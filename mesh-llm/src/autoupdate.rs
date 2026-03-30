use anyhow::{Context, Result};
#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use crate::{download, launch, plugin, Cli, VERSION};

const DEFAULT_RELEASE_REPO: &str = "michaelneale/mesh-llm";
const INSTALL_SCRIPT_URL: &str =
    "https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh";
const SELF_UPDATE_ATTEMPTED_ENV: &str = "MESH_LLM_SELF_UPDATE_ATTEMPTED";
const SELF_UPDATE_DISABLED_ENV: &str = "MESH_LLM_NO_SELF_UPDATE";
const SELF_UPDATE_REPO_ENV: &str = "MESH_LLM_SELF_UPDATE_REPO";

pub(crate) async fn check_for_update() {
    if let Some(latest) = latest_release_version().await {
        if version_newer(&latest, VERSION) {
            eprintln!(
                "💡 Update available: v{VERSION} → v{latest}  https://github.com/michaelneale/mesh-llm/releases"
            );
            eprintln!("   curl -fsSL {INSTALL_SCRIPT_URL} | bash");
        }
    }
}

pub(crate) async fn maybe_self_update(cli: &Cli) -> Result<bool> {
    if !should_attempt_self_update(cli) {
        return Ok(false);
    }

    let exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(_) => return Ok(false),
    };
    let Some((install_dir, bundle_flavor)) = bundle_install_dir(&exe, cli.llama_flavor) else {
        return Ok(false);
    };
    let Some(asset_name) = stable_release_asset_name(bundle_flavor) else {
        return Ok(false);
    };

    let Some(latest) = latest_release_version().await else {
        return Ok(true);
    };
    if !version_newer(&latest, VERSION) {
        return Ok(true);
    }
    if !path_is_writable(&exe) {
        eprintln!(
            "⚠️  Startup self-update skipped: {} is not writable",
            exe.display()
        );
        return Ok(true);
    }

    eprintln!(
        "⬇️ Updating mesh-llm v{VERSION} → v{latest} ({})...",
        bundle_flavor.suffix()
    );
    match install_latest_bundle(&install_dir, &asset_name, bundle_flavor).await {
        Ok(()) => {
            eprintln!("✅ Updated to v{latest}; restarting");
            std::env::set_var(SELF_UPDATE_ATTEMPTED_ENV, "1");
            exec_current_binary(&exe)?;
        }
        Err(err) => {
            eprintln!("⚠️  Startup self-update failed: {err}");
        }
    }

    Ok(true)
}

pub(crate) fn startup_self_update_enabled(cli: &Cli) -> bool {
    if !should_attempt_self_update(cli) {
        return false;
    }
    match plugin::load_config(cli.config.as_deref()) {
        Ok(config) => config.self_update_enabled(),
        Err(err) => {
            eprintln!("⚠️  Failed to read config for self-update: {err}");
            false
        }
    }
}

pub(crate) async fn latest_release_version() -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;
    let resp = client
        .get(latest_release_api_url())
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    let tag = body["tag_name"].as_str()?;
    let latest = tag.trim_start_matches('v').trim();
    if latest.is_empty() {
        None
    } else {
        Some(latest.to_string())
    }
}

pub(crate) fn version_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> { v.split('.').filter_map(|s| s.parse().ok()).collect() };
    parse(a) > parse(b)
}

fn should_attempt_self_update(cli: &Cli) -> bool {
    cli.command.is_none()
        && cli.plugin.is_none()
        && !cli.no_self_update
        && std::env::var_os(SELF_UPDATE_ATTEMPTED_ENV).is_none()
        && std::env::var_os(SELF_UPDATE_DISABLED_ENV).is_none()
}

fn stable_release_asset_name(flavor: launch::BinaryFlavor) -> Option<String> {
    match (std::env::consts::OS, std::env::consts::ARCH, flavor) {
        ("macos", "aarch64", launch::BinaryFlavor::Metal) => {
            Some("mesh-llm-aarch64-apple-darwin.tar.gz".to_string())
        }
        ("linux", "x86_64", launch::BinaryFlavor::Cpu) => {
            Some("mesh-llm-x86_64-unknown-linux-gnu.tar.gz".to_string())
        }
        ("linux", "x86_64", launch::BinaryFlavor::Cuda) => {
            Some("mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz".to_string())
        }
        ("linux", "x86_64", launch::BinaryFlavor::Rocm) => {
            Some("mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz".to_string())
        }
        ("linux", "x86_64", launch::BinaryFlavor::Vulkan) => {
            Some("mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz".to_string())
        }
        _ => None,
    }
}

fn bundled_server_flavor_name(name: &str, flavor: launch::BinaryFlavor) -> String {
    format!("{name}-{}", flavor.suffix())
}

fn has_bundled_server_pair(dir: &Path, flavor: launch::BinaryFlavor) -> bool {
    dir.join(bundled_server_flavor_name("rpc-server", flavor))
        .is_file()
        && dir
            .join(bundled_server_flavor_name("llama-server", flavor))
            .is_file()
}

fn bundled_server_flavors(dir: &Path) -> Vec<launch::BinaryFlavor> {
    launch::BinaryFlavor::ALL
        .into_iter()
        .filter(|flavor| has_bundled_server_pair(dir, *flavor))
        .collect()
}

fn installed_bundle_flavor(
    dir: &Path,
    requested: Option<launch::BinaryFlavor>,
) -> Option<launch::BinaryFlavor> {
    if let Some(flavor) = requested {
        return has_bundled_server_pair(dir, flavor).then_some(flavor);
    }

    let mut matches = bundled_server_flavors(dir);
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

fn release_repo() -> String {
    match std::env::var(SELF_UPDATE_REPO_ENV) {
        Ok(repo) if repo.contains('/') && !repo.trim().is_empty() => repo,
        _ => DEFAULT_RELEASE_REPO.to_string(),
    }
}

fn latest_release_api_url() -> String {
    format!(
        "https://api.github.com/repos/{}/releases/latest",
        release_repo()
    )
}

fn latest_release_asset_url(asset_name: &str) -> String {
    format!(
        "https://github.com/{}/releases/latest/download/{asset_name}",
        release_repo()
    )
}

fn path_is_writable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        let Ok(c_path) = CString::new(path.as_os_str().as_bytes()) else {
            return false;
        };
        unsafe { libc::access(c_path.as_ptr(), libc::W_OK) == 0 }
    }

    #[cfg(not(unix))]
    {
        std::fs::metadata(path)
            .map(|meta| !meta.permissions().readonly())
            .unwrap_or(false)
    }
}

fn bundle_install_dir(
    exe: &Path,
    requested_flavor: Option<launch::BinaryFlavor>,
) -> Option<(PathBuf, launch::BinaryFlavor)> {
    let dir = exe.parent()?;
    if exe.file_name()?.to_str()? != "mesh-llm" {
        return None;
    }
    let flavor = installed_bundle_flavor(dir, requested_flavor)?;
    Some((dir.to_path_buf(), flavor))
}

async fn install_latest_bundle(
    install_dir: &Path,
    asset_name: &str,
    expected_flavor: launch::BinaryFlavor,
) -> Result<()> {
    let unique = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let workspace = install_dir.join(format!(".mesh-llm-update-{unique}"));
    let extracted = workspace.join("bundle");
    let archive = workspace.join(asset_name);
    let backup = workspace.join("backup");

    std::fs::create_dir_all(&extracted)
        .with_context(|| format!("Failed to create update workspace {}", workspace.display()))?;

    let result = async {
        download::download_url(&latest_release_asset_url(asset_name), &archive).await?;
        extract_bundle_archive(&archive, &extracted)?;
        let staged_files = collect_bundle_files(&extracted, expected_flavor)?;
        replace_bundle_files(install_dir, &extracted, &backup, &staged_files)?;
        Ok::<(), anyhow::Error>(())
    }
    .await;

    let _ = std::fs::remove_dir_all(&workspace);
    result
}

fn extract_bundle_archive(archive: &Path, extracted: &Path) -> Result<()> {
    let status = std::process::Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(extracted)
        .arg("--strip-components=1")
        .status()
        .with_context(|| format!("Failed to extract {}", archive.display()))?;
    anyhow::ensure!(status.success(), "tar extraction failed");
    Ok(())
}

fn collect_bundle_files(
    extracted: &Path,
    expected_flavor: launch::BinaryFlavor,
) -> Result<Vec<String>> {
    let flavors = bundled_server_flavors(extracted);
    anyhow::ensure!(
        flavors.len() == 1,
        "Downloaded bundle must contain exactly one bundled server flavor"
    );
    anyhow::ensure!(
        flavors[0] == expected_flavor,
        "Downloaded bundle flavor mismatch: expected {}, found {}",
        expected_flavor.suffix(),
        flavors[0].suffix()
    );

    let mut files = Vec::new();
    for entry in std::fs::read_dir(extracted)
        .with_context(|| format!("Failed to read {}", extracted.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name = name.to_string_lossy().to_string();
        if file_type.is_dir() {
            anyhow::bail!("Unexpected directory in bundle: {name}");
        }
        if file_type.is_file() {
            files.push(name);
        }
    }

    anyhow::ensure!(!files.is_empty(), "Downloaded bundle was empty");
    anyhow::ensure!(
        files.iter().any(|name| name == "mesh-llm"),
        "Downloaded bundle missing mesh-llm"
    );
    files.sort_by_key(|name| (name == "mesh-llm", name.clone()));
    Ok(files)
}

fn backup_existing_file(
    install_dir: &Path,
    backup: &Path,
    name: &str,
    backed_up: &mut Vec<String>,
) -> Result<()> {
    if backed_up.iter().any(|existing| existing == name) {
        return Ok(());
    }

    let dest = install_dir.join(name);
    if !dest.exists() {
        return Ok(());
    }

    let backup_path = backup.join(name);
    std::fs::rename(&dest, &backup_path).with_context(|| {
        format!(
            "Failed to move {} to {}",
            dest.display(),
            backup_path.display()
        )
    })?;
    backed_up.push(name.to_string());
    Ok(())
}

fn replace_bundle_files(
    install_dir: &Path,
    extracted: &Path,
    backup: &Path,
    staged_files: &[String],
) -> Result<()> {
    use std::collections::BTreeSet;

    std::fs::create_dir_all(backup)
        .with_context(|| format!("Failed to create backup dir {}", backup.display()))?;

    let staged_names: BTreeSet<&str> = staged_files.iter().map(String::as_str).collect();
    let mut managed_names: BTreeSet<String> = BTreeSet::from(["mesh-llm".to_string()]);
    managed_names.extend(super::bundled_bin_names("rpc-server"));
    managed_names.extend(super::bundled_bin_names("llama-server"));

    let mut backed_up = Vec::new();
    for name in managed_names {
        if staged_names.contains(name.as_str()) {
            continue;
        }
        backup_existing_file(install_dir, backup, &name, &mut backed_up)?;
    }
    for name in staged_files {
        backup_existing_file(install_dir, backup, name, &mut backed_up)?;
    }

    let mut installed = Vec::new();
    for name in staged_files {
        let source = extracted.join(name);
        let dest = install_dir.join(name);
        if let Err(err) = std::fs::rename(&source, &dest) {
            rollback_bundle_replace(install_dir, backup, &installed, &backed_up);
            return Err(err).with_context(|| {
                format!(
                    "Failed to install {} into {}",
                    source.display(),
                    dest.display()
                )
            });
        }
        installed.push(name.clone());
    }

    Ok(())
}

fn rollback_bundle_replace(
    install_dir: &Path,
    backup: &Path,
    installed: &[String],
    backed_up: &[String],
) {
    for name in installed.iter().rev() {
        let dest = install_dir.join(name);
        let _ = std::fs::remove_file(&dest);
    }
    for name in backed_up.iter().rev() {
        let backup_path = backup.join(name);
        let dest = install_dir.join(name);
        let _ = std::fs::rename(&backup_path, &dest);
    }
}

#[cfg(unix)]
fn exec_current_binary(exe: &Path) -> Result<()> {
    let exe_c = CString::new(exe.as_os_str().as_bytes())
        .context("Executable path contains an unexpected NUL byte")?;
    let args: Vec<CString> = std::env::args_os()
        .map(|arg| {
            CString::new(arg.as_os_str().as_bytes())
                .context("Argument contains an unexpected NUL byte")
        })
        .collect::<Result<_>>()?;
    let mut argv: Vec<*const libc::c_char> = args.iter().map(|arg| arg.as_ptr()).collect();
    argv.push(std::ptr::null());
    let rc = unsafe { libc::execv(exe_c.as_ptr(), argv.as_ptr()) };
    let errno = std::io::Error::last_os_error();
    anyhow::ensure!(rc != 0, "execv unexpectedly returned success");
    Err(errno).context("Failed to restart updated mesh-llm")
}

#[cfg(not(unix))]
fn exec_current_binary(_exe: &Path) -> Result<()> {
    anyhow::bail!("Self-update restart is only supported on Unix")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serial_test::serial;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let unique = format!(
            "mesh-llm-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_version_newer() {
        assert!(version_newer("0.33.1", "0.33.0"));
        assert!(!version_newer("0.33.0", "0.33.0"));
        assert!(!version_newer("0.32.0", "0.33.0"));
    }

    #[test]
    #[serial]
    fn test_latest_release_asset_url() {
        std::env::remove_var(SELF_UPDATE_REPO_ENV);
        assert_eq!(
            latest_release_asset_url("mesh-llm-aarch64-apple-darwin.tar.gz"),
            "https://github.com/michaelneale/mesh-llm/releases/latest/download/mesh-llm-aarch64-apple-darwin.tar.gz"
        );
    }

    #[test]
    #[serial]
    fn test_release_repo_defaults_to_main_repo() {
        std::env::remove_var(SELF_UPDATE_REPO_ENV);
        assert_eq!(release_repo(), "michaelneale/mesh-llm");
        assert_eq!(
            latest_release_api_url(),
            "https://api.github.com/repos/michaelneale/mesh-llm/releases/latest"
        );
    }

    #[test]
    #[serial]
    fn test_release_repo_can_be_overridden_for_testing() {
        std::env::set_var(SELF_UPDATE_REPO_ENV, "jdumay/mesh-llm");
        assert_eq!(release_repo(), "jdumay/mesh-llm");
        assert_eq!(
            latest_release_api_url(),
            "https://api.github.com/repos/jdumay/mesh-llm/releases/latest"
        );
        assert_eq!(
            latest_release_asset_url("mesh-llm-x86_64-unknown-linux-gnu.tar.gz"),
            "https://github.com/jdumay/mesh-llm/releases/latest/download/mesh-llm-x86_64-unknown-linux-gnu.tar.gz"
        );
        std::env::remove_var(SELF_UPDATE_REPO_ENV);
    }

    #[test]
    fn test_stable_release_asset_name_matches_platform() {
        let expected = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => Some((
                launch::BinaryFlavor::Metal,
                "mesh-llm-aarch64-apple-darwin.tar.gz",
            )),
            ("linux", "x86_64") => Some((
                launch::BinaryFlavor::Cpu,
                "mesh-llm-x86_64-unknown-linux-gnu.tar.gz",
            )),
            _ => None,
        };

        let Some((flavor, asset)) = expected else {
            return;
        };
        assert_eq!(stable_release_asset_name(flavor), Some(asset.to_string()));
    }

    #[test]
    fn test_path_is_writable_for_temp_file() {
        let dir = temp_dir("self-update-writable");
        let path = dir.join("mesh-llm");
        std::fs::write(&path, b"binary").unwrap();
        assert!(path_is_writable(&path));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    #[serial]
    fn test_should_attempt_self_update_defaults_to_startup_only() {
        std::env::remove_var(SELF_UPDATE_ATTEMPTED_ENV);
        std::env::remove_var(SELF_UPDATE_DISABLED_ENV);
        let cli = Cli::parse_from(["mesh-llm", "--auto"]);
        assert!(should_attempt_self_update(&cli));

        let cli = Cli::parse_from(["mesh-llm", "download"]);
        assert!(!should_attempt_self_update(&cli));
    }

    #[test]
    #[serial]
    fn test_should_attempt_self_update_respects_disable_flags() {
        std::env::set_var(SELF_UPDATE_ATTEMPTED_ENV, "1");
        let cli = Cli::parse_from(["mesh-llm", "--auto"]);
        assert!(!should_attempt_self_update(&cli));
        std::env::remove_var(SELF_UPDATE_ATTEMPTED_ENV);

        std::env::set_var(SELF_UPDATE_DISABLED_ENV, "1");
        let cli = Cli::parse_from(["mesh-llm", "--auto"]);
        assert!(!should_attempt_self_update(&cli));
        std::env::remove_var(SELF_UPDATE_DISABLED_ENV);

        let cli = Cli::parse_from(["mesh-llm", "--auto", "--no-self-update"]);
        assert!(!should_attempt_self_update(&cli));
    }

    #[test]
    fn test_startup_self_update_enabled_respects_config() {
        let dir = temp_dir("self-update-config");
        let config = dir.join("config.toml");
        std::fs::write(&config, "self_update = false\n").unwrap();

        let cli = Cli::parse_from([
            "mesh-llm",
            "--auto",
            "--config",
            config.to_string_lossy().as_ref(),
        ]);
        assert!(!startup_self_update_enabled(&cli));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_bundle_install_dir_requires_matching_flavor_pair() {
        let dir = temp_dir("bundle-install");
        let exe = dir.join("mesh-llm");
        std::fs::write(&exe, b"binary").unwrap();
        std::fs::write(dir.join("rpc-server-cpu"), b"rpc").unwrap();
        std::fs::write(dir.join("llama-server-cpu"), b"llama").unwrap();

        assert_eq!(
            bundle_install_dir(&exe, None),
            Some((dir.clone(), launch::BinaryFlavor::Cpu))
        );
        assert_eq!(
            bundle_install_dir(&exe, Some(launch::BinaryFlavor::Cpu)),
            Some((dir.clone(), launch::BinaryFlavor::Cpu))
        );
        assert_eq!(
            bundle_install_dir(&exe, Some(launch::BinaryFlavor::Cuda)),
            None
        );

        let missing = temp_dir("bundle-missing");
        let missing_exe = missing.join("mesh-llm");
        std::fs::write(&missing_exe, b"binary").unwrap();
        assert_eq!(bundle_install_dir(&missing_exe, None), None);

        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(missing);
    }
}
