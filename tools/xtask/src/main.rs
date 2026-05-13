use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

type DynError = Box<dyn Error>;
type DynResult<T> = Result<T, DynError>;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> DynResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command, scope] if command == "repo-consistency" && scope == "release-targets" => {
            check_release_targets()
        }
        _ => Err(
            "usage:\n  cargo run -p xtask -- repo-consistency release-targets"
                .to_string()
                .into(),
        ),
    }
}

fn check_release_targets() -> DynResult<()> {
    let repo_root = repo_root()?;
    let fixture_rows = fixture_rows(&repo_root)?;
    let fixture_version = fixture_release_tag(&fixture_rows)?;

    if host_supports_shell_parity_checks() {
        check_installer_outcomes(&repo_root, &fixture_rows)?;
        check_package_release_assets(&repo_root, &fixture_rows, &fixture_version)?;
    } else {
        println!(
            "note: skipping bash-dependent release parity checks on native Windows; run `just check-release` on macOS/Linux for install.sh and package-release.sh parity"
        );
    }
    check_windows_name_invariance(&fixture_rows, &fixture_version)?;
    check_docs_and_workflow_invariants(&repo_root)?;

    println!("repo consistency checks passed: release-targets");
    Ok(())
}

fn host_supports_shell_parity_checks() -> bool {
    !cfg!(windows)
}

fn repo_root() -> DynResult<PathBuf> {
    // CARGO_MANIFEST_DIR is <repo>/tools/xtask; go up two levels to reach the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "could not determine repo root from xtask manifest directory".into())
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureRow {
    os: String,
    arch: String,
    flavor: String,
    support: String,
    stable_asset: Option<String>,
    versioned_asset: Option<String>,
}

fn fixture_rows(repo_root: &Path) -> DynResult<Vec<FixtureRow>> {
    let fixture_path = fixture_path(repo_root);
    let contents = fs::read_to_string(&fixture_path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn fixture_path(repo_root: &Path) -> PathBuf {
    repo_root
        .join("crates")
        .join("mesh-llm-system")
        .join("tests")
        .join("fixtures")
        .join("release-target-matrix.json")
}

fn fixture_release_tag(rows: &[FixtureRow]) -> DynResult<String> {
    for row in rows {
        let (Some(stable), Some(versioned)) = (&row.stable_asset, &row.versioned_asset) else {
            continue;
        };

        let stable_tail = stable
            .strip_prefix("mesh-llm-")
            .ok_or("stable asset missing mesh-llm- prefix")?;
        let versioned_tail = versioned
            .strip_prefix("mesh-llm-")
            .ok_or("versioned asset missing mesh-llm- prefix")?;
        let suffix = format!("-{stable_tail}");
        if let Some(version) = versioned_tail.strip_suffix(&suffix) {
            return Ok(version.to_string());
        }
    }

    Err("could not derive fixture release tag".into())
}

fn fixture_row<'a>(
    rows: &'a [FixtureRow],
    os: &str,
    arch: &str,
    flavor: &str,
) -> DynResult<&'a FixtureRow> {
    rows.iter()
        .find(|row| row.os == os && row.arch == arch && row.flavor == flavor)
        .ok_or_else(|| format!("missing fixture row for {os}/{arch}/{flavor}").into())
}

fn check_installer_outcomes(repo_root: &Path, rows: &[FixtureRow]) -> DynResult<()> {
    let linux_arm64_asset = fixture_row(rows, "linux", "aarch64", "cpu")?
        .stable_asset
        .clone()
        .ok_or("linux/aarch64/cpu stable asset missing")?;
    let macos_arm64_asset = fixture_row(rows, "macos", "aarch64", "metal")?
        .stable_asset
        .clone()
        .ok_or("macos/aarch64/metal stable asset missing")?;

    let cases = [
        InstallerCase {
            raw_os: "Linux",
            raw_arch: "arm64",
            flavor: "cpu",
            expected_platform: "Linux/aarch64",
            expected_supported_flavors: "cpu",
            expected_asset: linux_arm64_asset.as_str(),
            label: "Linux/arm64",
        },
        InstallerCase {
            raw_os: "Linux",
            raw_arch: "aarch64",
            flavor: "cpu",
            expected_platform: "Linux/aarch64",
            expected_supported_flavors: "cpu",
            expected_asset: linux_arm64_asset.as_str(),
            label: "Linux/aarch64",
        },
        InstallerCase {
            raw_os: "Darwin",
            raw_arch: "arm64",
            flavor: "metal",
            expected_platform: "Darwin/arm64",
            expected_supported_flavors: "metal",
            expected_asset: macos_arm64_asset.as_str(),
            label: "Darwin/arm64",
        },
    ];

    for case in cases {
        let envs = [
            ("MESH_LLM_TEST_UNAME_S", case.raw_os),
            ("MESH_LLM_TEST_UNAME_M", case.raw_arch),
        ];
        let actual_platform =
            sourced_script_stdout(repo_root, "install.sh", "platform_id", &envs, &[])?;
        ensure_eq(
            case.expected_platform,
            &actual_platform,
            &format!("{} normalized platform", case.label),
        )?;

        let actual_supported_flavors =
            sourced_script_stdout(repo_root, "install.sh", "supported_flavors", &envs, &[])?;
        ensure_eq(
            case.expected_supported_flavors,
            &actual_supported_flavors,
            &format!("{} supported flavors", case.label),
        )?;

        let actual_asset = sourced_script_stdout(
            repo_root,
            "install.sh",
            "asset_name \"$2\"",
            &envs,
            &[case.flavor],
        )?;
        ensure_eq(
            case.expected_asset,
            &actual_asset,
            &format!("{} asset parity", case.label),
        )?;
    }

    let arm_fixture = fixture_row(rows, "linux", "arm", "cpu")?;
    let arm_envs = [
        ("MESH_LLM_TEST_UNAME_S", "Linux"),
        ("MESH_LLM_TEST_UNAME_M", "armv7l"),
    ];
    let actual_support = sourced_script_stdout(
        repo_root,
        "install.sh",
        "platform_support_status",
        &arm_envs,
        &[],
    )?;
    ensure_eq(
        &arm_fixture.support,
        &actual_support,
        "Linux/armv7l installer support classification",
    )?;
    let actual_message = sourced_script_stdout(
        repo_root,
        "install.sh",
        "platform_error_message",
        &arm_envs,
        &[],
    )?;
    ensure_eq(
        "error: recognized but unsupported platform: Linux/arm (32-bit ARM release bundles are not published)",
        &actual_message,
        "Linux/armv7l installer error",
    )?;

    Ok(())
}

struct InstallerCase<'a> {
    raw_os: &'a str,
    raw_arch: &'a str,
    flavor: &'a str,
    expected_platform: &'a str,
    expected_supported_flavors: &'a str,
    expected_asset: &'a str,
    label: &'a str,
}

fn check_package_release_assets(
    repo_root: &Path,
    rows: &[FixtureRow],
    fixture_version: &str,
) -> DynResult<()> {
    for row in rows {
        if row.os != "linux" && row.os != "macos" {
            continue;
        }
        if row.support == "recognized-unsupported" {
            continue;
        }

        for raw_case in raw_targets(row)? {
            let mut envs = vec![
                ("MESH_RELEASE_OS", raw_case.raw_os),
                ("MESH_RELEASE_ARCH", raw_case.raw_arch),
            ];
            if row.flavor != implicit_release_flavor(row) {
                envs.push(("MESH_RELEASE_FLAVOR", row.flavor.as_str()));
            }

            let actual_support = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "release_target_support",
                &envs,
                &[],
            )?;
            ensure_eq(
                shell_support(row),
                &actual_support,
                &format!(
                    "{}/{}/{} package support ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;

            if row.support != "supported" {
                let tmp_output_dir = unique_temp_dir("check-release-unsupported");
                let output = run_command(
                    Command::new("bash")
                        .current_dir(repo_root)
                        .envs(envs.iter().copied())
                        .arg("scripts/package-release.sh")
                        .arg(fixture_version)
                        .arg(&tmp_output_dir),
                );
                let _ = std::fs::remove_dir_all(&tmp_output_dir);
                let output = output?;
                ensure_status(
                    1,
                    output.status.code(),
                    &format!(
                        "{}/{}/{} unsupported packaging exit code ({})",
                        row.os, row.arch, row.flavor, raw_case.label
                    ),
                )?;
                ensure_eq(
                    &unsupported_release_target_message(&raw_case, row),
                    &trimmed_stderr_or_stdout(&output),
                    &format!(
                        "{}/{}/{} unsupported packaging message ({})",
                        row.os, row.arch, row.flavor, raw_case.label
                    ),
                )?;
                continue;
            }

            let actual_stable = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "resolve_release_target; printf '%s\\n' \"$STABLE_ASSET\"",
                &envs,
                &[],
            )?;
            ensure_eq_option(
                row.stable_asset.as_deref(),
                Some(actual_stable.as_str()),
                &format!(
                    "{}/{}/{} package stable asset ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;

            let actual_versioned = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "versioned_asset_name \"$2\"",
                &envs,
                &[fixture_version],
            )?;
            ensure_eq_option(
                row.versioned_asset.as_deref(),
                Some(actual_versioned.as_str()),
                &format!(
                    "{}/{}/{} package versioned asset ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;
        }
    }

    let arm_row = fixture_row(rows, "linux", "arm", "cpu")?;
    ensure_eq(
        "recognized-unsupported",
        &arm_row.support,
        "linux/arm fixture support",
    )?;
    ensure_eq_option(
        None,
        arm_row.stable_asset.as_deref(),
        "linux/arm fixture stable asset",
    )?;
    ensure_eq_option(
        None,
        arm_row.versioned_asset.as_deref(),
        "linux/arm fixture versioned asset",
    )?;

    let tmp_output_dir = unique_temp_dir("check-release");
    let output = run_command(
        Command::new("bash")
            .current_dir(repo_root)
            .env("MESH_RELEASE_OS", "Linux")
            .env("MESH_RELEASE_ARCH", "armv7l")
            .arg("scripts/package-release.sh")
            .arg(fixture_version)
            .arg(&tmp_output_dir),
    );
    // Clean up before propagating any error so the temp dir is always removed.
    let _ = std::fs::remove_dir_all(&tmp_output_dir);
    let output = output?;
    ensure_status(1, output.status.code(), "Linux/armv7l packaging exit code")?;
    let actual_message = trimmed_stderr_or_stdout(&output);
    ensure_eq(
        "Recognized but unsupported release target: Linux/armv7l (normalized: linux/arm)",
        &actual_message,
        "Linux/armv7l packaging error",
    )?;

    Ok(())
}

struct RawTargetCase {
    raw_os: &'static str,
    raw_arch: &'static str,
    label: &'static str,
}

fn raw_targets(row: &FixtureRow) -> DynResult<Vec<RawTargetCase>> {
    match (row.os.as_str(), row.arch.as_str()) {
        ("macos", "aarch64") => Ok(vec![RawTargetCase {
            raw_os: "Darwin",
            raw_arch: "arm64",
            label: "Darwin/arm64",
        }]),
        ("linux", "x86_64") => Ok(vec![RawTargetCase {
            raw_os: "Linux",
            raw_arch: "x86_64",
            label: "Linux/x86_64",
        }]),
        ("linux", "aarch64") => Ok(vec![
            RawTargetCase {
                raw_os: "Linux",
                raw_arch: "arm64",
                label: "Linux/arm64",
            },
            RawTargetCase {
                raw_os: "Linux",
                raw_arch: "aarch64",
                label: "Linux/aarch64",
            },
        ]),
        _ => Err(format!("unsupported raw target mapping for {}/{}", row.os, row.arch).into()),
    }
}

fn implicit_release_flavor(row: &FixtureRow) -> &'static str {
    match (row.os.as_str(), row.arch.as_str()) {
        ("macos", "aarch64") => "metal",
        ("linux", "x86_64") | ("linux", "aarch64") | ("linux", "arm") => "cpu",
        _ => "",
    }
}

fn shell_support(row: &FixtureRow) -> &str {
    match row.support.as_str() {
        "unknown" => "unsupported",
        other => other,
    }
}

fn unsupported_release_target_message(raw_case: &RawTargetCase, row: &FixtureRow) -> String {
    format!(
        "Unsupported release target/flavor for packaging: {}/{} with flavor {} (normalized: {}/{})",
        raw_case.raw_os, raw_case.raw_arch, row.flavor, row.os, row.arch
    )
}

fn check_windows_name_invariance(rows: &[FixtureRow], fixture_version: &str) -> DynResult<()> {
    for row in rows {
        if row.os != "windows" {
            continue;
        }

        ensure_eq(
            "x86_64",
            &row.arch,
            &format!("windows/{}/{}/canonical arch", row.arch, row.flavor),
        )?;
        ensure_eq(
            "supported",
            &row.support,
            &format!("windows/{}/{}/support", row.arch, row.flavor),
        )?;
        let stable_expected = windows_asset_name(&row.flavor, "");
        let versioned_expected = windows_asset_name(&row.flavor, &format!("-{fixture_version}"));
        ensure_eq_option(
            Some(stable_expected.as_str()),
            row.stable_asset.as_deref(),
            &format!("windows/{}/{}/stable asset", row.arch, row.flavor),
        )?;
        ensure_eq_option(
            Some(versioned_expected.as_str()),
            row.versioned_asset.as_deref(),
            &format!("windows/{}/{}/versioned asset", row.arch, row.flavor),
        )?;
    }

    Ok(())
}

fn windows_asset_name(flavor: &str, version_prefix: &str) -> String {
    let suffix = match flavor {
        "cpu" | "metal" => "",
        other => other,
    };

    if suffix.is_empty() {
        format!("mesh-llm{version_prefix}-x86_64-pc-windows-msvc.zip")
    } else {
        format!("mesh-llm{version_prefix}-x86_64-pc-windows-msvc-{suffix}.zip")
    }
}

fn check_docs_and_workflow_invariants(repo_root: &Path) -> DynResult<()> {
    let readme = fs::read_to_string(repo_root.join("README.md"))?;
    let contributing = fs::read_to_string(repo_root.join("CONTRIBUTING.md"))?;
    let release = fs::read_to_string(repo_root.join("RELEASE.md"))?;
    let justfile = fs::read_to_string(repo_root.join("Justfile"))?;
    let release_workflow = fs::read_to_string(repo_root.join(".github/workflows/release.yml"))?;
    let ci_workflow = fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))?;

    ensure_contains(
        &readme,
        "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
        "README Linux ARM64 asset note",
    )?;
    ensure_contains(
        &release,
        "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
        "RELEASE Linux ARM64 asset note",
    )?;
    ensure_contains_normalized(
        &readme,
        "Windows CPU, Windows CUDA, Windows ROCm, and Windows Vulkan bundles",
        "README Windows publish note",
    )?;
    ensure_contains(
        &release,
        "Windows release artifacts use the `x86_64-pc-windows-msvc` target triple",
        "RELEASE Windows publish note",
    )?;
    ensure_contains(
        &release_workflow,
        "runs-on: ubuntu-24.04-arm",
        "release workflow ARM64 runner",
    )?;
    ensure_contains(
        &release_workflow,
        "name: release-linux-arm64",
        "release workflow ARM64 artifact",
    )?;
    ensure_contains(
        &release_workflow,
        "build_windows_cpu:",
        "release workflow Windows CPU build",
    )?;
    ensure_contains(
        &release_workflow,
        "build_windows_gpu:",
        "release workflow Windows GPU build",
    )?;
    ensure_contains(
        &release_workflow,
        "- build_windows_cpu",
        "release workflow Windows CPU publish need",
    )?;
    ensure_contains(
        &release_workflow,
        "- build_windows_gpu",
        "release workflow Windows GPU publish need",
    )?;
    ensure_contains(
        &justfile,
        "check-release:",
        "Justfile release consistency wrapper",
    )?;
    ensure_contains(
        &justfile,
        "cargo run -p xtask -- repo-consistency release-targets",
        "Justfile xtask command",
    )?;
    ensure_contains(
        &contributing,
        "just check-release",
        "CONTRIBUTING release consistency command",
    )?;
    ensure_contains(
        &contributing,
        "On native Windows, `just check-release` runs the host-safe Rust/doc invariant subset and skips the Bash-only `install.sh` / `package-release.sh` parity checks",
        "CONTRIBUTING Windows check-release note",
    )?;
    ensure_contains(
        &release,
        "On native Windows, `just check-release` still runs the Rust/docs/workflow invariant checks, but it skips the Bash-only `install.sh` and `scripts/package-release.sh` parity checks",
        "RELEASE Windows check-release note",
    )?;
    ensure_contains(
        &ci_workflow,
        "cargo run -p xtask -- repo-consistency release-targets",
        "CI xtask release-target check",
    )?;
    check_ci_crate_test_coverage(&ci_workflow)?;

    Ok(())
}

fn check_ci_crate_test_coverage(ci_workflow: &str) -> DynResult<()> {
    const REQUIRED_TEST_COMMANDS: &[(&str, &str)] = &[
        ("cargo test -p mesh-llm-client", "mesh client crate tests"),
        ("cargo test -p mesh-api", "mesh API crate tests"),
        ("cargo test -p mesh-api-ffi", "mesh API FFI crate tests"),
        (
            "cargo test -p skippy-protocol --lib",
            "skippy protocol crate tests",
        ),
        (
            "cargo test -p skippy-server --lib",
            "skippy server crate tests",
        ),
        (
            "cargo test -p openai-frontend --lib",
            "OpenAI frontend crate tests",
        ),
        ("cargo test -p skippy-runtime", "skippy runtime crate tests"),
        (
            "cargo test -p skippy-topology",
            "skippy topology crate tests",
        ),
        (
            "cargo test -p skippy-model-package",
            "skippy model-package crate tests",
        ),
        ("cargo test -p skippy-prompt", "skippy prompt crate tests"),
        ("cargo test -p metrics-server", "metrics server crate tests"),
    ];

    for (command, context) in REQUIRED_TEST_COMMANDS {
        ensure_contains(ci_workflow, command, &format!("CI {context}"))?;
    }

    Ok(())
}

fn sourced_script_stdout(
    repo_root: &Path,
    script_relative_path: &str,
    expression: &str,
    envs: &[(&str, &str)],
    extra_args: &[&str],
) -> DynResult<String> {
    let script_path = repo_root.join(script_relative_path);
    let command = format!("source \"$1\"; {expression}");
    let mut bash = Command::new("bash");
    bash.current_dir(repo_root)
        .arg("-lc")
        .arg(command)
        .arg("bash")
        .arg(script_path);
    for extra_arg in extra_args {
        bash.arg(extra_arg);
    }
    for (key, value) in envs {
        bash.env(key, value);
    }

    let output = run_command(&mut bash)?;
    if !output.status.success() {
        return Err(format!(
            "script command failed: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }
    Ok(trimmed_stdout(&output))
}

fn run_command(command: &mut Command) -> DynResult<Output> {
    Ok(command.output()?)
}

fn trimmed_stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn trimmed_stderr_or_stdout(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        stderr
    } else {
        trimmed_stdout(output)
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(".tmp-{prefix}-{}-{nanos}", std::process::id()))
}

fn ensure_eq(expected: &str, actual: &str, context: &str) -> DynResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(format!("{context}: expected `{expected}`, got `{actual}`").into())
    }
}

fn ensure_eq_option(expected: Option<&str>, actual: Option<&str>, context: &str) -> DynResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(format!("{context}: expected {:?}, got {:?}", expected, actual).into())
    }
}

fn ensure_status(expected: i32, actual: Option<i32>, context: &str) -> DynResult<()> {
    match actual {
        Some(status) if status == expected => Ok(()),
        Some(status) => {
            Err(format!("{context}: expected exit code {expected}, got {status}").into())
        }
        None => Err(format!("{context}: process terminated by signal").into()),
    }
}

fn ensure_contains(haystack: &str, needle: &str, context: &str) -> DynResult<()> {
    if haystack.contains(needle) {
        Ok(())
    } else {
        Err(format!("{context}: missing `{needle}`").into())
    }
}

fn ensure_contains_normalized(haystack: &str, needle: &str, context: &str) -> DynResult<()> {
    let normalized_haystack = normalize_whitespace(haystack);
    let normalized_needle = normalize_whitespace(needle);
    if normalized_haystack.contains(&normalized_needle) {
        Ok(())
    } else {
        Err(format!("{context}: missing `{needle}`").into())
    }
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}
