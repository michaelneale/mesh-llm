pub async fn latest_release_version() -> Option<String> {
    latest_release_info().await.map(|release| release.version)
}

async fn latest_release_info() -> Option<ReleaseInfo> {
    fetch_release_info(&latest_release_api_url()).await
}

async fn release_info_for_tag(tag: &str) -> Option<ReleaseInfo> {
    fetch_release_info(&release_api_url_for_tag(tag)).await
}

async fn fetch_release_info(url: &str) -> Option<ReleaseInfo> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;
    let resp = client
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    release_info_from_json(&body)
}

pub fn version_newer(a: &str, b: &str) -> bool {
    match (semver::Version::parse(a), semver::Version::parse(b)) {
        (Ok(a), Ok(b)) => a > b,
        _ => false,
    }
}

fn should_attempt_auto_update(options: AutoUpdateOptions) -> bool {
    options.auto_update
        && !options.plugin_requested
        && !options.command_is_update
        && std::env::var_os(SELF_UPDATE_ATTEMPTED_ENV).is_none()
}

fn discover_update_target(llama_flavor: Option<backend::BinaryFlavor>) -> Option<UpdateTarget> {
    let exe = std::env::current_exe().ok()?;
    let (install_dir, bundle_flavor) = bundle_install_dir(&exe, llama_flavor)?;
    let release_target = current_release_target(bundle_flavor)?;
    Some(UpdateTarget {
        exe,
        install_dir,
        release_target,
        bundle_flavor,
    })
}

fn require_update_target(llama_flavor: Option<backend::BinaryFlavor>) -> Result<UpdateTarget> {
    if !platform_has_release_assets() {
        bail!(
            "`mesh-llm update` is not supported on this platform. Download the latest release from {RELEASES_URL}."
        );
    }

    let exe = std::env::current_exe().context("Cannot determine mesh-llm executable path")?;
    let Some((install_dir, bundle_flavor)) = bundle_install_dir(&exe, llama_flavor) else {
        bail!(
            "`mesh-llm update` only works for release-bundle installs. Current executable: {}",
            exe.display()
        );
    };
    let Some(release_target) = current_release_target(bundle_flavor) else {
        #[cfg(not(windows))]
        bail!("No published release bundle matches this install. Reinstall with {INSTALL_SCRIPT_URL}.");
        #[cfg(windows)]
        bail!("No published release bundle matches this install. Download the latest release from {RELEASES_URL}.");
    };

    Ok(UpdateTarget {
        exe,
        install_dir,
        release_target,
        bundle_flavor,
    })
}

async fn apply_update_if_available(
    target: UpdateTarget,
    action: PostInstallAction,
    current_version: &str,
) -> Result<bool> {
    let Some(release) = latest_release_info().await else {
        return Ok(true);
    };
    if !version_newer(&release.version, current_version) {
        return Ok(true);
    }
    let Some(asset_name) = resolve_release_asset_name(
        &release,
        target.release_target,
        ReleaseAssetPreference::StableFirst,
    ) else {
        return Ok(false);
    };
    if !path_is_writable(&target.exe) {
        eprintln!(
            "⚠️  Auto-update skipped: {} is not writable",
            target.exe.display()
        );
        return Ok(true);
    }

    eprintln!(
        "⬇️ Updating mesh-llm v{current_version} -> v{} ({})...",
        release.version,
        target.bundle_flavor.suffix()
    );
    match install_latest_bundle(
        &target.exe,
        &target.install_dir,
        &release,
        &asset_name,
        target.bundle_flavor,
        action,
    )
    .await
    {
        Ok(InstallOutcome::RestartNow) => {
            eprintln!("✅ Updated to v{}; restarting", release.version);
            std::env::set_var(SELF_UPDATE_ATTEMPTED_ENV, "1");
            exec_current_binary(&target.exe)?;
        }
        Ok(InstallOutcome::ExitNow) => {
            eprintln!("✅ Updated to v{}", release.version);
        }
        Ok(InstallOutcome::HandoffAndExit) => {
            eprintln!("✅ Updated to v{}; restarting", release.version);
            std::process::exit(0);
        }
        Err(err) => {
            eprintln!("⚠️  Auto-update failed: {err}");
        }
    }

    Ok(true)
}

fn current_release_target(flavor: backend::BinaryFlavor) -> Option<ReleaseTarget> {
    ReleaseTarget::from_raw(std::env::consts::OS, std::env::consts::ARCH, flavor).ok()
}

#[cfg(test)]
fn stable_release_asset_name_for(
    os: &str,
    arch: &str,
    flavor: backend::BinaryFlavor,
) -> Option<String> {
    ReleaseTarget::from_raw(os, arch, flavor)
        .ok()
        .and_then(ReleaseTarget::stable_asset_name)
}

fn legacy_release_asset_name(target: ReleaseTarget) -> Option<String> {
    (target
        == ReleaseTarget::new(
            CanonicalOs::Macos,
            CanonicalArch::Aarch64,
            backend::BinaryFlavor::Metal,
        ))
    .then_some("mesh-bundle.tar.gz".to_string())
}

fn push_release_asset_candidate(candidates: &mut Vec<String>, asset_name: Option<String>) {
    let Some(asset_name) = asset_name else {
        return;
    };
    if !candidates.iter().any(|candidate| candidate == &asset_name) {
        candidates.push(asset_name);
    }
}

fn release_asset_candidates(
    target: ReleaseTarget,
    release_tag: &str,
    preference: ReleaseAssetPreference,
) -> Vec<String> {
    let mut candidates = Vec::new();
    match preference {
        ReleaseAssetPreference::StableFirst => {
            push_release_asset_candidate(&mut candidates, target.stable_asset_name());
            push_release_asset_candidate(&mut candidates, target.versioned_asset_name(release_tag));
        }
        ReleaseAssetPreference::VersionedFirst => {
            push_release_asset_candidate(&mut candidates, target.versioned_asset_name(release_tag));
            push_release_asset_candidate(&mut candidates, target.stable_asset_name());
        }
    }
    push_release_asset_candidate(&mut candidates, legacy_release_asset_name(target));
    candidates
}

fn resolve_release_asset_name(
    release: &ReleaseInfo,
    target: ReleaseTarget,
    preference: ReleaseAssetPreference,
) -> Option<String> {
    release_asset_candidates(target, &release.tag, preference)
        .into_iter()
        .find(|asset_name| {
            release
                .assets
                .iter()
                .any(|candidate| candidate == asset_name)
        })
}

fn release_has_any_platform_asset(release: &ReleaseInfo, os: &str, arch: &str) -> bool {
    backend::BinaryFlavor::ALL.into_iter().any(|flavor| {
        ReleaseTarget::from_raw(os, arch, flavor)
            .ok()
            .and_then(|target| {
                resolve_release_asset_name(release, target, ReleaseAssetPreference::StableFirst)
            })
            .is_some()
    })
}

fn mesh_binary_name() -> String {
    backend::platform_bin_name("mesh-llm")
}

fn installed_bundle_flavor(
    _dir: &Path,
    requested: Option<backend::BinaryFlavor>,
) -> Option<backend::BinaryFlavor> {
    if let Some(flavor) = requested {
        return Some(flavor);
    }

    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some(backend::BinaryFlavor::Metal)
    } else {
        Some(backend::BinaryFlavor::Cpu)
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

fn release_api_url_for_tag(tag: &str) -> String {
    format!(
        "https://api.github.com/repos/{}/releases/tags/{tag}",
        release_repo()
    )
}

fn release_asset_url(tag: &str, asset_name: &str) -> String {
    format!(
        "https://github.com/{}/releases/download/{tag}/{asset_name}",
        release_repo()
    )
}

fn release_info_from_json(body: &serde_json::Value) -> Option<ReleaseInfo> {
    let tag = body["tag_name"].as_str()?.trim();
    let version = tag.trim_start_matches('v').trim();
    if tag.is_empty() || version.is_empty() {
        return None;
    }

    let assets = body["assets"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item["name"].as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Some(ReleaseInfo {
        tag: tag.to_string(),
        version: version.to_string(),
        assets,
    })
}
