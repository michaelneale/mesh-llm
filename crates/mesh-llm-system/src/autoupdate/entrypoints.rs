pub async fn check_for_update(current_version: &str) {
    if !platform_has_release_assets() {
        return;
    }
    if let Some(release) = latest_release_info().await {
        if !version_newer(&release.version, current_version) {
            return;
        }
        // Determine whether this is a bundle install and, if so, whether the
        // specific installed flavor's asset is present in the new release.
        let bundle_asset = std::env::current_exe().ok().and_then(|exe| {
            let (_, flavor) = bundle_install_dir(&exe, None)?;
            current_release_target(flavor).and_then(|target| {
                resolve_release_asset_name(&release, target, ReleaseAssetPreference::StableFirst)
            })
        });
        match bundle_asset {
            Some(ref asset) if release.assets.iter().any(|a| a == asset) => {
                eprintln!(
                    "✨ New version: v{current_version} -> v{}. Run 'mesh-llm update'.",
                    release.version
                );
            }
            _ => {
                // Either not a bundle install, or the installed flavor's asset
                // is not published in the new release — fall back to generic guidance.
                #[cfg(not(windows))]
                if release_has_any_platform_asset(
                    &release,
                    std::env::consts::OS,
                    std::env::consts::ARCH,
                ) {
                    eprintln!(
                        "✨ New version: v{current_version} -> v{}. Reinstall with: curl -fsSL {INSTALL_SCRIPT_URL} | bash",
                        release.version
                    );
                }
                #[cfg(windows)]
                if release_has_any_platform_asset(
                    &release,
                    std::env::consts::OS,
                    std::env::consts::ARCH,
                ) {
                    eprintln!(
                        "✨ New version: v{current_version} -> v{}. Download from {RELEASES_URL}",
                        release.version
                    );
                }
            }
        }
    }
}

fn platform_has_release_assets() -> bool {
    platform_has_release_assets_for(std::env::consts::OS, std::env::consts::ARCH)
}

fn platform_has_release_assets_for(os: &str, arch: &str) -> bool {
    backend::BinaryFlavor::ALL.into_iter().any(|flavor| {
        ReleaseTarget::from_raw(os, arch, flavor)
            .map(|target| target.support_status().is_supported())
            .unwrap_or(false)
    })
}

pub async fn maybe_auto_update(options: AutoUpdateOptions) -> Result<bool> {
    if !should_attempt_auto_update(options) {
        return Ok(false);
    }
    let Some(target) = discover_update_target(options.llama_flavor) else {
        return Ok(false);
    };
    apply_update_if_available(
        target,
        PostInstallAction::RestartCurrentProcess,
        options.current_version,
    )
    .await
}

pub async fn run_update_command(options: UpdateCommandOptions<'_>) -> Result<()> {
    let target = require_update_target(options.llama_flavor)?;
    let requested_version = options.requested_version;
    let Some(release) = resolve_release_info(requested_version).await? else {
        bail!("Could not check for a release right now. Try again shortly.");
    };
    if requested_version.is_none() && !version_newer(&release.version, options.current_version) {
        eprintln!(
            "mesh-llm is already up to date (v{}).",
            options.current_version
        );
        return Ok(());
    }
    let asset_preference = if requested_version.is_some() {
        ReleaseAssetPreference::VersionedFirst
    } else {
        ReleaseAssetPreference::StableFirst
    };
    let Some(asset_name) =
        resolve_release_asset_name(&release, target.release_target, asset_preference)
    else {
        bail!(
            "Release v{} does not include a bundle for this install (tried: {}).",
            release.version,
            release_asset_candidates(target.release_target, &release.tag, asset_preference)
                .join(", ")
        );
    };
    if !path_is_writable(&target.exe) {
        bail!("{} is not writable.", target.exe.display());
    }

    eprintln!(
        "⬇️ {} mesh-llm v{} -> v{} ({})...",
        describe_requested_update(
            &release.version,
            options.current_version,
            requested_version.is_some()
        ),
        options.current_version,
        release.version,
        target.bundle_flavor.suffix()
    );
    match install_latest_bundle(
        &target.exe,
        &target.install_dir,
        &release,
        &asset_name,
        target.bundle_flavor,
        PostInstallAction::ExitAfterInstall,
    )
    .await
    {
        Ok(InstallOutcome::ExitNow) => {
            eprintln!("✅ Updated to v{}", release.version);
            Ok(())
        }
        Ok(InstallOutcome::HandoffAndExit) => {
            eprintln!(
                "✅ Applying update to v{}; exiting so the installer can finish",
                release.version
            );
            std::process::exit(0);
        }
        Ok(InstallOutcome::RestartNow) => {
            eprintln!("✅ Updated to v{}", release.version);
            Ok(())
        }
        Err(err) => Err(err),
    }
}
