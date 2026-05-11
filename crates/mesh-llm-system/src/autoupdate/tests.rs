#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(version_newer("0.33.0", "0.33.0-rc.1"));
        assert!(!version_newer("0.33.0-rc.1", "0.33.0"));
        assert!(version_newer("0.33.0-rc.2", "0.33.0-rc.1"));
    }

    #[test]
    #[serial]
    fn test_release_asset_url() {
        std::env::remove_var(SELF_UPDATE_REPO_ENV);
        assert_eq!(
            release_asset_url("v0.60.0", "mesh-llm-aarch64-apple-darwin.tar.gz"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.60.0/mesh-llm-aarch64-apple-darwin.tar.gz"
        );
    }

    #[test]
    #[serial]
    fn test_release_repo_defaults_to_main_repo() {
        std::env::remove_var(SELF_UPDATE_REPO_ENV);
        assert_eq!(release_repo(), "Mesh-LLM/mesh-llm");
        assert_eq!(
            latest_release_api_url(),
            "https://api.github.com/repos/Mesh-LLM/mesh-llm/releases/latest"
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
            release_api_url_for_tag("v0.60.0"),
            "https://api.github.com/repos/jdumay/mesh-llm/releases/tags/v0.60.0"
        );
        assert_eq!(
            release_asset_url("v0.60.0", "mesh-llm-x86_64-unknown-linux-gnu.tar.gz"),
            "https://github.com/jdumay/mesh-llm/releases/download/v0.60.0/mesh-llm-x86_64-unknown-linux-gnu.tar.gz"
        );
        std::env::remove_var(SELF_UPDATE_REPO_ENV);
    }

    #[test]
    fn test_normalize_release_tag() {
        assert_eq!(normalize_release_tag("v0.60.0").unwrap(), "v0.60.0");
        assert_eq!(
            normalize_release_tag("0.60.0-rc.1").unwrap(),
            "v0.60.0-rc.1"
        );
        assert!(normalize_release_tag("latest").is_err());
    }

    #[test]
    fn test_describe_requested_update() {
        assert_eq!(
            describe_requested_update("0.60.0", "0.65.1", false),
            "Updating"
        );
        assert_eq!(
            describe_requested_update("0.65.1", "0.65.1", true),
            "Reinstalling"
        );
        assert_eq!(
            describe_requested_update("0.0.1", "0.65.1", true),
            "Downgrading"
        );
        assert_eq!(
            describe_requested_update("999.0.0", "0.65.1", true),
            "Installing"
        );
    }

    #[test]
    fn test_stable_release_asset_name_matches_platform() {
        let expected = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => Some((
                backend::BinaryFlavor::Metal,
                "mesh-llm-aarch64-apple-darwin.tar.gz",
            )),
            ("linux", "x86_64") => Some((
                backend::BinaryFlavor::Cpu,
                "mesh-llm-x86_64-unknown-linux-gnu.tar.gz",
            )),
            _ => None,
        };

        let Some((flavor, asset)) = expected else {
            return;
        };
        assert_eq!(
            stable_release_asset_name_for(std::env::consts::OS, std::env::consts::ARCH, flavor),
            Some(asset.to_string())
        );
    }

    #[test]
    fn test_windows_release_asset_names() {
        assert!(platform_has_release_assets_for("windows", "x86_64"));
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Cpu),
            Some("mesh-llm-x86_64-pc-windows-msvc.zip".to_string())
        );
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Cuda),
            Some("mesh-llm-x86_64-pc-windows-msvc-cuda.zip".to_string())
        );
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Rocm),
            Some("mesh-llm-x86_64-pc-windows-msvc-rocm.zip".to_string())
        );
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Vulkan),
            Some("mesh-llm-x86_64-pc-windows-msvc-vulkan.zip".to_string())
        );
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec!["mesh-llm-v0.60.0-x86_64-pc-windows-msvc.zip".to_string()],
        };
        assert!(release_has_any_platform_asset(
            &release, "windows", "x86_64"
        ));
        let empty_release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: Vec::new(),
        };
        assert!(!release_has_any_platform_asset(
            &empty_release,
            "windows",
            "x86_64"
        ));
    }

    #[test]
    fn test_linux_arm64_release_asset_names() {
        let stable_asset = "mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string();
        assert!(platform_has_release_assets_for("linux", "aarch64"));
        assert_eq!(
            stable_release_asset_name_for("linux", "aarch64", backend::BinaryFlavor::Cpu),
            Some(stable_asset.clone())
        );

        let published_release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![stable_asset],
        };
        assert!(release_has_any_platform_asset(
            &published_release,
            "linux",
            "aarch64"
        ));

        let missing_release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: Vec::new(),
        };
        assert!(!release_has_any_platform_asset(
            &missing_release,
            "linux",
            "aarch64"
        ));
    }

    #[test]
    fn test_linux_arm64_aliases_resolve_identical_release_assets() {
        let arm64_asset =
            stable_release_asset_name_for("linux", "arm64", backend::BinaryFlavor::Cpu);
        let aarch64_asset =
            stable_release_asset_name_for("linux", "aarch64", backend::BinaryFlavor::Cpu);
        assert_eq!(arm64_asset, aarch64_asset);
        assert_eq!(
            arm64_asset,
            Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_prefers_stable_linux_arm64_asset() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![
                "mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string(),
                "mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string(),
            ],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "arm64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::StableFirst,
            ),
            Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_falls_back_to_versioned_linux_arm64_asset() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec!["mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string()],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "aarch64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::StableFirst,
            ),
            Some("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_prefers_versioned_for_explicit_install() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![
                "mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string(),
                "mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string(),
            ],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "aarch64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::VersionedFirst,
            ),
            Some("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_versioned_first_falls_back_to_stable() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec!["mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string()],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "arm64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::VersionedFirst,
            ),
            Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_macos_legacy_bundle_asset_remains_compatible() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec!["mesh-bundle.tar.gz".to_string()],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::new(
                    CanonicalOs::Macos,
                    CanonicalArch::Aarch64,
                    backend::BinaryFlavor::Metal,
                ),
                ReleaseAssetPreference::StableFirst,
            ),
            Some("mesh-bundle.tar.gz".to_string())
        );
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
    fn test_should_attempt_auto_update_only_when_flag_is_set() {
        std::env::remove_var(SELF_UPDATE_ATTEMPTED_ENV);
        assert!(should_attempt_auto_update(AutoUpdateOptions {
            auto_update: true,
            plugin_requested: false,
            command_is_update: false,
            llama_flavor: None,
            current_version: "0.65.1",
        }));

        assert!(!should_attempt_auto_update(AutoUpdateOptions {
            auto_update: false,
            plugin_requested: false,
            command_is_update: false,
            llama_flavor: None,
            current_version: "0.65.1",
        }));

        assert!(!should_attempt_auto_update(AutoUpdateOptions {
            auto_update: true,
            plugin_requested: false,
            command_is_update: true,
            llama_flavor: None,
            current_version: "0.65.1",
        }));
    }

    #[test]
    #[serial]
    fn test_should_attempt_auto_update_respects_restart_guard() {
        std::env::set_var(SELF_UPDATE_ATTEMPTED_ENV, "1");
        assert!(!should_attempt_auto_update(AutoUpdateOptions {
            auto_update: true,
            plugin_requested: false,
            command_is_update: false,
            llama_flavor: None,
            current_version: "0.65.1",
        }));
        std::env::remove_var(SELF_UPDATE_ATTEMPTED_ENV);
    }

    #[test]
    fn test_bundle_install_dir_uses_requested_or_platform_default_flavor() {
        let dir = temp_dir("bundle-install");
        let exe = dir.join(mesh_binary_name());
        std::fs::write(&exe, b"binary").unwrap();
        let default_flavor = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            backend::BinaryFlavor::Metal
        } else {
            backend::BinaryFlavor::Cpu
        };

        assert_eq!(
            bundle_install_dir(&exe, None),
            Some((dir.clone(), default_flavor))
        );
        assert_eq!(
            bundle_install_dir(&exe, Some(backend::BinaryFlavor::Vulkan)),
            Some((dir.clone(), backend::BinaryFlavor::Vulkan))
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
