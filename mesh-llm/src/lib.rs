mod api;
mod cli;
pub mod crypto;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
mod plugins;
mod protocol;
mod runtime;
mod system;

pub mod proto {
    pub mod node {
        include!(concat!(env!("OUT_DIR"), "/meshllm.node.v1.rs"));
    }
}

pub(crate) use plugins::blackboard;

use anyhow::Result;

pub const VERSION: &str = "0.60.0-rc.4";

pub async fn run() -> Result<()> {
    runtime::run().await
}

#[cfg(test)]
fn fixture_flavor(name: &str) -> BinaryFlavor {
    match name {
        "cpu" => BinaryFlavor::Cpu,
        "cuda" => BinaryFlavor::Cuda,
        "rocm" => BinaryFlavor::Rocm,
        "vulkan" => BinaryFlavor::Vulkan,
        "metal" => BinaryFlavor::Metal,
        other => panic!("unknown fixture flavor: {other}"),
    }
}

#[cfg(test)]
fn insert_release_tag(stable_asset: &str) -> String {
    let suffix = stable_asset
        .strip_prefix("mesh-llm-")
        .expect("fixture assets must start with mesh-llm-");
    format!("mesh-llm-{FIXTURE_RELEASE_TAG}-{suffix}")
}

#[cfg(test)]
fn temp_dir(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
#[test]
fn release_target_fixture() {
    for row in fixture_rows() {
        let flavor = fixture_flavor(&row.flavor);
        assert_eq!(
            system::autoupdate::release_target_stable_asset_name_for(&row.os, &row.arch, flavor),
            row.stable_asset,
            "stable asset mismatch for {} {} {}",
            row.os,
            row.arch,
            row.flavor
        );

        assert_eq!(
            system::autoupdate::release_target_has_release_assets_for(&row.os, &row.arch),
            row.support == "supported",
            "support mismatch for {} {} {}",
            row.os,
            row.arch,
            row.flavor
        );

        assert_eq!(
            system::autoupdate::release_target_recognizes_unsupported_for(&row.os, &row.arch),
            row.support == "recognized-unsupported",
            "recognized unsupported mismatch for {} {} {}",
            row.os,
            row.arch,
            row.flavor
        );

        if let Some(stable_asset) = row.stable_asset.as_deref() {
            assert_eq!(
                row.versioned_asset.as_deref(),
                Some(insert_release_tag(stable_asset).as_str()),
                "versioned fixture asset mismatch for {} {} {}",
                row.os,
                row.arch,
                row.flavor
            );
        } else {
            assert!(row.versioned_asset.is_none());
        }

        if row.os == "linux" && row.arch == "x86_64" {
            assert_eq!(
                cli::release_target_versioned_linux_asset_name(FIXTURE_RELEASE_TAG, flavor,),
                row.versioned_asset,
                "linux versioned asset mismatch for {} {} {}",
                row.os,
                row.arch,
                row.flavor
            );
        }
    }
}

#[cfg(test)]
#[test]
fn release_target_renders_stable_assets() {
    for row in fixture_rows() {
        let flavor = fixture_flavor(&row.flavor);
        assert_eq!(
            system::autoupdate::release_target_stable_asset_name_for(&row.os, &row.arch, flavor),
            row.stable_asset,
            "stable asset mismatch for {} {} {}",
            row.os,
            row.arch,
            row.flavor
        );
    }
}

#[cfg(test)]
#[test]
fn release_target_renders_versioned_assets() {
    for row in fixture_rows()
        .into_iter()
        .filter(|row| row.os == "linux" && row.arch == "x86_64")
    {
        let flavor = fixture_flavor(&row.flavor);
        assert_eq!(
            cli::release_target_versioned_linux_asset_name(FIXTURE_RELEASE_TAG, flavor,),
            row.versioned_asset,
            "versioned asset mismatch for {} {} {}",
            row.os,
            row.arch,
            row.flavor
        );
    }
}

#[cfg(test)]
#[test]
fn release_assets_match_expected_linux_names() {
    let linux_rows: Vec<_> = fixture_rows()
        .into_iter()
        .filter(|row| row.os == "linux" && row.arch == "x86_64")
        .collect();

    assert_eq!(linux_rows.len(), 4, "expected four x86_64 linux HF rows");
    for row in linux_rows {
        let flavor = fixture_flavor(&row.flavor);
        assert_eq!(
            cli::release_target_versioned_linux_asset_name(FIXTURE_RELEASE_TAG, flavor,),
            row.versioned_asset,
            "HF job linux release asset mismatch for {} {} {}",
            row.os,
            row.arch,
            row.flavor
        );
    }
}

#[cfg(test)]
#[test]
fn job_images_match_runtime_targets() {
    use crate::cli::commands::moe::hf_jobs::release_target_job_image_for;
    use cli::moe::HfJobReleaseTarget;

    assert_eq!(
        release_target_job_image_for(HfJobReleaseTarget::Cpu),
        "ghcr.io/astral-sh/uv:python3.12-bookworm"
    );
    assert_eq!(
        release_target_job_image_for(HfJobReleaseTarget::Cuda),
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
    );
    assert_eq!(
        release_target_job_image_for(HfJobReleaseTarget::Rocm),
        "rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0"
    );
    assert_eq!(
        release_target_job_image_for(HfJobReleaseTarget::Vulkan),
        "ghcr.io/astral-sh/uv:python3.12-bookworm"
    );
}

#[cfg(test)]
#[test]
fn release_target_hf_jobs_parity() {
    let linux_rows: Vec<_> = fixture_rows()
        .into_iter()
        .filter(|row| row.os == "linux" && row.arch == "x86_64")
        .collect();

    assert_eq!(linux_rows.len(), 4, "expected four x86_64 linux HF rows");
    for row in linux_rows {
        let flavor = fixture_flavor(&row.flavor);
        let asset = cli::release_target_versioned_linux_asset_name(FIXTURE_RELEASE_TAG, flavor);
        println!("{} -> {:?}", row.flavor, asset);
        assert_eq!(
            asset, row.versioned_asset,
            "HF jobs parity mismatch for {} {} {}",
            row.os, row.arch, row.flavor
        );
    }
}

#[cfg(test)]
#[test]
fn release_target_normalizes_aliases() {
    assert_eq!(
        system::release_target::release_target_canonical_arch_for("arm64"),
        Some("aarch64")
    );
    assert_eq!(
        system::release_target::release_target_canonical_arch_for("aarch64"),
        Some("aarch64")
    );
    assert_eq!(
        system::release_target::release_target_canonical_arch_for("amd64"),
        Some("x86_64")
    );
    assert_eq!(
        system::release_target::release_target_canonical_arch_for("armv7l"),
        Some("arm")
    );
    assert_eq!(
        system::release_target::release_target_canonical_arch_for("armv6hf"),
        Some("arm")
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "arm64",
            BinaryFlavor::Cpu
        ),
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "aarch64",
            BinaryFlavor::Cpu
        )
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "amd64",
            BinaryFlavor::Cpu
        ),
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "x86_64",
            BinaryFlavor::Cpu
        )
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "armv7l",
            BinaryFlavor::Cpu
        ),
        None
    );
    assert!(system::autoupdate::release_target_recognizes_unsupported_for("linux", "armv7l"));
    assert!(system::autoupdate::release_target_recognizes_unsupported_for("linux", "armv6hf"));
}

#[cfg(test)]
#[test]
fn release_target_rejects_unknown_arch() {
    assert_eq!(
        system::release_target::release_target_canonical_arch_for("mips64"),
        None
    );
    assert_eq!(
        system::release_target::release_target_parse_error_message_for(
            "linux",
            "mips64",
            BinaryFlavor::Cpu,
        ),
        Some("unknown release target arch: mips64".to_string())
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "mips64",
            BinaryFlavor::Cpu
        ),
        None
    );
    assert!(!system::autoupdate::release_target_has_release_assets_for(
        "linux", "mips64"
    ));
    assert!(!system::autoupdate::release_target_recognizes_unsupported_for("linux", "mips64"));
}

#[cfg(test)]
#[test]
fn release_target_linux_arm64_cpu_asset() {
    let row = fixture_rows()
        .into_iter()
        .find(|row| row.os == "linux" && row.arch == "aarch64" && row.flavor == "cpu")
        .expect("linux arm64 cpu row should exist");

    assert_eq!(row.support, "supported");
    assert_eq!(
        row.stable_asset.as_deref(),
        Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz")
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "aarch64",
            BinaryFlavor::Cpu
        ),
        row.stable_asset
    );
    assert!(system::autoupdate::release_target_has_release_assets_for(
        "linux", "aarch64"
    ));
}

#[cfg(test)]
#[test]
fn release_target_recognizes_unsupported_arm32() {
    let row = fixture_rows()
        .into_iter()
        .find(|row| row.os == "linux" && row.arch == "arm" && row.flavor == "cpu")
        .expect("linux arm32 row should exist");

    assert_eq!(row.support, "recognized-unsupported");
    assert_eq!(row.stable_asset, None);
    assert_eq!(row.versioned_asset, None);
    assert!(!system::autoupdate::release_target_has_release_assets_for(
        "linux", "arm"
    ));
    assert!(system::autoupdate::release_target_recognizes_unsupported_for("linux", "arm"));
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for("linux", "arm", BinaryFlavor::Cpu),
        None
    );
}

#[cfg(test)]
#[test]
fn test_stable_release_asset_name_matches_platform() {
    let expected = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some((BinaryFlavor::Metal, "mesh-llm-aarch64-apple-darwin.tar.gz")),
        ("linux", "x86_64") => Some((
            BinaryFlavor::Cpu,
            "mesh-llm-x86_64-unknown-linux-gnu.tar.gz",
        )),
        _ => None,
    };

    let Some((flavor, asset)) = expected else {
        return;
    };

    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            std::env::consts::OS,
            std::env::consts::ARCH,
            flavor,
        ),
        Some(asset.to_string())
    );
}

#[cfg(test)]
#[test]
fn test_linux_arm64_release_asset_names() {
    let stable_asset = "mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string();
    assert!(system::autoupdate::release_target_has_release_assets_for(
        "linux", "aarch64"
    ));
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "linux",
            "aarch64",
            BinaryFlavor::Cpu
        ),
        Some(stable_asset.clone())
    );
    assert!(
        system::autoupdate::release_target_platform_asset_published_for(
            "linux",
            "aarch64",
            FIXTURE_RELEASE_TAG,
            std::slice::from_ref(&stable_asset),
        )
    );
    assert!(
        !system::autoupdate::release_target_platform_asset_published_for(
            "linux",
            "aarch64",
            FIXTURE_RELEASE_TAG,
            &[],
        )
    );
}

#[cfg(test)]
#[test]
fn test_bundle_install_dir_requires_matching_flavor_pair() {
    let dir = temp_dir("bundle-install");
    let exe = dir.join(if cfg!(windows) {
        "mesh-llm.exe"
    } else {
        "mesh-llm"
    });
    std::fs::write(&exe, b"binary").unwrap();
    std::fs::write(
        dir.join(if cfg!(windows) {
            "rpc-server-cpu.exe"
        } else {
            "rpc-server-cpu"
        }),
        b"rpc",
    )
    .unwrap();
    std::fs::write(
        dir.join(if cfg!(windows) {
            "llama-server-cpu.exe"
        } else {
            "llama-server-cpu"
        }),
        b"llama",
    )
    .unwrap();

    assert_eq!(
        system::autoupdate::release_target_bundle_install_dir_for(&exe, None),
        Some((dir.clone(), BinaryFlavor::Cpu))
    );
    assert_eq!(
        system::autoupdate::release_target_bundle_install_dir_for(&exe, Some(BinaryFlavor::Cpu)),
        Some((dir.clone(), BinaryFlavor::Cpu))
    );
    assert_eq!(
        system::autoupdate::release_target_bundle_install_dir_for(&exe, Some(BinaryFlavor::Cuda)),
        None
    );

    let missing = temp_dir("bundle-missing");
    let missing_exe = missing.join(if cfg!(windows) {
        "mesh-llm.exe"
    } else {
        "mesh-llm"
    });
    std::fs::write(&missing_exe, b"binary").unwrap();
    assert_eq!(
        system::autoupdate::release_target_bundle_install_dir_for(&missing_exe, None),
        None
    );

    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(missing);
}

#[cfg(test)]
#[test]
fn test_windows_release_asset_names() {
    assert!(system::autoupdate::release_target_has_release_assets_for(
        "windows", "x86_64"
    ));
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "windows",
            "x86_64",
            BinaryFlavor::Cpu
        ),
        Some("mesh-llm-x86_64-pc-windows-msvc.zip".to_string())
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "windows",
            "x86_64",
            BinaryFlavor::Cuda
        ),
        Some("mesh-llm-x86_64-pc-windows-msvc-cuda.zip".to_string())
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "windows",
            "x86_64",
            BinaryFlavor::Rocm
        ),
        Some("mesh-llm-x86_64-pc-windows-msvc-rocm.zip".to_string())
    );
    assert_eq!(
        system::autoupdate::release_target_stable_asset_name_for(
            "windows",
            "x86_64",
            BinaryFlavor::Vulkan
        ),
        Some("mesh-llm-x86_64-pc-windows-msvc-vulkan.zip".to_string())
    );
    assert!(
        system::autoupdate::release_target_platform_asset_published_for(
            "windows",
            "x86_64",
            FIXTURE_RELEASE_TAG,
            &["mesh-llm-x86_64-pc-windows-msvc.zip".to_string()],
        )
    );
    assert!(
        !system::autoupdate::release_target_platform_asset_published_for(
            "windows",
            "x86_64",
            FIXTURE_RELEASE_TAG,
            &[],
        )
    );
}
