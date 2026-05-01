use crate::inference::launch::BinaryFlavor;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CanonicalOs {
    Macos,
    Linux,
    Windows,
}

impl CanonicalOs {
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "macos" | "darwin" => Some(Self::Macos),
            "linux" => Some(Self::Linux),
            "windows" => Some(Self::Windows),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CanonicalArch {
    X86_64,
    Aarch64,
    Arm,
}

impl CanonicalArch {
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "x86_64" | "amd64" => Some(Self::X86_64),
            "aarch64" | "arm64" => Some(Self::Aarch64),
            "arm" => Some(Self::Arm),
            _ if normalized.starts_with("armv6") || normalized.starts_with("armv7") => {
                Some(Self::Arm)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArchiveKind {
    TarGz,
    Zip,
}

impl ArchiveKind {
    pub(crate) fn extension(self) -> &'static str {
        match self {
            Self::TarGz => "tar.gz",
            Self::Zip => "zip",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SupportStatus {
    Supported,
    RecognizedUnsupported,
    Unknown,
}

impl SupportStatus {
    pub(crate) fn is_supported(self) -> bool {
        self == Self::Supported
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReleaseTargetParseError {
    UnknownOs(String),
    UnknownArch(String),
}

impl std::fmt::Display for ReleaseTargetParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOs(os) => write!(f, "unknown release target os: {os}"),
            Self::UnknownArch(arch) => write!(f, "unknown release target arch: {arch}"),
        }
    }
}

impl std::error::Error for ReleaseTargetParseError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReleaseTarget {
    os: CanonicalOs,
    arch: CanonicalArch,
    flavor: BinaryFlavor,
}

impl ReleaseTarget {
    pub(crate) fn new(os: CanonicalOs, arch: CanonicalArch, flavor: BinaryFlavor) -> Self {
        Self { os, arch, flavor }
    }

    pub(crate) fn from_raw(
        os: &str,
        arch: &str,
        flavor: BinaryFlavor,
    ) -> Result<Self, ReleaseTargetParseError> {
        let os = CanonicalOs::parse(os)
            .ok_or_else(|| ReleaseTargetParseError::UnknownOs(os.to_string()))?;
        let arch = CanonicalArch::parse(arch)
            .ok_or_else(|| ReleaseTargetParseError::UnknownArch(arch.to_string()))?;
        Ok(Self::new(os, arch, flavor))
    }

    pub(crate) fn support_status(self) -> SupportStatus {
        match (self.os, self.arch, self.flavor) {
            (CanonicalOs::Macos, CanonicalArch::Aarch64, BinaryFlavor::Metal)
            | (CanonicalOs::Linux, CanonicalArch::X86_64, BinaryFlavor::Cpu)
            | (CanonicalOs::Linux, CanonicalArch::X86_64, BinaryFlavor::Cuda)
            | (CanonicalOs::Linux, CanonicalArch::X86_64, BinaryFlavor::Rocm)
            | (CanonicalOs::Linux, CanonicalArch::X86_64, BinaryFlavor::Vulkan)
            | (CanonicalOs::Linux, CanonicalArch::Aarch64, BinaryFlavor::Cpu)
            | (CanonicalOs::Windows, CanonicalArch::X86_64, BinaryFlavor::Cpu)
            | (CanonicalOs::Windows, CanonicalArch::X86_64, BinaryFlavor::Cuda)
            | (CanonicalOs::Windows, CanonicalArch::X86_64, BinaryFlavor::Rocm)
            | (CanonicalOs::Windows, CanonicalArch::X86_64, BinaryFlavor::Vulkan) => {
                SupportStatus::Supported
            }
            (CanonicalOs::Linux, CanonicalArch::Arm, _) => SupportStatus::RecognizedUnsupported,
            _ => SupportStatus::Unknown,
        }
    }

    fn archive_kind(self) -> ArchiveKind {
        match self.os {
            CanonicalOs::Macos | CanonicalOs::Linux => ArchiveKind::TarGz,
            CanonicalOs::Windows => ArchiveKind::Zip,
        }
    }

    fn target_triple(self) -> Option<&'static str> {
        match (self.os, self.arch) {
            (CanonicalOs::Macos, CanonicalArch::Aarch64) => Some("aarch64-apple-darwin"),
            (CanonicalOs::Linux, CanonicalArch::X86_64) => Some("x86_64-unknown-linux-gnu"),
            (CanonicalOs::Linux, CanonicalArch::Aarch64) => Some("aarch64-unknown-linux-gnu"),
            (CanonicalOs::Linux, CanonicalArch::Arm) => Some("arm-unknown-linux-gnueabihf"),
            (CanonicalOs::Windows, CanonicalArch::X86_64) => Some("x86_64-pc-windows-msvc"),
            _ => None,
        }
    }

    pub(crate) fn stable_asset_name(self) -> Option<String> {
        self.asset_name(None)
    }

    pub(crate) fn versioned_asset_name(self, release_tag: &str) -> Option<String> {
        self.asset_name(Some(release_tag))
    }

    fn asset_name(self, release_tag: Option<&str>) -> Option<String> {
        if self.support_status() != SupportStatus::Supported {
            return None;
        }

        let triple = self.target_triple()?;
        let archive = self.archive_kind().extension();
        let flavor_suffix = match self.flavor {
            BinaryFlavor::Cpu | BinaryFlavor::Metal => "",
            BinaryFlavor::Cuda => "-cuda",
            BinaryFlavor::Rocm => "-rocm",
            BinaryFlavor::Vulkan => "-vulkan",
        };

        match release_tag {
            Some(tag) => Some(format!("mesh-llm-{tag}-{triple}{flavor_suffix}.{archive}")),
            None => Some(format!("mesh-llm-{triple}{flavor_suffix}.{archive}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::path::Path;

    const FIXTURE_RELEASE_TAG: &str = "v0.60.0";

    #[derive(Debug, Deserialize)]
    struct ReleaseTargetRow {
        os: String,
        arch: String,
        flavor: String,
        support: String,
        stable_asset: Option<String>,
        versioned_asset: Option<String>,
    }

    fn fixture_rows() -> Vec<ReleaseTargetRow> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("release-target-matrix.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn flavor(name: &str) -> BinaryFlavor {
        match name {
            "cpu" => BinaryFlavor::Cpu,
            "cuda" => BinaryFlavor::Cuda,
            "rocm" => BinaryFlavor::Rocm,
            "vulkan" => BinaryFlavor::Vulkan,
            "metal" => BinaryFlavor::Metal,
            other => panic!("unknown fixture flavor: {other}"),
        }
    }

    #[test]
    fn release_target_renders_stable_assets() {
        for row in fixture_rows() {
            let target = ReleaseTarget::from_raw(&row.os, &row.arch, flavor(&row.flavor)).unwrap();
            assert_eq!(
                target.stable_asset_name(),
                row.stable_asset,
                "stable asset mismatch for {} {} {}",
                row.os,
                row.arch,
                row.flavor
            );
        }
    }

    #[test]
    fn release_target_renders_versioned_assets() {
        for row in fixture_rows() {
            let target = ReleaseTarget::from_raw(&row.os, &row.arch, flavor(&row.flavor)).unwrap();
            assert_eq!(
                target.versioned_asset_name(FIXTURE_RELEASE_TAG),
                row.versioned_asset,
                "versioned asset mismatch for {} {} {}",
                row.os,
                row.arch,
                row.flavor
            );
        }
    }

    #[test]
    fn release_target_normalizes_aliases() {
        assert_eq!(
            ReleaseTarget::from_raw("linux", "arm64", BinaryFlavor::Cpu)
                .unwrap()
                .arch,
            CanonicalArch::Aarch64
        );
        assert_eq!(
            ReleaseTarget::from_raw("linux", "aarch64", BinaryFlavor::Cpu)
                .unwrap()
                .arch,
            CanonicalArch::Aarch64
        );
        assert_eq!(
            ReleaseTarget::from_raw("linux", "amd64", BinaryFlavor::Cpu)
                .unwrap()
                .arch,
            CanonicalArch::X86_64
        );
        assert_eq!(
            ReleaseTarget::from_raw("linux", "armv7l", BinaryFlavor::Cpu)
                .unwrap()
                .arch,
            CanonicalArch::Arm
        );
        assert_eq!(
            ReleaseTarget::from_raw("linux", "armv6hf", BinaryFlavor::Cpu)
                .unwrap()
                .arch,
            CanonicalArch::Arm
        );
    }

    #[test]
    fn release_target_arm64_aliases_have_identical_linux_assets() {
        let arm64 = ReleaseTarget::from_raw("linux", "arm64", BinaryFlavor::Cpu).unwrap();
        let aarch64 = ReleaseTarget::from_raw("linux", "aarch64", BinaryFlavor::Cpu).unwrap();

        assert_eq!(arm64.support_status(), aarch64.support_status());
        assert_eq!(arm64.stable_asset_name(), aarch64.stable_asset_name());
        assert_eq!(
            arm64.versioned_asset_name(FIXTURE_RELEASE_TAG),
            aarch64.versioned_asset_name(FIXTURE_RELEASE_TAG)
        );
        assert_eq!(
            arm64.stable_asset_name(),
            Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn release_target_rejects_unknown_arch() {
        assert_eq!(
            ReleaseTarget::from_raw("linux", "mips64", BinaryFlavor::Cpu),
            Err(ReleaseTargetParseError::UnknownArch("mips64".to_string()))
        );
    }

    #[test]
    fn release_target_round_trips_fixture_matrix() {
        for row in fixture_rows() {
            let target = ReleaseTarget::from_raw(&row.os, &row.arch, flavor(&row.flavor)).unwrap();
            assert_eq!(
                target.arch,
                CanonicalArch::parse(&row.arch).unwrap(),
                "arch mismatch for {} {} {}",
                row.os,
                row.arch,
                row.flavor
            );
            assert_eq!(
                target.flavor,
                flavor(&row.flavor),
                "flavor mismatch for {} {} {}",
                row.os,
                row.arch,
                row.flavor
            );

            let expected_support = match row.support.as_str() {
                "supported" => SupportStatus::Supported,
                "recognized-unsupported" => SupportStatus::RecognizedUnsupported,
                "unknown" => SupportStatus::Unknown,
                other => panic!("unknown fixture support status: {other}"),
            };
            assert_eq!(
                target.support_status(),
                expected_support,
                "support mismatch for {} {} {}",
                row.os,
                row.arch,
                row.flavor
            );
        }
    }
}
