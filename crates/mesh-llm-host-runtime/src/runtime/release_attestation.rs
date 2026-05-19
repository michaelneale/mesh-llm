use crate::ReleaseBuildAttestation;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const RELEASE_ATTESTATION_PATH_ENV: &str = "MESH_LLM_RELEASE_ATTESTATION_PATH";

#[derive(Debug, Clone)]
pub(crate) struct LoadedReleaseAttestation {
    pub(crate) path: PathBuf,
    pub(crate) attestation: ReleaseBuildAttestation,
}

pub(crate) fn load_for_current_binary() -> Result<Option<LoadedReleaseAttestation>> {
    let binary_path =
        std::env::current_exe().context("failed to determine mesh-llm binary path")?;
    load_for_binary_path(&binary_path)
}

pub(crate) fn sibling_attestation_path(binary_path: &Path) -> PathBuf {
    let file_name = binary_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("mesh-llm");
    binary_path.with_file_name(format!("{file_name}.attestation.json"))
}

fn resolve_attestation_path(binary_path: &Path) -> PathBuf {
    std::env::var(RELEASE_ATTESTATION_PATH_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| sibling_attestation_path(binary_path))
}

fn load_for_binary_path(binary_path: &Path) -> Result<Option<LoadedReleaseAttestation>> {
    let path = resolve_attestation_path(binary_path);
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read(&path)
        .with_context(|| format!("failed to read release attestation {}", path.display()))?;
    let attestation = serde_json::from_slice(&raw)
        .with_context(|| format!("failed to parse release attestation {}", path.display()))?;
    Ok(Some(LoadedReleaseAttestation { path, attestation }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_path_uses_binary_name() {
        let path = sibling_attestation_path(Path::new("/tmp/mesh-llm"));
        assert_eq!(path, PathBuf::from("/tmp/mesh-llm.attestation.json"));
    }
}
