use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use skippy_protocol::{LoadMode, StageConfig};
use skippy_runtime::package::{self, LayerPackageInfo, PackageStageRequest};

use super::StageLoadRequest;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StagePackageRef {
    LocalPackage(PathBuf),
    HuggingFacePackage {
        repo: String,
        revision: Option<String>,
    },
    SyntheticDirectGguf(PathBuf),
}

impl StagePackageRef {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        if let Some(rest) = value.strip_prefix("hf://") {
            let (repo, revision) = if let Some((repo, revision)) = rest.split_once('@') {
                (repo, Some(revision.to_string()))
            } else if let Some(index) = rest.rfind(':') {
                (&rest[..index], Some(rest[index + 1..].to_string()))
            } else {
                (rest, None)
            };
            if repo.split('/').count() != 2 || repo.contains(':') || repo.contains('@') {
                bail!("HF package repo id must look like namespace/repo");
            }
            return Ok(Self::HuggingFacePackage {
                repo: repo.to_string(),
                revision,
            });
        }

        let path = PathBuf::from(value);
        if path.join("model-package.json").is_file() {
            return Ok(Self::LocalPackage(path));
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            return Ok(Self::SyntheticDirectGguf(path));
        }

        bail!("not a skippy package ref: {value}");
    }

    pub(crate) fn is_distributable_package(&self) -> bool {
        matches!(
            self,
            Self::LocalPackage(_) | Self::HuggingFacePackage { .. }
        )
    }

    pub(crate) fn as_package_ref(&self) -> Option<String> {
        match self {
            Self::LocalPackage(path) => Some(path.to_string_lossy().to_string()),
            Self::HuggingFacePackage { repo, revision } => Some(match revision {
                Some(revision) => format!("hf://{repo}@{revision}"),
                None => format!("hf://{repo}"),
            }),
            Self::SyntheticDirectGguf(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StagePackageInfo {
    pub(crate) package_ref: String,
    pub(crate) package_dir: PathBuf,
    pub(crate) manifest_sha256: String,
    pub(crate) model_id: String,
    pub(crate) source_model_path: String,
    pub(crate) source_model_sha256: String,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) layer_count: u32,
    pub(crate) activation_width: u32,
    pub(crate) layers: Vec<StagePackageLayerInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StagePackageLayerInfo {
    pub(crate) layer_index: u32,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaterializedStageArtifact {
    pub(crate) path: PathBuf,
    pub(crate) manifest_sha256: String,
    pub(crate) source_model_path: String,
    pub(crate) source_model_sha256: String,
    pub(crate) source_model_bytes: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct MaterializedStagePin {
    path: PathBuf,
}

impl Drop for MaterializedStagePin {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PinFile {
    artifact_path: PathBuf,
    package_ref: String,
    topology_id: String,
    run_id: String,
    stage_id: String,
}

pub(crate) fn configure_materialized_stage_cache() {
    if std::env::var_os("SKIPPY_MATERIALIZED_DIR").is_none() {
        std::env::set_var("SKIPPY_MATERIALIZED_DIR", materialized_stage_cache_dir());
    }
}

pub(crate) fn materialized_stage_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("skippy-stages")
}

pub(crate) fn is_layer_package_ref(value: &str) -> bool {
    StagePackageRef::parse(value).is_ok_and(|package_ref| package_ref.is_distributable_package())
}

/// Resolve an `hf://` package ref to a local directory, downloading the manifest
/// and required layer files using the `hf_hub` Rust library.
///
/// Returns the local directory path containing the package files.
/// If `package_ref` is already a local path, returns it as-is.
pub(crate) fn resolve_hf_package_to_local(
    package_ref: &str,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let parsed = StagePackageRef::parse(package_ref)?;
    let (repo, revision) = match &parsed {
        StagePackageRef::HuggingFacePackage { repo, revision } => (
            repo.clone(),
            revision.clone().unwrap_or_else(|| "main".to_string()),
        ),
        _ => return Ok(package_ref.to_string()),
    };

    let api = crate::models::build_hf_api(false)?;
    let (owner, name) = repo.split_once('/').context("invalid HF repo format")?;
    let model_api = api.model(owner, name);

    // Download manifest first
    let manifest_path = model_api
        .download_file(
            &hf_hub::RepoDownloadFileParams::builder()
                .filename("model-package.json".to_string())
                .revision(revision.clone())
                .build(),
        )
        .context("download layer package manifest")?;

    let package_dir = manifest_path
        .parent()
        .context("manifest has no parent directory")?
        .to_path_buf();

    // Read manifest to determine which files we need
    let manifest_contents = fs::read(&manifest_path).context("read package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse package manifest")?;

    // Collect the files we need to download
    let mut needed_files: Vec<String> = Vec::new();

    // Always need shared/metadata.gguf
    if let Some(path) = manifest
        .pointer("/shared/metadata/path")
        .and_then(|v| v.as_str())
    {
        needed_files.push(path.to_string());
    }
    if include_embeddings {
        if let Some(path) = manifest
            .pointer("/shared/embeddings/path")
            .and_then(|v| v.as_str())
        {
            needed_files.push(path.to_string());
        }
    }
    if include_output {
        if let Some(path) = manifest
            .pointer("/shared/output/path")
            .and_then(|v| v.as_str())
        {
            needed_files.push(path.to_string());
        }
    }

    // Layer files for assigned range
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for (i, layer) in layers.iter().enumerate() {
            let idx = i as u32;
            if idx >= layer_start && idx < layer_end {
                if let Some(path) = layer.get("path").and_then(|a| a.as_str()) {
                    needed_files.push(path.to_string());
                }
            }
        }
    }

    // Download each needed file
    for file in &needed_files {
        let local_path = package_dir.join(file);
        if local_path.is_file() {
            continue; // already cached
        }
        model_api
            .download_file(
                &hf_hub::RepoDownloadFileParams::builder()
                    .filename(file.clone())
                    .revision(revision.clone())
                    .build(),
            )
            .with_context(|| format!("download layer package file: {file}"))?;
    }

    Ok(package_dir.to_string_lossy().to_string())
}

pub(crate) fn inspect_stage_package(package_ref: &str) -> Result<StagePackageInfo> {
    // Resolve hf:// to local (only downloads manifest for inspection)
    let local_ref = resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let info = package::inspect_layer_package(&local_ref)
        .with_context(|| format!("inspect skippy layer package {package_ref}"))?;
    stage_package_info(package_ref, info)
}

pub(crate) fn materialize_stage_load(
    load: &StageLoadRequest,
) -> Result<Option<(MaterializedStageArtifact, MaterializedStagePin)>> {
    if load.load_mode != LoadMode::LayerPackage {
        return Ok(None);
    }
    let is_final = load.downstream.is_none();
    let include_embeddings = load.layer_start == 0 || is_final;
    // Resolve hf:// to local dir with needed files downloaded
    let local_ref = resolve_hf_package_to_local(
        &load.package_ref,
        load.layer_start,
        load.layer_end,
        include_embeddings,
        is_final,
    )?;
    let request = package_stage_request(
        &load.model_id,
        &load.topology_id,
        &local_ref,
        &load.stage_id,
        load.layer_start,
        load.layer_end,
        is_final,
    );
    let materialized = package::materialize_layer_package_details(&request).with_context(|| {
        format!(
            "materialize skippy stage package {} layers {}..{}",
            load.stage_id, load.layer_start, load.layer_end
        )
    })?;
    let info = package::inspect_layer_package(&local_ref)?;
    let artifact = MaterializedStageArtifact {
        path: materialized.output_path,
        manifest_sha256: materialized.manifest_sha256,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
    };
    let pin = pin_materialized_stage(
        &artifact.path,
        &load.package_ref,
        &load.topology_id,
        &load.run_id,
        &load.stage_id,
    )?;
    Ok(Some((artifact, pin)))
}

pub(crate) fn materialize_stage_config(
    config: &StageConfig,
) -> Result<Option<(MaterializedStageArtifact, MaterializedStagePin)>> {
    if config.load_mode != LoadMode::LayerPackage {
        return Ok(None);
    }
    let package_ref = config
        .model_path
        .as_deref()
        .or(config.package_ref.as_deref())
        .context("layer-package config is missing package ref")?;
    let is_final = config.downstream.is_none();
    let include_embeddings = config.layer_start == 0 || is_final;
    // Resolve hf:// to local dir with needed files downloaded
    let local_ref = resolve_hf_package_to_local(
        package_ref,
        config.layer_start,
        config.layer_end,
        include_embeddings,
        is_final,
    )?;
    let request = package_stage_request(
        &config.model_id,
        &config.topology_id,
        &local_ref,
        &config.stage_id,
        config.layer_start,
        config.layer_end,
        is_final,
    );
    let materialized = package::materialize_layer_package_details(&request).with_context(|| {
        format!(
            "materialize skippy stage package {} layers {}..{}",
            config.stage_id, config.layer_start, config.layer_end
        )
    })?;
    let info = package::inspect_layer_package(&local_ref)?;
    let artifact = MaterializedStageArtifact {
        path: materialized.output_path,
        manifest_sha256: materialized.manifest_sha256,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
    };
    let pin = pin_materialized_stage(
        &artifact.path,
        package_ref,
        &config.topology_id,
        &config.run_id,
        &config.stage_id,
    )?;
    Ok(Some((artifact, pin)))
}

pub(crate) fn prune_unpinned_materialized_stages() -> Result<usize> {
    let root = materialized_stage_cache_dir();
    if !root.is_dir() {
        return Ok(0);
    }
    let pins = active_pin_artifacts(&root)?;
    let mut removed = 0usize;
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        if pins.iter().any(|pin| pin == &path) {
            continue;
        }
        fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        removed += 1;
    }
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("source-") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(index) = serde_json::from_slice::<SourceIndex>(&bytes) else {
            continue;
        };
        if !index.artifact_path.exists() && !pins.iter().any(|pin| pin == &index.artifact_path) {
            let _ = fs::remove_file(path);
        }
    }
    Ok(removed)
}

pub(crate) fn remove_materialized_stages_for_sources(sources: &[PathBuf]) -> Result<usize> {
    let candidates = materialized_stage_removal_candidates(sources)?;
    let mut removed = 0usize;
    for candidate in candidates {
        if candidate.artifact_path.exists() {
            fs::remove_file(&candidate.artifact_path)
                .with_context(|| format!("remove {}", candidate.artifact_path.display()))?;
            removed += 1;
        }
        let _ = fs::remove_file(candidate.source_index_path);
    }
    Ok(removed)
}

pub(crate) fn materialized_stages_for_sources(sources: &[PathBuf]) -> Result<Vec<PathBuf>> {
    Ok(materialized_stage_removal_candidates(sources)?
        .into_iter()
        .filter(|candidate| candidate.artifact_path.exists())
        .map(|candidate| candidate.artifact_path)
        .collect())
}

fn materialized_stage_removal_candidates(
    sources: &[PathBuf],
) -> Result<Vec<MaterializedStageRemovalCandidate>> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let root = materialized_stage_cache_dir();
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let source_strings = sources
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let pins = active_pin_artifacts(&root)?;
    let mut candidates = Vec::new();
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("source-") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(index) = serde_json::from_slice::<SourceIndex>(&bytes) else {
            continue;
        };
        if !source_strings
            .iter()
            .any(|source| source == &index.source_model_path)
        {
            continue;
        }
        if pins.iter().any(|pin| pin == &index.artifact_path) {
            continue;
        }
        candidates.push(MaterializedStageRemovalCandidate {
            artifact_path: index.artifact_path,
            source_index_path: path,
        });
    }
    candidates.sort_by(|left, right| left.artifact_path.cmp(&right.artifact_path));
    Ok(candidates)
}

#[derive(Debug)]
struct MaterializedStageRemovalCandidate {
    artifact_path: PathBuf,
    source_index_path: PathBuf,
}

fn stage_package_info(package_ref: &str, info: LayerPackageInfo) -> Result<StagePackageInfo> {
    let activation_width = info.activation_width.with_context(|| {
        format!(
            "layer package {package_ref} is missing activation_width; rebuild the package manifest"
        )
    })?;
    Ok(StagePackageInfo {
        package_ref: package_ref.to_string(),
        package_dir: info.package_dir,
        manifest_sha256: info.manifest_sha256,
        model_id: info.model_id,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
        layer_count: info.layer_count,
        activation_width,
        layers: info
            .layers
            .into_iter()
            .map(|layer| StagePackageLayerInfo {
                layer_index: layer.layer_index,
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
            })
            .collect(),
    })
}

fn package_stage_request(
    model_id: &str,
    topology_id: &str,
    package_ref: &str,
    stage_id: &str,
    layer_start: u32,
    layer_end: u32,
    is_final_stage: bool,
) -> PackageStageRequest {
    PackageStageRequest {
        model_id: model_id.to_string(),
        topology_id: topology_id.to_string(),
        package_ref: package_ref.to_string(),
        stage_id: stage_id.to_string(),
        layer_start,
        layer_end,
        include_embeddings: layer_start == 0 || is_final_stage,
        include_output: is_final_stage,
    }
}

fn pin_materialized_stage(
    artifact_path: &Path,
    package_ref: &str,
    topology_id: &str,
    run_id: &str,
    stage_id: &str,
) -> Result<MaterializedStagePin> {
    let root = materialized_stage_cache_dir();
    let pin_dir = root.join("pins");
    fs::create_dir_all(&pin_dir).with_context(|| format!("create {}", pin_dir.display()))?;
    let pin = PinFile {
        artifact_path: artifact_path.to_path_buf(),
        package_ref: package_ref.to_string(),
        topology_id: topology_id.to_string(),
        run_id: run_id.to_string(),
        stage_id: stage_id.to_string(),
    };
    let pin_path = pin_dir.join(format!(
        "{}.json",
        cache_key(&format!(
            "{package_ref}\0{topology_id}\0{run_id}\0{stage_id}"
        ))
    ));
    fs::write(&pin_path, serde_json::to_vec_pretty(&pin)?)
        .with_context(|| format!("write {}", pin_path.display()))?;
    write_source_index(artifact_path, &pin)?;
    Ok(MaterializedStagePin { path: pin_path })
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceIndex {
    artifact_path: PathBuf,
    source_model_path: String,
}

fn write_source_index(artifact_path: &Path, pin: &PinFile) -> Result<()> {
    let root = materialized_stage_cache_dir();
    let Ok(info) = package::inspect_layer_package(&pin.package_ref) else {
        return Ok(());
    };
    let index = SourceIndex {
        artifact_path: artifact_path.to_path_buf(),
        source_model_path: info.source_model_path,
    };
    let path = root.join(format!(
        "source-{}.json",
        cache_key(&format!(
            "{}\0{}",
            index.source_model_path,
            artifact_path.to_string_lossy()
        ))
    ));
    fs::write(path, serde_json::to_vec_pretty(&index)?).context("write source index")?;
    Ok(())
}

fn active_pin_artifacts(root: &Path) -> Result<Vec<PathBuf>> {
    let pin_dir = root.join("pins");
    if !pin_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut artifacts = Vec::new();
    for entry in fs::read_dir(&pin_dir).with_context(|| format!("read {}", pin_dir.display()))? {
        let path = entry?.path();
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(pin) = serde_json::from_slice::<PinFile>(&bytes) else {
            continue;
        };
        artifacts.push(pin.artifact_path);
    }
    Ok(artifacts)
}

fn cache_key(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(24);
    for byte in &digest[..12] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    use serial_test::serial;

    fn restore_env(key: &str, previous: Option<OsString>) {
        if let Some(value) = previous {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn layer_package_ref_detects_local_manifest_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-package.json"), "{}").unwrap();

        assert!(is_layer_package_ref(&dir.path().to_string_lossy()));
        assert!(!is_layer_package_ref("/tmp/not-a-package"));
        assert!(is_layer_package_ref("hf://Mesh-LLM/demo-package"));
    }

    #[test]
    fn package_ref_distinguishes_direct_gguf_from_distributable_packages() {
        let direct = StagePackageRef::parse("/models/model.gguf").unwrap();
        assert_eq!(
            direct,
            StagePackageRef::SyntheticDirectGguf(PathBuf::from("/models/model.gguf"))
        );
        assert!(!direct.is_distributable_package());
        assert!(direct.as_package_ref().is_none());

        let hf = StagePackageRef::parse("hf://Mesh-LLM/demo-package@abc123").unwrap();
        assert!(hf.is_distributable_package());
        assert_eq!(
            hf.as_package_ref().as_deref(),
            Some("hf://Mesh-LLM/demo-package@abc123")
        );
    }

    #[test]
    #[serial]
    fn materialized_stage_preview_matches_source_removal_candidates() {
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("XDG_CACHE_HOME", temp.path());

        let root = materialized_stage_cache_dir();
        fs::create_dir_all(&root).unwrap();
        let source = temp
            .path()
            .join("source-package")
            .join("model-package.json");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, b"{}").unwrap();
        let artifact = root.join("stage-000.gguf");
        fs::write(&artifact, b"stage").unwrap();
        let index = SourceIndex {
            artifact_path: artifact.clone(),
            source_model_path: source.to_string_lossy().to_string(),
        };
        let index_path = root.join("source-test.json");
        fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
        fs::create_dir(root.join("source-unreadable.json")).unwrap();

        let preview = materialized_stages_for_sources(std::slice::from_ref(&source)).unwrap();
        assert_eq!(preview, vec![artifact.clone()]);

        let removed =
            remove_materialized_stages_for_sources(std::slice::from_ref(&source)).unwrap();
        assert_eq!(removed, 1);
        assert!(!artifact.exists());
        assert!(!index_path.exists());

        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    /// Integration test: downloads only the manifest from a real layer package on HF.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_manifest_only -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_manifest_only() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request 0 layers — should only download the manifest
        let local_path = resolve_hf_package_to_local(package_ref, 0, 0, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(
            manifest.is_file(),
            "manifest should exist at {}",
            manifest.display()
        );

        // Verify manifest is valid JSON with expected fields
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert!(parsed["layers"].as_array().unwrap().len() > 50);

        // Verify the function didn't request any layer downloads
        // (we can't check the cache dir because previous test runs may have cached files)
    }

    /// Integration test: downloads manifest + a single layer file.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_single_layer -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_single_layer() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request just layer 0
        let local_path = resolve_hf_package_to_local(package_ref, 0, 1, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(manifest.is_file());

        // Read manifest to find layer 0's artifact path
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        let layer0_artifact = parsed["layers"][0]["path"].as_str().unwrap();

        // Verify that specific layer file was downloaded
        let layer0_path = std::path::Path::new(&local_path).join(layer0_artifact);
        assert!(
            layer0_path.is_file(),
            "layer 0 should be downloaded at {}",
            layer0_path.display()
        );
        // Should be non-trivial size (layer files are typically > 1 MB)
        let size = std::fs::metadata(&layer0_path).unwrap().len();
        assert!(size > 1_000_000, "layer file should be > 1MB, got {size}");
    }
}
