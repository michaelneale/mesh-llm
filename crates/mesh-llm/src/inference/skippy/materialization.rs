use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use skippy_protocol::{LoadMode, StageConfig};
use skippy_runtime::package::{self, LayerPackageInfo, PackageStageRequest};

use super::StageLoadRequest;

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
    package::is_hf_package_ref(value) || Path::new(value).join("model-package.json").is_file()
}

pub(crate) fn inspect_stage_package(package_ref: &str) -> Result<StagePackageInfo> {
    let info = package::inspect_layer_package(package_ref)
        .with_context(|| format!("inspect skippy layer package {package_ref}"))?;
    stage_package_info(package_ref, info)
}

pub(crate) fn materialize_stage_load(
    load: &StageLoadRequest,
) -> Result<Option<(MaterializedStageArtifact, MaterializedStagePin)>> {
    if load.load_mode != LoadMode::LayerPackage {
        return Ok(None);
    }
    let request = package_stage_request(
        &load.model_id,
        &load.topology_id,
        &load.package_ref,
        &load.stage_id,
        load.layer_start,
        load.layer_end,
        load.downstream.is_none(),
    );
    let materialized = package::materialize_layer_package_details(&request).with_context(|| {
        format!(
            "materialize skippy stage package {} layers {}..{}",
            load.stage_id, load.layer_start, load.layer_end
        )
    })?;
    let info = package::inspect_layer_package(&load.package_ref)?;
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
    let request = package_stage_request(
        &config.model_id,
        &config.topology_id,
        package_ref,
        &config.stage_id,
        config.layer_start,
        config.layer_end,
        config.downstream.is_none(),
    );
    let materialized = package::materialize_layer_package_details(&request).with_context(|| {
        format!(
            "materialize skippy stage package {} layers {}..{}",
            config.stage_id, config.layer_start, config.layer_end
        )
    })?;
    let info = package::inspect_layer_package(package_ref)?;
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
    if sources.is_empty() {
        return Ok(0);
    }
    let root = materialized_stage_cache_dir();
    if !root.is_dir() {
        return Ok(0);
    }
    let source_strings = sources
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let pins = active_pin_artifacts(&root)?;
    let mut removed = 0usize;
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
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
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
        if index.artifact_path.exists() {
            fs::remove_file(&index.artifact_path)
                .with_context(|| format!("remove {}", index.artifact_path.display()))?;
            removed += 1;
        }
        let _ = fs::remove_file(path);
    }
    Ok(removed)
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

    #[test]
    fn layer_package_ref_detects_local_manifest_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-package.json"), "{}").unwrap();

        assert!(is_layer_package_ref(&dir.path().to_string_lossy()));
        assert!(!is_layer_package_ref("/tmp/not-a-package"));
        assert!(is_layer_package_ref("hf://Mesh-LLM/demo-package"));
    }
}
