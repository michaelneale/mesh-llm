use anyhow::{anyhow, Result};
use skippy_protocol::StageConfig;
pub use skippy_runtime::package::is_hf_package_ref;
use skippy_runtime::package::{
    materialize_layer_package as materialize_runtime_package, select_layer_package_parts,
    PackageStageRequest, SelectedPackageParts,
};
use std::path::PathBuf;

pub fn materialize_layer_package(config: &StageConfig) -> Result<PathBuf> {
    let package_ref = config.model_path.as_deref().ok_or_else(|| {
        anyhow!("layer-package load mode requires model_path to point at a package directory")
    })?;
    let is_final_stage = config.downstream.is_none();
    materialize_runtime_package(&PackageStageRequest {
        model_id: config.model_id.clone(),
        topology_id: config.topology_id.clone(),
        package_ref: package_ref.to_string(),
        stage_id: config.stage_id.clone(),
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        // Include embeddings for first stage, and also for final stage
        // (needed for tied-embedding models where token_embd doubles as output weight)
        include_embeddings: config.layer_start == 0 || is_final_stage,
        include_output: is_final_stage,
    })
}

pub fn select_package_parts(config: &StageConfig) -> Result<SelectedPackageParts> {
    let package_ref = config.model_path.as_deref().ok_or_else(|| {
        anyhow!("layer-package load mode requires model_path to point at a package directory")
    })?;
    select_layer_package_parts(&PackageStageRequest {
        model_id: config.model_id.clone(),
        topology_id: config.topology_id.clone(),
        package_ref: package_ref.to_string(),
        stage_id: config.stage_id.clone(),
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        include_embeddings: config.layer_start == 0,
        include_output: config.downstream.is_none(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_hf_package_refs() {
        assert!(is_hf_package_ref("hf://Mesh-LLM/Qwen3.6-package"));
        assert!(!is_hf_package_ref("/tmp/package"));
    }
}
