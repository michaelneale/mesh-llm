use super::catalog;
use super::local::{huggingface_identity_for_path, huggingface_snapshot_path};
use hf_hub::RepoType;
use serde_json::Value;
use std::path::Path;

pub use mesh_client::models::topology::{ModelMoeInfo, ModelTopology};

#[allow(dead_code)]
pub fn infer_catalog_topology(model: &catalog::CatalogModel) -> Option<ModelTopology> {
    model.moe.as_ref().map(|moe| ModelTopology {
        moe: Some(ModelMoeInfo {
            expert_count: moe.n_expert,
            used_expert_count: moe.n_expert_used,
            min_experts_per_node: Some(moe.min_experts_per_node),
            source: Some("catalog".to_string()),
            ranking_source: None,
            ranking_origin: None,
            ranking: Vec::new(),
            ranking_prompt_count: None,
            ranking_tokens: None,
            ranking_layer_scope: None,
        }),
    })
}

#[allow(dead_code)]
pub fn infer_local_model_topology(
    path: &Path,
    catalog: Option<&catalog::CatalogModel>,
) -> Option<ModelTopology> {
    if let Some(model) = catalog.and_then(infer_catalog_topology) {
        return Some(model);
    }

    read_local_config(path).and_then(|config| infer_hf_metadata_topology(&config))
}

#[allow(dead_code)]
fn infer_hf_metadata_topology(config: &Value) -> Option<ModelTopology> {
    let expert_count = config.get("num_experts").and_then(|value| value.as_u64())? as u32;
    if expert_count <= 1 {
        return None;
    }
    // Omit topology entirely when num_experts_per_tok is missing or zero to
    // avoid surfacing impossible values like "top-0" in the UI.
    let used_expert_count = config
        .get("num_experts_per_tok")
        .and_then(|value| value.as_u64())
        .filter(|&v| v > 0)? as u32;
    Some(ModelTopology {
        moe: Some(ModelMoeInfo {
            expert_count,
            used_expert_count,
            min_experts_per_node: None,
            source: Some("hf_metadata".to_string()),
            ranking_source: None,
            ranking_origin: None,
            ranking: Vec::new(),
            ranking_prompt_count: None,
            ranking_tokens: None,
            ranking_layer_scope: None,
        }),
    })
}

#[allow(dead_code)]
fn read_local_config(path: &Path) -> Option<Value> {
    // Derive the snapshot root from the Hugging Face cache layout:
    // cache/models--{org}--{repo}/snapshots/{revision}/
    // config.json always lives at the snapshot root, even when the GGUF is in a subdirectory.
    let identity = huggingface_identity_for_path(path)?;
    let snapshot_root =
        huggingface_snapshot_path(&identity.repo_id, RepoType::Model, &identity.revision);
    let config_path = snapshot_root.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    serde_json::from_str(&text).ok()
}
