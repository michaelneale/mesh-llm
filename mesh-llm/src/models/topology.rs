use super::catalog;
use super::local::{huggingface_hub_cache, huggingface_identity_for_path};
use hf_hub::Repo;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelTopology {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moe: Option<ModelMoeInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelMoeInfo {
    pub expert_count: u32,
    pub used_expert_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_experts_per_node: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_origin: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ranking: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_prompt_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_layer_scope: Option<String>,
}

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

pub fn infer_local_model_topology(
    path: &Path,
    catalog: Option<&catalog::CatalogModel>,
) -> Option<ModelTopology> {
    if let Some(model) = catalog.and_then(infer_catalog_topology) {
        return Some(model);
    }

    read_local_config(path).and_then(|config| infer_hf_metadata_topology(&config))
}

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

fn read_local_config(path: &Path) -> Option<Value> {
    // Derive the snapshot root using the hf-hub CacheRepo::pointer_path() API.
    // This matches the authoritative cache layout: cache/models--{org}--{repo}/snapshots/{revision}/
    // config.json always lives at the snapshot root, even when the GGUF is in a subdirectory.
    let identity = huggingface_identity_for_path(path)?;
    let snapshot_root = huggingface_hub_cache()
        .repo(Repo::model(identity.repo_id))
        .pointer_path(&identity.revision);
    let config_path = snapshot_root.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    serde_json::from_str(&text).ok()
}
