use serde::{Deserialize, Serialize};

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
