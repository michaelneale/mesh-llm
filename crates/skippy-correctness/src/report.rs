use serde::Serialize;

pub use model_artifact::ModelIdentity;

#[derive(Debug, Serialize)]
pub struct BaselineReport {
    pub token_id: i32,
    pub predicted_token: i32,
}

#[derive(Debug, Serialize)]
pub struct BoundaryReport {
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub payload_bytes: u64,
    pub wire_payload_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct SplitReport {
    pub token_id: i32,
    pub predicted_token: i32,
    pub activation_width: i32,
    pub wire_dtype: String,
    pub boundary: BoundaryReport,
}

#[derive(Debug, Serialize)]
pub struct SingleStepReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub matches: bool,
    pub baseline: BaselineReport,
    pub split: SplitReport,
    pub stage_models: Vec<StageModelReport>,
}

#[derive(Debug, Serialize)]
pub struct ChainStageReport {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub payload_bytes: Option<u64>,
    pub wire_payload_bytes: Option<usize>,
    pub forwarded_over_binary: bool,
    pub returned_predicted_token: bool,
}

#[derive(Debug, Serialize)]
pub struct ChainReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub matches: bool,
    pub baseline: BaselineReport,
    pub token_id: i32,
    pub predicted_token: i32,
    pub activation_width: i32,
    pub wire_dtype: String,
    pub stages: Vec<ChainStageReport>,
    pub stage_models: Vec<StageModelReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct StageModelReport {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub load_mode: &'static str,
    pub model_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<PackageStageReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct PackageStageReport {
    pub package_ref: String,
    pub materialized_path: String,
    pub manifest_sha256: String,
    pub selected_parts: Vec<PackagePartReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct PackagePartReport {
    pub role: String,
    pub layer_index: Option<u32>,
    pub path: String,
    pub sha256: String,
    pub artifact_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct SplitScanReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub baseline: BaselineReport,
    pub split_count: usize,
    pub mismatch_count: usize,
    pub results: Vec<SingleStepReport>,
}

#[derive(Debug, Serialize)]
pub struct DtypeMatrixReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub baseline: BaselineReport,
    pub dtype_count: usize,
    pub mismatch_count: usize,
    pub results: Vec<SingleStepReport>,
}

#[derive(Debug, Serialize)]
pub struct StateHandoffReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub matches: bool,
    pub predicted_token_matches: bool,
    pub roundtrip_state_matches: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restored_output_matches: Option<bool>,
    pub cache_hit_matches: bool,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub handoff_transport: &'static str,
    pub state_payload_kind: &'static str,
    pub borrowed_resident_hits: bool,
    pub cached_decoded_result_hits: bool,
    pub source_predicted_token: i32,
    pub restored_predicted_token: i32,
    pub prompt_token_count: usize,
    pub benchmark_prompt_token_count: usize,
    pub benchmark_prompt_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_prefix_token_count: Option<usize>,
    pub activation_width: i32,
    pub state_bytes: usize,
    pub state_bytes_per_prompt_token: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_storage_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_storage_bytes_per_prompt_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_state_bytes: Option<usize>,
    pub roundtrip_state_bytes: usize,
    pub payload_digest: StatePayloadDigestReport,
    pub tokenize_ms: f64,
    pub source_prefill_ms: f64,
    pub source_export_ms: f64,
    pub source_decode_ms: f64,
    pub restore_import_ms: f64,
    pub restore_export_ms: f64,
    pub restore_decode_ms: f64,
    pub cache_hit_repeats: usize,
    pub recompute_total_ms: f64,
    pub cache_hit_total_ms: f64,
    pub cache_hit_speedup: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cache_hit_import_ms: Vec<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cache_hit_decode_ms: Vec<f64>,
    pub stage_models: Vec<StageModelReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct StatePayloadDigestReport {
    pub payload_kind: &'static str,
    pub payload_sha256: String,
    pub total_bytes: usize,
    pub kv_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_sha256: Option<String>,
    pub recurrent_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurrent_sha256: Option<String>,
    pub block_size_bytes: usize,
    pub block_count: usize,
    pub unique_block_count: usize,
    pub duplicate_block_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<StatePayloadBlockDigestReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct StatePayloadBlockDigestReport {
    pub component: &'static str,
    pub index: usize,
    pub offset: usize,
    pub bytes: usize,
    pub sha256: String,
}
