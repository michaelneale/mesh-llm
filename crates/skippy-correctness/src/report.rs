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
