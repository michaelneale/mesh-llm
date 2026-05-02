use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Deserialize)]
pub(crate) struct CreateRunRequest {
    pub(crate) run_id: Option<String>,
    #[serde(flatten)]
    pub(crate) config: Map<String, Value>,
}

#[derive(Serialize)]
pub(crate) struct CreateRunResponse {
    pub(crate) run_id: String,
    pub(crate) status: String,
}

#[derive(Serialize)]
pub(crate) struct RunStatusResponse {
    pub(crate) run_id: String,
    pub(crate) status: String,
    pub(crate) started_at_unix_nanos: i64,
    pub(crate) finished_at_unix_nanos: Option<i64>,
    pub(crate) request_count: i64,
    pub(crate) span_count: i64,
    pub(crate) metric_export_count: i64,
    pub(crate) log_export_count: i64,
}

#[derive(Serialize)]
pub(crate) struct ArtifactsResponse {
    pub(crate) artifacts: Vec<Artifact>,
}

#[derive(Serialize)]
pub(crate) struct Artifact {
    pub(crate) name: String,
    pub(crate) url: String,
    pub(crate) content_type: String,
}

#[derive(Serialize)]
pub(crate) struct Report {
    pub(crate) run: RunRecord,
    pub(crate) counts: BTreeMap<String, i64>,
    pub(crate) telemetry_loss: TelemetryLossReport,
    pub(crate) requests: Vec<RequestRecord>,
    pub(crate) stages: Vec<StageRecord>,
    pub(crate) stage_request_summaries: Vec<StageRequestSummary>,
    pub(crate) spans: Vec<SpanRecord>,
}

#[derive(Serialize)]
pub(crate) struct TelemetryLossReport {
    pub(crate) dropped_events: i64,
    pub(crate) export_errors: i64,
}

#[derive(Serialize)]
pub(crate) struct RunRecord {
    pub(crate) run_id: String,
    pub(crate) status: String,
    pub(crate) started_at_unix_nanos: i64,
    pub(crate) finished_at_unix_nanos: Option<i64>,
    pub(crate) config: Value,
}

#[derive(Serialize)]
pub(crate) struct RequestRecord {
    pub(crate) run_id: String,
    pub(crate) request_id: String,
    pub(crate) session_id: Option<String>,
    pub(crate) first_seen_unix_nanos: i64,
}

#[derive(Serialize)]
pub(crate) struct StageRecord {
    pub(crate) run_id: String,
    pub(crate) stage_id: String,
    pub(crate) first_seen_unix_nanos: i64,
    pub(crate) attributes: Value,
}

#[derive(Serialize)]
pub(crate) struct StageRequestSummary {
    pub(crate) run_id: String,
    pub(crate) request_id: String,
    pub(crate) stage_id: String,
    pub(crate) span_count: i64,
    pub(crate) first_start_unix_nanos: i64,
    pub(crate) last_end_unix_nanos: i64,
}

#[derive(Serialize)]
pub(crate) struct SpanRecord {
    pub(crate) run_id: String,
    pub(crate) request_id: Option<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) stage_id: Option<String>,
    pub(crate) trace_id: String,
    pub(crate) span_id: String,
    pub(crate) parent_span_id: Option<String>,
    pub(crate) name: String,
    pub(crate) kind: i32,
    pub(crate) start_time_unix_nanos: i64,
    pub(crate) end_time_unix_nanos: i64,
    pub(crate) attributes: Value,
}
