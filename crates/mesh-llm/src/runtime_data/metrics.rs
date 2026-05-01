use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RuntimeLlamaEndpointStatus {
    Ready,
    Error,
    #[default]
    Unavailable,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaMetricSample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaMetricsSnapshot {
    pub status: RuntimeLlamaEndpointStatus,
    pub last_attempt_unix_ms: Option<u64>,
    pub last_success_unix_ms: Option<u64>,
    pub error: Option<String>,
    pub raw_text: Option<String>,
    pub samples: Vec<RuntimeLlamaMetricSample>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaSlotSnapshot {
    pub id: Option<u64>,
    pub id_task: Option<u64>,
    pub n_ctx: Option<u64>,
    pub speculative: Option<bool>,
    pub is_processing: Option<bool>,
    pub next_token: Option<Value>,
    pub params: Option<Value>,
    pub extra: Value,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaSlotsSnapshot {
    pub status: RuntimeLlamaEndpointStatus,
    pub last_attempt_unix_ms: Option<u64>,
    pub last_success_unix_ms: Option<u64>,
    pub error: Option<String>,
    pub slots: Vec<RuntimeLlamaSlotSnapshot>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaMetricItem {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaSlotItem {
    pub index: usize,
    pub id: Option<u64>,
    pub id_task: Option<u64>,
    pub n_ctx: Option<u64>,
    pub is_processing: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaRuntimeItems {
    pub metrics: Vec<RuntimeLlamaMetricItem>,
    pub slots: Vec<RuntimeLlamaSlotItem>,
    pub slots_total: usize,
    pub slots_busy: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeLlamaRuntimeSnapshot {
    pub metrics: RuntimeLlamaMetricsSnapshot,
    pub slots: RuntimeLlamaSlotsSnapshot,
    pub items: RuntimeLlamaRuntimeItems,
}
