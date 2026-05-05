use skippy_runtime::ActivationFrame;

use crate::kv_proto::{KvPageManifest, PageIdentity};

#[derive(Debug, Clone)]
pub struct PrefillKvIdentity {
    pub identity: PageIdentity,
    pub page_id: String,
}

#[derive(Debug, Clone)]
pub struct LookupBatchOutcome {
    pub pages: Vec<KvPageManifest>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RecordPageOutcome {
    pub manifest: KvPageManifest,
    pub write_ms: f64,
    pub checksum_ms: f64,
}

#[derive(Debug)]
pub struct AttachedPage {
    pub manifest: KvPageManifest,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixRestore {
    pub page_id: String,
    pub token_count: usize,
    pub seq_id: i32,
    pub entries: usize,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixRecord {
    pub page_id: String,
    pub token_count: usize,
    pub seq_id: i32,
    pub stored: bool,
    pub evicted_entries: usize,
    pub evicted_tokens: u64,
    pub entries: usize,
    pub resident_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct ResidentActivationRestore {
    pub identity: PrefillKvIdentity,
    pub page_id: String,
    pub token_count: usize,
    pub payload_bytes: usize,
    pub entries: usize,
    pub frame: ActivationFrame,
}

#[derive(Debug, Clone)]
pub struct ResidentActivationRecord {
    pub page_id: String,
    pub token_count: usize,
    pub payload_bytes: usize,
    pub evicted_entries: usize,
    pub evicted_bytes: u64,
    pub entries: usize,
    pub resident_bytes: u64,
}

impl AttachedPage {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
