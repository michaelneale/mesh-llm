use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use anyhow::{bail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use skippy_cache::{
    ExactStateCache, PrefixCandidatePolicy, ResidentActivationCache, ResidentPrefixCache,
};
use skippy_runtime::{ActivationFrame, RuntimeKvPageDesc};

use crate::kv_proto::{
    Checksum, ChecksumAlgorithm, KvPageManifest, PageIdentity, PageState, MANIFEST_SCHEMA_VERSION,
};

mod activation;
mod config;
mod exact_state;
mod identity;
mod records;
mod resident_prefix;

pub use records::{
    AttachedPage, ExactStateRecord, ExactStateRestore, LookupBatchOutcome, PrefillKvIdentity,
    RecordPageOutcome, ResidentActivationRecord, ResidentActivationRestore, ResidentPrefixRecord,
    ResidentPrefixRestore,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKvMode {
    Disabled,
    Record,
    LookupRecord,
    Correctness,
}

#[derive(Clone)]
pub struct KvStageIntegration {
    pub(crate) mode: StageKvMode,
    pub(crate) payload: StagePrefixCachePayload,
    pub(crate) correctness_mode: bool,
    pub(crate) trust_local_writes: bool,
    pub(crate) candidate_policy: PrefixCandidatePolicy,
    pub(crate) inflight_records: Arc<Mutex<BTreeSet<String>>>,
    pub(crate) resident: Arc<Mutex<ResidentPrefixCache>>,
    pub(crate) activations: Arc<Mutex<ResidentActivationCache<ActivationFrame>>>,
    pub(crate) exact_states: Arc<Mutex<ExactStateCache<ExactStateExtra>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagePrefixCachePayload {
    Disabled,
    ResidentKv,
    KvRecurrent,
    FullState,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExactStateExtra {
    pub(crate) kv_desc: Option<RuntimeKvPageDesc>,
}

impl KvStageIntegration {
    pub fn mode(&self) -> StageKvMode {
        self.mode
    }

    pub fn should_lookup(&self) -> bool {
        matches!(
            self.mode,
            StageKvMode::LookupRecord | StageKvMode::Correctness
        )
    }

    pub fn should_record(&self) -> bool {
        matches!(
            self.mode,
            StageKvMode::Record | StageKvMode::LookupRecord | StageKvMode::Correctness
        )
    }

    pub fn try_begin_record(&self, page_id: &str) -> bool {
        self.inflight_records
            .lock()
            .expect("kv inflight record lock poisoned")
            .insert(page_id.to_string())
    }

    pub fn finish_record(&self, page_id: &str) {
        self.inflight_records
            .lock()
            .expect("kv inflight record lock poisoned")
            .remove(page_id);
    }

    pub async fn hello(&self) -> Result<()> {
        Ok(())
    }

    pub async fn lookup_prefixes(
        &self,
        _identities: Vec<PageIdentity>,
    ) -> Result<LookupBatchOutcome> {
        Ok(LookupBatchOutcome {
            pages: Vec::new(),
            errors: Vec::new(),
        })
    }

    #[allow(dead_code)]
    pub async fn record_page(
        &self,
        page_id: String,
        identity: PageIdentity,
        bytes: &[u8],
        annotations: BTreeMap<String, String>,
    ) -> Result<KvPageManifest> {
        Ok(self
            .record_page_into(page_id, identity, bytes.len(), annotations, |output| {
                output.copy_from_slice(bytes);
                Ok(())
            })
            .await?
            .manifest)
    }

    pub async fn record_page_into(
        &self,
        page_id: String,
        identity: PageIdentity,
        byte_size: usize,
        mut annotations: BTreeMap<String, String>,
        write_page: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<RecordPageOutcome> {
        let mut bytes = vec![0; byte_size];
        write_page(&mut bytes)?;
        let checksum = local_trust_checksum(&page_id, byte_size as u64);
        annotations.insert(
            "mesh.skippy.prefix-cache-disabled".to_string(),
            "true".to_string(),
        );
        Ok(RecordPageOutcome {
            manifest: KvPageManifest {
                schema_version: MANIFEST_SCHEMA_VERSION,
                page_id,
                identity: Some(identity),
                state: PageState::Empty as i32,
                byte_size: byte_size as u64,
                shm_offset: 0,
                shm_len: byte_size as u64,
                checksum: Some(checksum),
                lease: None,
                annotations,
            },
            write_ms: 0.0,
            checksum_ms: 0.0,
        })
    }

    pub async fn attach_page(&self, _page_id: &str) -> Result<AttachedPage> {
        bail!("prefix cache integration is not included in mesh skippy-server")
    }

    pub async fn drop_session(&self, _session_id: &str) -> Result<u64> {
        Ok(0)
    }

    pub fn attrs(&self) -> Vec<(&'static str, Value)> {
        let resident = self
            .resident
            .lock()
            .expect("resident prefix cache lock poisoned");
        let resident = resident.stats();
        let activations = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned");
        let activations = activations.stats();
        vec![
            ("skippy.kv.mode", json!(format!("{:?}", self.mode))),
            ("skippy.kv.payload", json!(format!("{:?}", self.payload))),
            (
                "skippy.kv.page_size_tokens",
                json!(self.candidate_policy.page_size_tokens),
            ),
            ("skippy.kv.resident_entries", json!(resident.entries)),
            ("skippy.kv.resident_tokens", json!(resident.resident_tokens)),
            (
                "skippy.kv.resident_estimated_bytes",
                json!(resident.estimated_bytes),
            ),
            ("skippy.kv.max_entries", json!(resident.max_entries)),
            ("skippy.kv.max_bytes", json!(resident.max_bytes)),
            (
                "skippy.activation_cache.entries",
                json!(activations.entries),
            ),
            (
                "skippy.activation_cache.resident_bytes",
                json!(activations.resident_bytes),
            ),
            (
                "skippy.exact_cache.entries",
                json!(self.exact_state_stats().entries),
            ),
            (
                "skippy.exact_cache.logical_bytes",
                json!(self.exact_state_stats().logical_bytes),
            ),
            (
                "skippy.exact_cache.physical_bytes",
                json!(self.exact_state_stats().physical_bytes),
            ),
            (
                "skippy.exact_cache.max_bytes",
                json!(self.exact_state_stats().max_bytes),
            ),
            ("skippy.kv.correctness_mode", json!(self.correctness_mode)),
            (
                "skippy.kv.trust_local_writes",
                json!(self.trust_local_writes),
            ),
            (
                "skippy.kv.shared_prefix_min_tokens",
                json!(self.candidate_policy.min_tokens),
            ),
            (
                "skippy.kv.shared_prefix_stride_tokens",
                json!(self.candidate_policy.stride_tokens),
            ),
            (
                "skippy.kv.shared_prefix_record_limit",
                json!(self.candidate_policy.record_limit),
            ),
        ]
    }

    fn exact_state_stats(&self) -> skippy_cache::ExactStateCacheStats {
        self.exact_states
            .lock()
            .expect("exact state cache lock poisoned")
            .stats()
    }

    fn record_candidate_token_counts(&self, token_count: u64) -> Vec<u64> {
        self.candidate_policy
            .record_candidate_token_counts(token_count)
    }
}

fn local_trust_checksum(page_id: &str, byte_size: u64) -> Checksum {
    let mut digest = Sha256::new();
    digest.update(b"skippy-local-trust-v1");
    digest.update(page_id.as_bytes());
    digest.update(byte_size.to_le_bytes());
    Checksum {
        algorithm: ChecksumAlgorithm::Sha256 as i32,
        digest: digest.finalize().to_vec(),
    }
}
