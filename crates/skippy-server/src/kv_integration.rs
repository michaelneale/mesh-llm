use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use anyhow::{bail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use skippy_protocol::{MessageBase, StageConfig};

use crate::kv_proto::{
    Checksum, ChecksumAlgorithm, KvCodec, KvPageManifest, PageIdentity, PageLayout,
};

const NATIVE_KV_RUNTIME_ABI_VERSION: &str = "stage-abi-0.1/native-kv-page-v1";
const NATIVE_KV_DTYPE: &str = "ggml-native-kv";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKvMode {
    Disabled,
    Record,
    LookupRecord,
    Correctness,
}

#[derive(Clone)]
pub struct KvStageIntegration {
    mode: StageKvMode,
    page_size_tokens: u64,
    correctness_mode: bool,
    trust_local_writes: bool,
    shared_prefix_min_tokens: u64,
    shared_prefix_stride_tokens: u64,
    shared_prefix_record_limit: u64,
    inflight_records: Arc<Mutex<BTreeSet<String>>>,
}

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

impl AttachedPage {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl KvStageIntegration {
    pub fn from_config(_config: &StageConfig) -> Result<Option<Self>> {
        Ok(None)
    }

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

    pub fn prefill_identity(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> PrefillKvIdentity {
        let token_count = token_ids.len() as u64;
        let prefix_hash = prefix_hash(config, token_start, token_ids);
        let identity = PageIdentity {
            model_id: config.model_id.clone(),
            model_revision: "unknown".to_string(),
            runtime_abi_version: NATIVE_KV_RUNTIME_ABI_VERSION.to_string(),
            topology_id: config.topology_id.clone(),
            stage_id: config.stage_id.clone(),
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            prefix_hash: prefix_hash.clone(),
            session_id: base.session_id.clone(),
            token_start,
            token_count,
            generation: 1,
            layout: PageLayout::LayerContiguous as i32,
            codec: KvCodec::Fp16 as i32,
            tokenizer_id: base
                .tokenizer_id
                .clone()
                .unwrap_or_else(|| config.model_id.clone()),
            chat_template_id: base.chat_template_id.clone().unwrap_or_default(),
            position_config_hash: format!("ctx:{}", config.ctx_size),
            kv_dtype: NATIVE_KV_DTYPE.to_string(),
        };
        let page_id = page_id(config, token_start, token_count, &prefix_hash);
        PrefillKvIdentity { identity, page_id }
    }

    pub fn lookup_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        self.candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
    }

    pub fn record_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        self.record_candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
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
                schema_version: crate::kv_proto::MANIFEST_SCHEMA_VERSION,
                page_id,
                identity: Some(identity),
                state: crate::kv_proto::PageState::Empty as i32,
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
        vec![
            ("skippy.kv.mode", json!(format!("{:?}", self.mode))),
            ("skippy.kv.page_size_tokens", json!(self.page_size_tokens)),
            ("skippy.kv.correctness_mode", json!(self.correctness_mode)),
            (
                "skippy.kv.trust_local_writes",
                json!(self.trust_local_writes),
            ),
            (
                "skippy.kv.shared_prefix_min_tokens",
                json!(self.shared_prefix_min_tokens),
            ),
            (
                "skippy.kv.shared_prefix_stride_tokens",
                json!(self.shared_prefix_stride_tokens),
            ),
            (
                "skippy.kv.shared_prefix_record_limit",
                json!(self.shared_prefix_record_limit),
            ),
        ]
    }

    fn candidate_token_counts(&self, token_count: u64) -> Vec<u64> {
        if token_count == 0 {
            return Vec::new();
        }
        let mut counts = vec![token_count];
        let min_tokens = self.shared_prefix_min_tokens;
        if min_tokens == 0 || token_count <= min_tokens {
            return counts;
        }
        let stride = self
            .shared_prefix_stride_tokens
            .max(1)
            .min(self.page_size_tokens.max(1));
        let mut candidate = token_count.saturating_sub(1);
        while candidate >= min_tokens {
            counts.push(candidate);
            if candidate == min_tokens {
                break;
            }
            let next = candidate.saturating_sub(stride);
            candidate = next.max(min_tokens);
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts.dedup();
        counts
    }

    fn record_candidate_token_counts(&self, token_count: u64) -> Vec<u64> {
        let candidates = self.candidate_token_counts(token_count);
        let limit = self.shared_prefix_record_limit as usize;
        if limit == 0 || candidates.len() <= limit {
            return candidates;
        }

        let mut selected = Vec::with_capacity(limit);
        selected.push(token_count);

        let min_tokens = self.shared_prefix_min_tokens;
        let lower_bound = candidates
            .iter()
            .copied()
            .filter(|candidate| *candidate != token_count)
            .min()
            .unwrap_or(token_count);
        let shared_slots = limit.saturating_sub(1);
        if shared_slots == 0 {
            return selected;
        }
        for slot in 0..shared_slots {
            let target = if shared_slots == 1 {
                lower_bound
            } else {
                let span = token_count.saturating_sub(lower_bound);
                token_count
                    .saturating_sub(span.saturating_mul((slot + 1) as u64) / shared_slots as u64)
            }
            .max(min_tokens)
            .min(token_count);
            if let Some(candidate) = candidates
                .iter()
                .copied()
                .filter(|candidate| *candidate <= target && *candidate != token_count)
                .max()
            {
                selected.push(candidate);
            }
        }
        if selected.len() < limit {
            for candidate in candidates.into_iter().rev() {
                if selected.len() >= limit {
                    break;
                }
                if !selected.contains(&candidate) {
                    selected.push(candidate);
                }
            }
        }
        selected.sort_unstable_by(|a, b| b.cmp(a));
        selected.dedup();
        selected
    }
}

fn prefix_hash(config: &StageConfig, token_start: u64, token_ids: &[i32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(config.model_id.as_bytes());
    hasher.update(config.topology_id.as_bytes());
    hasher.update(config.stage_id.as_bytes());
    hasher.update(config.stage_index.to_le_bytes());
    hasher.update(config.layer_start.to_le_bytes());
    hasher.update(config.layer_end.to_le_bytes());
    hasher.update(NATIVE_KV_RUNTIME_ABI_VERSION.as_bytes());
    hasher.update((PageLayout::LayerContiguous as i32).to_le_bytes());
    hasher.update(NATIVE_KV_DTYPE.as_bytes());
    hasher.update(format!("ctx:{}", config.ctx_size).as_bytes());
    hasher.update(token_start.to_le_bytes());
    for token_id in token_ids {
        hasher.update(token_id.to_le_bytes());
    }
    format!("sha256:{}", hex(&hasher.finalize()))
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

fn page_id(config: &StageConfig, token_start: u64, token_count: u64, prefix_hash: &str) -> String {
    let digest = prefix_hash.strip_prefix("sha256:").unwrap_or(prefix_hash);
    let short = digest.get(..16).unwrap_or(digest);
    format!(
        "{}:{}:{}:{}:{}",
        config.stage_id, token_start, token_count, config.layer_start, short
    )
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}
