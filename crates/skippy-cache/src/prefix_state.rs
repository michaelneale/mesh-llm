use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Instant,
};

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use skippy_protocol::{
    binary::{StageWireMessage, WireMessageKind},
    StageConfig, StageFullStateCacheMode, StageFullStateCachePayload,
};
use skippy_runtime::RuntimeKvPageDesc;

const DEFAULT_BLOCK_SIZE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FullStateCacheKey {
    namespace: Option<String>,
    model_id: String,
    topology_id: String,
    stage_id: String,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    cache_type_k: String,
    cache_type_v: String,
    load_mode: String,
    filter_tensors_on_load: bool,
    pub token_start: u64,
    pub token_count: u64,
    token_sha256: String,
}

impl FullStateCacheKey {
    pub fn short_id(&self) -> String {
        format!(
            "{}:{}:{}:{}..{}:{}+{}:{}",
            self.namespace.as_deref().unwrap_or("default"),
            self.topology_id,
            self.stage_id,
            self.layer_start,
            self.layer_end,
            self.token_start,
            self.token_count,
            &self.token_sha256[..12.min(self.token_sha256.len())]
        )
    }
}

#[derive(Debug, Clone)]
pub struct FullStateCacheLookup {
    pub key: FullStateCacheKey,
    pub payload: FullStateCachePayloadEntry,
    pub entries: usize,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub dedupe_saved_bytes: u64,
    pub blob_blocks: usize,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub enum FullStateCachePayloadEntry {
    FullState {
        bytes: CacheBytes,
    },
    RecurrentOnly {
        recurrent: CacheBytes,
    },
    KvRecurrent {
        kv_desc: RuntimeKvPageDesc,
        kv: CacheBytes,
        recurrent: CacheBytes,
    },
}

impl FullStateCachePayloadEntry {
    pub fn full_state(bytes: Vec<u8>) -> Self {
        Self::FullState {
            bytes: CacheBytes::inline(bytes),
        }
    }

    pub fn recurrent_only(recurrent: Vec<u8>) -> Self {
        Self::RecurrentOnly {
            recurrent: CacheBytes::inline(recurrent),
        }
    }

    pub fn kv_recurrent(kv_desc: RuntimeKvPageDesc, kv: Vec<u8>, recurrent: Vec<u8>) -> Self {
        Self::KvRecurrent {
            kv_desc,
            kv: CacheBytes::inline(kv),
            recurrent: CacheBytes::inline(recurrent),
        }
    }

    pub fn kind(&self) -> StageFullStateCachePayload {
        match self {
            Self::FullState { .. } => StageFullStateCachePayload::FullState,
            Self::RecurrentOnly { .. } => StageFullStateCachePayload::RecurrentOnly,
            Self::KvRecurrent { .. } => StageFullStateCachePayload::KvRecurrent,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self.kind() {
            StageFullStateCachePayload::FullState => "full-state",
            StageFullStateCachePayload::RecurrentOnly => "recurrent-only",
            StageFullStateCachePayload::KvRecurrent => "kv-recurrent",
        }
    }

    pub fn byte_len(&self) -> u64 {
        match self {
            Self::FullState { bytes } => bytes.len(),
            Self::RecurrentOnly { recurrent } => recurrent.len(),
            Self::KvRecurrent { kv, recurrent, .. } => kv.len().saturating_add(recurrent.len()),
        }
    }

    pub fn kv_bytes(&self) -> u64 {
        match self {
            Self::KvRecurrent { kv, .. } => kv.len(),
            _ => 0,
        }
    }

    pub fn recurrent_bytes(&self) -> u64 {
        match self {
            Self::FullState { .. } => 0,
            Self::RecurrentOnly { recurrent } | Self::KvRecurrent { recurrent, .. } => {
                recurrent.len()
            }
        }
    }

    pub fn block_ref_count(&self) -> usize {
        match self {
            Self::FullState { bytes } => bytes.block_ref_count(),
            Self::RecurrentOnly { recurrent } => recurrent.block_ref_count(),
            Self::KvRecurrent { kv, recurrent, .. } => kv
                .block_ref_count()
                .saturating_add(recurrent.block_ref_count()),
        }
    }

    pub fn full_state_bytes(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            Self::FullState { bytes } => bytes.as_cow(),
            _ => Err(anyhow!("cache payload is not full-state")),
        }
    }

    pub fn full_state_bytes_timed(&self) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        match self {
            Self::FullState { bytes } => bytes.as_cow_timed(),
            _ => Err(anyhow!("cache payload is not full-state")),
        }
    }

    pub fn recurrent_state_bytes(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            Self::RecurrentOnly { recurrent } | Self::KvRecurrent { recurrent, .. } => {
                recurrent.as_cow()
            }
            _ => Err(anyhow!("cache payload has no recurrent component")),
        }
    }

    pub fn recurrent_state_bytes_timed(
        &self,
    ) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        match self {
            Self::RecurrentOnly { recurrent } | Self::KvRecurrent { recurrent, .. } => {
                recurrent.as_cow_timed()
            }
            _ => Err(anyhow!("cache payload has no recurrent component")),
        }
    }

    pub fn kv_bytes_cow(&self) -> Result<Option<Cow<'_, [u8]>>> {
        match self {
            Self::KvRecurrent { kv, .. } => Ok(Some(kv.as_cow()?)),
            _ => Ok(None),
        }
    }

    pub fn kv_bytes_cow_timed(
        &self,
    ) -> Result<(Option<Cow<'_, [u8]>>, CacheBytesReconstructStats)> {
        match self {
            Self::KvRecurrent { kv, .. } => {
                let (bytes, stats) = kv.as_cow_timed()?;
                Ok((Some(bytes), stats))
            }
            _ => Ok((None, CacheBytesReconstructStats::default())),
        }
    }

    fn dedupe_into(self, blobs: &mut CacheBlobStore) -> (Self, CacheDedupeStats) {
        match self {
            Self::FullState { bytes } => {
                let (bytes, stats) = blobs.store_bytes(bytes);
                (Self::FullState { bytes }, stats)
            }
            Self::RecurrentOnly { recurrent } => {
                let (recurrent, stats) = blobs.store_bytes(recurrent);
                (Self::RecurrentOnly { recurrent }, stats)
            }
            Self::KvRecurrent {
                kv_desc,
                kv,
                recurrent,
            } => {
                let (kv, kv_stats) = blobs.store_bytes(kv);
                let (recurrent, recurrent_stats) = blobs.store_bytes(recurrent);
                (
                    Self::KvRecurrent {
                        kv_desc,
                        kv,
                        recurrent,
                    },
                    kv_stats.saturating_add(recurrent_stats),
                )
            }
        }
    }

    fn release_from(&self, blobs: &mut CacheBlobStore) -> u64 {
        match self {
            Self::FullState { bytes } => blobs.release_bytes(bytes),
            Self::RecurrentOnly { recurrent } => blobs.release_bytes(recurrent),
            Self::KvRecurrent { kv, recurrent, .. } => blobs
                .release_bytes(kv)
                .saturating_add(blobs.release_bytes(recurrent)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheBytes {
    len: u64,
    repr: CacheBytesRepr,
}

#[derive(Debug, Clone)]
enum CacheBytesRepr {
    Inline(Arc<Vec<u8>>),
    Blocks(Arc<[CacheBlockRef]>),
}

#[derive(Debug, Clone)]
struct CacheBlockRef {
    hash: String,
    bytes: Arc<Vec<u8>>,
}

impl CacheBytes {
    fn inline(bytes: Vec<u8>) -> Self {
        Self {
            len: bytes.len() as u64,
            repr: CacheBytesRepr::Inline(Arc::new(bytes)),
        }
    }

    fn blocks(len: u64, blocks: Vec<CacheBlockRef>) -> Self {
        Self {
            len,
            repr: CacheBytesRepr::Blocks(blocks.into()),
        }
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    fn block_ref_count(&self) -> usize {
        match &self.repr {
            CacheBytesRepr::Inline(_) => 0,
            CacheBytesRepr::Blocks(blocks) => blocks.len(),
        }
    }

    fn as_cow(&self) -> Result<Cow<'_, [u8]>> {
        match &self.repr {
            CacheBytesRepr::Inline(bytes) => Ok(Cow::Borrowed(bytes.as_slice())),
            CacheBytesRepr::Blocks(blocks) => {
                let capacity = usize::try_from(self.len)
                    .map_err(|_| anyhow!("cache payload too large to reconstruct"))?;
                let mut out = Vec::with_capacity(capacity);
                for block in blocks.iter() {
                    out.extend_from_slice(block.bytes.as_slice());
                }
                if out.len() as u64 != self.len {
                    return Err(anyhow!(
                        "cache payload reconstruction length mismatch: expected {} got {}",
                        self.len,
                        out.len()
                    ));
                }
                Ok(Cow::Owned(out))
            }
        }
    }

    fn as_cow_timed(&self) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        let started = Instant::now();
        let block_count = self.block_ref_count();
        let bytes = self.as_cow()?;
        Ok((
            bytes,
            CacheBytesReconstructStats {
                reconstruct_ms: started.elapsed().as_secs_f64() * 1000.0,
                reconstruct_bytes: self.len,
                reconstruct_blocks: block_count,
            },
        ))
    }
}

#[derive(Debug, Default)]
struct CacheBlobStore {
    block_size: usize,
    physical_bytes: u64,
    blocks: HashMap<String, CacheBlob>,
}

#[derive(Debug)]
struct CacheBlob {
    bytes: Arc<Vec<u8>>,
    ref_count: u64,
}

impl CacheBlobStore {
    fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            physical_bytes: 0,
            blocks: HashMap::new(),
        }
    }

    fn store_bytes(&mut self, bytes: CacheBytes) -> (CacheBytes, CacheDedupeStats) {
        let bytes = match bytes.repr {
            CacheBytesRepr::Inline(bytes) => bytes,
            CacheBytesRepr::Blocks(_) => return (bytes, CacheDedupeStats::default()),
        };
        let mut blocks = Vec::new();
        let started = Instant::now();
        let mut stats = CacheDedupeStats {
            hash_bytes: bytes.len() as u64,
            ..CacheDedupeStats::default()
        };
        for chunk in bytes.chunks(self.block_size) {
            stats.block_count = stats.block_count.saturating_add(1);
            let hash = blake3_hex(chunk);
            let entry = self.blocks.entry(hash.clone()).or_insert_with(|| {
                self.physical_bytes = self.physical_bytes.saturating_add(chunk.len() as u64);
                stats.new_block_count = stats.new_block_count.saturating_add(1);
                CacheBlob {
                    bytes: Arc::new(chunk.to_vec()),
                    ref_count: 0,
                }
            });
            if entry.ref_count > 0 {
                stats.reused_block_count = stats.reused_block_count.saturating_add(1);
            }
            entry.ref_count = entry.ref_count.saturating_add(1);
            blocks.push(CacheBlockRef {
                hash,
                bytes: entry.bytes.clone(),
            });
        }
        stats.hash_ms = started.elapsed().as_secs_f64() * 1000.0;
        (CacheBytes::blocks(bytes.len() as u64, blocks), stats)
    }

    fn release_bytes(&mut self, bytes: &CacheBytes) -> u64 {
        let CacheBytesRepr::Blocks(blocks) = &bytes.repr else {
            return 0;
        };
        let mut freed = 0u64;
        for block in blocks.iter() {
            let mut remove = false;
            if let Some(entry) = self.blocks.get_mut(&block.hash) {
                entry.ref_count = entry.ref_count.saturating_sub(1);
                if entry.ref_count == 0 {
                    freed = freed.saturating_add(entry.bytes.len() as u64);
                    remove = true;
                }
            }
            if remove {
                self.blocks.remove(&block.hash);
            }
        }
        self.physical_bytes = self.physical_bytes.saturating_sub(freed);
        freed
    }

    fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheDedupeStats {
    pub hash_ms: f64,
    pub hash_bytes: u64,
    pub block_count: usize,
    pub new_block_count: usize,
    pub reused_block_count: usize,
}

impl CacheDedupeStats {
    fn saturating_add(self, other: Self) -> Self {
        Self {
            hash_ms: self.hash_ms + other.hash_ms,
            hash_bytes: self.hash_bytes.saturating_add(other.hash_bytes),
            block_count: self.block_count.saturating_add(other.block_count),
            new_block_count: self.new_block_count.saturating_add(other.new_block_count),
            reused_block_count: self
                .reused_block_count
                .saturating_add(other.reused_block_count),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheBytesReconstructStats {
    pub reconstruct_ms: f64,
    pub reconstruct_bytes: u64,
    pub reconstruct_blocks: usize,
}

impl CacheBytesReconstructStats {
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            reconstruct_ms: self.reconstruct_ms + other.reconstruct_ms,
            reconstruct_bytes: self
                .reconstruct_bytes
                .saturating_add(other.reconstruct_bytes),
            reconstruct_blocks: self
                .reconstruct_blocks
                .saturating_add(other.reconstruct_blocks),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FullStateCacheRecordOutcome {
    pub key: FullStateCacheKey,
    pub stored: bool,
    pub bytes: u64,
    pub physical_bytes: u64,
    pub dedupe_saved_bytes: u64,
    pub dedupe_hash_ms: f64,
    pub dedupe_hash_bytes: u64,
    pub dedupe_block_count: usize,
    pub dedupe_new_block_count: usize,
    pub dedupe_reused_block_count: usize,
    pub evicted_entries: usize,
    pub evicted_bytes: u64,
    pub evicted_physical_bytes: u64,
    pub entries: usize,
    pub logical_bytes: u64,
    pub physical_total_bytes: u64,
    pub dedupe_saved_total_bytes: u64,
    pub blob_blocks: usize,
    pub total_bytes: u64,
    pub reason: Option<String>,
}

pub type PrefixStateCache = FullStateCache;
pub type PrefixStateKey = FullStateCacheKey;
pub type PrefixStateLookup = FullStateCacheLookup;
pub type PrefixStatePayloadEntry = FullStateCachePayloadEntry;
pub type PrefixStateRecordOutcome = FullStateCacheRecordOutcome;

#[derive(Debug)]
pub struct FullStateCache {
    mode: StageFullStateCacheMode,
    payload: StageFullStateCachePayload,
    max_entries: usize,
    max_bytes: u64,
    min_tokens: u64,
    clock: u64,
    logical_bytes: u64,
    total_bytes: u64,
    blobs: CacheBlobStore,
    entries: HashMap<FullStateCacheKey, FullStateCacheEntry>,
}

#[derive(Debug)]
struct FullStateCacheEntry {
    payload: FullStateCachePayloadEntry,
    byte_len: u64,
    last_used: u64,
}

impl FullStateCache {
    pub fn from_config(config: &StageConfig) -> Option<Self> {
        let cache = config.full_state_cache.as_ref()?;
        if cache.mode == StageFullStateCacheMode::Disabled {
            return None;
        }
        Some(Self {
            mode: cache.mode,
            payload: cache.payload,
            max_entries: cache.max_entries,
            max_bytes: cache.max_bytes,
            min_tokens: cache.min_tokens,
            clock: 0,
            logical_bytes: 0,
            total_bytes: 0,
            blobs: CacheBlobStore::new(DEFAULT_BLOCK_SIZE_BYTES),
            entries: HashMap::new(),
        })
    }

    pub fn should_lookup(&self) -> bool {
        matches!(self.mode, StageFullStateCacheMode::LookupRecord)
    }

    pub fn should_record(&self) -> bool {
        matches!(
            self.mode,
            StageFullStateCacheMode::Record | StageFullStateCacheMode::LookupRecord
        )
    }

    pub fn payload(&self) -> StageFullStateCachePayload {
        self.payload
    }

    pub fn lookup(
        &mut self,
        config: &StageConfig,
        message: &StageWireMessage,
        token_ids: &[i32],
    ) -> Result<Option<FullStateCacheLookup>> {
        if !self.should_lookup() {
            return Ok(None);
        }
        let Some(key) = self.key_for_message(config, message, token_ids)? else {
            return Ok(None);
        };
        self.clock = self.clock.saturating_add(1);
        let Some(entry) = self.entries.get_mut(&key) else {
            return Ok(None);
        };
        entry.last_used = self.clock;
        Ok(Some(FullStateCacheLookup {
            key,
            payload: entry.payload.clone(),
            entries: self.entries.len(),
            logical_bytes: self.logical_bytes,
            physical_bytes: self.total_bytes,
            dedupe_saved_bytes: self.logical_bytes.saturating_sub(self.total_bytes),
            blob_blocks: self.blobs.block_count(),
            total_bytes: self.total_bytes,
        }))
    }

    pub fn record(
        &mut self,
        config: &StageConfig,
        message: &StageWireMessage,
        token_ids: &[i32],
        payload: FullStateCachePayloadEntry,
    ) -> Result<Option<FullStateCacheRecordOutcome>> {
        if !self.should_record() {
            return Ok(None);
        }
        let Some(key) = self.key_for_message(config, message, token_ids)? else {
            return Ok(None);
        };
        let byte_len = payload.byte_len();

        self.clock = self.clock.saturating_add(1);
        if let Some(previous) = self.entries.remove(&key) {
            self.logical_bytes = self.logical_bytes.saturating_sub(previous.byte_len);
            previous.payload.release_from(&mut self.blobs);
            self.total_bytes = self.blobs.physical_bytes();
        }
        let (payload, dedupe_stats) = payload.dedupe_into(&mut self.blobs);
        self.logical_bytes = self.logical_bytes.saturating_add(byte_len);
        self.total_bytes = self.blobs.physical_bytes();
        let entry_physical_bytes = payload_unique_physical_bytes(&payload);
        let entry_dedupe_saved_bytes = byte_len.saturating_sub(entry_physical_bytes);
        self.entries.insert(
            key.clone(),
            FullStateCacheEntry {
                payload,
                byte_len,
                last_used: self.clock,
            },
        );
        let (evicted_entries, evicted_bytes, evicted_physical_bytes) =
            self.evict_until_within_limits(&key);
        let stored = self.entries.contains_key(&key);
        let reason = if stored {
            None
        } else if self.max_bytes > 0 && entry_physical_bytes > self.max_bytes {
            Some("entry_exceeds_max_bytes".to_string())
        } else {
            Some("evicted_immediately".to_string())
        };
        Ok(Some(FullStateCacheRecordOutcome {
            key,
            stored,
            bytes: byte_len,
            physical_bytes: entry_physical_bytes,
            dedupe_saved_bytes: entry_dedupe_saved_bytes,
            dedupe_hash_ms: dedupe_stats.hash_ms,
            dedupe_hash_bytes: dedupe_stats.hash_bytes,
            dedupe_block_count: dedupe_stats.block_count,
            dedupe_new_block_count: dedupe_stats.new_block_count,
            dedupe_reused_block_count: dedupe_stats.reused_block_count,
            evicted_entries,
            evicted_bytes,
            evicted_physical_bytes,
            entries: self.entries.len(),
            logical_bytes: self.logical_bytes,
            physical_total_bytes: self.total_bytes,
            dedupe_saved_total_bytes: self.logical_bytes.saturating_sub(self.total_bytes),
            blob_blocks: self.blobs.block_count(),
            total_bytes: self.total_bytes,
            reason,
        }))
    }

    pub fn key_for_message(
        &self,
        config: &StageConfig,
        message: &StageWireMessage,
        token_ids: &[i32],
    ) -> Result<Option<FullStateCacheKey>> {
        if !eligible_prefill_message(message) {
            return Ok(None);
        }
        if token_ids.is_empty() {
            return Ok(None);
        }
        let token_start = u64::try_from(message.pos_start)
            .map_err(|_| anyhow!("negative prefill token start is not cacheable"))?;
        let token_count = u64::try_from(message.token_count)
            .map_err(|_| anyhow!("negative prefill token count is not cacheable"))?;
        if token_count < self.min_tokens {
            return Ok(None);
        }
        if usize::try_from(token_count).ok() != Some(token_ids.len()) {
            return Ok(None);
        }
        Ok(Some(FullStateCacheKey {
            namespace: config
                .full_state_cache
                .as_ref()
                .and_then(|cache| cache.namespace.clone()),
            model_id: config.model_id.clone(),
            topology_id: config.topology_id.clone(),
            stage_id: config.stage_id.clone(),
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            ctx_size: config.ctx_size,
            cache_type_k: config.cache_type_k.clone(),
            cache_type_v: config.cache_type_v.clone(),
            load_mode: format!("{:?}", config.load_mode),
            filter_tensors_on_load: config.filter_tensors_on_load,
            token_start,
            token_count,
            token_sha256: token_hash(token_ids),
        }))
    }

    fn evict_until_within_limits(&mut self, protected: &FullStateCacheKey) -> (usize, u64, u64) {
        let mut evicted_entries = 0usize;
        let mut evicted_bytes = 0u64;
        let mut evicted_physical_bytes = 0u64;
        loop {
            let over_entries = self.entries.len() > self.max_entries;
            let over_bytes = self.max_bytes > 0 && self.total_bytes > self.max_bytes;
            if !over_entries && !over_bytes {
                break;
            }
            let victim = self
                .entries
                .iter()
                .filter(|(key, _)| *key != protected || self.entries.len() == 1)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());
            let Some(victim) = victim else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim) {
                evicted_entries += 1;
                evicted_bytes = evicted_bytes.saturating_add(entry.byte_len);
                self.logical_bytes = self.logical_bytes.saturating_sub(entry.byte_len);
                evicted_physical_bytes = evicted_physical_bytes
                    .saturating_add(entry.payload.release_from(&mut self.blobs));
                self.total_bytes = self.blobs.physical_bytes();
            }
        }
        (evicted_entries, evicted_bytes, evicted_physical_bytes)
    }
}

pub fn eligible_prefill_message(message: &StageWireMessage) -> bool {
    matches!(message.kind, WireMessageKind::PrefillEmbd) && !message.kind.requires_predicted_reply()
}

pub fn cache_attrs(
    prefix: &str,
    key: Option<&FullStateCacheKey>,
    entries: Option<usize>,
    total_bytes: Option<u64>,
) -> BTreeMap<String, serde_json::Value> {
    let mut attrs = BTreeMap::new();
    if let Some(key) = key {
        attrs.insert(format!("{prefix}.key"), serde_json::json!(key.short_id()));
        if let Some(namespace) = key.namespace.as_ref() {
            attrs.insert(format!("{prefix}.namespace"), serde_json::json!(namespace));
        }
        attrs.insert(
            format!("{prefix}.token_start"),
            serde_json::json!(key.token_start),
        );
        attrs.insert(
            format!("{prefix}.token_count"),
            serde_json::json!(key.token_count),
        );
    }
    if let Some(entries) = entries {
        attrs.insert(format!("{prefix}.entries"), serde_json::json!(entries));
    }
    if let Some(total_bytes) = total_bytes {
        attrs.insert(
            format!("{prefix}.total_bytes"),
            serde_json::json!(total_bytes),
        );
    }
    attrs
}

fn token_hash(token_ids: &[i32]) -> String {
    let mut hasher = Sha256::new();
    for token in token_ids {
        hasher.update(token.to_le_bytes());
    }
    hex_lower(&hasher.finalize())
}

fn payload_unique_physical_bytes(payload: &FullStateCachePayloadEntry) -> u64 {
    let mut sizes = HashMap::<String, u64>::new();
    collect_payload_block_sizes(payload, &mut sizes);
    sizes.values().copied().sum()
}

fn collect_payload_block_sizes(
    payload: &FullStateCachePayloadEntry,
    sizes: &mut HashMap<String, u64>,
) {
    match payload {
        FullStateCachePayloadEntry::FullState { bytes } => collect_cache_bytes_sizes(bytes, sizes),
        FullStateCachePayloadEntry::RecurrentOnly { recurrent } => {
            collect_cache_bytes_sizes(recurrent, sizes)
        }
        FullStateCachePayloadEntry::KvRecurrent { kv, recurrent, .. } => {
            collect_cache_bytes_sizes(kv, sizes);
            collect_cache_bytes_sizes(recurrent, sizes);
        }
    }
}

fn collect_cache_bytes_sizes(bytes: &CacheBytes, sizes: &mut HashMap<String, u64>) {
    let CacheBytesRepr::Blocks(blocks) = &bytes.repr else {
        return;
    };
    for block in blocks.iter() {
        sizes
            .entry(block.hash.clone())
            .or_insert_with(|| block.bytes.len() as u64);
    }
}

fn blake3_hex(bytes: &[u8]) -> String {
    let digest = blake3::hash(bytes);
    hex_lower(digest.as_bytes())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use skippy_protocol::{LoadMode, StageConfig, StageFullStateCacheConfig};

    use super::*;

    fn config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topo".to_string(),
            model_id: "model".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 512,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            filter_tensors_on_load: false,
            selected_device: None,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:1".to_string(),
            upstream: None,
            downstream: None,
            full_state_cache: Some(StageFullStateCacheConfig {
                mode: StageFullStateCacheMode::LookupRecord,
                payload: StageFullStateCachePayload::FullState,
                namespace: None,
                max_entries: 1,
                max_bytes: 0,
                min_tokens: 0,
            }),
        }
    }

    fn config_with_limits(max_entries: usize, max_bytes: u64) -> StageConfig {
        let mut config = config();
        config.full_state_cache = Some(StageFullStateCacheConfig {
            mode: StageFullStateCacheMode::LookupRecord,
            payload: StageFullStateCachePayload::FullState,
            namespace: None,
            max_entries,
            max_bytes,
            min_tokens: 0,
        });
        config
    }

    fn prefill(tokens: Vec<i32>) -> StageWireMessage {
        let mut state = skippy_protocol::binary::StageStateHeader::new(
            WireMessageKind::PrefillEmbd,
            skippy_protocol::binary::WireActivationDType::F32,
        );
        state.prompt_token_count = tokens.len() as i32;
        StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: tokens.len() as i32,
            state,
            request_id: 1,
            session_id: 1,
            sampling: None,
            tokens,
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn record_lookup_and_evict() {
        let config = config();
        let mut cache = FullStateCache::from_config(&config).expect("cache");
        let first = prefill(vec![1, 2, 3]);
        let second = prefill(vec![4, 5, 6]);

        cache
            .record(
                &config,
                &first,
                &first.tokens,
                FullStateCachePayloadEntry::full_state(vec![7; 8]),
            )
            .expect("record")
            .expect("outcome");
        assert!(cache
            .lookup(&config, &first, &first.tokens)
            .expect("lookup")
            .is_some());

        let outcome = cache
            .record(
                &config,
                &second,
                &second.tokens,
                FullStateCachePayloadEntry::full_state(vec![9; 8]),
            )
            .expect("record")
            .expect("outcome");
        assert_eq!(outcome.evicted_entries, 1);
        assert!(cache
            .lookup(&config, &first, &first.tokens)
            .expect("lookup")
            .is_none());
        assert!(cache
            .lookup(&config, &second, &second.tokens)
            .expect("lookup")
            .is_some());
    }

    #[test]
    fn record_dedupes_blocks_and_reconstructs_exact_bytes() {
        let config = config_with_limits(2, 0);
        let mut cache = FullStateCache::from_config(&config).expect("cache");
        let first = prefill(vec![1, 2, 3]);
        let second = prefill(vec![4, 5, 6]);
        let shared = vec![7u8; DEFAULT_BLOCK_SIZE_BYTES];
        let mut first_payload = shared.clone();
        first_payload.extend_from_slice(&[1, 2, 3, 4]);
        let mut second_payload = shared;
        second_payload.extend_from_slice(&[9, 8, 7, 6]);

        let first_outcome = cache
            .record(
                &config,
                &first,
                &first.tokens,
                FullStateCachePayloadEntry::full_state(first_payload.clone()),
            )
            .expect("record")
            .expect("outcome");
        assert_eq!(first_outcome.logical_bytes, first_payload.len() as u64);
        assert_eq!(
            first_outcome.physical_total_bytes,
            first_payload.len() as u64
        );
        assert_eq!(first_outcome.dedupe_saved_total_bytes, 0);

        let second_outcome = cache
            .record(
                &config,
                &second,
                &second.tokens,
                FullStateCachePayloadEntry::full_state(second_payload.clone()),
            )
            .expect("record")
            .expect("outcome");
        assert_eq!(
            second_outcome.logical_bytes,
            (first_payload.len() + second_payload.len()) as u64
        );
        assert_eq!(
            second_outcome.physical_total_bytes,
            (first_payload.len() + 4) as u64
        );
        assert_eq!(
            second_outcome.dedupe_saved_total_bytes,
            DEFAULT_BLOCK_SIZE_BYTES as u64
        );

        let lookup = cache
            .lookup(&config, &second, &second.tokens)
            .expect("lookup")
            .expect("hit");
        let restored = lookup.payload.full_state_bytes().expect("restore");
        assert_eq!(restored.as_ref(), second_payload.as_slice());
        assert_eq!(lookup.dedupe_saved_bytes, DEFAULT_BLOCK_SIZE_BYTES as u64);
    }

    #[test]
    fn namespace_partitions_cache_identity() {
        let mut alpha = config();
        alpha.full_state_cache.as_mut().unwrap().namespace = Some("tenant-alpha".to_string());
        let mut beta = alpha.clone();
        beta.full_state_cache.as_mut().unwrap().namespace = Some("tenant-beta".to_string());
        let message = prefill(vec![1, 2, 3]);

        let mut cache = FullStateCache::from_config(&alpha).expect("cache");
        cache
            .record(
                &alpha,
                &message,
                &message.tokens,
                FullStateCachePayloadEntry::full_state(vec![7; 8]),
            )
            .expect("record")
            .expect("outcome");

        assert!(cache
            .lookup(&alpha, &message, &message.tokens)
            .expect("alpha lookup")
            .is_some());
        assert!(cache
            .lookup(&beta, &message, &message.tokens)
            .expect("beta lookup")
            .is_none());
    }
}
