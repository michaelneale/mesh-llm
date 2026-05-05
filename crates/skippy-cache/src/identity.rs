use skippy_protocol::StageConfig;

pub const NATIVE_KV_RUNTIME_ABI_VERSION: &str = "stage-abi-0.1/native-kv-page-v1";
pub const NATIVE_KV_DTYPE: &str = "ggml-native-kv";
const NATIVE_KV_LAYER_CONTIGUOUS_LAYOUT: i32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixIdentity {
    pub prefix_hash: String,
    pub page_id: String,
    pub token_start: u64,
    pub token_count: u64,
}

pub fn prefix_identity(
    config: &StageConfig,
    token_start: u64,
    token_ids: &[i32],
) -> PrefixIdentity {
    let token_count = token_ids.len() as u64;
    let prefix_hash = prefix_hash(config, token_start, token_ids);
    let page_id = page_id(config, token_start, token_count, &prefix_hash);
    PrefixIdentity {
        prefix_hash,
        page_id,
        token_start,
        token_count,
    }
}

pub fn prefix_hash(config: &StageConfig, token_start: u64, token_ids: &[i32]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(config.model_id.as_bytes());
    hasher.update(config.topology_id.as_bytes());
    hasher.update(config.stage_id.as_bytes());
    hasher.update(&config.stage_index.to_le_bytes());
    hasher.update(&config.layer_start.to_le_bytes());
    hasher.update(&config.layer_end.to_le_bytes());
    hasher.update(NATIVE_KV_RUNTIME_ABI_VERSION.as_bytes());
    hasher.update(&NATIVE_KV_LAYER_CONTIGUOUS_LAYOUT.to_le_bytes());
    hasher.update(NATIVE_KV_DTYPE.as_bytes());
    hasher.update(format!("ctx:{}", config.ctx_size).as_bytes());
    hasher.update(&token_start.to_le_bytes());
    for token_id in token_ids {
        hasher.update(token_id.to_le_bytes().as_slice());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

pub fn page_id(
    config: &StageConfig,
    token_start: u64,
    token_count: u64,
    prefix_hash: &str,
) -> String {
    let digest = prefix_hash
        .strip_prefix("blake3:")
        .or_else(|| prefix_hash.strip_prefix("sha256:"))
        .unwrap_or(prefix_hash);
    let short = digest.get(..16).unwrap_or(digest);
    format!(
        "{}:{}:{}:{}:{}",
        config.stage_id, token_start, token_count, config.layer_start, short
    )
}

pub fn activation_page_id(page_id: &str, activation_width: i32) -> String {
    format!("act:{}:w{}", page_id, activation_width.max(0))
}
