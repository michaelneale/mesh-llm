use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[repr(i32)]
pub enum PageLayout {
    LayerContiguous = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[repr(i32)]
pub enum KvCodec {
    Fp16 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[repr(i32)]
pub enum PageState {
    Empty = 0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[repr(i32)]
pub enum ChecksumAlgorithm {
    Sha256 = 1,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct PageIdentity {
    pub model_id: String,
    pub model_revision: String,
    pub runtime_abi_version: String,
    pub topology_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub prefix_hash: String,
    pub session_id: String,
    pub token_start: u64,
    pub token_count: u64,
    pub generation: u64,
    pub layout: i32,
    pub codec: i32,
    pub tokenizer_id: String,
    pub chat_template_id: String,
    pub position_config_hash: String,
    pub kv_dtype: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Checksum {
    pub algorithm: i32,
    pub digest: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Lease {
    pub owner: String,
    pub expires_at_unix_nanos: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct KvPageManifest {
    pub schema_version: u32,
    pub page_id: String,
    pub identity: Option<PageIdentity>,
    pub state: i32,
    pub byte_size: u64,
    pub shm_offset: u64,
    pub shm_len: u64,
    pub checksum: Option<Checksum>,
    pub lease: Option<Lease>,
    pub annotations: BTreeMap<String, String>,
}
