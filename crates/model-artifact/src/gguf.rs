use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_ARRAY_DEPTH: u32 = 64;
const MAX_GGUF_TENSOR_DIMS: u32 = 8;
const MAX_GGUF_HEADER_KV_COUNT: usize = 1_000_000;
const MAX_GGUF_TENSOR_COUNT: usize = 1_000_000;

/// GGUF value types (matching gguf.h enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    fn fixed_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut std::fs::File) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(f: &mut std::fs::File) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64(f: &mut std::fs::File) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_gguf_header_count(
    f: &mut std::fs::File,
    max: usize,
    label: &str,
) -> std::io::Result<usize> {
    let value = read_i64(f)?;
    let count = usize::try_from(value).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("negative {label}"))
    })?;
    if count > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too large"),
        ));
    }
    Ok(count)
}

fn read_bounded_len(f: &mut std::fs::File, max: u64, label: &str) -> std::io::Result<usize> {
    let len = read_u64(f)?;
    if len > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too long"),
        ));
    }
    usize::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too large"),
        )
    })
}

fn read_gguf_string(f: &mut std::fs::File) -> std::io::Result<String> {
    let len = read_bounded_len(f, MAX_GGUF_STRING_BYTES, "string")?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid UTF-8 in GGUF string",
        )
    })
}

fn skip_gguf_value(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<()> {
    skip_gguf_value_with_depth(f, typ, 0)
}

fn skip_gguf_value_with_depth(
    f: &mut std::fs::File,
    typ: GgufType,
    depth: u32,
) -> std::io::Result<()> {
    match typ {
        GgufType::String => {
            let _ = read_gguf_string(f)?;
        }
        GgufType::Array => {
            if depth >= MAX_GGUF_ARRAY_DEPTH {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "GGUF nesting too deep",
                ));
            }
            let elem_type = GgufType::from_u32(read_u32(f)?).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type")
            })?;
            let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "array")?;
            for _ in 0..count {
                skip_gguf_value_with_depth(f, elem_type, depth + 1)?;
            }
        }
        other => {
            let size = other.fixed_size().unwrap_or(0);
            f.seek(SeekFrom::Current(size as i64))?;
        }
    }
    Ok(())
}

fn read_gguf_value_as_u32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<u32>> {
    match typ {
        GgufType::Uint32 => Ok(Some(read_u32(f)?)),
        GgufType::Int32 => {
            let value = read_i32(f)?;
            let value = u32::try_from(value).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "negative Int32 where unsigned GGUF value was expected",
                )
            })?;
            Ok(Some(value))
        }
        GgufType::Uint16 => {
            let mut buf = [0u8; 2];
            f.read_exact(&mut buf)?;
            Ok(Some(u16::from_le_bytes(buf) as u32))
        }
        GgufType::Uint8 => {
            let mut buf = [0u8; 1];
            f.read_exact(&mut buf)?;
            Ok(Some(buf[0] as u32))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_f32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<f32>> {
    match typ {
        GgufType::Float32 => {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(Some(f32::from_le_bytes(buf)))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_string_opt(
    f: &mut std::fs::File,
    typ: GgufType,
) -> std::io::Result<Option<String>> {
    match typ {
        GgufType::String => Ok(Some(read_gguf_string(f)?)),
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GgufCompactMeta {
    pub architecture: String,
    pub context_length: u32,
    pub vocab_size: u32,
    pub embedding_size: u32,
    pub head_count: u32,
    pub kv_head_count: u32,
    pub layer_count: u32,
    pub feed_forward_length: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub tokenizer_model_name: String,
    pub rope_scale: f32,
    pub rope_freq_base: f32,
    pub expert_count: u32,
    pub expert_used_count: u32,
}

impl GgufCompactMeta {
    pub fn effective_kv_head_count(&self) -> Option<u32> {
        if self.kv_head_count > 0 {
            Some(self.kv_head_count)
        } else if self.head_count > 0 {
            Some(self.head_count)
        } else {
            None
        }
    }

    pub fn k_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().k_cache_bytes_per_token(self)
    }

    pub fn v_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().v_cache_bytes_per_token(self)
    }

    pub fn kv_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().kv_cache_bytes_per_token(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufKvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl GgufKvCacheType {
    pub fn from_llama_arg(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "q8_0" => Some(Self::Q8_0),
            "q4_0" => Some(Self::Q4_0),
            _ => None,
        }
    }

    pub const fn as_llama_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    fn block_shape(self) -> (u64, u64) {
        match self {
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4_0 => (32, 18),
        }
    }

    fn bytes_for_elements(self, elements: u64) -> Option<u64> {
        let (block_elements, block_bytes) = self.block_shape();
        let blocks = elements
            .checked_add(block_elements.checked_sub(1)?)?
            .checked_div(block_elements)?;
        blocks.checked_mul(block_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufKvCacheQuant {
    pub k: GgufKvCacheType,
    pub v: GgufKvCacheType,
}

impl GgufKvCacheQuant {
    /// f16 K + f16 V — highest quality, largest KV cache.
    pub const F16: Self = Self {
        k: GgufKvCacheType::F16,
        v: GgufKvCacheType::F16,
    };

    /// q8_0 K + q8_0 V — moderate compression.
    pub const Q8_0: Self = Self {
        k: GgufKvCacheType::Q8_0,
        v: GgufKvCacheType::Q8_0,
    };

    /// q4_0 K + q4_0 V — most aggressive compression, smallest KV cache.
    pub const Q4_0: Self = Self {
        k: GgufKvCacheType::Q4_0,
        v: GgufKvCacheType::Q4_0,
    };

    pub const fn new(k: GgufKvCacheType, v: GgufKvCacheType) -> Self {
        Self { k, v }
    }

    pub const fn f16() -> Self {
        Self::F16
    }

    /// Returns `true` if `self` uses more aggressive (smaller) quantisation
    /// than `other`.
    pub const fn is_more_aggressive_than(self, other: Self) -> bool {
        Self::aggressiveness(self) > Self::aggressiveness(other)
    }

    const fn aggressiveness(q: Self) -> u8 {
        Self::type_aggressiveness(q.k) + Self::type_aggressiveness(q.v)
    }

    const fn type_aggressiveness(t: GgufKvCacheType) -> u8 {
        match t {
            GgufKvCacheType::F16 => 0,
            GgufKvCacheType::Q8_0 => 1,
            GgufKvCacheType::Q4_0 => 2,
        }
    }

    pub fn from_llama_args(cache_type_k: &str, cache_type_v: &str) -> Option<Self> {
        Some(Self {
            k: GgufKvCacheType::from_llama_arg(cache_type_k)?,
            v: GgufKvCacheType::from_llama_arg(cache_type_v)?,
        })
    }

    pub fn k_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        cache_bytes_per_token(meta, meta.key_length, self.k)
    }

    pub fn v_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        cache_bytes_per_token(meta, meta.value_length, self.v)
    }

    pub fn kv_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        self.k_cache_bytes_per_token(meta)?
            .checked_add(self.v_cache_bytes_per_token(meta)?)
    }
}

fn cache_bytes_per_token(
    meta: &GgufCompactMeta,
    vector_length: u32,
    cache_type: GgufKvCacheType,
) -> Option<u64> {
    let kv_heads = u64::from(meta.effective_kv_head_count()?);
    let vector_length = u64::from((vector_length > 0).then_some(vector_length)?);
    let layers = u64::from((meta.layer_count > 0).then_some(meta.layer_count)?);
    let elements_per_layer = kv_heads.checked_mul(vector_length)?;
    cache_type
        .bytes_for_elements(elements_per_layer)?
        .checked_mul(layers)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GgufTensorByteProfile {
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub full_model_bytes: u64,
    pub base_resident_bytes: u64,
    pub expert_tensor_bytes: u64,
    pub file_overhead_bytes: u64,
}

#[derive(Clone, Debug)]
struct GgufTensorInfo {
    name: String,
    offset: u64,
}

/// Scan a GGUF file header and return compact structural metadata.
/// Reads only the KV section, never tensor data. Returns None on any parse failure.
pub fn scan_gguf_compact_meta(path: &Path) -> Option<GgufCompactMeta> {
    let mut f = std::fs::File::open(path).ok()?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None;
    }
    let _n_tensors = read_gguf_header_count(&mut f, MAX_GGUF_TENSOR_COUNT, "tensor count").ok()?;
    let n_kv = read_gguf_header_count(&mut f, MAX_GGUF_HEADER_KV_COUNT, "KV count").ok()?;

    let mut meta = GgufCompactMeta::default();
    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key == "general.architecture" {
            meta.architecture = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key == "tokenizer.ggml.model" {
            meta.tokenizer_model_name = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key.ends_with(".context_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.context_length = v;
            }
        } else if key.ends_with(".embedding_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.embedding_size = v;
            }
        } else if key.ends_with(".head_count") && !key.ends_with("_kv") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.head_count = v;
            }
        } else if key.ends_with(".attention.head_count_kv") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.kv_head_count = v;
            }
        } else if key.ends_with(".block_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.layer_count = v;
            }
        } else if key.ends_with(".feed_forward_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.feed_forward_length = v;
            }
        } else if key.ends_with(".attention.key_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.key_length = v;
            }
        } else if key.ends_with(".attention.value_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.value_length = v;
            }
        } else if key.ends_with(".rope.scale") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_scale = v;
            }
        } else if key.ends_with(".rope.freq_base") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_freq_base = v;
            }
        } else if key.ends_with(".vocab_size") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.vocab_size = v;
            }
        } else if key.ends_with(".expert_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_count = v;
            }
        } else if key.ends_with(".expert_used_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_used_count = v;
            }
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }
    }

    if meta.key_length == 0 && meta.head_count > 0 {
        if let Some(key_length) = meta.embedding_size.checked_div(meta.head_count) {
            meta.key_length = key_length;
        }
    }
    if meta.value_length == 0 {
        if let Some(effective_kv) = meta.effective_kv_head_count() {
            if let Some(value_length) = meta.embedding_size.checked_div(effective_kv) {
                meta.value_length = value_length;
            }
        }
    }

    Some(meta)
}

fn align_offset(value: u64, alignment: u32) -> u64 {
    let alignment = u64::from(alignment.max(1));
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

fn read_tensor_infos(
    f: &mut std::fs::File,
    n_tensors: usize,
) -> std::io::Result<Vec<GgufTensorInfo>> {
    let mut tensors = Vec::new();
    tensors.try_reserve(n_tensors).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "GGUF tensor count requires too much memory",
        )
    })?;
    for _ in 0..n_tensors {
        let name = read_gguf_string(f)?;
        let n_dims = read_u32(f)?;
        if n_dims > MAX_GGUF_TENSOR_DIMS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "too many GGUF tensor dimensions",
            ));
        }
        for _ in 0..n_dims {
            let _ = read_u64(f)?;
        }
        let _ = read_u32(f)?;
        let offset = read_u64(f)?;
        tensors.push(GgufTensorInfo { name, offset });
    }
    Ok(tensors)
}

fn is_expert_partitioned_tensor(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.contains("shared_expert") || lower.contains("sharedexpert") || lower.contains("shexp")
    {
        return false;
    }

    lower.contains("ffn_gate_exps")
        || lower.contains("ffn_up_exps")
        || lower.contains("ffn_down_exps")
        || lower.contains("exp_probs")
        || lower.contains(".expert")
        || lower.contains("_expert")
}

/// Scan GGUF tensor metadata and estimate which bytes are always resident versus
/// expert-partitioned. Reads only the header and tensor-info tables.
pub fn scan_gguf_tensor_byte_profile(path: &Path) -> Option<GgufTensorByteProfile> {
    let mut f = std::fs::File::open(path).ok()?;
    let file_len = f.metadata().ok()?.len();

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None;
    }

    let n_tensors = read_gguf_header_count(&mut f, MAX_GGUF_TENSOR_COUNT, "tensor count").ok()?;
    let n_kv = read_gguf_header_count(&mut f, MAX_GGUF_HEADER_KV_COUNT, "KV count").ok()?;

    let mut expert_count = 0u32;
    let mut expert_used_count = 0u32;
    let mut alignment = 32u32;

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key == "general.alignment" {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                alignment = value.max(1);
            }
        } else if key.ends_with(".expert_count") {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                expert_count = value;
            }
        } else if key.ends_with(".expert_used_count") {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                expert_used_count = value;
            }
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }
    }

    let mut tensors = read_tensor_infos(&mut f, n_tensors).ok()?;
    if tensors.is_empty() {
        return Some(GgufTensorByteProfile {
            expert_count,
            expert_used_count,
            full_model_bytes: file_len,
            base_resident_bytes: 0,
            expert_tensor_bytes: 0,
            file_overhead_bytes: file_len,
        });
    }

    let tensor_info_end = f.stream_position().ok()?;
    let data_start = align_offset(tensor_info_end, alignment);
    if data_start > file_len {
        return None;
    }
    let data_len = file_len - data_start;

    tensors.sort_by_key(|tensor| tensor.offset);
    if tensors.first()?.offset > data_len {
        return None;
    }

    let mut base_resident_bytes = 0u64;
    let mut expert_tensor_bytes = 0u64;
    for (index, tensor) in tensors.iter().enumerate() {
        let next_offset = tensors
            .get(index + 1)
            .map(|next| next.offset)
            .unwrap_or(data_len);
        if next_offset < tensor.offset || next_offset > data_len {
            return None;
        }
        let tensor_bytes = next_offset - tensor.offset;
        if is_expert_partitioned_tensor(&tensor.name) {
            expert_tensor_bytes = expert_tensor_bytes.saturating_add(tensor_bytes);
        } else {
            base_resident_bytes = base_resident_bytes.saturating_add(tensor_bytes);
        }
    }

    let file_overhead_bytes = file_len.saturating_sub(base_resident_bytes + expert_tensor_bytes);
    Some(GgufTensorByteProfile {
        expert_count,
        expert_used_count,
        full_model_bytes: file_len,
        base_resident_bytes,
        expert_tensor_bytes,
        file_overhead_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file_path(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{unique}.gguf"))
    }

    fn write_bytes(prefix: &str, bytes: &[u8]) -> PathBuf {
        let path = temp_file_path(prefix);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        path
    }

    fn push_array_header(bytes: &mut Vec<u8>, elem_type: GgufType, count: u64) {
        bytes.extend_from_slice(&(elem_type as u32).to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&(GgufType::Uint32 as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_tensor_info(bytes: &mut Vec<u8>, name: &str, offset: u64) {
        push_gguf_string(bytes, name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes());
        bytes.extend_from_slice(&(GgufType::Uint8 as u32).to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    #[test]
    fn skip_gguf_value_rejects_excessive_array_depth() {
        let mut bytes = Vec::new();
        for _ in 0..=MAX_GGUF_ARRAY_DEPTH {
            push_array_header(&mut bytes, GgufType::Array, 1);
        }
        push_array_header(&mut bytes, GgufType::Uint8, 1);
        bytes.push(0);

        let path = write_bytes("model-artifact-gguf-depth", &bytes);
        let mut file = std::fs::File::open(&path).unwrap();
        let err = skip_gguf_value(&mut file, GgufType::Array).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("nesting too deep"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn skip_gguf_value_rejects_excessive_array_count() {
        let mut bytes = Vec::new();
        push_array_header(&mut bytes, GgufType::Uint8, MAX_GGUF_ARRAY_ELEMENTS + 1);

        let path = write_bytes("model-artifact-gguf-count", &bytes);
        let mut file = std::fs::File::open(&path).unwrap();
        let err = skip_gguf_value(&mut file, GgufType::Array).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("array too long"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_returns_none_on_malicious_nested_array() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&1i64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        for _ in 0..=MAX_GGUF_ARRAY_DEPTH {
            push_array_header(&mut bytes, GgufType::Array, 1);
        }
        push_array_header(&mut bytes, GgufType::Uint8, 1);
        bytes.push(0);

        let path = write_bytes("model-artifact-gguf-malicious", &bytes);
        assert!(scan_gguf_compact_meta(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_derives_value_length_from_kv_heads_without_head_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);

        let path = write_bytes("model-artifact-gguf-kv-heads", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.head_count, 0);
        assert_eq!(meta.kv_head_count, 8);
        assert_eq!(meta.key_length, 0);
        assert_eq!(meta.value_length, 512);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_preserves_kv_head_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&6i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
        push_u32_kv(&mut bytes, "llama.block_count", 24);
        push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
        push_u32_kv(&mut bytes, "llama.attention.value_length", 128);

        let path = write_bytes("model-artifact-gguf-kv-head-count", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.head_count, 32);
        assert_eq!(meta.kv_head_count, 8);
        assert_eq!(meta.effective_kv_head_count(), Some(8));
        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(49_152));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), Some(49_152));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn kv_cache_quant_prices_key_and_value_types_independently() {
        let meta = GgufCompactMeta {
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };
        let quant = GgufKvCacheQuant::new(GgufKvCacheType::Q8_0, GgufKvCacheType::Q4_0);

        assert_eq!(quant.k_cache_bytes_per_token(&meta), Some(26_112));
        assert_eq!(quant.v_cache_bytes_per_token(&meta), Some(13_824));
        assert_eq!(quant.kv_cache_bytes_per_token(&meta), Some(39_936));
    }

    #[test]
    fn kv_cache_bytes_per_token_returns_none_when_required_fields_are_missing() {
        let meta = GgufCompactMeta {
            head_count: 32,
            layer_count: 24,
            key_length: 128,
            ..Default::default()
        };

        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(196_608));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), None);
        assert_eq!(
            GgufKvCacheQuant::f16().kv_cache_bytes_per_token(&meta),
            None
        );
    }

    #[test]
    fn scan_gguf_compact_meta_rejects_negative_kv_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&(-1i64).to_le_bytes());

        let path = write_bytes("model-artifact-gguf-negative-kv", &bytes);
        assert!(scan_gguf_compact_meta(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_byte_profile_rejects_excessive_tensor_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&((MAX_GGUF_TENSOR_COUNT as i64) + 1).to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());

        let path = write_bytes("model-artifact-gguf-too-many-tensors", &bytes);
        assert!(scan_gguf_tensor_byte_profile(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_gguf_value_as_u32_rejects_negative_int32() {
        let path = write_bytes("model-artifact-gguf-negative-int32", &(-1i32).to_le_bytes());
        let mut file = std::fs::File::open(&path).unwrap();
        let err = read_gguf_value_as_u32(&mut file, GgufType::Int32).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err
            .to_string()
            .contains("negative Int32 where unsigned GGUF value was expected"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_byte_profile_splits_base_and_expert_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        bytes.extend_from_slice(&3i64.to_le_bytes());

        push_u32_kv(&mut bytes, "general.alignment", 32);
        push_u32_kv(&mut bytes, "llama.expert_count", 8);
        push_u32_kv(&mut bytes, "llama.expert_used_count", 2);

        push_tensor_info(&mut bytes, "blk.0.ffn_up_exps.weight", 0);
        push_tensor_info(&mut bytes, "blk.0.attn_q.weight", 64);

        let data_start = align_offset(bytes.len() as u64, 32) as usize;
        bytes.resize(data_start, 0);
        bytes.resize(data_start + 96, 0);

        let path = write_bytes("model-artifact-gguf-tensors", &bytes);
        let profile = scan_gguf_tensor_byte_profile(&path).unwrap();
        assert_eq!(profile.expert_count, 8);
        assert_eq!(profile.expert_used_count, 2);
        assert_eq!(profile.expert_tensor_bytes, 64);
        assert_eq!(profile.base_resident_bytes, 32);
        assert_eq!(profile.full_model_bytes, bytes.len() as u64);
        assert_eq!(
            profile.full_model_bytes,
            profile.base_resident_bytes + profile.expert_tensor_bytes + profile.file_overhead_bytes
        );
        let _ = std::fs::remove_file(path);
    }
}
