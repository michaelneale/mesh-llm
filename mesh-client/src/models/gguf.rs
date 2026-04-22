use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// MoE info extracted from a GGUF file header.
#[derive(Clone, Debug)]
pub struct GgufMoeInfo {
    pub expert_count: u32,
    pub expert_used_count: u32,
}

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

fn read_i64(f: &mut std::fs::File) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_gguf_string(f: &mut std::fs::File) -> std::io::Result<String> {
    let len = read_u64(f)? as usize;
    if len > 1_000_000 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "string too long",
        ));
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn skip_gguf_value(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<()> {
    match typ {
        GgufType::String => {
            let _ = read_gguf_string(f)?;
        }
        GgufType::Array => {
            let elem_type = GgufType::from_u32(read_u32(f)?).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type")
            })?;
            let count = read_u64(f)? as usize;
            for _ in 0..count {
                skip_gguf_value(f, elem_type)?;
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
        GgufType::Int32 => Ok(Some(read_u32(f)?)),
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

/// Detect MoE parameters from a GGUF file by reading its header KV pairs.
///
/// Scans for `*.expert_count` and `*.expert_used_count` keys.
/// Returns None if the file isn't MoE (no expert_count or expert_count <= 1).
/// Takes ~1ms for typical GGUF files and only reads the header, not tensor data.
pub fn detect_moe(path: &Path) -> Option<GgufMoeInfo> {
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

    let _n_tensors = read_i64(&mut f).ok()?;
    let n_kv = read_i64(&mut f).ok()?;

    let mut expert_count: Option<u32> = None;
    let mut expert_used_count: Option<u32> = None;

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key.ends_with(".expert_count") {
            expert_count = read_gguf_value_as_u32(&mut f, vtype).ok()?;
        } else if key.ends_with(".expert_used_count") {
            expert_used_count = read_gguf_value_as_u32(&mut f, vtype).ok()?;
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }

        if expert_count.is_some() && expert_used_count.is_some() {
            break;
        }
    }

    match (expert_count, expert_used_count) {
        (Some(ec), Some(euc)) if ec > 1 => Some(GgufMoeInfo {
            expert_count: ec,
            expert_used_count: euc,
        }),
        _ => None,
    }
}

#[derive(Clone, Debug, Default)]
pub struct GgufCompactMeta {
    pub architecture: String,
    pub context_length: u32,
    pub vocab_size: u32,
    pub embedding_size: u32,
    pub head_count: u32,
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
    let _n_tensors = read_i64(&mut f).ok()?;
    let n_kv = read_i64(&mut f).ok()?;

    let mut meta = GgufCompactMeta::default();
    let mut kv_head_count: u32 = 0;

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
                kv_head_count = v;
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

    if meta.key_length == 0 {
        if let Some(key_length) = meta.embedding_size.checked_div(meta.head_count) {
            meta.key_length = key_length;
        }
    }
    if meta.value_length == 0 {
        let effective_kv = if kv_head_count > 0 {
            kv_head_count
        } else {
            meta.head_count
        };
        if let Some(value_length) = meta.embedding_size.checked_div(effective_kv) {
            meta.value_length = value_length;
        }
    }

    Some(meta)
}
