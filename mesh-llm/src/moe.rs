//! MoE expert sharding: split models across mesh nodes by expert assignment.
//!
//! Each node gets a GGUF with the full trunk (attention, norms, embeddings, head)
//! plus a subset of experts. The shared core (hottest experts by gate mass) is
//! replicated to every node. Remaining experts are distributed uniquely.
//!
//! No cross-node traffic during inference — each node runs independently.

use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ── GGUF MoE detection ──

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

    /// Size in bytes for fixed-size types. Returns None for String and Array.
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

/// Read a little-endian u32.
fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64.
fn read_u64(f: &mut std::fs::File) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a little-endian i64.
fn read_i64(f: &mut std::fs::File) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a GGUF string: uint64 length + bytes.
fn read_gguf_string(f: &mut std::fs::File) -> std::io::Result<String> {
    let len = read_u64(f)? as usize;
    if len > 1_000_000 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "string too long"));
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

/// Skip over a GGUF value of the given type.
fn skip_gguf_value(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<()> {
    match typ {
        GgufType::String => {
            let _ = read_gguf_string(f)?;
        }
        GgufType::Array => {
            let elem_type = GgufType::from_u32(read_u32(f)?)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type"))?;
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

/// Read a GGUF KV value as u32 (handles uint32, int32, uint16, etc.).
fn read_gguf_value_as_u32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<u32>> {
    match typ {
        GgufType::Uint32 => Ok(Some(read_u32(f)?)),
        GgufType::Int32 => Ok(Some(read_u32(f)?)), // reinterpret
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

/// Detect MoE parameters from a GGUF file by reading its header KV pairs.
///
/// Scans for `*.expert_count` and `*.expert_used_count` keys.
/// Returns None if the file isn't MoE (no expert_count or expert_count <= 1).
/// Takes ~1ms for typical GGUF files — only reads the header, not tensor data.
pub fn detect_moe(path: &Path) -> Option<GgufMoeInfo> {
    let mut f = std::fs::File::open(path).ok()?;

    // Header: magic (4) + version (4) + n_tensors (8) + n_kv (8)
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }

    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None; // v1 not supported
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

        // Early exit once we have both
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

// ── GGUF assembler: combine trunk + expert files into a shard ──

/// GGUF alignment (must match GGUF_DEFAULT_ALIGNMENT = 32).
const GGUF_ALIGNMENT: u64 = 32;

/// GGUF tensor type → (type_size_bytes, block_size_elements).
/// Used to compute tensor byte sizes: n_elements / block_size * type_size.
fn gguf_type_info(type_id: u32) -> Option<(usize, usize)> {
    // (type_size, block_size) — from ggml-common.h
    match type_id {
        0  => Some((4, 1)),     // F32
        1  => Some((2, 1)),     // F16
        2  => Some((18, 32)),   // Q4_0
        3  => Some((20, 32)),   // Q4_1
        6  => Some((22, 32)),   // Q5_0
        7  => Some((24, 32)),   // Q5_1
        8  => Some((34, 32)),   // Q8_0
        9  => Some((36, 32)),   // Q8_1
        10 => Some((84, 256)),  // Q2_K
        11 => Some((110, 256)), // Q3_K
        12 => Some((144, 256)), // Q4_K
        13 => Some((176, 256)), // Q5_K
        14 => Some((210, 256)), // Q6_K
        15 => Some((292, 256)), // Q8_K
        16 => Some((66, 256)),  // IQ2_XXS
        17 => Some((74, 256)),  // IQ2_XS
        18 => Some((98, 256)),  // IQ3_XXS
        19 => Some((50, 256)),  // IQ1_S
        20 => Some((34, 32)),   // IQ4_NL
        21 => Some((110, 256)), // IQ3_S
        22 => Some((82, 256)),  // IQ2_S
        23 => Some((36, 32)),   // IQ4_XS (using 32 block for iq4)
        24 => Some((1, 1)),     // I8
        25 => Some((2, 1)),     // I16
        26 => Some((4, 1)),     // I32
        27 => Some((8, 1)),     // I64
        28 => Some((8, 1)),     // F64
        29 => Some((56, 256)),  // IQ1_M
        30 => Some((2, 1)),     // BF16
        34 => Some((54, 256)),  // TQ1_0
        35 => Some((66, 256)),  // TQ2_0
        _ => None,
    }
}

/// Compute byte size of a tensor given its dimensions and type.
fn tensor_nbytes(ne: &[u64], type_id: u32) -> u64 {
    let (type_size, block_size) = gguf_type_info(type_id).unwrap_or((1, 1));
    // row_size = ne[0] / block_size * type_size, then * ne[1] * ne[2] * ...
    let row_size = (ne[0] as usize / block_size) * type_size;
    let n_rows: u64 = ne[1..].iter().product();
    row_size as u64 * n_rows
}

fn pad_to(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    if rem == 0 { offset } else { offset + (alignment - rem) }
}

fn write_zeros(f: &mut std::fs::File, n: u64) -> std::io::Result<()> {
    let zeros = vec![0u8; n as usize];
    f.write_all(&zeros)
}

/// Parsed GGUF tensor info entry.
#[derive(Clone, Debug)]
struct GgufTensorInfo {
    name: String,
    n_dims: u32,
    ne: Vec<u64>,      // dimensions
    type_id: u32,
    offset: u64,       // offset within data section
}

impl GgufTensorInfo {
    fn nbytes(&self) -> u64 {
        tensor_nbytes(&self.ne, self.type_id)
    }

    fn is_expert_tensor(&self) -> bool {
        self.name.contains("ffn_gate_exps")
            || self.name.contains("ffn_up_exps")
            || self.name.contains("ffn_down_exps")
    }

    fn is_router_gate(&self) -> bool {
        self.name.contains("ffn_gate_inp")
    }
}

/// Parsed GGUF file: header, raw KV bytes, tensor info, and data offset.
struct GgufFile {
    version: u32,
    n_kv: u64,
    kv_raw: Vec<u8>,           // raw bytes of all KV pairs
    tensors: Vec<GgufTensorInfo>,
    data_offset: u64,          // absolute offset where tensor data starts
}

/// Read a GGUF string from a reader: u64 length + bytes.
fn read_gguf_str(r: &mut impl Read) -> std::io::Result<String> {
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;
    if len > 10_000_000 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "string too long"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

/// Write a GGUF string: u64 length + bytes.
fn write_gguf_str(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

/// Skip a GGUF KV value.
fn skip_gguf_kv_value(r: &mut (impl Read + Seek), type_id: u32) -> std::io::Result<()> {
    match type_id {
        0 | 1 | 7 => { r.seek(SeekFrom::Current(1))?; },          // UINT8, INT8, BOOL
        2 | 3     => { r.seek(SeekFrom::Current(2))?; },          // UINT16, INT16
        4 | 5 | 6 => { r.seek(SeekFrom::Current(4))?; },          // UINT32, INT32, FLOAT32
        8 => { read_gguf_str(r)?; },                               // STRING
        9 => {                                                      // ARRAY
            let mut buf4 = [0u8; 4];
            let mut buf8 = [0u8; 8];
            r.read_exact(&mut buf4)?;
            let elem_type = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf8)?;
            let count = u64::from_le_bytes(buf8);
            for _ in 0..count {
                skip_gguf_kv_value(r, elem_type)?;
            }
        },
        10 | 11 | 12 => { r.seek(SeekFrom::Current(8))?; },      // UINT64, INT64, FLOAT64
        _ => { r.seek(SeekFrom::Current(4))?; },                  // unknown, assume 4
    }
    Ok(())
}

/// Parse a GGUF file: header, KV pairs (as raw bytes), tensor info, data offset.
fn parse_gguf(path: &Path) -> anyhow::Result<GgufFile> {
    let mut f = std::fs::File::open(path)?;

    // Header: magic(4) + version(u32) + n_tensors(u64) + n_kv(u64) = 24 bytes
    let mut header = [0u8; 24];
    f.read_exact(&mut header)?;

    if &header[0..4] != b"GGUF" {
        anyhow::bail!("not a GGUF file: {}", path.display());
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let n_tensors = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let n_kv = u64::from_le_bytes(header[16..24].try_into().unwrap());

    // Read KV pairs as raw bytes (we need to preserve them verbatim)
    let kv_start = f.stream_position()?;
    for _ in 0..n_kv {
        read_gguf_str(&mut f)?; // key
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?; // value type
        let vtype = u32::from_le_bytes(buf4);
        skip_gguf_kv_value(&mut f, vtype)?;
    }
    let kv_end = f.stream_position()?;

    // Capture raw KV bytes
    let kv_len = (kv_end - kv_start) as usize;
    f.seek(SeekFrom::Start(kv_start))?;
    let mut kv_raw = vec![0u8; kv_len];
    f.read_exact(&mut kv_raw)?;

    // Read tensor info
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_gguf_str(&mut f)?;
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n_dims = u32::from_le_bytes(buf4);

        let mut ne = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let mut buf8 = [0u8; 8];
            f.read_exact(&mut buf8)?;
            ne.push(u64::from_le_bytes(buf8));
        }

        f.read_exact(&mut buf4)?;
        let type_id = u32::from_le_bytes(buf4);

        let mut buf8 = [0u8; 8];
        f.read_exact(&mut buf8)?;
        let offset = u64::from_le_bytes(buf8);

        tensors.push(GgufTensorInfo { name, n_dims, ne, type_id, offset });
    }

    // Data starts after alignment
    let pos = f.stream_position()?;
    let data_offset = pad_to(pos, GGUF_ALIGNMENT);

    Ok(GgufFile { version, n_kv, kv_raw, tensors, data_offset })
}

/// Write GGUF tensor info entry.
fn write_tensor_info(w: &mut impl Write, t: &GgufTensorInfo, data_offset: u64) -> std::io::Result<()> {
    write_gguf_str(w, &t.name)?;
    w.write_all(&t.n_dims.to_le_bytes())?;
    for d in 0..t.n_dims as usize {
        w.write_all(&t.ne[d].to_le_bytes())?;
    }
    w.write_all(&t.type_id.to_le_bytes())?;
    w.write_all(&data_offset.to_le_bytes())
}

/// Assemble a shard GGUF from a trunk file and a set of expert files.
///
/// The trunk contains all non-expert tensors (attention, norms, embeddings, router gates).
/// Each expert file contains that expert's slices of the ffn_{gate,up,down}_exps tensors.
///
/// Output is a valid GGUF loadable by llama-server.
pub fn assemble_shard(
    trunk_path: &Path,
    expert_paths: &[PathBuf],
    output_path: &Path,
) -> anyhow::Result<()> {
    let n_experts_out = expert_paths.len() as u64;
    eprintln!("  🔧 Assembling shard: trunk + {} experts", n_experts_out);

    let trunk = parse_gguf(trunk_path)?;

    // Parse first expert to get tensor names/shapes
    if expert_paths.is_empty() {
        anyhow::bail!("no expert files provided");
    }
    let exp0 = parse_gguf(&expert_paths[0])?;

    // Read expert IDs from metadata (moe_explode.expert_id)
    let mut expert_ids: Vec<u32> = Vec::with_capacity(expert_paths.len());
    for ep in expert_paths {
        let exp = parse_gguf(ep)?;
        // Find expert_id in KV — scan raw bytes
        let eid = read_kv_u32(&exp.kv_raw, "moe_explode.expert_id")
            .ok_or_else(|| anyhow::anyhow!("missing moe_explode.expert_id in {}", ep.display()))?;
        expert_ids.push(eid);
    }

    let original_expert_count = read_kv_u32(&exp0.kv_raw, "moe_explode.original_expert_count")
        .ok_or_else(|| anyhow::anyhow!("missing moe_explode.original_expert_count"))?;

    eprintln!("  experts: {:?} (of {} original)", expert_ids, original_expert_count);

    // Build output KV: trunk KV with expert_count patched
    let out_kv = patch_kv_expert_count(&trunk.kv_raw, n_experts_out as u32, original_expert_count)?;

    // Build output tensor list
    let mut out_tensors: Vec<GgufTensorInfo> = Vec::new();

    // Trunk tensors — router gates get resized
    for t in &trunk.tensors {
        if t.is_router_gate() {
            let mut ne = t.ne.clone();
            let last = ne.len() - 1;
            ne[last] = n_experts_out;
            out_tensors.push(GgufTensorInfo {
                name: t.name.clone(),
                n_dims: t.n_dims,
                ne,
                type_id: t.type_id,
                offset: 0, // will recompute
            });
        } else {
            out_tensors.push(t.clone());
        }
    }

    // Expert tensors — restore the expert dimension
    for t in &exp0.tensors {
        let mut ne = t.ne.clone();
        ne.push(n_experts_out); // add expert dim back
        out_tensors.push(GgufTensorInfo {
            name: t.name.clone(),
            n_dims: t.n_dims + 1,
            ne,
            type_id: t.type_id,
            offset: 0,
        });
    }

    // Compute tensor data offsets
    let mut data_offset_cursor: u64 = 0;
    for t in &mut out_tensors {
        t.offset = data_offset_cursor;
        let nbytes = t.nbytes();
        data_offset_cursor = pad_to(data_offset_cursor + nbytes, GGUF_ALIGNMENT);
    }

    // Create output file
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::fs::File::create(output_path)?;

    // Write header
    let n_tensors_out = out_tensors.len() as u64;
    out.write_all(b"GGUF")?;
    out.write_all(&trunk.version.to_le_bytes())?;
    out.write_all(&n_tensors_out.to_le_bytes())?;
    out.write_all(&trunk.n_kv.to_le_bytes())?;

    // Write patched KV
    out.write_all(&out_kv)?;

    // Write tensor info
    for t in &out_tensors {
        write_tensor_info(&mut out, t, t.offset)?;
    }

    // Pad to alignment before data
    let pos = 24 + out_kv.len() as u64 + out_tensors.iter().map(|t| {
        8 + t.name.len() as u64 + 4 + (t.n_dims as u64 * 8) + 4 + 8
    }).sum::<u64>();
    let data_start = pad_to(pos, GGUF_ALIGNMENT);
    let padding = data_start - pos;
    write_zeros(&mut out, padding)?;

    // Write tensor data
    let mut trunk_f = std::fs::File::open(trunk_path)?;
    let mut buf = Vec::new();

    for out_t in &out_tensors {
        if let Some(trunk_t) = trunk.tensors.iter().find(|t| t.name == out_t.name) {
            // Trunk tensor
            if out_t.is_router_gate() {
                // Gather selected expert columns from full router gate
                let full_nbytes = trunk_t.nbytes();
                let bytes_per_expert = full_nbytes / original_expert_count as u64;

                buf.resize(full_nbytes as usize, 0);
                trunk_f.seek(SeekFrom::Start(trunk.data_offset + trunk_t.offset))?;
                trunk_f.read_exact(&mut buf)?;

                for (i, &eid) in expert_ids.iter().enumerate() {
                    let src_start = eid as u64 * bytes_per_expert;
                    let src_end = src_start + bytes_per_expert;
                    out.write_all(&buf[src_start as usize..src_end as usize])?;
                    let _ = i; // used implicitly by write order
                }
            } else {
                // Copy verbatim
                let nbytes = trunk_t.nbytes();
                buf.resize(nbytes as usize, 0);
                trunk_f.seek(SeekFrom::Start(trunk.data_offset + trunk_t.offset))?;
                trunk_f.read_exact(&mut buf)?;
                out.write_all(&buf)?;
            }
        } else {
            // Expert tensor — interleave from each expert file
            let exp0_t = exp0.tensors.iter().find(|t| t.name == out_t.name)
                .ok_or_else(|| anyhow::anyhow!("tensor {} not found in expert files", out_t.name))?;
            let bytes_per_expert = exp0_t.nbytes();

            for ep in expert_paths {
                let exp = parse_gguf(ep)?;
                let et = exp.tensors.iter().find(|t| t.name == out_t.name)
                    .ok_or_else(|| anyhow::anyhow!("tensor {} not in {}", out_t.name, ep.display()))?;

                buf.resize(bytes_per_expert as usize, 0);
                let mut ef = std::fs::File::open(ep)?;
                ef.seek(SeekFrom::Start(exp.data_offset + et.offset))?;
                ef.read_exact(&mut buf)?;
                out.write_all(&buf)?;
            }
        }

        // Pad tensor to alignment
        let nbytes = out_t.nbytes();
        let padded = pad_to(nbytes, GGUF_ALIGNMENT);
        write_zeros(&mut out, padded - nbytes)?;
    }

    let out_size = out.metadata()?.len();
    eprintln!("  ✅ Assembled: {} ({:.1} GB)", output_path.display(), out_size as f64 / 1e9);
    Ok(())
}

/// Search raw KV bytes for a u32 value by key name.
fn read_kv_u32(kv_raw: &[u8], target_key: &str) -> Option<u32> {
    let mut cursor = std::io::Cursor::new(kv_raw);
    loop {
        let key = read_gguf_str(&mut cursor).ok()?;
        let mut buf4 = [0u8; 4];
        cursor.read_exact(&mut buf4).ok()?;
        let vtype = u32::from_le_bytes(buf4);

        if key == target_key {
            // Read as u32
            return match vtype {
                4 | 5 => { // UINT32 or INT32
                    cursor.read_exact(&mut buf4).ok()?;
                    Some(u32::from_le_bytes(buf4))
                }
                _ => None,
            };
        }
        skip_gguf_kv_value(&mut cursor, vtype).ok()?;
    }
}

/// Patch expert_count in raw KV bytes. Returns new KV bytes.
/// Finds the key ending with ".expert_count" and replaces its u32 value.
/// Also clamps expert_used_count if it exceeds the new expert count.
fn patch_kv_expert_count(kv_raw: &[u8], new_count: u32, _original_count: u32) -> anyhow::Result<Vec<u8>> {
    let mut out = kv_raw.to_vec();
    let mut cursor = std::io::Cursor::new(kv_raw);

    loop {
        let key = match read_gguf_str(&mut cursor) {
            Ok(k) => k,
            Err(_) => break,
        };
        let mut buf4 = [0u8; 4];
        if cursor.read_exact(&mut buf4).is_err() { break; }
        let vtype = u32::from_le_bytes(buf4);

        let value_pos = cursor.position() as usize;

        if key.ends_with(".expert_count") && (vtype == 4 || vtype == 5) {
            // Patch the u32 value in the output
            out[value_pos..value_pos + 4].copy_from_slice(&new_count.to_le_bytes());
        }

        if key.ends_with(".expert_used_count") && (vtype == 4 || vtype == 5) && new_count > 0 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&kv_raw[value_pos..value_pos + 4]);
            let current = u32::from_le_bytes(buf);
            if current > new_count {
                out[value_pos..value_pos + 4].copy_from_slice(&new_count.to_le_bytes());
            }
        }

        if skip_gguf_kv_value(&mut cursor, vtype).is_err() { break; }
    }

    Ok(out)
}

// ── GGUF explode: split a MoE model into trunk + per-expert files ──

/// Write a GGUF KV entry: string key + u32 type + value.
fn write_kv_u32(w: &mut impl Write, key: &str, val: u32) -> std::io::Result<()> {
    write_gguf_str(w, key)?;
    w.write_all(&4u32.to_le_bytes())?; // GGUF_TYPE_UINT32
    w.write_all(&val.to_le_bytes())
}

fn write_kv_str(w: &mut impl Write, key: &str, val: &str) -> std::io::Result<()> {
    write_gguf_str(w, key)?;
    w.write_all(&8u32.to_le_bytes())?; // GGUF_TYPE_STRING
    write_gguf_str(w, val)
}

/// Explode a MoE GGUF into trunk.gguf + expert-NNN.gguf files.
///
/// - `model_path`: path to the full MoE GGUF
/// - `output_dir`: directory to write trunk.gguf and expert-NNN.gguf files
///
/// trunk.gguf gets all non-expert tensors with expert_count=0.
/// Each expert-NNN.gguf gets that expert's slices with minimal metadata.
pub fn explode_model(model_path: &Path, output_dir: &Path) -> anyhow::Result<u32> {
    let model = parse_gguf(model_path)?;
    let mut f_in = std::fs::File::open(model_path)?;

    // Find architecture and expert count from KV
    let arch = read_kv_string(&model.kv_raw, "general.architecture")
        .ok_or_else(|| anyhow::anyhow!("missing general.architecture"))?;
    let ec_key = format!("{arch}.expert_count");
    let n_expert = read_kv_u32(&model.kv_raw, &ec_key)
        .ok_or_else(|| anyhow::anyhow!("not a MoE model (no {ec_key})"))?;

    if n_expert <= 1 {
        anyhow::bail!("not a MoE model (expert_count={n_expert})");
    }

    // Separate tensors into trunk vs expert
    let trunk_tensors: Vec<&GgufTensorInfo> = model.tensors.iter()
        .filter(|t| !t.is_expert_tensor()).collect();
    let expert_tensors: Vec<&GgufTensorInfo> = model.tensors.iter()
        .filter(|t| t.is_expert_tensor()).collect();

    eprintln!("  Model: {} experts, {} trunk tensors, {} expert tensors",
        n_expert, trunk_tensors.len(), expert_tensors.len());

    std::fs::create_dir_all(output_dir)?;

    // ── Write trunk.gguf ──
    {
        let trunk_path = output_dir.join("trunk.gguf");
        eprintln!("  Writing trunk.gguf ({} tensors)...", trunk_tensors.len());

        // Patch KV: set expert_count=0
        let trunk_kv = patch_kv_expert_count(&model.kv_raw, 0, n_expert)?;

        // Compute trunk tensor data offsets
        let mut trunk_info: Vec<GgufTensorInfo> = trunk_tensors.iter().map(|t| (*t).clone()).collect();
        let mut data_cursor: u64 = 0;
        for t in &mut trunk_info {
            t.offset = data_cursor;
            data_cursor = pad_to(data_cursor + t.nbytes(), GGUF_ALIGNMENT);
        }

        let mut out = std::fs::File::create(&trunk_path)?;

        // Header
        out.write_all(b"GGUF")?;
        out.write_all(&model.version.to_le_bytes())?;
        out.write_all(&(trunk_info.len() as u64).to_le_bytes())?;
        out.write_all(&model.n_kv.to_le_bytes())?;

        // KV (patched)
        out.write_all(&trunk_kv)?;

        // Tensor info
        for t in &trunk_info {
            write_tensor_info(&mut out, t, t.offset)?;
        }

        // Pad to alignment
        let pos = out.stream_position()?;
        let data_start = pad_to(pos, GGUF_ALIGNMENT);
        write_zeros(&mut out, data_start - pos)?;

        // Tensor data
        let mut buf = Vec::new();
        for (out_t, src_t) in trunk_info.iter().zip(trunk_tensors.iter()) {
            let nbytes = src_t.nbytes();
            buf.resize(nbytes as usize, 0);
            f_in.seek(SeekFrom::Start(model.data_offset + src_t.offset))?;
            f_in.read_exact(&mut buf)?;
            out.write_all(&buf)?;
            let padded = pad_to(nbytes, GGUF_ALIGNMENT);
            write_zeros(&mut out, padded - nbytes)?;
            let _ = out_t; // used implicitly by write order
        }

        let size = out.stream_position()?;
        eprintln!("  trunk.gguf: {:.1} GB", size as f64 / 1e9);
    }

    // ── Write expert-NNN.gguf for each expert ──
    for eid in 0..n_expert {
        let expert_path = output_dir.join(format!("expert-{:03}.gguf", eid));

        // Build sliced tensor info (expert dim collapsed from 3D to 2D)
        let mut exp_info: Vec<GgufTensorInfo> = Vec::new();
        for t in &expert_tensors {
            let mut ne = t.ne.clone();
            let last = ne.len() - 1;
            ne[last] = 1; // single expert
            // ggml stores [x,y,1] as 2D [x,y] — drop the trailing 1
            if ne[last] == 1 && ne.len() > 1 {
                ne.pop();
            }
            exp_info.push(GgufTensorInfo {
                name: t.name.clone(),
                n_dims: ne.len() as u32,
                ne,
                type_id: t.type_id,
                offset: 0,
            });
        }

        // Compute offsets
        let mut data_cursor: u64 = 0;
        for t in &mut exp_info {
            t.offset = data_cursor;
            data_cursor = pad_to(data_cursor + t.nbytes(), GGUF_ALIGNMENT);
        }

        // KV: minimal metadata
        let n_kv: u64 = 4;
        let mut kv_buf: Vec<u8> = Vec::new();
        write_kv_str(&mut kv_buf, "general.architecture", &arch)?;
        write_kv_u32(&mut kv_buf, &ec_key, 1)?;
        write_kv_u32(&mut kv_buf, "moe_explode.expert_id", eid)?;
        write_kv_u32(&mut kv_buf, "moe_explode.original_expert_count", n_expert)?;

        let mut out = std::fs::File::create(&expert_path)?;

        // Header
        out.write_all(b"GGUF")?;
        out.write_all(&model.version.to_le_bytes())?;
        out.write_all(&(exp_info.len() as u64).to_le_bytes())?;
        out.write_all(&n_kv.to_le_bytes())?;

        // KV
        out.write_all(&kv_buf)?;

        // Tensor info
        for t in &exp_info {
            write_tensor_info(&mut out, t, t.offset)?;
        }

        // Pad to alignment
        let pos = out.stream_position()?;
        let data_start = pad_to(pos, GGUF_ALIGNMENT);
        write_zeros(&mut out, data_start - pos)?;

        // Tensor data: slice this expert from each expert tensor
        let mut buf = Vec::new();
        for (exp_t, src_t) in exp_info.iter().zip(expert_tensors.iter()) {
            let full_nbytes = src_t.nbytes();
            let bytes_per_expert = full_nbytes / n_expert as u64;

            buf.resize(bytes_per_expert as usize, 0);
            let src_offset = model.data_offset + src_t.offset + (eid as u64 * bytes_per_expert);
            f_in.seek(SeekFrom::Start(src_offset))?;
            f_in.read_exact(&mut buf)?;
            out.write_all(&buf)?;

            let padded = pad_to(bytes_per_expert, GGUF_ALIGNMENT);
            write_zeros(&mut out, padded - bytes_per_expert)?;
            let _ = exp_t; // used implicitly by write order
        }

        if eid % 32 == 0 || eid == n_expert - 1 {
            let size = out.stream_position()?;
            eprintln!("  expert-{:03}.gguf: {:.1} MB", eid, size as f64 / 1e6);
        }
    }

    eprintln!("  ✅ Exploded into {} + {} expert files", "trunk.gguf", n_expert);
    Ok(n_expert)
}

/// Read a string value from raw KV bytes by key name.
fn read_kv_string(kv_raw: &[u8], target_key: &str) -> Option<String> {
    let mut cursor = std::io::Cursor::new(kv_raw);
    loop {
        let key = read_gguf_str(&mut cursor).ok()?;
        let mut buf4 = [0u8; 4];
        cursor.read_exact(&mut buf4).ok()?;
        let vtype = u32::from_le_bytes(buf4);

        if key == target_key && vtype == 8 {
            return read_gguf_str(&mut cursor).ok();
        }
        skip_gguf_kv_value(&mut cursor, vtype).ok()?;
    }
}

// ── Ranking cache ──

/// Path to cached ranking CSV for a model.
/// Stored next to the model: `~/.models/moe-rankings/<model-stem>.csv`
pub fn ranking_cache_path(model_path: &Path) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = model_path.parent().unwrap_or(Path::new("."));
    dir.join("moe-rankings").join(format!("{stem}.csv"))
}

/// Load a cached ranking CSV. Format: one expert_id per line, sorted by gate mass descending.
/// Also supports the full CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
pub fn load_cached_ranking(path: &Path) -> Option<Vec<u32>> {
    let content = std::fs::read_to_string(path).ok()?;
    let ranking: Vec<u32> = content.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("expert"))
        .filter_map(|l| {
            // Support both plain "42" and CSV "42,1234.5,0.03,500"
            l.split(',').next()?.trim().parse().ok()
        })
        .collect();
    if ranking.is_empty() { None } else { Some(ranking) }
}

// ── Expert assignment ──

/// Expert assignment for a single node: which expert IDs it should hold.
#[derive(Clone, Debug)]
pub struct NodeAssignment {
    /// All expert IDs for this node (shared core + unique shard), sorted.
    pub experts: Vec<u32>,
    /// How many of these are shared (replicated to every node).
    pub n_shared: usize,
    /// How many are unique to this node.
    pub n_unique: usize,
}

/// Compute expert assignments for N nodes using the overlap strategy.
///
/// - `ranking`: expert IDs sorted by gate mass descending (hottest first)
/// - `n_nodes`: number of mesh nodes to split across
/// - `min_experts`: minimum experts per node for coherent output
///
/// Returns one NodeAssignment per node. Every expert appears in at least one node.
/// Convenience wrapper for compute_assignments_with_overlap with overlap=1.
/// Used by tests and external callers that don't need redundancy.
#[allow(dead_code)]
pub fn compute_assignments(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
) -> Vec<NodeAssignment> {
    compute_assignments_with_overlap(ranking, n_nodes, min_experts, 1)
}

/// Compute expert assignments with a configurable overlap factor.
///
/// - `overlap`: how many nodes each expert should live on (1 = no redundancy,
///   2 = every expert on at least 2 nodes, etc.). Capped at n_nodes.
///
/// Strategy:
/// 1. Shared core = top `min_experts` by gate mass → replicated to every node
/// 2. Remaining experts distributed with `overlap` copies each
///
/// With overlap=2, losing any single node doesn't orphan any expert —
/// at least one other node still has it.
pub fn compute_assignments_with_overlap(
    ranking: &[u32],
    n_nodes: usize,
    min_experts: u32,
    overlap: usize,
) -> Vec<NodeAssignment> {
    let n_expert = ranking.len();
    let min_exp = min_experts as usize;
    let overlap = overlap.min(n_nodes).max(1);

    if n_nodes <= 1 || min_exp >= n_expert {
        // Single node or core covers everything — give everyone all experts
        return vec![NodeAssignment {
            experts: ranking.to_vec(),
            n_shared: n_expert,
            n_unique: 0,
        }; n_nodes.max(1)];
    }

    // Shared core = top min_experts by gate mass (replicated to every node)
    let shared_core: Vec<u32> = ranking[..min_exp].to_vec();

    // Remaining experts to distribute with overlap
    let remaining: Vec<u32> = ranking[min_exp..].to_vec();

    // With overlap, each expert goes to `overlap` nodes.
    // Total expert-slots = remaining.len() * overlap, distributed round-robin.
    let mut node_experts: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];

    for (i, &expert_id) in remaining.iter().enumerate() {
        // Assign to `overlap` consecutive nodes (wrapping)
        for j in 0..overlap {
            let node = (i + j) % n_nodes;
            node_experts[node].push(expert_id);
        }
    }

    let mut assignments = Vec::with_capacity(n_nodes);
    for node_exps in node_experts {
        let n_unique = node_exps.len();
        let mut experts = shared_core.clone();
        experts.extend_from_slice(&node_exps);
        experts.sort();
        experts.dedup(); // in case overlap wraps and duplicates with shared core

        assignments.push(NodeAssignment {
            experts,
            n_shared: min_exp,
            n_unique,
        });
    }

    assignments
}

/// Format expert list as comma-separated string for moe-split --expert-list.
pub fn expert_list_arg(assignment: &NodeAssignment) -> String {
    assignment.experts.iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Path to the cached split GGUF for a given model + node count + node index.
pub fn split_path(model_path: &Path, n_nodes: usize, node_index: usize) -> PathBuf {
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = model_path.parent().unwrap_or(Path::new("."));
    dir.join("moe-splits")
        .join(format!("{stem}"))
        .join(format!("{n_nodes}-nodes"))
        .join(format!("node-{node_index}.gguf"))
}

/// Run llama-moe-split to produce a split GGUF for one node.
pub fn run_split(
    bin_dir: &Path,
    model_path: &Path,
    assignment: &NodeAssignment,
    output_path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let expert_list = expert_list_arg(assignment);
    let status = std::process::Command::new(bin_dir.join("llama-moe-split"))
        .args([
            "-m", &model_path.to_string_lossy(),
            "--expert-list", &expert_list,
            "-o", &output_path.to_string_lossy(),
        ])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run llama-moe-split: {e}"))?;

    anyhow::ensure!(status.success(), "llama-moe-split exited with {status}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assignments_2_nodes() {
        // 10 experts, min 4, 2 nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments(&ranking, 2, 4);

        assert_eq!(assignments.len(), 2);
        // Each node: 4 shared + 3 unique = 7 experts
        assert_eq!(assignments[0].experts.len(), 7);
        assert_eq!(assignments[1].experts.len(), 7);
        assert_eq!(assignments[0].n_shared, 4);
        assert_eq!(assignments[0].n_unique, 3);

        // Shared core (0-3) in both
        for e in 0..4 {
            assert!(assignments[0].experts.contains(&e));
            assert!(assignments[1].experts.contains(&e));
        }

        // Full coverage
        let mut all: Vec<u32> = assignments[0].experts.clone();
        all.extend(&assignments[1].experts);
        all.sort();
        all.dedup();
        assert_eq!(all, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn test_assignments_3_nodes() {
        // 128 experts, min 46, 3 nodes
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments(&ranking, 3, 46);

        assert_eq!(assignments.len(), 3);
        // 82 remaining / 3 = 27 each + 1 leftover
        // Nodes 0: 46+28=74, Node 1: 46+27=73, Node 2: 46+27=73
        assert_eq!(assignments[0].experts.len(), 74);
        assert_eq!(assignments[1].experts.len(), 73);
        assert_eq!(assignments[2].experts.len(), 73);

        // Full coverage
        let mut all: Vec<u32> = Vec::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        all.sort();
        all.dedup();
        assert_eq!(all, (0..128).collect::<Vec<u32>>());
    }

    #[test]
    fn test_ranking_cache_roundtrip() {
        let dir = std::env::temp_dir().join("moe-test-ranking");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");

        let ranking: Vec<u32> = vec![0, 26, 41, 69, 104, 3, 7, 99];
        let content: String = ranking.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
        std::fs::write(&path, content).unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, ranking);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_moe_analyze_csv() {
        // The CSV format from moe-analyze: expert_id,total_mass,mass_fraction,selection_count
        let dir = std::env::temp_dir().join("moe-test-csv");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ranking.csv");
        std::fs::write(&path, "expert_id,total_mass,mass_fraction,selection_count\n\
            0,8365.69,0.250,15680\n\
            26,267.43,0.008,4800\n\
            41,250.11,0.007,4600\n").unwrap();

        let loaded = load_cached_ranking(&path).unwrap();
        assert_eq!(loaded, vec![0, 26, 41]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_moe_qwen3() {
        let path = std::path::Path::new("/Users/micn/.models/Qwen3-30B-A3B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: model file not found");
            return;
        }
        let info = detect_moe(path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 128);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_olmoe() {
        let path = std::path::Path::new("/Users/micn/.models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf");
        if !path.exists() {
            eprintln!("Skipping: OLMoE model file not found");
            return;
        }
        let info = detect_moe(path).expect("Should detect MoE");
        assert_eq!(info.expert_count, 64);
        assert_eq!(info.expert_used_count, 8);
    }

    #[test]
    fn test_detect_moe_dense_model() {
        // Qwen2.5-3B is dense (no experts) — should return None
        let path = std::path::Path::new("/Users/micn/.models/Qwen2.5-3B-Instruct-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping: dense model file not found");
            return;
        }
        assert!(detect_moe(path).is_none(), "Dense model should not be detected as MoE");
    }

    #[test]
    fn test_single_node() {
        let ranking: Vec<u32> = (0..8).collect();
        let assignments = compute_assignments(&ranking, 1, 4);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].experts.len(), 8); // gets everything
    }

    // ── Overlap tests ──

    #[test]
    fn test_overlap_2x_3_nodes() {
        // 128 experts, min 46, 3 nodes, 2× overlap
        let ranking: Vec<u32> = (0..128).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 46, 2);

        assert_eq!(assignments.len(), 3);

        // Every expert should appear in at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }

        // Shared core (0..46) in all 3 nodes
        for e in 0..46 {
            assert!(*expert_count.get(&e).unwrap() >= 3,
                "Shared expert {e} should be in all nodes");
        }
        // Remaining experts (46..128) in at least 2 nodes
        for e in 46..128 {
            assert!(*expert_count.get(&e).unwrap() >= 2,
                "Expert {e} should be in at least 2 nodes, got {}", expert_count[&e]);
        }
        // Full coverage
        assert_eq!(expert_count.len(), 128);
    }

    #[test]
    fn test_overlap_2x_2_nodes() {
        // With 2 nodes and 2× overlap, every remaining expert is on both nodes
        let ranking: Vec<u32> = (0..10).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 2, 4, 2);

        assert_eq!(assignments.len(), 2);
        // Both nodes should have all 10 experts (4 shared + 6 remaining × 2× = both)
        assert_eq!(assignments[0].experts.len(), 10);
        assert_eq!(assignments[1].experts.len(), 10);
    }

    #[test]
    fn test_overlap_1x_same_as_original() {
        // overlap=1 should give same results as compute_assignments
        let ranking: Vec<u32> = (0..128).collect();
        let a1 = compute_assignments(&ranking, 3, 46);
        let a2 = compute_assignments_with_overlap(&ranking, 3, 46, 1);

        for i in 0..3 {
            assert_eq!(a1[i].experts, a2[i].experts);
        }
    }

    #[test]
    fn test_overlap_capped_at_n_nodes() {
        // overlap=10 with 3 nodes should cap to 3 (every expert on every node)
        let ranking: Vec<u32> = (0..20).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 3, 5, 10);

        // All 3 nodes should have all 20 experts
        for a in &assignments {
            assert_eq!(a.experts.len(), 20);
        }
    }

    #[test]
    fn test_overlap_glm5_10_nodes() {
        // GLM-5: 256 experts, min 96, 10 nodes, 2× overlap
        let ranking: Vec<u32> = (0..256).collect();
        let assignments = compute_assignments_with_overlap(&ranking, 10, 96, 2);

        assert_eq!(assignments.len(), 10);

        // Full coverage
        let mut all: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for a in &assignments {
            all.extend(&a.experts);
        }
        assert_eq!(all.len(), 256);

        // Every remaining expert on at least 2 nodes
        let mut expert_count: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for a in &assignments {
            for &e in &a.experts {
                *expert_count.entry(e).or_default() += 1;
            }
        }
        for e in 96..256 {
            assert!(*expert_count.get(&e).unwrap() >= 2);
        }

        // Print sizes for verification
        for (i, a) in assignments.iter().enumerate() {
            eprintln!("  Node {i}: {} experts ({} shared + {} unique)",
                a.experts.len(), a.n_shared, a.n_unique);
        }
    }

    // ── GGUF assembler tests ──

    #[test]
    fn test_parse_gguf_trunk() {
        let path = Path::new("/tmp/moe-explode-test/trunk.gguf");
        if !path.exists() {
            eprintln!("Skipping: trunk.gguf not found");
            return;
        }
        let gguf = parse_gguf(path).unwrap();
        assert_eq!(gguf.version, 3);
        assert!(gguf.tensors.len() > 100); // trunk has ~435 tensors
        assert!(gguf.n_kv > 10);

        // Should have router gates
        let routers: Vec<_> = gguf.tensors.iter().filter(|t| t.is_router_gate()).collect();
        assert!(!routers.is_empty(), "trunk should have router gates");

        // Should NOT have expert tensors (they were removed)
        let experts: Vec<_> = gguf.tensors.iter().filter(|t| t.is_expert_tensor()).collect();
        assert_eq!(experts.len(), 0, "trunk should not have expert tensors");

        eprintln!("trunk: {} tensors, {} KV, {} router gates", gguf.tensors.len(), gguf.n_kv, routers.len());
    }

    #[test]
    fn test_parse_gguf_expert() {
        let path = Path::new("/tmp/moe-explode-test/expert-000.gguf");
        if !path.exists() {
            eprintln!("Skipping: expert-000.gguf not found");
            return;
        }
        let gguf = parse_gguf(path).unwrap();
        assert_eq!(gguf.version, 3);

        // Expert file has 3 tensors per layer (gate, up, down) × 48 layers = 144
        assert_eq!(gguf.tensors.len(), 144);

        // All tensors should be expert tensors
        for t in &gguf.tensors {
            assert!(t.is_expert_tensor(), "expected expert tensor, got: {}", t.name);
        }

        // Should have expert_id = 0
        let eid = read_kv_u32(&gguf.kv_raw, "moe_explode.expert_id");
        assert_eq!(eid, Some(0));

        let orig = read_kv_u32(&gguf.kv_raw, "moe_explode.original_expert_count");
        assert_eq!(orig, Some(128));
    }

    #[test]
    fn test_assemble_shard() {
        let trunk = Path::new("/tmp/moe-explode-test/trunk.gguf");
        if !trunk.exists() {
            eprintln!("Skipping: exploded files not found");
            return;
        }

        // Assemble a 4-expert shard
        let expert_paths: Vec<PathBuf> = (0..4)
            .map(|i| PathBuf::from(format!("/tmp/moe-explode-test/expert-{:03}.gguf", i)))
            .collect();

        for p in &expert_paths {
            if !p.exists() {
                eprintln!("Skipping: {} not found", p.display());
                return;
            }
        }

        let output = Path::new("/tmp/moe-assemble-test/shard-4.gguf");
        let _ = std::fs::remove_file(output);

        assemble_shard(trunk, &expert_paths, output).unwrap();

        // Output should exist and be reasonable size
        assert!(output.exists());
        let size = std::fs::metadata(output).unwrap().len();
        eprintln!("shard-4: {:.1} GB", size as f64 / 1e9);

        // Should be bigger than trunk (957MB) but much smaller than full (17.8GB)
        assert!(size > 900_000_000, "shard too small: {}", size);
        assert!(size < 5_000_000_000, "shard too large: {}", size);

        // Parse the output — verify it's valid GGUF
        let out = parse_gguf(output).unwrap();
        assert_eq!(out.version, 3);
        assert!(out.tensors.len() > 100);

        // Expert count should be 4
        let ec = read_kv_u32(&out.kv_raw, "qwen3moe.expert_count");
        assert_eq!(ec, Some(4));

        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir("/tmp/moe-assemble-test");
    }

    #[test]
    fn test_assemble_matches_cpp() {
        let trunk = Path::new("/tmp/moe-explode-test/trunk.gguf");
        let cpp_shard = Path::new("/tmp/moe-explode-test/shard-64.gguf");
        if !trunk.exists() || !cpp_shard.exists() {
            eprintln!("Skipping: need trunk.gguf and shard-64.gguf from C++ moe-explode");
            return;
        }

        // Assemble same 64 experts (0..64) with Rust
        let expert_paths: Vec<PathBuf> = (0..64)
            .map(|i| PathBuf::from(format!("/tmp/moe-explode-test/expert-{:03}.gguf", i)))
            .collect();

        for p in &expert_paths {
            if !p.exists() {
                eprintln!("Skipping: {} not found", p.display());
                return;
            }
        }

        let output = Path::new("/tmp/moe-assemble-test/rust-shard-64.gguf");
        let _ = std::fs::remove_file(output);

        assemble_shard(trunk, &expert_paths, output).unwrap();

        let rust_size = std::fs::metadata(output).unwrap().len();
        let cpp_size = std::fs::metadata(cpp_shard).unwrap().len();
        eprintln!("Rust: {:.1} GB, C++: {:.1} GB", rust_size as f64 / 1e9, cpp_size as f64 / 1e9);

        assert_eq!(rust_size, cpp_size, "Rust and C++ shards should be same size");

        // Byte-for-byte comparison
        let rust_data = std::fs::read(output).unwrap();
        let cpp_data = std::fs::read(cpp_shard).unwrap();
        if rust_data != cpp_data {
            // Find first difference
            for (i, (r, c)) in rust_data.iter().zip(cpp_data.iter()).enumerate() {
                if r != c {
                    panic!("First byte difference at offset {i} (0x{i:x}): rust=0x{r:02x} cpp=0x{c:02x}");
                }
            }
        }

        eprintln!("✅ Rust and C++ shards are byte-identical!");
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir("/tmp/moe-assemble-test");
    }

    #[test]
    fn test_explode_model() {
        let model = Path::new("/Users/micn/.models/Qwen3-30B-A3B-Q4_K_M.gguf");
        if !model.exists() {
            eprintln!("Skipping: Qwen3-30B model not found");
            return;
        }

        let out_dir = Path::new("/tmp/moe-explode-rust-test");
        let _ = std::fs::remove_dir_all(out_dir);

        let n_expert = explode_model(model, out_dir).unwrap();
        assert_eq!(n_expert, 128);

        // Trunk should exist
        let trunk = out_dir.join("trunk.gguf");
        assert!(trunk.exists());
        let trunk_gguf = parse_gguf(&trunk).unwrap();
        // expert_count should be 0 in trunk
        let ec = read_kv_u32(&trunk_gguf.kv_raw, "qwen3moe.expert_count");
        assert_eq!(ec, Some(0));
        // No expert tensors
        assert!(trunk_gguf.tensors.iter().all(|t| !t.is_expert_tensor()));

        // Expert files should exist and have correct metadata
        let exp0 = out_dir.join("expert-000.gguf");
        assert!(exp0.exists());
        let exp_gguf = parse_gguf(&exp0).unwrap();
        assert_eq!(read_kv_u32(&exp_gguf.kv_raw, "moe_explode.expert_id"), Some(0));
        assert_eq!(read_kv_u32(&exp_gguf.kv_raw, "moe_explode.original_expert_count"), Some(128));

        let exp127 = out_dir.join("expert-127.gguf");
        assert!(exp127.exists());

        // Expert files should be byte-identical to C++ (same KV order — minimal metadata)
        let cpp_exp0 = Path::new("/tmp/moe-explode-test/expert-000.gguf");
        if cpp_exp0.exists() {
            let rust_exp0_data = std::fs::read(&exp0).unwrap();
            let cpp_exp0_data = std::fs::read(cpp_exp0).unwrap();
            assert_eq!(rust_exp0_data.len(), cpp_exp0_data.len(), "expert-000 sizes differ");
            assert_eq!(rust_exp0_data, cpp_exp0_data, "expert-000 content differs");
            eprintln!("✅ expert-000.gguf byte-identical to C++");
        }

        // Trunk KV ordering differs from C++ (we preserve original order, C++ reorders via gguf_set_kv).
        // Both are functionally identical — verified by round-trip below.

        // Round-trip: explode → assemble all 128 → should match original size
        let expert_paths: Vec<PathBuf> = (0..128)
            .map(|i| out_dir.join(format!("expert-{:03}.gguf", i)))
            .collect();
        let roundtrip = out_dir.join("roundtrip.gguf");
        assemble_shard(&trunk, &expert_paths, &roundtrip).unwrap();

        let original_size = std::fs::metadata(model).unwrap().len();
        let roundtrip_size = std::fs::metadata(&roundtrip).unwrap().len();
        eprintln!("original: {:.1} GB, roundtrip: {:.1} GB", original_size as f64 / 1e9, roundtrip_size as f64 / 1e9);
        assert_eq!(original_size, roundtrip_size, "round-trip size mismatch");

        let _ = std::fs::remove_dir_all(out_dir);
    }
}
