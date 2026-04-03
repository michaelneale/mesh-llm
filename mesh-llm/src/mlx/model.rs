//! Qwen2/Llama-style transformer model running on MLX via mlx-rs.
//!
//! Loads quantized safetensors and runs inference entirely on Metal GPU.
//! No Python, no subprocess — just Rust + MLX C library.

use anyhow::{bail, Context, Result};
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::Array;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    #[allow(dead_code)]
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[allow(dead_code)]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization: Option<QuantConfig>,
    /// EOS token ID(s) — can be a single int or array in config.json.
    #[serde(default, deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
}

fn deserialize_eos_token_id<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<u32>, D::Error> {
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum EosId {
        Single(u32),
        Multiple(Vec<u32>),
    }
    Ok(match EosId::deserialize(deserializer)? {
        EosId::Single(id) => vec![id],
        EosId::Multiple(ids) => ids,
    })
}

fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Debug, serde::Deserialize)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

// ── Layer primitives ──

pub struct QuantizedLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
}

impl QuantizedLinear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let out = mlx_rs::ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )?;
        Ok(if let Some(ref bias) = self.bias {
            &out + bias
        } else {
            out
        })
    }
}

pub struct RMSNorm {
    weight: Array,
    eps: f32,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::rms_norm(x, &self.weight, self.eps)?)
    }
}

// ── Attention ──

pub struct Attention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    o_proj: QuantizedLinear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_theta: f32,
}

impl Attention {
    pub fn forward(&self, x: &Array, cache: &mut KVCache) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let q = if let Some(norm) = &self.q_norm {
            norm.forward(&q)?
        } else {
            q
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
        let k = if let Some(norm) = &self.k_norm {
            norm.forward(&k)?
        } else {
            k
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, l, self.num_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.offset as i32;
        let q = mlx_rs::fast::rope(
            &q,
            self.head_dim,
            false,
            Some(self.rope_theta),
            1.0,
            offset,
            None::<&Array>,
        )?;
        let k = mlx_rs::fast::rope(
            &k,
            self.head_dim,
            false,
            Some(self.rope_theta),
            1.0,
            offset,
            None::<&Array>,
        )?;

        let (k, v) = cache.update(k, v)?;

        // Causal mask for prefill (multi-token). Decode (l=1) needs no mask.
        let mask = if l > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };

        let attn = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        let attn =
            attn.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.num_heads * self.head_dim])?;

        self.o_proj.forward(&attn)
    }
}

// ── MLP ──

pub struct MLP {
    gate_proj: QuantizedLinear,
    up_proj: QuantizedLinear,
    down_proj: QuantizedLinear,
}

impl MLP {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let gate = &mlx_rs::ops::sigmoid(&gate)? * &gate; // SiLU
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(&gate * &up))
    }
}

// ── Transformer layer ──

pub struct Layer {
    attn: Attention,
    mlp: MLP,
    attn_norm: RMSNorm,
    mlp_norm: RMSNorm,
}

impl Layer {
    pub fn forward(&self, x: &Array, cache: &mut KVCache) -> Result<Array> {
        let h = &self.attn.forward(&self.attn_norm.forward(x)?, cache)? + x;
        Ok(&self.mlp.forward(&self.mlp_norm.forward(&h)?)? + &h)
    }
}

// ── KV cache ──
//
// Pre-allocated KV cache following mlx-lm's approach:
//   - Allocate in chunks of STEP (256) positions
//   - Use slice assignment (index_mut) to write new KV entries in-place
//   - Return a view [0..offset] to SDPA — no allocations per token
//
// This eliminates the O(n²) concatenation cost that killed prefill performance.

const KV_CACHE_STEP: usize = 256;

pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    pub offset: usize,
}

impl KVCache {
    pub fn new() -> Self {
        KVCache {
            keys: None,
            values: None,
            offset: 0,
        }
    }

    /// Return references to cached arrays (for eval/materialization).
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = Vec::new();
        if let Some(ref k) = self.keys {
            out.push(k);
        }
        if let Some(ref v) = self.values {
            out.push(v);
        }
        out
    }

    pub fn update(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        use std::ops::RangeFull;

        let seq_len = k.shape()[2] as usize;
        let prev = self.offset;

        if self.keys.is_none() || (prev + seq_len) > self.keys.as_ref().unwrap().shape()[2] as usize
        {
            // Grow: pre-allocate in steps, matching the incoming dtype
            let [b, n_kv_heads, _, k_head_dim] = k.shape()[..4] else {
                bail!("unexpected k shape");
            };
            let v_head_dim = v.shape()[3];
            let k_dtype = k.dtype();
            let v_dtype = v.dtype();

            let n_steps = ((KV_CACHE_STEP + seq_len - 1) / KV_CACHE_STEP) * KV_CACHE_STEP;
            let k_shape = &[b, n_kv_heads, n_steps as i32, k_head_dim];
            let v_shape = &[b, n_kv_heads, n_steps as i32, v_head_dim];

            let new_k = mlx_rs::ops::zeros_dtype(k_shape, k_dtype)?;
            let new_v = mlx_rs::ops::zeros_dtype(v_shape, v_dtype)?;

            if let (Some(ref mut old_k), Some(ref mut old_v)) = (&mut self.keys, &mut self.values) {
                if prev % KV_CACHE_STEP != 0 {
                    *old_k = old_k.index((RangeFull, RangeFull, ..(prev as i32), RangeFull));
                    *old_v = old_v.index((RangeFull, RangeFull, ..(prev as i32), RangeFull));
                }
                self.keys = Some(mlx_rs::ops::concatenate_axis(
                    &[old_k as &Array, &new_k],
                    2,
                )?);
                self.values = Some(mlx_rs::ops::concatenate_axis(
                    &[old_v as &Array, &new_v],
                    2,
                )?);
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset = prev + seq_len;
        let prev_i = prev as i32;
        let end_i = self.offset as i32;

        // Slice-assign into pre-allocated buffer (no copy of existing data)
        self.keys
            .as_mut()
            .unwrap()
            .try_index_mut((RangeFull, RangeFull, prev_i..end_i, RangeFull), &k)?;
        self.values
            .as_mut()
            .unwrap()
            .try_index_mut((RangeFull, RangeFull, prev_i..end_i, RangeFull), &v)?;

        // Return views up to current offset
        let k_out = self
            .keys
            .as_ref()
            .unwrap()
            .index((RangeFull, RangeFull, ..end_i, RangeFull));
        let v_out = self
            .values
            .as_ref()
            .unwrap()
            .index((RangeFull, RangeFull, ..end_i, RangeFull));

        Ok((k_out, v_out))
    }

    /// Rewind the cache to `n` tokens. The underlying buffer is kept —
    /// only the offset is moved back so new tokens overwrite the tail.
    pub fn trim_to(&mut self, n: usize) {
        self.offset = n;
    }
}

// ── Quantized embedding ──

pub struct QuantizedEmbedding {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QuantizedEmbedding {
    pub fn forward(&self, indices: &Array) -> Result<Array> {
        let w = self.weight.take_axis(indices, 0)?;
        let s = self.scales.take_axis(indices, 0)?;
        let b = self.biases.take_axis(indices, 0)?;
        Ok(mlx_rs::ops::dequantize(
            &w,
            &s,
            &b,
            self.group_size,
            self.bits,
        )?)
    }

    pub fn as_linear(&self) -> QuantizedLinear {
        QuantizedLinear {
            weight: self.weight.clone(),
            scales: self.scales.clone(),
            biases: self.biases.clone(),
            bias: None,
            group_size: self.group_size,
            bits: self.bits,
        }
    }
}

// ── Full model ──

pub struct MlxModel {
    embed_tokens: QuantizedEmbedding,
    layers: Vec<Layer>,
    norm: RMSNorm,
    lm_head: Option<QuantizedLinear>,
    pub config: ModelConfig,
    pub tokenizer: tokenizers::Tokenizer,
    pub prompt_template: crate::mlx::template::PromptTemplate,
}

impl MlxModel {
    /// Load a quantized MLX model from a directory containing config.json,
    /// tokenizer.json, and model.safetensors.
    pub fn load(dir: &Path) -> Result<Self> {
        let config_text =
            std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let config_json: Value =
            serde_json::from_str(&config_text).context("parsing config.json")?;
        ensure_supported_mlx_model(dir, &config_json)?;
        let config: ModelConfig =
            serde_json::from_str(&config_text).context("parsing config.json")?;

        let qcfg = config
            .quantization
            .as_ref()
            .context("expected quantized model (quantization field in config.json)")?;
        let group_size = qcfg.group_size;
        let bits = qcfg.bits;

        tracing::info!(
            "MLX: loading {} layers, hidden={}, heads={}/{}, quant={}bit/g{}",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            bits,
            group_size,
        );

        let start = std::time::Instant::now();
        let tensors = load_all_safetensors(dir)?;
        tracing::info!(
            "MLX: loaded {} tensors in {:.2}s",
            tensors.len(),
            start.elapsed().as_secs_f64()
        );

        let load_qlinear = |prefix: &str| -> Result<QuantizedLinear> {
            Ok(QuantizedLinear {
                weight: tensors
                    .get(&format!("{prefix}.weight"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.weight"))?,
                scales: tensors
                    .get(&format!("{prefix}.scales"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.scales"))?,
                biases: tensors
                    .get(&format!("{prefix}.biases"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.biases"))?,
                bias: tensors.get(&format!("{prefix}.bias")).cloned(),
                group_size,
                bits,
            })
        };

        let embed_tokens = QuantizedEmbedding {
            weight: tensors
                .get("model.embed_tokens.weight")
                .cloned()
                .context("missing model.embed_tokens.weight")?,
            scales: tensors
                .get("model.embed_tokens.scales")
                .cloned()
                .context("missing model.embed_tokens.scales")?,
            biases: tensors
                .get("model.embed_tokens.biases")
                .cloned()
                .context("missing model.embed_tokens.biases")?,
            group_size,
            bits,
        };

        let norm = RMSNorm {
            weight: tensors
                .get("model.norm.weight")
                .cloned()
                .context("missing model.norm.weight")?,
            eps: config.rms_norm_eps,
        };

        // Qwen3-class configs can declare an explicit head_dim that differs
        // from hidden_size / num_attention_heads.
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / config.num_attention_heads);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                attn: Attention {
                    q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
                    k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
                    v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
                    o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
                    q_norm: tensors
                        .get(&format!("{p}.self_attn.q_norm.weight"))
                        .cloned()
                        .map(|weight| RMSNorm {
                            weight,
                            eps: config.rms_norm_eps,
                        }),
                    k_norm: tensors
                        .get(&format!("{p}.self_attn.k_norm.weight"))
                        .cloned()
                        .map(|weight| RMSNorm {
                            weight,
                            eps: config.rms_norm_eps,
                        }),
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim,
                    scale,
                    rope_theta: config.rope_theta,
                },
                mlp: MLP {
                    gate_proj: load_qlinear(&format!("{p}.mlp.gate_proj"))?,
                    up_proj: load_qlinear(&format!("{p}.mlp.up_proj"))?,
                    down_proj: load_qlinear(&format!("{p}.mlp.down_proj"))?,
                },
                attn_norm: RMSNorm {
                    weight: tensors
                        .get(&format!("{p}.input_layernorm.weight"))
                        .cloned()
                        .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
                    eps: config.rms_norm_eps,
                },
                mlp_norm: RMSNorm {
                    weight: tensors
                        .get(&format!("{p}.post_attention_layernorm.weight"))
                        .cloned()
                        .with_context(|| format!("missing {p}.post_attention_layernorm.weight"))?,
                    eps: config.rms_norm_eps,
                },
            });
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else if tensors.contains_key("lm_head.weight") {
            Some(load_qlinear("lm_head")?)
        } else {
            None
        };

        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        let prompt_template = crate::mlx::template::PromptTemplate::detect(dir, &config_json);

        Ok(MlxModel {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            tokenizer,
            prompt_template,
        })
    }

    /// Run a forward pass. Input shape: [1, seq_len] of u32 token IDs.
    /// Returns logits [1, seq_len, vocab_size].
    pub fn forward(&self, tokens: &Array, caches: &mut [KVCache]) -> Result<Array> {
        let mut h = self.embed_tokens.forward(tokens)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i])?;
        }
        let h = self.norm.forward(&h)?;

        Ok(if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)?
        } else {
            self.embed_tokens.as_linear().forward(&h)?
        })
    }

    pub fn new_caches(&self) -> Vec<KVCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }
}

/// Argmax over the last position's logits. Returns the token ID.
pub fn argmax_last(logits: &Array) -> Result<u32> {
    let shape = logits.shape();
    let flat = if shape.len() == 3 {
        let last_idx = (shape[1] - 1) as i32;
        let idx = Array::from_int(last_idx);
        logits.take_axis(&idx, 1)?.reshape(&[-1])?
    } else {
        logits.reshape(&[-1])?
    };
    let token = mlx_rs::ops::indexing::argmax(&flat, false)?;
    mlx_rs::transforms::eval([&token])?;
    Ok(token.as_slice::<u32>()[0])
}

/// Load all safetensors from a model directory (handles single file and sharded).
fn load_all_safetensors(dir: &Path) -> Result<HashMap<String, Array>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .context("missing weight_map in index")?;
        let mut tensors = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for file in weight_map.values() {
            let file = file.as_str().context("weight_map value not a string")?;
            if seen.insert(file.to_string()) {
                tensors.extend(Array::load_safetensors(dir.join(file))?);
            }
        }
        Ok(tensors)
    } else {
        let st_path = dir.join("model.safetensors");
        if !st_path.exists() {
            bail!("no model.safetensors found in {}", dir.display());
        }
        Ok(Array::load_safetensors(st_path)?)
    }
}

fn normalize_model_dir(path: &Path) -> Option<&Path> {
    if path.is_dir() {
        return Some(path);
    }
    let name = path.file_name()?.to_str()?;
    match name {
        "config.json"
        | "chat_template.jinja"
        | "tokenizer.json"
        | "tokenizer_config.json"
        | "model.safetensors"
        | "model.safetensors.index.json" => path.parent(),
        _ => None,
    }
}

fn has_required_model_files(dir: &Path) -> bool {
    let has_config = dir.join("config.json").exists();
    let has_tokenizer =
        dir.join("tokenizer_config.json").exists() || dir.join("tokenizer.json").exists();
    let has_weights =
        dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists();
    has_config && has_tokenizer && has_weights
}

fn config_supports_mlx(config: &Value) -> bool {
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str());
    let model_type = config.get("model_type").and_then(|value| value.as_str());

    architectures.chain(model_type).any(|name| {
        let name = name.to_ascii_lowercase();
        matches!(
            name.as_str(),
            "llama"
                | "qwen2"
                | "qwen3"
                | "llamaforcausallm"
                | "qwen2forcausallm"
                | "qwen3forcausallm"
        )
    })
}

fn read_model_config(dir: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

fn ensure_supported_mlx_model(dir: &Path, config: &Value) -> Result<()> {
    if config_supports_mlx(config) {
        return Ok(());
    }
    if let Some(architecture) = detect_architecture_from_safetensors_header(dir) {
        tracing::info!(
            "MLX loader: config.json did not identify a supported architecture, but safetensors headers matched {}",
            architecture
        );
        return Ok(());
    }

    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "none".to_string());
    bail!(
        "unsupported MLX model architecture in {} (model_type={}, architectures={}); supported MLX models currently cover Llama/Qwen-style safetensors checkpoints",
        dir.display(),
        model_type,
        architectures,
    )
}

fn detect_architecture_from_safetensors_header(dir: &Path) -> Option<String> {
    let path = if dir.join("model.safetensors").exists() {
        dir.join("model.safetensors")
    } else {
        let text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?;
        let index: Value = serde_json::from_str(&text).ok()?;
        let file = index
            .get("weight_map")
            .and_then(|value| value.as_object())?
            .values()
            .find_map(|value| value.as_str())?;
        dir.join(file)
    };

    let mut file = File::open(path).ok()?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return None;
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).ok()?;
    let json: Value = serde_json::from_slice(&header).ok()?;
    let map = json.as_object()?;

    let keys: Vec<&str> = map
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .map(|key| key.as_str())
        .collect();

    if keys.iter().any(|key| key.starts_with("model.layers."))
        && keys
            .iter()
            .any(|key| key.starts_with("model.embed_tokens."))
        && keys
            .iter()
            .any(|key| key.contains(".self_attn.q_proj.") || key.contains(".self_attn.q_proj"))
    {
        return Some("llama_like".to_string());
    }

    None
}

pub fn mlx_model_dir(path: &Path) -> Option<&Path> {
    let dir = normalize_model_dir(path)?;
    if has_required_model_files(dir) {
        Some(dir)
    } else {
        None
    }
}

/// Check whether a path resolves to a supported MLX safetensors model.
///
/// Prefers explicit config metadata and only falls back to safetensors-header
/// inspection when the config does not identify the architecture.
pub fn is_mlx_model_dir(path: &Path) -> bool {
    let Some(dir) = mlx_model_dir(path) else {
        return false;
    };

    if let Some(config) = read_model_config(dir) {
        if config_supports_mlx(&config) {
            return true;
        }
    }

    detect_architecture_from_safetensors_header(dir).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_model_dir_accepts_directory_and_known_files() {
        let root = std::env::temp_dir().join(format!("mesh-llm-mlx-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), "{}").unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"12345678").unwrap();

        assert_eq!(mlx_model_dir(&root), Some(root.as_path()));
        assert_eq!(
            mlx_model_dir(&root.join("config.json")),
            Some(root.as_path())
        );
        assert_eq!(
            mlx_model_dir(&root.join("model.safetensors")),
            Some(root.as_path())
        );
    }

    #[test]
    fn config_supports_known_mlx_architectures() {
        let qwen: Value = serde_json::json!({
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"]
        });
        let llama: Value = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"]
        });
        let unsupported: Value = serde_json::json!({
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"]
        });

        assert!(config_supports_mlx(&qwen));
        assert!(config_supports_mlx(&llama));
        assert!(!config_supports_mlx(&unsupported));
    }

    #[test]
    fn model_config_honors_explicit_head_dim() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 40960,
            "tie_word_embeddings": false,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": 151645
        }))
        .unwrap();

        let derived = config.hidden_size / config.num_attention_heads;
        assert_eq!(derived, 64);
        assert_eq!(config.head_dim, Some(128));
        assert_eq!(
            config
                .head_dim
                .unwrap_or_else(|| config.hidden_size / config.num_attention_heads),
            128
        );
    }

    #[test]
    fn unsupported_architecture_error_mentions_model_type() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-mlx-unsupported-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let config = serde_json::json!({
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"]
        });

        let err = ensure_supported_mlx_model(&root, &config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported MLX model architecture"));
        assert!(err.contains("model_type=gemma3"));
        assert!(err.contains("Gemma3ForConditionalGeneration"));
    }
}
