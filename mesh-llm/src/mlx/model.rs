//! Qwen2/Llama-style transformer model running on MLX via mlx-rs.
//!
//! Loads quantized safetensors and runs inference entirely on Metal GPU.
//! No Python, no subprocess — just Rust + MLX C library.

use anyhow::{bail, Context, Result};
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    #[allow(dead_code)]
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
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

        let q = q
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, l, self.num_kv_heads, self.head_dim])?
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
        let seq_len = k.shape()[2] as usize;
        if let (Some(prev_k), Some(prev_v)) = (&self.keys, &self.values) {
            let new_k = mlx_rs::ops::concatenate_axis(&[prev_k, &k], 2)?;
            let new_v = mlx_rs::ops::concatenate_axis(&[prev_v, &v], 2)?;
            self.offset += seq_len;
            self.keys = Some(new_k.clone());
            self.values = Some(new_v.clone());
            Ok((new_k, new_v))
        } else {
            self.offset = seq_len;
            self.keys = Some(k.clone());
            self.values = Some(v.clone());
            Ok((k, v))
        }
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
}

impl MlxModel {
    /// Load a quantized MLX model from a directory containing config.json,
    /// tokenizer.json, and model.safetensors.
    pub fn load(dir: &Path) -> Result<Self> {
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?,
        )
        .context("parsing config.json")?;

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

        let head_dim = config.hidden_size / config.num_attention_heads;
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

        Ok(MlxModel {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            tokenizer,
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

/// Check if a directory looks like an MLX safetensors model.
pub fn is_mlx_model_dir(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let has_config = path.join("config.json").exists();
    let has_tokenizer =
        path.join("tokenizer_config.json").exists() || path.join("tokenizer.json").exists();
    let has_weights = path.join("model.safetensors").exists()
        || path.join("model.safetensors.index.json").exists();
    has_config && has_tokenizer && has_weights
}
