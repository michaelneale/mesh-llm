//! Qwen2/Llama-style transformer model running on MLX via mlx-rs.
//!
//! Loads quantized safetensors and runs inference entirely on Metal GPU.
//! No Python, no subprocess — just Rust + MLX C library.

use anyhow::{bail, Context, Result};
use mlx_rs::array;
use mlx_rs::ops::dequantize_device;
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::Array;
use mlx_rs::{Dtype, StreamOrDevice};
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
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    pub vocab_size: i32,
    #[serde(default)]
    #[allow(dead_code)]
    pub vocab_size_per_layer_input: Option<i32>,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[allow(dead_code)]
    pub max_position_embeddings: i32,
    #[serde(default, deserialize_with = "deserialize_nullable_bool")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<i32>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_rope_parameters")]
    pub rope_parameters: Option<HashMap<String, RopeParameters>>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub cache_implementation: Option<String>,
    pub quantization: Option<QuantConfig>,
    /// EOS token ID(s) — can be a single int or array in config.json.
    #[serde(default, deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
}

#[derive(Debug, serde::Deserialize, Clone)]
pub struct RopeParameters {
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

fn deserialize_nullable_bool<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<bool, D::Error> {
    use serde::Deserialize;

    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or(false))
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

fn deserialize_rope_parameters<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<HashMap<String, RopeParameters>>, D::Error> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RopeParametersField {
        PerLayer(HashMap<String, RopeParameters>),
        Flat(RopeParameters),
    }

    Ok(
        match Option::<RopeParametersField>::deserialize(deserializer)? {
            None => None,
            Some(RopeParametersField::PerLayer(map)) => Some(map),
            Some(RopeParametersField::Flat(params)) => {
                let mut map = HashMap::new();
                map.insert("default".to_string(), params);
                Some(map)
            }
        },
    )
}

fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Debug, serde::Deserialize)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, serde::Deserialize)]
struct QuantOverride {
    #[serde(default)]
    group_size: Option<i32>,
    #[serde(default)]
    bits: Option<i32>,
}

// ── Layer primitives ──

pub struct QuantizedLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    dense_weight_t: Option<Array>,
}

impl QuantizedLinear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let out = if let Some(dense_weight_t) = &self.dense_weight_t {
            mlx_rs::ops::matmul(x, dense_weight_t)?
        } else {
            mlx_rs::ops::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                &self.biases,
                true,
                self.group_size,
                self.bits,
            )?
        };
        Ok(if let Some(ref bias) = self.bias {
            &out + bias
        } else {
            out
        })
    }
}

fn cpu_dense_weight_t(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    let dense_cpu = dequantize_device(
        weight,
        scales,
        biases,
        group_size,
        bits,
        StreamOrDevice::cpu(),
    )?;
    let dense_cpu = if dense_cpu.dtype() == Dtype::Float32 {
        dense_cpu
    } else if matches!(dense_cpu.dtype(), Dtype::Bfloat16 | Dtype::Float16) {
        dense_cpu.as_dtype(Dtype::Float32)?
    } else {
        bail!(
            "unsupported dense dequantized dtype for CPU fallback: {:?}",
            dense_cpu.dtype()
        );
    };
    let dense = Array::from_slice(dense_cpu.as_slice::<f32>(), dense_cpu.shape());

    Ok(dense.transpose_axes(&[1, 0])?)
}

pub struct RMSNorm {
    weight: Array,
    eps: f32,
    add_unit_offset: bool,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        if self.add_unit_offset {
            let one = array!(1.0f32).as_dtype(self.weight.dtype())?;
            let weight = self.weight.add(&one)?;
            Ok(mlx_rs::fast::rms_norm(x, &weight, self.eps)?)
        } else {
            Ok(mlx_rs::fast::rms_norm(x, &self.weight, self.eps)?)
        }
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
    v_norm: Option<RMSNorm>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    attn_logit_softcapping: Option<f32>,
    rope_dim: i32,
    rope_theta: f32,
    kv_shared_source: Option<usize>,
}

impl Attention {
    pub fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let q = q.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let q = if let Some(norm) = &self.q_norm {
            norm.forward(&q)?
        } else {
            q
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let q = apply_rope(&q, self.rope_dim, self.head_dim, self.rope_theta, 0)?;

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let k = k.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
        let k = if let Some(norm) = &self.k_norm {
            norm.forward(&k)?
        } else {
            k
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
        let v = if let Some(norm) = &self.v_norm {
            norm.forward(&v)?
        } else {
            v
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, self.rope_dim, self.head_dim, self.rope_theta, 0)?;

        let attn = if let Some(softcap) = self.attn_logit_softcapping {
            manual_scaled_dot_product_attention(&q, &k, &v, self.scale, softcap, l > 1)?
        } else {
            let mask = if l > 1 {
                Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
            } else {
                None
            };
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?
        };

        let attn =
            attn.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&attn)
    }

    pub fn forward(
        &self,
        x: &Array,
        cache: &mut KVCache,
        shared_cache: Option<&KVCache>,
    ) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let q = q.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let q = if let Some(norm) = &self.q_norm {
            norm.forward(&q)?
        } else {
            q
        }
        .transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.offset as i32;
        let q = apply_rope(&q, self.rope_dim, self.head_dim, self.rope_theta, offset)?;
        let (k, v) = if let Some(shared_cache) = shared_cache {
            shared_cache
                .views()
                .context("Gemma4 shared KV cache was empty")?
        } else {
            let k = self.k_proj.forward(x)?;
            let v = self.v_proj.forward(x)?;
            let k = k.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
            let k = if let Some(norm) = &self.k_norm {
                norm.forward(&k)?
            } else {
                k
            }
            .transpose_axes(&[0, 2, 1, 3])?;
            let v = v.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
            let v = if let Some(norm) = &self.v_norm {
                norm.forward(&v)?
            } else {
                v
            }
            .transpose_axes(&[0, 2, 1, 3])?;
            let k = apply_rope(&k, self.rope_dim, self.head_dim, self.rope_theta, offset)?;
            cache.update(k, v)?
        };

        // Causal mask for prefill (multi-token). Decode (l=1) needs no mask.
        let attn = if let Some(softcap) = self.attn_logit_softcapping {
            manual_scaled_dot_product_attention(&q, &k, &v, self.scale, softcap, l > 1)?
        } else {
            let mask = if l > 1 {
                Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
            } else {
                None
            };
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?
        };

        let attn =
            attn.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.num_heads * self.head_dim])?;

        self.o_proj.forward(&attn)
    }
}

fn causal_mask(query_len: i32, key_len: i32) -> Result<Array> {
    let key_positions = mlx_rs::ops::arange::<_, i32>(0, key_len, 1)?;
    let query_positions = if key_len > query_len {
        mlx_rs::ops::arange::<_, i32>(key_len - query_len, key_len, 1)?
    } else {
        mlx_rs::ops::arange::<_, i32>(0, query_len, 1)?
    };
    let left = query_positions.expand_dims(1)?;
    let right = key_positions.expand_dims(0)?;
    Ok(left.ge(&right)?)
}

fn manual_scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    softcap: f32,
    causal: bool,
) -> Result<Array> {
    let num_heads = q.shape()[1];
    let num_kv_heads = k.shape()[1];
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "cannot align attention heads: q_heads={}, kv_heads={}",
        num_heads,
        num_kv_heads
    );
    let repeats = num_heads / num_kv_heads;
    let batch = q.shape()[0];
    let query_len = q.shape()[2];
    let head_dim = q.shape()[3];
    let key_len = k.shape()[2];
    let mask = if causal {
        Some(causal_mask(query_len, key_len)?)
    } else {
        None
    };

    let mut queries = q.clone();
    if scale != 1.0 {
        queries = queries.multiply(&array!(scale))?;
    }

    let (queries, keys, values) = if repeats > 1 {
        (
            queries.reshape(&[batch, num_kv_heads, repeats, query_len, head_dim])?,
            k.expand_dims(2)?,
            v.expand_dims(2)?,
        )
    } else {
        (queries, k.clone(), v.clone())
    };

    let key_t = if repeats > 1 {
        keys.transpose_axes(&[0, 1, 2, 4, 3])?
    } else {
        keys.transpose_axes(&[0, 1, 3, 2])?
    };
    let mut scores = mlx_rs::ops::matmul(&queries, &key_t)?;
    scores = scores.divide(&array!(softcap))?;
    scores = mlx_rs::ops::tanh(&scores)?.multiply(&array!(softcap))?;
    if let Some(mask) = &mask {
        let fill = array!(scores.dtype().finfo_min()? as f32).as_dtype(scores.dtype())?;
        scores = mlx_rs::ops::r#where(mask, &scores, &fill)?;
    }
    let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
    let mut output = mlx_rs::ops::matmul(&probs, &values)?;
    if repeats > 1 {
        output = output.reshape(&[batch, num_heads, query_len, head_dim])?;
    }
    Ok(output)
}

fn apply_rope(
    x: &Array,
    rope_dim: i32,
    head_dim: i32,
    rope_theta: f32,
    offset: i32,
) -> Result<Array> {
    if rope_dim == head_dim {
        return Ok(mlx_rs::fast::rope(
            x,
            head_dim,
            false,
            Some(rope_theta),
            1.0,
            offset,
            None::<&Array>,
        )?);
    }

    let rotated = x.index((
        std::ops::RangeFull,
        std::ops::RangeFull,
        std::ops::RangeFull,
        ..rope_dim,
    ));
    let rotated = mlx_rs::fast::rope(
        &rotated,
        rope_dim,
        false,
        Some(rope_theta),
        1.0,
        offset,
        None::<&Array>,
    )?;
    let tail = x.index((
        std::ops::RangeFull,
        std::ops::RangeFull,
        std::ops::RangeFull,
        rope_dim..,
    ));
    Ok(mlx_rs::ops::concatenate_axis(&[&rotated, &tail], 3)?)
}

// ── MLP ──

pub struct MLP {
    gate_proj: QuantizedLinear,
    up_proj: QuantizedLinear,
    down_proj: QuantizedLinear,
    activation: Activation,
}

#[derive(Clone, Copy)]
pub enum Activation {
    Silu,
    GeluApproximate,
}

impl MLP {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let gate = match self.activation {
            Activation::Silu => &mlx_rs::ops::sigmoid(&gate)? * &gate,
            Activation::GeluApproximate => mlx_rs::nn::gelu_approximate(&gate)?,
        };
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(&gate * &up))
    }
}

// ── Transformer layer ──

pub struct Layer {
    attn: Attention,
    mlp: MLP,
    attn_in_norm: RMSNorm,
    attn_out_norm: Option<RMSNorm>,
    mlp_in_norm: RMSNorm,
    mlp_out_norm: Option<RMSNorm>,
    per_layer_input: Option<PerLayerInputBlock>,
    layer_scalar: Option<Array>,
}

impl Layer {
    pub fn forward_no_cache(&self, x: &Array, per_layer_input: Option<&Array>) -> Result<Array> {
        let attn = self.attn.forward_no_cache(&self.attn_in_norm.forward(x)?)?;
        let attn = if let Some(norm) = &self.attn_out_norm {
            norm.forward(&attn)?
        } else {
            attn
        };
        let h = &attn + x;
        let mlp = self.mlp.forward(&self.mlp_in_norm.forward(&h)?)?;
        let mlp = if let Some(norm) = &self.mlp_out_norm {
            norm.forward(&mlp)?
        } else {
            mlp
        };
        let mut out = &mlp + &h;

        if let (Some(block), Some(per_layer_input)) = (&self.per_layer_input, per_layer_input) {
            let residual = out.clone();
            let mut gated = block.input_gate.forward(&out)?;
            gated = match block.activation {
                Activation::Silu => &mlx_rs::ops::sigmoid(&gated)? * &gated,
                Activation::GeluApproximate => mlx_rs::nn::gelu_approximate(&gated)?,
            };
            gated = gated.multiply(per_layer_input)?;
            gated = block.projection.forward(&gated)?;
            gated = block.post_norm.forward(&gated)?;
            out = &gated + &residual;
        }

        if let Some(layer_scalar) = &self.layer_scalar {
            out = out.multiply(layer_scalar)?;
        }

        Ok(out)
    }

    pub fn forward(
        &self,
        x: &Array,
        per_layer_input: Option<&Array>,
        cache: &mut KVCache,
        shared_cache: Option<&KVCache>,
    ) -> Result<Array> {
        let attn = self
            .attn
            .forward(&self.attn_in_norm.forward(x)?, cache, shared_cache)?;
        let attn = if let Some(norm) = &self.attn_out_norm {
            norm.forward(&attn)?
        } else {
            attn
        };
        let h = &attn + x;
        let mlp = self.mlp.forward(&self.mlp_in_norm.forward(&h)?)?;
        let mlp = if let Some(norm) = &self.mlp_out_norm {
            norm.forward(&mlp)?
        } else {
            mlp
        };
        let mut out = &mlp + &h;

        if let (Some(block), Some(per_layer_input)) = (&self.per_layer_input, per_layer_input) {
            let residual = out.clone();
            let mut gated = block.input_gate.forward(&out)?;
            gated = match block.activation {
                Activation::Silu => &mlx_rs::ops::sigmoid(&gated)? * &gated,
                Activation::GeluApproximate => mlx_rs::nn::gelu_approximate(&gated)?,
            };
            gated = gated.multiply(per_layer_input)?;
            gated = block.projection.forward(&gated)?;
            gated = block.post_norm.forward(&gated)?;
            out = &gated + &residual;
        }

        if let Some(layer_scalar) = &self.layer_scalar {
            out = out.multiply(layer_scalar)?;
        }

        Ok(out)
    }
}

pub struct PerLayerInputBlock {
    input_gate: QuantizedLinear,
    projection: QuantizedLinear,
    post_norm: RMSNorm,
    activation: Activation,
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

    pub fn views(&self) -> Option<(Array, Array)> {
        use std::ops::RangeFull;

        if self.offset == 0 {
            return None;
        }
        let end_i = self.offset as i32;
        Some((
            self.keys
                .as_ref()?
                .index((RangeFull, RangeFull, ..end_i, RangeFull)),
            self.values
                .as_ref()?
                .index((RangeFull, RangeFull, ..end_i, RangeFull)),
        ))
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
    dense_weight: Option<Array>,
    dense_weight_t: Option<Array>,
}

impl QuantizedEmbedding {
    pub fn forward(&self, indices: &Array) -> Result<Array> {
        if let Some(dense_weight) = &self.dense_weight {
            return Ok(dense_weight.take_axis(indices, 0)?);
        }
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
            dense_weight_t: self.dense_weight_t.clone(),
        }
    }
}

fn quant_params_for(
    config: &Value,
    prefix: &str,
    default_group_size: i32,
    default_bits: i32,
) -> (i32, i32) {
    let override_cfg = config
        .get("quantization")
        .and_then(Value::as_object)
        .and_then(|q| q.get(prefix))
        .cloned()
        .and_then(|value| serde_json::from_value::<QuantOverride>(value).ok());

    (
        override_cfg
            .as_ref()
            .and_then(|cfg| cfg.group_size)
            .unwrap_or(default_group_size),
        override_cfg
            .as_ref()
            .and_then(|cfg| cfg.bits)
            .unwrap_or(default_bits),
    )
}

// ── Full model ──

pub struct MlxModel {
    embed_tokens: QuantizedEmbedding,
    embed_scale: f32,
    embed_tokens_per_layer: Option<QuantizedEmbedding>,
    embed_tokens_per_layer_scale: Option<f32>,
    per_layer_projection_norm: Option<RMSNorm>,
    per_layer_model_projection: Option<QuantizedLinear>,
    per_layer_model_projection_scale: Option<f32>,
    per_layer_input_scale: Option<f32>,
    layers: Vec<Layer>,
    norm: RMSNorm,
    lm_head: Option<QuantizedLinear>,
    final_logit_softcapping: Option<f32>,
    pub config: ModelConfig,
    pub tokenizer: tokenizers::Tokenizer,
    pub prompt_template: crate::mlx::template::PromptTemplate,
    tokenwise_prefill: bool,
    cacheless_generation: bool,
}

impl MlxModel {
    /// Load a quantized MLX model from a directory containing config.json,
    /// tokenizer.json, and model.safetensors.
    pub fn load(dir: &Path) -> Result<Self> {
        tracing::info!("MLX: loading model directory {}", dir.display());
        let config_text =
            std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let config_json: Value =
            serde_json::from_str(&config_text).context("parsing config.json")?;
        ensure_supported_mlx_model(dir, &config_json)?;
        let effective_config_json = effective_text_config_json(&config_json);
        let config: ModelConfig =
            serde_json::from_value(effective_config_json).context("parsing config.json")?;
        let arch = model_architecture(&config_json);

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
        let prefixes = tensor_prefixes(&tensors)?;

        let load_qlinear = |prefix: &str| -> Result<QuantizedLinear> {
            let (group_size, bits) = quant_params_for(&config_json, prefix, group_size, bits);
            let weight = tensors
                .get(&format!("{prefix}.weight"))
                .cloned()
                .with_context(|| format!("missing {prefix}.weight"))?;
            let scales = tensors
                .get(&format!("{prefix}.scales"))
                .cloned()
                .with_context(|| format!("missing {prefix}.scales"))?;
            let biases = tensors
                .get(&format!("{prefix}.biases"))
                .cloned()
                .with_context(|| format!("missing {prefix}.biases"))?;
            // Some Gemma4 MLX checkpoints use 5-bit weights for a subset of MLP
            // blocks, and current Metal qmm kernels are missing for that shape.
            let dense_weight_t = if bits == 5 {
                Some(cpu_dense_weight_t(
                    &weight, &scales, &biases, group_size, bits,
                )?)
            } else {
                None
            };
            Ok(QuantizedLinear {
                weight,
                scales,
                biases,
                bias: tensors.get(&format!("{prefix}.bias")).cloned(),
                group_size,
                bits,
                dense_weight_t,
            })
        };

        let (embed_group_size, embed_bits) = quant_params_for(
            &config_json,
            &format!("{}.embed_tokens", prefixes.model),
            group_size,
            bits,
        );
        let embed_weight = tensors
            .get(&format!("{}.embed_tokens.weight", prefixes.model))
            .cloned()
            .with_context(|| format!("missing {}.embed_tokens.weight", prefixes.model))?;
        let embed_scales = tensors
            .get(&format!("{}.embed_tokens.scales", prefixes.model))
            .cloned()
            .with_context(|| format!("missing {}.embed_tokens.scales", prefixes.model))?;
        let embed_biases = tensors
            .get(&format!("{}.embed_tokens.biases", prefixes.model))
            .cloned()
            .with_context(|| format!("missing {}.embed_tokens.biases", prefixes.model))?;
        let embed_dense_weight = None;
        let embed_dense_weight_t = None;
        let embed_tokens = QuantizedEmbedding {
            weight: embed_weight,
            scales: embed_scales,
            biases: embed_biases,
            group_size: embed_group_size,
            bits: embed_bits,
            dense_weight: embed_dense_weight,
            dense_weight_t: embed_dense_weight_t,
        };
        let embed_scale = if arch.uses_gemma_scaled_embeddings() {
            (config.hidden_size as f32).sqrt()
        } else {
            1.0
        };
        let embed_tokens_per_layer = if arch.is_gemma4() {
            let (group_size, bits) = quant_params_for(
                &config_json,
                &format!("{}.embed_tokens_per_layer", prefixes.model),
                group_size,
                bits,
            );
            Some(QuantizedEmbedding {
                weight: tensors
                    .get(&format!("{}.embed_tokens_per_layer.weight", prefixes.model))
                    .cloned()
                    .with_context(|| {
                        format!("missing {}.embed_tokens_per_layer.weight", prefixes.model)
                    })?,
                scales: tensors
                    .get(&format!("{}.embed_tokens_per_layer.scales", prefixes.model))
                    .cloned()
                    .with_context(|| {
                        format!("missing {}.embed_tokens_per_layer.scales", prefixes.model)
                    })?,
                biases: tensors
                    .get(&format!("{}.embed_tokens_per_layer.biases", prefixes.model))
                    .cloned()
                    .with_context(|| {
                        format!("missing {}.embed_tokens_per_layer.biases", prefixes.model)
                    })?,
                group_size,
                bits,
                dense_weight: None,
                dense_weight_t: None,
            })
        } else {
            None
        };
        let per_layer_projection_norm = if arch.is_gemma4() {
            Some(RMSNorm {
                weight: tensors
                    .get(&format!(
                        "{}.per_layer_projection_norm.weight",
                        prefixes.model
                    ))
                    .cloned()
                    .with_context(|| {
                        format!(
                            "missing {}.per_layer_projection_norm.weight",
                            prefixes.model
                        )
                    })?,
                eps: config.rms_norm_eps,
                add_unit_offset: false,
            })
        } else {
            None
        };
        let per_layer_model_projection = if arch.is_gemma4() {
            Some(load_qlinear(&format!(
                "{}.per_layer_model_projection",
                prefixes.model
            ))?)
        } else {
            None
        };

        let norm = RMSNorm {
            weight: tensors
                .get(&format!("{}.norm.weight", prefixes.model))
                .cloned()
                .with_context(|| format!("missing {}.norm.weight", prefixes.model))?,
            eps: config.rms_norm_eps,
            add_unit_offset: arch.uses_gemma_norm_offset(),
        };

        // Qwen3-class configs can declare an explicit head_dim that differs
        // from hidden_size / num_attention_heads.
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / config.num_attention_heads);
        let activation = match config.hidden_activation.as_deref() {
            Some("gelu_pytorch_tanh") | Some("gelu") => Activation::GeluApproximate,
            _ => Activation::Silu,
        };
        let first_kv_shared_layer_idx = config
            .num_kv_shared_layers
            .map(|n| (config.num_hidden_layers - n).max(0) as usize)
            .unwrap_or(config.num_hidden_layers as usize);
        let non_shared_layer_types = config
            .layer_types
            .as_ref()
            .map(|types| types[..first_kv_shared_layer_idx.min(types.len())].to_vec());

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("{}.layers.{i}", prefixes.model);
            let layer_type = config
                .layer_types
                .as_ref()
                .and_then(|types| types.get(i as usize))
                .map(String::as_str);
            let is_full_attention =
                arch.is_gemma4() && matches!(layer_type, Some("full_attention"));
            let layer_head_dim = if is_full_attention {
                config.global_head_dim.unwrap_or(head_dim)
            } else {
                head_dim
            };
            let rope_parameters = layer_type.and_then(|name| {
                config
                    .rope_parameters
                    .as_ref()
                    .and_then(|map| map.get(name))
            });
            let rope_dim = if is_full_attention {
                ((layer_head_dim as f32)
                    * rope_parameters
                        .and_then(|params| params.partial_rotary_factor)
                        .unwrap_or(1.0))
                .round() as i32
            } else {
                layer_head_dim
            };
            let rope_theta = rope_parameters
                .and_then(|params| params.rope_theta)
                .unwrap_or(config.rope_theta);
            let kv_shared_source = if arch.is_gemma4() && (i as usize) >= first_kv_shared_layer_idx
            {
                non_shared_layer_types.as_ref().and_then(|types| {
                    layer_type.and_then(|current| {
                        types
                            .iter()
                            .rposition(|candidate| candidate == current)
                            .map(|index| index)
                    })
                })
            } else {
                None
            };
            let scale = if arch.is_gemma4() {
                1.0
            } else if let Some(query_pre_attn_scalar) = config.query_pre_attn_scalar {
                1.0 / query_pre_attn_scalar.sqrt()
            } else {
                1.0 / (layer_head_dim as f32).sqrt()
            };
            let mlp_in_norm_key = if arch.is_gemma2() || arch.is_gemma3() || arch.is_gemma4() {
                format!("{p}.pre_feedforward_layernorm.weight")
            } else {
                format!("{p}.post_attention_layernorm.weight")
            };
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
                            add_unit_offset: arch.uses_gemma_norm_offset(),
                        }),
                    k_norm: tensors
                        .get(&format!("{p}.self_attn.k_norm.weight"))
                        .cloned()
                        .map(|weight| RMSNorm {
                            weight,
                            eps: config.rms_norm_eps,
                            add_unit_offset: arch.uses_gemma_norm_offset(),
                        }),
                    v_norm: arch.is_gemma4().then(|| RMSNorm {
                        weight: mlx_rs::ops::ones::<f32>(&[layer_head_dim])
                            .expect("allocating v_norm scale"),
                        eps: config.rms_norm_eps,
                        add_unit_offset: false,
                    }),
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: layer_head_dim,
                    scale,
                    attn_logit_softcapping: arch
                        .is_gemma2()
                        .then_some(config.attn_logit_softcapping.unwrap_or(50.0)),
                    rope_dim,
                    rope_theta,
                    kv_shared_source,
                },
                mlp: MLP {
                    gate_proj: load_qlinear(&format!("{p}.mlp.gate_proj"))?,
                    up_proj: load_qlinear(&format!("{p}.mlp.up_proj"))?,
                    down_proj: load_qlinear(&format!("{p}.mlp.down_proj"))?,
                    activation: activation,
                },
                attn_in_norm: RMSNorm {
                    weight: tensors
                        .get(&format!("{p}.input_layernorm.weight"))
                        .cloned()
                        .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
                    eps: config.rms_norm_eps,
                    add_unit_offset: arch.uses_gemma_norm_offset(),
                },
                attn_out_norm: (arch.is_gemma2() || arch.is_gemma3() || arch.is_gemma4())
                    .then(|| -> Result<RMSNorm> {
                        Ok(RMSNorm {
                            weight: tensors
                                .get(&format!("{p}.post_attention_layernorm.weight"))
                                .cloned()
                                .with_context(|| {
                                    format!("missing {p}.post_attention_layernorm.weight")
                                })?,
                            eps: config.rms_norm_eps,
                            add_unit_offset: arch.uses_gemma_norm_offset(),
                        })
                    })
                    .transpose()?,
                mlp_in_norm: RMSNorm {
                    weight: tensors.get(&mlp_in_norm_key).cloned().with_context(|| {
                        if arch.is_gemma2() || arch.is_gemma3() {
                            format!("missing {p}.pre_feedforward_layernorm.weight")
                        } else {
                            format!("missing {p}.post_attention_layernorm.weight")
                        }
                    })?,
                    eps: config.rms_norm_eps,
                    add_unit_offset: arch.uses_gemma_norm_offset(),
                },
                mlp_out_norm: (arch.is_gemma2() || arch.is_gemma3() || arch.is_gemma4())
                    .then(|| -> Result<RMSNorm> {
                        Ok(RMSNorm {
                            weight: tensors
                                .get(&format!("{p}.post_feedforward_layernorm.weight"))
                                .cloned()
                                .with_context(|| {
                                    format!("missing {p}.post_feedforward_layernorm.weight")
                                })?,
                            eps: config.rms_norm_eps,
                            add_unit_offset: arch.uses_gemma_norm_offset(),
                        })
                    })
                    .transpose()?,
                per_layer_input: arch
                    .is_gemma4()
                    .then(|| -> Result<PerLayerInputBlock> {
                        Ok(PerLayerInputBlock {
                            input_gate: load_qlinear(&format!("{p}.per_layer_input_gate"))?,
                            projection: load_qlinear(&format!("{p}.per_layer_projection"))?,
                            post_norm: RMSNorm {
                                weight: tensors
                                    .get(&format!("{p}.post_per_layer_input_norm.weight"))
                                    .cloned()
                                    .with_context(|| {
                                        format!("missing {p}.post_per_layer_input_norm.weight")
                                    })?,
                                eps: config.rms_norm_eps,
                                add_unit_offset: false,
                            },
                            activation,
                        })
                    })
                    .transpose()?,
                layer_scalar: arch
                    .is_gemma4()
                    .then(|| tensors.get(&format!("{p}.layer_scalar")).cloned())
                    .flatten(),
            });
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else if let Some(prefix) = prefixes.lm_head.as_deref() {
            if tensors.contains_key(&format!("{prefix}.weight")) {
                Some(load_qlinear(prefix)?)
            } else {
                None
            }
        } else {
            None
        };

        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        let prompt_template = crate::mlx::template::PromptTemplate::detect(dir, &config_json);

        Ok(MlxModel {
            embed_tokens,
            embed_scale,
            embed_tokens_per_layer,
            embed_tokens_per_layer_scale: arch.is_gemma4().then(|| {
                config
                    .hidden_size_per_layer_input
                    .map(|dim| (dim as f32).sqrt())
                    .unwrap_or(1.0)
            }),
            per_layer_projection_norm,
            per_layer_model_projection,
            per_layer_model_projection_scale: arch
                .is_gemma4()
                .then_some((config.hidden_size as f32).powf(-0.5)),
            per_layer_input_scale: arch.is_gemma4().then_some(2.0f32.powf(-0.5)),
            layers,
            norm,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping,
            config,
            tokenizer,
            prompt_template,
            tokenwise_prefill: arch.is_gemma2() || arch.is_gemma4(),
            cacheless_generation: arch.is_gemma2(),
        })
    }

    /// Run a forward pass. Input shape: [1, seq_len] of u32 token IDs.
    /// Returns logits [1, seq_len, vocab_size].
    pub fn forward(&self, tokens: &Array, caches: &mut [KVCache]) -> Result<Array> {
        let mut h = self.embed_tokens.forward(tokens)?;
        if self.embed_scale != 1.0 {
            h = h.multiply(&array!(self.embed_scale))?;
        }
        let per_layer_inputs = if let (
            Some(embed_tokens_per_layer),
            Some(embed_tokens_per_layer_scale),
            Some(per_layer_projection_norm),
            Some(per_layer_model_projection),
            Some(per_layer_model_projection_scale),
            Some(per_layer_input_scale),
            Some(hidden_size_per_layer_input),
        ) = (
            &self.embed_tokens_per_layer,
            self.embed_tokens_per_layer_scale,
            &self.per_layer_projection_norm,
            &self.per_layer_model_projection,
            self.per_layer_model_projection_scale,
            self.per_layer_input_scale,
            self.config.hidden_size_per_layer_input,
        ) {
            let per_layer_inputs = embed_tokens_per_layer
                .forward(tokens)?
                .multiply(&array!(embed_tokens_per_layer_scale))?
                .reshape(&[
                    tokens.shape()[0],
                    tokens.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_model_projection
                .forward(&h)?
                .multiply(&array!(per_layer_model_projection_scale))?
                .reshape(&[
                    h.shape()[0],
                    h.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_projection_norm.forward(&per_layer_projection)?;
            Some(
                (&per_layer_projection + &per_layer_inputs)
                    .multiply(&array!(per_layer_input_scale))?,
            )
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_input = per_layer_inputs.as_ref().map(|inputs| {
                inputs.index((
                    std::ops::RangeFull,
                    std::ops::RangeFull,
                    i as i32,
                    std::ops::RangeFull,
                ))
            });
            let (before, current_and_after) = caches.split_at_mut(i);
            let current_cache = &mut current_and_after[0];
            let shared_cache = layer
                .attn
                .kv_shared_source
                .and_then(|source| before.get(source));
            h = layer.forward(&h, layer_input.as_ref(), current_cache, shared_cache)?;
        }
        let h = self.norm.forward(&h)?;

        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)?
        } else {
            self.embed_tokens.as_linear().forward(&h)?
        };
        if let Some(softcap) = self.final_logit_softcapping {
            let scaled = logits.divide(&array!(softcap))?;
            Ok(mlx_rs::ops::tanh(&scaled)?.multiply(&array!(softcap))?)
        } else {
            Ok(logits)
        }
    }

    pub fn forward_no_cache(&self, tokens: &Array) -> Result<Array> {
        let mut h = self.embed_tokens.forward(tokens)?;
        if self.embed_scale != 1.0 {
            h = h.multiply(&array!(self.embed_scale))?;
        }
        let per_layer_inputs = if let (
            Some(embed_tokens_per_layer),
            Some(embed_tokens_per_layer_scale),
            Some(per_layer_projection_norm),
            Some(per_layer_model_projection),
            Some(per_layer_model_projection_scale),
            Some(per_layer_input_scale),
            Some(hidden_size_per_layer_input),
        ) = (
            &self.embed_tokens_per_layer,
            self.embed_tokens_per_layer_scale,
            &self.per_layer_projection_norm,
            &self.per_layer_model_projection,
            self.per_layer_model_projection_scale,
            self.per_layer_input_scale,
            self.config.hidden_size_per_layer_input,
        ) {
            let per_layer_inputs = embed_tokens_per_layer
                .forward(tokens)?
                .multiply(&array!(embed_tokens_per_layer_scale))?
                .reshape(&[
                    tokens.shape()[0],
                    tokens.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_model_projection
                .forward(&h)?
                .multiply(&array!(per_layer_model_projection_scale))?
                .reshape(&[
                    h.shape()[0],
                    h.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_projection_norm.forward(&per_layer_projection)?;
            Some(
                (&per_layer_projection + &per_layer_inputs)
                    .multiply(&array!(per_layer_input_scale))?,
            )
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_input = per_layer_inputs.as_ref().map(|inputs| {
                inputs.index((
                    std::ops::RangeFull,
                    std::ops::RangeFull,
                    i as i32,
                    std::ops::RangeFull,
                ))
            });
            h = layer.forward_no_cache(&h, layer_input.as_ref())?;
        }
        let h = self.norm.forward(&h)?;

        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)?
        } else {
            self.embed_tokens.as_linear().forward(&h)?
        };
        if let Some(softcap) = self.final_logit_softcapping {
            let scaled = logits.divide(&array!(softcap))?;
            Ok(mlx_rs::ops::tanh(&scaled)?.multiply(&array!(softcap))?)
        } else {
            Ok(logits)
        }
    }

    pub fn new_caches(&self) -> Vec<KVCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    pub fn tokenwise_prefill(&self) -> bool {
        self.tokenwise_prefill
    }

    pub fn cacheless_generation(&self) -> bool {
        self.cacheless_generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelArchitecture {
    LlamaLike,
    Gemma2,
    Gemma3,
    Gemma4,
}

impl ModelArchitecture {
    fn is_gemma2(self) -> bool {
        matches!(self, Self::Gemma2)
    }

    fn is_gemma3(self) -> bool {
        matches!(self, Self::Gemma3)
    }

    fn is_gemma4(self) -> bool {
        matches!(self, Self::Gemma4)
    }

    fn uses_gemma_norm_offset(self) -> bool {
        self.is_gemma2() || self.is_gemma3()
    }

    fn uses_gemma_scaled_embeddings(self) -> bool {
        self.is_gemma2() || self.is_gemma3() || self.is_gemma4()
    }
}

struct TensorPrefixes {
    model: String,
    lm_head: Option<String>,
}

fn model_architecture(config: &Value) -> ModelArchitecture {
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|value| value.get("model_type"))
                .and_then(|value| value.as_str())
        })
        .unwrap_or_default()
        .to_ascii_lowercase();

    if model_type.starts_with("gemma4") {
        ModelArchitecture::Gemma4
    } else if model_type.starts_with("gemma2") {
        ModelArchitecture::Gemma2
    } else if model_type.starts_with("gemma3") {
        ModelArchitecture::Gemma3
    } else {
        ModelArchitecture::LlamaLike
    }
}

fn effective_text_config_json(config: &Value) -> Value {
    let Some(text_config) = config
        .get("text_config")
        .and_then(|value| value.as_object())
    else {
        return config.clone();
    };

    let mut merged = serde_json::Map::new();
    for (key, value) in text_config {
        merged.insert(key.clone(), value.clone());
    }
    for key in [
        "quantization",
        "eos_token_id",
        "rope_theta",
        "rms_norm_eps",
        "head_dim",
        "max_position_embeddings",
        "tie_word_embeddings",
        "hidden_activation",
        "query_pre_attn_scalar",
        "global_head_dim",
        "vocab_size_per_layer_input",
        "vocab_size",
        "hidden_size_per_layer_input",
        "num_kv_shared_layers",
        "layer_types",
        "rope_parameters",
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "sliding_window",
        "cache_implementation",
    ] {
        if !merged.contains_key(key) || merged.get(key).is_some_and(Value::is_null) {
            if let Some(value) = config.get(key) {
                merged.insert(key.to_string(), value.clone());
            }
        }
    }
    if !merged.contains_key("architectures") {
        if let Some(value) = config.get("architectures") {
            merged.insert("architectures".to_string(), value.clone());
        }
    }

    Value::Object(merged)
}

fn tensor_prefixes(tensors: &HashMap<String, Array>) -> Result<TensorPrefixes> {
    if tensors.contains_key("model.embed_tokens.weight") {
        return Ok(TensorPrefixes {
            model: "model".to_string(),
            lm_head: Some("lm_head".to_string()),
        });
    }
    if tensors.contains_key("language_model.model.embed_tokens.weight") {
        return Ok(TensorPrefixes {
            model: "language_model.model".to_string(),
            lm_head: Some("language_model.lm_head".to_string()),
        });
    }
    bail!("unsupported MLX tensor prefix layout")
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
                | "gemma2"
                | "gemma3"
                | "gemma3_text"
                | "gemma4"
                | "gemma4_text"
                | "llamaforcausallm"
                | "qwen2forcausallm"
                | "qwen3forcausallm"
                | "gemma2forcausallm"
                | "gemma3forcausallm"
                | "gemma3forconditionalgeneration"
                | "gemma4forcausallm"
                | "gemma4forconditionalgeneration"
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
        "unsupported MLX model architecture in {} (model_type={}, architectures={}); supported MLX models currently cover Llama/Qwen/Gemma2/Gemma3/Gemma4-style safetensors checkpoints",
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
        let gemma2: Value = serde_json::json!({
            "model_type": "gemma2",
            "architectures": ["Gemma2ForCausalLM"]
        });
        let gemma3: Value = serde_json::json!({
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"]
        });
        let gemma4: Value = serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {"model_type": "gemma4_text"}
        });

        assert!(config_supports_mlx(&qwen));
        assert!(config_supports_mlx(&llama));
        assert!(config_supports_mlx(&gemma2));
        assert!(config_supports_mlx(&gemma3));
        assert!(config_supports_mlx(&gemma4));
    }

    #[test]
    fn config_rejects_other_reasoning_families_for_runtime_loading() {
        let glm: Value = serde_json::json!({
            "model_type": "glm",
            "architectures": ["GlmForCausalLM"]
        });
        let kimi: Value = serde_json::json!({
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"]
        });
        let gpt_oss: Value = serde_json::json!({
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"]
        });
        let lfm2: Value = serde_json::json!({
            "model_type": "lfm2_moe",
            "architectures": ["Lfm2MoeForCausalLM"]
        });

        assert!(!config_supports_mlx(&glm));
        assert!(!config_supports_mlx(&kimi));
        assert!(!config_supports_mlx(&gpt_oss));
        assert!(!config_supports_mlx(&lfm2));
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
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"]
        });

        let err = ensure_supported_mlx_model(&root, &config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported MLX model architecture"));
        assert!(err.contains("model_type=mistral"));
        assert!(err.contains("MistralForCausalLM"));
    }

    #[test]
    fn unsupported_reasoning_family_errors_are_explicit() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-mlx-unsupported-reasoning-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        for config in [
            serde_json::json!({
                "model_type": "glm",
                "architectures": ["GlmForCausalLM"]
            }),
            serde_json::json!({
                "model_type": "kimi_k25",
                "architectures": ["KimiK25ForConditionalGeneration"]
            }),
            serde_json::json!({
                "model_type": "gpt_oss",
                "architectures": ["GptOssForCausalLM"]
            }),
            serde_json::json!({
                "model_type": "lfm2_moe",
                "architectures": ["Lfm2MoeForCausalLM"]
            }),
        ] {
            let err = ensure_supported_mlx_model(&root, &config)
                .unwrap_err()
                .to_string();
            assert!(err.contains("unsupported MLX model architecture"));
            assert!(err.contains("model_type="));
            assert!(err.contains("architectures="));
        }
    }

    #[test]
    fn effective_text_config_extracts_gemma3_text_config() {
        let config = serde_json::json!({
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"],
            "quantization": {"group_size": 64, "bits": 4},
            "eos_token_id": [1, 106],
            "tie_word_embeddings": null,
            "head_dim": 256,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000,
            "max_position_embeddings": 32768,
            "hidden_activation": "gelu_pytorch_tanh",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 1152,
                "num_hidden_layers": 26,
                "intermediate_size": 6912,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "vocab_size": 262144
            }
        });

        let effective = effective_text_config_json(&config);
        let parsed: ModelConfig = serde_json::from_value(effective).unwrap();
        assert_eq!(parsed.hidden_size, 1152);
        assert_eq!(parsed.head_dim, Some(256));
        assert_eq!(parsed.query_pre_attn_scalar, Some(256.0));
        assert_eq!(
            parsed.hidden_activation.as_deref(),
            Some("gelu_pytorch_tanh")
        );
        assert!(!parsed.tie_word_embeddings);
        assert_eq!(parsed.eos_token_id, vec![1, 106]);
    }

    #[test]
    fn model_architecture_detects_gemma3_from_text_config() {
        let config = serde_json::json!({
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"],
            "text_config": {"model_type": "gemma3_text"}
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::Gemma3);
    }

    #[test]
    fn model_architecture_detects_gemma2() {
        let config = serde_json::json!({
            "model_type": "gemma2",
            "architectures": ["Gemma2ForCausalLM"]
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::Gemma2);
    }

    #[test]
    fn gemma2_config_parses_attention_softcaps() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "intermediate_size": 9216,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "query_pre_attn_scalar": 256,
            "vocab_size": 256000,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 8192,
            "tie_word_embeddings": false,
            "hidden_activation": "gelu_pytorch_tanh",
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "sliding_window": 4096,
            "cache_implementation": "hybrid",
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": 1
        }))
        .unwrap();

        assert_eq!(config.attn_logit_softcapping, Some(50.0));
        assert_eq!(config.final_logit_softcapping, Some(30.0));
        assert_eq!(config.sliding_window, Some(4096));
        assert_eq!(config.cache_implementation.as_deref(), Some("hybrid"));
    }

    #[test]
    fn effective_text_config_extracts_gemma4_text_config() {
        let config = serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "quantization": {"group_size": 64, "bits": 4},
            "eos_token_id": [1, 106],
            "tie_word_embeddings": false,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 2560,
                "hidden_size_per_layer_input": 256,
                "num_hidden_layers": 42,
                "intermediate_size": 10240,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_kv_shared_layers": 18,
                "head_dim": 256,
                "global_head_dim": 512,
                "query_pre_attn_scalar": 256.0,
                "vocab_size": 262400,
                "vocab_size_per_layer_input": 128,
                "layer_types": ["sliding_attention", "full_attention"],
                "final_logit_softcapping": 30.0
            }
        });

        let effective = effective_text_config_json(&config);
        let parsed: ModelConfig = serde_json::from_value(effective).unwrap();
        assert_eq!(parsed.hidden_size, 2560);
        assert_eq!(parsed.hidden_size_per_layer_input, Some(256));
        assert_eq!(parsed.head_dim, Some(256));
        assert_eq!(parsed.global_head_dim, Some(512));
        assert_eq!(parsed.num_kv_shared_layers, Some(18));
        assert_eq!(parsed.vocab_size_per_layer_input, Some(128));
        assert_eq!(
            parsed.layer_types.as_deref(),
            Some(
                &[
                    "sliding_attention".to_string(),
                    "full_attention".to_string()
                ][..]
            )
        );
        assert_eq!(parsed.final_logit_softcapping, Some(30.0));
    }

    #[test]
    fn model_architecture_detects_gemma4_from_text_config() {
        let config = serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {"model_type": "gemma4_text"}
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::Gemma4);
    }

    #[test]
    fn qwen3_flat_rope_parameters_are_accepted() {
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
            "tie_word_embeddings": true,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": 151645,
            "rope_theta": 1000000.0,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "rope_type": "default"
            }
        }))
        .unwrap();

        let params = config.rope_parameters.unwrap();
        assert_eq!(
            params.get("default").and_then(|p| p.rope_theta),
            Some(1000000.0)
        );
    }

    #[test]
    fn gemma3_uses_scaled_embeddings() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 1152,
            "num_hidden_layers": 26,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "query_pre_attn_scalar": 256,
            "vocab_size": 262144,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": null,
            "hidden_activation": "gelu_pytorch_tanh",
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": [1, 106]
        }))
        .unwrap();

        let embed_scale = (config.hidden_size as f32).sqrt();
        assert!((embed_scale - 33.941124).abs() < 0.001);
    }

    #[test]
    fn gemma4_uses_scaled_main_and_per_layer_embeddings() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 2560,
            "hidden_size_per_layer_input": 256,
            "num_hidden_layers": 42,
            "intermediate_size": 10240,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_kv_shared_layers": 18,
            "vocab_size": 262400,
            "vocab_size_per_layer_input": 128,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": false,
            "query_pre_attn_scalar": 256.0,
            "rope_theta": 10000.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": [1, 106]
        }))
        .unwrap();

        let embed_scale = (config.hidden_size as f32).sqrt();
        let per_layer_scale = (config.hidden_size_per_layer_input.unwrap() as f32).sqrt();
        assert!((embed_scale - 50.596443).abs() < 0.001);
        assert!((per_layer_scale - 16.0).abs() < 0.001);
    }

    #[test]
    fn quant_params_for_uses_tensor_specific_overrides() {
        let config = serde_json::json!({
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "language_model.model.embed_tokens": {"group_size": 64, "bits": 6},
                "language_model.model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 8}
            }
        });

        assert_eq!(
            quant_params_for(&config, "language_model.model.embed_tokens", 64, 4),
            (64, 6)
        );
        assert_eq!(
            quant_params_for(
                &config,
                "language_model.model.layers.0.self_attn.q_proj",
                64,
                4
            ),
            (64, 8)
        );
        assert_eq!(
            quant_params_for(
                &config,
                "language_model.model.layers.0.mlp.down_proj",
                64,
                4
            ),
            (64, 4)
        );
    }
}
