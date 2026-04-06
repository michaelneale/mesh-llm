//! Qwen2/Llama-style transformer model running on MLX via mlx-rs.
//!
//! Loads quantized safetensors and runs inference entirely on Metal GPU.
//! No Python, no subprocess — just Rust + MLX C library.

use anyhow::{bail, Context, Result};
use mlx_rs::array;
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::ops::{conv1d, dequantize, dequantize_device, pad, quantize};
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
    #[serde(default)]
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
    #[serde(alias = "norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[allow(dead_code)]
    #[serde(alias = "model_max_length")]
    pub max_position_embeddings: i32,
    #[serde(default, deserialize_with = "deserialize_nullable_bool")]
    pub tie_word_embeddings: bool,
    #[serde(default, alias = "hidden_act")]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default, alias = "num_shared_experts")]
    pub n_shared_experts: Option<i32>,
    #[serde(default, alias = "num_experts")]
    pub n_routed_experts: Option<i32>,
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default)]
    pub kv_lora_rank: Option<i32>,
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<i32>,
    #[serde(default)]
    pub v_head_dim: Option<i32>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub topk_method: Option<String>,
    #[serde(default, alias = "moe_renormalize")]
    pub norm_topk_prob: Option<bool>,
    #[serde(default, alias = "num_expert_group")]
    pub n_group: Option<i32>,
    #[serde(default)]
    pub topk_group: Option<i32>,
    #[serde(default, alias = "num_experts_per_token")]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub num_local_experts: Option<i32>,
    #[serde(default)]
    pub moe_layer_freq: Option<i32>,
    #[serde(default)]
    pub first_k_dense_replace: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub attention_bias: Option<bool>,
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
    pub sliding_window_pattern: Option<i32>,
    #[serde(default)]
    pub rope_local_base_freq: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub cache_implementation: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub conv_bias: Option<bool>,
    #[serde(default, alias = "conv_L_cache")]
    pub conv_l_cache: Option<i32>,
    #[serde(default)]
    pub block_norm_eps: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_dim: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_ff_dim: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_multiple_of: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_ffn_dim_multiplier: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_auto_adjust_ff_dim: Option<bool>,
    #[serde(default)]
    pub full_attn_idxs: Option<Vec<i32>>,
    #[serde(default)]
    pub linear_attn_config: Option<LinearAttnConfig>,
    #[serde(default)]
    #[allow(dead_code)]
    pub moe_router_activation_func: Option<String>,
    pub quantization: Option<QuantConfig>,
    /// EOS token ID(s) — can be a single int or array in config.json.
    #[serde(default, deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
}

#[derive(Debug, serde::Deserialize, Clone)]
pub struct LinearAttnConfig {
    #[allow(dead_code)]
    pub full_attn_layers: Vec<i32>,
    pub kda_layers: Vec<i32>,
    pub num_heads: i32,
    pub head_dim: i32,
    #[serde(default)]
    pub short_conv_kernel_size: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFamily {
    None,
    Qwen3,
    Glm,
    Kimi,
    GptOss,
    Lfm2,
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

fn force_dense_quantized_linear() -> bool {
    matches!(
        std::env::var("MESH_MLX_FORCE_DENSE_QUANTIZED_LINEAR").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
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

fn unit_rms_norm(x: &Array, eps: f32) -> Result<Array> {
    let width = x.shape()[x.shape().len() - 1];
    let weight = mlx_rs::ops::ones::<f32>(&[width])?.as_dtype(x.dtype())?;
    Ok(mlx_rs::fast::rms_norm(x, &weight, eps)?)
}

pub struct QuantizedMultiLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QuantizedMultiLinear {
    fn forward(&self, x: &Array, transpose: bool) -> Result<Array> {
        let num_heads = self.weight.shape()[0];
        let mut outputs = Vec::with_capacity(num_heads as usize);
        for head in 0..num_heads {
            let idx = Array::from_int(head);
            let w = self
                .weight
                .take_axis(&idx, 0)?
                .reshape(&[self.weight.shape()[1], self.weight.shape()[2]])?;
            let s = self
                .scales
                .take_axis(&idx, 0)?
                .reshape(&[self.scales.shape()[1], self.scales.shape()[2]])?;
            let b = self
                .biases
                .take_axis(&idx, 0)?
                .reshape(&[self.biases.shape()[1], self.biases.shape()[2]])?;
            let xh = x.index((
                std::ops::RangeFull,
                head,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let out = mlx_rs::ops::quantized_matmul(
                &xh,
                &w,
                &s,
                &b,
                transpose,
                self.group_size,
                self.bits,
            )?;
            outputs.push(out.expand_dims(1)?);
        }
        let output_refs: Vec<&Array> = outputs.iter().collect();
        Ok(mlx_rs::ops::concatenate_axis(&output_refs, 1)?)
    }
}

fn quantize_stacked_weights(
    dense: &Array,
    group_size: i32,
    bits: i32,
) -> Result<(Array, Array, Array)> {
    let num_heads = dense.shape()[0];
    let mut q_weights = Vec::with_capacity(num_heads as usize);
    let mut q_scales = Vec::with_capacity(num_heads as usize);
    let mut q_biases = Vec::with_capacity(num_heads as usize);
    for head in 0..num_heads {
        let slice = dense
            .index((head, std::ops::RangeFull, std::ops::RangeFull))
            .reshape(&[dense.shape()[1], dense.shape()[2]])?;
        let (w, s, b) = quantize(&slice, group_size, bits)?;
        q_weights.push(w.expand_dims(0)?);
        q_scales.push(s.expand_dims(0)?);
        q_biases.push(b.expand_dims(0)?);
    }
    let q_weight_refs: Vec<&Array> = q_weights.iter().collect();
    let q_scale_refs: Vec<&Array> = q_scales.iter().collect();
    let q_bias_refs: Vec<&Array> = q_biases.iter().collect();
    Ok((
        mlx_rs::ops::concatenate_axis(&q_weight_refs, 0)?,
        mlx_rs::ops::concatenate_axis(&q_scale_refs, 0)?,
        mlx_rs::ops::concatenate_axis(&q_bias_refs, 0)?,
    ))
}

fn expert_slice_2d(array: &Array, expert: i32) -> Result<Array> {
    Ok(array
        .take_axis(&Array::from_int(expert), 0)?
        .reshape(&[array.shape()[1], array.shape()[2]])?)
}

pub struct QuantizedSwitchLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
}

impl QuantizedSwitchLinear {
    fn forward_single(&self, x: &Array, expert: i32) -> Result<Array> {
        let out = mlx_rs::ops::quantized_matmul(
            x,
            &expert_slice_2d(&self.weight, expert)?,
            &expert_slice_2d(&self.scales, expert)?,
            &expert_slice_2d(&self.biases, expert)?,
            true,
            self.group_size,
            self.bits,
        )?;
        Ok(if let Some(bias) = &self.bias {
            let bias = bias
                .take_axis(&Array::from_int(expert), 0)?
                .reshape(&[1, bias.shape()[1]])?;
            out.add(&bias)?
        } else {
            out
        })
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
    window_size: Option<i32>,
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

        let mask = if self.window_size.is_some() {
            attention_mask(l, l, 0, self.window_size)?
        } else {
            None
        };
        let attn = if self.attn_logit_softcapping.is_some() || mask.is_some() {
            manual_scaled_dot_product_attention_with_mask(
                &q,
                &k,
                &v,
                self.scale,
                self.attn_logit_softcapping,
                mask.as_ref(),
            )?
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
        let mask = if self.window_size.is_some() {
            attention_mask(l, k.shape()[2], offset, self.window_size)?
        } else {
            None
        };
        let attn = if self.attn_logit_softcapping.is_some() || mask.is_some() {
            manual_scaled_dot_product_attention_with_mask(
                &q,
                &k,
                &v,
                self.scale,
                self.attn_logit_softcapping,
                mask.as_ref(),
            )?
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

pub struct DeepseekV3Attention {
    q_proj: Option<QuantizedLinear>,
    q_a_proj: Option<QuantizedLinear>,
    q_a_layernorm: Option<RMSNorm>,
    q_b_proj: Option<QuantizedLinear>,
    kv_a_proj_with_mqa: QuantizedLinear,
    kv_a_layernorm: RMSNorm,
    embed_q: QuantizedMultiLinear,
    unembed_out: QuantizedMultiLinear,
    o_proj: QuantizedLinear,
    num_heads: i32,
    q_head_dim: i32,
    qk_rope_head_dim: i32,
    qk_nope_head_dim: i32,
    kv_lora_rank: i32,
    v_head_dim: i32,
    scale: f32,
    rope_theta: f32,
}

impl DeepseekV3Attention {
    fn build_q(&self, x: &Array) -> Result<Array> {
        let q = if let Some(q_proj) = &self.q_proj {
            q_proj.forward(x)?
        } else {
            self.q_b_proj
                .as_ref()
                .context("missing q_b_proj for DeepSeekV3 attention")?
                .forward(
                    &self
                        .q_a_layernorm
                        .as_ref()
                        .context("missing q_a_layernorm for DeepSeekV3 attention")?
                        .forward(
                            &self
                                .q_a_proj
                                .as_ref()
                                .context("missing q_a_proj for DeepSeekV3 attention")?
                                .forward(x)?,
                        )?,
                )?
        };
        Ok(q)
    }

    fn attention_mask(&self, q_pe: &Array, k_pe: &Array, causal: bool) -> Result<Array> {
        let mut pe_scores = mlx_rs::ops::matmul(
            &q_pe.multiply(&array!(self.scale))?,
            &k_pe.transpose_axes(&[0, 1, 3, 2])?,
        )?;
        if causal {
            let mask = attention_mask(q_pe.shape()[2], k_pe.shape()[2], 0, None)?
                .context("expected causal mask")?;
            let fill = array!(pe_scores.dtype().finfo_min()? as f32).as_dtype(pe_scores.dtype())?;
            pe_scores = mlx_rs::ops::r#where(&mask, &pe_scores, &fill)?;
        }
        Ok(pe_scores)
    }

    fn forward_impl(&self, x: &Array, cache: Option<&mut KVCache>) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self
            .build_q(x)?
            .reshape(&[b, l, self.num_heads, self.q_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            std::ops::RangeFull,
            ..self.qk_nope_head_dim,
        ));
        let q_pe = q.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            std::ops::RangeFull,
            self.qk_nope_head_dim..,
        ));

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_latent = compressed_kv.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            ..self.kv_lora_rank,
        ));
        let k_pe = compressed_kv.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            self.kv_lora_rank..,
        ));
        let kv_latent = self.kv_a_layernorm.forward(&kv_latent)?.expand_dims(1)?;
        let k_pe = k_pe
            .reshape(&[b, l, 1, self.qk_rope_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.as_ref().map(|cache| cache.offset as i32).unwrap_or(0);
        let q_pe = apply_rope(
            &q_pe,
            self.qk_rope_head_dim,
            self.qk_rope_head_dim,
            self.rope_theta,
            offset,
        )?;
        let k_pe = apply_rope(
            &k_pe,
            self.qk_rope_head_dim,
            self.qk_rope_head_dim,
            self.rope_theta,
            offset,
        )?;

        let (kv_latent, k_pe) = if let Some(cache) = cache {
            cache.update(kv_latent, k_pe)?
        } else {
            (kv_latent, k_pe)
        };

        let mask = self.attention_mask(&q_pe, &k_pe, l > 1)?;
        let output = if l == 1 {
            let q_nope = self.embed_q.forward(&q_nope, true)?;
            let output = mlx_rs::fast::scaled_dot_product_attention(
                &q_nope,
                &kv_latent,
                &kv_latent,
                self.scale,
                Some((&mask).into()),
            )?;
            self.unembed_out.forward(&output, true)?
        } else {
            let k = self.embed_q.forward(&kv_latent, false)?;
            let v = self.unembed_out.forward(&kv_latent, true)?;
            mlx_rs::fast::scaled_dot_product_attention(
                &q_nope,
                &k,
                &v,
                self.scale,
                Some((&mask).into()),
            )?
        };

        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            l,
            self.num_heads * self.v_head_dim,
        ])?;
        self.o_proj.forward(&output)
    }

    pub fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        self.forward_impl(x, None)
    }

    pub fn forward(&self, x: &Array, cache: &mut KVCache) -> Result<Array> {
        self.forward_impl(x, Some(cache))
    }
}

fn attention_mask(
    query_len: i32,
    key_len: i32,
    offset: i32,
    window_size: Option<i32>,
) -> Result<Option<Array>> {
    if query_len == 1 && window_size.is_none() {
        return Ok(None);
    }

    let key_positions = mlx_rs::ops::arange::<_, i32>(0, key_len, 1)?;
    let query_positions = mlx_rs::ops::arange::<_, i32>(offset, offset + query_len, 1)?;
    let left = query_positions.expand_dims(1)?;
    let right = key_positions.expand_dims(0)?;
    let mut mask = left.ge(&right)?;
    if let Some(window_size) = window_size {
        let upper_bound = right.add(&array!(window_size))?;
        mask = mask.logical_and(&left.lt(&upper_bound)?)?;
    }
    Ok(Some(mask))
}

fn manual_scaled_dot_product_attention_with_mask(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    softcap: Option<f32>,
    mask: Option<&Array>,
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
    if let Some(softcap) = softcap {
        scores = scores.divide(&array!(softcap))?;
        scores = mlx_rs::ops::tanh(&scores)?.multiply(&array!(softcap))?;
    }
    if let Some(mask) = mask {
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
    gate_up_proj: Option<QuantizedLinear>,
    gate_proj: Option<QuantizedLinear>,
    up_proj: Option<QuantizedLinear>,
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
        let (gate, up) = if let Some(gate_up_proj) = &self.gate_up_proj {
            let gate_up = gate_up_proj.forward(x)?;
            let hidden = gate_up.shape()[gate_up.shape().len() - 1] / 2;
            let gate = gate_up.index((std::ops::RangeFull, std::ops::RangeFull, 0..hidden));
            let up = gate_up.index((
                std::ops::RangeFull,
                std::ops::RangeFull,
                hidden..(hidden * 2),
            ));
            (gate, up)
        } else {
            (
                self.gate_proj
                    .as_ref()
                    .context("missing gate_proj for unfused MLP")?
                    .forward(x)?,
                self.up_proj
                    .as_ref()
                    .context("missing up_proj for unfused MLP")?
                    .forward(x)?,
            )
        };
        let gate = match self.activation {
            Activation::Silu => &mlx_rs::ops::sigmoid(&gate)? * &gate,
            Activation::GeluApproximate => mlx_rs::nn::gelu_approximate(&gate)?,
        };
        self.down_proj.forward(&(&gate * &up))
    }
}

pub struct DeepseekV3MoE {
    switch_gate_proj: QuantizedSwitchLinear,
    switch_up_proj: QuantizedSwitchLinear,
    switch_down_proj: QuantizedSwitchLinear,
    gate_weight: Array,
    gate_bias: Array,
    top_k: i32,
    n_group: i32,
    topk_group: i32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
    shared_experts: Option<MLP>,
}

impl DeepseekV3MoE {
    fn gate(&self, x: &Array) -> Result<(Array, Array)> {
        let mut scores = mlx_rs::ops::matmul(x, &self.gate_weight.transpose_axes(&[1, 0])?)?;
        scores = mlx_rs::ops::sigmoid(&scores.as_dtype(Dtype::Float32)?)?;
        let orig_scores = scores.clone();
        scores = scores.add(&self.gate_bias)?;

        if self.n_group > 1 {
            let experts_per_group = scores.shape()[scores.shape().len() - 1] / self.n_group;
            let scores_grouped = scores.reshape(&[-1, self.n_group, experts_per_group])?;
            let top2 = mlx_rs::ops::indexing::topk_axis(&scores_grouped, 2, -1)?;
            let group_scores = top2.sum_axes(&[-1], true)?;
            let k = self.n_group - self.topk_group;
            let group_idx = mlx_rs::ops::argpartition_axis(&group_scores, k - 1, -2)?.index((
                std::ops::RangeFull,
                ..k,
                std::ops::RangeFull,
            ));
            let scores_grouped = mlx_rs::ops::indexing::put_along_axis(
                &scores_grouped,
                &group_idx,
                &array!(0.0f32),
                -2,
            )?;
            scores = scores_grouped.reshape(&[-1, self.gate_weight.shape()[0]])?;
        }

        let inds = mlx_rs::ops::argpartition_axis(
            &scores.multiply(&array!(-1.0f32))?,
            self.top_k - 1,
            -1,
        )?
        .index((std::ops::RangeFull, ..self.top_k));
        let mut probs = mlx_rs::ops::indexing::take_along_axis(&orig_scores, &inds, -1)?
            .as_dtype(Dtype::Float32)?;
        if self.top_k > 1 && self.norm_topk_prob {
            probs = probs.divide(&probs.sum_axes(&[-1], true)?)?;
        }
        probs = probs.multiply(&array!(self.routed_scaling_factor))?;
        Ok((inds, probs))
    }

    fn switch_forward_single(&self, x: &Array, expert: i32) -> Result<Array> {
        let x_up = self.switch_up_proj.forward_single(x, expert)?;
        let x_gate = self.switch_gate_proj.forward_single(x, expert)?;
        let activated = &mlx_rs::ops::sigmoid(&x_gate)? * &x_gate;
        self.switch_down_proj
            .forward_single(&activated.multiply(&x_up)?, expert)
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let b = x.shape()[0];
        let l = x.shape()[1];
        let hidden = x.shape()[2];
        let flat = x.reshape(&[b * l, hidden])?;
        let (inds, scores) = self.gate(&flat)?;
        mlx_rs::transforms::eval([&inds, &scores])?;
        let inds_slice = inds.as_slice::<u32>();
        let scores_slice = scores.as_slice::<f32>();
        let mut outputs = Vec::with_capacity((b * l) as usize);
        for token_idx in 0..(b * l) {
            let x_tok = flat.index((token_idx..token_idx + 1, std::ops::RangeFull));
            let mut token_out: Option<Array> = None;
            for expert_slot in 0..self.top_k {
                let offset = (token_idx * self.top_k + expert_slot) as usize;
                let expert = inds_slice[offset] as i32;
                let score = scores_slice[offset];
                let routed = self
                    .switch_forward_single(&x_tok, expert)?
                    .multiply(&array!(score))?;
                token_out = Some(match token_out {
                    Some(acc) => acc.add(&routed)?,
                    None => routed,
                });
            }
            let token_out = if let Some(shared) = &self.shared_experts {
                token_out.unwrap().add(&shared.forward(&x_tok)?)?
            } else {
                token_out.unwrap()
            };
            outputs.push(token_out);
        }
        let output_refs: Vec<&Array> = outputs.iter().collect();
        Ok(mlx_rs::ops::concatenate_axis(&output_refs, 0)?.reshape(&[b, l, hidden])?)
    }
}

pub struct GptOssMoE {
    switch_gate_proj: QuantizedSwitchLinear,
    switch_up_proj: QuantizedSwitchLinear,
    switch_down_proj: QuantizedSwitchLinear,
    router: QuantizedLinear,
    top_k: i32,
}

impl GptOssMoE {
    fn switch_forward_single(&self, x: &Array, expert: i32) -> Result<Array> {
        let x_linear = self.switch_up_proj.forward_single(x, expert)?;
        let x_glu = self.switch_gate_proj.forward_single(x, expert)?;
        let x_glu = mlx_rs::ops::clip(&x_glu, ((), 7.0f32))?;
        let x_linear = mlx_rs::ops::clip(&x_linear, (-7.0f32, 7.0f32))?;
        let out_glu =
            x_glu.multiply(&mlx_rs::ops::sigmoid(&x_glu.multiply(&array!(1.702f32))?)?)?;
        let activated = out_glu.multiply(&x_linear.add(&array!(1.0f32))?)?;
        self.switch_down_proj.forward_single(&activated, expert)
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let b = x.shape()[0];
        let l = x.shape()[1];
        let hidden = x.shape()[2];
        let flat = x.reshape(&[b * l, hidden])?;
        let router_logits = self.router.forward(&flat)?.as_dtype(Dtype::Float32)?;
        let inds = mlx_rs::ops::argpartition_axis(
            &router_logits.multiply(&array!(-1.0f32))?,
            self.top_k - 1,
            -1,
        )?
        .index((std::ops::RangeFull, ..self.top_k));
        let weights = mlx_rs::ops::indexing::take_along_axis(&router_logits, &inds, -1)?;
        let weights = mlx_rs::ops::softmax_axis(&weights, -1, true)?;
        mlx_rs::transforms::eval([&inds, &weights])?;
        let inds_slice = inds.as_slice::<u32>();
        let weights_slice = weights.as_slice::<f32>();
        let mut outputs = Vec::with_capacity((b * l) as usize);
        for token_idx in 0..(b * l) {
            let x_tok = flat.index((token_idx..token_idx + 1, std::ops::RangeFull));
            let mut token_out: Option<Array> = None;
            for expert_slot in 0..self.top_k {
                let offset = (token_idx * self.top_k + expert_slot) as usize;
                let expert = inds_slice[offset] as i32;
                let weight = weights_slice[offset];
                let routed = self
                    .switch_forward_single(&x_tok, expert)?
                    .multiply(&array!(weight))?;
                token_out = Some(match token_out {
                    Some(acc) => acc.add(&routed)?,
                    None => routed,
                });
            }
            outputs.push(token_out.context("gpt-oss moe produced no experts")?);
        }
        let output_refs: Vec<&Array> = outputs.iter().collect();
        Ok(mlx_rs::ops::concatenate_axis(&output_refs, 0)?.reshape(&[b, l, hidden])?)
    }
}

pub struct KimiMlaAttention {
    q_proj: QuantizedLinear,
    kv_a_proj_with_mqa: QuantizedLinear,
    kv_a_layernorm: RMSNorm,
    embed_q: QuantizedMultiLinear,
    unembed_out: QuantizedMultiLinear,
    o_proj: QuantizedLinear,
    num_heads: i32,
    q_head_dim: i32,
    qk_rope_head_dim: i32,
    qk_nope_head_dim: i32,
    kv_lora_rank: i32,
    v_head_dim: i32,
    scale: f32,
}

impl KimiMlaAttention {
    pub fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, self.num_heads, self.q_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            std::ops::RangeFull,
            ..self.qk_nope_head_dim,
        ));
        let q_pe = q.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            std::ops::RangeFull,
            self.qk_nope_head_dim..,
        ));

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_latent = compressed_kv.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            ..self.kv_lora_rank,
        ));
        let k_pe = compressed_kv.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            self.kv_lora_rank..,
        ));
        let kv_latent = self.kv_a_layernorm.forward(&kv_latent)?.expand_dims(1)?;
        let k_pe = k_pe
            .reshape(&[b, l, 1, self.qk_rope_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let pe_scores = mlx_rs::ops::matmul(
            &q_pe.multiply(&array!(self.scale))?,
            &k_pe.transpose_axes(&[0, 1, 3, 2])?,
        )?;

        let output = if l == 1 {
            let q_nope = self.embed_q.forward(&q_nope, true)?;
            let scores = mlx_rs::ops::matmul(
                &q_nope.multiply(&array!(self.scale))?,
                &kv_latent.transpose_axes(&[0, 1, 3, 2])?,
            )?
            .add(&pe_scores)?;
            let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
            let output = mlx_rs::ops::matmul(&probs, &kv_latent)?;
            self.unembed_out.forward(&output, true)?
        } else {
            let k = self.embed_q.forward(&kv_latent, false)?;
            let v = self.unembed_out.forward(&kv_latent, true)?;
            let mask = attention_mask(l, l, 0, None)?.context("expected kimi mla mask")?;
            let scores = mlx_rs::ops::matmul(
                &q_nope.multiply(&array!(self.scale))?,
                &k.transpose_axes(&[0, 1, 3, 2])?,
            )?
            .add(&pe_scores)?;
            let fill = array!(scores.dtype().finfo_min()? as f32).as_dtype(scores.dtype())?;
            let scores = mlx_rs::ops::r#where(&mask, &scores, &fill)?;
            let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
            mlx_rs::ops::matmul(&probs, &v)?
        };

        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            l,
            self.num_heads * self.v_head_dim,
        ])?;
        self.o_proj.forward(&output)
    }
}

pub struct KimiShortConv {
    conv_weight: Array,
    kernel_size: i32,
    channels: i32,
}

impl KimiShortConv {
    fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let x = pad(
            x,
            &[(0, 0), (self.kernel_size - 1, 0), (0, 0)],
            None::<Array>,
            None::<mlx_rs::ops::PadMode>,
        )?;
        let x = conv1d(
            &x,
            &self.conv_weight,
            None::<i32>,
            None::<i32>,
            None::<i32>,
            Some(self.channels),
        )?;
        Ok(&mlx_rs::ops::sigmoid(&x)? * &x)
    }
}

pub struct KimiDeltaAttention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    q_conv: KimiShortConv,
    k_conv: KimiShortConv,
    v_conv: KimiShortConv,
    f_a_proj: QuantizedLinear,
    f_b_proj: QuantizedLinear,
    b_proj: QuantizedLinear,
    g_a_proj: QuantizedLinear,
    g_b_proj: QuantizedLinear,
    a_log: Array,
    dt_bias: Array,
    o_norm: RMSNorm,
    o_proj: QuantizedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl KimiDeltaAttention {
    fn gated_delta_update(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        a: &Array,
        b: &Array,
    ) -> Result<Array> {
        let bsz = q.shape()[0];
        let seq = q.shape()[1];
        let heads = q.shape()[2];
        let dim = q.shape()[3];
        let mut state = mlx_rs::ops::zeros_dtype(&[bsz, heads, dim, dim], q.dtype())?;
        let beta = mlx_rs::ops::sigmoid(b)?;
        let a = a.add(
            &self
                .dt_bias
                .reshape(&[1, 1, self.num_heads, self.head_dim])?,
        )?;
        let g = mlx_rs::ops::exp(&mlx_rs::ops::negative(
            &mlx_rs::ops::exp(
                &self
                    .a_log
                    .reshape(&[1, 1, self.num_heads, 1])?
                    .as_dtype(Dtype::Float32)?,
            )?
            .multiply(&mlx_rs::nn::softplus(&a)?)?,
        )?)?
        .as_dtype(q.dtype())?;

        let mut ys = Vec::with_capacity(seq as usize);
        for t in 0..seq {
            let q_t = q.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let k_t = k.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let v_t = v.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let beta_t = beta.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let g_t = g.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            state = state.multiply(&g_t.expand_dims(2)?)?;
            let kv_mem = state
                .multiply(&k_t.expand_dims(2)?)?
                .sum_axes(&[-1], false)?;
            let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?;
            state = state.add(&k_t.expand_dims(2)?.multiply(&delta.expand_dims(3)?)?)?;
            ys.push(
                state
                    .multiply(&q_t.expand_dims(2)?)?
                    .sum_axes(&[-1], false)?,
            );
        }
        let y_refs: Vec<&Array> = ys.iter().collect();
        Ok(mlx_rs::ops::stack(&y_refs)?.swap_axes(0, 1)?)
    }

    pub fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);
        let q_conv = self.q_conv.forward_no_cache(&self.q_proj.forward(x)?)?;
        let k_conv = self.k_conv.forward_no_cache(&self.k_proj.forward(x)?)?;
        let v_conv = self.v_conv.forward_no_cache(&self.v_proj.forward(x)?)?;

        let mut q = q_conv.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let mut k = k_conv.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let v = v_conv.reshape(&[b, l, self.num_heads, self.head_dim])?;

        q = unit_rms_norm(&q, 1e-6)?.multiply(&array!(self.scale * self.scale))?;
        k = unit_rms_norm(&k, 1e-6)?.multiply(&array!(self.scale))?;

        let a_logits = self
            .f_b_proj
            .forward(&self.f_a_proj.forward(x)?)?
            .reshape(&[b, l, self.num_heads, self.head_dim])?;
        let b_logits = self
            .b_proj
            .forward(x)?
            .reshape(&[b, l, self.num_heads, 1])?;
        let out = self.gated_delta_update(&q, &k, &v, &a_logits, &b_logits)?;
        let gate = self
            .g_b_proj
            .forward(&self.g_a_proj.forward(x)?)?
            .reshape(&[b, l, self.num_heads, self.head_dim])?;
        let out = self
            .o_norm
            .forward(&out)?
            .multiply(&mlx_rs::ops::sigmoid(&gate)?)?
            .reshape(&[b, l, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&out)
    }
}

pub struct Lfm2ShortConv {
    conv_weight: Array,
    in_proj: QuantizedLinear,
    out_proj: QuantizedLinear,
    hidden_size: i32,
    conv_l_cache: i32,
}

impl Lfm2ShortConv {
    fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let bcx = self.in_proj.forward(x)?;
        let hidden = self.hidden_size;
        let b = bcx.index((std::ops::RangeFull, std::ops::RangeFull, 0..hidden));
        let c = bcx.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            hidden..(hidden * 2),
        ));
        let x_proj = bcx.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            (hidden * 2)..(hidden * 3),
        ));
        let bx = b.multiply(&x_proj)?;
        let bx = pad(
            &bx,
            &[(0, 0), (self.conv_l_cache - 1, 0), (0, 0)],
            None::<Array>,
            None::<mlx_rs::ops::PadMode>,
        )?;
        let conv_out = conv1d(
            &bx,
            &self.conv_weight,
            None::<i32>,
            None::<i32>,
            None::<i32>,
            Some(self.hidden_size),
        )?;
        let y = c.multiply(&conv_out)?;
        self.out_proj.forward(&y)
    }
}

pub enum AttentionKind {
    Standard(Attention),
    DeepseekV3(DeepseekV3Attention),
    KimiMla(KimiMlaAttention),
    KimiDelta(KimiDeltaAttention),
    Lfm2ShortConv(Lfm2ShortConv),
}

impl AttentionKind {
    fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Standard(attn) => attn.forward_no_cache(x),
            Self::DeepseekV3(attn) => attn.forward_no_cache(x),
            Self::KimiMla(attn) => attn.forward_no_cache(x),
            Self::KimiDelta(attn) => attn.forward_no_cache(x),
            Self::Lfm2ShortConv(conv) => conv.forward_no_cache(x),
        }
    }

    fn forward(
        &self,
        x: &Array,
        cache: &mut KVCache,
        shared_cache: Option<&KVCache>,
    ) -> Result<Array> {
        match self {
            Self::Standard(attn) => attn.forward(x, cache, shared_cache),
            Self::DeepseekV3(attn) => attn.forward(x, cache),
            Self::KimiMla(_) | Self::KimiDelta(_) => {
                bail!("Kimi Linear currently requires cacheless generation")
            }
            Self::Lfm2ShortConv(_) => {
                bail!("LFM2 ShortConv currently requires cacheless generation")
            }
        }
    }

    fn kv_shared_source(&self) -> Option<usize> {
        match self {
            Self::Standard(attn) => attn.kv_shared_source,
            Self::DeepseekV3(_) => None,
            Self::KimiMla(_) | Self::KimiDelta(_) => None,
            Self::Lfm2ShortConv(_) => None,
        }
    }
}

pub enum MlpKind {
    Dense(MLP),
    DeepseekV3MoE(DeepseekV3MoE),
    GptOssMoE(GptOssMoE),
}

impl MlpKind {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::DeepseekV3MoE(moe) => moe.forward(x),
            Self::GptOssMoE(moe) => moe.forward(x),
        }
    }
}

// ── Transformer layer ──

pub struct Layer {
    attn: AttentionKind,
    mlp: MlpKind,
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

fn transform_deepseek_v3_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
    config_json: &Value,
    default_group_size: i32,
    default_bits: i32,
) -> Result<()> {
    let num_heads = config.num_attention_heads;
    let qk_nope_head_dim = config
        .qk_nope_head_dim
        .context("missing qk_nope_head_dim for DeepSeekV3")?;
    let v_head_dim = config
        .v_head_dim
        .context("missing v_head_dim for DeepSeekV3")?;
    let kv_lora_rank = config
        .kv_lora_rank
        .context("missing kv_lora_rank for DeepSeekV3")?;

    for i in 0..config.num_hidden_layers {
        let prefix = format!("{}.layers.{i}.self_attn", prefixes.model);
        if !tensors.contains_key(&format!("{prefix}.kv_b_proj.weight"))
            || tensors.contains_key(&format!("{prefix}.embed_q.weight"))
        {
            continue;
        }

        let (group_size, bits) = quant_params_for(
            config_json,
            &format!("{prefix}.kv_b_proj"),
            default_group_size,
            default_bits,
        );
        let weight = tensors
            .get(&format!("{prefix}.kv_b_proj.weight"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.weight"))?;
        let scales = tensors
            .get(&format!("{prefix}.kv_b_proj.scales"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.scales"))?;
        let biases = tensors
            .get(&format!("{prefix}.kv_b_proj.biases"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.biases"))?;
        let dense = dequantize(&weight, &scales, &biases, group_size, bits)?;
        let dense = dense.reshape(&[num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank])?;
        let wk = dense
            .index((std::ops::RangeFull, ..qk_nope_head_dim, std::ops::RangeFull))
            .transpose_axes(&[0, 2, 1])?;
        let wv = dense.index((std::ops::RangeFull, qk_nope_head_dim.., std::ops::RangeFull));
        let (wk_q, wk_s, wk_b) = quantize_stacked_weights(&wk, group_size, bits)?;
        let (wv_q, wv_s, wv_b) = quantize_stacked_weights(&wv, group_size, bits)?;
        tensors.insert(format!("{prefix}.embed_q.weight"), wk_q);
        tensors.insert(format!("{prefix}.embed_q.scales"), wk_s);
        tensors.insert(format!("{prefix}.embed_q.biases"), wk_b);
        tensors.insert(format!("{prefix}.unembed_out.weight"), wv_q);
        tensors.insert(format!("{prefix}.unembed_out.scales"), wv_s);
        tensors.insert(format!("{prefix}.unembed_out.biases"), wv_b);
    }

    Ok(())
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
    pub reasoning_family: ReasoningFamily,
    tokenwise_prefill: bool,
    cacheless_generation: bool,
}

impl MlxModel {
    /// Load an MLX model from a directory containing config.json,
    /// tokenizer.json, and model.safetensors.
    pub fn load(dir: &Path) -> Result<Self> {
        tracing::info!("MLX: loading model directory {}", dir.display());
        let config_text =
            std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let config_json: Value =
            serde_json::from_str(&config_text).context("parsing config.json")?;
        ensure_supported_mlx_model(dir, &config_json)?;
        let effective_config_json = normalized_model_config_json(&config_json);
        let config: ModelConfig =
            serde_json::from_value(effective_config_json).context("parsing config.json")?;
        let arch = model_architecture(&config_json);

        let quantized = config.quantization.as_ref();
        let default_group_size = quantized.map(|q| q.group_size).unwrap_or(0);
        let default_bits = quantized.map(|q| q.bits).unwrap_or(0);

        if let Some(qcfg) = quantized {
            tracing::info!(
                "MLX: loading {} layers, hidden={}, heads={}/{}, quant={}bit/g{}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                qcfg.bits,
                qcfg.group_size,
            );
        } else {
            tracing::info!(
                "MLX: loading {} layers, hidden={}, heads={}/{}, dense_dtype={:?}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config_json
                    .get("torch_dtype")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown"),
            );
        }

        let start = std::time::Instant::now();
        let mut tensors = load_all_safetensors(dir)?;
        tracing::info!(
            "MLX: loaded {} tensors in {:.2}s",
            tensors.len(),
            start.elapsed().as_secs_f64()
        );
        let prefixes = tensor_prefixes(&tensors)?;
        if arch.is_deepseek_v3() || arch.is_kimi_linear() {
            transform_deepseek_v3_tensors(
                &mut tensors,
                &prefixes,
                &config,
                &config_json,
                default_group_size,
                default_bits,
            )?;
        }

        let load_qlinear = |prefix: &str| -> Result<QuantizedLinear> {
            let weight = tensors
                .get(&format!("{prefix}.weight"))
                .cloned()
                .with_context(|| format!("missing {prefix}.weight"))?;
            let bias = tensors.get(&format!("{prefix}.bias")).cloned();
            let scales_key = format!("{prefix}.scales");
            let biases_key = format!("{prefix}.biases");
            let has_quantized_tensors =
                tensors.contains_key(&scales_key) && tensors.contains_key(&biases_key);
            let dense_weight_t = if quantized.is_none() || !has_quantized_tensors {
                Some(weight.transpose_axes(&[1, 0])?)
            } else {
                let (group_size, bits) =
                    quant_params_for(&config_json, prefix, default_group_size, default_bits);
                let scales = tensors
                    .get(&scales_key)
                    .cloned()
                    .with_context(|| format!("missing {scales_key}"))?;
                let biases = tensors
                    .get(&biases_key)
                    .cloned()
                    .with_context(|| format!("missing {biases_key}"))?;
                // Some Gemma4 MLX checkpoints use 5-bit weights for a subset of MLP
                // blocks, and current Metal qmm kernels are missing for that shape.
                // Allow forcing the dense path for diagnostics when comparing qmm
                // behavior against dequantized matmul on the same artifact.
                if bits == 5 || force_dense_quantized_linear() {
                    Some(cpu_dense_weight_t(
                        &weight, &scales, &biases, group_size, bits,
                    )?)
                } else {
                    None
                }
            };
            let (group_size, bits) = if quantized.is_some() && has_quantized_tensors {
                quant_params_for(&config_json, prefix, default_group_size, default_bits)
            } else {
                (0, 0)
            };
            let scales = tensors
                .get(&scales_key)
                .cloned()
                .unwrap_or_else(|| array!(0.0f32));
            let biases = tensors
                .get(&biases_key)
                .cloned()
                .unwrap_or_else(|| array!(0.0f32));
            Ok(QuantizedLinear {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                dense_weight_t,
            })
        };

        let load_multi_linear = |prefix: &str| -> Result<QuantizedMultiLinear> {
            let (group_size, bits) =
                quant_params_for(&config_json, prefix, default_group_size, default_bits);
            Ok(QuantizedMultiLinear {
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
                group_size,
                bits,
            })
        };

        let load_switch_linear = |prefix: &str| -> Result<QuantizedSwitchLinear> {
            let (group_size, bits) =
                quant_params_for(&config_json, prefix, default_group_size, default_bits);
            Ok(QuantizedSwitchLinear {
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

        let load_lfm2_conv_weight = |prefix: &str| -> Result<Array> {
            let weight = tensors
                .get(&format!("{prefix}.weight"))
                .cloned()
                .with_context(|| format!("missing {prefix}.weight"))?;
            if weight.ndim() == 3 && weight.shape()[2] > weight.shape()[1] {
                Ok(weight.transpose_axes(&[0, 2, 1])?)
            } else {
                Ok(weight)
            }
        };

        let (embed_group_size, embed_bits) = quant_params_for(
            &config_json,
            &format!("{}.embed_tokens", prefixes.model),
            default_group_size,
            default_bits,
        );
        let embed_weight = tensors
            .get(&format!("{}.embed_tokens.weight", prefixes.model))
            .cloned()
            .with_context(|| format!("missing {}.embed_tokens.weight", prefixes.model))?;
        let embed_scales = tensors
            .get(&format!("{}.embed_tokens.scales", prefixes.model))
            .cloned()
            .unwrap_or_else(|| array!(0.0f32));
        let embed_biases = tensors
            .get(&format!("{}.embed_tokens.biases", prefixes.model))
            .cloned()
            .unwrap_or_else(|| array!(0.0f32));
        let embed_dense_weight = quantized.is_none().then(|| embed_weight.clone());
        let embed_dense_weight_t = if quantized.is_none() {
            Some(embed_weight.transpose_axes(&[1, 0])?)
        } else {
            None
        };
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
                default_group_size,
                default_bits,
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
                    .unwrap_or_else(|| array!(0.0f32)),
                biases: tensors
                    .get(&format!("{}.embed_tokens_per_layer.biases", prefixes.model))
                    .cloned()
                    .unwrap_or_else(|| array!(0.0f32)),
                group_size,
                bits,
                dense_weight: quantized.is_none().then(|| {
                    tensors[&format!("{}.embed_tokens_per_layer.weight", prefixes.model)].clone()
                }),
                dense_weight_t: if quantized.is_none() {
                    Some(
                        tensors[&format!("{}.embed_tokens_per_layer.weight", prefixes.model)]
                            .transpose_axes(&[1, 0])?,
                    )
                } else {
                    None
                },
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
            weight: if arch.is_lfm2() {
                tensors
                    .get(&format!("{}.embedding_norm.weight", prefixes.model))
                    .cloned()
                    .with_context(|| format!("missing {}.embedding_norm.weight", prefixes.model))?
            } else {
                tensors
                    .get(&format!("{}.norm.weight", prefixes.model))
                    .cloned()
                    .with_context(|| format!("missing {}.norm.weight", prefixes.model))?
            },
            eps: config.block_norm_eps.unwrap_or(config.rms_norm_eps),
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
            if arch.is_deepseek_v3() {
                let qk_nope_head_dim = config
                    .qk_nope_head_dim
                    .context("missing qk_nope_head_dim for DeepSeekV3")?;
                let qk_rope_head_dim = config
                    .qk_rope_head_dim
                    .context("missing qk_rope_head_dim for DeepSeekV3")?;
                let kv_lora_rank = config
                    .kv_lora_rank
                    .context("missing kv_lora_rank for DeepSeekV3")?;
                let v_head_dim = config
                    .v_head_dim
                    .context("missing v_head_dim for DeepSeekV3")?;
                let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
                let is_moe_layer = config.n_routed_experts.is_some()
                    && (i >= config.first_k_dense_replace.unwrap_or(0))
                    && (i % config.moe_layer_freq.unwrap_or(1) == 0);
                let shared_intermediate = config
                    .n_shared_experts
                    .zip(config.moe_intermediate_size)
                    .map(|(n_shared, hidden)| n_shared * hidden);
                let mlp_kind = if is_moe_layer {
                    MlpKind::DeepseekV3MoE(DeepseekV3MoE {
                        switch_gate_proj: load_switch_linear(&format!(
                            "{p}.mlp.switch_mlp.gate_proj"
                        ))?,
                        switch_up_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.up_proj"))?,
                        switch_down_proj: load_switch_linear(&format!(
                            "{p}.mlp.switch_mlp.down_proj"
                        ))?,
                        gate_weight: tensors
                            .get(&format!("{p}.mlp.gate.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.mlp.gate.weight"))?,
                        gate_bias: tensors
                            .get(&format!("{p}.mlp.gate.e_score_correction_bias"))
                            .cloned()
                            .with_context(|| {
                                format!("missing {p}.mlp.gate.e_score_correction_bias")
                            })?,
                        top_k: config.num_experts_per_tok.unwrap_or(1),
                        n_group: config.n_group.unwrap_or(1),
                        topk_group: config.topk_group.unwrap_or(1),
                        routed_scaling_factor: config.routed_scaling_factor.unwrap_or(1.0),
                        norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
                        shared_experts: shared_intermediate
                            .map(|_intermediate_size| -> Result<MLP> {
                                Ok(MLP {
                                    gate_up_proj: None,
                                    gate_proj: Some(load_qlinear(&format!(
                                        "{p}.mlp.shared_experts.gate_proj"
                                    ))?),
                                    up_proj: Some(load_qlinear(&format!(
                                        "{p}.mlp.shared_experts.up_proj"
                                    ))?),
                                    down_proj: load_qlinear(&format!(
                                        "{p}.mlp.shared_experts.down_proj"
                                    ))?,
                                    activation: Activation::Silu,
                                })
                            })
                            .transpose()?,
                    })
                } else {
                    MlpKind::Dense(MLP {
                        gate_up_proj: None,
                        gate_proj: Some(load_qlinear(&format!("{p}.mlp.gate_proj"))?),
                        up_proj: Some(load_qlinear(&format!("{p}.mlp.up_proj"))?),
                        down_proj: load_qlinear(&format!("{p}.mlp.down_proj"))?,
                        activation: Activation::Silu,
                    })
                };

                layers.push(Layer {
                    attn: AttentionKind::DeepseekV3(DeepseekV3Attention {
                        q_proj: if config.q_lora_rank.is_some() {
                            None
                        } else {
                            Some(load_qlinear(&format!("{p}.self_attn.q_proj"))?)
                        },
                        q_a_proj: config
                            .q_lora_rank
                            .is_some()
                            .then(|| load_qlinear(&format!("{p}.self_attn.q_a_proj")))
                            .transpose()?,
                        q_a_layernorm: tensors
                            .get(&format!("{p}.self_attn.q_a_layernorm.weight"))
                            .cloned()
                            .map(|weight| RMSNorm {
                                weight,
                                eps: 1e-6,
                                add_unit_offset: false,
                            }),
                        q_b_proj: config
                            .q_lora_rank
                            .is_some()
                            .then(|| load_qlinear(&format!("{p}.self_attn.q_b_proj")))
                            .transpose()?,
                        kv_a_proj_with_mqa: load_qlinear(&format!(
                            "{p}.self_attn.kv_a_proj_with_mqa"
                        ))?,
                        kv_a_layernorm: RMSNorm {
                            weight: tensors
                                .get(&format!("{p}.self_attn.kv_a_layernorm.weight"))
                                .cloned()
                                .with_context(|| {
                                    format!("missing {p}.self_attn.kv_a_layernorm.weight")
                                })?,
                            eps: 1e-6,
                            add_unit_offset: false,
                        },
                        embed_q: load_multi_linear(&format!("{p}.self_attn.embed_q"))?,
                        unembed_out: load_multi_linear(&format!("{p}.self_attn.unembed_out"))?,
                        o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
                        num_heads: config.num_attention_heads,
                        q_head_dim,
                        qk_rope_head_dim,
                        qk_nope_head_dim,
                        kv_lora_rank,
                        v_head_dim,
                        scale: 1.0 / (q_head_dim as f32).sqrt(),
                        rope_theta: config.rope_theta,
                    }),
                    mlp: mlp_kind,
                    attn_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.input_layernorm.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
                        eps: config.rms_norm_eps,
                        add_unit_offset: false,
                    },
                    attn_out_norm: None,
                    mlp_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.post_attention_layernorm.weight"))
                            .cloned()
                            .with_context(|| {
                                format!("missing {p}.post_attention_layernorm.weight")
                            })?,
                        eps: config.rms_norm_eps,
                        add_unit_offset: false,
                    },
                    mlp_out_norm: None,
                    per_layer_input: None,
                    layer_scalar: None,
                });
                continue;
            }
            if arch.is_lfm2() {
                let full_attn_idxs = config
                    .full_attn_idxs
                    .as_ref()
                    .with_context(|| format!("missing full_attn_idxs for LFM2 layer {}", i))?;
                let is_attention_layer = full_attn_idxs.contains(&i);
                let operator = if is_attention_layer {
                    AttentionKind::Standard(Attention {
                        q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
                        k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
                        v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
                        o_proj: load_qlinear(&format!("{p}.self_attn.out_proj"))?,
                        q_norm: tensors
                            .get(&format!("{p}.self_attn.q_layernorm.weight"))
                            .cloned()
                            .map(|weight| RMSNorm {
                                weight,
                                eps: config.block_norm_eps.unwrap_or(config.rms_norm_eps),
                                add_unit_offset: false,
                            }),
                        k_norm: tensors
                            .get(&format!("{p}.self_attn.k_layernorm.weight"))
                            .cloned()
                            .map(|weight| RMSNorm {
                                weight,
                                eps: config.block_norm_eps.unwrap_or(config.rms_norm_eps),
                                add_unit_offset: false,
                            }),
                        v_norm: None,
                        num_heads: config.num_attention_heads,
                        num_kv_heads: config.num_key_value_heads,
                        head_dim,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                        attn_logit_softcapping: None,
                        rope_dim: head_dim,
                        rope_theta: config.rope_theta,
                        window_size: None,
                        kv_shared_source: None,
                    })
                } else {
                    AttentionKind::Lfm2ShortConv(Lfm2ShortConv {
                        conv_weight: load_lfm2_conv_weight(&format!("{p}.conv.conv"))?,
                        in_proj: load_qlinear(&format!("{p}.conv.in_proj"))?,
                        out_proj: load_qlinear(&format!("{p}.conv.out_proj"))?,
                        hidden_size: config.hidden_size,
                        conv_l_cache: config.conv_l_cache.unwrap_or(3),
                    })
                };

                layers.push(Layer {
                    attn: operator,
                    mlp: MlpKind::Dense(MLP {
                        gate_up_proj: None,
                        gate_proj: Some(load_qlinear(&format!("{p}.feed_forward.w1"))?),
                        up_proj: Some(load_qlinear(&format!("{p}.feed_forward.w3"))?),
                        down_proj: load_qlinear(&format!("{p}.feed_forward.w2"))?,
                        activation: Activation::Silu,
                    }),
                    attn_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.operator_norm.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.operator_norm.weight"))?,
                        eps: config.block_norm_eps.unwrap_or(config.rms_norm_eps),
                        add_unit_offset: false,
                    },
                    attn_out_norm: None,
                    mlp_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.ffn_norm.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.ffn_norm.weight"))?,
                        eps: config.block_norm_eps.unwrap_or(config.rms_norm_eps),
                        add_unit_offset: false,
                    },
                    mlp_out_norm: None,
                    per_layer_input: None,
                    layer_scalar: None,
                });
                continue;
            }
            if arch.is_kimi_linear() {
                let linear_cfg = config
                    .linear_attn_config
                    .as_ref()
                    .context("missing linear_attn_config for Kimi Linear")?;
                let is_linear_layer = linear_cfg.kda_layers.contains(&(i + 1));
                let projection_dim = linear_cfg.num_heads * linear_cfg.head_dim;
                let is_moe_layer = config.n_routed_experts.unwrap_or(0) > 0
                    && (i >= config.first_k_dense_replace.unwrap_or(0))
                    && (i % config.moe_layer_freq.unwrap_or(1) == 0);
                let mlp = if is_moe_layer {
                    MlpKind::DeepseekV3MoE(DeepseekV3MoE {
                        switch_gate_proj: load_switch_linear(&format!(
                            "{p}.mlp.switch_mlp.gate_proj"
                        ))?,
                        switch_up_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.up_proj"))?,
                        switch_down_proj: load_switch_linear(&format!(
                            "{p}.mlp.switch_mlp.down_proj"
                        ))?,
                        gate_weight: tensors
                            .get(&format!("{p}.mlp.gate.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.mlp.gate.weight"))?,
                        gate_bias: tensors
                            .get(&format!("{p}.mlp.e_score_correction_bias"))
                            .cloned()
                            .with_context(|| format!("missing {p}.mlp.e_score_correction_bias"))?,
                        top_k: config.num_experts_per_tok.unwrap_or(1),
                        n_group: config.n_group.unwrap_or(1),
                        topk_group: config.topk_group.unwrap_or(1),
                        routed_scaling_factor: config.routed_scaling_factor.unwrap_or(1.0),
                        norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
                        shared_experts: config
                            .n_shared_experts
                            .filter(|n| *n > 0)
                            .map(|_| -> Result<MLP> {
                                Ok(MLP {
                                    gate_up_proj: None,
                                    gate_proj: Some(load_qlinear(&format!(
                                        "{p}.mlp.shared_experts.gate_proj"
                                    ))?),
                                    up_proj: Some(load_qlinear(&format!(
                                        "{p}.mlp.shared_experts.up_proj"
                                    ))?),
                                    down_proj: load_qlinear(&format!(
                                        "{p}.mlp.shared_experts.down_proj"
                                    ))?,
                                    activation: Activation::Silu,
                                })
                            })
                            .transpose()?,
                    })
                } else {
                    MlpKind::Dense(MLP {
                        gate_up_proj: None,
                        gate_proj: Some(load_qlinear(&format!("{p}.mlp.gate_proj"))?),
                        up_proj: Some(load_qlinear(&format!("{p}.mlp.up_proj"))?),
                        down_proj: load_qlinear(&format!("{p}.mlp.down_proj"))?,
                        activation: Activation::Silu,
                    })
                };

                let attn = if is_linear_layer {
                    AttentionKind::KimiDelta(KimiDeltaAttention {
                        q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
                        k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
                        v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
                        q_conv: KimiShortConv {
                            conv_weight: load_lfm2_conv_weight(&format!(
                                "{p}.self_attn.q_conv.conv"
                            ))?,
                            kernel_size: linear_cfg.short_conv_kernel_size.unwrap_or(4),
                            channels: projection_dim,
                        },
                        k_conv: KimiShortConv {
                            conv_weight: load_lfm2_conv_weight(&format!(
                                "{p}.self_attn.k_conv.conv"
                            ))?,
                            kernel_size: linear_cfg.short_conv_kernel_size.unwrap_or(4),
                            channels: projection_dim,
                        },
                        v_conv: KimiShortConv {
                            conv_weight: load_lfm2_conv_weight(&format!(
                                "{p}.self_attn.v_conv.conv"
                            ))?,
                            kernel_size: linear_cfg.short_conv_kernel_size.unwrap_or(4),
                            channels: projection_dim,
                        },
                        f_a_proj: load_qlinear(&format!("{p}.self_attn.f_a_proj"))?,
                        f_b_proj: load_qlinear(&format!("{p}.self_attn.f_b_proj"))?,
                        b_proj: load_qlinear(&format!("{p}.self_attn.b_proj"))?,
                        g_a_proj: load_qlinear(&format!("{p}.self_attn.g_a_proj"))?,
                        g_b_proj: load_qlinear(&format!("{p}.self_attn.g_b_proj"))?,
                        a_log: tensors
                            .get(&format!("{p}.self_attn.A_log"))
                            .cloned()
                            .with_context(|| format!("missing {p}.self_attn.A_log"))?,
                        dt_bias: tensors
                            .get(&format!("{p}.self_attn.dt_bias"))
                            .cloned()
                            .with_context(|| format!("missing {p}.self_attn.dt_bias"))?,
                        o_norm: RMSNorm {
                            weight: tensors
                                .get(&format!("{p}.self_attn.o_norm.weight"))
                                .cloned()
                                .with_context(|| format!("missing {p}.self_attn.o_norm.weight"))?,
                            eps: config.rms_norm_eps,
                            add_unit_offset: false,
                        },
                        o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
                        num_heads: linear_cfg.num_heads,
                        head_dim: linear_cfg.head_dim,
                        scale: (linear_cfg.head_dim as f32).powf(-0.5),
                    })
                } else {
                    let qk_nope_head_dim = config
                        .qk_nope_head_dim
                        .context("missing qk_nope_head_dim for Kimi Linear MLA")?;
                    let qk_rope_head_dim = config
                        .qk_rope_head_dim
                        .context("missing qk_rope_head_dim for Kimi Linear MLA")?;
                    let kv_lora_rank = config
                        .kv_lora_rank
                        .context("missing kv_lora_rank for Kimi Linear MLA")?;
                    let v_head_dim = config
                        .v_head_dim
                        .context("missing v_head_dim for Kimi Linear MLA")?;
                    AttentionKind::KimiMla(KimiMlaAttention {
                        q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
                        kv_a_proj_with_mqa: load_qlinear(&format!(
                            "{p}.self_attn.kv_a_proj_with_mqa"
                        ))?,
                        kv_a_layernorm: RMSNorm {
                            weight: tensors
                                .get(&format!("{p}.self_attn.kv_a_layernorm.weight"))
                                .cloned()
                                .with_context(|| {
                                    format!("missing {p}.self_attn.kv_a_layernorm.weight")
                                })?,
                            eps: config.rms_norm_eps,
                            add_unit_offset: false,
                        },
                        embed_q: load_multi_linear(&format!("{p}.self_attn.embed_q"))?,
                        unembed_out: load_multi_linear(&format!("{p}.self_attn.unembed_out"))?,
                        o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
                        num_heads: config.num_attention_heads,
                        q_head_dim: qk_nope_head_dim + qk_rope_head_dim,
                        qk_rope_head_dim,
                        qk_nope_head_dim,
                        kv_lora_rank,
                        v_head_dim,
                        scale: 1.0 / ((qk_nope_head_dim + qk_rope_head_dim) as f32).sqrt(),
                    })
                };

                layers.push(Layer {
                    attn,
                    mlp,
                    attn_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.input_layernorm.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
                        eps: config.rms_norm_eps,
                        add_unit_offset: false,
                    },
                    attn_out_norm: None,
                    mlp_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.post_attention_layernorm.weight"))
                            .cloned()
                            .with_context(|| {
                                format!("missing {p}.post_attention_layernorm.weight")
                            })?,
                        eps: config.rms_norm_eps,
                        add_unit_offset: false,
                    },
                    mlp_out_norm: None,
                    per_layer_input: None,
                    layer_scalar: None,
                });
                continue;
            }
            if arch.is_gpt_oss() {
                let window_size = if matches!(layer_type, Some("sliding_attention")) {
                    Some(
                        config
                            .sliding_window
                            .context("missing sliding_window for gpt-oss sliding layer")?,
                    )
                } else {
                    None
                };
                layers.push(Layer {
                    attn: AttentionKind::Standard(Attention {
                        q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
                        k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
                        v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
                        o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
                        q_norm: None,
                        k_norm: None,
                        v_norm: None,
                        num_heads: config.num_attention_heads,
                        num_kv_heads: config.num_key_value_heads,
                        head_dim,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                        attn_logit_softcapping: None,
                        rope_dim: head_dim,
                        rope_theta: config.rope_theta,
                        window_size,
                        kv_shared_source: None,
                    }),
                    mlp: MlpKind::GptOssMoE(GptOssMoE {
                        switch_gate_proj: load_switch_linear(&format!(
                            "{p}.mlp.experts.gate_proj"
                        ))?,
                        switch_up_proj: load_switch_linear(&format!("{p}.mlp.experts.up_proj"))?,
                        switch_down_proj: load_switch_linear(&format!(
                            "{p}.mlp.experts.down_proj"
                        ))?,
                        router: load_qlinear(&format!("{p}.mlp.router"))?,
                        top_k: config.num_experts_per_tok.unwrap_or(1),
                    }),
                    attn_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.input_layernorm.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
                        eps: config.rms_norm_eps,
                        add_unit_offset: false,
                    },
                    attn_out_norm: None,
                    mlp_in_norm: RMSNorm {
                        weight: tensors
                            .get(&format!("{p}.post_attention_layernorm.weight"))
                            .cloned()
                            .with_context(|| {
                                format!("missing {p}.post_attention_layernorm.weight")
                            })?,
                        eps: config.rms_norm_eps,
                        add_unit_offset: false,
                    },
                    mlp_out_norm: None,
                    per_layer_input: None,
                    layer_scalar: None,
                });
                continue;
            }
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
            } else if arch.is_glm4() {
                ((layer_head_dim as f32) * config.partial_rotary_factor.unwrap_or(1.0)).round()
                    as i32
            } else {
                layer_head_dim
            };
            let gemma3_global_layer = arch.is_gemma3()
                && config
                    .sliding_window_pattern
                    .map(|pattern| pattern > 0 && ((i + 1) % pattern == 0))
                    .unwrap_or(false);
            let rope_theta = if arch.is_gemma3() && !gemma3_global_layer {
                config.rope_local_base_freq.unwrap_or(10_000.0)
            } else {
                rope_parameters
                    .and_then(|params| params.rope_theta)
                    .unwrap_or(config.rope_theta)
            };
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
            let mlp_in_norm_key = if arch.is_glm4() {
                format!("{p}.post_attention_layernorm.weight")
            } else if arch.is_gemma2() || arch.is_gemma3() || arch.is_gemma4() {
                format!("{p}.pre_feedforward_layernorm.weight")
            } else {
                format!("{p}.post_attention_layernorm.weight")
            };
            layers.push(Layer {
                attn: AttentionKind::Standard(Attention {
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
                    window_size: if arch.is_gemma3() && !gemma3_global_layer {
                        config.sliding_window
                    } else {
                        None
                    },
                    kv_shared_source,
                }),
                mlp: MlpKind::Dense(MLP {
                    gate_up_proj: tensors
                        .contains_key(&format!("{p}.mlp.gate_up_proj.weight"))
                        .then(|| load_qlinear(&format!("{p}.mlp.gate_up_proj")))
                        .transpose()?,
                    gate_proj: tensors
                        .contains_key(&format!("{p}.mlp.gate_proj.weight"))
                        .then(|| load_qlinear(&format!("{p}.mlp.gate_proj")))
                        .transpose()?,
                    up_proj: tensors
                        .contains_key(&format!("{p}.mlp.up_proj.weight"))
                        .then(|| load_qlinear(&format!("{p}.mlp.up_proj")))
                        .transpose()?,
                    down_proj: load_qlinear(&format!("{p}.mlp.down_proj"))?,
                    activation: activation,
                }),
                attn_in_norm: RMSNorm {
                    weight: tensors
                        .get(&format!("{p}.input_layernorm.weight"))
                        .cloned()
                        .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
                    eps: config.rms_norm_eps,
                    add_unit_offset: arch.uses_gemma_norm_offset(),
                },
                attn_out_norm: (arch.is_glm4()
                    || arch.is_gemma2()
                    || arch.is_gemma3()
                    || arch.is_gemma4())
                .then(|| -> Result<RMSNorm> {
                    let key = if arch.is_glm4() {
                        format!("{p}.post_self_attn_layernorm.weight")
                    } else {
                        format!("{p}.post_attention_layernorm.weight")
                    };
                    Ok(RMSNorm {
                        weight: tensors
                            .get(&key)
                            .cloned()
                            .with_context(|| format!("missing {key}"))?,
                        eps: config.rms_norm_eps,
                        add_unit_offset: arch.uses_gemma_norm_offset(),
                    })
                })
                .transpose()?,
                mlp_in_norm: RMSNorm {
                    weight: tensors.get(&mlp_in_norm_key).cloned().with_context(|| {
                        if arch.is_gemma2() || arch.is_gemma3() {
                            format!("missing {p}.pre_feedforward_layernorm.weight")
                        } else if arch.is_glm4() {
                            format!("missing {p}.post_attention_layernorm.weight")
                        } else {
                            format!("missing {p}.post_attention_layernorm.weight")
                        }
                    })?,
                    eps: config.rms_norm_eps,
                    add_unit_offset: arch.uses_gemma_norm_offset(),
                },
                mlp_out_norm: (arch.is_glm4()
                    || arch.is_gemma2()
                    || arch.is_gemma3()
                    || arch.is_gemma4())
                .then(|| -> Result<RMSNorm> {
                    let key = if arch.is_glm4() {
                        format!("{p}.post_mlp_layernorm.weight")
                    } else {
                        format!("{p}.post_feedforward_layernorm.weight")
                    };
                    Ok(RMSNorm {
                        weight: tensors
                            .get(&key)
                            .cloned()
                            .with_context(|| format!("missing {key}"))?,
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
            reasoning_family: reasoning_family(&config_json),
            tokenwise_prefill: arch.is_gemma2() || arch.is_gemma4(),
            cacheless_generation: arch.is_gemma2()
                || arch.is_gpt_oss()
                || arch.is_kimi_linear()
                || arch.is_lfm2(),
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
                .kv_shared_source()
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
    DeepseekV3,
    GptOss,
    KimiLinear,
    Lfm2,
    Glm4,
    Gemma2,
    Gemma3,
    Gemma4,
}

impl ModelArchitecture {
    fn is_deepseek_v3(self) -> bool {
        matches!(self, Self::DeepseekV3)
    }

    fn is_glm4(self) -> bool {
        matches!(self, Self::Glm4)
    }

    fn is_gpt_oss(self) -> bool {
        matches!(self, Self::GptOss)
    }

    fn is_kimi_linear(self) -> bool {
        matches!(self, Self::KimiLinear)
    }

    fn is_lfm2(self) -> bool {
        matches!(self, Self::Lfm2)
    }

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

    if model_type.starts_with("glm4") {
        ModelArchitecture::Glm4
    } else if model_type.starts_with("gpt_oss") {
        ModelArchitecture::GptOss
    } else if model_type.starts_with("kimi_linear") {
        ModelArchitecture::KimiLinear
    } else if model_type.starts_with("deepseek_v3")
        || model_type.starts_with("kimi_k2")
        || model_type.starts_with("kimi_k25")
    {
        ModelArchitecture::DeepseekV3
    } else if model_type.starts_with("lfm2") {
        ModelArchitecture::Lfm2
    } else if model_type.starts_with("gemma4") {
        ModelArchitecture::Gemma4
    } else if model_type.starts_with("gemma2") {
        ModelArchitecture::Gemma2
    } else if model_type.starts_with("gemma3") {
        ModelArchitecture::Gemma3
    } else {
        ModelArchitecture::LlamaLike
    }
}

fn reasoning_family(config: &Value) -> ReasoningFamily {
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
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();

    if model_type == "qwen3" || architectures.iter().any(|value| value.contains("qwen3")) {
        return ReasoningFamily::Qwen3;
    }
    if model_type.starts_with("glm") || architectures.iter().any(|value| value.contains("glm")) {
        return ReasoningFamily::Glm;
    }
    if model_type == "gpt_oss" || architectures.iter().any(|value| value.contains("gptoss")) {
        return ReasoningFamily::GptOss;
    }
    if model_type.starts_with("lfm2") || architectures.iter().any(|value| value.contains("lfm2")) {
        return ReasoningFamily::Lfm2;
    }
    if model_type.contains("kimi") || architectures.iter().any(|value| value.contains("kimi")) {
        return ReasoningFamily::Kimi;
    }
    ReasoningFamily::None
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
        "moe_intermediate_size",
        "n_shared_experts",
        "n_routed_experts",
        "routed_scaling_factor",
        "kv_lora_rank",
        "q_lora_rank",
        "qk_rope_head_dim",
        "v_head_dim",
        "qk_nope_head_dim",
        "topk_method",
        "norm_topk_prob",
        "n_group",
        "topk_group",
        "num_experts_per_tok",
        "moe_layer_freq",
        "first_k_dense_replace",
        "attention_bias",
        "num_kv_shared_layers",
        "layer_types",
        "rope_parameters",
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "sliding_window",
        "cache_implementation",
        "conv_bias",
        "conv_L_cache",
        "block_dim",
        "block_ff_dim",
        "block_multiple_of",
        "block_ffn_dim_multiplier",
        "block_auto_adjust_ff_dim",
        "full_attn_idxs",
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

fn normalized_model_config_json(config: &Value) -> Value {
    let mut normalized = effective_text_config_json(config);
    let Some(object) = normalized.as_object_mut() else {
        return normalized;
    };

    if !object.contains_key("hidden_activation") {
        if let Some(value) = object.get("hidden_act").cloned() {
            object.insert("hidden_activation".to_string(), value);
        }
    }
    object.remove("hidden_act");

    normalized
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
                | "glm4"
                | "deepseek_v3"
                | "lfm2"
                | "qwen2"
                | "qwen3"
                | "gpt_oss"
                | "kimi_linear"
                | "gemma2"
                | "gemma3"
                | "gemma3_text"
                | "gemma4"
                | "gemma4_text"
                | "glm4forcausallm"
                | "deepseekv3forcausallm"
                | "lfm2forcausallm"
                | "llamaforcausallm"
                | "qwen2forcausallm"
                | "qwen3forcausallm"
                | "gptossforcausallm"
                | "kimilinearforcausallm"
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
        "unsupported MLX model architecture in {} (model_type={}, architectures={}); supported MLX models currently cover Llama/DeepSeekV3/GPT-OSS/Kimi-Linear/LFM2/GLM4/Qwen/Gemma2/Gemma3/Gemma4-style safetensors checkpoints",
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
        let deepseek: Value = serde_json::json!({
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"]
        });
        let kimi: Value = serde_json::json!({
            "model_type": "kimi_k2",
            "architectures": ["DeepseekV3ForCausalLM"]
        });
        let glm4: Value = serde_json::json!({
            "model_type": "glm4",
            "architectures": ["Glm4ForCausalLM"]
        });
        let lfm2: Value = serde_json::json!({
            "model_type": "lfm2",
            "architectures": ["Lfm2ForCausalLM"]
        });
        let qwen: Value = serde_json::json!({
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"]
        });
        let gpt_oss: Value = serde_json::json!({
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"]
        });
        let kimi_linear: Value = serde_json::json!({
            "model_type": "kimi_linear",
            "architectures": ["KimiLinearForCausalLM"]
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

        assert!(config_supports_mlx(&deepseek));
        assert!(config_supports_mlx(&kimi));
        assert!(config_supports_mlx(&glm4));
        assert!(config_supports_mlx(&lfm2));
        assert!(config_supports_mlx(&qwen));
        assert!(config_supports_mlx(&gpt_oss));
        assert!(config_supports_mlx(&kimi_linear));
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
        let lfm2: Value = serde_json::json!({
            "model_type": "lfm2_moe",
            "architectures": ["Lfm2MoeForCausalLM"]
        });

        assert!(!config_supports_mlx(&glm));
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
    fn model_architecture_detects_glm4() {
        let config = serde_json::json!({
            "model_type": "glm4",
            "architectures": ["Glm4ForCausalLM"]
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::Glm4);
    }

    #[test]
    fn model_architecture_detects_lfm2() {
        let config = serde_json::json!({
            "model_type": "lfm2",
            "architectures": ["Lfm2ForCausalLM"]
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::Lfm2);
    }

    #[test]
    fn model_architecture_detects_deepseek_v3() {
        let config = serde_json::json!({
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"]
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::DeepseekV3);
    }

    #[test]
    fn model_architecture_detects_gpt_oss() {
        let config = serde_json::json!({
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"]
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::GptOss);
    }

    #[test]
    fn model_architecture_detects_kimi_linear() {
        let config = serde_json::json!({
            "model_type": "kimi_linear",
            "architectures": ["KimiLinearForCausalLM"]
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::KimiLinear);
    }

    #[test]
    fn model_architecture_detects_kimi_k2_as_deepseek_v3_runtime() {
        let config = serde_json::json!({
            "model_type": "kimi_k25",
            "architectures": ["DeepseekV3ForCausalLM"]
        });

        assert_eq!(model_architecture(&config), ModelArchitecture::DeepseekV3);
    }

    #[test]
    fn glm4_config_parses_partial_rotary_factor() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "glm4",
            "hidden_size": 4096,
            "num_hidden_layers": 40,
            "intermediate_size": 13696,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 151552,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": 151329
        }))
        .unwrap();

        assert_eq!(config.partial_rotary_factor, Some(0.5));
        assert_eq!(config.head_dim, Some(128));
    }

    #[test]
    fn deepseek_v3_config_parses_moe_and_mla_fields() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "n_shared_experts": 1,
            "n_routed_experts": 256,
            "routed_scaling_factor": 2.5,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_group": 8,
            "topk_group": 4,
            "num_experts_per_tok": 8,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 3,
            "vocab_size": 129280,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "max_position_embeddings": 163840,
            "tie_word_embeddings": false,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": [0, 1]
        }))
        .unwrap();

        assert_eq!(config.moe_intermediate_size, Some(2048));
        assert_eq!(config.n_routed_experts, Some(256));
        assert_eq!(config.kv_lora_rank, Some(512));
        assert_eq!(config.q_lora_rank, Some(1536));
        assert_eq!(config.qk_rope_head_dim, Some(64));
        assert_eq!(config.qk_nope_head_dim, Some(128));
        assert_eq!(config.v_head_dim, Some(128));
        assert_eq!(config.n_group, Some(8));
        assert_eq!(config.topk_group, Some(4));
        assert_eq!(config.num_experts_per_tok, Some(8));
        assert_eq!(config.first_k_dense_replace, Some(3));
    }

    #[test]
    fn lfm2_config_parses_conv_and_attention_layout() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "lfm2",
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "intermediate_size": 6656,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 65536,
            "rms_norm_eps": 0.00001,
            "max_position_embeddings": 128000,
            "tie_word_embeddings": false,
            "rope_theta": 1000000.0,
            "conv_bias": false,
            "conv_L_cache": 3,
            "block_norm_eps": 0.00001,
            "block_dim": 1024,
            "block_ff_dim": 6656,
            "block_multiple_of": 256,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "eos_token_id": 7
        }))
        .unwrap();

        assert_eq!(config.conv_l_cache, Some(3));
        assert_eq!(
            config.full_attn_idxs.as_deref(),
            Some(&[2, 5, 8, 10, 12, 14][..])
        );
        assert_eq!(config.block_norm_eps, Some(0.00001));
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
    fn gemma3_config_parses_sliding_window_fields() {
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
            "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "cache_implementation": "hybrid",
            "eos_token_id": [1, 106]
        }))
        .unwrap();

        assert_eq!(config.sliding_window, Some(512));
        assert_eq!(config.sliding_window_pattern, Some(6));
        assert_eq!(config.rope_local_base_freq, Some(10000.0));
        assert_eq!(config.query_pre_attn_scalar, Some(256.0));
    }

    #[test]
    fn gemma2_real_hf_config_parses() {
        let raw = serde_json::json!({
            "architectures": ["Gemma2ForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attn_logit_softcapping": 50.0,
            "bos_token_id": 2,
            "cache_implementation": "hybrid",
            "eos_token_id": [1, 107],
            "final_logit_softcapping": 30.0,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2304,
            "initializer_range": 0.02,
            "intermediate_size": 9216,
            "max_position_embeddings": 8192,
            "model_type": "gemma2",
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "pad_token_id": 0,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.42.4",
            "use_cache": true,
            "vocab_size": 256000
        });
        let config: ModelConfig =
            serde_json::from_value(normalized_model_config_json(&raw)).unwrap();

        assert_eq!(config.eos_token_id, vec![1, 107]);
        assert_eq!(config.cache_implementation.as_deref(), Some("hybrid"));
        assert_eq!(
            config.hidden_activation.as_deref(),
            Some("gelu_pytorch_tanh")
        );
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

    #[test]
    fn quantized_repo_can_keep_specific_linear_dense() {
        let config = serde_json::json!({
            "quantization": {
                "group_size": 64,
                "bits": 8
            }
        });
        let prefix = "language_model.model.per_layer_model_projection";
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            format!("{prefix}.weight"),
            array!([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]),
        );

        let weight = tensors[&format!("{prefix}.weight")].clone();
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");
        let has_quantized_tensors =
            tensors.contains_key(&scales_key) && tensors.contains_key(&biases_key);

        assert!(!has_quantized_tensors);

        let dense_weight_t = if !has_quantized_tensors {
            Some(weight.transpose_axes(&[1, 0]).unwrap())
        } else {
            None
        };
        let (group_size, bits) = if has_quantized_tensors {
            quant_params_for(&config, prefix, 64, 8)
        } else {
            (0, 0)
        };

        assert!(dense_weight_t.is_some());
        assert_eq!((group_size, bits), (0, 0));
    }

    #[test]
    fn dense_model_config_is_allowed_without_quantization_block() {
        let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 151936,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 40960,
            "tie_word_embeddings": true,
            "eos_token_id": 151645
        }))
        .unwrap();

        assert!(config.quantization.is_none());
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn dense_embeddings_can_project_logits_through_as_linear() {
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let embedding = QuantizedEmbedding {
            weight: weight.clone(),
            scales: array!(0.0f32),
            biases: array!(0.0f32),
            group_size: 0,
            bits: 0,
            dense_weight: Some(weight.clone()),
            dense_weight_t: Some(weight.transpose_axes(&[1, 0]).unwrap()),
        };
        let hidden = Array::from_slice(&[10.0f32, 20.0], &[1, 1, 2]);

        let logits = embedding.as_linear().forward(&hidden).unwrap();

        assert_eq!(logits.as_slice::<f32>(), &[50.0, 110.0, 170.0]);
    }
}
