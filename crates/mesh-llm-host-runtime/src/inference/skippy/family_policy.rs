use std::path::Path;

use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};
use skippy_topology::{infer_family_capability, FamilyCapabilityRecord, WireDType};

use super::StageWireDType;
use crate::models::gguf::{scan_gguf_compact_meta, GgufCompactMeta};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FamilyPolicy {
    pub(crate) activation_wire_dtype: StageWireDType,
    pub(crate) prefix_cache: FamilyPrefixCachePolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FamilyPrefixCachePolicy {
    Disabled {
        reason: &'static str,
    },
    Auto {
        payload: FamilyPrefixCachePayload,
        min_tokens: u64,
        max_entries: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FamilyPrefixCachePayload {
    ResidentKv,
    KvRecurrent,
}

impl FamilyPrefixCachePayload {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ResidentKv => "resident-kv",
            Self::KvRecurrent => "kv-recurrent",
        }
    }

    fn as_stage_payload(self) -> StageKvCachePayload {
        match self {
            Self::ResidentKv => StageKvCachePayload::ResidentKv,
            Self::KvRecurrent => StageKvCachePayload::KvRecurrent,
        }
    }
}

impl FamilyPolicy {
    pub(crate) fn stage_kv_cache_config_for_stage(
        &self,
        config: &StageConfig,
    ) -> Option<StageKvCacheConfig> {
        match self.prefix_cache {
            FamilyPrefixCachePolicy::Disabled { .. } => None,
            FamilyPrefixCachePolicy::Auto {
                payload,
                min_tokens,
                max_entries,
            } => {
                let max_bytes = derive_stage_cache_max_bytes(config)?;
                // The family policy's `max_entries` is a generous
                // upper bound on cache cardinality. The real ceiling
                // is the unified KV cell pool size: each resident
                // prefix pins `token_count` cells across `stage_layers`
                // in the same `n_ctx` pool the active lanes use. If
                // we let the cache fill to `max_entries` it can
                // starve the active lanes of cells and surface as
                // HTTP 502 `RuntimeError: llama_decode failed`
                // (`decode: failed to find a memory slot`).
                //
                // Cap entries so the cache cannot overcommit the
                // pool. See `derive_max_entries_from_kv_cells` below.
                let bounded_entries =
                    derive_max_entries_from_kv_cells(config, min_tokens, max_entries);
                Some(StageKvCacheConfig {
                    mode: StageKvCacheMode::LookupRecord,
                    payload: payload.as_stage_payload(),
                    max_entries: bounded_entries,
                    max_bytes,
                    min_tokens,
                    shared_prefix_stride_tokens: 128,
                    shared_prefix_record_limit: 2,
                })
            }
        }
    }
}

/// Cap the prefix-cache `max_entries` so resident prefixes cannot
/// exhaust the unified KV cell pool.
///
/// Skippy's stage runtime serves with `kv_unified = true` whenever
/// `lane_count > 1` (patch `0034-Add-shared-execution-lanes-to-skippy-ABI.patch`).
/// In unified mode the KV cache is a single pool of `n_ctx` cells
/// shared across all `n_seq_max` sequences. The resident-prefix cache
/// pins prefixes onto dedicated sequence ids in *the same pool*, so
/// every cached entry consumes cells that the active lanes can no
/// longer use. Without a cap, the family default of 128 entries can
/// accumulate enough pinned prefixes to starve the active lanes,
/// surfacing as HTTP 502 `RuntimeError: llama_decode failed`
/// (`decode: failed to find a memory slot`) after a dozen or so
/// agent-style requests.
///
/// Budget: the cache may use at most half the cell pool. Each entry
/// is at least `min_tokens` cells, so `max_entries ≤ n_ctx / (2 *
/// min_tokens)`. The LRU in `ResidentPrefixCache` evicts when this
/// ceiling is hit. The other half of the pool stays available for
/// the active lanes' fresh prompts.
///
/// Never lifted above the family-policy default; never below 1.
fn derive_max_entries_from_kv_cells(
    config: &StageConfig,
    min_tokens: u64,
    family_default: usize,
) -> usize {
    if min_tokens == 0 {
        return family_default;
    }
    let n_ctx = u64::from(config.ctx_size.max(1));
    let cache_budget_cells = n_ctx / 2;
    let kv_capped = (cache_budget_cells / min_tokens) as usize;
    kv_capped.clamp(1, family_default)
}

pub(crate) fn family_policy_for_stage_config(config: &StageConfig) -> FamilyPolicy {
    [
        config.materialized_path.as_deref(),
        config.source_model_path.as_deref(),
        config.model_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(|path| family_policy_for_gguf_path(path, Some(&config.model_id)))
    .unwrap_or_else(|| family_policy_for_model_id(&config.model_id))
}

pub(crate) fn family_policy_for_model_path(
    path: impl AsRef<Path>,
    model_id: Option<&str>,
) -> FamilyPolicy {
    family_policy_for_gguf_path(path, model_id)
        .unwrap_or_else(|| family_policy_for_model_id(model_id.unwrap_or_default()))
}

fn family_policy_for_gguf_path(
    path: impl AsRef<Path>,
    model_id: Option<&str>,
) -> Option<FamilyPolicy> {
    let meta = scan_gguf_compact_meta(path.as_ref())?;
    Some(family_policy_for_gguf_meta(&meta, model_id))
}

fn family_policy_for_gguf_meta(meta: &GgufCompactMeta, model_id: Option<&str>) -> FamilyPolicy {
    capability_from_gguf_meta(meta, model_id)
        .as_ref()
        .map(family_policy_for_capability)
        .unwrap_or_else(|| family_policy_for_model_id(model_id.unwrap_or_default()))
}

fn family_policy_for_capability(capability: &FamilyCapabilityRecord) -> FamilyPolicy {
    family_policy_for_normalized_family_id(
        capability.family_id.as_str(),
        wire_dtype_from_capability(capability.default_wire_dtype),
    )
}

fn family_policy_for_model_id(model_id: &str) -> FamilyPolicy {
    if model_id.trim().is_empty() {
        return unknown_family_policy();
    }
    infer_family_capability(model_id, 0, 0)
        .as_ref()
        .map(family_policy_for_capability)
        .unwrap_or_else(|| unknown_family_policy_with_wire_dtype(StageWireDType::F16))
}

fn capability_from_gguf_meta(
    meta: &GgufCompactMeta,
    model_id: Option<&str>,
) -> Option<FamilyCapabilityRecord> {
    if let Some(capability) = model_id.and_then(|model_id| {
        infer_family_capability(model_id, meta.layer_count, meta.embedding_size)
    }) {
        return Some(capability);
    }

    if !meta.architecture.trim().is_empty() {
        if let Some(capability) =
            infer_family_capability(&meta.architecture, meta.layer_count, meta.embedding_size)
        {
            return Some(capability);
        }
    }

    None
}

fn family_policy_for_normalized_family_id(
    family_id: &str,
    activation_wire_dtype: StageWireDType,
) -> FamilyPolicy {
    if matches!(family_id, "dream" | "llada" | "llada_moe") {
        return disabled_family_policy(
            activation_wire_dtype,
            "non-causal diffusion family has no resident KV state to cache",
        );
    }

    if let Some(expected) = skippy_topology::STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS
        .iter()
        .find(|expected| expected.family_id == family_id)
    {
        return if expected.recurrent_or_hybrid {
            kv_recurrent_policy(activation_wire_dtype)
        } else {
            resident_kv_policy(activation_wire_dtype)
        };
    }

    match family_id {
        "qwen2" | "qwen3_dense" | "llama" | "deepseek" | "deepseek2" | "deepseek3" | "glm4"
        | "glm4_moe" | "olmo" | "olmo2" | "olmoe" | "gemma2" | "gemma" | "gemma3" | "gemma4"
        | "gemma4_a4b" | "gemma4_e4b" | "glm47_flash" | "minimax_m27" | "qwen2moe" | "qwen3moe"
        | "granite" | "granite_moe" | "hunyuan_dense" | "hunyuan_moe" | "hunyuan_vl"
        | "gptneox" | "bloom" | "stablelm" | "starcoder2" | "mpt" | "phi" | "phi2" | "phimoe"
        | "gpt2" | "mistral" | "internlm2" | "baichuan" | "exaone" | "exaone4" | "cohere2"
        | "command_r" | "falcon" | "qwen2vl" | "qwen3vl" | "deepseek2ocr" | "qwen3vlmoe"
        | "openai_moe" | "ernie4_5_moe" | "llama4" | "mistral4" | "seed_oss" => {
            resident_kv_policy(activation_wire_dtype)
        }
        "qwen3next" | "falcon_h1" | "jamba" | "lfm2" | "mamba" | "mamba2" | "rwkv6" | "rwkv7"
        | "granite_hybrid" | "qwen35" | "qwen35moe" | "nemotron_h_moe" => {
            kv_recurrent_policy(activation_wire_dtype)
        }
        _ => unknown_family_policy_with_wire_dtype(activation_wire_dtype),
    }
}

fn resident_kv_policy(activation_wire_dtype: StageWireDType) -> FamilyPolicy {
    FamilyPolicy {
        activation_wire_dtype,
        prefix_cache: FamilyPrefixCachePolicy::Auto {
            payload: FamilyPrefixCachePayload::ResidentKv,
            min_tokens: 256,
            // Was 128. Real-world OpenAI surface workloads (Goose,
            // OpenCode, pi) record prefixes that average 1.5–2k
            // tokens — much larger than `min_tokens`. 128 entries at
            // ~2k tokens each pins ~256k cells, which exceeds even a
            // 131k-`n_ctx` model's unified KV pool. The active lanes
            // then can't find a slot and the embedded runtime
            // returns HTTP 502
            // `RuntimeError: llama_decode failed`
            // (`decode: failed to find a memory slot`).
            //
            // 16 entries at ~2k tokens ≈ 32k cells; comfortable
            // headroom under any model that gets `kv_unified = true`
            // serving (`lane_count > 1`). The LRU in
            // `ResidentPrefixCache` evicts older entries as new
            // prefixes are recorded, so cache hit rate for the
            // recent workload is preserved.
            //
            // The entry-count cap is the *coarse* lever: it bounds
            // how many distinct prefixes the cache can hold, but with
            // `kv_unified = true` even 16 long prefixes can pin the
            // full cell pool. The complementary fine-grained cell
            // budget (`max_resident_tokens` in
            // `ResidentCacheConfig::from_stage`, landed in PR #566)
            // closes the remaining gap by evicting on token pressure
            // before the cell pool runs out. The 16-entry cap is
            // still useful as a structural ceiling.
            max_entries: 16,
        },
    }
}

fn kv_recurrent_policy(activation_wire_dtype: StageWireDType) -> FamilyPolicy {
    FamilyPolicy {
        activation_wire_dtype,
        prefix_cache: FamilyPrefixCachePolicy::Auto {
            payload: FamilyPrefixCachePayload::KvRecurrent,
            min_tokens: 256,
            // See `resident_kv_policy` for the rationale; recurrent
            // state lanes share the same n_ctx cell pool under
            // `kv_unified = true`.
            max_entries: 16,
        },
    }
}

fn unknown_family_policy() -> FamilyPolicy {
    unknown_family_policy_with_wire_dtype(StageWireDType::F16)
}

fn unknown_family_policy_with_wire_dtype(activation_wire_dtype: StageWireDType) -> FamilyPolicy {
    disabled_family_policy(
        activation_wire_dtype,
        "family cache policy is not certified",
    )
}

fn disabled_family_policy(
    activation_wire_dtype: StageWireDType,
    reason: &'static str,
) -> FamilyPolicy {
    FamilyPolicy {
        activation_wire_dtype,
        prefix_cache: FamilyPrefixCachePolicy::Disabled { reason },
    }
}

fn wire_dtype_from_capability(dtype: WireDType) -> StageWireDType {
    match dtype {
        WireDType::F32 => StageWireDType::F32,
        WireDType::F16 => StageWireDType::F16,
        WireDType::Q8 => StageWireDType::Q8,
    }
}

fn derive_stage_cache_max_bytes(config: &StageConfig) -> Option<u64> {
    [
        config.materialized_path.as_deref(),
        config.source_model_path.as_deref(),
        config.model_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(|path| scan_gguf_compact_meta(Path::new(path)))
    .and_then(|meta| estimate_stage_cache_max_bytes(config, &meta))
}

fn estimate_stage_cache_max_bytes(config: &StageConfig, meta: &GgufCompactMeta) -> Option<u64> {
    let stage_layers = config.layer_end.checked_sub(config.layer_start)?;
    if stage_layers == 0 {
        return None;
    }

    let kv_heads = if meta.kv_head_count > 0 {
        meta.kv_head_count
    } else {
        meta.head_count
    };
    let key_width = if meta.key_length > 0 {
        meta.key_length
    } else if meta.embedding_size > 0 && kv_heads > 0 {
        meta.embedding_size.checked_div(kv_heads)?
    } else {
        return None;
    };
    let value_width = if meta.value_length > 0 {
        meta.value_length
    } else if meta.embedding_size > 0 && kv_heads > 0 {
        meta.embedding_size.checked_div(kv_heads)?
    } else {
        return None;
    };

    let key_elems_per_token = u64::from(key_width).checked_mul(u64::from(kv_heads))?;
    let value_elems_per_token = u64::from(value_width).checked_mul(u64::from(kv_heads))?;
    let key_bytes_per_token = dtype_bytes(key_elems_per_token, &config.cache_type_k)?;
    let value_bytes_per_token = dtype_bytes(value_elems_per_token, &config.cache_type_v)?;
    let bytes_per_token_layer = key_bytes_per_token.checked_add(value_bytes_per_token)?;

    // The prefix cache shares the same `n_ctx` cell pool the active
    // lanes use (skippy patches set `kv_unified = true` whenever
    // `lane_count > 1`; see patch 0034). The total native KV memory
    // is `bytes_per_token_layer * stage_layers * n_ctx` — NOT
    // multiplied by `lane_count` (lanes share, they do not multiply
    // the budget). Cap the cache at *half* that total so the other
    // half stays free for the lanes' fresh prompts.
    //
    // The previous code (a) included the lane_count multiplier (so
    // budget was 2–4× the actual pool) and (b) didn't reserve any
    // pool for active lanes. Under sustained agent-style traffic
    // (Goose, OpenCode, pi against `model: auto`) the cache filled
    // until it crowded the lanes out and the embedded runtime
    // returned HTTP 502 `RuntimeError: llama_decode failed`
    // (`decode: failed to find a memory slot`).
    let full_pool_bytes = bytes_per_token_layer
        .checked_mul(u64::from(stage_layers))?
        .checked_mul(u64::from(config.ctx_size.max(1)))?;
    let cache_budget_bytes = full_pool_bytes / 2;
    if cache_budget_bytes == 0 {
        return None;
    }
    Some(cache_budget_bytes)
}

fn dtype_bytes(elements: u64, dtype: &str) -> Option<u64> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "f32" => elements.checked_mul(4),
        "f16" | "bf16" => elements.checked_mul(2),
        "q8" | "q8_0" => ggml_block_bytes(elements, 32, 34),
        "q8_1" => ggml_block_bytes(elements, 32, 36),
        "q4" | "q4_0" | "iq4_nl" => ggml_block_bytes(elements, 32, 18),
        "q4_1" => ggml_block_bytes(elements, 32, 20),
        _ => None,
    }
}

fn ggml_block_bytes(elements: u64, block_size: u64, type_size: u64) -> Option<u64> {
    elements.div_ceil(block_size).checked_mul(type_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{FlashAttentionType, LoadMode};
    use skippy_topology::{reviewed_capability_records, STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS};

    fn meta(architecture: &str) -> GgufCompactMeta {
        GgufCompactMeta {
            architecture: architecture.to_string(),
            layer_count: 28,
            embedding_size: 1024,
            ..Default::default()
        }
    }

    fn stage_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "test/model:Q4_K_M".to_string(),
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
            layer_end: 2,
            ctx_size: 1024,
            lane_count: 2,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "q8_0".to_string(),
            flash_attn_type: FlashAttentionType::Disabled,
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    fn kv_meta() -> GgufCompactMeta {
        GgufCompactMeta {
            architecture: "llama".to_string(),
            layer_count: 32,
            embedding_size: 4096,
            head_count: 32,
            kv_head_count: 8,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        }
    }

    #[test]
    fn qwen_policy_comes_from_gguf_architecture() {
        let policy = family_policy_for_gguf_meta(&meta("qwen3"), None);

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert_eq!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::ResidentKv,
                min_tokens: 256,
                max_entries: 16,
            }
        );
    }

    #[test]
    fn llama_policy_comes_from_capability_family_id() {
        let policy = family_policy_for_model_id("llama");

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::ResidentKv,
                ..
            }
        ));
    }

    #[test]
    fn falcon_h1_uses_kv_recurrent_cache_shape() {
        let policy = family_policy_for_model_id("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M");

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::KvRecurrent,
                ..
            }
        ));
    }

    #[test]
    fn deepseek3_uses_resident_kv_cache_shape_until_mla_is_certified() {
        let policy = family_policy_for_model_id("unsloth/DeepSeek-V3.2-GGUF:Q4_K_M");

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::ResidentKv,
                ..
            }
        ));
    }

    #[test]
    fn qwen3_coder_active_parameter_package_uses_resident_kv_cache_shape() {
        let policy =
            family_policy_for_model_id("unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:UD-Q4_K_XL");

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::ResidentKv,
                ..
            }
        ));
    }

    #[test]
    fn gemma_family_uses_resident_kv_cache_shape() {
        let policy = family_policy_for_gguf_meta(&meta("gemma3"), None);

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::ResidentKv,
                ..
            }
        ));
    }

    #[test]
    fn gemma_small_reviewed_policy_uses_f32_activation_wire() {
        let policy = family_policy_for_model_id("ggml-org/gemma-3-270m-it-GGUF:Q8_0");

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F32);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::ResidentKv,
                ..
            }
        ));
    }

    #[test]
    fn apertus_reviewed_policy_uses_f32_activation_wire() {
        let policy = family_policy_for_model_id("unsloth/Apertus-8B-Instruct-2509-GGUF:UD-IQ2_M");

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F32);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::ResidentKv,
                ..
            }
        ));
    }

    #[test]
    fn every_reviewed_family_has_an_explicit_cache_policy() {
        for record in reviewed_capability_records() {
            let policy = family_policy_for_capability(&record.capability);
            let family_id = record.capability.family_id.as_str();

            match family_id {
                "dream" | "llada" | "llada_moe" => assert_eq!(
                    policy.prefix_cache,
                    FamilyPrefixCachePolicy::Disabled {
                        reason: "non-causal diffusion family has no resident KV state to cache",
                    },
                    "{family_id}"
                ),
                "qwen2" | "qwen3_dense" | "llama" | "deepseek" | "deepseek2" | "deepseek3"
                | "glm4" | "glm4_moe" | "olmo" | "olmo2" | "olmoe" | "gemma" | "gemma2"
                | "gemma3" | "gemma3n" | "gemma4_a4b" | "gemma4_e4b" | "glm47_flash"
                | "minimax_m27" | "qwen2moe" | "qwen3moe" | "granite" | "granite_moe"
                | "hunyuan_dense" | "hunyuan_moe" | "hunyuan_vl" | "gptneox" | "bloom"
                | "stablelm" | "starcoder2" | "mpt" | "phi" | "phi2" | "phimoe" | "gpt2"
                | "mistral" | "internlm2" | "baichuan" | "exaone" | "exaone4" | "cohere2"
                | "exaone_moe" | "falcon" | "openai_moe" | "qwen2vl" | "qwen3vl"
                | "deepseek2ocr" | "qwen3vlmoe" | "maincoder" | "openelm" | "minicpm"
                | "minicpm3" | "plamo" | "plamo3" | "plm" | "refact" | "smallthinker"
                | "smollm3" | "arcee" | "chatglm" | "codeshell" | "deci" | "xverse" | "apertus"
                | "bitnet" | "command_r" | "starcoder" | "ernie4_5" | "ernie4_5_moe" | "qwen"
                | "jais" | "jais2" | "nemotron" | "llama4" | "mistral4" | "seed_oss" => {
                    assert_eq!(
                        policy.prefix_cache,
                        FamilyPrefixCachePolicy::Auto {
                            payload: FamilyPrefixCachePayload::ResidentKv,
                            min_tokens: 256,
                            max_entries: 16,
                        },
                        "{family_id}"
                    )
                }
                "qwen3next" | "falcon_h1" | "jamba" | "lfm2" | "mamba" | "mamba2" | "rwkv6"
                | "rwkv7" | "granite_hybrid" | "qwen35" | "qwen35moe" | "plamo2" | "nemotron_h"
                | "nemotron_h_moe" | "lfm2moe" | "kimi_linear" => assert_eq!(
                    policy.prefix_cache,
                    FamilyPrefixCachePolicy::Auto {
                        payload: FamilyPrefixCachePayload::KvRecurrent,
                        min_tokens: 256,
                        max_entries: 16,
                    },
                    "{family_id}"
                ),
                other => panic!("reviewed family {other} has no explicit policy assertion"),
            }
        }
    }

    #[test]
    fn every_stage_runtime_llama_architecture_has_cache_policy() {
        for expected in STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS {
            let policy = family_policy_for_model_id(expected.llama_architecture);
            if matches!(expected.family_id, "dream" | "llada" | "llada_moe") {
                assert_eq!(
                    policy.prefix_cache,
                    FamilyPrefixCachePolicy::Disabled {
                        reason: "non-causal diffusion family has no resident KV state to cache",
                    },
                    "{} ({})",
                    expected.llama_architecture,
                    expected.family_id
                );
                continue;
            }

            let expected_payload = if expected.recurrent_or_hybrid {
                FamilyPrefixCachePayload::KvRecurrent
            } else {
                FamilyPrefixCachePayload::ResidentKv
            };

            assert_eq!(
                policy.prefix_cache,
                FamilyPrefixCachePolicy::Auto {
                    payload: expected_payload,
                    min_tokens: 256,
                    max_entries: 16,
                },
                "{} ({})",
                expected.llama_architecture,
                expected.family_id
            );
        }
    }

    #[test]
    fn production_cache_policy_never_selects_full_state() {
        for record in reviewed_capability_records() {
            let policy = family_policy_for_capability(&record.capability);

            if let FamilyPrefixCachePolicy::Auto { payload, .. } = policy.prefix_cache {
                assert!(
                    matches!(
                        payload,
                        FamilyPrefixCachePayload::ResidentKv
                            | FamilyPrefixCachePayload::KvRecurrent
                    ),
                    "{}",
                    record.capability.family_id
                );
            }
        }
    }

    #[test]
    fn certified_recurrent_families_never_use_resident_kv_policy() {
        for model_id in [
            "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M",
            "bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS",
            "bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M",
            "meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M",
            "mradermacher/mamba-130m-hf-GGUF:Q4_K_M",
            "mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M",
            "latestissue/rwkv-6-finch-1b6-gguf:Q4_K",
            "Mungert/rwkv7-191M-world-GGUF:Q4_K",
            "mradermacher/UnifiedReward-Edit-qwen35-4b-i1-GGUF:IQ2_M",
        ] {
            let policy = family_policy_for_model_id(model_id);
            assert!(
                matches!(
                    policy.prefix_cache,
                    FamilyPrefixCachePolicy::Auto {
                        payload: FamilyPrefixCachePayload::KvRecurrent,
                        ..
                    }
                ),
                "{model_id}: {:?}",
                policy.prefix_cache
            );
        }
    }

    #[test]
    fn stage_cache_cap_tracks_ctx_layers_and_kv_types() {
        let config = stage_config();

        let bytes = estimate_stage_cache_max_bytes(&config, &kv_meta()).unwrap();

        // 4096 bytes/token/layer * 2 stage_layers * 1024 ctx_size /
        // 2 (cache may use at most half the unified KV pool).
        //
        // Crucially does NOT include lane_count: skippy's unified KV
        // shares one cell pool across all lanes (`kv_unified = true`
        // patch 0034), so the cache budget is independent of lane
        // count. The previous formula multiplied by lane_count AND
        // didn't reserve any of the pool for active lanes, which
        // produced an over-generous cache budget that surfaced as
        // `decode: failed to find a memory slot` failures under
        // sustained agent traffic.
        assert_eq!(bytes, 3_211_264);
    }

    #[test]
    fn stage_cache_cap_tracks_quantized_kv_types() {
        let mut config = stage_config();
        config.cache_type_k = "q4_0".to_string();
        config.cache_type_v = "q4_0".to_string();

        let bytes = estimate_stage_cache_max_bytes(&config, &kv_meta()).unwrap();

        // q4_0 packs 32 elements into 18 bytes (= 0.5625 bytes/element
        // vs 2.0 for f16). Same `2 stage_layers * 1024 ctx_size / 2`
        // and no lane_count multiplier; see
        // `stage_cache_cap_tracks_ctx_layers_and_kv_types` for why.
        assert_eq!(bytes, 1_179_648);
    }

    #[test]
    fn stage_cache_cap_rejects_unknown_kv_type() {
        let mut config = stage_config();
        config.cache_type_k = "mystery".to_string();

        assert!(estimate_stage_cache_max_bytes(&config, &kv_meta()).is_none());
    }
}
