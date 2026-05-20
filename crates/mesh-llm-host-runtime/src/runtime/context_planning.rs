use crate::models::gguf::{GgufCompactMeta, GgufKvCacheQuant};

const DEFAULT_CONTEXT_LENGTH: u32 = 4096;
const DEFAULT_PARALLEL_SLOTS: usize = 4;
const MIN_AUTO_CONTEXT_LENGTH: u32 = 512;
/// Auto-planner ceiling on concurrent lanes.
///
/// Matches upstream llama-server: when `--parallel` is left to auto,
/// llama-server picks `n_parallel = 4` and turns on `kv_unified = true`
/// (see `tools/server/server.cpp`,
/// `"n_parallel is set to auto, using n_parallel = 4 and kv_unified = true"`).
///
/// Skippy's stage-runtime patches also set `kv_unified = true` whenever
/// `lane_count > 1` (`third_party/llama.cpp/patches/0034-*.patch`). In
/// unified mode llama allocates exactly `n_ctx` cells total, shared
/// across all `n_seq_max` sequences. The previous ceiling of 16 was
/// inherited from a VRAM-based slot calculation that pretended each
/// lane carved off its own `n_ctx × bytes_per_token` allocation —
/// which is the `kv_unified = false` semantics, not what skippy
/// actually does. On any node with comfortable VRAM that math
/// happily picked 16 lanes even though all 16 raced for the *same*
/// pool of `n_ctx` cells.
///
/// Concrete failure mode that prompted this change: Qwen3-8B on a
/// 32k `n_ctx` got `slots = 16`. Three concurrent agent-shape
/// requests (~14k tokens each — OpenCode system prompt plus tools
/// plus a tool-result follow-up) need ~45k cells in the shared 32k
/// pool; llama's `find_slot` fails on the third request and skippy
/// surfaces it as an HTTP 502 with body `skippy ABI call failed:
/// RuntimeError: llama_decode failed`.
///
/// 4 is the same conservative ceiling llama-server uses for the
/// same `kv_unified = true` reason. Operators who know their
/// workload (e.g. all short chat turns, or a single-user MoA host)
/// can still go higher via `parallel_override` /
/// `[models.throughput] parallel = N` in the TOML config.
const MAX_AUTO_PARALLEL_SLOTS: usize = 4;
const KV_CACHE_BUDGET_NUMERATOR: u64 = 85;
const KV_CACHE_BUDGET_DENOMINATOR: u64 = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeResourcePlan {
    pub(super) context_length: u32,
    pub(super) slots: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RuntimeResourcePlanInput<'a> {
    pub(super) ctx_size_override: Option<u32>,
    pub(super) parallel_override: Option<usize>,
    /// Model weight bytes **local to this node**.  For a split/layer-package
    /// load, pass only this node's share of the model weights.
    pub(super) model_bytes: u64,
    pub(super) vram_bytes: u64,
    pub(super) metadata: Option<&'a GgufCompactMeta>,
    /// The KV cache quant that will be used.  Default is Q8_0 everywhere.
    /// Only differs when the user explicitly passes `--cache-type-k/v`.
    pub(super) kv_cache_quant: GgufKvCacheQuant,
    /// Fraction of the model's layers that reside on this node (0.0–1.0).
    /// `None` means the whole model is local (fraction = 1.0).
    pub(super) local_layer_fraction: Option<f64>,
}

/// Plan context length and parallel slots.
///
/// Strategy: maximise context up to the model's native context length using
/// the provided KV quant (default Q8_0).  No negotiation — the quant is
/// decided upstream (Q8_0 default, or user override via CLI flags).
pub(super) fn plan_runtime_resources(input: RuntimeResourcePlanInput<'_>) -> RuntimeResourcePlan {
    let context_length = input
        .ctx_size_override
        .unwrap_or_else(|| planned_context_length(&input));
    let slots = input
        .parallel_override
        .unwrap_or_else(|| planned_parallel_slots(&input, context_length));

    RuntimeResourcePlan {
        context_length,
        slots,
    }
}

fn planned_context_length(input: &RuntimeResourcePlanInput<'_>) -> u32 {
    let Some(metadata) = input.metadata else {
        return DEFAULT_CONTEXT_LENGTH;
    };
    let native_context = metadata.context_length;
    if native_context == 0 {
        return DEFAULT_CONTEXT_LENGTH;
    }
    let Some(kv_bytes_per_token_full) = input.kv_cache_quant.kv_cache_bytes_per_token(metadata)
    else {
        return DEFAULT_CONTEXT_LENGTH.min(native_context);
    };

    // In a pipeline-parallel split each stage only holds KV state for its
    // own layers.  Scale the per-token cost by the local layer fraction.
    let kv_bytes_per_token = scale_by_layer_fraction(kv_bytes_per_token_full, input);

    let kv_budget = usable_kv_cache_budget(input.vram_bytes, input.model_bytes);
    if kv_bytes_per_token == 0 {
        return native_context;
    }
    let max_affordable_context = kv_budget / kv_bytes_per_token;
    if max_affordable_context == 0 {
        return MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    }

    let planned = max_affordable_context
        .min(u64::from(native_context))
        .min(u64::from(u32::MAX)) as u32;
    let minimum = MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    if planned < minimum {
        minimum
    } else {
        snap_context_length_down(planned).max(minimum)
    }
}

fn planned_parallel_slots(input: &RuntimeResourcePlanInput<'_>, context_length: u32) -> usize {
    let Some(metadata) = input.metadata else {
        return DEFAULT_PARALLEL_SLOTS;
    };
    let Some(kv_bytes_per_token_full) = input.kv_cache_quant.kv_cache_bytes_per_token(metadata)
    else {
        return DEFAULT_PARALLEL_SLOTS;
    };

    let kv_bytes_per_token = scale_by_layer_fraction(kv_bytes_per_token_full, input);

    let Some(bytes_per_slot) = u64::from(context_length).checked_mul(kv_bytes_per_token) else {
        return DEFAULT_PARALLEL_SLOTS;
    };
    if bytes_per_slot == 0 {
        return DEFAULT_PARALLEL_SLOTS;
    }

    let raw_slots = usable_kv_cache_budget(input.vram_bytes, input.model_bytes) / bytes_per_slot;
    snap_parallel_slots_down(raw_slots)
}

fn scale_by_layer_fraction(kv_bytes_per_token: u64, input: &RuntimeResourcePlanInput<'_>) -> u64 {
    let fraction = input.local_layer_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    if fraction < 1.0 && fraction > 0.0 {
        ((kv_bytes_per_token as f64) * fraction).ceil() as u64
    } else {
        kv_bytes_per_token
    }
}

fn usable_kv_cache_budget(vram_bytes: u64, model_bytes: u64) -> u64 {
    let free_bytes = vram_bytes.saturating_sub(model_bytes);
    let budget = u128::from(free_bytes) * u128::from(KV_CACHE_BUDGET_NUMERATOR)
        / u128::from(KV_CACHE_BUDGET_DENOMINATOR);
    budget.min(u128::from(u64::MAX)) as u64
}

fn snap_parallel_slots_down(raw_slots: u64) -> usize {
    match raw_slots.min(MAX_AUTO_PARALLEL_SLOTS as u64) {
        0 => 1,
        1 => 1,
        2 | 3 => 2,
        4..=7 => 4,
        8..=15 => 8,
        _ => MAX_AUTO_PARALLEL_SLOTS,
    }
}

fn snap_context_length_down(value: u32) -> u32 {
    const CONTEXT_STEPS: &[u32] = &[512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];
    CONTEXT_STEPS
        .iter()
        .rev()
        .copied()
        .find(|step| *step <= value)
        .unwrap_or(MIN_AUTO_CONTEXT_LENGTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gqa_metadata(context_length: u32) -> GgufCompactMeta {
        GgufCompactMeta {
            context_length,
            head_count: 32,
            kv_head_count: 8,
            layer_count: 32,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        }
    }

    #[test]
    fn explicit_overrides_are_preserved() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: Some(16_384),
            parallel_override: Some(7),
            model_bytes: 10_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 16_384);
        assert_eq!(plan.slots, 7);
    }

    #[test]
    fn auto_context_clamped_to_native() {
        let metadata = gqa_metadata(16_384);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        assert_eq!(
            plan.context_length, 16_384,
            "should reach native context, not exceed it"
        );
    }

    #[test]
    fn q8_default_reaches_larger_context_than_f16() {
        // Tight VRAM so f16 can only reach 8K but q8_0 reaches 16K.
        // KV budget = (7.0 - 5.0) * 0.85 = 1.7 GB.
        // f16: 131072 B/tok → 1.7G / 131K ≈ 12K → snaps 8K
        // q8:   69632 B/tok → 1.7G / 69K  ≈ 24K → snaps 16K
        let metadata = gqa_metadata(131_072);
        let f16_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(1),
            model_bytes: 5_000_000_000,
            vram_bytes: 7_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::F16,
            local_layer_fraction: None,
        });
        let q8_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(1),
            model_bytes: 5_000_000_000,
            vram_bytes: 7_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        assert!(
            q8_plan.context_length > f16_plan.context_length,
            "q8_0 should afford more context: q8={}K, f16={}K",
            q8_plan.context_length / 1024,
            f16_plan.context_length / 1024
        );
    }

    #[test]
    fn fallback_defaults_without_metadata() {
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: None,
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 4096);
        assert_eq!(plan.slots, 4);
    }

    #[test]
    fn explicit_parallel_with_auto_context() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(2),
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 32_768);
        assert_eq!(plan.slots, 2);
    }

    #[test]
    fn auto_slots_capped_at_llama_server_default() {
        // Regression: a small model on a huge-VRAM box used to plan
        // `slots = 16` because the VRAM-derived per-lane math pretended
        // each lane carved off its own `n_ctx × bytes/token` allocation.
        // With `kv_unified = true` (skippy patch 0034) those 16 lanes
        // race for the same `n_ctx` cell pool, and 3 concurrent agent
        // requests at ~14k tokens each blow it up with
        // `find_slot` failures → HTTP 502
        // `RuntimeError: llama_decode failed`.
        //
        // Match llama-server's auto default of 4 (see
        // `.deps/llama.cpp/tools/server/server.cpp`: "n_parallel is
        // set to auto, using n_parallel = 4 and kv_unified = true").
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            // 128GB free — plenty for many "per-lane" slots under the
            // old broken math.
            vram_bytes: 128_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        assert_eq!(plan.context_length, 32_768);
        assert!(
            plan.slots <= 4,
            "auto-planner should not exceed llama-server's 4-lane unified-KV ceiling; got {}",
            plan.slots
        );
    }

    #[test]
    fn explicit_parallel_can_exceed_auto_ceiling() {
        // Operators who know their workload can still go higher than
        // the auto ceiling via `parallel_override`.
        let metadata = gqa_metadata(131_072);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(8),
            model_bytes: 5_000_000_000,
            vram_bytes: 128_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        assert_eq!(plan.slots, 8);
    }

    #[test]
    fn split_model_uses_local_layer_fraction() {
        // 480B-class model: 94 layers, 264GB total, host holds 62/94 layers.
        let metadata = GgufCompactMeta {
            context_length: 131_072,
            head_count: 64,
            kv_head_count: 8,
            layer_count: 94,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };
        let total_model_bytes: u64 = 264_000_000_000;
        let local_fraction = 62.0 / 94.0;
        let local_model_bytes = (total_model_bytes as f64 * local_fraction) as u64;

        // Without split awareness: 206 GB VRAM, 264 GB model → negative budget → minimum
        let no_split = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: total_model_bytes,
            vram_bytes: 206_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });

        // With split awareness: local model ~174 GB, local KV fraction 0.66
        let split = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: local_model_bytes,
            vram_bytes: 206_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: Some(local_fraction),
        });

        assert!(
            split.context_length > no_split.context_length,
            "split-aware should produce larger context: split={}K, no_split={}K",
            split.context_length / 1024,
            no_split.context_length / 1024
        );
        assert!(
            split.context_length >= 65_536,
            "480B split on 206+103 GB with q8_0 should get at least 64K, got {}K",
            split.context_length / 1024
        );
    }

    #[test]
    fn q4_more_slots_than_q8_at_same_context() {
        let metadata = gqa_metadata(131_072);
        let q8_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
        });
        let q4_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q4_0,
            local_layer_fraction: None,
        });

        assert_eq!(q8_plan.context_length, q4_plan.context_length);
        assert!(
            q4_plan.slots >= q8_plan.slots,
            "q4_0 should allow at least as many slots: q4={}, q8={}",
            q4_plan.slots,
            q8_plan.slots
        );
    }
}
