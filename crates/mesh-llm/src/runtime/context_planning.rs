use crate::models::gguf::{GgufCompactMeta, GgufKvCacheQuant};

const DEFAULT_CONTEXT_LENGTH: u32 = 4096;
const DEFAULT_PARALLEL_SLOTS: usize = 4;
const MIN_AUTO_CONTEXT_LENGTH: u32 = 512;
const MAX_AUTO_PARALLEL_SLOTS: usize = 16;
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
    pub(super) model_bytes: u64,
    pub(super) vram_bytes: u64,
    pub(super) metadata: Option<&'a GgufCompactMeta>,
    pub(super) kv_cache_quant: GgufKvCacheQuant,
}

pub(super) fn plan_runtime_resources(input: RuntimeResourcePlanInput<'_>) -> RuntimeResourcePlan {
    let context_length = input
        .ctx_size_override
        .unwrap_or_else(|| planned_context_length(input));
    let slots = input
        .parallel_override
        .unwrap_or_else(|| planned_parallel_slots(input, context_length));

    RuntimeResourcePlan {
        context_length,
        slots,
    }
}

fn planned_context_length(input: RuntimeResourcePlanInput<'_>) -> u32 {
    let Some(metadata) = input.metadata else {
        return DEFAULT_CONTEXT_LENGTH;
    };
    let native_context = metadata.context_length;
    if native_context == 0 {
        return DEFAULT_CONTEXT_LENGTH;
    }
    let Some(kv_bytes_per_token) = input.kv_cache_quant.kv_cache_bytes_per_token(metadata) else {
        return DEFAULT_CONTEXT_LENGTH.min(native_context);
    };
    let kv_budget = usable_kv_cache_budget(input.vram_bytes, input.model_bytes);
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

fn planned_parallel_slots(input: RuntimeResourcePlanInput<'_>, context_length: u32) -> usize {
    let Some(metadata) = input.metadata else {
        return DEFAULT_PARALLEL_SLOTS;
    };
    let Some(kv_bytes_per_token) = input.kv_cache_quant.kv_cache_bytes_per_token(metadata) else {
        return DEFAULT_PARALLEL_SLOTS;
    };
    let Some(bytes_per_slot) = u64::from(context_length).checked_mul(kv_bytes_per_token) else {
        return DEFAULT_PARALLEL_SLOTS;
    };
    if bytes_per_slot == 0 {
        return DEFAULT_PARALLEL_SLOTS;
    }

    let raw_slots = usable_kv_cache_budget(input.vram_bytes, input.model_bytes) / bytes_per_slot;
    snap_parallel_slots_down(raw_slots)
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
    fn plan_runtime_resources_preserves_explicit_overrides() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: Some(16_384),
            parallel_override: Some(7),
            model_bytes: 10_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::f16(),
        });

        assert_eq!(
            plan,
            RuntimeResourcePlan {
                context_length: 16_384,
                slots: 7,
            }
        );
    }

    #[test]
    fn plan_runtime_resources_clamps_auto_context_to_native_metadata() {
        let metadata = gqa_metadata(16_384);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::f16(),
        });

        assert_eq!(plan.context_length, 16_384);
        assert!(
            plan.slots > 4,
            "expected metadata-based slots, got {plan:?}"
        );
    }

    #[test]
    fn plan_runtime_resources_snaps_auto_context_down() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(1),
            model_bytes: 5_000_000_000,
            vram_bytes: 6_300_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::f16(),
        });

        assert_eq!(plan.context_length, 8192);
        assert_eq!(plan.slots, 1);
    }

    #[test]
    fn plan_runtime_resources_uses_effective_kv_quant_for_slot_budget() {
        let metadata = gqa_metadata(131_072);
        let f16_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::f16(),
        });
        let q4_plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::new(
                model_artifact::gguf::GgufKvCacheType::Q4_0,
                model_artifact::gguf::GgufKvCacheType::Q4_0,
            ),
        });

        assert_eq!(f16_plan.context_length, q4_plan.context_length);
        assert!(
            q4_plan.slots > f16_plan.slots,
            "expected quantized KV cache to allow more slots: f16={f16_plan:?}, q4={q4_plan:?}"
        );
    }

    #[test]
    fn plan_runtime_resources_falls_back_to_legacy_defaults_without_metadata() {
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            vram_bytes: 24_000_000_000,
            metadata: None,
            kv_cache_quant: GgufKvCacheQuant::f16(),
        });

        assert_eq!(
            plan,
            RuntimeResourcePlan {
                context_length: 4096,
                slots: 4,
            }
        );
    }

    #[test]
    fn plan_runtime_resources_uses_explicit_parallel_with_metadata_context() {
        let metadata = gqa_metadata(32_768);
        let plan = plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: Some(2),
            model_bytes: 5_000_000_000,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::f16(),
        });

        assert_eq!(plan.context_length, 32_768);
        assert_eq!(plan.slots, 2);
    }
}
