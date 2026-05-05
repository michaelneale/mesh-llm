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
    pub(crate) fn stage_kv_cache_config(&self) -> Option<StageKvCacheConfig> {
        const DEFAULT_EXACT_CACHE_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
        match self.prefix_cache {
            FamilyPrefixCachePolicy::Disabled { .. } => None,
            FamilyPrefixCachePolicy::Auto {
                payload,
                min_tokens,
                max_entries,
            } => Some(StageKvCacheConfig {
                mode: StageKvCacheMode::LookupRecord,
                payload: payload.as_stage_payload(),
                max_entries,
                max_bytes: DEFAULT_EXACT_CACHE_MAX_BYTES,
                min_tokens,
                shared_prefix_stride_tokens: 128,
                shared_prefix_record_limit: 2,
            }),
        }
    }
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
    match family_id {
        "qwen3_dense" | "llama" | "deepseek2" | "deepseek3" | "glm4" | "olmo" | "gemma2"
        | "gemma3" | "gemma4_a4b" | "gemma4_e4b" | "glm47_flash" | "minimax_m27" => {
            resident_kv_policy(activation_wire_dtype)
        }
        "qwen3next" | "falcon_h1" => kv_recurrent_policy(activation_wire_dtype),
        _ => unknown_family_policy_with_wire_dtype(activation_wire_dtype),
    }
}

fn resident_kv_policy(activation_wire_dtype: StageWireDType) -> FamilyPolicy {
    FamilyPolicy {
        activation_wire_dtype,
        prefix_cache: FamilyPrefixCachePolicy::Auto {
            payload: FamilyPrefixCachePayload::ResidentKv,
            min_tokens: 256,
            max_entries: 128,
        },
    }
}

fn kv_recurrent_policy(activation_wire_dtype: StageWireDType) -> FamilyPolicy {
    FamilyPolicy {
        activation_wire_dtype,
        prefix_cache: FamilyPrefixCachePolicy::Auto {
            payload: FamilyPrefixCachePayload::KvRecurrent,
            min_tokens: 256,
            max_entries: 128,
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

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_topology::reviewed_capability_records;

    fn meta(architecture: &str) -> GgufCompactMeta {
        GgufCompactMeta {
            architecture: architecture.to_string(),
            layer_count: 28,
            embedding_size: 1024,
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
                max_entries: 128,
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
    fn every_reviewed_family_has_an_explicit_cache_policy() {
        for record in reviewed_capability_records() {
            let policy = family_policy_for_capability(&record.capability);
            let family_id = record.capability.family_id.as_str();

            match family_id {
                "qwen3_dense" | "llama" | "deepseek2" | "deepseek3" | "glm4" | "olmo"
                | "gemma2" | "gemma3" | "gemma4_a4b" | "gemma4_e4b" | "glm47_flash"
                | "minimax_m27" => {
                    assert_eq!(
                        policy.prefix_cache,
                        FamilyPrefixCachePolicy::Auto {
                            payload: FamilyPrefixCachePayload::ResidentKv,
                            min_tokens: 256,
                            max_entries: 128,
                        },
                        "{family_id}"
                    )
                }
                "qwen3next" | "falcon_h1" => assert_eq!(
                    policy.prefix_cache,
                    FamilyPrefixCachePolicy::Auto {
                        payload: FamilyPrefixCachePayload::KvRecurrent,
                        min_tokens: 256,
                        max_entries: 128,
                    },
                    "{family_id}"
                ),
                other => panic!("reviewed family {other} has no explicit policy assertion"),
            }
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
}
