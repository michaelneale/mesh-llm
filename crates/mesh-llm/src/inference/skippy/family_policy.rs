use std::path::Path;

use skippy_protocol::StageConfig;
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
    KvRecurrent,
    FullState,
}

impl FamilyPrefixCachePayload {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::KvRecurrent => "kv-recurrent",
            Self::FullState => "full-state",
        }
    }
}

pub(crate) fn family_policy_for_stage_config(config: &StageConfig) -> FamilyPolicy {
    [
        config.materialized_path.as_deref(),
        config.model_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(|path| family_policy_for_gguf_path(path, Some(&config.model_id)))
    .unwrap_or_else(|| family_policy_for_family_id(None))
}

pub(crate) fn family_policy_for_gguf_path(
    path: impl AsRef<Path>,
    model_id: Option<&str>,
) -> Option<FamilyPolicy> {
    let meta = scan_gguf_compact_meta(path.as_ref())?;
    Some(family_policy_for_gguf_meta(&meta, model_id))
}

pub(crate) fn family_policy_for_gguf_meta(
    meta: &GgufCompactMeta,
    model_id: Option<&str>,
) -> FamilyPolicy {
    capability_from_gguf_meta(meta, model_id)
        .as_ref()
        .map(family_policy_for_capability)
        .unwrap_or_else(|| family_policy_for_family_id(None))
}

pub(crate) fn family_policy_for_capability(capability: &FamilyCapabilityRecord) -> FamilyPolicy {
    family_policy_for_normalized_family_id(
        capability.family_id.as_str(),
        wire_dtype_from_capability(capability.default_wire_dtype),
    )
}

pub(crate) fn family_policy_for_family_id(family_id: Option<&str>) -> FamilyPolicy {
    let Some(family_id) = family_id else {
        return unknown_family_policy();
    };
    family_policy_for_normalized_family_id(family_id, StageWireDType::F16)
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
        "qwen3_dense" | "qwen3next" | "llama" | "deepseek2" | "glm4" | "olmo" | "falcon_h1"
        | "minimax_m27" => kv_recurrent_policy(activation_wire_dtype),
        "gemma2" | "gemma3" | "gemma4_a4b" | "gemma4_e4b" | "glm47_flash" => {
            full_state_policy(activation_wire_dtype)
        }
        _ => unknown_family_policy_with_wire_dtype(activation_wire_dtype),
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

fn full_state_policy(activation_wire_dtype: StageWireDType) -> FamilyPolicy {
    FamilyPolicy {
        activation_wire_dtype,
        prefix_cache: FamilyPrefixCachePolicy::Auto {
            payload: FamilyPrefixCachePayload::FullState,
            min_tokens: 256,
            max_entries: 128,
        },
    }
}

fn unknown_family_policy() -> FamilyPolicy {
    unknown_family_policy_with_wire_dtype(StageWireDType::F16)
}

fn unknown_family_policy_with_wire_dtype(activation_wire_dtype: StageWireDType) -> FamilyPolicy {
    FamilyPolicy {
        activation_wire_dtype,
        prefix_cache: FamilyPrefixCachePolicy::Disabled {
            reason: "family cache policy is not certified",
        },
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
                payload: FamilyPrefixCachePayload::KvRecurrent,
                min_tokens: 256,
                max_entries: 128,
            }
        );
    }

    #[test]
    fn llama_policy_comes_from_capability_family_id() {
        let policy = family_policy_for_family_id(Some("llama"));

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
    fn gemma_family_uses_full_state_cache_for_now() {
        let policy = family_policy_for_gguf_meta(&meta("gemma3"), None);

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::FullState,
                ..
            }
        ));
    }

    #[test]
    fn minimax_cache_uses_kv_recurrent_after_artifact_slice_certification() {
        let policy = family_policy_for_family_id(Some("minimax_m27"));

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
    fn unknown_family_disables_cache() {
        let policy = family_policy_for_family_id(None);

        assert_eq!(policy.activation_wire_dtype, StageWireDType::F16);
        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Disabled { reason }
                if reason.contains("not certified")
        ));
    }

    #[test]
    fn every_reviewed_family_has_an_explicit_cache_policy() {
        for record in reviewed_capability_records() {
            let policy = family_policy_for_capability(&record.capability);
            let family_id = record.capability.family_id.as_str();

            match family_id {
                "qwen3_dense" | "qwen3next" | "llama" | "deepseek2" | "glm4" | "olmo"
                | "falcon_h1" | "minimax_m27" => assert_eq!(
                    policy.prefix_cache,
                    FamilyPrefixCachePolicy::Auto {
                        payload: FamilyPrefixCachePayload::KvRecurrent,
                        min_tokens: 256,
                        max_entries: 128,
                    },
                    "{family_id}"
                ),
                "gemma2" | "gemma3" | "gemma4_a4b" | "gemma4_e4b" | "glm47_flash" => {
                    assert_eq!(
                        policy.prefix_cache,
                        FamilyPrefixCachePolicy::Auto {
                            payload: FamilyPrefixCachePayload::FullState,
                            min_tokens: 256,
                            max_entries: 128,
                        },
                        "{family_id}"
                    )
                }
                other => panic!("reviewed family {other} has no explicit policy assertion"),
            }
        }
    }

    #[test]
    fn reviewed_model_identity_refines_coarse_gguf_architecture() {
        let mut meta = meta("glm4");
        meta.layer_count = 47;
        meta.embedding_size = 2048;

        let policy = family_policy_for_gguf_meta(&meta, Some("unsloth/GLM-4.7-Flash-GGUF:Q4_K_M"));

        assert!(matches!(
            policy.prefix_cache,
            FamilyPrefixCachePolicy::Auto {
                payload: FamilyPrefixCachePayload::FullState,
                ..
            }
        ));
    }
}
