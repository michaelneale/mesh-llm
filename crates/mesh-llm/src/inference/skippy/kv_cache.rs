const GB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    pub(crate) fn as_config_value(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvCachePolicy {
    pub(crate) k_type: KvCacheType,
    pub(crate) v_type: KvCacheType,
}

impl KvCachePolicy {
    pub(crate) const MEDIUM_TIER_MIN_BYTES: u64 = 5 * GB;
    pub(crate) const LARGE_TIER_MIN_BYTES: u64 = 50 * GB;

    pub(crate) fn for_model_size(model_bytes: u64) -> Self {
        if model_bytes >= Self::LARGE_TIER_MIN_BYTES {
            Self {
                k_type: KvCacheType::Q4_0,
                v_type: KvCacheType::Q4_0,
            }
        } else if model_bytes >= Self::MEDIUM_TIER_MIN_BYTES {
            Self {
                k_type: KvCacheType::Q8_0,
                v_type: KvCacheType::Q4_0,
            }
        } else {
            Self {
                k_type: KvCacheType::F16,
                v_type: KvCacheType::F16,
            }
        }
    }

    pub(crate) fn cache_type_k(self) -> &'static str {
        self.k_type.as_config_value()
    }

    pub(crate) fn cache_type_v(self) -> &'static str {
        self.v_type.as_config_value()
    }

    pub(crate) fn requires_flash_attention(self) -> bool {
        !matches!(self.v_type, KvCacheType::F16)
    }

    pub(crate) fn label(self, model_bytes: u64) -> String {
        let tier = if model_bytes >= Self::LARGE_TIER_MIN_BYTES {
            "model >= 50GB"
        } else if model_bytes >= Self::MEDIUM_TIER_MIN_BYTES {
            "model 5-50GB"
        } else {
            "model < 5GB"
        };
        format!(
            "{} K + {} V ({tier})",
            self.cache_type_k().to_ascii_uppercase(),
            self.cache_type_v().to_ascii_uppercase()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_policy_matches_legacy_llama_server_tiers() {
        assert_eq!(
            KvCachePolicy::for_model_size(KvCachePolicy::MEDIUM_TIER_MIN_BYTES - 1),
            KvCachePolicy {
                k_type: KvCacheType::F16,
                v_type: KvCacheType::F16,
            }
        );
        assert_eq!(
            KvCachePolicy::for_model_size(KvCachePolicy::MEDIUM_TIER_MIN_BYTES),
            KvCachePolicy {
                k_type: KvCacheType::Q8_0,
                v_type: KvCacheType::Q4_0,
            }
        );
        assert_eq!(
            KvCachePolicy::for_model_size(KvCachePolicy::LARGE_TIER_MIN_BYTES),
            KvCachePolicy {
                k_type: KvCacheType::Q4_0,
                v_type: KvCacheType::Q4_0,
            }
        );
    }
}
