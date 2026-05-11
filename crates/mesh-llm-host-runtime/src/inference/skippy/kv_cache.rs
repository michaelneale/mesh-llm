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
    const LARGE_MODEL_MIN_BYTES: u64 = 50 * 1024 * 1024 * 1024;

    /// Default KV cache policy, tiered by model size.
    ///
    /// Models >= 50 GB use Q4_0 K + Q4_0 V to keep KV cache small enough
    /// that unified-memory machines don't thrash.  On a 480B MoE split
    /// across two Apple Silicon nodes the difference between Q8_0 and Q4_0
    /// is the difference between swap-thrashing at 1 tok/s and running at
    /// 20+ tok/s.
    ///
    /// Smaller models use Q8_0 K + Q8_0 V which gives ~2× compression over
    /// f16 with negligible quality loss.
    ///
    /// Users can override via `--cache-type-k` / `--cache-type-v`.
    pub(crate) fn for_model_size(model_bytes: u64) -> Self {
        if model_bytes >= Self::LARGE_MODEL_MIN_BYTES {
            Self {
                k_type: KvCacheType::Q4_0,
                v_type: KvCacheType::Q4_0,
            }
        } else {
            Self {
                k_type: KvCacheType::Q8_0,
                v_type: KvCacheType::Q8_0,
            }
        }
    }

    pub(crate) fn cache_type_k(self) -> &'static str {
        self.k_type.as_config_value()
    }

    pub(crate) fn cache_type_v(self) -> &'static str {
        self.v_type.as_config_value()
    }

    pub(crate) fn label(self) -> String {
        format!(
            "{} K + {} V",
            self.cache_type_k().to_ascii_uppercase(),
            self.cache_type_v().to_ascii_uppercase()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_model_uses_q8_0() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        assert_eq!(policy.k_type, KvCacheType::Q8_0);
        assert_eq!(policy.v_type, KvCacheType::Q8_0);
    }

    #[test]
    fn large_model_uses_q4_0() {
        let policy = KvCachePolicy::for_model_size(50 * 1024 * 1024 * 1024);
        assert_eq!(policy.k_type, KvCacheType::Q4_0);
        assert_eq!(policy.v_type, KvCacheType::Q4_0);
    }
}
