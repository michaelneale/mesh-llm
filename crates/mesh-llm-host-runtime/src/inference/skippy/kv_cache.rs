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
    /// Default KV cache policy: Q8_0 for both K and V.
    ///
    /// Q8_0 gives ~2× compression over f16 with <5% speed cost across all
    /// context lengths.  This is the universal default regardless of model
    /// size — benchmarks show no meaningful quality degradation.
    ///
    /// Users can override via `--cache-type-k` / `--cache-type-v` if they
    /// want f16 (maximum precision) or q4_0 (maximum compression).
    pub(crate) fn default() -> Self {
        Self {
            k_type: KvCacheType::Q8_0,
            v_type: KvCacheType::Q8_0,
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
    fn default_kv_cache_is_q8_0() {
        let policy = KvCachePolicy::default();
        assert_eq!(policy.k_type, KvCacheType::Q8_0);
        assert_eq!(policy.v_type, KvCacheType::Q8_0);
        assert_eq!(policy.cache_type_k(), "q8_0");
        assert_eq!(policy.cache_type_v(), "q8_0");
    }
}
