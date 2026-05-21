use skippy_protocol::{StageConfig, StageKvCacheConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentCacheConfig {
    pub max_entries: usize,
    pub max_bytes: u64,
    pub min_tokens: u64,
    pub reserved_seq_count: i32,
    /// Maximum number of native KV cell positions the cache may hold
    /// at one time, in tokens. Under `kv_unified = true` (skippy patch
    /// 0034) the resident prefix cache shares one `n_ctx` cell pool
    /// with the active execution lanes. Without this cap the cache
    /// budget is bounded only by `max_entries` and `max_bytes`, both
    /// of which can easily allow more pinned tokens than the cell
    /// pool has cells — the lanes then can't find a free slot and
    /// the embedded runtime surfaces HTTP 502
    /// `RuntimeError: llama_decode failed`
    /// (`decode: failed to find a memory slot`).
    ///
    /// Set this to a fraction of the model's `n_ctx` (typically
    /// `n_ctx / 2` or similar). A value of 0 disables the cap and
    /// behaves like the legacy unbounded-by-tokens cache. The cap is
    /// only useful when `n_ctx` is comfortably larger than
    /// `min_tokens`; see [`derive_max_resident_tokens`] for the floor.
    pub max_resident_tokens: u64,
}

impl ResidentCacheConfig {
    pub fn from_stage(config: &StageConfig, cache: &StageKvCacheConfig) -> Self {
        let reserved_seq_count = i32::try_from(config.lane_count.saturating_mul(2))
            .unwrap_or(i32::MAX)
            .max(2);
        let max_resident_tokens = derive_max_resident_tokens(u64::from(config.ctx_size));
        Self {
            max_entries: cache.max_entries.clamp(1, 512),
            max_bytes: cache.max_bytes,
            min_tokens: cache.min_tokens,
            reserved_seq_count,
            max_resident_tokens,
        }
    }
}

/// Derive `max_resident_tokens` from the model's `n_ctx` cell pool.
///
/// The cache shares the `n_ctx` cell pool with the active lanes under
/// `kv_unified = true`. The cap reserves half of the pool for in-flight
/// lane prefills and lets the cache use at most the other half.
///
/// For small contexts (smoke-test / tiny-model configs) the half-pool
/// can be smaller than a single typical prompt; applying the cap then
/// rejects the very first record and degrades the cache without
/// preventing any real wedge. The cap is therefore disabled when the
/// model's `n_ctx` is below `MIN_CTX_FOR_CELL_CAP` cells. The original
/// failure mode this cap fixes is large-context unified-KV serving
/// (e.g. `n_ctx = 131072`), which comfortably clears this floor.
///
/// Picking `min_tokens` as the floor would be tempting but does not
/// match the actual wedge: callers can configure `min_tokens` as low
/// as 64 while still using a small `n_ctx`, and the cap would still
/// be smaller than typical prompts. A hard cell-count floor is easier
/// to reason about and matches the real-world contexts the cap is
/// designed for (long-context unified-KV serving).
const MIN_CTX_FOR_CELL_CAP: u64 = 8192;

fn derive_max_resident_tokens(ctx_size: u64) -> u64 {
    if ctx_size < MIN_CTX_FOR_CELL_CAP {
        return 0;
    }
    ctx_size.saturating_div(2)
}

#[cfg(test)]
mod resident_cache_config_tests {
    use super::*;

    #[test]
    fn cap_disabled_for_smoke_test_ctx_size() {
        // Smoke-test / SmolLM2 scenario: ctx_size=768. Half=384 would
        // be smaller than a typical 533-token smoke prompt; cap stays
        // disabled.
        assert_eq!(derive_max_resident_tokens(768), 0);
    }

    #[test]
    fn cap_enabled_for_production_ctx_size() {
        // Production failure mode the cap is designed for.
        assert_eq!(derive_max_resident_tokens(131072), 65536);
        // Exactly at the floor.
        assert_eq!(derive_max_resident_tokens(8192), 4096);
        // Just above the floor.
        assert_eq!(derive_max_resident_tokens(16384), 8192);
    }

    #[test]
    fn cap_disabled_just_below_floor() {
        // Below the hard floor.
        assert_eq!(derive_max_resident_tokens(8191), 0);
        assert_eq!(derive_max_resident_tokens(4096), 0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCandidatePolicy {
    pub min_tokens: u64,
    pub stride_tokens: u64,
    pub record_limit: u64,
    pub page_size_tokens: u64,
}

impl PrefixCandidatePolicy {
    pub fn from_cache(cache: &StageKvCacheConfig) -> Self {
        Self {
            min_tokens: cache.min_tokens,
            stride_tokens: cache.shared_prefix_stride_tokens,
            record_limit: cache.shared_prefix_record_limit,
            page_size_tokens: cache.min_tokens.max(1),
        }
    }

    pub fn candidate_token_counts(self, token_count: u64) -> Vec<u64> {
        if token_count == 0 {
            return Vec::new();
        }
        let mut counts = vec![token_count];
        if self.min_tokens == 0 || token_count <= self.min_tokens {
            return counts;
        }
        let stride = self.stride_tokens.max(1).min(self.page_size_tokens.max(1));
        let mut candidate = token_count.saturating_sub(1);
        while candidate >= self.min_tokens {
            counts.push(candidate);
            if candidate == self.min_tokens {
                break;
            }
            let next = candidate.saturating_sub(stride);
            candidate = next.max(self.min_tokens);
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts.dedup();
        counts
    }

    pub fn record_candidate_token_counts(self, token_count: u64) -> Vec<u64> {
        let candidates = self.candidate_token_counts(token_count);
        let limit = self.record_limit as usize;
        if limit == 0 || candidates.len() <= limit {
            return candidates;
        }

        let mut selected = Vec::with_capacity(limit);
        selected.push(token_count);

        let lower_bound = candidates
            .iter()
            .copied()
            .filter(|candidate| *candidate != token_count)
            .min()
            .unwrap_or(token_count);
        let shared_slots = limit.saturating_sub(1);
        if shared_slots == 0 {
            return selected;
        }
        for slot in 0..shared_slots {
            let target = if shared_slots == 1 {
                lower_bound
            } else {
                let span = token_count.saturating_sub(lower_bound);
                token_count
                    .saturating_sub(span.saturating_mul((slot + 1) as u64) / shared_slots as u64)
            }
            .max(self.min_tokens)
            .min(token_count);
            if let Some(candidate) = candidates
                .iter()
                .copied()
                .filter(|candidate| *candidate <= target && *candidate != token_count)
                .max()
            {
                selected.push(candidate);
            }
        }
        if selected.len() < limit {
            for candidate in candidates.into_iter().rev() {
                if selected.len() >= limit {
                    break;
                }
                if !selected.contains(&candidate) {
                    selected.push(candidate);
                }
            }
        }
        selected.sort_unstable_by(|a, b| b.cmp(a));
        selected.dedup();
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_candidates_prefer_longest_prefix_first() {
        let policy = PrefixCandidatePolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 2,
            page_size_tokens: 64,
        };

        assert_eq!(
            policy.candidate_token_counts(160),
            vec![160, 159, 127, 95, 64]
        );
    }

    #[test]
    fn record_candidates_are_limited_but_keep_current_and_shared_prefix() {
        let policy = PrefixCandidatePolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 2,
            page_size_tokens: 64,
        };

        assert_eq!(policy.record_candidate_token_counts(160), vec![160, 64]);
    }

    #[test]
    fn candidates_below_min_only_use_exact_request() {
        let policy = PrefixCandidatePolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 2,
            page_size_tokens: 64,
        };

        assert_eq!(policy.candidate_token_counts(63), vec![63]);
        assert_eq!(policy.record_candidate_token_counts(63), vec![63]);
    }

    #[test]
    fn unlimited_record_candidates_keep_shared_prefix_grid() {
        let policy = PrefixCandidatePolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 0,
            page_size_tokens: 64,
        };

        assert_eq!(
            policy.record_candidate_token_counts(160),
            vec![160, 159, 127, 95, 64]
        );
    }
}
