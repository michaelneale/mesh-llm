use skippy_protocol::{StageConfig, StageKvCacheConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentCacheConfig {
    pub max_entries: usize,
    pub max_bytes: u64,
    pub min_tokens: u64,
    pub reserved_seq_count: i32,
}

impl ResidentCacheConfig {
    pub fn from_stage(config: &StageConfig, cache: &StageKvCacheConfig) -> Self {
        let reserved_seq_count = i32::try_from(config.lane_count.saturating_mul(2))
            .unwrap_or(i32::MAX)
            .max(2);
        Self {
            max_entries: cache.max_entries.clamp(1, 512),
            max_bytes: cache.max_bytes,
            min_tokens: cache.min_tokens,
            reserved_seq_count,
        }
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
