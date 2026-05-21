use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::ResidentCacheConfig;

#[derive(Debug)]
pub struct ResidentPrefixCache {
    max_entries: usize,
    max_bytes: u64,
    /// 0 means "unlimited (legacy)".
    max_resident_tokens: u64,
    min_tokens: u64,
    reserved_seq_count: i32,
    next_seq_id: i32,
    clock: u64,
    resident_tokens: u64,
    estimated_bytes: u64,
    entries: HashMap<String, ResidentPrefixEntry>,
    free_seq_ids: Vec<i32>,
}

#[derive(Debug)]
struct ResidentPrefixEntry {
    seq_id: i32,
    token_count: u64,
    estimated_bytes: u64,
    last_used: u64,
    borrowed: bool,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixLookup {
    pub seq_id: i32,
    pub entries: usize,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixEviction {
    pub page_id: String,
    pub seq_id: i32,
    pub token_count: u64,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixAllocation {
    pub seq_id: i32,
    pub evictions: Vec<ResidentPrefixEviction>,
    pub should_save: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentPrefixCacheStats {
    pub entries: usize,
    pub resident_tokens: u64,
    pub estimated_bytes: u64,
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl ResidentPrefixCache {
    pub fn new(config: ResidentCacheConfig) -> Self {
        Self {
            max_entries: config.max_entries,
            max_bytes: config.max_bytes,
            max_resident_tokens: config.max_resident_tokens,
            min_tokens: config.min_tokens,
            reserved_seq_count: config.reserved_seq_count,
            next_seq_id: config.reserved_seq_count,
            clock: 0,
            resident_tokens: 0,
            estimated_bytes: 0,
            entries: HashMap::new(),
            free_seq_ids: Vec::new(),
        }
    }

    pub fn lookup(&mut self, page_id: &str) -> Option<ResidentPrefixLookup> {
        self.clock = self.clock.saturating_add(1);
        let entries = self.entries.len();
        let entry = self.entries.get_mut(page_id)?;
        if entry.borrowed {
            return None;
        }
        entry.last_used = self.clock;
        Some(ResidentPrefixLookup {
            seq_id: entry.seq_id,
            entries,
        })
    }

    pub fn acquire(&mut self, page_id: &str) -> Option<ResidentPrefixLookup> {
        self.clock = self.clock.saturating_add(1);
        let entries = self.entries.len();
        let entry = self.entries.get_mut(page_id)?;
        if entry.borrowed {
            return None;
        }
        entry.borrowed = true;
        entry.last_used = self.clock;
        Some(ResidentPrefixLookup {
            seq_id: entry.seq_id,
            entries,
        })
    }

    pub fn release(&mut self, page_id: &str) {
        if let Some(entry) = self.entries.get_mut(page_id) {
            entry.borrowed = false;
            self.clock = self.clock.saturating_add(1);
            entry.last_used = self.clock;
        }
    }

    pub fn allocate_for_record(
        &mut self,
        page_id: &str,
        token_count: u64,
        estimated_bytes: u64,
        mut drop_evicted: impl FnMut(i32) -> Result<()>,
    ) -> Result<ResidentPrefixAllocation> {
        if token_count < self.min_tokens {
            bail!("resident prefix has fewer tokens than cache minimum");
        }
        self.clock = self.clock.saturating_add(1);
        if let Some(entry) = self.entries.get(page_id) {
            return Ok(ResidentPrefixAllocation {
                seq_id: entry.seq_id,
                evictions: Vec::new(),
                should_save: false,
            });
        }

        let evictions =
            self.evict_until_room_for(estimated_bytes, token_count, &mut drop_evicted)?;
        let seq_id = self.next_sequence_id()?;
        Ok(ResidentPrefixAllocation {
            seq_id,
            evictions,
            should_save: true,
        })
    }

    pub fn commit_record(
        &mut self,
        page_id: String,
        seq_id: i32,
        token_count: u64,
        estimated_bytes: u64,
    ) {
        self.clock = self.clock.saturating_add(1);
        if let Some(previous) = self.entries.remove(&page_id) {
            self.resident_tokens = self.resident_tokens.saturating_sub(previous.token_count);
            self.estimated_bytes = self
                .estimated_bytes
                .saturating_sub(previous.estimated_bytes);
        }
        self.resident_tokens = self.resident_tokens.saturating_add(token_count);
        self.estimated_bytes = self.estimated_bytes.saturating_add(estimated_bytes);
        self.entries.insert(
            page_id,
            ResidentPrefixEntry {
                seq_id,
                token_count,
                estimated_bytes,
                last_used: self.clock,
                borrowed: false,
            },
        );
    }

    pub fn stats(&self) -> ResidentPrefixCacheStats {
        ResidentPrefixCacheStats {
            entries: self.entries.len(),
            resident_tokens: self.resident_tokens,
            estimated_bytes: self.estimated_bytes,
            max_entries: self.max_entries,
            max_bytes: self.max_bytes,
        }
    }

    fn evict_until_room_for(
        &mut self,
        estimated_bytes: u64,
        token_count: u64,
        drop_evicted: &mut impl FnMut(i32) -> Result<()>,
    ) -> Result<Vec<ResidentPrefixEviction>> {
        let mut evictions = Vec::new();
        loop {
            let over_entries = self.entries.len().saturating_add(1) > self.max_entries;
            let over_bytes = self.max_bytes > 0
                && self.estimated_bytes.saturating_add(estimated_bytes) > self.max_bytes;
            // Under unified-KV serving the prefix cache shares the
            // `n_ctx` cell pool with the active lanes. `max_entries`
            // and `max_bytes` alone do not bound the cell footprint:
            // 12 entries averaging 10k tokens each pin 120k cells in
            // a 131k-`n_ctx` pool with no LRU pressure. Add an
            // explicit token budget so eviction kicks in before the
            // cells run out.
            let over_tokens = self.max_resident_tokens > 0
                && self.resident_tokens.saturating_add(token_count) > self.max_resident_tokens;
            if !over_entries && !over_bytes && !over_tokens {
                break;
            }
            let Some(victim) = self
                .entries
                .iter()
                .filter(|(_, entry)| !entry.borrowed)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                bail!("resident prefix cache has no releasable entries");
            };
            if let Some(entry) = self.entries.remove(&victim) {
                drop_evicted(entry.seq_id)?;
                self.free_seq_ids.push(entry.seq_id);
                self.resident_tokens = self.resident_tokens.saturating_sub(entry.token_count);
                self.estimated_bytes = self.estimated_bytes.saturating_sub(entry.estimated_bytes);
                evictions.push(ResidentPrefixEviction {
                    page_id: victim,
                    seq_id: entry.seq_id,
                    token_count: entry.token_count,
                });
            }
        }
        Ok(evictions)
    }

    fn next_sequence_id(&mut self) -> Result<i32> {
        if let Some(seq_id) = self.free_seq_ids.pop() {
            return Ok(seq_id);
        }
        let seq_id = self.next_seq_id;
        self.next_seq_id = self
            .next_seq_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("resident prefix sequence id overflow"))?;
        if seq_id < self.reserved_seq_count || seq_id >= 1024 {
            bail!("resident prefix sequence id capacity exhausted");
        }
        Ok(seq_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_entries: usize, max_bytes: u64, max_resident_tokens: u64) -> ResidentCacheConfig {
        ResidentCacheConfig {
            max_entries,
            max_bytes,
            max_resident_tokens,
            min_tokens: 256,
            reserved_seq_count: 2,
        }
    }

    #[test]
    fn token_budget_triggers_lru_before_entry_cap_under_unified_kv() {
        // Regression: under skippy `kv_unified = true` the prefix cache
        // shares the model's `n_ctx` cell pool with the active lanes.
        // Before this fix, the cache only evicted on `max_entries` and
        // `max_bytes`. Live data on a 131k-`n_ctx` MiniMax showed the
        // cache happily filling to ~124k pinned tokens (~95% of the
        // pool) across just 12 entries with no LRU pressure, starving
        // the lanes and surfacing as HTTP 502
        // `RuntimeError: llama_decode failed`.
        //
        // With `max_resident_tokens = 4096` and entries of 1500 tokens
        // each, the third record must trigger eviction even though
        // we're well under `max_entries = 16`.
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 4096));
        let mut dropped: Vec<i32> = Vec::new();

        let alloc1 = cache
            .allocate_for_record("page-1", 1500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        assert!(alloc1.should_save);
        cache.commit_record("page-1".to_string(), alloc1.seq_id, 1500, 100);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().resident_tokens, 1500);

        let alloc2 = cache
            .allocate_for_record("page-2", 1500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        cache.commit_record("page-2".to_string(), alloc2.seq_id, 1500, 100);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().resident_tokens, 3000);
        // Two entries fit; we are at 3000 / 4096 tokens.
        assert!(dropped.is_empty(), "should not have evicted yet");

        let alloc3 = cache
            .allocate_for_record("page-3", 1500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        cache.commit_record("page-3".to_string(), alloc3.seq_id, 1500, 100);
        // 3000 + 1500 = 4500 > 4096 — must have evicted at least one
        // entry to make room. LRU picks page-1.
        assert_eq!(dropped, vec![alloc1.seq_id], "LRU should evict oldest");
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().resident_tokens, 3000);
    }

    #[test]
    fn small_ctx_smoke_test_scenario_records_without_eviction_loop() {
        // Regression for skippy-ci-smoke `prompt exact-prefix hit and
        // live-session reuse` (CI run 26193173851). The smoke test
        // ships SmolLM2-135M with `PROMPT_CTX_SIZE=768` and a 533-token
        // prompt. With a naive `n_ctx / 2 = 384` cap, the cap was
        // *smaller than a single prompt*, so the first record attempt
        // would call `evict_until_room_for` with `over_tokens`
        // permanently true on an empty cache and bail with
        // "no releasable entries".
        //
        // `ResidentCacheConfig::from_stage` derives the cap from the
        // model's `n_ctx` via `derive_max_resident_tokens`, which uses
        // a hard `MIN_CTX_FOR_CELL_CAP = 8192` floor: below that, the
        // cap is 0 (disabled) regardless of `min_tokens`. The smoke
        // test runs with `ctx_size = 768`, well below the floor, so
        // its derived cap is 0. This unit test pins that path: cap=0,
        // record a 533-token prompt against a 256-token min, no
        // eviction loop.
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let alloc = cache
            .allocate_for_record("page-0", 533, 100, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc.seq_id, 533, 100);
        assert_eq!(cache.stats().resident_tokens, 533);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn zero_token_budget_disables_the_check() {
        // max_resident_tokens = 0 means "unlimited" — legacy behavior.
        // 12 entries at 10k tokens each = 120k tokens, which the
        // previous unbounded cache happily accepted.
        let mut cache = ResidentPrefixCache::new(cfg(64, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();

        for i in 0..12 {
            let alloc = cache
                .allocate_for_record(&format!("page-{i}"), 10_000, 100, |sid| {
                    dropped.push(sid);
                    Ok(())
                })
                .unwrap();
            cache.commit_record(format!("page-{i}"), alloc.seq_id, 10_000, 100);
        }
        assert!(
            dropped.is_empty(),
            "zero token budget should not trigger evictions"
        );
        assert_eq!(cache.stats().entries, 12);
        assert_eq!(cache.stats().resident_tokens, 120_000);
    }
}
