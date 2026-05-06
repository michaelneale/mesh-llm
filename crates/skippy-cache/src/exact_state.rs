use std::collections::HashMap;

use crate::{CacheBlobStore, CacheDedupeStats, ExactStatePayload};

#[derive(Debug)]
pub struct ExactStateCache<E> {
    max_entries: usize,
    max_bytes: u64,
    clock: u64,
    logical_bytes: u64,
    blobs: CacheBlobStore,
    entries: HashMap<String, ExactStateEntry<E>>,
}

#[derive(Debug, Clone)]
struct ExactStateEntry<E> {
    token_count: u64,
    logical_bytes: u64,
    last_used: u64,
    payload: ExactStatePayload,
    extra: E,
}

#[derive(Debug, Clone)]
pub struct ExactStateLookup<E> {
    pub page_id: String,
    pub token_count: u64,
    pub logical_bytes: u64,
    pub entries: usize,
    pub payload: ExactStatePayload,
    pub extra: E,
}

#[derive(Debug, Clone)]
pub struct ExactStateRecordOutcome {
    pub stored: bool,
    pub page_id: String,
    pub token_count: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub entries: usize,
    pub evicted_entries: usize,
    pub evicted_logical_bytes: u64,
    pub dedupe: CacheDedupeStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExactStateCacheStats {
    pub entries: usize,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub block_count: usize,
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl<E: Clone> ExactStateCache<E> {
    pub fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            max_entries: max_entries.max(1),
            max_bytes,
            clock: 0,
            logical_bytes: 0,
            blobs: CacheBlobStore::default(),
            entries: HashMap::new(),
        }
    }

    pub fn lookup(&mut self, page_id: &str) -> Option<ExactStateLookup<E>> {
        self.clock = self.clock.saturating_add(1);
        let entries = self.entries.len();
        let entry = self.entries.get_mut(page_id)?;
        entry.last_used = self.clock;
        Some(ExactStateLookup {
            page_id: page_id.to_string(),
            token_count: entry.token_count,
            logical_bytes: entry.logical_bytes,
            entries,
            payload: entry.payload.clone(),
            extra: entry.extra.clone(),
        })
    }

    pub fn record(
        &mut self,
        page_id: String,
        token_count: u64,
        payload: ExactStatePayload,
        extra: E,
    ) -> ExactStateRecordOutcome {
        self.clock = self.clock.saturating_add(1);
        if let Some(previous) = self.entries.remove(&page_id) {
            self.remove_entry(previous);
        }

        let logical_bytes = payload.byte_len();
        let (payload, dedupe) = payload.dedupe_into(&mut self.blobs);
        self.logical_bytes = self.logical_bytes.saturating_add(logical_bytes);
        self.entries.insert(
            page_id.clone(),
            ExactStateEntry {
                token_count,
                logical_bytes,
                last_used: self.clock,
                payload,
                extra,
            },
        );

        let (mut evicted_entries, mut evicted_logical_bytes) = self.evict_until_within_limits();
        if self.max_bytes > 0 && self.blobs.physical_bytes() > self.max_bytes {
            if let Some(entry) = self.entries.remove(&page_id) {
                evicted_entries = evicted_entries.saturating_add(1);
                evicted_logical_bytes = evicted_logical_bytes.saturating_add(entry.logical_bytes);
                self.remove_entry(entry);
            }
        }
        let stored = self.entries.contains_key(&page_id);
        let stats = self.stats();
        ExactStateRecordOutcome {
            stored,
            page_id,
            token_count,
            logical_bytes,
            physical_bytes: stats.physical_bytes,
            entries: stats.entries,
            evicted_entries,
            evicted_logical_bytes,
            dedupe,
        }
    }

    pub fn stats(&self) -> ExactStateCacheStats {
        ExactStateCacheStats {
            entries: self.entries.len(),
            logical_bytes: self.logical_bytes,
            physical_bytes: self.blobs.physical_bytes(),
            block_count: self.blobs.block_count(),
            max_entries: self.max_entries,
            max_bytes: self.max_bytes,
        }
    }

    fn evict_until_within_limits(&mut self) -> (usize, u64) {
        let mut evicted_entries = 0usize;
        let mut evicted_logical_bytes = 0u64;
        loop {
            let over_entries = self.entries.len() > self.max_entries;
            let over_bytes = self.max_bytes > 0 && self.blobs.physical_bytes() > self.max_bytes;
            if !over_entries && !over_bytes {
                break;
            }
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(page_id, _)| page_id.clone())
            else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim) {
                evicted_entries = evicted_entries.saturating_add(1);
                evicted_logical_bytes = evicted_logical_bytes.saturating_add(entry.logical_bytes);
                self.remove_entry(entry);
            }
        }
        (evicted_entries, evicted_logical_bytes)
    }

    fn remove_entry(&mut self, entry: ExactStateEntry<E>) {
        self.logical_bytes = self.logical_bytes.saturating_sub(entry.logical_bytes);
        entry.payload.release_from(&mut self.blobs);
    }
}

#[cfg(test)]
mod tests {
    use crate::{exact_state::ExactStateCache, ExactStatePayload};

    #[test]
    fn exact_state_cache_evicts_lru_by_entry_cap() {
        let mut cache = ExactStateCache::new(1, 0);
        cache.record(
            "first".to_string(),
            2,
            ExactStatePayload::full_state(vec![1, 2]),
            (),
        );
        cache.record(
            "second".to_string(),
            2,
            ExactStatePayload::full_state(vec![3, 4]),
            (),
        );

        assert!(cache.lookup("first").is_none());
        assert!(cache.lookup("second").is_some());
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn exact_state_cache_releases_deduped_blocks_on_eviction() {
        let mut cache = ExactStateCache::new(1, 0);
        cache.record(
            "first".to_string(),
            8,
            ExactStatePayload::full_state(vec![7; 1024 * 1024]),
            (),
        );
        cache.record(
            "second".to_string(),
            8,
            ExactStatePayload::full_state(vec![7; 1024 * 1024]),
            (),
        );

        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().physical_bytes, 1024 * 1024);
        assert_eq!(cache.stats().block_count, 1);
    }
}
