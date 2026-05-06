use std::collections::HashMap;

use crate::ResidentCacheConfig;

#[derive(Debug)]
pub struct ResidentActivationCache<F> {
    max_entries: usize,
    max_bytes: u64,
    min_tokens: u64,
    clock: u64,
    resident_bytes: u64,
    entries: HashMap<String, ResidentActivationEntry<F>>,
}

#[derive(Debug, Clone)]
struct ResidentActivationEntry<F> {
    token_count: u64,
    byte_size: u64,
    last_used: u64,
    frame: F,
}

#[derive(Debug, Clone)]
pub struct ResidentActivationLookup<F> {
    pub token_count: u64,
    pub byte_size: u64,
    pub entries: usize,
    pub frame: F,
}

#[derive(Debug, Clone)]
pub struct ResidentActivationRecordOutcome {
    pub evicted_entries: usize,
    pub evicted_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentActivationStats {
    pub entries: usize,
    pub resident_bytes: u64,
}

impl<F: Clone> ResidentActivationCache<F> {
    pub fn new(config: ResidentCacheConfig) -> Self {
        Self {
            max_entries: config.max_entries,
            max_bytes: config.max_bytes,
            min_tokens: config.min_tokens,
            clock: 0,
            resident_bytes: 0,
            entries: HashMap::new(),
        }
    }

    pub fn lookup(&mut self, page_id: &str) -> Option<ResidentActivationLookup<F>> {
        self.clock = self.clock.saturating_add(1);
        let entries = self.entries.len();
        let entry = self.entries.get_mut(page_id)?;
        entry.last_used = self.clock;
        Some(ResidentActivationLookup {
            token_count: entry.token_count,
            byte_size: entry.byte_size,
            entries,
            frame: entry.frame.clone(),
        })
    }

    pub fn record(
        &mut self,
        page_id: String,
        token_count: u64,
        byte_size: u64,
        frame: F,
    ) -> ResidentActivationRecordOutcome {
        if token_count < self.min_tokens || byte_size == 0 {
            return ResidentActivationRecordOutcome {
                evicted_entries: 0,
                evicted_bytes: 0,
            };
        }
        self.clock = self.clock.saturating_add(1);
        if let Some(previous) = self.entries.remove(&page_id) {
            self.resident_bytes = self.resident_bytes.saturating_sub(previous.byte_size);
        }
        let (evicted_entries, evicted_bytes) = self.evict_until_room_for(byte_size);
        self.resident_bytes = self.resident_bytes.saturating_add(byte_size);
        self.entries.insert(
            page_id,
            ResidentActivationEntry {
                token_count,
                byte_size,
                last_used: self.clock,
                frame,
            },
        );
        ResidentActivationRecordOutcome {
            evicted_entries,
            evicted_bytes,
        }
    }

    pub fn stats(&self) -> ResidentActivationStats {
        ResidentActivationStats {
            entries: self.entries.len(),
            resident_bytes: self.resident_bytes,
        }
    }

    fn evict_until_room_for(&mut self, byte_size: u64) -> (usize, u64) {
        let mut evicted_entries = 0usize;
        let mut evicted_bytes = 0u64;
        loop {
            let over_entries = self.entries.len().saturating_add(1) > self.max_entries;
            let over_bytes = self.max_bytes > 0
                && self.resident_bytes.saturating_add(byte_size) > self.max_bytes;
            if !over_entries && !over_bytes {
                break;
            }
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim) {
                self.resident_bytes = self.resident_bytes.saturating_sub(entry.byte_size);
                evicted_entries = evicted_entries.saturating_add(1);
                evicted_bytes = evicted_bytes.saturating_add(entry.byte_size);
            }
        }
        (evicted_entries, evicted_bytes)
    }
}
