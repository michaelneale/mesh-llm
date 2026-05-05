use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::ResidentCacheConfig;

#[derive(Debug)]
pub struct ResidentPrefixCache {
    max_entries: usize,
    max_bytes: u64,
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
        entry.last_used = self.clock;
        Some(ResidentPrefixLookup {
            seq_id: entry.seq_id,
            entries,
        })
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
            });
        }

        let evictions = self.evict_until_room_for(estimated_bytes, &mut drop_evicted)?;
        let seq_id = self.next_sequence_id()?;
        Ok(ResidentPrefixAllocation { seq_id, evictions })
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
        drop_evicted: &mut impl FnMut(i32) -> Result<()>,
    ) -> Result<Vec<ResidentPrefixEviction>> {
        let mut evictions = Vec::new();
        loop {
            let over_entries = self.entries.len().saturating_add(1) > self.max_entries;
            let over_bytes = self.max_bytes > 0
                && self.estimated_bytes.saturating_add(estimated_bytes) > self.max_bytes;
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
