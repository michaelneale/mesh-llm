use std::{collections::HashMap, sync::Arc, time::Instant};

use super::bytes::{CacheBlockRef, CacheBytesRepr};
use super::CacheBytes;

const DEFAULT_BLOCK_SIZE_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CacheBlobStore {
    block_size: usize,
    physical_bytes: u64,
    blocks: HashMap<String, CacheBlob>,
}

impl Default for CacheBlobStore {
    fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE_BYTES)
    }
}

#[derive(Debug)]
struct CacheBlob {
    bytes: Arc<Vec<u8>>,
    ref_count: u64,
}

impl CacheBlobStore {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            physical_bytes: 0,
            blocks: HashMap::new(),
        }
    }

    pub fn store_bytes(&mut self, bytes: CacheBytes) -> (CacheBytes, CacheDedupeStats) {
        let len = bytes.len;
        let bytes = match bytes.repr {
            CacheBytesRepr::Inline(bytes) => bytes,
            repr @ CacheBytesRepr::Blocks(_) => {
                return (CacheBytes { len, repr }, CacheDedupeStats::default());
            }
        };
        let mut blocks = Vec::new();
        let started = Instant::now();
        let mut stats = CacheDedupeStats {
            hash_bytes: bytes.len() as u64,
            ..CacheDedupeStats::default()
        };
        for chunk in bytes.chunks(self.block_size) {
            stats.block_count = stats.block_count.saturating_add(1);
            let hash = blake3::hash(chunk).to_hex().to_string();
            let entry = self.blocks.entry(hash).or_insert_with(|| {
                self.physical_bytes = self.physical_bytes.saturating_add(chunk.len() as u64);
                stats.new_block_count = stats.new_block_count.saturating_add(1);
                CacheBlob {
                    bytes: Arc::new(chunk.to_vec()),
                    ref_count: 0,
                }
            });
            if entry.ref_count > 0 {
                stats.reused_block_count = stats.reused_block_count.saturating_add(1);
            }
            entry.ref_count = entry.ref_count.saturating_add(1);
            blocks.push(CacheBlockRef::new(entry.bytes.clone()));
        }
        stats.hash_ms = started.elapsed().as_secs_f64() * 1000.0;
        (CacheBytes::blocks(bytes.len() as u64, blocks), stats)
    }

    pub fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheDedupeStats {
    pub hash_ms: f64,
    pub hash_bytes: u64,
    pub block_count: usize,
    pub new_block_count: usize,
    pub reused_block_count: usize,
}

impl CacheDedupeStats {
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            hash_ms: self.hash_ms + other.hash_ms,
            hash_bytes: self.hash_bytes.saturating_add(other.hash_bytes),
            block_count: self.block_count.saturating_add(other.block_count),
            new_block_count: self.new_block_count.saturating_add(other.new_block_count),
            reused_block_count: self
                .reused_block_count
                .saturating_add(other.reused_block_count),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::payload::{CacheBlobStore, ExactStatePayload};

    #[test]
    fn block_store_dedupes_repeated_payload_blocks() {
        let mut blobs = CacheBlobStore::new(4);
        let first = ExactStatePayload::full_state(b"aaaabbbb".to_vec());
        let second = ExactStatePayload::full_state(b"aaaacccc".to_vec());

        let (_, first_stats) = first.dedupe_into(&mut blobs);
        let (second, second_stats) = second.dedupe_into(&mut blobs);

        assert_eq!(first_stats.new_block_count, 2);
        assert_eq!(second_stats.new_block_count, 1);
        assert_eq!(second_stats.reused_block_count, 1);
        assert_eq!(blobs.physical_bytes(), 12);
        assert_eq!(second.byte_len(), 8);
    }
}
