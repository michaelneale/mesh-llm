use std::{borrow::Cow, fmt};

use anyhow::{anyhow, Result};

mod blob_store;
pub(super) mod bytes;

pub use blob_store::{CacheBlobStore, CacheDedupeStats};
pub use bytes::{CacheBytes, CacheBytesReconstructStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactStatePayloadKind {
    FullState,
    RecurrentOnly,
    KvRecurrent,
}

#[derive(Debug, Clone)]
pub enum ExactStatePayload {
    FullState {
        bytes: CacheBytes,
    },
    RecurrentOnly {
        recurrent: CacheBytes,
    },
    KvRecurrent {
        kv: CacheBytes,
        recurrent: CacheBytes,
    },
}

impl ExactStatePayload {
    pub fn full_state(bytes: Vec<u8>) -> Self {
        Self::FullState {
            bytes: CacheBytes::inline(bytes),
        }
    }

    pub fn recurrent_only(recurrent: Vec<u8>) -> Self {
        Self::RecurrentOnly {
            recurrent: CacheBytes::inline(recurrent),
        }
    }

    pub fn kv_recurrent(kv: Vec<u8>, recurrent: Vec<u8>) -> Self {
        Self::KvRecurrent {
            kv: CacheBytes::inline(kv),
            recurrent: CacheBytes::inline(recurrent),
        }
    }

    pub fn kind(&self) -> ExactStatePayloadKind {
        match self {
            Self::FullState { .. } => ExactStatePayloadKind::FullState,
            Self::RecurrentOnly { .. } => ExactStatePayloadKind::RecurrentOnly,
            Self::KvRecurrent { .. } => ExactStatePayloadKind::KvRecurrent,
        }
    }

    pub fn byte_len(&self) -> u64 {
        match self {
            Self::FullState { bytes } => bytes.len(),
            Self::RecurrentOnly { recurrent } => recurrent.len(),
            Self::KvRecurrent { kv, recurrent } => kv.len().saturating_add(recurrent.len()),
        }
    }

    pub fn recurrent_state_bytes(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            Self::RecurrentOnly { recurrent } | Self::KvRecurrent { recurrent, .. } => {
                recurrent.as_cow()
            }
            _ => Err(anyhow!("cache payload has no recurrent component")),
        }
    }

    pub fn recurrent_state_bytes_timed(
        &self,
    ) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        match self {
            Self::RecurrentOnly { recurrent } | Self::KvRecurrent { recurrent, .. } => {
                recurrent.as_cow_timed()
            }
            _ => Err(anyhow!("cache payload has no recurrent component")),
        }
    }

    pub fn full_state_bytes_timed(&self) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        match self {
            Self::FullState { bytes } => bytes.as_cow_timed(),
            _ => Err(anyhow!("cache payload is not full-state")),
        }
    }

    pub fn kv_bytes(&self) -> Result<Option<Cow<'_, [u8]>>> {
        match self {
            Self::KvRecurrent { kv, .. } => Ok(Some(kv.as_cow()?)),
            _ => Ok(None),
        }
    }

    pub fn kv_bytes_timed(&self) -> Result<Option<(Cow<'_, [u8]>, CacheBytesReconstructStats)>> {
        match self {
            Self::KvRecurrent { kv, .. } => Ok(Some(kv.as_cow_timed()?)),
            _ => Ok(None),
        }
    }

    pub fn dedupe_into(self, blobs: &mut CacheBlobStore) -> (Self, CacheDedupeStats) {
        match self {
            Self::FullState { bytes } => {
                let (bytes, stats) = blobs.store_bytes(bytes);
                (Self::FullState { bytes }, stats)
            }
            Self::RecurrentOnly { recurrent } => {
                let (recurrent, stats) = blobs.store_bytes(recurrent);
                (Self::RecurrentOnly { recurrent }, stats)
            }
            Self::KvRecurrent { kv, recurrent } => {
                let (kv, kv_stats) = blobs.store_bytes(kv);
                let (recurrent, recurrent_stats) = blobs.store_bytes(recurrent);
                (
                    Self::KvRecurrent { kv, recurrent },
                    kv_stats.saturating_add(recurrent_stats),
                )
            }
        }
    }

    pub fn release_from(&self, blobs: &mut CacheBlobStore) {
        match self {
            Self::FullState { bytes } => blobs.release_bytes(bytes),
            Self::RecurrentOnly { recurrent } => blobs.release_bytes(recurrent),
            Self::KvRecurrent { kv, recurrent } => {
                blobs.release_bytes(kv);
                blobs.release_bytes(recurrent);
            }
        }
    }
}

impl fmt::Display for ExactStatePayloadKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullState => f.write_str("full-state"),
            Self::RecurrentOnly => f.write_str("recurrent-only"),
            Self::KvRecurrent => f.write_str("kv-recurrent"),
        }
    }
}
