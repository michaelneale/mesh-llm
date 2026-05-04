//! Shared cache primitives for staged serving.
//!
//! `skippy-cache` owns cache identity, payload storage, block dedupe, eviction,
//! and cache accounting. Runtime export/import remains in serving adapters.

pub mod prefix_state;

pub use prefix_state::{
    cache_attrs, eligible_prefill_message, CacheBytesReconstructStats, CacheDedupeStats,
    FullStateCache, FullStateCacheKey, FullStateCacheLookup, FullStateCachePayloadEntry,
    FullStateCacheRecordOutcome, PrefixStateCache, PrefixStateKey, PrefixStateLookup,
    PrefixStatePayloadEntry, PrefixStateRecordOutcome,
};
