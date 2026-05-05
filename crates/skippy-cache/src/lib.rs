pub mod config;
pub mod exact_state;
pub mod identity;
pub mod payload;
pub mod resident;

pub use config::{PrefixCandidatePolicy, ResidentCacheConfig};
pub use exact_state::{
    ExactStateCache, ExactStateCacheStats, ExactStateLookup, ExactStateRecordOutcome,
};
pub use identity::{
    activation_page_id, prefix_hash, prefix_identity, PrefixIdentity, NATIVE_KV_DTYPE,
    NATIVE_KV_RUNTIME_ABI_VERSION,
};
pub use payload::{
    CacheBlobStore, CacheBytes, CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayload,
    ExactStatePayloadKind,
};
pub use resident::{
    ResidentActivationCache, ResidentActivationLookup, ResidentActivationRecordOutcome,
    ResidentActivationStats, ResidentPrefixAllocation, ResidentPrefixCache,
    ResidentPrefixCacheStats, ResidentPrefixEviction, ResidentPrefixLookup,
};
