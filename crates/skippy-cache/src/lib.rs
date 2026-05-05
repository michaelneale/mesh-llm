pub mod config;
pub mod identity;
pub mod payload;
pub mod resident;

pub use config::{PrefixCandidatePolicy, ResidentCacheConfig};
pub use identity::{
    activation_page_id, prefix_hash, prefix_identity, PrefixIdentity, NATIVE_KV_DTYPE,
    NATIVE_KV_RUNTIME_ABI_VERSION,
};
pub use payload::{
    CacheBytes, CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayload,
    ExactStatePayloadKind,
};
pub use resident::{
    ResidentActivationCache, ResidentActivationLookup, ResidentActivationRecordOutcome,
    ResidentActivationStats, ResidentPrefixAllocation, ResidentPrefixCache,
    ResidentPrefixCacheStats, ResidentPrefixEviction, ResidentPrefixLookup,
};
