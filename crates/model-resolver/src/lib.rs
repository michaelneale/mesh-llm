pub mod catalog;
pub mod hf_catalog;
pub mod refs;
pub mod resolve;
pub mod types;

pub use catalog::{
    CatalogEntry, CatalogPackage, CatalogSidecarAsset, CatalogSidecarRef, CatalogSource,
    CatalogVariant, CuratedMeta,
};
pub use hf_catalog::HfCatalogProvider;
pub use refs::{
    file_basename, format_huggingface_display_ref, format_huggingface_exact_ref,
    huggingface_resolve_url, is_primary_mlx_weight_file, is_split_mlx_first_shard,
    parse_hf_resolve_url, parse_huggingface_file_ref, parse_huggingface_repo_ref,
    parse_huggingface_repo_url, remote_filename,
};
pub use resolve::{CatalogProvider, MemoryCatalog, ModelResolver};
pub use types::{
    LocalGguf, LocalLayerPackage, ModelArtifactCandidate, RemoteGguf, RemoteLayerPackage,
};
