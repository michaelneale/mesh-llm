pub mod capabilities;
pub mod catalog;
pub mod gguf;
pub mod inventory;
pub mod local;
mod maintenance;
mod resolve;
pub mod search;
pub mod topology;

use anyhow::{Context, Result};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::api::tokio::{Api as TokioApi, ApiBuilder as TokioApiBuilder};

pub use capabilities::{CapabilityLevel, ModelCapabilities};
pub use inventory::{scan_local_inventory_snapshot_with_progress, LocalModelInventorySnapshot};
pub use local::{
    find_mmproj_path, find_model_path, huggingface_hub_cache, huggingface_hub_cache_dir,
    huggingface_identity_for_path, mesh_llm_cache_dir, resolve_mmproj_path, scan_installed_models,
    scan_local_models,
};
pub use maintenance::{run_update, warn_about_updates_for_paths};
pub(crate) use resolve::resolve_model_spec_with_progress;
pub use resolve::{
    download_exact_ref, find_catalog_model_exact, installed_model_capabilities,
    installed_model_display_name, resolve_huggingface_model_identity, resolve_model_spec,
    show_exact_model, show_model_variants_with_progress, ModelDetails, ShowVariantsProgress,
};
pub use search::{
    search_catalog_models, search_huggingface, SearchArtifactFilter, SearchHit, SearchProgress,
};
pub use topology::{infer_local_model_topology, ModelMoeInfo, ModelTopology};

pub(crate) fn build_hf_api(progress: bool) -> Result<Api> {
    let mut builder = ApiBuilder::from_cache(huggingface_hub_cache()).with_progress(progress);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        let endpoint = endpoint.trim();
        if !endpoint.is_empty() {
            builder = builder.with_endpoint(endpoint.to_string());
        }
    }
    builder = builder.with_token(hf_token_override());
    builder.build().context("Build Hugging Face API client")
}

pub(crate) fn build_hf_tokio_api(progress: bool) -> Result<TokioApi> {
    let mut builder = TokioApiBuilder::from_cache(huggingface_hub_cache()).with_progress(progress);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        let endpoint = endpoint.trim();
        if !endpoint.is_empty() {
            builder = builder.with_endpoint(endpoint.to_string());
        }
    }
    builder = builder.with_token(hf_token_override());
    builder
        .build()
        .context("Build Hugging Face async API client")
}

pub(crate) fn hf_token_override() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(key) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

fn format_size_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0}MB", bytes as f64 / 1e6)
    }
}

fn short_revision(revision: &str) -> String {
    if revision.len() <= 12 {
        revision.to_string()
    } else {
        revision[..12].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::maintenance::cache_repo_id_from_dir;
    use crate::models::resolve::{parse_hf_resolve_url, parse_huggingface_ref};

    #[test]
    fn parse_hf_resolve_url_extracts_repo_revision_and_file() {
        let (repo, revision, file) = parse_hf_resolve_url(
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        )
        .unwrap();
        assert_eq!(repo, "Qwen/Qwen3-8B-GGUF");
        assert_eq!(revision.as_deref(), Some("main"));
        assert_eq!(file, "Qwen3-8B-Q4_K_M.gguf");
    }

    #[test]
    fn cache_repo_id_from_dir_decodes_hf_cache_names() {
        assert_eq!(
            cache_repo_id_from_dir("models--Qwen--Qwen3-8B-GGUF"),
            Some("Qwen/Qwen3-8B-GGUF".to_string())
        );
    }

    #[test]
    fn parse_huggingface_ref_accepts_revision_shorthand() {
        let (repo, revision, file) =
            parse_huggingface_ref("Qwen/Qwen3-8B-GGUF@main/Qwen3-8B-Q4_K_M.gguf").unwrap();
        assert_eq!(repo, "Qwen/Qwen3-8B-GGUF");
        assert_eq!(revision.as_deref(), Some("main"));
        assert_eq!(file, "Qwen3-8B-Q4_K_M.gguf");
    }

    #[test]
    fn find_catalog_model_exact_matches_filename_stem() {
        let model = find_catalog_model_exact("Qwen3-8B-Q4_K_M").unwrap();
        assert_eq!(model.name, "Qwen3-8B-Q4_K_M");
    }
}
