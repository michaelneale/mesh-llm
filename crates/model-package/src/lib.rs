pub mod jobs;
pub mod permissions;
pub mod prepare;
pub mod script;

use anyhow::{Context, Result};

/// Build an `HFClient` from environment, suitable for API calls.
///
/// Resolves endpoint from `HF_ENDPOINT` and token from `HF_TOKEN` /
/// `HUGGING_FACE_HUB_TOKEN`.
pub fn build_hf_client() -> Result<hf_hub::HFClient> {
    let mut builder =
        hf_hub::HFClientBuilder::new().cache_dir(model_hf::huggingface_hub_cache_dir());

    if let Some(endpoint) = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        builder = builder.endpoint(endpoint);
    }

    if let Some(token) = model_hf::hf_token_override() {
        builder = builder.token(token);
    }

    builder.build().context("build HuggingFace API client")
}
