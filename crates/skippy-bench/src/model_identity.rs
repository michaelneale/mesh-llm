use std::path::Path;

use anyhow::{Context, Result};
use model_artifact::ModelIdentity;
use model_hf::HfModelRepository;
use model_ref::ModelRef;

pub fn explicit_model_identity(model_id: &str) -> Result<ModelIdentity> {
    let model_ref = ModelRef::parse(model_id)
        .with_context(|| format!("--model-id must be a model coordinate, got {model_id:?}"))?;
    Ok(ModelIdentity::from_model_id(model_ref.display_id()))
}

pub fn model_identity_for_path(model_id: &str, model_path: Option<&Path>) -> Result<ModelIdentity> {
    let explicit = explicit_model_identity(model_id)?;
    let Some(model_path) = model_path else {
        return Ok(explicit);
    };
    Ok(hf_identity_for_path(model_path).unwrap_or(explicit))
}

fn hf_identity_for_path(model_path: &Path) -> Option<ModelIdentity> {
    HfModelRepository::from_env()
        .ok()
        .and_then(|repository| repository.identity_for_path(model_path))
        .map(|identity| identity.to_model_identity())
}
