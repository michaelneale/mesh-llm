use super::super::{ModelConfig, TensorPrefixes};
use anyhow::Result;
use mlx_rs::Array;
use std::collections::HashMap;

pub(crate) fn transform_gemma3_tensors(
    tensors: &mut HashMap<String, Array>,
    _prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    tensors.retain(|key, _| {
        !key.starts_with("vision_tower.") && !key.starts_with("multi_modal_projector.")
    });

    if config.tie_word_embeddings {
        tensors.remove("language_model.lm_head.weight");
        tensors.remove("language_model.lm_head.scales");
        tensors.remove("language_model.lm_head.biases");
        tensors.remove("language_model.lm_head.bias");
    }

    Ok(())
}
