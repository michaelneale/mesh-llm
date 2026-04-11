use super::super::{ModelConfig, TensorPrefixes};
use anyhow::Result;
use mlx_rs::Array;
use std::collections::HashMap;

pub(crate) fn transform_gemma4_tensors(
    tensors: &mut HashMap<String, Array>,
    _prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    let mut normalized = HashMap::with_capacity(tensors.len());
    for (key, value) in tensors.drain() {
        let starts_w_model = key.starts_with("model.");
        let key = key.strip_prefix("model.").unwrap_or(&key).to_string();

        if key.starts_with("vision_tower.")
            || key.starts_with("multi_modal_projector.")
            || key.starts_with("audio_tower.")
            || key.starts_with("embed_audio.")
            || key.starts_with("embed_vision.")
        {
            continue;
        }

        let normalized_key = if starts_w_model && key.starts_with("language_model.") {
            key.replacen("language_model.", "language_model.model.", 1)
        } else {
            key
        };
        normalized.insert(normalized_key, value);
    }
    *tensors = normalized;

    if config.tie_word_embeddings {
        tensors.remove("lm_head.weight");
        tensors.remove("lm_head.scales");
        tensors.remove("lm_head.biases");
        tensors.remove("lm_head.bias");
    }

    Ok(())
}
