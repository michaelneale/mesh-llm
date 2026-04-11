use super::super::{ModelConfig, TensorPrefixes};
use anyhow::Result;
use mlx_rs::Array;
use std::collections::HashMap;

pub(crate) fn transform_llama_like_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    tensors.retain(|key, _| !key.contains("self_attn.rotary_emb.inv_freq"));

    if config.tie_word_embeddings {
        if let Some(prefix) = prefixes.lm_head.as_deref() {
            tensors.remove(&format!("{prefix}.weight"));
            tensors.remove(&format!("{prefix}.scales"));
            tensors.remove(&format!("{prefix}.biases"));
            tensors.remove(&format!("{prefix}.bias"));
        }
    }

    Ok(())
}
