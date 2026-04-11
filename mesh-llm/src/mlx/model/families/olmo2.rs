use super::super::{ModelConfig, TensorPrefixes};
use anyhow::Result;
use mlx_rs::Array;
use std::collections::HashMap;

pub(crate) fn transform_olmo2_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    for i in 0..config.num_hidden_layers {
        tensors.remove(&format!(
            "{}.layers.{i}.self_attn.rotary_emb.inv_freq",
            prefixes.model
        ));
    }
    Ok(())
}
