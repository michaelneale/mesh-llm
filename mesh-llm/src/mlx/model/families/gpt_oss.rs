use super::super::{ModelConfig, TensorPrefixes};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::collections::HashMap;

fn split_even_odd_axis(tensor: &Array, axis: i32) -> Result<(Array, Array)> {
    let shape = tensor.shape();
    let ndim = shape.len() as i32;
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        anyhow::bail!("axis {axis} out of bounds for GPT-OSS tensor shape {shape:?}");
    }
    let axis_len = shape[axis as usize];
    let even_idx: Vec<u32> = (0..axis_len).step_by(2).map(|idx| idx as u32).collect();
    let odd_idx: Vec<u32> = (1..axis_len).step_by(2).map(|idx| idx as u32).collect();
    Ok((
        tensor.take_axis(
            &Array::from_slice(&even_idx, &[even_idx.len() as i32]),
            axis,
        )?,
        tensor.take_axis(&Array::from_slice(&odd_idx, &[odd_idx.len() as i32]), axis)?,
    ))
}

fn split_gate_up_proj(prefix: &str, tensors: &mut HashMap<String, Array>) -> Result<()> {
    if tensors.contains_key(&format!("{prefix}.gate_proj.weight")) {
        return Ok(());
    }

    for suffix in ["weight", "scales", "biases"] {
        let key = format!("{prefix}.gate_up_proj.{suffix}");
        if let Some(fused) = tensors.get(&key).cloned() {
            let (gate, up) = split_even_odd_axis(&fused, -2)?;
            tensors.insert(format!("{prefix}.gate_proj.{suffix}"), gate);
            tensors.insert(format!("{prefix}.up_proj.{suffix}"), up);
        }
    }

    let bias_key = format!("{prefix}.gate_up_proj_bias");
    if let Some(fused_bias) = tensors.get(&bias_key).cloned() {
        let (gate_bias, up_bias) = split_even_odd_axis(&fused_bias, -1)?;
        tensors.insert(format!("{prefix}.gate_proj.bias"), gate_bias);
        tensors.insert(format!("{prefix}.up_proj.bias"), up_bias);
    }

    Ok(())
}

fn normalize_down_proj_bias(prefix: &str, tensors: &mut HashMap<String, Array>) {
    let legacy_key = format!("{prefix}.down_proj_bias");
    let normalized_key = format!("{prefix}.down_proj.bias");
    if let Some(bias) = tensors.get(&legacy_key).cloned() {
        tensors.entry(normalized_key).or_insert(bias);
    }
}

pub(crate) fn transform_gpt_oss_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    for i in 0..config.num_hidden_layers {
        let mlp_prefix = format!("{}.layers.{i}.mlp.experts", prefixes.model);
        if tensors
            .keys()
            .any(|key| key.starts_with(&format!("{mlp_prefix}.gate_up_proj")))
        {
            split_gate_up_proj(&mlp_prefix, tensors)
                .with_context(|| format!("failed to sanitize GPT-OSS tensors for {mlp_prefix}"))?;
        }
        normalize_down_proj_bias(&mlp_prefix, tensors);
    }

    Ok(())
}
