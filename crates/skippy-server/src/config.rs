use std::path::PathBuf;

use anyhow::{bail, Result};
use serde_json::{json, Value};
use skippy_protocol::{LoadMode, StageConfig, StageTopology};

use crate::package::is_hf_package_ref;

pub fn validate_config(config: &StageConfig, topology: Option<&StageTopology>) -> Result<()> {
    if config.layer_start >= config.layer_end {
        bail!("layer_start must be less than layer_end");
    }
    if config.lane_count == 0 {
        bail!("lane_count must be greater than zero");
    }
    if config
        .selected_device
        .as_ref()
        .is_some_and(|device| device.backend_device.is_empty())
    {
        bail!("selected_device.backend_device must not be empty");
    }
    if config.projector_path.as_deref().is_some_and(str::is_empty) {
        bail!("projector_path must not be empty");
    }
    match config.load_mode {
        LoadMode::RuntimeSlice => {}
        LoadMode::ArtifactSlice => {
            if !config.filter_tensors_on_load {
                bail!("artifact-slice load mode requires filter_tensors_on_load=true")
            }
        }
        LoadMode::LayerPackage => {
            if !config.filter_tensors_on_load {
                bail!("layer-package load mode requires filter_tensors_on_load=true")
            }
            let Some(model_path) = config.model_path.as_ref() else {
                bail!("layer-package load mode requires model_path to point at a package directory")
            };
            if !is_hf_package_ref(model_path) && !std::path::Path::new(model_path).is_dir() {
                bail!("layer-package model_path must be a package directory")
            }
        }
    }
    if let Some(topology) = topology {
        if topology.topology_id != config.topology_id {
            bail!("topology_id mismatch between config and topology");
        }
        if topology.model_id != config.model_id {
            bail!("model_id mismatch between config and topology");
        }
        let Some(stage) = topology
            .stages
            .iter()
            .find(|stage| stage.stage_id == config.stage_id)
        else {
            bail!("stage_id not found in topology");
        };
        if stage.stage_index != config.stage_index
            || stage.layer_start != config.layer_start
            || stage.layer_end != config.layer_end
            || stage.load_mode != config.load_mode
        {
            bail!("stage config does not match topology entry");
        }
    }
    Ok(())
}

pub fn load_json<T>(path: &PathBuf) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let contents = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

pub fn example_config() -> Value {
    json!({
        "run_id": "run-local",
        "topology_id": "single-stage-fixture",
        "model_id": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "model_path": null,
        "projector_path": null,
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "ctx_size": 512,
        "lane_count": 4,
        "n_gpu_layers": 0,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "filter_tensors_on_load": false,
        "load_mode": "runtime-slice",
        "bind_addr": "127.0.0.1:19000",
        "upstream": null,
        "downstream": null
    })
}
