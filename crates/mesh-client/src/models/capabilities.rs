use super::catalog;
pub use mesh_llm_types::models::capabilities::{
    merge_config_signals, merge_name_signals, merge_sibling_signals, CapabilityLevel,
    ModelCapabilities,
};
use serde_json::Value;
use std::path::Path;

pub fn infer_catalog_capabilities(model: &catalog::CatalogModel) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    if model.mmproj.is_some() {
        caps.upgrade_vision(CapabilityLevel::Supported);
    }
    caps = merge_name_signals(
        caps,
        &[
            model.name.as_str(),
            model.file.as_str(),
            model.description.as_str(),
        ],
    );
    caps.normalize()
}

pub fn infer_local_model_capabilities(
    model_name: &str,
    path: &Path,
    catalog_entry: Option<&catalog::CatalogModel>,
) -> ModelCapabilities {
    let mut caps = catalog_entry
        .map(infer_catalog_capabilities)
        .unwrap_or_default();
    caps = merge_name_signals(
        caps,
        &[
            model_name,
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        ],
    );
    for config in read_local_metadata_jsons(path) {
        caps = merge_config_signals(caps, &config);
    }
    caps.normalize()
}

fn read_local_metadata_jsons(path: &Path) -> Vec<Value> {
    let mut values = Vec::new();
    for dir in path.ancestors().skip(1).take(6) {
        for name in ["config.json", "tokenizer_config.json", "chat_template.json"] {
            let candidate = dir.join(name);
            if !candidate.is_file() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&candidate) else {
                continue;
            };
            if let Ok(value) = serde_json::from_str(&text) {
                values.push(value);
            }
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::{merge_name_signals, CapabilityLevel};

    #[test]
    fn qwen3vl_name_signal_is_supported_vision() {
        let caps = merge_name_signals(
            Default::default(),
            &[
                "Qwen3VL-2B-Instruct-Q4_K_M",
                "Qwen/Qwen3-VL-2B-Instruct-GGUF",
            ],
        );
        assert_eq!(caps.vision, CapabilityLevel::Supported);
        assert!(caps.multimodal);
    }
}
