//! Local model inventory — scans GGUF files on disk and extracts metadata.
//!
//! These helpers run at startup and on gossip to report what models are
//! available locally, along with their sizes and quantization metadata.

use std::collections::HashMap;

/// Scan model directories for GGUF files and return a map of stem name to file size in bytes.
pub fn scan_local_model_sizes() -> HashMap<String, u64> {
    let mut sizes: HashMap<String, u64> = HashMap::new();
    for models_dir in crate::models::model_dirs() {
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        if size > 500_000_000 {
                            let name = split_gguf_base_name(stem).unwrap_or(stem).to_string();
                            sizes.entry(name).and_modify(|e| *e += size).or_insert(size);
                        }
                    }
                }
            }
        }
    }
    sizes
}

fn derive_quantization_type(stem: &str) -> String {
    let parts: Vec<&str> = stem.split('-').collect();
    for &part in parts.iter().rev() {
        let upper = part.to_uppercase();
        if upper.starts_with('Q')
            || upper.starts_with("IQ")
            || upper.starts_with('F')
            || upper.starts_with("BF")
        {
            if upper.len() >= 2
                && upper
                    .chars()
                    .nth(1)
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
                || upper.starts_with("IQ")
                || upper.starts_with("BF")
            {
                return part.to_string();
            }
        }
    }
    String::new()
}

pub fn scan_all_model_metadata() -> Vec<crate::proto::node::CompactModelMetadata> {
    let mut result = Vec::new();
    for models_dir in crate::models::model_dirs() {
        let Ok(entries) = std::fs::read_dir(&models_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if size < 500_000_000 {
                continue;
            }
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let model_key = split_gguf_base_name(&stem).unwrap_or(&stem).to_string();
            let quantization_type = derive_quantization_type(&model_key);
            let meta = if let Some(m) = crate::moe::scan_gguf_compact_meta(&path) {
                crate::proto::node::CompactModelMetadata {
                    model_key: model_key.clone(),
                    context_length: m.context_length,
                    vocab_size: m.vocab_size,
                    embedding_size: m.embedding_size,
                    head_count: m.head_count,
                    layer_count: m.layer_count,
                    feed_forward_length: m.feed_forward_length,
                    key_length: m.key_length,
                    value_length: m.value_length,
                    architecture: m.architecture,
                    tokenizer_model_name: m.tokenizer_model_name,
                    special_tokens: vec![],
                    rope_scale: m.rope_scale,
                    rope_freq_base: m.rope_freq_base,
                    is_moe: m.expert_count > 1,
                    expert_count: m.expert_count,
                    used_expert_count: m.expert_used_count,
                    quantization_type,
                }
            } else {
                crate::proto::node::CompactModelMetadata {
                    model_key,
                    quantization_type,
                    ..Default::default()
                }
            };
            if !result
                .iter()
                .any(|e: &crate::proto::node::CompactModelMetadata| e.model_key == meta.model_key)
            {
                result.push(meta);
            }
        }
    }
    result
}

/// Extract the base model name from a split GGUF stem.
/// "GLM-5-UD-IQ2_XXS-00001-of-00006" → Some("GLM-5-UD-IQ2_XXS")
/// "Qwen3-8B-Q4_K_M" → None (not a split file)
fn split_gguf_base_name(stem: &str) -> Option<&str> {
    // Pattern: ...-NNNNN-of-NNNNN
    let suffix = stem.rfind("-of-")?;
    let part_num = &stem[suffix + 4..];
    if part_num.len() != 5 || !part_num.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let dash = stem[..suffix].rfind('-')?;
    let seq = &stem[dash + 1..suffix];
    if seq.len() != 5 || !seq.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&stem[..dash])
}

/// Detect available VRAM. On Apple Silicon, uses ~75% of system RAM
/// (the rest is reserved for OS/apps on unified memory).
/// Detect available memory for model loading, capped by max_vram_gb if set.
/// "VRAM" is a misnomer — on macOS unified memory and Linux CPU-only, this
/// is system RAM. On Linux with a GPU, it's actual GPU VRAM.
pub fn detect_vram_bytes_capped(max_vram_gb: Option<f64>) -> u64 {
    let mut detected = crate::hardware::survey().vram_bytes;
    if let Some(cap) = max_vram_gb {
        let cap_bytes = (cap * 1e9) as u64;
        if cap_bytes < detected {
            detected = cap_bytes;
        }
    }
    detected
}
