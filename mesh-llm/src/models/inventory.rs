use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::local::{
    direct_hf_cache_root_gguf_paths, gguf_metadata_cache_path, huggingface_hub_cache,
    huggingface_hub_cache_dir,
};

#[derive(Clone, Debug, Default)]
pub struct LocalModelInventorySnapshot {
    pub model_names: HashSet<String>,
    pub size_by_name: HashMap<String, u64>,
    pub metadata_by_name: HashMap<String, crate::proto::node::CompactModelMetadata>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ModelMetadataCacheProgress {
    pub missing_cache_files_total: usize,
    pub missing_cache_files_done: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedCompactModelMetadata {
    model_key: String,
    context_length: u32,
    vocab_size: u32,
    embedding_size: u32,
    head_count: u32,
    layer_count: u32,
    feed_forward_length: u32,
    key_length: u32,
    value_length: u32,
    architecture: String,
    tokenizer_model_name: String,
    rope_scale: f32,
    rope_freq_base: f32,
    is_moe: bool,
    expert_count: u32,
    used_expert_count: u32,
    quantization_type: String,
}

#[derive(Clone, Debug)]
struct InventoryScanEntry {
    path: PathBuf,
    size: u64,
    model_key: String,
    quantization_type: String,
    scans_metadata: bool,
    missing_cache_file: bool,
}

impl CachedCompactModelMetadata {
    fn into_proto(self) -> crate::proto::node::CompactModelMetadata {
        crate::proto::node::CompactModelMetadata {
            model_key: self.model_key,
            context_length: self.context_length,
            vocab_size: self.vocab_size,
            embedding_size: self.embedding_size,
            head_count: self.head_count,
            layer_count: self.layer_count,
            feed_forward_length: self.feed_forward_length,
            key_length: self.key_length,
            value_length: self.value_length,
            architecture: self.architecture,
            tokenizer_model_name: self.tokenizer_model_name,
            special_tokens: vec![],
            rope_scale: self.rope_scale,
            rope_freq_base: self.rope_freq_base,
            is_moe: self.is_moe,
            expert_count: self.expert_count,
            used_expert_count: self.used_expert_count,
            quantization_type: self.quantization_type,
        }
    }

    fn from_proto(meta: &crate::proto::node::CompactModelMetadata) -> Self {
        Self {
            model_key: meta.model_key.clone(),
            context_length: meta.context_length,
            vocab_size: meta.vocab_size,
            embedding_size: meta.embedding_size,
            head_count: meta.head_count,
            layer_count: meta.layer_count,
            feed_forward_length: meta.feed_forward_length,
            key_length: meta.key_length,
            value_length: meta.value_length,
            architecture: meta.architecture.clone(),
            tokenizer_model_name: meta.tokenizer_model_name.clone(),
            rope_scale: meta.rope_scale,
            rope_freq_base: meta.rope_freq_base,
            is_moe: meta.is_moe,
            expert_count: meta.expert_count,
            used_expert_count: meta.used_expert_count,
            quantization_type: meta.quantization_type.clone(),
        }
    }
}

fn local_gguf_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    // Use CacheInfo to enumerate GGUF files in the HF cache instead of
    // recursively walking the entire cache root (which includes blobs, refs,
    // lock files, and other non-model subdirectories that are expensive to scan).
    let hf_cache_dir = huggingface_hub_cache_dir();
    if hf_cache_dir.exists() {
        for path in direct_hf_cache_root_gguf_paths(&hf_cache_dir) {
            let normalized = path.canonicalize().unwrap_or_else(|_| path.clone());
            if seen.insert(normalized) {
                out.push(path);
            }
        }

        let cache = huggingface_hub_cache();
        if let Ok(cache_info) = hf_hub::cache::CacheInfo::scan_dir(Some(cache.path())) {
            for repo in &cache_info.repos {
                if !repo.cache_id().starts_with("model/") {
                    continue;
                }
                for revision in &repo.revisions {
                    for file in &revision.files {
                        if !file.file_name.ends_with(".gguf") {
                            continue;
                        }
                        let path = file.file_path.clone();
                        let normalized = path.canonicalize().unwrap_or_else(|_| path.clone());
                        if seen.insert(normalized) {
                            out.push(path);
                        }
                    }
                }
            }
        }
    }

    out.sort();
    out
}

pub(crate) fn derive_quantization_type(stem: &str) -> String {
    let parts: Vec<&str> = stem.split('-').collect();
    for &part in parts.iter().rev() {
        let upper = part.to_uppercase();
        if (upper.starts_with('Q')
            || upper.starts_with("IQ")
            || upper.starts_with('F')
            || upper.starts_with("BF"))
            && ((upper.len() >= 2
                && upper
                    .chars()
                    .nth(1)
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false))
                || upper.starts_with("IQ")
                || upper.starts_with("BF"))
        {
            return part.to_string();
        }
    }
    String::new()
}

fn split_gguf_base_name(stem: &str) -> Option<&str> {
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

fn compact_metadata_from_gguf(
    path: &Path,
    model_key: String,
    quantization_type: String,
) -> crate::proto::node::CompactModelMetadata {
    if let Some(m) = crate::models::gguf::scan_gguf_compact_meta(path) {
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
    }
}

fn cached_compact_metadata_for_path(
    path: &Path,
    model_key: String,
    quantization_type: String,
) -> crate::proto::node::CompactModelMetadata {
    let computed =
        || compact_metadata_from_gguf(path, model_key.clone(), quantization_type.clone());
    let Some(cache_path) = gguf_metadata_cache_path(path) else {
        return computed();
    };
    if let Ok(bytes) = std::fs::read(&cache_path) {
        if let Ok(cached) = serde_json::from_slice::<CachedCompactModelMetadata>(&bytes) {
            return cached.into_proto();
        }
    }
    let meta = computed();
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(bytes) = serde_json::to_vec(&CachedCompactModelMetadata::from_proto(&meta)) {
        let _ = std::fs::write(cache_path, bytes);
    }
    meta
}

fn metadata_cache_missing_for_path(path: &Path) -> bool {
    let Some(cache_path) = gguf_metadata_cache_path(path) else {
        return true;
    };
    let Ok(bytes) = std::fs::read(cache_path) else {
        return true;
    };
    serde_json::from_slice::<CachedCompactModelMetadata>(&bytes).is_err()
}

fn inventory_scan_entries() -> Vec<InventoryScanEntry> {
    let mut entries = Vec::new();
    let mut metadata_seen = HashSet::new();
    for path in local_gguf_paths() {
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if size < 500_000_000 {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let model_key = split_gguf_base_name(stem).unwrap_or(stem).to_string();
        let quantization_type = derive_quantization_type(&model_key);
        let scans_metadata = metadata_seen.insert(model_key.clone());
        let missing_cache_file = scans_metadata && metadata_cache_missing_for_path(&path);
        entries.push(InventoryScanEntry {
            path,
            size,
            model_key,
            quantization_type,
            scans_metadata,
            missing_cache_file,
        });
    }
    entries
}

pub fn scan_local_inventory_snapshot_with_progress<F>(
    mut on_progress: F,
) -> LocalModelInventorySnapshot
where
    F: FnMut(ModelMetadataCacheProgress),
{
    let entries = inventory_scan_entries();
    let missing_cache_files_total = entries
        .iter()
        .filter(|entry| entry.missing_cache_file)
        .count();
    let mut missing_cache_files_done = 0usize;
    if missing_cache_files_total > 0 {
        on_progress(ModelMetadataCacheProgress {
            missing_cache_files_total,
            missing_cache_files_done,
        });
    }

    let mut snapshot = LocalModelInventorySnapshot::default();
    for entry in entries {
        snapshot.model_names.insert(entry.model_key.clone());
        snapshot
            .size_by_name
            .entry(entry.model_key.clone())
            .and_modify(|total| *total += entry.size)
            .or_insert(entry.size);
        if !entry.scans_metadata {
            continue;
        }
        let meta = cached_compact_metadata_for_path(
            &entry.path,
            entry.model_key.clone(),
            entry.quantization_type,
        );
        if entry.missing_cache_file {
            missing_cache_files_done += 1;
            on_progress(ModelMetadataCacheProgress {
                missing_cache_files_total,
                missing_cache_files_done,
            });
        }
        snapshot.metadata_by_name.insert(entry.model_key, meta);
    }
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    #[serial]
    fn local_gguf_paths_includes_direct_hf_cache_root_files() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-inventory-direct-cache-root-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let model = temp.join("Inventory-Root-Q4_K_M.gguf");
        std::fs::write(&model, b"gguf").unwrap();

        std::env::set_var("HF_HUB_CACHE", &temp);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let paths = local_gguf_paths();
        assert!(paths.iter().any(|path| path == &model));

        let _ = std::fs::remove_dir_all(&temp);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }
}
