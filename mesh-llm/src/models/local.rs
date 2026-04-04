use hf_hub::cache::CacheInfo;
use hf_hub::Cache;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuggingFaceModelIdentity {
    pub repo_id: String,
    pub revision: String,
    pub file: String,
    pub canonical_ref: String,
    pub local_file_name: String,
}

/// Directories to scan for GGUF models.
pub fn model_dirs() -> Vec<PathBuf> {
    let canonical = huggingface_hub_cache_dir();
    let legacy = legacy_models_dir();
    let mut dirs = vec![canonical];
    if legacy.exists() {
        dirs.push(legacy);
    }
    dirs
}

fn hf_hub_cache_override() -> Option<PathBuf> {
    let path = std::env::var("HF_HUB_CACHE").ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

/// Build the effective Hugging Face cache handle.
///
/// `hf-hub` already resolves `HF_HOME` and the default cache location.
/// We only patch in `HF_HUB_CACHE` here because the crate does not honor it.
pub fn huggingface_hub_cache() -> Cache {
    if let Some(path) = hf_hub_cache_override() {
        Cache::new(path)
    } else {
        Cache::from_env()
    }
}

pub fn huggingface_hub_cache_dir() -> PathBuf {
    huggingface_hub_cache().path().clone()
}

pub fn legacy_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".models")
}

pub fn mesh_llm_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
        })
        .join("mesh-llm")
}

pub fn model_metadata_cache_dir() -> PathBuf {
    mesh_llm_cache_dir().join("model-meta")
}

pub fn legacy_models_present() -> bool {
    let legacy_dir = legacy_models_dir();
    if !legacy_dir.exists() {
        return false;
    }
    tree_contains_gguf(&legacy_dir)
}

fn parse_model_repo_folder_name(folder: &str) -> Option<String> {
    folder
        .strip_prefix("models--")
        .map(|value| value.replace("--", "/"))
}

fn identity_from_cache_snapshot_path(
    path: &Path,
    cache_root: &Path,
) -> Option<HuggingFaceModelIdentity> {
    let relative = path.strip_prefix(cache_root).ok()?;
    let mut components = relative.components();
    let repo_folder = components.next()?.as_os_str().to_str()?;
    let repo_id = parse_model_repo_folder_name(repo_folder)?;
    if components.next()?.as_os_str() != OsStr::new("snapshots") {
        return None;
    }
    let revision = components.next()?.as_os_str().to_str()?.to_string();
    let relative_file = components
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()?
        .join("/");
    if relative_file.is_empty() {
        return None;
    }
    let local_file_name = Path::new(&relative_file)
        .file_name()
        .and_then(|value| value.to_str())?
        .to_string();
    let canonical_ref = format!("{repo_id}@{revision}/{relative_file}");
    Some(HuggingFaceModelIdentity {
        repo_id,
        revision,
        file: relative_file,
        canonical_ref,
        local_file_name,
    })
}

fn scan_hf_cache_identity_for_path(path: &Path, cache: &Cache) -> Option<HuggingFaceModelIdentity> {
    let cache_info = CacheInfo::scan_dir(Some(cache.path())).ok()?;
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    for repo in &cache_info.repos {
        let cache_id = repo.cache_id();
        let Some(repo_id) = cache_id.strip_prefix("model/") else {
            continue;
        };
        for revision in &repo.revisions {
            for file in &revision.files {
                let candidate = file
                    .file_path
                    .canonicalize()
                    .unwrap_or_else(|_| file.file_path.clone());
                if file.file_path != path && candidate != resolved {
                    continue;
                }

                let relative_path = file
                    .file_path
                    .strip_prefix(&revision.snapshot_path)
                    .ok()?
                    .to_string_lossy()
                    .replace('\\', "/");
                if relative_path.is_empty() {
                    return None;
                }

                let canonical_ref = format!(
                    "{repo_id}@{revision}/{relative_path}",
                    revision = revision.commit_hash
                );

                return Some(HuggingFaceModelIdentity {
                    repo_id: repo_id.to_string(),
                    revision: revision.commit_hash.clone(),
                    file: relative_path,
                    canonical_ref,
                    local_file_name: file.file_name.clone(),
                });
            }
        }
    }

    None
}

pub fn huggingface_identity_for_path(path: &Path) -> Option<HuggingFaceModelIdentity> {
    let cache = huggingface_hub_cache();
    let cache_root = cache.path();
    if let Some(identity) = identity_from_cache_snapshot_path(path, cache_root) {
        return Some(identity);
    }
    let resolved_cache_root = cache_root
        .canonicalize()
        .unwrap_or_else(|_| cache_root.clone());
    if resolved_cache_root != *cache_root {
        if let Some(identity) = identity_from_cache_snapshot_path(path, &resolved_cache_root) {
            return Some(identity);
        }
    }
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if resolved != path {
        if let Some(identity) = identity_from_cache_snapshot_path(&resolved, cache_root) {
            return Some(identity);
        }
        if resolved_cache_root != *cache_root {
            if let Some(identity) =
                identity_from_cache_snapshot_path(&resolved, &resolved_cache_root)
            {
                return Some(identity);
            }
        }
    }
    scan_hf_cache_identity_for_path(path, &cache)
}

pub fn gguf_metadata_cache_path(path: &Path) -> Option<PathBuf> {
    let key = if let Some(identity) = huggingface_identity_for_path(path) {
        format!("hf:{}", identity.canonical_ref)
    } else {
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos();
        format!(
            "local:{}:{}:{}",
            path.to_string_lossy(),
            metadata.len(),
            modified
        )
    };
    let digest = Sha256::digest(key.as_bytes());
    Some(model_metadata_cache_dir().join(format!("{digest:x}.json")))
}

pub fn path_is_in_legacy_models_dir(path: &Path) -> bool {
    path.starts_with(legacy_models_dir())
}

fn cache_scanned_file_path(
    cache_root: &Path,
    repo: &hf_hub::cache::CachedRepo,
    revision: &hf_hub::cache::CachedRevision,
    file: &hf_hub::cache::CachedFile,
) -> PathBuf {
    let relative = file
        .file_path
        .strip_prefix(&revision.snapshot_path)
        .unwrap_or(file.file_path.as_path());
    cache_root
        .join(repo.repo.folder_name())
        .join("snapshots")
        .join(&revision.commit_hash)
        .join(relative)
}

fn push_model_name(
    path: &Path,
    names: &mut Vec<String>,
    seen: &mut HashSet<String>,
    min_size_bytes: u64,
) {
    if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        return;
    }
    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return;
    };
    if stem.contains("mmproj") {
        return;
    }
    let size = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    if size <= min_size_bytes {
        return;
    }
    let name = split_gguf_base_name(stem).unwrap_or(stem).to_string();
    if seen.insert(name.clone()) {
        names.push(name);
    }
}

fn tree_contains_gguf(root: &Path) -> bool {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if (file_type.is_file() || file_type.is_symlink())
                && path.extension().and_then(|ext| ext.to_str()) == Some("gguf")
            {
                return true;
            }
        }
    }
    false
}

fn scan_hf_cache_models(names: &mut Vec<String>, seen: &mut HashSet<String>, min_size_bytes: u64) {
    let cache = huggingface_hub_cache();
    let cache_root = cache.path().clone();
    let Ok(cache_info) = CacheInfo::scan_dir(Some(cache.path())) else {
        return;
    };
    for repo in &cache_info.repos {
        if !repo.cache_id().starts_with("model/") {
            continue;
        }
        for revision in &repo.revisions {
            for file in &revision.files {
                if !file.file_name.ends_with(".gguf") {
                    continue;
                }
                let path = cache_scanned_file_path(&cache_root, repo, revision, file);
                push_model_name(&path, names, seen, min_size_bytes);
            }
        }
    }
}

fn scan_model_tree(
    root: &Path,
    names: &mut Vec<String>,
    seen: &mut HashSet<String>,
    min_size_bytes: u64,
) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() || file_type.is_symlink() {
                push_model_name(&path, names, seen, min_size_bytes);
            }
        }
    }
}

fn scan_models_with_min_size(min_size_bytes: u64) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = HashSet::new();
    let canonical_dir = huggingface_hub_cache_dir();
    if canonical_dir.exists() {
        scan_hf_cache_models(&mut names, &mut seen, min_size_bytes);
    }
    let legacy_dir = legacy_models_dir();
    if legacy_dir.exists() {
        scan_model_tree(&legacy_dir, &mut names, &mut seen, min_size_bytes);
    }
    names.sort();
    names
}

/// Scan model directories for GGUF files and return their stem names.
pub fn scan_local_models() -> Vec<String> {
    scan_models_with_min_size(500_000_000)
}

/// Scan installed GGUF models, including small draft models.
pub fn scan_installed_models() -> Vec<String> {
    scan_models_with_min_size(0)
}

fn find_hf_cache_model_path(root: &Path, stem: &str) -> Option<PathBuf> {
    let filename = format!("{stem}.gguf");
    let direct = root.join(&filename);
    if direct.exists() {
        return Some(direct);
    }

    let split_prefix = format!("{stem}-00001-of-");
    let cache = huggingface_hub_cache();
    let cache_root = cache.path().clone();
    let Ok(cache_info) = CacheInfo::scan_dir(Some(cache.path())) else {
        return None;
    };
    for repo in &cache_info.repos {
        if !repo.cache_id().starts_with("model/") {
            continue;
        }
        for revision in &repo.revisions {
            for file in &revision.files {
                let Some(name) = Path::new(&file.file_name)
                    .file_name()
                    .and_then(|value| value.to_str())
                else {
                    continue;
                };
                if name == filename || (name.starts_with(&split_prefix) && name.ends_with(".gguf"))
                {
                    return Some(cache_scanned_file_path(&cache_root, repo, revision, file));
                }
            }
        }
    }
    None
}

fn find_model_tree_path(root: &Path, stem: &str) -> Option<PathBuf> {
    let filename = format!("{stem}.gguf");
    let split_prefix = format!("{stem}-00001-of-");
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name == filename || (name.starts_with(&split_prefix) && name.ends_with(".gguf")) {
                return Some(path);
            }
        }
    }
    None
}

/// Extract the base model name from a split GGUF stem.
/// "GLM-5-UD-IQ2_XXS-00001-of-00006" → Some("GLM-5-UD-IQ2_XXS")
/// "Qwen3-8B-Q4_K_M" → None (not a split file)
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

/// Find a GGUF model file by stem name, searching all model directories.
/// Returns the first match found (prefers the Hugging Face cache, then legacy ~/.models).
/// For split GGUFs, finds the first part (name-00001-of-NNNNN.gguf).
pub fn find_model_path(stem: &str) -> PathBuf {
    let filename = format!("{stem}.gguf");
    let canonical_dir = huggingface_hub_cache_dir();
    if let Some(found) = find_hf_cache_model_path(&canonical_dir, stem) {
        return found;
    }

    let legacy_dir = legacy_models_dir();
    if let Some(found) = find_model_tree_path(&legacy_dir, stem) {
        return found;
    }

    canonical_dir.join(&filename)
}

pub fn find_mmproj_path(model_name: &str, model_path: &Path) -> Option<PathBuf> {
    if let Some(path) = crate::models::catalog::MODEL_CATALOG
        .iter()
        .find(|m| {
            m.name == model_name || m.file.strip_suffix(".gguf").unwrap_or(&m.file) == model_name
        })
        .and_then(|m| m.mmproj.as_ref())
        .map(|asset| crate::models::catalog::models_dir().join(&asset.file))
        .filter(|p| p.exists())
    {
        return Some(path);
    }

    let parent = model_path.parent()?;
    let mut candidates = std::fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != model_path)
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("gguf"))
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| stem.to_ascii_lowercase().contains("mmproj"))
                .unwrap_or(false)
        });

    let candidate = candidates.next()?;
    if candidates.next().is_some() {
        return None;
    }
    Some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn huggingface_cache_prefers_explicit_hub_cache() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::set_var("HF_HUB_CACHE", "/tmp/mesh-llm-hub-cache");
        std::env::set_var("HF_HOME", "/tmp/mesh-llm-hf-home");
        std::env::set_var("XDG_CACHE_HOME", "/tmp/mesh-llm-xdg");

        assert_eq!(
            huggingface_hub_cache_dir(),
            PathBuf::from("/tmp/mesh-llm-hub-cache")
        );

        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    #[serial]
    fn huggingface_cache_falls_back_to_hf_home() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::remove_var("HF_HUB_CACHE");
        std::env::set_var("HF_HOME", "/tmp/mesh-llm-hf-home");
        std::env::set_var("XDG_CACHE_HOME", "/tmp/mesh-llm-xdg");

        assert_eq!(
            huggingface_hub_cache_dir(),
            PathBuf::from("/tmp/mesh-llm-hf-home").join("hub")
        );

        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    fn legacy_tree_detection_finds_nested_gguf_files() {
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-legacy-detect-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested = temp.join("nested").join("models");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("Qwen3-8B-Q4_K_M.gguf"), b"gguf").unwrap();

        assert!(tree_contains_gguf(&temp));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_split_gguf_base_name() {
        assert_eq!(
            split_gguf_base_name("GLM-5-UD-IQ2_XXS-00001-of-00006"),
            Some("GLM-5-UD-IQ2_XXS")
        );
        assert_eq!(
            split_gguf_base_name("GLM-5-UD-IQ2_XXS-00006-of-00006"),
            Some("GLM-5-UD-IQ2_XXS")
        );
        assert_eq!(split_gguf_base_name("Qwen3-8B-Q4_K_M"), None);
        assert_eq!(split_gguf_base_name("model-001-of-003"), None);
        assert_eq!(split_gguf_base_name("model-00001-of-00003"), Some("model"));
    }

    #[test]
    #[serial]
    fn huggingface_identity_for_path_parses_snapshot_path_directly() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-hf-identity-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let snapshot_path = temp
            .join("models--bartowski--Llama-3.2-1B-Instruct-GGUF")
            .join("snapshots")
            .join("abcdef1234567890")
            .join("nested")
            .join("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        std::fs::create_dir_all(snapshot_path.parent().unwrap()).unwrap();
        std::fs::write(&snapshot_path, b"gguf").unwrap();

        std::env::set_var("HF_HUB_CACHE", &temp);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let identity = huggingface_identity_for_path(&snapshot_path).unwrap();
        assert_eq!(identity.repo_id, "bartowski/Llama-3.2-1B-Instruct-GGUF");
        assert_eq!(identity.revision, "abcdef1234567890");
        assert_eq!(identity.file, "nested/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        assert_eq!(
            identity.canonical_ref,
            "bartowski/Llama-3.2-1B-Instruct-GGUF@abcdef1234567890/nested/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        );
        assert_eq!(
            identity.local_file_name,
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        );

        let _ = std::fs::remove_dir_all(&temp);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    fn mmproj_path_falls_back_to_single_sibling_sidecar() {
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-mmproj-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let model = temp.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let mmproj = temp.join("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&mmproj, b"mmproj").unwrap();

        let found = find_mmproj_path("Qwen3VL-2B-Instruct-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(mmproj.as_path()));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn mmproj_path_ignores_ambiguous_sibling_sidecars() {
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-mmproj-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let model = temp.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let mmproj_a = temp.join("mmproj-a.gguf");
        let mmproj_b = temp.join("mmproj-b.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&mmproj_a, b"mmproj").unwrap();
        std::fs::write(&mmproj_b, b"mmproj").unwrap();

        assert!(find_mmproj_path("Qwen3VL-2B-Instruct-Q4_K_M", &model).is_none());

        let _ = std::fs::remove_dir_all(&temp);
    }

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }
}
