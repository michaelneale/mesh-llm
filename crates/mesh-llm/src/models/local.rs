use hf_hub::cache::scan_cache_dir;
use hf_hub::types::cache::{CachedFileInfo, CachedRepoInfo, CachedRevisionInfo, HFCacheInfo};
use hf_hub::RepoType;
use model_ref::{format_model_ref, gguf_matches_quant_selector, normalize_gguf_distribution_id};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuggingFaceModelIdentity {
    pub repo_id: String,
    pub revision: String,
    pub file: String,
    pub canonical_ref: String,
    pub local_file_name: String,
}

static MODEL_REF_PATHS: OnceLock<Mutex<HashMap<String, PathBuf>>> = OnceLock::new();

fn model_ref_paths() -> &'static Mutex<HashMap<String, PathBuf>> {
    MODEL_REF_PATHS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn remember_model_ref_path(model_ref: &str, path: &Path) {
    if let Ok(mut paths) = model_ref_paths().lock() {
        match paths.get(model_ref) {
            Some(existing)
                if model_ref_path_preference_key(existing)
                    <= model_ref_path_preference_key(path) => {}
            _ => {
                paths.insert(model_ref.to_string(), path.to_path_buf());
            }
        }
    }
}

fn remembered_model_ref_path(model_ref: &str) -> Option<PathBuf> {
    model_ref_paths()
        .lock()
        .ok()
        .and_then(|paths| paths.get(model_ref).cloned())
        .filter(|path| path.exists())
}

impl HuggingFaceModelIdentity {
    #[cfg(test)]
    pub fn distribution_ref(&self) -> String {
        format!(
            "{}@{}/{}",
            self.repo_id,
            self.revision,
            distribution_ref_file(&self.file)
        )
    }
}

#[cfg(test)]
fn distribution_ref_file(file: &str) -> String {
    let path = Path::new(file);
    let file_name = path.file_name().and_then(|value| value.to_str());
    let Some(file_name) = file_name else {
        return file.to_string();
    };
    let Some(stem) = file_name.strip_suffix(".gguf") else {
        return file.to_string();
    };
    let Some((prefix, suffix)) = stem.rsplit_once("-of-") else {
        return file.to_string();
    };
    let Some((prefix, shard_no)) = prefix.rsplit_once('-') else {
        return file.to_string();
    };
    if shard_no.len() != 5
        || suffix.len() != 5
        || !shard_no.chars().all(|ch| ch.is_ascii_digit())
        || !suffix.chars().all(|ch| ch.is_ascii_digit())
    {
        return file.to_string();
    }
    path.with_file_name(prefix)
        .to_string_lossy()
        .replace('\\', "/")
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

pub fn huggingface_hub_cache() -> PathBuf {
    if let Some(path) = hf_hub_cache_override() {
        path
    } else {
        if let Ok(path) = std::env::var("HF_HOME") {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join("hub");
            }
        }
        if let Ok(path) = std::env::var("XDG_CACHE_HOME") {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed).join("huggingface").join("hub");
            }
        }
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".cache")
            .join("huggingface")
            .join("hub")
    }
}

pub fn huggingface_hub_cache_dir() -> PathBuf {
    huggingface_hub_cache()
}

pub(crate) fn huggingface_repo_folder_name(repo_id: &str, repo_type: RepoType) -> String {
    let type_plural = format!("{}s", repo_type);
    std::iter::once(type_plural.as_str())
        .chain(repo_id.split('/'))
        .collect::<Vec<_>>()
        .join("--")
}

#[cfg(test)]
pub(crate) fn huggingface_snapshot_path(
    repo_id: &str,
    repo_type: RepoType,
    revision: &str,
) -> PathBuf {
    huggingface_hub_cache_dir()
        .join(huggingface_repo_folder_name(repo_id, repo_type))
        .join("snapshots")
        .join(revision)
}

pub(crate) fn scan_hf_cache_info(cache_root: &Path) -> Option<HFCacheInfo> {
    let cache_root = cache_root.to_path_buf();
    let scan = move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        runtime.block_on(scan_cache_dir(&cache_root)).ok()
    };

    if tokio::runtime::Handle::try_current().is_ok() {
        std::thread::spawn(scan).join().ok().flatten()
    } else {
        scan()
    }
}

fn cache_repo_id(repo: &CachedRepoInfo) -> Option<&str> {
    (repo.repo_type == RepoType::Model).then_some(repo.repo_id.as_str())
}

pub fn mesh_llm_cache_dir() -> PathBuf {
    if let Ok(path) = std::env::var("XDG_CACHE_HOME") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("mesh-llm");
        }
    }

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

fn scan_hf_cache_identity_for_path(
    path: &Path,
    cache_root: &Path,
) -> Option<HuggingFaceModelIdentity> {
    let cache_info = scan_hf_cache_info(cache_root)?;
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    for repo in &cache_info.repos {
        let Some(repo_id) = cache_repo_id(repo) else {
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
    let cache_root = huggingface_hub_cache_dir();
    if let Some(identity) = identity_from_cache_snapshot_path(path, &cache_root) {
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
        if let Some(identity) = identity_from_cache_snapshot_path(&resolved, &cache_root) {
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
    scan_hf_cache_identity_for_path(path, &cache_root)
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

pub(crate) fn direct_hf_cache_root_gguf_paths(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !(file_type.is_file() || file_type.is_symlink()) {
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            != Some(true)
        {
            continue;
        }
        out.push(path);
    }
    out.sort();
    out
}

fn cache_scanned_file_path(
    cache_root: &Path,
    repo: &CachedRepoInfo,
    revision: &CachedRevisionInfo,
    file: &CachedFileInfo,
) -> PathBuf {
    let relative = file
        .file_path
        .strip_prefix(&revision.snapshot_path)
        .unwrap_or(file.file_path.as_path());
    cache_root
        .join(huggingface_repo_folder_name(&repo.repo_id, repo.repo_type))
        .join("snapshots")
        .join(&revision.commit_hash)
        .join(relative)
}

fn cached_relative_file(revision: &CachedRevisionInfo, file: &CachedFileInfo) -> String {
    file.file_path
        .strip_prefix(&revision.snapshot_path)
        .unwrap_or(file.file_path.as_path())
        .to_string_lossy()
        .replace('\\', "/")
}

fn layered_package_relative_preference(relative_file: &str) -> u8 {
    if relative_file == "shared/output.gguf" {
        0
    } else if is_layered_package_direct_shared_relative_file(relative_file) {
        1
    } else if layered_package_layer_index(relative_file).is_some() {
        2
    } else if is_layered_package_gguf_relative_file(relative_file) {
        3
    } else {
        4
    }
}

fn model_ref_path_preference_key(path: &Path) -> (u8, String) {
    let rank = huggingface_identity_for_path(path)
        .filter(|identity| identity.repo_id.ends_with("-layers"))
        .map(|identity| layered_package_relative_preference(&identity.file))
        .unwrap_or(0);
    (rank, path.to_string_lossy().to_string())
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
    let name = model_ref_for_path(path);
    if seen.insert(name.clone()) {
        names.push(name);
    }
}

fn scan_hf_cache_models(names: &mut Vec<String>, seen: &mut HashSet<String>, min_size_bytes: u64) {
    let cache_root = huggingface_hub_cache_dir();

    for path in direct_hf_cache_root_gguf_paths(&cache_root) {
        push_model_name(&path, names, seen, min_size_bytes);
    }

    let Some(cache_info) = scan_hf_cache_info(&cache_root) else {
        return;
    };
    for repo in &cache_info.repos {
        if repo.repo_type != RepoType::Model {
            continue;
        }
        for revision in &repo.revisions {
            let mut files = revision.files.iter().collect::<Vec<_>>();
            files.sort_by(|left, right| {
                let left_relative = cached_relative_file(revision, left);
                let right_relative = cached_relative_file(revision, right);
                layered_package_relative_preference(&left_relative)
                    .cmp(&layered_package_relative_preference(&right_relative))
                    .then_with(|| left_relative.cmp(&right_relative))
            });
            for file in files {
                if !file.file_name.ends_with(".gguf") {
                    continue;
                }
                let path = cache_scanned_file_path(&cache_root, repo, revision, file);
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
    names.sort();
    names
}

/// Scan model directories for GGUF files and return canonical model refs.
pub fn scan_local_models() -> Vec<String> {
    scan_models_with_min_size(500_000_000)
}

/// Scan installed GGUF models, including small draft models, and return canonical model refs.
pub fn scan_installed_models() -> Vec<String> {
    scan_models_with_min_size(0)
}

fn hf_identity_model_ref(identity: &HuggingFaceModelIdentity) -> String {
    if let Some(model_ref) = layered_package_model_ref(identity) {
        return model_ref;
    }
    let selector = model_ref::quant_selector_from_gguf_file(&identity.file)
        .or_else(|| normalize_gguf_distribution_id(&identity.file));
    format_model_ref(&identity.repo_id, None, selector.as_deref())
}

fn layered_package_model_ref(identity: &HuggingFaceModelIdentity) -> Option<String> {
    if identity.repo_id.ends_with("-layers")
        && is_layered_package_gguf_relative_file(&identity.file)
    {
        Some(format_model_ref(&identity.repo_id, None, None))
    } else {
        None
    }
}

fn is_layered_package_gguf_relative_file(relative_file: &str) -> bool {
    (relative_file.starts_with("shared/") || relative_file.starts_with("layers/"))
        && relative_file.ends_with(".gguf")
        && Path::new(relative_file).file_name().is_some()
}

fn is_layered_package_direct_shared_relative_file(relative_file: &str) -> bool {
    let Some(file_name) = relative_file.strip_prefix("shared/") else {
        return false;
    };
    !file_name.is_empty() && !file_name.contains('/') && file_name.ends_with(".gguf")
}

fn layered_package_layer_index(relative_file: &str) -> Option<usize> {
    let relative = relative_file.strip_prefix("layers/")?;
    let file_name = Path::new(relative).file_name()?.to_str()?;
    let index = file_name.strip_prefix("layer-")?.strip_suffix(".gguf")?;
    if index.is_empty() || !index.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    index.parse().ok()
}

fn layered_package_snapshot_root(
    path: &Path,
    identity: &HuggingFaceModelIdentity,
) -> Option<PathBuf> {
    let mut root = path.to_path_buf();
    for _ in Path::new(&identity.file).components() {
        if !root.pop() {
            return None;
        }
    }
    Some(root)
}

fn layered_package_gguf_paths(path: &Path) -> Option<(PathBuf, Vec<PathBuf>)> {
    let identity = huggingface_identity_for_path(path)?;
    layered_package_model_ref(&identity)?;
    let root = layered_package_snapshot_root(path, &identity)?;
    let mut paths = Vec::new();
    for subdir in ["shared", "layers"] {
        collect_gguf_paths_recursive(&root.join(subdir), &mut paths);
    }
    paths.sort();
    Some((root, paths))
}

fn collect_gguf_paths_recursive(dir: &Path, paths: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_paths_recursive(&path, paths);
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            paths.push(path);
        }
    }
}

pub fn layered_package_layer_count_for_path(path: &Path) -> Option<usize> {
    let (root, paths) = layered_package_gguf_paths(path)?;
    let layers = paths
        .iter()
        .filter(|path| {
            path.strip_prefix(&root)
                .ok()
                .map(|relative| relative.to_string_lossy().replace('\\', "/"))
                .as_deref()
                .and_then(layered_package_layer_index)
                .is_some()
        })
        .count();
    (layers > 0).then_some(layers)
}

pub fn layered_package_total_bytes_for_path(path: &Path) -> Option<u64> {
    let (_root, paths) = layered_package_gguf_paths(path)?;
    let total = paths
        .iter()
        .map(|path| std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0))
        .sum();
    Some(total)
}

fn synthetic_local_gguf_model_ref(path: &Path) -> String {
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.gguf");
    let metadata = std::fs::metadata(path).ok();
    let len = metadata.as_ref().map(std::fs::Metadata::len).unwrap_or(0);
    let modified = metadata
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(b"\0");
    hasher.update(filename.as_bytes());
    hasher.update(b"\0");
    hasher.update(len.to_le_bytes());
    hasher.update(modified.to_le_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format_model_ref(&format!("local-gguf/sha256-{}", &digest[..16]), None, None)
}

pub fn model_ref_for_path(path: &Path) -> String {
    let model_ref = huggingface_identity_for_path(path)
        .map(|identity| hf_identity_model_ref(&identity))
        .unwrap_or_else(|| synthetic_local_gguf_model_ref(path));
    remember_model_ref_path(&model_ref, path);
    model_ref
}

fn find_hf_cache_model_ref_path(root: &Path, model: &model_ref::ModelRef) -> Option<PathBuf> {
    if model.repo.starts_with("local-gguf/") {
        return find_synthetic_local_gguf_path(root, model);
    }
    let cache_info = scan_hf_cache_info(root)?;
    let mut candidates = Vec::new();
    for repo in &cache_info.repos {
        if repo.repo_type != RepoType::Model || repo.repo_id != model.repo {
            continue;
        }
        for revision in &repo.revisions {
            if let Some(wanted_revision) = model.revision.as_deref() {
                if revision.commit_hash != wanted_revision {
                    continue;
                }
            }
            for file in &revision.files {
                if !file.file_name.ends_with(".gguf") {
                    continue;
                }
                let matches = match model.selector.as_deref() {
                    Some(selector) => {
                        gguf_matches_quant_selector(&file.file_name, selector)
                            || normalize_gguf_distribution_id(&file.file_name).as_deref()
                                == Some(selector)
                    }
                    None => true,
                };
                if matches {
                    candidates.push(cache_scanned_file_path(root, repo, revision, file));
                }
            }
        }
    }
    candidates.sort();
    candidates.into_iter().next()
}

fn find_synthetic_local_gguf_path(root: &Path, model: &model_ref::ModelRef) -> Option<PathBuf> {
    let wanted = model.display_id();
    let mut candidates = direct_hf_cache_root_gguf_paths(root);
    let cache_info = scan_hf_cache_info(root);
    if let Some(cache_info) = cache_info {
        for repo in &cache_info.repos {
            if repo.repo_type != RepoType::Model {
                continue;
            }
            for revision in &repo.revisions {
                for file in &revision.files {
                    if file.file_name.ends_with(".gguf") {
                        candidates.push(cache_scanned_file_path(root, repo, revision, file));
                    }
                }
            }
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .find(|path| model_ref_for_path(path) == wanted)
}

fn find_hf_cache_model_path(root: &Path, stem: &str) -> Option<PathBuf> {
    let filename = format!("{stem}.gguf");
    let direct = root.join(&filename);
    if direct.exists() {
        return Some(direct);
    }

    let split_prefix = format!("{stem}-00001-of-");
    let cache_root = huggingface_hub_cache_dir();
    let cache_info = scan_hf_cache_info(&cache_root)?;
    for repo in &cache_info.repos {
        if repo.repo_type != RepoType::Model {
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

/// Extract the base model name from a split GGUF stem.
/// "GLM-5-UD-IQ2_XXS-00001-of-00006" → Some("GLM-5-UD-IQ2_XXS")
/// "Qwen3-8B-Q4_K_M" → None (not a split file)
pub(crate) fn split_gguf_base_name(stem: &str) -> Option<&str> {
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

/// Find a GGUF model file by canonical model ref in the Hugging Face cache.
/// For split GGUFs, finds the first part (name-00001-of-NNNNN.gguf).
pub fn find_model_path(model_ref: &str) -> PathBuf {
    if let Some(path) = remembered_model_ref_path(model_ref) {
        return path;
    }
    let canonical_dir = huggingface_hub_cache_dir();
    if let Ok(parsed) = model_ref::ModelRef::parse(model_ref) {
        if let Some(found) = find_hf_cache_model_ref_path(&canonical_dir, &parsed) {
            return found;
        }
    }

    if let Some(found) = find_hf_cache_model_path(&canonical_dir, model_ref) {
        return found;
    }

    canonical_dir.join(format!("{model_ref}.gguf"))
}

/// Strip common GGUF quantization suffixes from a lowercased stem.
/// e.g. "qwen3vl-2b-instruct-q4_k_m" → "qwen3vl-2b-instruct"
fn strip_quant_suffix(stem: &str) -> &str {
    // Quant suffixes are typically the last hyphen-separated component:
    // Q4_K_M, Q8_0, BF16, F16, F32, IQ4_NL, etc.
    if let Some(pos) = stem.rfind('-') {
        let suffix = &stem[pos + 1..];
        // Starts with 'q', 'iq', 'f', or 'bf' followed by a digit → quant suffix
        let is_quant = suffix.starts_with("q")
            || suffix.starts_with("iq")
            || suffix.starts_with("f16")
            || suffix.starts_with("f32")
            || suffix.starts_with("bf16");
        if is_quant {
            return &stem[..pos];
        }
    }
    stem
}

/// Extract the quantization suffix from a lowercased model stem, if present.
/// e.g. "qwen3vl-2b-instruct-q4_k_m" → Some("q4_k_m")
///      "my-model"                    → None
fn extract_quant_suffix(stem: &str) -> Option<String> {
    let stripped = strip_quant_suffix(stem);
    if stripped.len() < stem.len() {
        // +1 to skip the '-' separator; use .get() for safe UTF-8 slicing
        stem.get(stripped.len() + 1..).map(|s| s.to_string())
    } else {
        None
    }
}

/// Return the sole candidate from `candidates` whose lowercased filename
/// contains `quant`, or `None` if zero or multiple candidates match.
fn pick_quant_match(candidates: &[PathBuf], quant: &str) -> Option<PathBuf> {
    let mut matches: Vec<_> = candidates
        .iter()
        .filter(|path| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase().contains(quant))
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

fn is_named_mmproj_match(lower: &str, model_base: &str, model_stem: &str) -> bool {
    // Try pattern: <model>-mmproj... (model name before mmproj)
    if let Some((prefix, _)) = lower
        .split_once("-mmproj")
        .or_else(|| lower.split_once("_mmproj"))
    {
        if model_base.starts_with(prefix) || model_stem.starts_with(prefix) {
            return true;
        }
    }
    // Try pattern: mmproj-<model>... (model name after mmproj)
    if let Some(after) = lower
        .strip_prefix("mmproj-")
        .or_else(|| lower.strip_prefix("mmproj_"))
    {
        let mmproj_model_base = strip_quant_suffix(after);
        if model_base.starts_with(mmproj_model_base) || mmproj_model_base.starts_with(model_base) {
            return true;
        }
    }
    false
}

fn mmproj_precision_variant_key(path: &Path) -> Option<(String, u8)> {
    let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
    let split = stem.rfind(['-', '_'])?;
    let base = stem[..split].trim_end_matches(['-', '_']).to_string();
    let precision = &stem[split + 1..];
    let rank = match precision {
        "bf16" => 0,
        "f16" => 1,
        "f32" => 2,
        _ => return None,
    };
    Some((base, rank))
}

fn choose_mmproj_candidate(candidates: &[PathBuf]) -> Option<PathBuf> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates[0].clone());
    }

    let parsed: Vec<_> = candidates
        .iter()
        .map(|path| mmproj_precision_variant_key(path).map(|(base, rank)| (path, base, rank)))
        .collect::<Option<Vec<_>>>()?;
    let base = &parsed.first()?.1;
    if parsed.iter().any(|(_, other_base, _)| other_base != base) {
        return None;
    }

    parsed
        .into_iter()
        .min_by_key(|(_, _, rank)| *rank)
        .map(|(path, _, _)| path.clone())
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

    // Scan the model's parent directory for a matching mmproj file.
    // This is safe for the HF hub cache because each model lives in its own
    // isolated snapshot subdirectory alongside only its companion files.
    //
    // Preferred resolution order within that exact directory:
    // 1. Model-name-aware matches (single → return immediately).
    // 2. Among multiple name-matched candidates: quant-aware selection —
    //    prefer the mmproj whose filename contains the same quantization as
    //    the model (e.g. Q4_K_M), matching LM Studio's heuristic.
    // 3. Precision-variant fallback: if all remaining candidates are the same
    //    projector in different precisions, prefer BF16 over F16 over F32.
    // 4. Return None when the choice is genuinely ambiguous.
    let parent = model_path.parent()?;
    let model_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())?;
    // Strip the quant suffix from the model stem to get the base model name
    // e.g. "qwen3vl-2b-instruct-q4_k_m" → "qwen3vl-2b-instruct"
    let model_base = strip_quant_suffix(&model_stem);
    // Extract the quantization suffix for quant-aware matching below
    // e.g. "qwen3vl-2b-instruct-q4_k_m" → Some("q4_k_m")
    let model_quant = extract_quant_suffix(&model_stem);
    let mmproj_siblings: Vec<PathBuf> = std::fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != model_path)
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("gguf"))
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| {
                    let lower = stem.to_ascii_lowercase();
                    lower.contains("mmproj")
                })
                .unwrap_or(false)
        })
        .collect();

    let named_matches: Vec<PathBuf> = mmproj_siblings
        .iter()
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| {
                    is_named_mmproj_match(&stem.to_ascii_lowercase(), model_base, &model_stem)
                })
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    if !named_matches.is_empty() {
        // Multiple named matches: try quant-aware selection before precision fallback
        if named_matches.len() > 1 {
            if let Some(ref quant) = model_quant {
                if let Some(candidate) = pick_quant_match(&named_matches, quant) {
                    return Some(candidate);
                }
            }
        }
        // Single named match, or quant-match failed: precision-variant pick or None
        return choose_mmproj_candidate(&named_matches);
    }

    // No named matches: try quant-aware selection among all siblings, then precision fallback
    if mmproj_siblings.len() > 1 {
        if let Some(ref quant) = model_quant {
            if let Some(candidate) = pick_quant_match(&mmproj_siblings, quant) {
                return Some(candidate);
            }
        }
    }
    choose_mmproj_candidate(&mmproj_siblings)
}

#[cfg(test)]
pub fn resolve_mmproj_path(
    model_name: &str,
    model_path: &Path,
    explicit_mmproj: Option<&Path>,
) -> Option<PathBuf> {
    explicit_mmproj
        .map(Path::to_path_buf)
        .or_else(|| find_mmproj_path(model_name, model_path))
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

    #[test]
    fn mmproj_path_prefers_bf16_generic_precision_variants() {
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-mmproj-precision-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let model = temp.join("Qwen3.5-0.8B-Q4_K_M.gguf");
        let f32 = temp.join("mmproj-F32.gguf");
        let f16 = temp.join("mmproj-F16.gguf");
        let bf16 = temp.join("mmproj-BF16.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&f32, b"mmproj").unwrap();
        std::fs::write(&f16, b"mmproj").unwrap();
        std::fs::write(&bf16, b"mmproj").unwrap();

        let found = find_mmproj_path("Qwen3.5-0.8B-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(bf16.as_path()));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn resolve_mmproj_path_prefers_explicit_override() {
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-mmproj-override-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let model = temp.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let sibling = temp.join("mmproj-sibling.gguf");
        let explicit = temp.join("mmproj-explicit.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&sibling, b"mmproj").unwrap();
        std::fs::write(&explicit, b"mmproj").unwrap();

        let found = resolve_mmproj_path(
            "Qwen3VL-2B-Instruct-Q4_K_M",
            &model,
            Some(explicit.as_path()),
        );
        assert_eq!(found.as_deref(), Some(explicit.as_path()));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    #[serial]
    fn scan_installed_models_includes_direct_hf_cache_root_files() {
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-direct-cache-root-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        std::fs::write(temp.join("Direct-Root-Q4_K_M.gguf"), b"gguf").unwrap();

        std::env::set_var("HF_HUB_CACHE", &temp);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let installed = scan_installed_models();
        assert!(installed
            .iter()
            .any(|name| name.starts_with("local-gguf/sha256-")));

        let _ = std::fs::remove_dir_all(&temp);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    #[serial]
    fn scan_installed_models_collapses_layered_package_files() {
        if let Ok(mut paths) = model_ref_paths().lock() {
            paths.clear();
        }
        let prev_hub_cache = std::env::var_os("HF_HUB_CACHE");
        let prev_hf_home = std::env::var_os("HF_HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");

        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-layered-cache-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let repo_dir = temp.join("models--meshllm--DeepSeek-V3.2-UD-Q4_K_XL-layers");
        let revision = "abcdef1234567890";
        let snapshot = repo_dir.join("snapshots").join(revision);
        let shared = snapshot.join("shared").join("embeddings.gguf");
        let layer_000 = snapshot.join("layers").join("layer-000.gguf");
        let layer_001 = snapshot.join("layers").join("layer-001.gguf");
        let nested_layer_002 = snapshot
            .join("layers")
            .join("blocks")
            .join("layer-002.gguf");
        let nested_shared = snapshot.join("shared").join("nested").join("extra.gguf");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::create_dir_all(layer_000.parent().unwrap()).unwrap();
        std::fs::create_dir_all(nested_layer_002.parent().unwrap()).unwrap();
        std::fs::create_dir_all(nested_shared.parent().unwrap()).unwrap();
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), revision).unwrap();
        std::fs::write(&shared, b"shared").unwrap();
        std::fs::write(&layer_000, b"layer-000").unwrap();
        std::fs::write(&layer_001, b"layer-001").unwrap();
        std::fs::write(&nested_layer_002, b"layer-002").unwrap();
        std::fs::write(&nested_shared, b"nested").unwrap();

        std::env::set_var("HF_HUB_CACHE", &temp);
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");

        let installed = scan_installed_models();

        assert_eq!(
            installed,
            vec!["meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers".to_string()]
        );
        assert_eq!(
            find_model_path("meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers"),
            shared
        );
        assert_eq!(layered_package_layer_count_for_path(&layer_000), Some(3));
        assert_eq!(
            layered_package_total_bytes_for_path(&layer_000),
            Some(6 + 9 + 9 + 9 + 6)
        );

        let _ = std::fs::remove_dir_all(&temp);
        restore_env("HF_HUB_CACHE", prev_hub_cache);
        restore_env("HF_HOME", prev_hf_home);
        restore_env("XDG_CACHE_HOME", prev_xdg);
    }

    #[test]
    fn hf_identity_model_ref_preserves_layer_selector_outside_layered_packages() {
        let identity = HuggingFaceModelIdentity {
            repo_id: "example/Regular-GGUF".to_string(),
            revision: "deadbeef".to_string(),
            file: "layers/layer-000.gguf".to_string(),
            canonical_ref: "example/Regular-GGUF@deadbeef/layers/layer-000.gguf".to_string(),
            local_file_name: "layer-000.gguf".to_string(),
        };

        assert_eq!(
            hf_identity_model_ref(&identity),
            "example/Regular-GGUF:layer-000"
        );
    }

    #[test]
    fn layered_package_shared_matching_accepts_nested_package_artifacts() {
        let direct = HuggingFaceModelIdentity {
            repo_id: "meshllm/Demo-layers".to_string(),
            revision: "deadbeef".to_string(),
            file: "shared/embeddings.gguf".to_string(),
            canonical_ref: "meshllm/Demo-layers@deadbeef/shared/embeddings.gguf".to_string(),
            local_file_name: "embeddings.gguf".to_string(),
        };
        let nested = HuggingFaceModelIdentity {
            file: "shared/nested/embeddings.gguf".to_string(),
            canonical_ref: "meshllm/Demo-layers@deadbeef/shared/nested/embeddings.gguf".to_string(),
            ..direct.clone()
        };

        assert_eq!(
            layered_package_model_ref(&direct),
            Some("meshllm/Demo-layers".to_string())
        );
        assert_eq!(
            layered_package_model_ref(&nested),
            Some("meshllm/Demo-layers".to_string())
        );
    }

    #[test]
    fn layered_package_layer_matching_is_separator_safe_after_normalization() {
        let relative = "layers\\block-0\\layer-000.gguf".replace('\\', "/");

        assert_eq!(layered_package_layer_index(&relative), Some(0));
    }

    #[test]
    fn distribution_ref_preserves_unsplit_exact_file() {
        let identity = HuggingFaceModelIdentity {
            repo_id: "unsloth/Qwen3.6-35B-A3B-GGUF".to_string(),
            revision: "deadbeef".to_string(),
            file: "BF16/Qwen3.6-35B-A3B-BF16.gguf".to_string(),
            canonical_ref: "unsloth/Qwen3.6-35B-A3B-GGUF@deadbeef/BF16/Qwen3.6-35B-A3B-BF16.gguf"
                .to_string(),
            local_file_name: "Qwen3.6-35B-A3B-BF16.gguf".to_string(),
        };

        assert_eq!(
            identity.distribution_ref(),
            "unsloth/Qwen3.6-35B-A3B-GGUF@deadbeef/BF16/Qwen3.6-35B-A3B-BF16.gguf"
        );
    }

    #[test]
    fn distribution_ref_strips_split_suffix_for_split_gguf() {
        let identity = HuggingFaceModelIdentity {
            repo_id: "unsloth/Qwen3.6-35B-A3B-GGUF".to_string(),
            revision: "deadbeef".to_string(),
            file: "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf".to_string(),
            canonical_ref: "unsloth/Qwen3.6-35B-A3B-GGUF@deadbeef/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf".to_string(),
            local_file_name: "Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf".to_string(),
        };

        assert_eq!(
            identity.distribution_ref(),
            "unsloth/Qwen3.6-35B-A3B-GGUF@deadbeef/BF16/Qwen3.6-35B-A3B-BF16"
        );
    }

    #[test]
    fn distribution_ref_strips_non_first_split_suffix_for_split_gguf() {
        let identity = HuggingFaceModelIdentity {
            repo_id: "unsloth/Qwen3.6-35B-A3B-GGUF".to_string(),
            revision: "deadbeef".to_string(),
            file: "BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf".to_string(),
            canonical_ref: "unsloth/Qwen3.6-35B-A3B-GGUF@deadbeef/BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf".to_string(),
            local_file_name: "Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf".to_string(),
        };

        assert_eq!(
            identity.distribution_ref(),
            "unsloth/Qwen3.6-35B-A3B-GGUF@deadbeef/BF16/Qwen3.6-35B-A3B-BF16"
        );
    }

    fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn mmproj_path_prefers_quant_matched_named_candidate() {
        // When multiple named mmproj candidates exist (model-name prefix matches
        // both), quant-aware selection should pick the one whose filename contains
        // the same quantization as the model (Q4_K_M in this case).
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-mmproj-quant-named-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let model = temp.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let q4_mmproj = temp.join("mmproj-Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let q8_mmproj = temp.join("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&q4_mmproj, b"mmproj").unwrap();
        std::fs::write(&q8_mmproj, b"mmproj").unwrap();

        let found = find_mmproj_path("Qwen3VL-2B-Instruct-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(q4_mmproj.as_path()));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn mmproj_path_prefers_quant_matched_generic_sibling() {
        // When there are no model-name-aware matches but the siblings include
        // a projector with the same quant as the model, select that one.
        let temp = std::env::temp_dir().join(format!(
            "mesh-llm-mmproj-quant-sibling-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let model = temp.join("my-model-Q4_K_M.gguf");
        // Generic projector names without a matching model prefix
        let q4_mmproj = temp.join("mmproj-Q4_K_M.gguf");
        let q8_mmproj = temp.join("mmproj-Q8_0.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&q4_mmproj, b"mmproj").unwrap();
        std::fs::write(&q8_mmproj, b"mmproj").unwrap();

        let found = find_mmproj_path("my-model-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(q4_mmproj.as_path()));

        let _ = std::fs::remove_dir_all(&temp);
    }
}
