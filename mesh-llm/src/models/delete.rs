use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hf_hub::types::cache::HFCacheInfo;

use crate::models::local::{
    gguf_metadata_cache_path, huggingface_hub_cache_dir, huggingface_identity_for_path,
    mesh_llm_cache_dir, scan_hf_cache_info, split_gguf_base_name,
};
use crate::models::resolve::{
    parse_delete_model_ref, resolve_huggingface_file_from_sibling_entries, DeleteModelRef,
};
use crate::models::usage;

#[derive(Debug)]
pub struct DeleteResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
    pub removed_metadata_files: usize,
    pub removed_usage_records: usize,
}

pub async fn resolve_model_identifier(identifier: &str) -> Result<Vec<PathBuf>> {
    match parse_delete_model_ref(identifier).await? {
        DeleteModelRef::LocalStem(stem) => {
            let path = crate::models::find_model_path(&stem);
            if !path.exists() {
                bail!("Model not found: {}", identifier);
            }
            let mut resolved = BTreeSet::from([normalize_path(&path)]);
            if let Some(cache_info) = scan_hf_cache_info(&huggingface_hub_cache_dir()) {
                resolved.extend(find_related_hf_cache_paths(&cache_info, &path));
            }
            Ok(resolved.into_iter().collect())
        }
        DeleteModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => resolve_cached_hf_ref(&repo, revision.as_deref(), &file)
            .await
            .with_context(|| format!("Resolve installed model ref {identifier}")),
    }
}

fn normalized_gguf_stem(stem: &str) -> &str {
    let stem = stem.strip_suffix(".gguf").unwrap_or(stem);
    split_gguf_base_name(stem).unwrap_or(stem)
}

async fn resolve_cached_hf_ref(
    repo_id: &str,
    revision: Option<&str>,
    file: &str,
) -> Result<Vec<PathBuf>> {
    let cache_root = huggingface_hub_cache_dir();
    let Some(cache_info) = scan_hf_cache_info(&cache_root) else {
        bail!("Model not found: {repo_id}");
    };

    for repo in &cache_info.repos {
        use hf_hub::RepoType;
        if repo.repo_type != RepoType::Model || repo.repo_id != repo_id {
            continue;
        }
        for cached_revision in &repo.revisions {
            if revision.is_some_and(|requested| {
                requested != cached_revision.commit_hash
                    && !cached_revision.refs.iter().any(|r| r == requested)
            }) {
                continue;
            }
            let sibling_entries: Vec<(String, Option<u64>)> = cached_revision
                .files
                .iter()
                .map(|entry| {
                    let size = std::fs::metadata(&entry.file_path)
                        .ok()
                        .map(|meta| meta.len());
                    (entry.file_name.clone(), size)
                })
                .collect();
            let resolved_file = resolve_huggingface_file_from_sibling_entries(
                repo_id,
                revision.or_else(|| cached_revision.refs.first().map(String::as_str)),
                file,
                &sibling_entries,
            )
            .await?;
            if !resolved_file.ends_with(".gguf") {
                bail!("Delete only supports GGUF models: {repo_id}");
            }
            let expected = normalized_gguf_stem(&resolved_file);
            let mut matches: Vec<PathBuf> = cached_revision
                .files
                .iter()
                .filter(|entry| entry.file_name.ends_with(".gguf"))
                .filter(|entry| {
                    normalized_gguf_stem(&entry.file_name).eq_ignore_ascii_case(expected)
                })
                .map(|entry| entry.file_path.clone())
                .collect();
            if !matches.is_empty() {
                matches.sort();
                return Ok(matches);
            }
        }
    }

    bail!("Model not found: {repo_id}")
}

fn find_related_hf_cache_paths(cache_info: &HFCacheInfo, path: &Path) -> Vec<PathBuf> {
    let mut results = BTreeSet::new();
    let Some(identity) = huggingface_identity_for_path(path) else {
        return Vec::new();
    };
    let Some(file_name) = Path::new(&identity.file)
        .file_name()
        .and_then(|value| value.to_str())
    else {
        return Vec::new();
    };
    let expected = normalized_gguf_stem(file_name);

    for repo in &cache_info.repos {
        use hf_hub::RepoType;
        if repo.repo_type != RepoType::Model || repo.repo_id != identity.repo_id {
            continue;
        }
        for revision in &repo.revisions {
            if revision.commit_hash != identity.revision {
                continue;
            }
            for file in &revision.files {
                if !file.file_name.ends_with(".gguf") {
                    continue;
                }
                if normalized_gguf_stem(&file.file_name).eq_ignore_ascii_case(expected) {
                    results.insert(file.file_path.clone());
                }
            }
        }
    }

    results.into_iter().collect()
}

pub fn collect_delete_paths(resolved_paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut to_delete: BTreeSet<PathBuf> = BTreeSet::new();
    if resolved_paths.is_empty() {
        return Ok(Vec::new());
    }

    for path in resolved_paths {
        ensure_delete_path_allowed(path)?;
        to_delete.insert(normalize_path(path));
    }

    let primary_path = &resolved_paths[0];
    if let Some(record) = usage::load_model_usage_record_for_path(primary_path) {
        if record.mesh_managed && !record.managed_paths.is_empty() {
            for p in &record.managed_paths {
                to_delete.insert(normalize_path(p));
            }
        }
    }

    Ok(to_delete.into_iter().collect())
}

pub async fn delete_model_by_identifier(identifier: &str) -> Result<DeleteResult> {
    let resolved_paths = resolve_model_identifier(identifier).await?;

    if resolved_paths.is_empty() {
        bail!("Model not found: {}", identifier);
    }

    let all_paths = collect_delete_paths(&resolved_paths)?;

    if all_paths.is_empty() {
        bail!(
            "No GGUF files found at resolved path: {}",
            resolved_paths[0].display()
        );
    }

    let mut reclaimed_bytes: u64 = 0;
    let mut removed_metadata_files: usize = 0;
    let mut removed_usage_records: usize = 0;
    let mut deleted_paths: Vec<PathBuf> = Vec::new();
    let mut removed_record_paths = BTreeSet::new();

    for path in &all_paths {
        if path.exists() {
            if let Ok(meta) = std::fs::metadata(path) {
                reclaimed_bytes += meta.len();
            }
            std::fs::remove_file(path).with_context(|| format!("Remove {}", path.display()))?;
            deleted_paths.push(path.clone());

            if let Some(metadata_path) = gguf_metadata_cache_path(path) {
                if metadata_path.exists() {
                    std::fs::remove_file(&metadata_path).with_context(|| {
                        format!("Remove metadata cache {}", metadata_path.display())
                    })?;
                    removed_metadata_files += 1;
                }
            }

            prune_empty_ancestors(path, &huggingface_hub_cache_dir());
        }
    }

    for path in &all_paths {
        if let Some(record) = load_model_usage_record_for_path(path) {
            let usage_dir = usage::model_usage_cache_dir();
            let record_path = usage::usage_record_path(&usage_dir, &record.lookup_key);
            if removed_record_paths.insert(record_path.clone()) && record_path.exists() {
                std::fs::remove_file(&record_path)
                    .with_context(|| format!("Remove usage record {}", record_path.display()))?;
                removed_usage_records += 1;
            }
        }
    }

    Ok(DeleteResult {
        deleted_paths,
        reclaimed_bytes,
        removed_metadata_files,
        removed_usage_records,
    })
}

/// Load a model usage record for a given path.
fn load_model_usage_record_for_path(path: &std::path::Path) -> Option<usage::ModelUsageRecord> {
    usage::load_model_usage_record_for_path(path)
}

fn ensure_delete_path_allowed(path: &Path) -> Result<()> {
    let normalized = normalize_path(path);
    let hf_root = normalize_path(&huggingface_hub_cache_dir());
    let mesh_root = normalize_path(&mesh_llm_cache_dir());
    if normalized.starts_with(&hf_root) || normalized.starts_with(&mesh_root) {
        Ok(())
    } else {
        bail!(
            "Deletion target outside known model roots: {}",
            normalized.display()
        );
    }
}

/// Prune empty ancestor directories up to (but not including) stop_at.
fn prune_empty_ancestors(path: &std::path::Path, stop_at: &std::path::Path) {
    let stop_at = normalize_path(stop_at);
    let mut current = path.parent().map(normalize_path);
    while let Some(dir) = current {
        if dir == stop_at {
            break;
        }
        let Ok(mut entries) = std::fs::read_dir(&dir) else {
            break;
        };
        if entries.next().is_some() {
            break;
        }
        if std::fs::remove_dir(&dir).is_err() {
            break;
        }
        current = dir.parent().map(normalize_path);
    }
}

fn normalize_path(path: &std::path::Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_gguf_stem_collapses_split_shards() {
        assert_eq!(
            normalized_gguf_stem("GLM-5-UD-IQ2_XXS-00001-of-00006.gguf"),
            "GLM-5-UD-IQ2_XXS"
        );
        assert_eq!(normalized_gguf_stem("Qwen3-8B-Q4_K_M"), "Qwen3-8B-Q4_K_M");
    }
}
