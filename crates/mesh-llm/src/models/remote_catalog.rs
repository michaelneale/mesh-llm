//! Fetches and caches the meshllm/catalog HuggingFace dataset for layer package discovery.
//!
//! The catalog lives at <https://huggingface.co/datasets/meshllm/catalog> with entries like:
//! ```text
//! entries/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF.json
//! ```
#![allow(dead_code)]

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::RwLock,
    time::{Duration, SystemTime},
};

use hf_hub::{RepoDownloadFileParams, RepoInfo, RepoInfoParams};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tracing::warn;

// ---------------------------------------------------------------------------
// Schema types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct CatalogEntry {
    pub schema_version: u32,
    pub source_repo: String,
    pub variants: std::collections::HashMap<String, CatalogVariant>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CatalogVariant {
    pub source: CatalogSource,
    pub curated: CatalogCurated,
    #[serde(default)]
    pub packages: Vec<CatalogPackage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CatalogSource {
    pub repo: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub file: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CatalogCurated {
    pub name: String,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub draft: Option<bool>,
    #[serde(default)]
    pub moe: Option<String>,
    #[serde(default)]
    pub extra_files: Vec<serde_json::Value>,
    #[serde(default)]
    pub mmproj: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CatalogPackage {
    #[serde(rename = "type")]
    pub package_type: String,
    pub repo: String,
    #[serde(default)]
    pub layer_count: Option<u32>,
    #[serde(default)]
    pub total_bytes: Option<u64>,
}

// ---------------------------------------------------------------------------
// Static catalog cache
// ---------------------------------------------------------------------------

static CATALOG_ENTRIES: RwLock<Option<Vec<CatalogEntry>>> = RwLock::new(None);

/// Returns the directory where the catalog dataset is cached locally.
pub fn catalog_cache_dir() -> PathBuf {
    std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .map(|path| path.join("meshllm-catalog"))
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|path| path.join(".cache/meshllm/catalog"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("meshllm/catalog"))
}

/// Returns true if the catalog cache is older than 24 hours or doesn't exist.
pub fn is_catalog_stale() -> bool {
    let cache_dir = catalog_cache_dir();
    let entries_dir = cache_dir.join("entries");
    if !entries_dir.is_dir() {
        return true;
    }
    let refresh_marker = entries_dir.join(".last_refresh");
    let Ok(metadata) = fs::metadata(&refresh_marker) else {
        return true;
    };
    let Ok(modified) = metadata.modified() else {
        return true;
    };
    let Ok(elapsed) = SystemTime::now().duration_since(modified) else {
        return true;
    };
    elapsed > Duration::from_secs(24 * 60 * 60)
}

/// Downloads/refreshes the catalog dataset from HuggingFace and loads entries into memory.
///
/// Lists all files in the `meshllm/catalog` dataset via the HF API, then downloads
/// every `entries/**/*.json` file. No hardcoded file list — new models added to the
/// catalog are discovered automatically.
pub fn refresh_catalog() -> Result<()> {
    let api = super::build_hf_api(false)?;
    let dataset = api.dataset("meshllm", "catalog");

    // List all files in the dataset repo
    let info = dataset
        .info(
            &RepoInfoParams::builder()
                .revision("main".to_string())
                .build(),
        )
        .context("fetch meshllm/catalog dataset info")?;

    let siblings = match info {
        RepoInfo::Dataset(ref d) => d.siblings.as_ref(),
        _ => None,
    };
    let Some(siblings) = siblings else {
        bail!("meshllm/catalog dataset info has no file listing");
    };

    let entry_files: Vec<&str> = siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|f| f.starts_with("entries/") && f.ends_with(".json"))
        .collect();

    if entry_files.is_empty() {
        bail!("meshllm/catalog has no entry files");
    }

    let cache_dir = catalog_cache_dir();
    let entries_dir = cache_dir.join("entries");
    fs::create_dir_all(&entries_dir)
        .with_context(|| format!("create catalog cache dir {}", entries_dir.display()))?;

    // Download each entry file
    for entry_file in &entry_files {
        let downloaded = dataset
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(entry_file.to_string())
                    .revision("main".to_string())
                    .build(),
            )
            .with_context(|| format!("download catalog entry {entry_file}"))?;

        // Copy to our cache dir structure if needed
        let dest = cache_dir.join(entry_file);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        if downloaded != dest {
            fs::copy(&downloaded, &dest)
                .with_context(|| format!("copy catalog entry to cache: {entry_file}"))?;
        }
    }

    prune_stale_catalog_entry_files(&cache_dir, &entry_files)?;

    // Touch marker to update mtime for staleness check. Directory mtimes do
    // not reliably change when existing entry files are overwritten.
    let _ = fs::File::create(entries_dir.join(".last_refresh"));

    load_catalog_from_disk()
}

/// Loads catalog entries from the on-disk cache without downloading.
/// Useful if the cache is already fresh.
pub fn load_catalog_from_disk() -> Result<()> {
    let cache_dir = catalog_cache_dir();
    let entries_dir = cache_dir.join("entries");
    if !entries_dir.is_dir() {
        bail!(
            "catalog entries directory does not exist: {}",
            entries_dir.display()
        );
    }

    let entries = parse_entries_recursive(&entries_dir)?;
    let mut lock = CATALOG_ENTRIES
        .write()
        .map_err(|_| anyhow::anyhow!("catalog lock poisoned"))?;
    *lock = Some(entries);
    Ok(())
}

/// Ensures the catalog is loaded — refreshes if stale, otherwise loads from disk.
pub fn ensure_catalog() -> Result<()> {
    {
        let lock = CATALOG_ENTRIES
            .read()
            .map_err(|_| anyhow::anyhow!("catalog lock poisoned"))?;
        if lock.is_some() && !is_catalog_stale() {
            return Ok(());
        }
    }
    if is_catalog_stale() {
        refresh_catalog()
    } else {
        load_catalog_from_disk()
    }
}

/// Searches the cached catalog for a layer-package matching `model_query`.
///
/// The query is matched (case-insensitive contains) against:
/// - variant name (the key in the variants map)
/// - curated name
/// - source_repo
///
/// Returns the first matching layer-package repo as an `hf://` reference.
/// Catalog entries, variants, and package repos are traversed in sorted order
/// so overlapping contains-matches resolve deterministically.
pub fn find_layer_package(model_query: &str) -> Option<String> {
    let lock = CATALOG_ENTRIES.read().ok()?;
    let entries = lock.as_ref()?;
    let query_lower = model_query.to_lowercase();

    for entry in entries {
        let mut variants: Vec<_> = entry.variants.iter().collect();
        variants.sort_by(|(left_name, _), (right_name, _)| left_name.cmp(right_name));
        for (variant_name, variant) in variants {
            let matches = variant_name.to_lowercase().contains(&query_lower)
                || variant.curated.name.to_lowercase().contains(&query_lower)
                || entry.source_repo.to_lowercase().contains(&query_lower);

            if matches {
                let mut packages: Vec<_> = variant.packages.iter().collect();
                packages.sort_by(|left, right| left.repo.cmp(&right.repo));
                for package in packages {
                    if package.package_type == "layer-package" {
                        return Some(format!("hf://{}", package.repo));
                    }
                }
            }
        }
    }

    None
}

/// A resolved model download reference from the remote catalog.
pub struct RemoteModelRef {
    pub name: String,
    pub repo: String,
    pub revision: Option<String>,
    pub file: String,
}

/// Searches the remote catalog for a model matching `query` and returns
/// download coordinates (repo, revision, file) if found.
///
/// This enables models not in the baked-in catalog to be resolved and
/// downloaded from HuggingFace when they exist in the remote catalog.
pub fn resolve_model_download(query: &str) -> Option<RemoteModelRef> {
    if ensure_catalog().is_err() {
        return None;
    }
    let lock = CATALOG_ENTRIES.read().ok()?;
    let entries = lock.as_ref()?;
    let query_lower = query.to_lowercase();

    for entry in entries {
        let mut variants: Vec<_> = entry.variants.iter().collect();
        variants.sort_by(|(left_name, _), (right_name, _)| left_name.cmp(right_name));
        for (variant_name, variant) in variants {
            let matches = variant_name.to_lowercase().contains(&query_lower)
                || variant.curated.name.to_lowercase().contains(&query_lower)
                || entry.source_repo.to_lowercase().contains(&query_lower);

            if matches {
                let repo = variant.source.repo.clone();
                let file = variant.source.file.clone().unwrap_or_else(|| {
                    // Default: use variant name as filename
                    format!("{variant_name}.gguf")
                });
                return Some(RemoteModelRef {
                    name: variant.curated.name.clone(),
                    repo,
                    revision: variant.source.revision.clone(),
                    file,
                });
            }
        }
    }

    None
}

/// Returns all loaded catalog entries (if any).
pub fn catalog_entries() -> Option<Vec<CatalogEntry>> {
    let lock = CATALOG_ENTRIES.read().ok()?;
    lock.clone()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_entries_recursive(dir: &std::path::Path) -> Result<Vec<CatalogEntry>> {
    let mut entries = Vec::new();
    visit_json_files(dir, &mut entries)?;
    Ok(entries)
}

fn prune_stale_catalog_entry_files(cache_dir: &Path, entry_files: &[&str]) -> Result<()> {
    let entries_dir = cache_dir.join("entries");
    if !entries_dir.is_dir() {
        return Ok(());
    }

    let expected_paths: HashSet<PathBuf> = entry_files
        .iter()
        .map(|path| cache_dir.join(path))
        .collect();
    prune_stale_json_files(&entries_dir, &expected_paths)
}

fn prune_stale_json_files(dir: &Path, expected_paths: &HashSet<PathBuf>) -> Result<()> {
    let read_dir =
        fs::read_dir(dir).with_context(|| format!("read catalog cache dir {}", dir.display()))?;
    let mut dir_entries = read_dir
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("read cached catalog entries under {}", dir.display()))?;
    dir_entries.sort_by_key(|dir_entry| dir_entry.path());

    for dir_entry in dir_entries {
        let path = dir_entry.path();
        if path.is_dir() {
            prune_stale_json_files(&path, expected_paths)?;
            continue;
        }
        if path.extension().is_some_and(|ext| ext == "json") && !expected_paths.contains(&path) {
            fs::remove_file(&path)
                .with_context(|| format!("remove stale catalog entry {}", path.display()))?;
        }
    }
    Ok(())
}

fn visit_json_files(dir: &std::path::Path, entries: &mut Vec<CatalogEntry>) -> Result<()> {
    let read_dir = fs::read_dir(dir)
        .with_context(|| format!("read catalog entries directory {}", dir.display()))?;
    let mut dir_entries = read_dir
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("read catalog entries under {}", dir.display()))?;
    dir_entries.sort_by_key(|dir_entry| dir_entry.path());

    for dir_entry in dir_entries {
        let path = dir_entry.path();
        if path.is_dir() {
            visit_json_files(&path, entries)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            match parse_catalog_entry(&path) {
                Ok(entry) => entries.push(entry),
                Err(err) => {
                    warn!(
                        path = %path.display(),
                        error = %err,
                        "skipping malformed catalog entry"
                    );
                }
            }
        }
    }
    Ok(())
}

fn parse_catalog_entry(path: &std::path::Path) -> Result<CatalogEntry> {
    let contents =
        fs::read(path).with_context(|| format!("read catalog entry {}", path.display()))?;
    serde_json::from_slice(&contents)
        .with_context(|| format!("parse catalog entry {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use serial_test::serial;

    #[test]
    fn deserializes_catalog_entry() {
        let json = r#"{
            "schema_version": 1,
            "source_repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF",
            "variants": {
                "Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL": {
                    "source": { "repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF", "revision": "main", "file": "Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL.gguf" },
                    "curated": { "name": "Qwen3 Coder 480B Q4_K_XL", "size": "294GB", "description": "Large MoE coding model", "draft": null, "moe": "480B/35B", "extra_files": [], "mmproj": null },
                    "packages": [
                        { "type": "layer-package", "repo": "meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers", "layer_count": 62, "total_bytes": 315680000000 }
                    ]
                }
            }
        }"#;

        let entry: CatalogEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.schema_version, 1);
        assert_eq!(
            entry.source_repo,
            "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
        );
        assert_eq!(entry.variants.len(), 1);

        let variant = entry
            .variants
            .get("Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL")
            .unwrap();
        assert_eq!(variant.curated.name, "Qwen3 Coder 480B Q4_K_XL");
        assert_eq!(variant.curated.moe.as_deref(), Some("480B/35B"));
        assert_eq!(variant.packages.len(), 1);
        assert_eq!(variant.packages[0].package_type, "layer-package");
        assert_eq!(
            variant.packages[0].repo,
            "meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers"
        );
        assert_eq!(variant.packages[0].layer_count, Some(62));
    }

    #[test]
    fn catalog_cache_dir_uses_hf_home() {
        // Just verify it returns a path (env-dependent)
        let dir = catalog_cache_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    fn test_variant(curated_name: &str, repo: &str, package_repos: &[&str]) -> CatalogVariant {
        CatalogVariant {
            source: CatalogSource {
                repo: repo.to_string(),
                revision: Some("main".to_string()),
                file: Some(format!("{curated_name}.gguf")),
            },
            curated: CatalogCurated {
                name: curated_name.to_string(),
                size: None,
                description: None,
                draft: None,
                moe: None,
                extra_files: Vec::new(),
                mmproj: None,
            },
            packages: package_repos
                .iter()
                .map(|repo| CatalogPackage {
                    package_type: "layer-package".to_string(),
                    repo: (*repo).to_string(),
                    layer_count: None,
                    total_bytes: None,
                })
                .collect(),
        }
    }

    #[test]
    #[serial]
    fn layer_package_lookup_uses_deterministic_variant_and_package_order() {
        let previous = CATALOG_ENTRIES.write().unwrap().take();
        let mut variants = HashMap::new();
        variants.insert(
            "z-variant".to_string(),
            test_variant(
                "Shared Match Z",
                "example/shared-source",
                &["meshllm/z-package"],
            ),
        );
        variants.insert(
            "a-variant".to_string(),
            test_variant(
                "Shared Match A",
                "example/shared-source",
                &["meshllm/b-package", "meshllm/a-package"],
            ),
        );
        *CATALOG_ENTRIES.write().unwrap() = Some(vec![CatalogEntry {
            schema_version: 1,
            source_repo: "example/shared-source".to_string(),
            variants,
        }]);

        assert_eq!(
            find_layer_package("shared"),
            Some("hf://meshllm/a-package".to_string())
        );

        *CATALOG_ENTRIES.write().unwrap() = previous;
    }

    #[test]
    fn parse_entries_recursive_uses_sorted_directory_order() {
        let temp = tempfile::tempdir().unwrap();
        let z_dir = temp.path().join("z");
        let a_dir = temp.path().join("a");
        fs::create_dir_all(&z_dir).unwrap();
        fs::create_dir_all(&a_dir).unwrap();
        fs::write(
            z_dir.join("entry.json"),
            r#"{
                "schema_version": 1,
                "source_repo": "z/source",
                "variants": {}
            }"#,
        )
        .unwrap();
        fs::write(
            a_dir.join("entry.json"),
            r#"{
                "schema_version": 1,
                "source_repo": "a/source",
                "variants": {}
            }"#,
        )
        .unwrap();

        let entries = parse_entries_recursive(temp.path()).unwrap();
        let repos: Vec<_> = entries
            .iter()
            .map(|entry| entry.source_repo.as_str())
            .collect();
        assert_eq!(repos, vec!["a/source", "z/source"]);
    }

    #[test]
    fn prune_stale_catalog_entry_files_removes_deleted_upstream_entries() {
        let temp = tempfile::tempdir().unwrap();
        let entries_dir = temp.path().join("entries");
        fs::create_dir_all(entries_dir.join("current")).unwrap();
        fs::create_dir_all(entries_dir.join("removed")).unwrap();
        let current = entries_dir.join("current/entry.json");
        let stale = entries_dir.join("removed/entry.json");
        fs::write(&current, b"{}").unwrap();
        fs::write(&stale, b"{}").unwrap();

        prune_stale_catalog_entry_files(temp.path(), &["entries/current/entry.json"]).unwrap();

        assert!(current.is_file());
        assert!(!stale.exists());
    }

    #[test]
    #[serial]
    fn stale_check_returns_true_for_nonexistent() {
        let prev = std::env::var_os("HF_HOME");
        std::env::set_var("HF_HOME", "/tmp/meshllm-test-nonexistent-dir-xyz");
        let result = is_catalog_stale();
        match prev {
            Some(val) => std::env::set_var("HF_HOME", val),
            None => std::env::remove_var("HF_HOME"),
        }
        assert!(result);
    }

    #[test]
    #[serial]
    fn stale_check_uses_last_refresh_marker() {
        let prev = std::env::var_os("HF_HOME");
        let temp = std::env::temp_dir().join(format!(
            "meshllm-catalog-stale-marker-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::env::set_var("HF_HOME", &temp);

        let entries_dir = catalog_cache_dir().join("entries");
        fs::create_dir_all(&entries_dir).unwrap();
        assert!(is_catalog_stale());

        fs::File::create(entries_dir.join(".last_refresh")).unwrap();
        assert!(!is_catalog_stale());

        let _ = fs::remove_dir_all(&temp);
        match prev {
            Some(val) => std::env::set_var("HF_HOME", val),
            None => std::env::remove_var("HF_HOME"),
        }
    }
}
