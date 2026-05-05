//! Fetches and caches the meshllm/catalog HuggingFace dataset for layer package discovery.
//!
//! The catalog lives at <https://huggingface.co/datasets/meshllm/catalog> with entries like:
//! ```text
//! entries/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF.json
//! ```
#![allow(dead_code)]

use std::{
    ffi::OsString,
    fs,
    path::PathBuf,
    process::Command,
    sync::OnceLock,
    time::{Duration, SystemTime},
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

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

static CATALOG_ENTRIES: OnceLock<Vec<CatalogEntry>> = OnceLock::new();

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
    let Ok(metadata) = fs::metadata(&entries_dir) else {
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
/// Uses the `hf` CLI to download `meshllm/catalog` as a dataset, then parses all
/// JSON files under `entries/**/*.json`.
pub fn refresh_catalog() -> Result<()> {
    let cache_dir = catalog_cache_dir();
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("create catalog cache dir {}", cache_dir.display()))?;

    let args: Vec<OsString> = vec![
        OsString::from("download"),
        OsString::from("meshllm/catalog"),
        OsString::from("--repo-type"),
        OsString::from("dataset"),
        OsString::from("--local-dir"),
        cache_dir.as_os_str().to_os_string(),
        OsString::from("--quiet"),
    ];

    let status = Command::new("hf")
        .args(&args)
        .status()
        .context("run hf download for meshllm/catalog dataset")?;
    if !status.success() {
        bail!("hf download failed for meshllm/catalog dataset");
    }

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
    // Set the static; if already set (race), ignore.
    let _ = CATALOG_ENTRIES.set(entries);
    Ok(())
}

/// Ensures the catalog is loaded — refreshes if stale, otherwise loads from disk.
pub fn ensure_catalog() -> Result<()> {
    if CATALOG_ENTRIES.get().is_some() {
        return Ok(());
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
pub fn find_layer_package(model_query: &str) -> Option<String> {
    let entries = CATALOG_ENTRIES.get()?;
    let query_lower = model_query.to_lowercase();

    for entry in entries {
        for (variant_name, variant) in &entry.variants {
            let matches = variant_name.to_lowercase().contains(&query_lower)
                || variant.curated.name.to_lowercase().contains(&query_lower)
                || entry.source_repo.to_lowercase().contains(&query_lower);

            if matches {
                for package in &variant.packages {
                    if package.package_type == "layer-package" {
                        return Some(format!("hf://{}", package.repo));
                    }
                }
            }
        }
    }

    None
}

/// Returns all loaded catalog entries (if any).
pub fn catalog_entries() -> Option<&'static Vec<CatalogEntry>> {
    CATALOG_ENTRIES.get()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_entries_recursive(dir: &std::path::Path) -> Result<Vec<CatalogEntry>> {
    let mut entries = Vec::new();
    visit_json_files(dir, &mut entries)?;
    Ok(entries)
}

fn visit_json_files(dir: &std::path::Path, entries: &mut Vec<CatalogEntry>) -> Result<()> {
    let read_dir = fs::read_dir(dir)
        .with_context(|| format!("read catalog entries directory {}", dir.display()))?;

    for dir_entry in read_dir {
        let dir_entry = dir_entry?;
        let path = dir_entry.path();
        if path.is_dir() {
            visit_json_files(&path, entries)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            match parse_catalog_entry(&path) {
                Ok(entry) => entries.push(entry),
                Err(err) => {
                    // Log but don't fail on individual malformed entries
                    eprintln!(
                        "warning: skipping malformed catalog entry {}: {err}",
                        path.display()
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

    #[test]
    fn stale_check_returns_true_for_nonexistent() {
        std::env::set_var("HF_HOME", "/tmp/meshllm-test-nonexistent-dir-xyz");
        assert!(is_catalog_stale());
        std::env::remove_var("HF_HOME");
    }
}
