use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::catalog::CatalogEntry;
use crate::resolve::CatalogProvider;

/// Catalog provider backed by a local checkout/cache of the `meshllm/catalog`
/// Hugging Face dataset.
///
/// Network refresh remains the caller's responsibility. This provider owns the
/// portable part of the catalog contract: reading `entries/**/*.json` files in a
/// deterministic order and exposing them to `ModelResolver`.
#[derive(Debug)]
pub struct HfCatalogProvider {
    root_dir: PathBuf,
    entries: Vec<CatalogEntry>,
}

impl HfCatalogProvider {
    /// Loads catalog entries from a cache root containing an `entries/` tree.
    pub fn load(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.into();
        let entries_dir = cache_dir.join("entries");
        let entries = parse_entries_recursive(&entries_dir)?;
        Ok(Self {
            root_dir: cache_dir,
            entries,
        })
    }

    /// Loads catalog entries directly from an `entries/` directory.
    pub fn from_entries_dir(entries_dir: impl Into<PathBuf>) -> Result<Self> {
        let entries_dir = entries_dir.into();
        let entries = parse_entries_recursive(&entries_dir)?;
        let root_dir = entries_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| entries_dir.clone());
        Ok(Self { root_dir, entries })
    }

    /// Creates a provider from already-loaded entries. Useful for tests and for
    /// callers that own their own refresh/cache lifecycle.
    pub fn from_entries(entries: Vec<CatalogEntry>) -> Self {
        Self {
            root_dir: PathBuf::new(),
            entries,
        }
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }
}

impl CatalogProvider for HfCatalogProvider {
    fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }
}

fn parse_entries_recursive(entries_dir: &Path) -> Result<Vec<CatalogEntry>> {
    if !entries_dir.is_dir() {
        bail!(
            "catalog entries directory does not exist: {}",
            entries_dir.display()
        );
    }

    let mut entries = Vec::new();
    visit_json_files(entries_dir, &mut entries)?;
    Ok(entries)
}

fn visit_json_files(dir: &Path, entries: &mut Vec<CatalogEntry>) -> Result<()> {
    let read_dir = fs::read_dir(dir)
        .with_context(|| format!("read catalog entries under {}", dir.display()))?;
    let mut dir_entries = read_dir
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("read catalog directory entries under {}", dir.display()))?;
    dir_entries.sort_by_key(|dir_entry| dir_entry.path());

    for dir_entry in dir_entries {
        let path = dir_entry.path();
        if path.is_dir() {
            visit_json_files(&path, entries)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            entries.push(parse_catalog_entry(&path)?);
        }
    }

    Ok(())
}

fn parse_catalog_entry(path: &Path) -> Result<CatalogEntry> {
    let contents =
        fs::read(path).with_context(|| format!("read catalog entry {}", path.display()))?;
    serde_json::from_slice(&contents)
        .with_context(|| format!("parse catalog entry {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CatalogSidecarAsset, CatalogSidecarRef, CatalogSource, CatalogVariant, CuratedMeta,
    };
    use std::collections::HashMap;

    fn entry_json(source_repo: &str, variant_name: &str) -> String {
        serde_json::to_string(&CatalogEntry {
            schema_version: 1,
            source_repo: source_repo.to_string(),
            variants: {
                let mut variants = HashMap::new();
                variants.insert(
                    variant_name.to_string(),
                    CatalogVariant {
                        source: CatalogSource {
                            repo: source_repo.to_string(),
                            revision: Some("main".to_string()),
                            file: Some(format!("{variant_name}.gguf")),
                        },
                        curated: CuratedMeta {
                            name: variant_name.to_string(),
                            size: None,
                            description: None,
                            draft: Some(format!("{variant_name}-draft")),
                            moe: None,
                            extra_files: Vec::new(),
                            mmproj: Some(CatalogSidecarRef::Asset(CatalogSidecarAsset {
                                file: "mmproj-BF16.gguf".to_string(),
                                repo: source_repo.to_string(),
                                revision: Some("main".to_string()),
                                source_file: None,
                            })),
                        },
                        packages: Vec::new(),
                    },
                );
                variants
            },
        })
        .unwrap()
    }

    #[test]
    fn loads_entries_from_cache_root() {
        let temp = tempfile::tempdir().unwrap();
        let entries_dir = temp.path().join("entries").join("org");
        fs::create_dir_all(&entries_dir).unwrap();
        fs::write(
            entries_dir.join("repo.json"),
            entry_json("org/repo", "repo-Q4"),
        )
        .unwrap();

        let provider = HfCatalogProvider::load(temp.path()).unwrap();

        assert_eq!(provider.root_dir(), temp.path());
        assert_eq!(provider.entries().len(), 1);
        assert_eq!(provider.entries()[0].source_repo, "org/repo");
    }

    #[test]
    fn loads_entries_in_deterministic_recursive_order() {
        let temp = tempfile::tempdir().unwrap();
        let z_dir = temp.path().join("entries").join("z");
        let a_dir = temp.path().join("entries").join("a");
        fs::create_dir_all(&z_dir).unwrap();
        fs::create_dir_all(&a_dir).unwrap();
        fs::write(z_dir.join("repo.json"), entry_json("z/repo", "z-Q4")).unwrap();
        fs::write(a_dir.join("repo.json"), entry_json("a/repo", "a-Q4")).unwrap();

        let provider = HfCatalogProvider::load(temp.path()).unwrap();
        let repos: Vec<_> = provider
            .entries()
            .iter()
            .map(|entry| entry.source_repo.as_str())
            .collect();

        assert_eq!(repos, vec!["a/repo", "z/repo"]);
    }

    #[test]
    fn errors_when_entries_dir_is_missing() {
        let temp = tempfile::tempdir().unwrap();
        let error = HfCatalogProvider::load(temp.path()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("catalog entries directory does not exist"),
            "unexpected error: {error}"
        );
    }
}
