use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use model_ref::ModelRef;

use crate::catalog::{CatalogEntry, CatalogSource};
use crate::types::{
    LocalGguf, LocalLayerPackage, ModelArtifactCandidate, RemoteGguf, RemoteLayerPackage,
};

/// Provides access to catalog entries. Implementations can load from HF dataset,
/// local fixture files, or in-memory test data.
pub trait CatalogProvider: Send + Sync {
    fn entries(&self) -> &[CatalogEntry];
}

/// In-memory catalog for testing.
pub struct MemoryCatalog {
    entries: Vec<CatalogEntry>,
}

impl MemoryCatalog {
    pub fn new(entries: Vec<CatalogEntry>) -> Self {
        Self { entries }
    }
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl CatalogProvider for MemoryCatalog {
    fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }
}

/// Model resolver that searches local paths and catalog entries.
pub struct ModelResolver<C: CatalogProvider> {
    catalog: C,
    local_model_dirs: Vec<PathBuf>,
}

impl<C: CatalogProvider> ModelResolver<C> {
    pub fn new(catalog: C, local_model_dirs: Vec<PathBuf>) -> Self {
        Self {
            catalog,
            local_model_dirs,
        }
    }

    /// Resolve a user input string to candidate model artifacts.
    /// Returns candidates in preference order (local before remote, layer-package before GGUF).
    pub fn resolve(&self, input: &str) -> Result<Vec<ModelArtifactCandidate>> {
        let mut candidates = Vec::new();

        // 1. Check if it's a local file or layer-package path
        let path = PathBuf::from(input);
        if let Some(package) = local_layer_package_candidate(&path)? {
            candidates.push(ModelArtifactCandidate::LocalLayerPackage(package));
            return Ok(candidates);
        }
        if path.exists() && path.extension().is_some_and(|ext| ext == "gguf") {
            candidates.push(ModelArtifactCandidate::LocalGguf(LocalGguf {
                path: path.canonicalize().unwrap_or(path),
                identity: None,
            }));
            return Ok(candidates);
        }

        // 2. Search local model dirs for bare names
        if !input.contains('/') {
            let stem = input.strip_suffix(".gguf").unwrap_or(input);
            for dir in &self.local_model_dirs {
                let package_path = dir.join(stem);
                if let Some(package) = local_layer_package_candidate(&package_path)? {
                    candidates.push(ModelArtifactCandidate::LocalLayerPackage(package));
                    return Ok(candidates);
                }

                let candidate_path = dir.join(format!("{stem}.gguf"));
                if candidate_path.exists() {
                    candidates.push(ModelArtifactCandidate::LocalGguf(LocalGguf {
                        path: candidate_path,
                        identity: None,
                    }));
                    return Ok(candidates);
                }
            }
        }

        // 3. Search catalog
        let query_lower = input.to_lowercase();
        for entry in self.catalog.entries() {
            let mut variants: Vec<_> = entry.variants.iter().collect();
            variants.sort_by_key(|(name, _)| *name);

            for (variant_name, variant) in variants {
                let matches = variant_name.to_lowercase().contains(&query_lower)
                    || variant.curated.name.to_lowercase().contains(&query_lower)
                    || entry.source_repo.to_lowercase().contains(&query_lower);

                if !matches {
                    continue;
                }

                // Check for layer packages first (preferred)
                let mut packages: Vec<_> = variant.packages.iter().collect();
                packages.sort_by(|a, b| a.repo.cmp(&b.repo));
                for package in &packages {
                    if package.package_type == "layer-package" {
                        candidates.push(ModelArtifactCandidate::RemoteLayerPackage(
                            RemoteLayerPackage {
                                package_repo: package.repo.clone(),
                                layer_count: package.layer_count,
                                total_bytes: package.total_bytes,
                                source: catalog_source_with_default_file(
                                    &variant.source,
                                    variant_name,
                                ),
                                curated: Some(variant.curated.clone()),
                            },
                        ));
                    }
                }

                // Then add the GGUF candidate
                candidates.push(ModelArtifactCandidate::RemoteGguf(RemoteGguf {
                    source: catalog_source_with_default_file(&variant.source, variant_name),
                    curated: Some(variant.curated.clone()),
                }));

                // Return first match's candidates
                if !candidates.is_empty() {
                    return Ok(candidates);
                }
            }
        }

        // 4. Try parsing as a ModelRef (org/repo:selector)
        if let Ok(model_ref) = ModelRef::parse(input) {
            candidates.push(ModelArtifactCandidate::RemoteGguf(RemoteGguf {
                source: CatalogSource {
                    repo: model_ref.repo,
                    revision: model_ref.revision,
                    file: None,
                },
                curated: None,
            }));
            return Ok(candidates);
        }

        Ok(candidates)
    }
}

fn local_layer_package_candidate(path: &Path) -> Result<Option<LocalLayerPackage>> {
    if !path.is_dir() {
        return Ok(None);
    }

    let manifest_path = path.join("model-package.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }

    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest = serde_json::from_slice(&manifest_bytes)?;
    Ok(Some(LocalLayerPackage {
        path: path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
        manifest,
    }))
}

fn catalog_source_with_default_file(source: &CatalogSource, variant_name: &str) -> CatalogSource {
    CatalogSource {
        repo: source.repo.clone(),
        revision: source.revision.clone(),
        file: source
            .file
            .clone()
            .or_else(|| Some(format!("{variant_name}.gguf"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{CatalogPackage, CatalogVariant, CuratedMeta};
    use std::collections::HashMap;

    fn make_entry(
        source_repo: &str,
        variant_name: &str,
        curated_name: &str,
        packages: &[(&str, Option<u32>, Option<u64>)],
    ) -> CatalogEntry {
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
                    name: curated_name.to_string(),
                    size: None,
                    description: None,
                    draft: None,
                    moe: None,
                    extra_files: Vec::new(),
                    mmproj: None,
                },
                packages: packages
                    .iter()
                    .map(|(repo, layer_count, total_bytes)| CatalogPackage {
                        package_type: "layer-package".to_string(),
                        repo: repo.to_string(),
                        layer_count: *layer_count,
                        total_bytes: *total_bytes,
                    })
                    .collect(),
            },
        );
        CatalogEntry {
            schema_version: 1,
            source_repo: source_repo.to_string(),
            variants,
        }
    }

    #[test]
    fn resolves_bare_name_from_catalog() {
        let entry = make_entry(
            "test-org/test-repo",
            "test-model-Q4_K_M",
            "Test Model Q4",
            &[],
        );
        let catalog = MemoryCatalog::new(vec![entry]);
        let resolver = ModelResolver::new(catalog, vec![]);

        let candidates = resolver.resolve("test-model").unwrap();
        assert_eq!(candidates.len(), 1);
        match &candidates[0] {
            ModelArtifactCandidate::RemoteGguf(remote) => {
                assert_eq!(remote.source.repo, "test-org/test-repo");
                assert!(remote.curated.is_some());
                assert_eq!(remote.curated.as_ref().unwrap().name, "Test Model Q4");
            }
            other => panic!("expected RemoteGguf, got {other:?}"),
        }
    }

    #[test]
    fn resolves_name_with_layer_package() {
        let entry = make_entry(
            "test-org/test-repo",
            "test-model-Q4_K_M",
            "Test Model Q4",
            &[("meshllm/test-layers", Some(32), Some(1_000_000))],
        );
        let catalog = MemoryCatalog::new(vec![entry]);
        let resolver = ModelResolver::new(catalog, vec![]);

        let candidates = resolver.resolve("test-model").unwrap();
        assert_eq!(candidates.len(), 2);
        assert!(
            matches!(&candidates[0], ModelArtifactCandidate::RemoteLayerPackage(p) if p.package_repo == "meshllm/test-layers")
        );
        assert!(matches!(
            &candidates[1],
            ModelArtifactCandidate::RemoteGguf(_)
        ));
    }

    #[test]
    fn catalog_candidates_default_missing_source_file_to_variant_name() {
        let mut variants = HashMap::new();
        variants.insert(
            "test-model-Q4_K_M".to_string(),
            CatalogVariant {
                source: CatalogSource {
                    repo: "test-org/test-repo".to_string(),
                    revision: Some("main".to_string()),
                    file: None,
                },
                curated: CuratedMeta {
                    name: "Test Model Q4".to_string(),
                    size: None,
                    description: None,
                    draft: None,
                    moe: None,
                    extra_files: Vec::new(),
                    mmproj: None,
                },
                packages: vec![CatalogPackage {
                    package_type: "layer-package".to_string(),
                    repo: "meshllm/test-layers".to_string(),
                    layer_count: None,
                    total_bytes: None,
                }],
            },
        );
        let resolver = ModelResolver::new(
            MemoryCatalog::new(vec![CatalogEntry {
                schema_version: 1,
                source_repo: "test-org/test-repo".to_string(),
                variants,
            }]),
            vec![],
        );

        let candidates = resolver.resolve("test-model").unwrap();

        match &candidates[0] {
            ModelArtifactCandidate::RemoteLayerPackage(package) => {
                assert_eq!(
                    package.source.file.as_deref(),
                    Some("test-model-Q4_K_M.gguf")
                );
            }
            other => panic!("expected RemoteLayerPackage, got {other:?}"),
        }
        match &candidates[1] {
            ModelArtifactCandidate::RemoteGguf(remote) => {
                assert_eq!(
                    remote.source.file.as_deref(),
                    Some("test-model-Q4_K_M.gguf")
                );
            }
            other => panic!("expected RemoteGguf, got {other:?}"),
        }
    }

    #[test]
    fn resolves_local_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("test-model.gguf");
        std::fs::write(&gguf_path, b"fake gguf").unwrap();

        let resolver = ModelResolver::new(MemoryCatalog::empty(), vec![]);
        let candidates = resolver.resolve(gguf_path.to_str().unwrap()).unwrap();

        assert_eq!(candidates.len(), 1);
        match &candidates[0] {
            ModelArtifactCandidate::LocalGguf(local) => {
                assert_eq!(
                    local.path.file_name().unwrap().to_str().unwrap(),
                    "test-model.gguf"
                );
                assert!(local.identity.is_none());
            }
            other => panic!("expected LocalGguf, got {other:?}"),
        }
    }

    #[test]
    fn resolves_local_layer_package_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model-package.json"),
            r#"{"source_model":{"size_bytes":4800000000},"layers":[]}"#,
        )
        .unwrap();

        let resolver = ModelResolver::new(MemoryCatalog::empty(), vec![]);
        let candidates = resolver.resolve(dir.path().to_str().unwrap()).unwrap();

        assert_eq!(candidates.len(), 1);
        match &candidates[0] {
            ModelArtifactCandidate::LocalLayerPackage(local) => {
                assert_eq!(local.path, dir.path().canonicalize().unwrap());
                assert_eq!(
                    local.manifest["source_model"]["size_bytes"],
                    4_800_000_000u64
                );
            }
            other => panic!("expected LocalLayerPackage, got {other:?}"),
        }
    }

    #[test]
    fn resolves_local_layer_package_from_model_dirs() {
        let models_dir = tempfile::tempdir().unwrap();
        let package_dir = models_dir.path().join("test-package");
        std::fs::create_dir(&package_dir).unwrap();
        std::fs::write(package_dir.join("model-package.json"), r#"{"layers":[1]}"#).unwrap();

        let resolver = ModelResolver::new(MemoryCatalog::empty(), vec![models_dir.path().into()]);
        let candidates = resolver.resolve("test-package").unwrap();

        assert_eq!(candidates.len(), 1);
        assert!(matches!(
            &candidates[0],
            ModelArtifactCandidate::LocalLayerPackage(local) if local.path == package_dir.canonicalize().unwrap()
        ));
    }

    #[test]
    fn resolves_unknown_returns_empty() {
        let resolver = ModelResolver::new(MemoryCatalog::empty(), vec![]);
        let candidates = resolver.resolve("nonexistent-model").unwrap();
        assert!(candidates.is_empty());
    }

    #[test]
    fn resolves_model_ref_to_remote_gguf() {
        let resolver = ModelResolver::new(MemoryCatalog::empty(), vec![]);
        let candidates = resolver.resolve("testorg/testrepo:Q4_K_M").unwrap();

        assert_eq!(candidates.len(), 1);
        match &candidates[0] {
            ModelArtifactCandidate::RemoteGguf(remote) => {
                assert_eq!(remote.source.repo, "testorg/testrepo");
                assert!(remote.source.revision.is_none());
                assert!(remote.curated.is_none());
            }
            other => panic!("expected RemoteGguf, got {other:?}"),
        }
    }
}
