use std::path::PathBuf;

use model_artifact::ModelIdentity;

use crate::catalog::{CatalogSource, CuratedMeta};

/// A discovered model artifact candidate, ranked by preference.
/// The resolver returns one or more of these; the caller picks the best.
#[derive(Debug, Clone)]
pub enum ModelArtifactCandidate {
    LocalGguf(LocalGguf),
    RemoteGguf(RemoteGguf),
    LocalLayerPackage(LocalLayerPackage),
    RemoteLayerPackage(RemoteLayerPackage),
}

#[derive(Debug, Clone)]
pub struct LocalGguf {
    pub path: PathBuf,
    pub identity: Option<ModelIdentity>,
}

#[derive(Debug, Clone)]
pub struct RemoteGguf {
    pub source: CatalogSource,
    pub curated: Option<CuratedMeta>,
}

#[derive(Debug, Clone)]
pub struct LocalLayerPackage {
    pub path: PathBuf,
    pub manifest: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct RemoteLayerPackage {
    pub package_repo: String,
    pub layer_count: Option<u32>,
    pub total_bytes: Option<u64>,
    pub source: CatalogSource,
    pub curated: Option<CuratedMeta>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_local_gguf() {
        let candidate = ModelArtifactCandidate::LocalGguf(LocalGguf {
            path: PathBuf::from("/models/test.gguf"),
            identity: None,
        });
        if let ModelArtifactCandidate::LocalGguf(local) = &candidate {
            assert_eq!(local.path, PathBuf::from("/models/test.gguf"));
            assert!(local.identity.is_none());
        } else {
            panic!("expected LocalGguf");
        }
    }

    #[test]
    fn construct_remote_gguf() {
        let candidate = ModelArtifactCandidate::RemoteGguf(RemoteGguf {
            source: CatalogSource {
                repo: "org/repo".to_string(),
                revision: Some("main".to_string()),
                file: Some("model.gguf".to_string()),
            },
            curated: Some(CuratedMeta {
                name: "Test Model".to_string(),
                size: None,
                description: None,
                draft: None,
                moe: None,
                extra_files: Vec::new(),
                mmproj: None,
            }),
        });
        if let ModelArtifactCandidate::RemoteGguf(remote) = &candidate {
            assert_eq!(remote.source.repo, "org/repo");
            assert_eq!(remote.source.revision.as_deref(), Some("main"));
            assert!(remote.curated.is_some());
            assert_eq!(remote.curated.as_ref().unwrap().name, "Test Model");
        } else {
            panic!("expected RemoteGguf");
        }
    }

    #[test]
    fn construct_local_layer_package() {
        let candidate = ModelArtifactCandidate::LocalLayerPackage(LocalLayerPackage {
            path: PathBuf::from("/packages/test"),
            manifest: serde_json::json!({"layers": 10}),
        });
        if let ModelArtifactCandidate::LocalLayerPackage(local) = &candidate {
            assert_eq!(local.path, PathBuf::from("/packages/test"));
            assert_eq!(local.manifest["layers"], 10);
        } else {
            panic!("expected LocalLayerPackage");
        }
    }

    #[test]
    fn construct_remote_layer_package() {
        let candidate = ModelArtifactCandidate::RemoteLayerPackage(RemoteLayerPackage {
            package_repo: "meshllm/test-layers".to_string(),
            layer_count: Some(32),
            total_bytes: Some(1_000_000),
            source: CatalogSource {
                repo: "org/repo".to_string(),
                revision: None,
                file: None,
            },
            curated: None,
        });
        if let ModelArtifactCandidate::RemoteLayerPackage(remote) = &candidate {
            assert_eq!(remote.package_repo, "meshllm/test-layers");
            assert_eq!(remote.layer_count, Some(32));
            assert_eq!(remote.total_bytes, Some(1_000_000));
            assert_eq!(remote.source.repo, "org/repo");
            assert!(remote.curated.is_none());
        } else {
            panic!("expected RemoteLayerPackage");
        }
    }
}
