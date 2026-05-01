use std::path::Path;

use anyhow::{bail, Result};
use async_trait::async_trait;
use model_ref::{
    format_canonical_ref, gguf_matches_quant_selector, normalize_gguf_distribution_id,
    parse_model_ref, split_gguf_shard_info, ModelRef,
};
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait ModelRepository: Send + Sync {
    async fn resolve_revision(&self, repo: &str, revision: Option<&str>) -> Result<String>;

    async fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<ModelArtifactFile>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModelArtifact {
    pub model_id: String,
    pub source_repo: String,
    pub source_revision: String,
    pub selector: Option<String>,
    pub format: ModelFormat,
    pub files: Vec<ModelArtifactFile>,
    pub primary_file: String,
    pub canonical_ref: String,
    pub distribution_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selector: Option<String>,
}

impl ModelIdentity {
    pub fn from_model_id(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            source_repo: None,
            source_revision: None,
            source_file: None,
            canonical_ref: None,
            distribution_id: None,
            selector: None,
        }
    }
}

impl From<&ResolvedModelArtifact> for ModelIdentity {
    fn from(artifact: &ResolvedModelArtifact) -> Self {
        Self {
            model_id: artifact.model_id.clone(),
            source_repo: Some(artifact.source_repo.clone()),
            source_revision: Some(artifact.source_revision.clone()),
            source_file: Some(artifact.primary_file.clone()),
            canonical_ref: Some(artifact.canonical_ref.clone()),
            distribution_id: Some(artifact.distribution_id.clone()),
            selector: artifact.selector.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactFile {
    pub path: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

impl ModelArtifactFile {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            size_bytes: None,
            sha256: None,
        }
    }
}

pub async fn resolve_model_artifact_ref(
    model_ref: &str,
    repository: &impl ModelRepository,
) -> Result<ResolvedModelArtifact> {
    let parsed = parse_model_ref(model_ref)?;
    resolve_model_artifact(&parsed, repository).await
}

pub async fn resolve_model_artifact(
    model_ref: &ModelRef,
    repository: &impl ModelRepository,
) -> Result<ResolvedModelArtifact> {
    let source_revision = repository
        .resolve_revision(&model_ref.repo, model_ref.revision.as_deref())
        .await?;
    let mut repo_files = repository
        .list_files(&model_ref.repo, &source_revision)
        .await?;
    repo_files.sort_by(|left, right| left.path.cmp(&right.path));

    let primary_file = select_primary_file(model_ref.selector.as_deref(), &repo_files)?;
    let format = format_for_file(&primary_file.path)?;
    let files = artifact_file_set(&primary_file.path, &repo_files);
    let distribution_id = distribution_id_for_file(&primary_file.path)?;

    Ok(ResolvedModelArtifact {
        model_id: model_ref.display_id(),
        source_repo: model_ref.repo.clone(),
        source_revision: source_revision.clone(),
        selector: model_ref.selector.clone(),
        format,
        files,
        primary_file: primary_file.path.clone(),
        canonical_ref: format_canonical_ref(&model_ref.repo, &source_revision, &primary_file.path),
        distribution_id,
    })
}

fn select_primary_file(
    selector: Option<&str>,
    files: &[ModelArtifactFile],
) -> Result<ModelArtifactFile> {
    let Some(selector) = selector else {
        return select_default_file(files);
    };

    let selector_lower = selector.to_ascii_lowercase();
    let gguf_exact = format!("{selector}.gguf").to_ascii_lowercase();
    let gguf_split_prefix = format!("{selector}-00001-of-").to_ascii_lowercase();
    let safetensors_exact = format!("{selector}.safetensors").to_ascii_lowercase();
    let safetensors_split_prefix = format!("{selector}-00001-of-").to_ascii_lowercase();

    files
        .iter()
        .filter_map(|file| {
            let lower = file.path.to_ascii_lowercase();
            let basename = basename_lower(&file.path);
            let rank = if lower == selector_lower || basename == selector_lower {
                0
            } else if gguf_matches_quant_selector(&file.path, selector) {
                1
            } else if basename == safetensors_exact {
                2
            } else if basename.starts_with(&safetensors_split_prefix)
                && basename.ends_with(".safetensors")
            {
                3
            } else if basename == gguf_exact {
                4
            } else if basename.starts_with(&gguf_split_prefix) && basename.ends_with(".gguf") {
                5
            } else {
                return None;
            };
            Some((rank, file.path.clone(), file.clone()))
        })
        .min_by(|left, right| (left.0, &left.1).cmp(&(right.0, &right.1)))
        .map(|(_, _, file)| file)
        .ok_or_else(|| {
            anyhow::anyhow!("no model artifact matching selector '{selector}' in repository")
        })
}

fn select_default_file(files: &[ModelArtifactFile]) -> Result<ModelArtifactFile> {
    files
        .iter()
        .filter_map(|file| {
            let lower = file.path.to_ascii_lowercase();
            let basename = basename_lower(&file.path);
            let rank = if basename == "model.safetensors" {
                0
            } else if is_split_safetensors_first_shard(&basename) {
                1
            } else if lower.ends_with(".gguf") {
                if is_known_gguf_sidecar(&basename) {
                    return None;
                }
                if lower.contains("-000") && !lower.contains("-00001-of-") {
                    return None;
                }
                if lower.contains("-00001-of-") {
                    2
                } else {
                    3
                }
            } else {
                return None;
            };
            Some((rank, file.path.clone(), file.clone()))
        })
        .min_by(|left, right| (left.0, &left.1).cmp(&(right.0, &right.1)))
        .map(|(_, _, file)| file)
        .ok_or_else(|| anyhow::anyhow!("no supported model artifact files found in repository"))
}

fn artifact_file_set(primary_file: &str, files: &[ModelArtifactFile]) -> Vec<ModelArtifactFile> {
    if let Some(primary) = split_gguf_shard_info(primary_file) {
        let mut shards = files
            .iter()
            .filter(|file| {
                split_gguf_shard_info(&file.path)
                    .map(|candidate| {
                        candidate.prefix == primary.prefix && candidate.total == primary.total
                    })
                    .unwrap_or(false)
            })
            .cloned()
            .collect::<Vec<_>>();
        shards.sort_by(|left, right| left.path.cmp(&right.path));
        if !shards.is_empty() {
            return shards;
        }
    }

    vec![files
        .iter()
        .find(|file| file.path == primary_file)
        .cloned()
        .unwrap_or_else(|| ModelArtifactFile::new(primary_file))]
}

fn format_for_file(file: &str) -> Result<ModelFormat> {
    if file.ends_with(".gguf") {
        return Ok(ModelFormat::Gguf);
    }
    if file.ends_with(".safetensors") || file.ends_with(".safetensors.index.json") {
        return Ok(ModelFormat::Safetensors);
    }
    bail!("unsupported model artifact file format: {file}")
}

fn distribution_id_for_file(file: &str) -> Result<String> {
    if file.ends_with(".gguf") {
        return normalize_gguf_distribution_id(file)
            .ok_or_else(|| anyhow::anyhow!("invalid GGUF artifact file name: {file}"));
    }
    let basename = Path::new(file)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(file);
    let stem = basename.strip_suffix(".safetensors").unwrap_or(basename);
    Ok(split_safetensors_shard_stem_prefix(stem)
        .unwrap_or(stem)
        .to_string())
}

fn basename_lower(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(path)
        .to_ascii_lowercase()
}

fn is_known_gguf_sidecar(basename_lower: &str) -> bool {
    basename_lower.starts_with("mmproj")
}

fn is_split_safetensors_first_shard(basename_lower: &str) -> bool {
    let Some(stem) = basename_lower.strip_suffix(".safetensors") else {
        return false;
    };
    split_safetensors_shard_info(stem)
        .map(|(_, part, _)| part == "00001")
        .unwrap_or(false)
}

fn split_safetensors_shard_stem_prefix(stem: &str) -> Option<&str> {
    split_safetensors_shard_info(stem).map(|(prefix, _, _)| prefix)
}

fn split_safetensors_shard_info(stem: &str) -> Option<(&str, &str, &str)> {
    let (prefix_and_part, total) = stem.rsplit_once("-of-")?;
    if total.len() != 5 || !total.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let (prefix, part) = prefix_and_part.rsplit_once('-')?;
    if part.len() != 5 || !part.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some((prefix, part, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MemoryRepository {
        revision: String,
        files: HashMap<String, Vec<ModelArtifactFile>>,
    }

    #[async_trait]
    impl ModelRepository for MemoryRepository {
        async fn resolve_revision(&self, _repo: &str, revision: Option<&str>) -> Result<String> {
            Ok(revision.unwrap_or(&self.revision).to_string())
        }

        async fn list_files(&self, repo: &str, _revision: &str) -> Result<Vec<ModelArtifactFile>> {
            Ok(self.files.get(repo).cloned().unwrap_or_default())
        }
    }

    fn repo(files: Vec<&str>) -> MemoryRepository {
        MemoryRepository {
            revision: "abc123".to_string(),
            files: HashMap::from([(
                "org/repo".to_string(),
                files.into_iter().map(ModelArtifactFile::new).collect(),
            )]),
        }
    }

    #[tokio::test]
    async fn resolves_quant_selector_to_gguf_file() {
        let repository = repo(vec!["Model-Q5_K_M.gguf", "Model-Q4_K_M.gguf", "README.md"]);

        let resolved = resolve_model_artifact_ref("org/repo:Q4_K_M", &repository)
            .await
            .unwrap();

        assert_eq!(resolved.model_id, "org/repo:Q4_K_M");
        assert_eq!(resolved.source_revision, "abc123");
        assert_eq!(resolved.primary_file, "Model-Q4_K_M.gguf");
        assert_eq!(resolved.canonical_ref, "org/repo@abc123/Model-Q4_K_M.gguf");
        assert_eq!(resolved.distribution_id, "Model-Q4_K_M");
        assert_eq!(resolved.files.len(), 1);
    }

    #[tokio::test]
    async fn resolves_split_gguf_selector_to_all_shards() {
        let repository = repo(vec![
            "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00002-of-00003.gguf",
            "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00003.gguf",
            "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00003-of-00003.gguf",
            "UD-Q4_K_M/GLM-5.1-UD-Q4_K_M-00001-of-00003.gguf",
        ]);

        let resolved = resolve_model_artifact_ref("org/repo:UD-IQ2_M", &repository)
            .await
            .unwrap();

        assert_eq!(
            resolved.primary_file,
            "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00003.gguf"
        );
        assert_eq!(resolved.distribution_id, "GLM-5.1-UD-IQ2_M");
        assert_eq!(resolved.files.len(), 3);
        assert_eq!(
            resolved.files[2].path,
            "UD-IQ2_M/GLM-5.1-UD-IQ2_M-00003-of-00003.gguf"
        );
    }

    #[tokio::test]
    async fn accepts_revisioned_selector_refs() {
        let repository = repo(vec!["Model-Q4_K_M.gguf"]);

        let resolved = resolve_model_artifact_ref("org/repo:Q4_K_M@rev-1", &repository)
            .await
            .unwrap();

        assert_eq!(resolved.model_id, "org/repo@rev-1:Q4_K_M");
        assert_eq!(resolved.source_revision, "rev-1");
        assert_eq!(resolved.canonical_ref, "org/repo@rev-1/Model-Q4_K_M.gguf");
    }

    #[tokio::test]
    async fn default_selection_prefers_primary_weights() {
        let repository = repo(vec![
            "README.md",
            "Qwen3-8B-Q4_K_M.gguf",
            "Qwen3-8B-Q5_K_M.gguf",
        ]);

        let resolved = resolve_model_artifact_ref("org/repo", &repository)
            .await
            .unwrap();

        assert_eq!(resolved.primary_file, "Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(resolved.format, ModelFormat::Gguf);
    }

    #[tokio::test]
    async fn unknown_selector_returns_error() {
        let repository = repo(vec!["Model-Q4_K_M.gguf"]);

        let error = resolve_model_artifact_ref("org/repo:Q5_K_M", &repository)
            .await
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("no model artifact matching selector"));
    }
}
