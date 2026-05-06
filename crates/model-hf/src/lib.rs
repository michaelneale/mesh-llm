use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use hf_hub::{
    cache::scan_cache_dir,
    types::cache::{CachedRepoInfo, HFCacheInfo},
    types::repo::ModelInfo,
    HFClient, HFClientBuilder, RepoDownloadFileParams, RepoInfo, RepoInfoParams, RepoType,
};
use model_artifact::{ModelArtifactFile, ModelIdentity, ModelRepository, ResolvedModelArtifact};
use model_ref::{
    format_canonical_ref, format_model_ref, normalize_gguf_distribution_id,
    quant_selector_from_gguf_file,
};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct HfModelRepository {
    api: HFClient,
    cache_dir: PathBuf,
}

impl HfModelRepository {
    pub fn from_env() -> Result<Self> {
        Self::builder().build()
    }

    pub fn builder() -> HfModelRepositoryBuilder {
        HfModelRepositoryBuilder::default()
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub async fn download_file(&self, repo: &str, revision: &str, file: &str) -> Result<PathBuf> {
        let (owner, name) = repo_parts(repo);
        self.api
            .repo(RepoType::Model, owner, name)
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(file.to_string())
                    .revision(revision.to_string())
                    .build(),
            )
            .await
            .with_context(|| format!("download Hugging Face model file {repo}@{revision}/{file}"))
    }

    pub async fn download_artifact_files(
        &self,
        artifact: &ResolvedModelArtifact,
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::with_capacity(artifact.files.len());
        for file in &artifact.files {
            paths.push(
                self.download_file(&artifact.source_repo, &artifact.source_revision, &file.path)
                    .await?,
            );
        }
        Ok(paths)
    }

    pub fn identity_for_path(&self, path: &Path) -> Option<HfModelIdentity> {
        huggingface_identity_for_path_in_cache(path, &self.cache_dir)
    }
}

#[derive(Default)]
pub struct HfModelRepositoryBuilder {
    cache_dir: Option<PathBuf>,
    endpoint: Option<String>,
    token: Option<String>,
}

impl HfModelRepositoryBuilder {
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn build(self) -> Result<HfModelRepository> {
        let cache_dir = self.cache_dir.unwrap_or_else(huggingface_hub_cache_dir);
        let mut builder = HFClientBuilder::new().cache_dir(cache_dir.clone());

        let endpoint = self
            .endpoint
            .or_else(|| std::env::var("HF_ENDPOINT").ok())
            .map(|endpoint| endpoint.trim().to_string())
            .filter(|endpoint| !endpoint.is_empty());
        if let Some(endpoint) = endpoint {
            builder = builder.endpoint(endpoint);
        }

        let token = self.token.or_else(hf_token_override);
        if let Some(token) = token {
            builder = builder.token(token);
        }

        let api = builder.build().context("build Hugging Face API client")?;
        Ok(HfModelRepository { api, cache_dir })
    }
}

#[async_trait]
impl ModelRepository for HfModelRepository {
    async fn resolve_revision(&self, repo: &str, revision: Option<&str>) -> Result<String> {
        let revision = revision.unwrap_or("main");
        self.repo_info(repo, revision)
            .await?
            .sha
            .with_context(|| format!("Hugging Face repo {repo}@{revision} did not return a sha"))
    }

    async fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<ModelArtifactFile>> {
        let info = self.repo_info(repo, revision).await?;
        Ok(info
            .siblings
            .unwrap_or_default()
            .into_iter()
            .map(|sibling| ModelArtifactFile {
                path: sibling.rfilename,
                size_bytes: sibling.size,
                sha256: None,
            })
            .collect())
    }
}

impl HfModelRepository {
    async fn repo_info(&self, repo: &str, revision: &str) -> Result<ModelInfo> {
        let (owner, name) = repo_parts(repo);
        let detail = self
            .api
            .model(owner, name)
            .info(
                &RepoInfoParams::builder()
                    .revision(revision.to_string())
                    .build(),
            )
            .await
            .with_context(|| format!("fetch Hugging Face model repo {repo}@{revision}"))?;
        let RepoInfo::Model(detail) = detail else {
            bail!("expected Hugging Face model repo info for {repo}@{revision}");
        };
        Ok(detail)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfModelIdentity {
    pub model_id: String,
    pub repo_id: String,
    pub revision: String,
    pub file: String,
    pub canonical_ref: String,
    pub distribution_id: Option<String>,
    pub selector: Option<String>,
}

impl HfModelIdentity {
    pub fn to_model_identity(&self) -> ModelIdentity {
        ModelIdentity {
            model_id: self.model_id.clone(),
            source_repo: Some(self.repo_id.clone()),
            source_revision: Some(self.revision.clone()),
            source_file: Some(self.file.clone()),
            canonical_ref: Some(self.canonical_ref.clone()),
            distribution_id: self.distribution_id.clone(),
            selector: self.selector.clone(),
        }
    }

    pub fn distribution_ref(&self) -> Option<String> {
        self.distribution_id.as_ref().map(|distribution_id| {
            format!("{}@{}/{}", self.repo_id, self.revision, distribution_id)
        })
    }
}

pub fn huggingface_hub_cache_dir() -> PathBuf {
    if let Some(path) = env_path("HF_HUB_CACHE") {
        return path;
    }
    if let Some(path) = env_path("HUGGINGFACE_HUB_CACHE") {
        return path;
    }
    if let Some(path) = env_path("HF_HOME") {
        return path.join("hub");
    }
    if let Some(path) = env_path("XDG_CACHE_HOME") {
        return path.join("huggingface").join("hub");
    }
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

pub fn hf_token_override() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(key) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

pub fn huggingface_repo_folder_name(repo_id: &str, repo_type: RepoType) -> String {
    let type_plural = format!("{}s", repo_type);
    std::iter::once(type_plural.as_str())
        .chain(repo_id.split('/'))
        .collect::<Vec<_>>()
        .join("--")
}

pub fn huggingface_snapshot_path(repo_id: &str, repo_type: RepoType, revision: &str) -> PathBuf {
    huggingface_hub_cache_dir()
        .join(huggingface_repo_folder_name(repo_id, repo_type))
        .join("snapshots")
        .join(revision)
}

pub fn huggingface_identity_for_path_in_cache(
    path: &Path,
    cache_root: &Path,
) -> Option<HfModelIdentity> {
    if let Some(identity) = identity_from_cache_snapshot_path(path, cache_root) {
        return Some(identity);
    }
    let resolved_cache_root = cache_root
        .canonicalize()
        .unwrap_or_else(|_| cache_root.to_path_buf());
    if resolved_cache_root != cache_root {
        if let Some(identity) = identity_from_cache_snapshot_path(path, &resolved_cache_root) {
            return Some(identity);
        }
    }
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if resolved != path {
        if let Some(identity) = identity_from_cache_snapshot_path(&resolved, cache_root) {
            return Some(identity);
        }
        if resolved_cache_root != cache_root {
            if let Some(identity) =
                identity_from_cache_snapshot_path(&resolved, &resolved_cache_root)
            {
                return Some(identity);
            }
        }
    }
    scan_hf_cache_identity_for_path(path, cache_root)
}

fn identity_from_cache_snapshot_path(path: &Path, cache_root: &Path) -> Option<HfModelIdentity> {
    let relative = path.strip_prefix(cache_root).ok()?;
    let mut components = relative.components();
    let repo_folder = components.next()?.as_os_str().to_str()?;
    let repo_id = parse_model_repo_folder_name(repo_folder)?;
    if components.next()?.as_os_str() != OsStr::new("snapshots") {
        return None;
    }
    let revision = components.next()?.as_os_str().to_str()?.to_string();
    let file = components
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()?
        .join("/");
    if file.is_empty() {
        return None;
    }
    Some(identity_from_parts(repo_id, revision, file))
}

fn scan_hf_cache_identity_for_path(path: &Path, cache_root: &Path) -> Option<HfModelIdentity> {
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

                return Some(identity_from_parts(
                    repo_id.to_string(),
                    revision.commit_hash.clone(),
                    relative_path,
                ));
            }
        }
    }
    None
}

fn scan_hf_cache_info(cache_root: &Path) -> Option<HFCacheInfo> {
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

fn identity_from_parts(repo_id: String, revision: String, file: String) -> HfModelIdentity {
    let selector = quant_selector_from_gguf_file(&file);
    let model_id = format_model_ref(&repo_id, None, selector.as_deref());
    let distribution_id = normalize_gguf_distribution_id(&file);
    let canonical_ref = format_canonical_ref(&repo_id, &revision, &file);
    HfModelIdentity {
        model_id,
        repo_id,
        revision,
        file,
        canonical_ref,
        distribution_id,
        selector,
    }
}

fn cache_repo_id(repo: &CachedRepoInfo) -> Option<&str> {
    (repo.repo_type == RepoType::Model).then_some(repo.repo_id.as_str())
}

fn parse_model_repo_folder_name(folder: &str) -> Option<String> {
    folder
        .strip_prefix("models--")
        .map(|value| value.replace("--", "/"))
}

fn repo_parts(repo: &str) -> (&str, &str) {
    repo.split_once('/').unwrap_or(("", repo))
}

fn env_path(key: &str) -> Option<PathBuf> {
    let value = std::env::var(key).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_path_identity_matches_mesh_snapshot_layout() {
        let cache_root = PathBuf::from("/cache/hub");
        let path = cache_root
            .join("models--org--repo")
            .join("snapshots")
            .join("abc123")
            .join("Qwen3-8B-Q4_K_M.gguf");

        let identity = huggingface_identity_for_path_in_cache(&path, &cache_root).unwrap();
        assert_eq!(identity.model_id, "org/repo:Q4_K_M");
        assert_eq!(identity.repo_id, "org/repo");
        assert_eq!(identity.revision, "abc123");
        assert_eq!(identity.file, "Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(
            identity.canonical_ref,
            "org/repo@abc123/Qwen3-8B-Q4_K_M.gguf"
        );
        assert_eq!(identity.distribution_id.as_deref(), Some("Qwen3-8B-Q4_K_M"));
        assert_eq!(
            identity.distribution_ref().as_deref(),
            Some("org/repo@abc123/Qwen3-8B-Q4_K_M")
        );
    }

    #[test]
    fn cache_path_identity_collapses_split_gguf_distribution() {
        let cache_root = PathBuf::from("/cache/hub");
        let path = cache_root
            .join("models--org--repo")
            .join("snapshots")
            .join("abc123")
            .join("UD-IQ2_M")
            .join("GLM-5.1-UD-IQ2_M-00001-of-00006.gguf");

        let identity = huggingface_identity_for_path_in_cache(&path, &cache_root).unwrap();
        assert_eq!(identity.model_id, "org/repo:UD-IQ2_M");
        assert_eq!(identity.selector.as_deref(), Some("UD-IQ2_M"));
        assert_eq!(
            identity.distribution_id.as_deref(),
            Some("GLM-5.1-UD-IQ2_M")
        );
    }

    #[test]
    fn repo_folder_name_matches_huggingface_cache_layout() {
        assert_eq!(
            huggingface_repo_folder_name("org/repo", RepoType::Model),
            "models--org--repo"
        );
    }
}
