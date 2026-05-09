use std::path::{Component, Path, PathBuf};
use std::{fs, io::Read};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(crate) const MODEL_TRANSFER_SUBPROTOCOL_NAME: &str = "mesh-model-transfer";
pub(crate) const MODEL_TRANSFER_SUBPROTOCOL_MAJOR: u32 = 1;
pub(crate) const MODEL_TRANSFER_FEATURE_FILE_GET: &str = "file-get";
pub(crate) const MODEL_TRANSFER_FEATURE_RANGE: &str = "range";
pub(crate) const MODEL_TRANSFER_FEATURE_SHA256: &str = "sha256";
pub(crate) const MODEL_TRANSFER_STREAM_FILE_GET: u8 = 0x01;
pub(crate) const MODEL_TRANSFER_GENERATION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub(crate) struct ModelFileTransferRequest {
    pub(crate) gen: u32,
    pub(crate) requester_id: Vec<u8>,
    pub(crate) request_id: String,
    pub(crate) repo: String,
    pub(crate) revision: String,
    pub(crate) file: String,
    pub(crate) offset: u64,
    pub(crate) expected_size: Option<u64>,
    pub(crate) expected_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub(crate) struct ModelFileTransferResponse {
    pub(crate) gen: u32,
    pub(crate) accepted: bool,
    pub(crate) resolved_revision: Option<String>,
    pub(crate) total_size: u64,
    pub(crate) sha256: Option<String>,
    pub(crate) offset: u64,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServableModelFile {
    pub(crate) path: PathBuf,
    pub(crate) resolved_revision: String,
    pub(crate) size: u64,
    pub(crate) sha256: String,
}

pub(crate) fn model_transfer_enabled() -> bool {
    std::env::var("MESH_LLM_MODEL_TRANSFER")
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

pub(crate) fn safe_relative_model_file_path(path: &str) -> Result<PathBuf> {
    anyhow::ensure!(!path.trim().is_empty(), "model file path is empty");
    anyhow::ensure!(
        !path.contains('\\'),
        "model file path must use Hugging Face forward-slash separators"
    );
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("model file path is empty");
    };
    anyhow::ensure!(
        matches!(first, Component::Normal(_))
            && components.all(|component| matches!(component, Component::Normal(_))),
        "model file path must be a safe relative path"
    );
    Ok(path.to_path_buf())
}

pub(crate) fn hf_model_cache_path(repo: &str, revision: &str, file: &str) -> Result<PathBuf> {
    validate_hf_model_repo(repo)?;
    validate_immutable_revision(revision)?;
    let file = safe_relative_model_file_path(file)?;
    Ok(hf_repo_cache_root(repo)
        .join("snapshots")
        .join(revision)
        .join(file))
}

pub(crate) fn cached_hf_model_file(
    repo: &str,
    revision: &str,
    file: &str,
) -> Result<Option<PathBuf>> {
    let resolved_revision = resolve_cached_hf_revision(repo, revision)?;
    let path = hf_model_cache_path(repo, &resolved_revision, file)?;
    if !path.exists() {
        return Ok(None);
    }
    ensure_path_inside_repo_root(&hf_repo_cache_root(repo), &path)?;
    let metadata = fs::metadata(&path).context("stat cached HF model file")?;
    anyhow::ensure!(metadata.is_file(), "cached HF model path is not a file");
    Ok(Some(path))
}

pub(crate) fn resolve_cached_hf_revision(repo: &str, revision: &str) -> Result<String> {
    validate_hf_model_repo(repo)?;
    validate_hf_revision_ref(revision)?;
    if is_sha_revision(revision) {
        return Ok(revision.to_ascii_lowercase());
    }
    let refs_path = hf_repo_cache_root(repo).join("refs").join(revision);
    let parent = refs_path
        .parent()
        .context("HF revision ref path has no parent directory")?;
    ensure_path_inside_repo_root(&hf_repo_cache_root(repo), parent)
        .context("HF revision ref escapes the managed cache repo")?;
    let resolved = fs::read_to_string(&refs_path)
        .with_context(|| format!("read cached HF revision ref {}", refs_path.display()))?;
    let resolved = resolved.trim();
    validate_immutable_revision(resolved)?;
    Ok(resolved.to_ascii_lowercase())
}

pub(crate) fn install_hf_revision_ref(
    repo: &str,
    revision: &str,
    resolved_revision: &str,
) -> Result<()> {
    validate_hf_model_repo(repo)?;
    validate_hf_revision_ref(revision)?;
    validate_immutable_revision(resolved_revision)?;
    if is_sha_revision(revision) {
        return Ok(());
    }
    let refs_path = hf_repo_cache_root(repo).join("refs").join(revision);
    let parent = refs_path
        .parent()
        .context("HF revision ref path has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create HF revision refs dir {}", parent.display()))?;
    ensure_path_inside_repo_root(&hf_repo_cache_root(repo), parent)
        .context("HF revision ref escapes the managed cache repo")?;
    fs::write(&refs_path, resolved_revision.to_ascii_lowercase())
        .with_context(|| format!("write HF revision ref {}", refs_path.display()))?;
    Ok(())
}

pub(crate) fn is_immutable_hf_revision(revision: &str) -> bool {
    is_sha_revision(revision)
}

pub(crate) fn validate_model_file_transfer_request(
    request: &ModelFileTransferRequest,
) -> Result<()> {
    anyhow::ensure!(
        request.gen == MODEL_TRANSFER_GENERATION,
        "unsupported model transfer generation"
    );
    anyhow::ensure!(
        request.requester_id.len() == 32,
        "requester_id must be a 32-byte endpoint id"
    );
    anyhow::ensure!(
        !request.request_id.trim().is_empty(),
        "request_id is required"
    );
    validate_hf_model_repo(&request.repo)?;
    validate_hf_revision_ref(&request.revision)?;
    safe_relative_model_file_path(&request.file)?;
    if let Some(expected_sha) = request.expected_sha256.as_deref() {
        validate_sha256(expected_sha)?;
    }
    Ok(())
}

pub(crate) fn validate_model_file_transfer_response(
    response: &ModelFileTransferResponse,
    requested_offset: u64,
) -> Result<()> {
    anyhow::ensure!(
        response.gen == MODEL_TRANSFER_GENERATION,
        "unsupported model transfer response generation"
    );
    anyhow::ensure!(
        response.offset == requested_offset,
        "model transfer response offset mismatch"
    );
    if !response.accepted {
        return Ok(());
    }
    let resolved_revision = response
        .resolved_revision
        .as_deref()
        .context("accepted model transfer response missing resolved revision")?;
    validate_immutable_revision(resolved_revision)?;
    if let Some(sha256) = response.sha256.as_deref() {
        validate_sha256(sha256)?;
    } else {
        bail!("accepted model transfer response missing sha256");
    }
    anyhow::ensure!(
        requested_offset <= response.total_size,
        "model transfer response is smaller than resume offset"
    );
    Ok(())
}

pub(crate) fn servable_model_file_from_request(
    request: &ModelFileTransferRequest,
) -> Result<ServableModelFile> {
    validate_model_file_transfer_request(request)?;
    let resolved_revision = resolve_cached_hf_revision(&request.repo, &request.revision)?;
    let path = hf_model_cache_path(&request.repo, &resolved_revision, &request.file)?;
    let repo_root = hf_repo_cache_root(&request.repo);
    ensure_path_inside_repo_root(&repo_root, &path)?;
    let metadata = fs::metadata(&path).context("model file is not cached")?;
    anyhow::ensure!(metadata.is_file(), "model cache entry is not a file");
    if let Some(expected_size) = request.expected_size {
        anyhow::ensure!(metadata.len() == expected_size, "model file size mismatch");
    }
    anyhow::ensure!(
        request.offset <= metadata.len(),
        "model transfer offset exceeds file size"
    );
    let sha256 = file_sha256_hex(&path)?;
    if let Some(expected_sha) = request.expected_sha256.as_deref() {
        anyhow::ensure!(
            sha256.eq_ignore_ascii_case(expected_sha),
            "model file sha256 mismatch"
        );
    }
    Ok(ServableModelFile {
        path,
        resolved_revision,
        size: metadata.len(),
        sha256,
    })
}

pub(crate) fn ensure_model_file_install_parent(repo: &str, destination: &Path) -> Result<()> {
    validate_hf_model_repo(repo)?;
    let parent = destination
        .parent()
        .context("model file destination has no parent directory")?;
    ensure_path_inside_repo_root(&hf_repo_cache_root(repo), parent)
        .context("model file destination escapes the managed HF cache repo")
}

pub(crate) fn model_file_transfer_features() -> Vec<String> {
    vec![
        MODEL_TRANSFER_FEATURE_FILE_GET.to_string(),
        MODEL_TRANSFER_FEATURE_RANGE.to_string(),
        MODEL_TRANSFER_FEATURE_SHA256.to_string(),
    ]
}

pub(crate) fn file_sha256_hex(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).context("open model file for sha256")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .context("read model file for sha256")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn validate_hf_model_repo(repo: &str) -> Result<()> {
    let parts = repo.split('/').collect::<Vec<_>>();
    anyhow::ensure!(
        parts.len() == 2 && parts.iter().all(|part| safe_hf_component(part)),
        "HF model repo id must look like namespace/repo"
    );
    Ok(())
}

fn safe_hf_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && !component.contains('/')
        && !component.contains(':')
        && !component.contains('@')
        && !component.contains('\\')
}

fn validate_immutable_revision(revision: &str) -> Result<()> {
    anyhow::ensure!(!revision.trim().is_empty(), "HF model revision is empty");
    anyhow::ensure!(
        is_sha_revision(revision),
        "peer model transfer requires an immutable Hugging Face commit revision"
    );
    Ok(())
}

fn validate_hf_revision_ref(revision: &str) -> Result<()> {
    anyhow::ensure!(!revision.trim().is_empty(), "HF model revision is empty");
    anyhow::ensure!(
        is_sha_revision(revision) || safe_hf_component(revision),
        "HF model revision must be a safe branch, tag, or commit id"
    );
    Ok(())
}

fn is_sha_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_sha256(value: &str) -> Result<()> {
    anyhow::ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "sha256 must be 64 hex characters"
    );
    Ok(())
}

fn hf_repo_cache_root(repo: &str) -> PathBuf {
    crate::models::huggingface_hub_cache_dir().join(
        crate::models::local::huggingface_repo_folder_name(repo, hf_hub::RepoType::Model),
    )
}

fn ensure_path_inside_repo_root(repo_root: &Path, path: &Path) -> Result<()> {
    let repo_root = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    let path = path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", path.display()))?;
    anyhow::ensure!(
        path.starts_with(&repo_root),
        "{} escapes {}",
        path.display(),
        repo_root.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn restore_env(key: &str, previous: Option<std::ffi::OsString>) {
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    #[test]
    #[serial]
    fn model_transfer_enabled_defaults_on_and_honors_opt_out() {
        let previous = std::env::var_os("MESH_LLM_MODEL_TRANSFER");
        std::env::remove_var("MESH_LLM_MODEL_TRANSFER");
        assert!(model_transfer_enabled());

        for disabled in ["0", "false", "off", "no"] {
            std::env::set_var("MESH_LLM_MODEL_TRANSFER", disabled);
            assert!(
                !model_transfer_enabled(),
                "{disabled} should disable transfer"
            );
        }

        std::env::set_var("MESH_LLM_MODEL_TRANSFER", "true");
        assert!(model_transfer_enabled());
        restore_env("MESH_LLM_MODEL_TRANSFER", previous);
    }

    #[test]
    fn safe_relative_model_file_path_rejects_escape_paths() {
        assert_eq!(
            safe_relative_model_file_path("nested/model.gguf").unwrap(),
            std::path::PathBuf::from("nested/model.gguf")
        );

        for value in [
            "",
            ".",
            "..",
            "../model.gguf",
            "nested/../model.gguf",
            "/tmp/model.gguf",
            "nested\\model.gguf",
        ] {
            assert!(
                safe_relative_model_file_path(value).is_err(),
                "{value:?} must be rejected"
            );
        }
    }

    #[test]
    fn model_transfer_request_rejects_unsafe_revision() {
        let request = ModelFileTransferRequest {
            gen: MODEL_TRANSFER_GENERATION,
            requester_id: vec![7; 32],
            request_id: "request-a".to_string(),
            repo: "org/model".to_string(),
            revision: "../main".to_string(),
            file: "model.gguf".to_string(),
            offset: 0,
            expected_size: Some(5),
            expected_sha256: Some(sha256_hex(b"model")),
        };

        assert!(validate_model_file_transfer_request(&request).is_err());
    }

    #[test]
    #[serial]
    fn servable_model_file_requires_managed_hf_cache_file() {
        let previous = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", temp.path());
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let model_path = hf_model_cache_path("org/model", revision, "model.gguf").unwrap();
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"model").unwrap();

        let request = ModelFileTransferRequest {
            gen: MODEL_TRANSFER_GENERATION,
            requester_id: vec![7; 32],
            request_id: "request-a".to_string(),
            repo: "org/model".to_string(),
            revision: revision.to_string(),
            file: "model.gguf".to_string(),
            offset: 0,
            expected_size: Some(5),
            expected_sha256: Some(sha256_hex(b"model")),
        };

        let artifact = servable_model_file_from_request(&request).unwrap();
        assert_eq!(artifact.path, model_path);
        assert_eq!(artifact.resolved_revision, revision);
        assert_eq!(artifact.size, 5);
        assert_eq!(artifact.sha256, sha256_hex(b"model"));

        restore_env("HF_HUB_CACHE", previous);
    }

    #[test]
    #[serial]
    fn servable_model_file_resolves_mutable_ref_to_cached_commit() {
        let previous = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", temp.path());
        let revision = "0123456789abcdef0123456789abcdef01234567";
        install_hf_revision_ref("org/model", "main", revision).unwrap();
        let model_path = hf_model_cache_path("org/model", revision, "model.gguf").unwrap();
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"model").unwrap();

        let request = ModelFileTransferRequest {
            gen: MODEL_TRANSFER_GENERATION,
            requester_id: vec![7; 32],
            request_id: "request-a".to_string(),
            repo: "org/model".to_string(),
            revision: "main".to_string(),
            file: "model.gguf".to_string(),
            offset: 0,
            expected_size: Some(5),
            expected_sha256: Some(sha256_hex(b"model")),
        };

        let artifact = servable_model_file_from_request(&request).unwrap();
        assert_eq!(artifact.path, model_path);
        assert_eq!(artifact.resolved_revision, revision);

        restore_env("HF_HUB_CACHE", previous);
    }

    #[test]
    #[serial]
    fn cached_hf_model_file_returns_safe_existing_snapshot_file() {
        let previous = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", temp.path());
        let revision = "0123456789abcdef0123456789abcdef01234567";
        install_hf_revision_ref("org/model", "main", revision).unwrap();
        let model_path = hf_model_cache_path("org/model", revision, "model.gguf").unwrap();
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"model").unwrap();

        assert_eq!(
            cached_hf_model_file("org/model", "main", "model.gguf").unwrap(),
            Some(model_path)
        );

        restore_env("HF_HUB_CACHE", previous);
    }

    #[test]
    fn model_transfer_response_requires_commit_and_sha_when_accepted() {
        let valid = ModelFileTransferResponse {
            gen: MODEL_TRANSFER_GENERATION,
            accepted: true,
            resolved_revision: Some("0123456789abcdef0123456789abcdef01234567".to_string()),
            total_size: 5,
            sha256: Some(sha256_hex(b"model")),
            offset: 0,
            error: None,
        };
        validate_model_file_transfer_response(&valid, 0).unwrap();

        let mut missing_revision = valid.clone();
        missing_revision.resolved_revision = None;
        assert!(validate_model_file_transfer_response(&missing_revision, 0).is_err());

        let mut missing_sha = valid;
        missing_sha.sha256 = None;
        assert!(validate_model_file_transfer_response(&missing_sha, 0).is_err());
    }

    #[test]
    #[cfg(unix)]
    #[serial]
    fn servable_model_file_rejects_symlink_escape_from_hf_cache() {
        use std::os::unix::fs as unix_fs;

        let previous = std::env::var_os("HF_HUB_CACHE");
        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", temp.path());
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let model_path = hf_model_cache_path("org/model", revision, "model.gguf").unwrap();
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        let outside = temp.path().join("outside.gguf");
        std::fs::write(&outside, b"model").unwrap();
        unix_fs::symlink(&outside, &model_path).unwrap();

        let request = ModelFileTransferRequest {
            gen: MODEL_TRANSFER_GENERATION,
            requester_id: vec![7; 32],
            request_id: "request-a".to_string(),
            repo: "org/model".to_string(),
            revision: revision.to_string(),
            file: "model.gguf".to_string(),
            offset: 0,
            expected_size: Some(5),
            expected_sha256: Some(sha256_hex(b"model")),
        };

        assert!(servable_model_file_from_request(&request).is_err());

        restore_env("HF_HUB_CACHE", previous);
    }
}
