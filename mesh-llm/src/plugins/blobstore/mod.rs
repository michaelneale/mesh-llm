use anyhow::{Context, Result};
use base64::Engine;
use mesh_llm_plugin::{
    async_trait, json_response, parse_rpc_params, Plugin, PluginContext, PluginError, PluginResult,
    PluginRpcResult, PluginRuntime,
};
use rand::Rng;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::plugin::blobstore::{
    FinishRequestRequest, FinishRequestResponse, GetRequestObjectRequest, GetRequestObjectResponse,
    PutRequestObjectRequest, PutRequestObjectResponse, ABORT_REQUEST_METHOD,
    COMPLETE_REQUEST_METHOD, GET_REQUEST_OBJECT_METHOD, PUT_REQUEST_OBJECT_METHOD,
};

const DEFAULT_REQUEST_OBJECT_TTL_SECS: u64 = 15 * 60;
const DEFAULT_USES_REMAINING: u32 = 3;
/// Maximum decoded size for a single uploaded object (50 MiB).
const MAX_OBJECT_BYTES: usize = 50 * 1024 * 1024;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn default_blobstore_root() -> PathBuf {
    crate::models::local::mesh_llm_cache_dir().join("blobstore")
}

fn url_safe_base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn sanitize_id(value: &str, field: &str) -> PluginResult<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(PluginError::invalid_params(format!(
            "Missing required '{field}' value"
        )));
    }
    if !trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.')
    {
        return Err(PluginError::invalid_params(format!(
            "Invalid '{field}' value '{}'",
            value
        )));
    }
    Ok(trimmed.to_string())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RequestIndex {
    request_id: String,
    tokens: Vec<String>,
    created_at: u64,
    updated_at: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredObject {
    token: String,
    request_id: String,
    mime_type: String,
    file_name: Option<String>,
    sha256_hex: String,
    size_bytes: u64,
    created_at: u64,
    expires_at: u64,
    uses_remaining: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ReapStats {
    removed_tokens: usize,
    removed_bytes: u64,
}

impl ReapStats {
    fn merge(&mut self, other: Self) {
        self.removed_tokens += other.removed_tokens;
        self.removed_bytes += other.removed_bytes;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BlobStore {
    root: PathBuf,
}

impl BlobStore {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn tokens_dir(&self) -> PathBuf {
        self.root.join("tokens")
    }

    fn requests_dir(&self) -> PathBuf {
        self.root.join("requests")
    }

    fn object_path(&self, token: &str) -> PathBuf {
        self.objects_dir().join(format!("{token}.bin"))
    }

    fn token_path(&self, token: &str) -> PathBuf {
        self.tokens_dir().join(format!("{token}.json"))
    }

    fn request_path(&self, request_id: &str) -> PathBuf {
        self.requests_dir().join(format!("{request_id}.json"))
    }

    fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(self.objects_dir())
            .with_context(|| format!("Create {}", self.objects_dir().display()))?;
        std::fs::create_dir_all(self.tokens_dir())
            .with_context(|| format!("Create {}", self.tokens_dir().display()))?;
        std::fs::create_dir_all(self.requests_dir())
            .with_context(|| format!("Create {}", self.requests_dir().display()))?;
        Ok(())
    }

    fn reap_expired(&self) -> Result<ReapStats> {
        self.ensure_dirs()?;
        let now = now_secs();
        let mut stats = ReapStats::default();
        for entry in std::fs::read_dir(self.tokens_dir())
            .with_context(|| format!("Read {}", self.tokens_dir().display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(stored) = serde_json::from_str::<StoredObject>(&raw) else {
                continue;
            };
            if stored.expires_at > now {
                continue;
            }
            stats.merge(self.delete_token(&stored.token)?);
            let request_path = self.request_path(&stored.request_id);
            self.compact_request_index(&request_path)?;
        }
        Ok(stats)
    }

    pub(crate) fn put_request_object(
        &self,
        request: PutRequestObjectRequest,
    ) -> PluginResult<PutRequestObjectResponse> {
        self.ensure_dirs()
            .map_err(|err| PluginError::internal(err.to_string()))?;
        self.reap_expired()
            .map_err(|err| PluginError::internal(err.to_string()))?;

        let request_id = sanitize_id(&request.request_id, "request_id")?;
        let mime_type = request.mime_type.trim().to_string();
        if mime_type.is_empty() {
            return Err(PluginError::invalid_params(
                "Missing required 'mime_type' value",
            ));
        }

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(request.bytes_base64)
            .map_err(|err| PluginError::invalid_params(format!("Invalid bytes_base64: {err}")))?;
        if bytes.len() > MAX_OBJECT_BYTES {
            return Err(PluginError::invalid_params(format!(
                "Object too large: {} bytes exceeds the {} byte limit",
                bytes.len(),
                MAX_OBJECT_BYTES
            )));
        }
        let size_bytes = bytes.len() as u64;
        let created_at = now_secs();
        let expires_at = created_at
            + request
                .expires_in_secs
                .unwrap_or(DEFAULT_REQUEST_OBJECT_TTL_SECS);
        let uses_remaining = request
            .uses_remaining
            .unwrap_or(DEFAULT_USES_REMAINING)
            .max(1);

        let token = self.generate_token();
        let sha256_hex = hex::encode(Sha256::digest(&bytes));
        let stored = StoredObject {
            token: token.clone(),
            request_id: request_id.clone(),
            mime_type: mime_type.clone(),
            file_name: request.file_name.clone(),
            sha256_hex: sha256_hex.clone(),
            size_bytes,
            created_at,
            expires_at,
            uses_remaining,
        };

        let object_path = self.object_path(&token);
        self.write_atomic(&object_path, &bytes)
            .map_err(|err| PluginError::internal(err.to_string()))?;
        self.write_json(&self.token_path(&token), &stored)
            .map_err(|err| PluginError::internal(err.to_string()))?;
        self.add_request_token(&request_id, &token, created_at)
            .map_err(|err| PluginError::internal(err.to_string()))?;

        Ok(PutRequestObjectResponse {
            token,
            request_id,
            mime_type,
            file_name: request.file_name,
            size_bytes,
            sha256_hex,
            created_at,
            expires_at,
            uses_remaining,
        })
    }

    pub(crate) fn get_request_object(
        &self,
        request: GetRequestObjectRequest,
    ) -> PluginResult<GetRequestObjectResponse> {
        self.ensure_dirs()
            .map_err(|err| PluginError::internal(err.to_string()))?;
        self.reap_expired()
            .map_err(|err| PluginError::internal(err.to_string()))?;

        let token = sanitize_id(&request.token, "token")?;
        let token_path = self.token_path(&token);
        let mut stored = self
            .read_json::<StoredObject>(&token_path)
            .map_err(|_| PluginError::invalid_params("Unknown or expired blob token"))?;
        if let Some(expected_request_id) = request.request_id.as_deref() {
            let expected_request_id = sanitize_id(expected_request_id, "request_id")?;
            if stored.request_id != expected_request_id {
                return Err(PluginError::invalid_params(
                    "Blob token does not belong to the requested completion",
                ));
            }
        }
        if stored.uses_remaining == 0 {
            return Err(PluginError::invalid_params("Blob token is spent"));
        }

        let bytes = std::fs::read(self.object_path(&token))
            .with_context(|| format!("Read object for token {token}"))
            .map_err(|err| PluginError::internal(err.to_string()))?;
        stored.uses_remaining = stored.uses_remaining.saturating_sub(1);
        self.write_json(&token_path, &stored)
            .map_err(|err| PluginError::internal(err.to_string()))?;

        Ok(GetRequestObjectResponse {
            token: stored.token,
            request_id: stored.request_id,
            mime_type: stored.mime_type,
            file_name: stored.file_name,
            bytes_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
            size_bytes: stored.size_bytes,
            sha256_hex: stored.sha256_hex,
            created_at: stored.created_at,
            expires_at: stored.expires_at,
            uses_remaining: stored.uses_remaining,
        })
    }

    pub(crate) fn finish_request(&self, request_id: &str) -> PluginResult<FinishRequestResponse> {
        self.ensure_dirs()
            .map_err(|err| PluginError::internal(err.to_string()))?;
        self.reap_expired()
            .map_err(|err| PluginError::internal(err.to_string()))?;

        let request_id = sanitize_id(request_id, "request_id")?;
        let request_path = self.request_path(&request_id);
        let Ok(index) = self.read_json::<RequestIndex>(&request_path) else {
            return Ok(FinishRequestResponse {
                request_id,
                removed_tokens: 0,
                removed_bytes: 0,
            });
        };

        let mut stats = ReapStats::default();
        for token in index.tokens {
            stats.merge(
                self.delete_token(&token)
                    .map_err(|err| PluginError::internal(err.to_string()))?,
            );
        }
        let _ = std::fs::remove_file(&request_path);

        Ok(FinishRequestResponse {
            request_id,
            removed_tokens: stats.removed_tokens,
            removed_bytes: stats.removed_bytes,
        })
    }

    fn generate_token(&self) -> String {
        let mut bytes = [0u8; 24];
        rand::rng().fill(&mut bytes);
        format!("obj_{}", url_safe_base64(&bytes))
    }

    fn add_request_token(&self, request_id: &str, token: &str, created_at: u64) -> Result<()> {
        let path = self.request_path(request_id);
        let mut index = match self.read_json::<RequestIndex>(&path) {
            Ok(existing) => existing,
            Err(_) => RequestIndex {
                request_id: request_id.to_string(),
                tokens: Vec::new(),
                created_at,
                updated_at: created_at,
            },
        };
        if !index.tokens.iter().any(|existing| existing == token) {
            index.tokens.push(token.to_string());
        }
        index.updated_at = now_secs();
        self.write_json(&path, &index)
    }

    fn compact_request_index(&self, path: &Path) -> Result<()> {
        let mut index = match self.read_json::<RequestIndex>(path) {
            Ok(index) => index,
            Err(_) => return Ok(()),
        };
        index.tokens.retain(|token| {
            let has_metadata = self.token_path(token).exists();
            if !has_metadata {
                // Remove orphaned object file that reap_expired cannot discover
                if let Err(err) = std::fs::remove_file(self.object_path(token)) {
                    tracing::debug!("compact: could not remove orphaned blob for {token}: {err}");
                }
            }
            has_metadata
        });
        if index.tokens.is_empty() {
            let _ = std::fs::remove_file(path);
            return Ok(());
        }
        index.updated_at = now_secs();
        self.write_json(path, &index)
    }

    fn delete_token(&self, token: &str) -> Result<ReapStats> {
        let token_path = self.token_path(token);
        let metadata = self.read_json::<StoredObject>(&token_path).ok();
        let mut stats = ReapStats::default();
        if let Some(stored) = metadata {
            stats.removed_tokens = 1;
            stats.removed_bytes = stored.size_bytes;
        }
        let _ = std::fs::remove_file(token_path);
        let _ = std::fs::remove_file(self.object_path(token));
        Ok(stats)
    }

    fn read_json<T: for<'de> Deserialize<'de>>(&self, path: &Path) -> Result<T> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("Read {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("Parse {}", path.display()))
    }

    fn write_atomic(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        let tmp_path = path.with_extension(format!(
            "tmp-{}-{:016x}",
            std::process::id(),
            rand::rng().random::<u64>()
        ));
        std::fs::write(&tmp_path, bytes)
            .with_context(|| format!("Write staging file {}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, path).with_context(|| {
            let _ = std::fs::remove_file(&tmp_path);
            format!("Rename {} -> {}", tmp_path.display(), path.display())
        })
    }

    fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value).context("Serialize blobstore metadata")?;
        self.write_atomic(path, &bytes)
    }
}

struct BlobstorePlugin {
    plugin_id: String,
    plugin_version: String,
    server_info: ServerInfo,
    store: BlobStore,
}

impl BlobstorePlugin {
    fn new(plugin_id: String) -> Self {
        Self {
            plugin_id,
            plugin_version: crate::VERSION.to_string(),
            server_info: ServerInfo::new(ServerCapabilities::builder().build())
                .with_server_info(
                    Implementation::new("mesh-blobstore", crate::VERSION)
                        .with_title("Mesh Blobstore Plugin")
                        .with_description(
                            "Ingress-local request-scoped media object storage for multimodal requests.",
                        ),
                )
                .with_instructions(
                    "Provides internal request object storage for the mesh-llm host. Not intended for direct user tools.",
                ),
            store: BlobStore::new(default_blobstore_root()),
        }
    }
}

#[async_trait]
impl Plugin for BlobstorePlugin {
    fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    fn plugin_version(&self) -> String {
        self.plugin_version.clone()
    }

    fn server_info(&self) -> ServerInfo {
        self.server_info.clone()
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["internal:blobstore".into()]
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        self.store.reap_expired()?;
        let token_count = std::fs::read_dir(self.store.tokens_dir())
            .map(|entries| entries.count())
            .unwrap_or(0);
        Ok(format!(
            "root={} tokens={}",
            self.store.root.display(),
            token_count
        ))
    }

    async fn handle_rpc(
        &mut self,
        request: mesh_llm_plugin::proto::RpcRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginRpcResult {
        match request.method.as_str() {
            PUT_REQUEST_OBJECT_METHOD => {
                let params: PutRequestObjectRequest = parse_rpc_params(&request)?;
                json_response(&self.store.put_request_object(params)?)
            }
            GET_REQUEST_OBJECT_METHOD => {
                let params: GetRequestObjectRequest = parse_rpc_params(&request)?;
                json_response(&self.store.get_request_object(params)?)
            }
            COMPLETE_REQUEST_METHOD | ABORT_REQUEST_METHOD => {
                let params: FinishRequestRequest = parse_rpc_params(&request)?;
                json_response(&self.store.finish_request(&params.request_id)?)
            }
            _ => Err(PluginError::method_not_found(format!(
                "Unsupported blobstore RPC '{}'",
                request.method
            ))),
        }
    }
}

pub(crate) async fn run_plugin(name: String) -> anyhow::Result<()> {
    PluginRuntime::run(BlobstorePlugin::new(name)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_blobstore_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mesh-llm-blobstore-{name}-{}",
            rand::random::<u64>()
        ))
    }

    #[test]
    fn put_get_complete_roundtrip() {
        let root = temp_blobstore_root("roundtrip");
        let store = BlobStore::new(root.clone());
        let response = store
            .put_request_object(PutRequestObjectRequest {
                request_id: "req_123".into(),
                mime_type: "audio/wav".into(),
                file_name: Some("clip.wav".into()),
                bytes_base64: base64::engine::general_purpose::STANDARD.encode(b"hello world"),
                expires_in_secs: Some(60),
                uses_remaining: Some(2),
            })
            .unwrap();

        assert!(response.token.starts_with("obj_"));
        let first_get = store
            .get_request_object(GetRequestObjectRequest {
                token: response.token.clone(),
                request_id: Some("req_123".into()),
            })
            .unwrap();
        assert_eq!(first_get.mime_type, "audio/wav");
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(first_get.bytes_base64)
                .unwrap(),
            b"hello world"
        );
        assert_eq!(first_get.uses_remaining, 1);

        let finished = store.finish_request("req_123").unwrap();
        assert_eq!(finished.removed_tokens, 1);
        assert_eq!(finished.removed_bytes, 11);
        assert!(!store.token_path(&response.token).exists());
        assert!(!store.object_path(&response.token).exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn get_rejects_mismatched_request_id() {
        let root = temp_blobstore_root("request-check");
        let store = BlobStore::new(root.clone());
        let response = store
            .put_request_object(PutRequestObjectRequest {
                request_id: "req_abc".into(),
                mime_type: "image/png".into(),
                file_name: None,
                bytes_base64: base64::engine::general_purpose::STANDARD.encode(b"png"),
                expires_in_secs: Some(60),
                uses_remaining: Some(1),
            })
            .unwrap();

        let error = store
            .get_request_object(GetRequestObjectRequest {
                token: response.token,
                request_id: Some("req_other".into()),
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("Blob token does not belong to the requested completion"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn expired_tokens_are_reaped() {
        let root = temp_blobstore_root("reap");
        let store = BlobStore::new(root.clone());
        let response = store
            .put_request_object(PutRequestObjectRequest {
                request_id: "req_expired".into(),
                mime_type: "application/octet-stream".into(),
                file_name: None,
                bytes_base64: base64::engine::general_purpose::STANDARD.encode(b"bye"),
                expires_in_secs: Some(60),
                uses_remaining: Some(1),
            })
            .unwrap();

        let path = store.token_path(&response.token);
        let mut stored = store.read_json::<StoredObject>(&path).unwrap();
        stored.expires_at = 0;
        store.write_json(&path, &stored).unwrap();

        let stats = store.reap_expired().unwrap();
        assert_eq!(stats.removed_tokens, 1);
        assert_eq!(stats.removed_bytes, 3);
        assert!(!path.exists());
        assert!(!store.object_path(&response.token).exists());
        assert!(!store.request_path("req_expired").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
