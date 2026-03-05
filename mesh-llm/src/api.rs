//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state (JSON)
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   POST /api/chat      — proxy to inference API
//!   GET  /              — embedded web dashboard
//!
//! The dashboard is read-only — shows status, topology, models.
//! All mutations happen via CLI flags (--join, --model, --auto).

use crate::{download, election, mesh, nostr};
use base64::Engine as _;
use ed25519_dalek::{Signer, SigningKey};
use include_dir::{include_dir, Dir};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static CONSOLE_DIST: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");
const MESH_LLM_VERSION: &str = crate::VERSION;
const HUB_MEMBERSHIP_SYNC_INTERVAL_SECS: u64 = 60;
const HUB_MEMBERSHIP_TTL_SECS: u64 = 15 * 60;
const HUB_OUTAGE_GRACE_SECS: u64 = 10 * 60;
const HUB_RUNTIME_SESSION_TTL_SECS: u64 = 15 * 60;
const HUB_RUNTIME_SESSION_REFRESH_BEFORE_SECS: u64 = 5 * 60;
const HUB_TELEMETRY_SYNC_INTERVAL_SECS: u64 = 30;

// ── Shared state ──

/// Shared live state — written by the main process, read by API handlers.
#[derive(Clone)]
pub struct MeshApi {
    inner: Arc<Mutex<ApiInner>>,
}

struct ApiInner {
    node: mesh::Node,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    model_name: String,
    draft_name: Option<String>,
    api_port: u16,
    model_size_bytes: u64,
    mesh_name: Option<String>,
    latest_version: Option<String>,
    nostr_relays: Vec<String>,
    hub: HubState,
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
}

#[derive(Clone, Default)]
struct HubState {
    base_url: String,
    access_token: Option<String>,
    auth_pending: bool,
    profile: Option<HubProfile>,
    default_mesh_selector: Option<String>,
    default_invite_token: Option<String>,
    node_id: Option<String>,
    linked_mesh_id: Option<String>,
    link_state: String,
    membership_enforcement: String,
    last_membership_sync_ok_unix: Option<u64>,
    runtime_access_token: Option<String>,
    runtime_access_token_expires_unix: Option<u64>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct HubProfile {
    name: Option<String>,
    handle: Option<String>,
    avatar_url: Option<String>,
}

#[derive(Serialize, Deserialize, Default)]
struct HubSessionFile {
    access_token: Option<String>,
    profile: Option<HubProfile>,
    default_mesh_selector: Option<String>,
    default_invite_token: Option<String>,
    node_id: Option<String>,
    linked_mesh_id: Option<String>,
    link_state: Option<String>,
    membership_enforcement: Option<String>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct HubIdentityFile {
    signing_key_b64: String,
    public_key_b64: String,
}

struct HubTelemetrySnapshot {
    node_id: String,
    hub_mesh_id: String,
    active_nodes: u64,
    total_vram_gb: f64,
    node_total_vram_gb: f64,
    node_available_vram_gb: f64,
    node_gpu_count: u64,
    node_warm_models: Vec<String>,
    mesh_warm_models: Vec<String>,
}

#[derive(Serialize)]
struct StatusPayload {
    version: String,
    latest_version: Option<String>,
    node_id: String,
    token: String,
    node_status: String,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    model_name: String,
    draft_name: Option<String>,
    api_port: u16,
    my_vram_gb: f64,
    model_size_gb: f64,
    peers: Vec<PeerPayload>,
    launch_pi: Option<String>,
    launch_goose: Option<String>,
    mesh_models: Vec<MeshModelPayload>,
    inflight_requests: u64,
    /// Mesh identity (for matching against discovered meshes)
    mesh_id: Option<String>,
    /// Human-readable mesh name (from Nostr publishing)
    mesh_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_auth_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_user_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_user_handle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_user_avatar_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_profile_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_meshes_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_mesh_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh_link_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    membership_enforcement: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_default_mesh_selector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hub_default_invite_configured: Option<bool>,
}

#[derive(Serialize)]
struct PeerPayload {
    id: String,
    role: String,
    models: Vec<String>,
    vram_gb: f64,
    serving: Option<String>,
    rtt_ms: Option<u32>,
}

#[derive(Serialize)]
struct MeshModelPayload {
    name: String,
    status: String,
    node_count: usize,
    size_gb: f64,
    /// Total requests seen across the mesh (from demand map)
    #[serde(skip_serializing_if = "Option::is_none")]
    request_count: Option<u64>,
    /// Seconds since last request or declaration (None if no demand data)
    #[serde(skip_serializing_if = "Option::is_none")]
    last_active_secs_ago: Option<u64>,
}

fn hub_session_path() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".mesh-llm").join("hub-session.json"))
}

fn hub_identity_path() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".mesh-llm").join("hub-identity.json"))
}

fn load_hub_session() -> Option<HubSessionFile> {
    let path = hub_session_path()?;
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<HubSessionFile>(&data).ok()
}

fn save_hub_session(session: &HubSessionFile) -> anyhow::Result<()> {
    let Some(path) = hub_session_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(session)?;
    std::fs::write(path, payload)?;
    Ok(())
}

fn load_hub_identity() -> Option<HubIdentityFile> {
    let path = hub_identity_path()?;
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<HubIdentityFile>(&data).ok()
}

fn save_hub_identity(identity: &HubIdentityFile) -> anyhow::Result<()> {
    let Some(path) = hub_identity_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(identity)?;
    std::fs::write(path, payload)?;
    Ok(())
}

fn load_or_create_hub_identity() -> anyhow::Result<HubIdentityFile> {
    if let Some(existing) = load_hub_identity() {
        if !existing.signing_key_b64.trim().is_empty() && !existing.public_key_b64.trim().is_empty()
        {
            return Ok(existing);
        }
    }
    let sk: [u8; 32] = rand::random();
    let signing_key = SigningKey::from_bytes(&sk);
    let verifying_key = signing_key.verifying_key();
    let identity = HubIdentityFile {
        signing_key_b64: base64::engine::general_purpose::STANDARD.encode(signing_key.to_bytes()),
        public_key_b64: base64::engine::general_purpose::STANDARD.encode(verifying_key.to_bytes()),
    };
    save_hub_identity(&identity)?;
    Ok(identity)
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn default_node_display_name() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "mesh-node".to_string())
}

fn sign_hub_message(identity: &HubIdentityFile, message: &str) -> anyhow::Result<String> {
    let secret_bytes = base64::engine::general_purpose::STANDARD
        .decode(identity.signing_key_b64.as_bytes())
        .map_err(|e| anyhow::anyhow!("invalid stored signing key: {e}"))?;
    let signing_key_bytes: [u8; 32] = secret_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("stored signing key has invalid length"))?;
    let signing_key = SigningKey::from_bytes(&signing_key_bytes);
    let signature = signing_key.sign(message.as_bytes());
    Ok(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()))
}

impl MeshApi {
    fn persist_hub_state(inner: &ApiInner) {
        let _ = save_hub_session(&HubSessionFile {
            access_token: inner.hub.access_token.clone(),
            profile: inner.hub.profile.clone(),
            default_mesh_selector: inner.hub.default_mesh_selector.clone(),
            default_invite_token: inner.hub.default_invite_token.clone(),
            node_id: inner.hub.node_id.clone(),
            linked_mesh_id: inner.hub.linked_mesh_id.clone(),
            link_state: Some(inner.hub.link_state.clone()),
            membership_enforcement: Some(inner.hub.membership_enforcement.clone()),
        });
    }

    pub fn new(node: mesh::Node, model_name: String, api_port: u16, model_size_bytes: u64) -> Self {
        let hub_base_url = std::env::var("MESH_LLM_HUB_BASE_URL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "https://www.inferencehub.cc".to_string())
            .trim_end_matches('/')
            .to_string();
        let session = load_hub_session().unwrap_or_default();
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                is_host: false,
                is_client: false,
                llama_ready: false,
                llama_port: None,
                model_name,
                draft_name: None,
                api_port,
                model_size_bytes,
                mesh_name: None,
                latest_version: None,
                nostr_relays: nostr::DEFAULT_RELAYS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                hub: HubState {
                    base_url: hub_base_url,
                    access_token: session.access_token,
                    auth_pending: false,
                    profile: session.profile,
                    default_mesh_selector: session.default_mesh_selector,
                    default_invite_token: session.default_invite_token,
                    node_id: session.node_id,
                    linked_mesh_id: session.linked_mesh_id,
                    link_state: session.link_state.unwrap_or_else(|| "unlinked".to_string()),
                    membership_enforcement: session
                        .membership_enforcement
                        .unwrap_or_else(|| "local".to_string()),
                    last_membership_sync_ok_unix: None,
                    runtime_access_token: None,
                    runtime_access_token_expires_unix: None,
                },
                sse_clients: Vec::new(),
            })),
        }
    }

    pub async fn set_draft_name(&self, name: String) {
        self.inner.lock().await.draft_name = Some(name);
    }

    pub async fn set_client(&self, is_client: bool) {
        self.inner.lock().await.is_client = is_client;
    }

    pub async fn set_mesh_name(&self, name: String) {
        self.inner.lock().await.mesh_name = Some(name);
    }

    pub async fn set_nostr_relays(&self, relays: Vec<String>) {
        self.inner.lock().await.nostr_relays = relays;
    }

    pub async fn update(&self, is_host: bool, llama_ready: bool) {
        {
            let mut inner = self.inner.lock().await;
            inner.is_host = is_host;
            inner.llama_ready = llama_ready;
        }
        self.push_status().await;
    }

    pub async fn set_llama_port(&self, port: Option<u16>) {
        self.inner.lock().await.llama_port = port;
    }

    async fn hub_base_url(&self) -> String {
        self.inner.lock().await.hub.base_url.clone()
    }

    async fn hub_access_token(&self) -> Option<String> {
        self.inner.lock().await.hub.access_token.clone()
    }

    async fn set_hub_auth_pending(&self, pending: bool) {
        self.inner.lock().await.hub.auth_pending = pending;
    }

    async fn set_hub_access_token(&self, token: Option<String>) {
        let mut inner = self.inner.lock().await;
        inner.hub.access_token = token;
        if inner.hub.access_token.is_none() {
            inner.hub.profile = None;
            inner.hub.last_membership_sync_ok_unix = None;
            inner.hub.runtime_access_token = None;
            inner.hub.runtime_access_token_expires_unix = None;
        }
        Self::persist_hub_state(&inner);
    }

    async fn set_hub_profile(&self, profile: Option<HubProfile>) {
        let mut inner = self.inner.lock().await;
        inner.hub.profile = profile;
        Self::persist_hub_state(&inner);
    }

    async fn set_hub_node_id(&self, node_id: Option<String>) {
        let (node, mesh_id) = {
            let mut inner = self.inner.lock().await;
            inner.hub.node_id = node_id.clone();
            if inner.hub.node_id.is_none() {
                inner.hub.runtime_access_token = None;
                inner.hub.runtime_access_token_expires_unix = None;
            }
            let mesh_id = inner.hub.linked_mesh_id.clone();
            Self::persist_hub_state(&inner);
            (inner.node.clone(), mesh_id)
        };
        node.set_hub_identity(node_id, mesh_id).await;
    }

    async fn set_hub_link_mode(
        &self,
        link_state: &str,
        membership_enforcement: &str,
        linked_mesh_id: Option<String>,
    ) {
        let (node, hub_node_id, hub_mesh_id) = {
            let mut inner = self.inner.lock().await;
            inner.hub.link_state = link_state.to_string();
            inner.hub.membership_enforcement = membership_enforcement.to_string();
            inner.hub.linked_mesh_id = linked_mesh_id;
            if inner.hub.membership_enforcement != "hub_enforced"
                || inner.hub.linked_mesh_id.is_none()
            {
                inner.hub.runtime_access_token = None;
                inner.hub.runtime_access_token_expires_unix = None;
            }
            inner.hub.last_membership_sync_ok_unix = if membership_enforcement == "hub_enforced" {
                Some(now_unix_secs())
            } else {
                None
            };
            Self::persist_hub_state(&inner);
            (
                inner.node.clone(),
                inner.hub.node_id.clone(),
                inner.hub.linked_mesh_id.clone(),
            )
        };
        node.set_hub_identity(hub_node_id, hub_mesh_id).await;
    }

    async fn set_hub_default_target(
        &self,
        default_mesh_selector: Option<String>,
        default_invite_token: Option<String>,
    ) {
        let mut inner = self.inner.lock().await;
        inner.hub.default_mesh_selector = default_mesh_selector;
        inner.hub.default_invite_token = default_invite_token;
        Self::persist_hub_state(&inner);
    }

    async fn set_runtime_access_token(&self, token: Option<String>, expires_unix: Option<u64>) {
        let mut inner = self.inner.lock().await;
        inner.hub.runtime_access_token = token;
        inner.hub.runtime_access_token_expires_unix = expires_unix;
    }

    async fn is_hub_enforced_mode(&self) -> bool {
        let inner = self.inner.lock().await;
        inner.hub.membership_enforcement == "hub_enforced" && inner.hub.linked_mesh_id.is_some()
    }

    async fn hub_fetch_profile(&self) -> anyhow::Result<Option<HubProfile>> {
        let base_url = self.hub_base_url().await;
        let Some(token) = self.hub_access_token().await else {
            return Ok(None);
        };
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{base_url}/api/profile"))
            .bearer_auth(token)
            .send()
            .await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            self.set_hub_access_token(None).await;
            self.set_hub_auth_pending(false).await;
            self.push_status().await;
            return Ok(None);
        }
        if !response.status().is_success() {
            anyhow::bail!("hub profile request failed with {}", response.status());
        }
        let payload = response.json::<serde_json::Value>().await?;
        let name = payload
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let email = payload
            .get("email")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let handle = email
            .as_deref()
            .map(|e| e.split('@').next().unwrap_or_default().trim().to_string())
            .filter(|s| !s.is_empty());
        let avatar_url = payload
            .get("image")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        Ok(Some(HubProfile {
            name,
            handle,
            avatar_url,
        }))
    }

    async fn hub_registered_node_id(&self) -> Option<String> {
        self.inner.lock().await.hub.node_id.clone()
    }

    async fn hub_ensure_registered_node(&self) -> anyhow::Result<String> {
        if let Some(existing) = self.hub_registered_node_id().await {
            return Ok(existing);
        }
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let identity = load_or_create_hub_identity()?;
        let client = reqwest::Client::new();

        let challenge_resp = client
            .post(format!("{base_url}/api/v0/nodes/register"))
            .bearer_auth(&token)
            .json(&serde_json::json!({ "public_key": identity.public_key_b64 }))
            .send()
            .await?;
        if !challenge_resp.status().is_success() {
            anyhow::bail!(
                "node register challenge failed with {}",
                challenge_resp.status()
            );
        }
        let challenge_payload = challenge_resp.json::<serde_json::Value>().await?;
        let challenge_id = challenge_payload
            .get("challenge_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let nonce = challenge_payload
            .get("challenge_nonce")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if challenge_id.is_empty() || nonce.is_empty() {
            anyhow::bail!("node register challenge payload missing required fields");
        }

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let signed_message = format!(
            "inferencehub-v1\npurpose=node_register\nchallenge_id={challenge_id}\nnonce={nonce}\nnode_id=\nts={ts}"
        );
        let signature_b64 = sign_hub_message(&identity, &signed_message)?;
        let hostname = default_node_display_name();
        let complete_resp = client
            .post(format!("{base_url}/api/v0/nodes/register/complete"))
            .bearer_auth(&token)
            .json(&serde_json::json!({
                "challenge_id": challenge_id,
                "public_key": identity.public_key_b64,
                "signature": signature_b64,
                "signed_message": signed_message,
                "display_name": hostname,
                "client": {
                    "version": crate::VERSION,
                    "platform": std::env::consts::OS,
                }
            }))
            .send()
            .await?;
        if !complete_resp.status().is_success() {
            anyhow::bail!(
                "node register complete failed with {}",
                complete_resp.status()
            );
        }
        let complete_payload = complete_resp.json::<serde_json::Value>().await?;
        let node_id = complete_payload
            .get("node_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if node_id.is_empty() {
            anyhow::bail!("node register response missing node_id");
        }
        self.set_hub_node_id(Some(node_id.clone())).await;
        Ok(node_id)
    }

    async fn hub_fetch_mesh_node_ids(&self, hub_mesh_id: &str) -> anyhow::Result<HashSet<String>> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{base_url}/api/v0/meshes/{hub_mesh_id}/nodes"))
            .bearer_auth(token)
            .send()
            .await?;
        if !response.status().is_success() {
            anyhow::bail!("mesh nodes fetch failed with {}", response.status());
        }
        let payload = response.json::<serde_json::Value>().await?;
        let ids = payload
            .get("items")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        item.get("node_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect::<HashSet<String>>()
            })
            .unwrap_or_default();
        Ok(ids)
    }

    async fn hub_resolve_mesh_id_by_selector(&self, selector: &str) -> anyhow::Result<String> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let selector = selector.trim();
        if selector.is_empty() {
            anyhow::bail!("empty mesh selector");
        }
        let client = reqwest::Client::new();

        // First try as mesh ID.
        let by_id_resp = client
            .get(format!("{base_url}/api/v0/meshes/{selector}"))
            .bearer_auth(&token)
            .send()
            .await?;
        if by_id_resp.status().is_success() {
            let payload = by_id_resp.json::<serde_json::Value>().await?;
            let id = payload
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or_default()
                .to_string();
            if !id.is_empty() {
                return Ok(id);
            }
        }

        let list_resp = client
            .get(format!("{base_url}/api/v0/meshes"))
            .bearer_auth(token)
            .send()
            .await?;
        if !list_resp.status().is_success() {
            anyhow::bail!("mesh list request failed with {}", list_resp.status());
        }
        let payload = list_resp.json::<serde_json::Value>().await?;
        let Some(items) = payload.as_array() else {
            anyhow::bail!("mesh list response format was invalid");
        };

        let selector_lower = selector.to_lowercase();
        let mut matches: Vec<String> = items
            .iter()
            .filter_map(|item| {
                let id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|v| !v.is_empty())?;
                let slug = item
                    .get("slug")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                if id == selector
                    || slug
                        .as_deref()
                        .map(|s| s.eq_ignore_ascii_case(selector))
                        .unwrap_or(false)
                    || name
                        .as_deref()
                        .map(|s| s.to_lowercase() == selector_lower)
                        .unwrap_or(false)
                {
                    Some(id.to_string())
                } else {
                    None
                }
            })
            .collect();

        matches.sort();
        matches.dedup();
        match matches.as_slice() {
            [single] => Ok(single.clone()),
            [] => anyhow::bail!("no InferenceHub mesh matched '{selector}'"),
            _ => anyhow::bail!("multiple meshes matched '{selector}'. Use an exact mesh ID."),
        }
    }

    async fn hub_redeem_invite_for_mesh_id(&self, invite_token: &str) -> anyhow::Result<String> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{base_url}/api/v0/invites/redeem"))
            .bearer_auth(token)
            .json(&serde_json::json!({
                "invite_token": invite_token,
            }))
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let payload = response.text().await.unwrap_or_default();
            anyhow::bail!("invite redeem failed with {status}: {payload}");
        }
        let payload = response.json::<serde_json::Value>().await?;
        let hub_mesh_id = payload
            .get("hub_mesh_id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or_default()
            .to_string();
        if hub_mesh_id.is_empty() {
            anyhow::bail!("invite redeem response missing hub_mesh_id");
        }
        Ok(hub_mesh_id)
    }

    async fn hub_resolve_target_mesh_id(
        &self,
        requested_hub_mesh_id: Option<String>,
    ) -> anyhow::Result<String> {
        if let Some(id) = requested_hub_mesh_id
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            return Ok(id);
        }

        let (default_mesh_selector, default_invite_token, linked_mesh_id) = {
            let inner = self.inner.lock().await;
            (
                inner.hub.default_mesh_selector.clone(),
                inner.hub.default_invite_token.clone(),
                inner.hub.linked_mesh_id.clone(),
            )
        };

        if let Some(invite) = default_invite_token.as_deref() {
            let hub_mesh_id = self.hub_redeem_invite_for_mesh_id(invite).await?;
            self.set_hub_default_target(Some(hub_mesh_id.clone()), None)
                .await;
            return Ok(hub_mesh_id);
        }

        if let Some(selector) = default_mesh_selector.as_deref() {
            return self.hub_resolve_mesh_id_by_selector(selector).await;
        }

        if let Some(linked) = linked_mesh_id {
            return Ok(linked);
        }

        anyhow::bail!(
            "hub_mesh_id is required (or pass --hub-mesh / --hub-invite when starting mesh-llm)"
        )
    }

    async fn hub_telemetry_snapshot(
        &self,
        hub_mesh_id: &str,
        node_id: &str,
    ) -> anyhow::Result<HubTelemetrySnapshot> {
        let node = self.inner.lock().await.node.clone();
        let peers = node.peers().await;
        let linked_peers: Vec<_> = peers
            .into_iter()
            .filter(|peer| peer.hub_mesh_id.as_deref() == Some(hub_mesh_id))
            .collect();

        let node_total_vram_gb = node.vram_bytes() as f64 / 1e9;
        let peer_total_vram_gb = linked_peers
            .iter()
            .map(|peer| peer.vram_bytes as f64 / 1e9)
            .sum::<f64>();
        let node_warm_models = node
            .serving()
            .await
            .map(|name| vec![name])
            .unwrap_or_default();

        Ok(HubTelemetrySnapshot {
            node_id: node_id.to_string(),
            hub_mesh_id: hub_mesh_id.to_string(),
            active_nodes: 1 + linked_peers.len() as u64,
            total_vram_gb: node_total_vram_gb + peer_total_vram_gb,
            node_total_vram_gb,
            node_available_vram_gb: node_total_vram_gb,
            node_gpu_count: if node_total_vram_gb > 0.0 { 1 } else { 0 },
            node_warm_models,
            mesh_warm_models: node.models_being_served().await,
        })
    }

    async fn hub_ensure_runtime_access_token(&self, node_id: &str) -> anyhow::Result<String> {
        let now = now_unix_secs();
        {
            let inner = self.inner.lock().await;
            if let (Some(token), Some(expires_unix)) = (
                inner.hub.runtime_access_token.clone(),
                inner.hub.runtime_access_token_expires_unix,
            ) {
                if expires_unix > now + HUB_RUNTIME_SESSION_REFRESH_BEFORE_SECS {
                    return Ok(token);
                }
            }
        }

        let base_url = self.hub_base_url().await;
        let identity = load_or_create_hub_identity()?;
        let client = reqwest::Client::new();
        let challenge_resp = client
            .post(format!(
                "{base_url}/api/v0/nodes/{node_id}/session/challenge"
            ))
            .send()
            .await?;
        if !challenge_resp.status().is_success() {
            let status = challenge_resp.status();
            let body = challenge_resp.text().await.unwrap_or_default();
            anyhow::bail!("runtime challenge failed with {status}: {body}");
        }
        let challenge_payload = challenge_resp.json::<serde_json::Value>().await?;
        let challenge_id = challenge_payload
            .get("challenge_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let nonce = challenge_payload
            .get("challenge_nonce")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if challenge_id.is_empty() || nonce.is_empty() {
            anyhow::bail!("runtime challenge payload missing required fields");
        }

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let signed_message = format!(
            "inferencehub-v1\npurpose=node_session\nchallenge_id={challenge_id}\nnonce={nonce}\nnode_id={node_id}\nts={ts}"
        );
        let signature_b64 = sign_hub_message(&identity, &signed_message)?;

        let session_resp = client
            .post(format!("{base_url}/api/v0/nodes/{node_id}/session"))
            .json(&serde_json::json!({
                "challenge_id": challenge_id,
                "signature": signature_b64,
                "signed_message": signed_message,
            }))
            .send()
            .await?;
        if !session_resp.status().is_success() {
            let status = session_resp.status();
            let body = session_resp.text().await.unwrap_or_default();
            anyhow::bail!("runtime session exchange failed with {status}: {body}");
        }
        let session_payload = session_resp.json::<serde_json::Value>().await?;
        let access_token = session_payload
            .get("access_token")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if access_token.is_empty() {
            anyhow::bail!("runtime session response missing access_token");
        }
        self.set_runtime_access_token(
            Some(access_token.clone()),
            Some(now + HUB_RUNTIME_SESSION_TTL_SECS),
        )
        .await;
        Ok(access_token)
    }

    async fn hub_post_runtime_json(
        &self,
        node_id: &str,
        path: &str,
        body: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let base_url = self.hub_base_url().await;
        let url = format!("{base_url}{path}");
        let client = reqwest::Client::new();

        for attempt in 0..2u8 {
            let token = self.hub_ensure_runtime_access_token(node_id).await?;
            let response = client
                .post(&url)
                .bearer_auth(&token)
                .json(body)
                .send()
                .await?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED {
                self.set_runtime_access_token(None, None).await;
                if attempt == 0 {
                    continue;
                }
            }
            if !response.status().is_success() {
                let status = response.status();
                let payload = response.text().await.unwrap_or_default();
                anyhow::bail!("hub telemetry post {path} failed with {status}: {payload}");
            }
            return Ok(());
        }

        anyhow::bail!("hub telemetry auth failed for {path}")
    }

    async fn publish_hub_telemetry_once(&self) -> anyhow::Result<()> {
        let (node_id, hub_mesh_id, should_publish) = {
            let inner = self.inner.lock().await;
            (
                inner.hub.node_id.clone(),
                inner.hub.linked_mesh_id.clone(),
                inner.hub.membership_enforcement == "hub_enforced",
            )
        };
        if !should_publish {
            return Ok(());
        }
        let Some(node_id) = node_id else {
            return Ok(());
        };
        let Some(hub_mesh_id) = hub_mesh_id else {
            return Ok(());
        };

        let telemetry = self.hub_telemetry_snapshot(&hub_mesh_id, &node_id).await?;
        let hostname = default_node_display_name();

        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/heartbeat"),
            &serde_json::json!({}),
        )
        .await?;
        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/capabilities"),
            &serde_json::json!({
                "capabilities": {
                    "total_vram_gb": telemetry.node_total_vram_gb,
                    "available_vram_gb": telemetry.node_available_vram_gb,
                    "gpu_count": telemetry.node_gpu_count,
                    "hostname": hostname,
                    "platform": std::env::consts::OS,
                    "client_version": crate::VERSION,
                }
            }),
        )
        .await?;
        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/metrics"),
            &serde_json::json!({
                "hub_mesh_id": telemetry.hub_mesh_id,
                "metrics": {
                    "node_total_vram_gb": telemetry.node_total_vram_gb,
                    "node_warm_model_count": telemetry.node_warm_models.len(),
                    "active_nodes": telemetry.active_nodes,
                    "total_vram_gb": telemetry.total_vram_gb,
                    "warm_model_count": telemetry.mesh_warm_models.len(),
                }
            }),
        )
        .await?;
        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/models"),
            &serde_json::json!({
                "hub_mesh_id": telemetry.hub_mesh_id,
                "warm_models": telemetry.node_warm_models,
            }),
        )
        .await?;

        let otlp_payload = serde_json::json!({
            "resourceMetrics": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "mesh-llm"}},
                        {"key": "service.version", "value": {"stringValue": MESH_LLM_VERSION}},
                        {"key": "host.name", "value": {"stringValue": hostname}},
                        {"key": "inferencehub.node_id", "value": {"stringValue": telemetry.node_id}},
                        {"key": "inferencehub.mesh_id", "value": {"stringValue": telemetry.hub_mesh_id}},
                    ]
                },
                "scopeMetrics": [{
                    "metrics": [
                        {
                            "name": "inferencehub.node.total_vram_gb",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_total_vram_gb }] }
                        },
                        {
                            "name": "inferencehub.node.available_vram_gb",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_available_vram_gb }] }
                        },
                        {
                            "name": "inferencehub.node.gpu_count",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_gpu_count }] }
                        },
                        {
                            "name": "inferencehub.node.warm_model_count",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_warm_models.len() }] }
                        },
                        {
                            "name": "inferencehub.node.heartbeat",
                            "gauge": { "dataPoints": [{ "asDouble": 1 }] }
                        },
                        {
                            "name": "inferencehub.mesh.active_nodes",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.active_nodes }] }
                        },
                        {
                            "name": "inferencehub.mesh.total_vram_gb",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.total_vram_gb }] }
                        },
                        {
                            "name": "inferencehub.mesh.warm_model_count",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.mesh_warm_models.len() }] }
                        }
                    ]
                }]
            }]
        });
        self.hub_post_runtime_json(&node_id, "/api/v0/oltp/v1/metrics", &otlp_payload)
            .await?;

        Ok(())
    }

    async fn attach_node_to_hub_mesh(
        &self,
        hub_mesh_id: &str,
        node_id: &str,
    ) -> anyhow::Result<()> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let node = self.inner.lock().await.node.clone();
        let local_mesh_id = node.mesh_id().await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!(
                "{base_url}/api/v0/meshes/{hub_mesh_id}/nodes/attach"
            ))
            .bearer_auth(token)
            .json(&serde_json::json!({
                "node_id": node_id,
                "local_mesh_id": local_mesh_id,
                "local_mesh_label": std::env::var("HOSTNAME").ok(),
            }))
            .send()
            .await?;
        if !response.status().is_success() {
            anyhow::bail!("node attach failed with {}", response.status());
        }
        Ok(())
    }

    async fn compute_link_preflight(&self, hub_mesh_id: &str) -> anyhow::Result<serde_json::Value> {
        let node = self.inner.lock().await.node.clone();
        let hub_member_nodes = self.hub_fetch_mesh_node_ids(hub_mesh_id).await?;
        let peers = node.peers().await;
        let mut peer_verified = 0usize;
        let mut peer_unverified = 0usize;
        let mut verified_host_capable_count = 0usize;
        let mut drop_ids = Vec::new();

        for peer in &peers {
            let in_mesh = peer.hub_mesh_id.as_deref() == Some(hub_mesh_id);
            let member_ok = peer
                .hub_node_id
                .as_ref()
                .map(|id| hub_member_nodes.contains(id))
                .unwrap_or(false);
            let verified = in_mesh && member_ok;
            if verified {
                peer_verified += 1;
                if !matches!(peer.role, mesh::NodeRole::Client) {
                    verified_host_capable_count += 1;
                }
            } else {
                peer_unverified += 1;
                drop_ids.push(peer.id);
            }
        }

        let self_host_capable = {
            let inner = self.inner.lock().await;
            !inner.is_client && inner.node.vram_bytes() > 0
        };
        let would_block_reason = if verified_host_capable_count == 0 && !self_host_capable {
            Some("No verified host-capable peers and this node cannot host".to_string())
        } else {
            None
        };

        Ok(serde_json::json!({
            "hub_mesh_id": hub_mesh_id,
            "peer_total": peers.len(),
            "peer_verified": peer_verified,
            "peer_unverified": peer_unverified,
            "verified_host_capable_count": verified_host_capable_count,
            "self_host_capable": self_host_capable,
            "would_block_reason": would_block_reason,
            "drop_peer_ids": drop_ids.iter().map(|id| id.fmt_short().to_string()).collect::<Vec<_>>(),
        }))
    }

    async fn leave_local_mesh_for_hub_join(&self) -> usize {
        let node = self.inner.lock().await.node.clone();
        let peers = node.peers().await;
        let drop_ids: Vec<_> = peers.into_iter().map(|p| p.id).collect();
        let dropped = drop_ids.len();
        if !drop_ids.is_empty() {
            node.drop_peers(&drop_ids).await;
        }
        self.set_hub_link_mode("unlinked", "local", None).await;
        dropped
    }

    async fn enforce_hub_membership_once(&self) -> (usize, bool) {
        let (hub_mesh_id, should_enforce) = {
            let inner = self.inner.lock().await;
            (
                inner.hub.linked_mesh_id.clone(),
                inner.hub.membership_enforcement == "hub_enforced",
            )
        };
        if !should_enforce {
            return (0, false);
        }
        let Some(hub_mesh_id) = hub_mesh_id else {
            return (0, false);
        };

        let now = now_unix_secs();
        let hub_member_nodes = match self.hub_fetch_mesh_node_ids(&hub_mesh_id).await {
            Ok(ids) => {
                let mut status_changed = false;
                {
                    let mut inner = self.inner.lock().await;
                    inner.hub.last_membership_sync_ok_unix = Some(now);
                    if inner.hub.link_state != "linked" {
                        inner.hub.link_state = "linked".to_string();
                        Self::persist_hub_state(&inner);
                        status_changed = true;
                    }
                }
                if status_changed {
                    self.push_status().await;
                }
                ids
            }
            Err(err) => {
                let (node, should_block, status_changed, age_secs) = {
                    let mut inner = self.inner.lock().await;
                    if inner.hub.last_membership_sync_ok_unix.is_none() {
                        inner.hub.last_membership_sync_ok_unix = Some(now);
                    }
                    let last_ok = inner.hub.last_membership_sync_ok_unix.unwrap_or(now);
                    let age = now.saturating_sub(last_ok);
                    let block = age > (HUB_MEMBERSHIP_TTL_SECS + HUB_OUTAGE_GRACE_SECS);
                    let mut changed = false;
                    if inner.hub.link_state != "degraded" {
                        inner.hub.link_state = "degraded".to_string();
                        Self::persist_hub_state(&inner);
                        changed = true;
                    }
                    (inner.node.clone(), block, changed, age)
                };
                tracing::warn!(
                    "Hub membership sync failed (age={}s, ttl={}s, grace={}s): {}",
                    age_secs,
                    HUB_MEMBERSHIP_TTL_SECS,
                    HUB_OUTAGE_GRACE_SECS,
                    err
                );
                if should_block {
                    let peers = node.peers().await;
                    let drop_ids: Vec<_> = peers.into_iter().map(|p| p.id).collect();
                    let dropped = drop_ids.len();
                    if !drop_ids.is_empty() {
                        node.drop_peers(&drop_ids).await;
                    }
                    return (dropped, true);
                }
                return (0, status_changed);
            }
        };

        let node = self.inner.lock().await.node.clone();
        let peers = node.peers().await;
        let mut drop_ids = Vec::new();
        for peer in peers {
            let in_mesh = peer.hub_mesh_id.as_deref() == Some(hub_mesh_id.as_str());
            let member_ok = peer
                .hub_node_id
                .as_ref()
                .map(|id| hub_member_nodes.contains(id))
                .unwrap_or(false);
            if !(in_mesh && member_ok) {
                drop_ids.push(peer.id);
            }
        }
        let dropped = drop_ids.len();
        if !drop_ids.is_empty() {
            node.drop_peers(&drop_ids).await;
        }
        (dropped, dropped > 0)
    }

    async fn apply_hub_identity_to_node(&self) {
        let (node, hub_node_id, hub_mesh_id) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.hub.node_id.clone(),
                inner.hub.linked_mesh_id.clone(),
            )
        };
        node.set_hub_identity(hub_node_id, hub_mesh_id).await;
    }

    async fn status(&self) -> StatusPayload {
        // Snapshot inner fields and drop the lock before any async node queries.
        // This prevents deadlock: if node.peers() etc. block on node.state.lock(),
        // we don't hold inner.lock() hostage, so other handlers can still proceed.
        let (
            node,
            node_id,
            token,
            my_vram_gb,
            inflight_requests,
            model_name,
            model_size_bytes,
            llama_ready,
            is_host,
            is_client,
            api_port,
            draft_name,
            mesh_name,
            latest_version,
            hub_state,
        ) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.node.id().fmt_short().to_string(),
                inner.node.invite_token(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.node.inflight_requests(),
                inner.model_name.clone(),
                inner.model_size_bytes,
                inner.llama_ready,
                inner.is_host,
                inner.is_client,
                inner.api_port,
                inner.draft_name.clone(),
                inner.mesh_name.clone(),
                inner.latest_version.clone(),
                inner.hub.clone(),
            )
        }; // inner lock dropped here

        let all_peers = node.peers().await;
        let peers: Vec<PeerPayload> = all_peers
            .iter()
            .map(|p| PeerPayload {
                id: p.id.fmt_short().to_string(),
                role: match p.role {
                    mesh::NodeRole::Worker => "Worker".into(),
                    mesh::NodeRole::Host { .. } => "Host".into(),
                    mesh::NodeRole::Client => "Client".into(),
                },
                models: p.models.clone(),
                vram_gb: p.vram_bytes as f64 / 1e9,
                serving: p.serving.clone(),
                rtt_ms: p.rtt_ms,
            })
            .collect();

        let catalog = node.mesh_catalog().await;
        let served = node.models_being_served().await;
        let active_demand = node.active_demand().await;
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mesh_models: Vec<MeshModelPayload> = catalog
            .iter()
            .map(|name| {
                let is_warm = served.contains(name);
                let node_count = if is_warm {
                    let peer_count = all_peers
                        .iter()
                        .filter(|p| p.serving.as_deref() == Some(name.as_str()))
                        .count();
                    let me = if *name == model_name { 1 } else { 0 };
                    peer_count + me
                } else {
                    0
                };
                let size_gb = if *name == model_name && model_size_bytes > 0 {
                    model_size_bytes as f64 / 1e9
                } else {
                    download::parse_size_gb(
                        download::MODEL_CATALOG
                            .iter()
                            .find(|m| {
                                m.file.strip_suffix(".gguf").unwrap_or(m.file) == name.as_str()
                                    || m.name == name.as_str()
                            })
                            .map(|m| m.size)
                            .unwrap_or("0"),
                    )
                };
                let (request_count, last_active_secs_ago) = match active_demand.get(name) {
                    Some(d) => (
                        Some(d.request_count),
                        Some(now_ts.saturating_sub(d.last_active)),
                    ),
                    None => (None, None),
                };
                MeshModelPayload {
                    name: name.clone(),
                    status: if is_warm {
                        "warm".into()
                    } else {
                        "cold".into()
                    },
                    node_count,
                    size_gb,
                    request_count,
                    last_active_secs_ago,
                }
            })
            .collect();

        let (launch_pi, launch_goose) = if llama_ready {
            (
                Some(format!("pi --provider mesh --model {model_name}")),
                Some(format!("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={model_name} goose session")),
            )
        } else {
            (None, None)
        };

        let mesh_id = node.mesh_id().await;
        let hub_profile_url = format!("{}/profile", hub_state.base_url);
        let hub_meshes_url = format!("{}/meshes", hub_state.base_url);
        let hub_mesh_url = hub_state
            .linked_mesh_id
            .as_ref()
            .map(|id| format!("{}/meshes/{}", hub_state.base_url, id));
        let hub_auth_state = if hub_state.access_token.is_some() {
            "logged_in".to_string()
        } else if hub_state.auth_pending {
            "auth_pending".to_string()
        } else {
            "logged_out".to_string()
        };
        let status_token = if hub_state.membership_enforcement == "hub_enforced" {
            String::new()
        } else {
            token
        };

        // Derive node status for display
        let node_status = if is_client {
            "Client".to_string()
        } else if is_host && llama_ready {
            let has_split_workers = all_peers.iter().any(|p| {
                matches!(p.role, mesh::NodeRole::Worker)
                    && p.serving.as_deref() == Some(model_name.as_str())
            });
            if has_split_workers {
                "Serving (split)".to_string()
            } else {
                "Serving".to_string()
            }
        } else if !is_host && model_name != "(idle)" && !model_name.is_empty() {
            "Worker (split)".to_string()
        } else if model_name == "(idle)" || model_name.is_empty() {
            if all_peers.is_empty() {
                "Idle".to_string()
            } else {
                "Standby".to_string()
            }
        } else {
            "Standby".to_string()
        };

        StatusPayload {
            version: MESH_LLM_VERSION.to_string(),
            latest_version,
            node_id,
            token: status_token,
            node_status,
            is_host,
            is_client,
            llama_ready,
            model_name,
            draft_name,
            api_port,
            my_vram_gb,
            model_size_gb: model_size_bytes as f64 / 1e9,
            peers,
            launch_pi,
            launch_goose,
            mesh_models,
            inflight_requests,
            mesh_id,
            mesh_name,
            hub_base_url: Some(hub_state.base_url),
            hub_auth_state: Some(hub_auth_state),
            hub_user_name: hub_state.profile.as_ref().and_then(|p| p.name.clone()),
            hub_user_handle: hub_state.profile.as_ref().and_then(|p| p.handle.clone()),
            hub_user_avatar_url: hub_state
                .profile
                .as_ref()
                .and_then(|p| p.avatar_url.clone()),
            hub_profile_url: Some(hub_profile_url),
            hub_meshes_url: Some(hub_meshes_url),
            hub_mesh_url,
            mesh_link_state: Some(hub_state.link_state),
            membership_enforcement: Some(hub_state.membership_enforcement),
            hub_default_mesh_selector: hub_state.default_mesh_selector,
            hub_default_invite_configured: Some(hub_state.default_invite_token.is_some()),
        }
    }

    async fn push_status(&self) {
        let status = self.status().await;
        if let Ok(json) = serde_json::to_string(&status) {
            let event = format!("data: {json}\n\n");
            let mut inner = self.inner.lock().await;
            inner.sse_clients.retain(|tx| !tx.is_closed());
            for tx in &inner.sse_clients {
                let _ = tx.send(event.clone());
            }
        }
    }
}

// ── Server ──

/// Start the mesh management API server.
pub async fn start(
    port: u16,
    state: MeshApi,
    mut target_rx: watch::Receiver<election::InferenceTarget>,
    listen_all: bool,
) {
    // Watch election target changes
    let state2 = state.clone();
    tokio::spawn(async move {
        loop {
            if target_rx.changed().await.is_err() {
                break;
            }
            let target = target_rx.borrow().clone();
            match target {
                election::InferenceTarget::Local(port)
                | election::InferenceTarget::MoeLocal(port) => {
                    state2.set_llama_port(Some(port)).await;
                }
                election::InferenceTarget::Remote(_) | election::InferenceTarget::MoeRemote(_) => {
                    let mut inner = state2.inner.lock().await;
                    inner.llama_ready = true;
                    inner.llama_port = None;
                }
                election::InferenceTarget::None => {
                    state2.set_llama_port(None).await;
                }
            }
            state2.push_status().await;
        }
    });

    // Push status when peers join/leave.
    let mut peer_rx = {
        let inner = state.inner.lock().await;
        inner.node.peer_change_rx.clone()
    };
    let state3 = state.clone();
    tokio::spawn(async move {
        loop {
            if peer_rx.changed().await.is_err() {
                break;
            }
            state3.push_status().await;
        }
    });

    // Push status when in-flight request count changes.
    let mut inflight_rx = {
        let inner = state.inner.lock().await;
        inner.node.inflight_change_rx()
    };
    let state4 = state.clone();
    tokio::spawn(async move {
        loop {
            if inflight_rx.changed().await.is_err() {
                break;
            }
            state4.push_status().await;
        }
    });

    // One-shot check for newer public release (for UI footer indicator).
    let state5 = state.clone();
    tokio::spawn(async move {
        let Some(latest) = crate::latest_release_version().await else {
            return;
        };
        if !crate::version_newer(&latest, crate::VERSION) {
            return;
        }
        {
            let mut inner = state5.inner.lock().await;
            inner.latest_version = Some(latest);
        }
        state5.push_status().await;
    });

    // Apply persisted hub link identity to gossip announcements on startup.
    state.apply_hub_identity_to_node().await;
    if state.hub_access_token().await.is_some() {
        if let Ok(profile) = state.hub_fetch_profile().await {
            state.set_hub_profile(profile).await;
        }
    }

    // Runtime membership enforcement for linked meshes.
    // In hub_enforced mode, periodically re-validate peers against hub membership.
    let state6 = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            HUB_MEMBERSHIP_SYNC_INTERVAL_SECS,
        ));
        interval.tick().await;
        loop {
            interval.tick().await;
            if !state6.is_hub_enforced_mode().await {
                continue;
            }
            let (dropped, status_changed) = state6.enforce_hub_membership_once().await;
            if dropped > 0 || status_changed {
                state6.push_status().await;
            }
        }
    });

    // Runtime telemetry publishing for hub-linked meshes.
    let state7 = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            HUB_TELEMETRY_SYNC_INTERVAL_SECS,
        ));
        interval.tick().await;
        loop {
            interval.tick().await;
            if let Err(err) = state7.publish_hub_telemetry_once().await {
                tracing::warn!("Hub telemetry publish failed: {err}");
            }
        }
    });

    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Management API: failed to bind :{port}: {e}");
            return;
        }
    };
    tracing::info!("Management API on http://localhost:{port}");

    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_request(stream, &state).await {
                tracing::debug!("API connection error: {e}");
            }
        });
    }
}

// ── Request dispatch ──

async fn handle_request(mut stream: TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let method = req.split_whitespace().next().unwrap_or("GET");
    let path = req.split_whitespace().nth(1).unwrap_or("/");
    let path_only = path.split('?').next().unwrap_or(path);

    match (method, path_only) {
        // ── Dashboard UI ──
        ("GET", "/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", "/dashboard") | ("GET", "/chat") | ("GET", "/dashboard/") | ("GET", "/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", p) if p.starts_with("/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        // ── Frontend static assets ──
        ("GET", p) if p.starts_with("/assets/") => {
            if !respond_console_asset(&mut stream, p).await? {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        // ── Discover meshes via Nostr ──
        ("GET", "/api/discover") => {
            if state.is_hub_enforced_mode().await {
                respond_error(
                    &mut stream,
                    403,
                    "Local discovery is disabled for InferenceHub-linked meshes",
                )
                .await?;
                return Ok(());
            }
            let relays = state.inner.lock().await.nostr_relays.clone();
            let filter = nostr::MeshFilter::default();
            match nostr::discover(&relays, &filter).await {
                Ok(meshes) => {
                    if let Ok(json) = serde_json::to_string(&meshes) {
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(), json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    } else {
                        respond_error(&mut stream, 500, "Failed to serialize").await?;
                    }
                }
                Err(e) => {
                    respond_error(&mut stream, 500, &format!("Discovery failed: {e}")).await?;
                }
            }
        }

        // ── InferenceHub device auth + profile ──
        ("POST", "/api/hub/login-device/start") => {
            let body = parse_json_body(&req).unwrap_or_default();
            let client_name = body
                .get("client_name")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("mesh-llm");
            let base_url = state.hub_base_url().await;
            let client = reqwest::Client::new();
            match client
                .post(format!("{base_url}/api/v0/device-auth/start"))
                .json(&serde_json::json!({ "client_name": client_name }))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_else(|_| {
                        "{\"error\":\"failed to read hub response\"}".to_string()
                    });
                    if (200..300).contains(&status) {
                        state.set_hub_auth_pending(true).await;
                        state.push_status().await;
                    }
                    respond_json_raw(&mut stream, status, &text).await?;
                }
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Hub auth start failed: {e}")).await?;
                }
            }
        }

        ("POST", "/api/hub/login-device/poll") => {
            let body = parse_json_body(&req).unwrap_or_default();
            let Some(device_code) = body
                .get("device_code")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            else {
                respond_error(&mut stream, 400, "device_code is required").await?;
                return Ok(());
            };
            let base_url = state.hub_base_url().await;
            let client = reqwest::Client::new();
            match client
                .post(format!("{base_url}/api/v0/device-auth/poll"))
                .json(&serde_json::json!({ "device_code": device_code }))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_else(|_| {
                        "{\"error\":\"failed to read hub response\"}".to_string()
                    });
                    if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) {
                        match payload.get("status").and_then(|v| v.as_str()) {
                            Some("authorized") => {
                                if let Some(token) = payload
                                    .get("access_token")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                {
                                    state.set_hub_access_token(Some(token)).await;
                                    state.set_hub_auth_pending(false).await;
                                    if let Ok(profile) = state.hub_fetch_profile().await {
                                        state.set_hub_profile(profile).await;
                                    }
                                }
                            }
                            Some("pending") => {
                                state.set_hub_auth_pending(true).await;
                            }
                            _ => {
                                state.set_hub_auth_pending(false).await;
                            }
                        }
                        state.push_status().await;
                    }
                    respond_json_raw(&mut stream, status, &text).await?;
                }
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Hub auth poll failed: {e}")).await?;
                }
            }
        }

        ("GET", "/api/hub/profile") => {
            if state.hub_access_token().await.is_none() {
                let body = serde_json::json!({ "authenticated": false, "profile": serde_json::Value::Null });
                respond_json_value(&mut stream, 200, &body).await?;
                return Ok(());
            }
            match state.hub_fetch_profile().await {
                Ok(profile) => {
                    state.set_hub_profile(profile.clone()).await;
                    state.push_status().await;
                    let body = if let Some(profile) = profile {
                        serde_json::json!({ "authenticated": true, "profile": profile })
                    } else {
                        serde_json::json!({ "authenticated": false, "profile": serde_json::Value::Null })
                    };
                    respond_json_value(&mut stream, 200, &body).await?;
                }
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Hub profile fetch failed: {e}"))
                        .await?;
                }
            }
        }

        ("POST", "/api/hub/logout") => {
            state.set_hub_auth_pending(false).await;
            state.set_hub_access_token(None).await;
            state.set_hub_profile(None).await;
            state.set_hub_node_id(None).await;
            state.set_hub_link_mode("unlinked", "local", None).await;
            state.push_status().await;
            respond_json_value(&mut stream, 200, &serde_json::json!({ "ok": true })).await?;
        }

        ("POST", "/api/hub/leave-local") => {
            let dropped = state.leave_local_mesh_for_hub_join().await;
            state.push_status().await;
            respond_json_value(
                &mut stream,
                200,
                &serde_json::json!({
                    "ok": true,
                    "dropped_peer_count": dropped,
                    "mode": "local",
                }),
            )
            .await?;
        }

        ("POST", "/api/hub/link-preflight") => {
            let body = parse_json_body(&req).unwrap_or_default();
            let requested_hub_mesh_id = body
                .get("hub_mesh_id")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            if state.hub_access_token().await.is_none() {
                respond_error(&mut stream, 401, "Login with InferenceHub required").await?;
                return Ok(());
            }
            let hub_mesh_id = match state
                .hub_resolve_target_mesh_id(requested_hub_mesh_id)
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    respond_error(&mut stream, 400, &e.to_string()).await?;
                    return Ok(());
                }
            };
            let node_id = match state.hub_ensure_registered_node().await {
                Ok(id) => id,
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Node registration failed: {e}"))
                        .await?;
                    return Ok(());
                }
            };
            match state.compute_link_preflight(&hub_mesh_id).await {
                Ok(mut payload) => {
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert("node_id".to_string(), serde_json::Value::String(node_id));
                    }
                    respond_json_value(&mut stream, 200, &payload).await?;
                }
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Preflight failed: {e}")).await?;
                }
            }
        }

        ("POST", "/api/hub/link-commit") => {
            let body = parse_json_body(&req).unwrap_or_default();
            let requested_hub_mesh_id = body
                .get("hub_mesh_id")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let confirm_phrase = body
                .get("confirm_phrase")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            if confirm_phrase != "LINK MESH NOW" {
                respond_error(&mut stream, 400, "confirm_phrase must be LINK MESH NOW").await?;
                return Ok(());
            }
            if state.hub_access_token().await.is_none() {
                respond_error(&mut stream, 401, "Login with InferenceHub required").await?;
                return Ok(());
            }
            let hub_mesh_id = match state
                .hub_resolve_target_mesh_id(requested_hub_mesh_id)
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    respond_error(&mut stream, 400, &e.to_string()).await?;
                    return Ok(());
                }
            };

            let preflight = match state.compute_link_preflight(&hub_mesh_id).await {
                Ok(v) => v,
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Preflight failed: {e}")).await?;
                    return Ok(());
                }
            };
            if preflight
                .get("would_block_reason")
                .and_then(|v| v.as_str())
                .is_some()
            {
                respond_json_value(&mut stream, 409, &preflight).await?;
                return Ok(());
            }

            let node_id = match state.hub_ensure_registered_node().await {
                Ok(id) => id,
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Node registration failed: {e}"))
                        .await?;
                    return Ok(());
                }
            };
            if let Err(e) = state.attach_node_to_hub_mesh(&hub_mesh_id, &node_id).await {
                respond_error(&mut stream, 502, &format!("Attach failed: {e}")).await?;
                return Ok(());
            }

            state
                .set_hub_link_mode("linked", "hub_enforced", Some(hub_mesh_id.clone()))
                .await;
            state.apply_hub_identity_to_node().await;

            let hub_member_nodes = match state.hub_fetch_mesh_node_ids(&hub_mesh_id).await {
                Ok(ids) => ids,
                Err(e) => {
                    respond_error(&mut stream, 502, &format!("Membership fetch failed: {e}"))
                        .await?;
                    return Ok(());
                }
            };
            let node = state.inner.lock().await.node.clone();
            let peers = node.peers().await;
            let mut drop_ids = Vec::new();
            for peer in peers {
                let in_mesh = peer.hub_mesh_id.as_deref() == Some(hub_mesh_id.as_str());
                let member_ok = peer
                    .hub_node_id
                    .as_ref()
                    .map(|id| hub_member_nodes.contains(id))
                    .unwrap_or(false);
                if !(in_mesh && member_ok) {
                    drop_ids.push(peer.id);
                }
            }
            let dropped_count = drop_ids.len();
            if !drop_ids.is_empty() {
                node.drop_peers(&drop_ids).await;
            }
            state.push_status().await;
            respond_json_value(
                &mut stream,
                200,
                &serde_json::json!({
                    "mode": "hub_enforced",
                    "hub_mesh_id": hub_mesh_id,
                    "node_id": node_id,
                    "dropped_peer_count": dropped_count,
                }),
            )
            .await?;
        }

        // ── Live status ──
        ("GET", "/api/status") => {
            let status = state.status().await;
            let json = serde_json::to_string(&status)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        // ── SSE event stream ──
        ("GET", "/api/events") => {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            stream.write_all(header.as_bytes()).await?;

            let status = state.status().await;
            if let Ok(json) = serde_json::to_string(&status) {
                stream
                    .write_all(format!("data: {json}\n\n").as_bytes())
                    .await?;
            }

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            state.inner.lock().await.sse_clients.push(tx);

            while let Some(event) = rx.recv().await {
                if stream.write_all(event.as_bytes()).await.is_err() {
                    break;
                }
            }
        }

        // ── Chat proxy (routes through inference API port) ──
        (m, p) if m != "POST" && p.starts_with("/api/chat") => {
            respond_error(&mut stream, 405, "Method Not Allowed").await?;
        }
        ("POST", p) if p.starts_with("/api/chat") => {
            let inner = state.inner.lock().await;
            if !inner.llama_ready && !inner.is_client {
                drop(inner);
                return respond_error(&mut stream, 503, "LLM not ready").await;
            }
            let port = inner.api_port;
            drop(inner);
            let target = format!("127.0.0.1:{port}");
            if let Ok(mut upstream) = TcpStream::connect(&target).await {
                let rewritten = req.replacen("/api/chat", "/v1/chat/completions", 1);
                upstream.write_all(rewritten.as_bytes()).await?;
                tokio::io::copy_bidirectional(&mut stream, &mut upstream).await?;
            } else {
                respond_error(&mut stream, 502, "Cannot reach LLM server").await?;
            }
        }

        _ => {
            respond_error(&mut stream, 404, "Not found").await?;
        }
    }
    Ok(())
}

async fn respond_error(stream: &mut TcpStream, code: u16, msg: &str) -> anyhow::Result<()> {
    let body = format!("{{\"error\":\"{msg}\"}}");
    let status = match code {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        405 => "Method Not Allowed",
        409 => "Conflict",
        410 => "Gone",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Not Found",
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

fn parse_json_body(req: &str) -> Option<serde_json::Value> {
    let (_head, body) = req.split_once("\r\n\r\n")?;
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(trimmed).ok()
}

async fn respond_json_value(
    stream: &mut TcpStream,
    code: u16,
    body: &serde_json::Value,
) -> anyhow::Result<()> {
    let payload = serde_json::to_string(body)?;
    respond_json_raw(stream, code, &payload).await
}

async fn respond_json_raw(stream: &mut TcpStream, code: u16, body: &str) -> anyhow::Result<()> {
    let status = match code {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        410 => "Gone",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let payload = if body.trim().is_empty() { "{}" } else { body };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload,
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn respond_console_index(stream: &mut TcpStream) -> anyhow::Result<bool> {
    if let Some(file) = CONSOLE_DIST.get_file("index.html") {
        respond_bytes(
            stream,
            200,
            "OK",
            "text/html; charset=utf-8",
            file.contents(),
        )
        .await?;
        return Ok(true);
    }
    Ok(false)
}

async fn respond_console_asset(stream: &mut TcpStream, path: &str) -> anyhow::Result<bool> {
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return Ok(false);
    }
    let Some(file) = CONSOLE_DIST.get_file(rel) else {
        return Ok(false);
    };
    let content_type = match rel.rsplit('.').next().unwrap_or("") {
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    };
    respond_bytes(stream, 200, "OK", content_type, file.contents()).await?;
    Ok(true)
}

async fn respond_bytes(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok(())
}
