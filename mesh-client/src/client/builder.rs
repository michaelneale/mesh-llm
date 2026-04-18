use crate::crypto::keys::OwnerKeypair;
use crate::protocol::{connect_mesh, STREAM_TUNNEL_HTTP};
use crate::runtime::CoreRuntime;
use base64::Engine as _;
use iroh::endpoint::{Connection, QuicTransportConfig, RelayMode};
use iroh::{Endpoint, EndpointAddr, RelayConfig, RelayMap, SecretKey};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;
type CancelFlagMap =
    Arc<Mutex<HashMap<String, (Arc<AtomicBool>, Arc<dyn crate::events::EventListener>)>>>;

const DEFAULT_RELAY_URLS: &[&str] = &[
    "https://usw1-2.relay.michaelneale.mesh-llm.iroh.link./",
    "https://aps1-1.relay.michaelneale.mesh-llm.iroh.link./",
];

pub const MAX_RECONNECT_ATTEMPTS: u32 = 10;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("runtime error: {0}")]
    Runtime(#[from] crate::runtime::RuntimeError),
    #[error("endpoint error: {0}")]
    Endpoint(String),
    #[error("join error: {0}")]
    Join(String),
}

#[derive(Clone, Debug)]
pub struct InviteToken(pub String);

impl InviteToken {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn decode_endpoint_addr(&self) -> Result<EndpointAddr, String> {
        decode_invite_token(&self.0)
    }
}

impl std::str::FromStr for InviteToken {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.trim().is_empty() {
            return Err("empty invite token".to_string());
        }
        Ok(Self(s.to_string()))
    }
}

pub struct ClientConfig {
    pub owner_keypair: OwnerKeypair,
    pub invite_token: InviteToken,
    pub user_agent: String,
    pub connect_timeout: Duration,
    pub api_base_url: Option<String>,
}

pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    pub fn new(owner_keypair: OwnerKeypair, invite_token: InviteToken) -> Self {
        Self {
            config: ClientConfig {
                owner_keypair,
                invite_token,
                user_agent: format!("mesh-client/{}", env!("CARGO_PKG_VERSION")),
                connect_timeout: Duration::from_secs(30),
                api_base_url: None,
            },
        }
    }

    pub fn with_user_agent(mut self, ua: String) -> Self {
        self.config.user_agent = ua;
        self
    }

    pub fn with_connect_timeout(mut self, d: Duration) -> Self {
        self.config.connect_timeout = d;
        self
    }

    pub fn with_api_base_url(mut self, api_base_url: String) -> Self {
        self.config.api_base_url = Some(api_base_url);
        self
    }

    pub fn build(self) -> Result<MeshClient, ClientError> {
        let runtime = CoreRuntime::new()?;
        let invite_token = self.config.invite_token.0.clone();
        let owner_keypair = self.config.owner_keypair.clone();
        let endpoint = runtime_block_on(&runtime, async move { create_endpoint(&owner_keypair).await })
            .map_err(ClientError::Endpoint)?;

        Ok(MeshClient {
            runtime,
            config: self.config,
            state: Arc::new(Mutex::new(ClientState {
                endpoint,
                invite_token,
                bootstrap_addr: None,
                connection: None,
            })),
            cancel_flags: Arc::new(Mutex::new(HashMap::new())),
            listeners: Arc::new(Mutex::new(
                Vec::<Arc<dyn crate::events::EventListener>>::new(),
            )),
            reconnect_attempts: 0,
            user_disconnected: false,
        })
    }
}

fn runtime_block_on<T, Fut>(runtime: &CoreRuntime, future: Fut) -> T
where
    T: Send + 'static,
    Fut: Future<Output = T> + Send + 'static,
{
    let handle = runtime.handle().clone();
    std::thread::spawn(move || handle.block_on(future))
        .join()
        .expect("runtime helper thread panicked")
}

struct ClientState {
    endpoint: Arc<Endpoint>,
    invite_token: String,
    bootstrap_addr: Option<EndpointAddr>,
    connection: Option<Connection>,
}

pub struct MeshClient {
    runtime: CoreRuntime,
    pub(crate) config: ClientConfig,
    state: Arc<Mutex<ClientState>>,
    pub(crate) cancel_flags: CancelFlagMap,
    pub listeners: Arc<Mutex<Vec<Arc<dyn crate::events::EventListener>>>>,
    pub reconnect_attempts: u32,
    pub user_disconnected: bool,
}

impl MeshClient {
    /// Join the mesh using the invite token.
    pub async fn join(&mut self) -> Result<(), ClientError> {
        let _ = self.ensure_connected().await?;
        Ok(())
    }

    /// List available models on the mesh.
    pub async fn list_models(&self) -> Result<Vec<Model>, ClientError> {
        let request = format!(
            "GET /v1/models HTTP/1.1\r\nHost: mesh\r\nUser-Agent: {}\r\nConnection: close\r\n\r\n",
            self.config.user_agent
        );
        let response = mesh_http_request(
            self.state.clone(),
            self.config.connect_timeout,
            request.into_bytes(),
        )
        .await
        .map_err(ClientError::Endpoint)?;
        let response: ModelsResponse = parse_json_response(&response).map_err(ClientError::Endpoint)?;
        let models: Vec<Model> = response
            .data
            .into_iter()
            .map(|model| Model {
                id: model.id.clone(),
                name: model.id,
            })
            .collect();

        self.emit_event(crate::events::Event::ModelsUpdated {
            models: models.clone(),
        });

        Ok(models)
    }

    /// Start a chat completion request. Sync — returns a `RequestId` immediately.
    /// Streaming tokens are delivered via `listener.on_event()` on the runtime thread.
    pub fn chat(
        &self,
        request: ChatRequest,
        listener: Arc<dyn crate::events::EventListener>,
    ) -> RequestId {
        let id = RequestId::new();
        let request_id = id.0.clone();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags
            .lock()
            .unwrap()
            .insert(request_id.clone(), (cancel_flag.clone(), listener.clone()));

        let state = self.state.clone();
        let connect_timeout = self.config.connect_timeout;
        let user_agent = self.config.user_agent.clone();
        let cancel_flags = self.cancel_flags.clone();

        self.runtime.handle().spawn(async move {
            let body = serde_json::json!({
                "model": request.model,
                "messages": request.messages.iter().map(|m| serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })).collect::<Vec<_>>(),
                "stream": false,
            });
            let body = body.to_string();
            let request = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: mesh\r\nUser-Agent: {user_agent}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );

            let result = mesh_http_request(state, connect_timeout, request.into_bytes()).await;

            match result.and_then(|response| parse_json_response::<ChatCompletionResponse>(&response)) {
                Ok(response) => {
                    let cancelled = cancel_flag.load(Ordering::Relaxed);
                    if !cancelled {
                        if let Some(content) = response
                            .choices
                            .first()
                            .map(|choice| choice.message.content.clone())
                        {
                            listener.on_event(crate::events::Event::TokenDelta {
                                request_id: request_id.clone(),
                                delta: content,
                            });
                        }
                        listener.on_event(crate::events::Event::Completed {
                            request_id: request_id.clone(),
                        });
                    }
                }
                Err(error) => {
                    if !cancel_flag.load(Ordering::Relaxed) {
                        listener.on_event(crate::events::Event::Failed {
                            request_id: request_id.clone(),
                            error,
                        });
                    }
                }
            }

            cancel_flags.lock().unwrap().remove(&request_id);
        });
        id
    }

    /// Start a responses request. Sync — returns a `RequestId` immediately.
    pub fn responses(
        &self,
        request: ResponsesRequest,
        listener: Arc<dyn crate::events::EventListener>,
    ) -> RequestId {
        self.chat(
            ChatRequest {
                model: request.model,
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: request.input,
                }],
            },
            listener,
        )
    }

    /// Cancel an in-flight request. No-op if the `request_id` is unknown.
    /// Emits `Event::Failed { error: "cancelled" }` to the request's listener when found.
    pub fn cancel(&self, request_id: RequestId) {
        let entry = self.cancel_flags.lock().unwrap().remove(&request_id.0);
        if let Some((flag, listener)) = entry {
            flag.store(true, Ordering::Relaxed);
            listener.on_event(crate::events::Event::Failed {
                request_id: request_id.0.clone(),
                error: "cancelled".to_string(),
            });
        }
    }

    /// Return the current mesh connection status.
    pub async fn status(&self) -> Status {
        let state = self.state.lock().unwrap();
        let connected = state.connection.is_some();
        Status {
            connected,
            peer_count: usize::from(connected),
        }
    }

    pub async fn disconnect(&mut self) {
        self.user_disconnected = true;
        self.clear_connection();
        self.emit_event(crate::events::Event::Disconnected {
            reason: "disconnect_requested".to_string(),
        });
    }

    pub async fn reconnect(&mut self) -> Result<(), ClientError> {
        self.user_disconnected = false;
        self.reconnect_attempts = 0;
        self.clear_connection();
        self.emit_event(crate::events::Event::Disconnected {
            reason: "reconnect_requested".to_string(),
        });
        self.join().await
    }

    async fn ensure_connected(&self) -> Result<Connection, ClientError> {
        let (endpoint, bootstrap_addr, existing_connection) = {
            let state = self.state.lock().unwrap();
            (
                state.endpoint.clone(),
                state
                    .bootstrap_addr
                    .clone()
                    .map(Ok)
                    .unwrap_or_else(|| decode_invite_token(&state.invite_token)),
                state.connection.clone(),
            )
        };

        if let Some(connection) = existing_connection {
            return Ok(connection);
        }

        self.emit_event(crate::events::Event::Connecting);

        let bootstrap_addr = bootstrap_addr.map_err(ClientError::Join)?;

        let connection = tokio::time::timeout(
            self.config.connect_timeout,
            connect_mesh(endpoint.as_ref(), bootstrap_addr.clone()),
        )
        .await
        .map_err(|_| {
            ClientError::Join(format!(
                "timed out after {}s while connecting to mesh",
                self.config.connect_timeout.as_secs()
            ))
        })?
        .map_err(|err| ClientError::Join(err.to_string()))?;

        {
            let mut state = self.state.lock().unwrap();
            if let Some(existing) = state.connection.clone() {
                return Ok(existing);
            }
            state.bootstrap_addr = Some(bootstrap_addr.clone());
            state.connection = Some(connection.clone());
        }

        self.emit_event(crate::events::Event::Joined {
            node_id: hex::encode(bootstrap_addr.id.as_bytes()),
        });

        Ok(connection)
    }

    fn clear_connection(&self) {
        self.state.lock().unwrap().connection = None;
    }

    fn emit_event(&self, event: crate::events::Event) {
        for listener in self.listeners.lock().unwrap().iter() {
            listener.on_event(event.clone());
        }
    }
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ResponsesRequest {
    pub model: String,
    pub input: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub name: String,
}

pub struct Status {
    pub connected: bool,
    pub peer_count: usize,
}

pub struct RequestId(pub String);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

fn decode_invite_token(invite_token: &str) -> Result<EndpointAddr, String> {
    let json = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(invite_token)
        .map_err(|err| format!("invalid invite token encoding: {err}"))?;
    serde_json::from_slice(&json).map_err(|err| format!("invalid invite token JSON: {err}"))
}

async fn create_endpoint(owner_keypair: &OwnerKeypair) -> Result<Arc<Endpoint>, String> {
    let seed = derive_node_secret_seed(owner_keypair);
    let secret_key = SecretKey::from_bytes(&seed);
    let transport_config = QuicTransportConfig::builder()
        .max_concurrent_bidi_streams(1024u32.into())
        .build();
    let relay_map = RelayMap::from_iter(DEFAULT_RELAY_URLS.iter().map(|url| RelayConfig {
        url: url.parse().expect("default relay URL must be valid"),
        quic: None,
    }));

    let endpoint = Endpoint::empty_builder()
        .secret_key(secret_key)
        .alpns(vec![crate::protocol::ALPN_V1.to_vec()])
        .relay_mode(RelayMode::Custom(relay_map))
        .transport_config(transport_config)
        .bind()
        .await
        .map_err(|err| format!("bind endpoint: {err}"))?;

    let _ = tokio::time::timeout(Duration::from_secs(5), endpoint.online()).await;
    Ok(Arc::new(endpoint))
}

fn derive_node_secret_seed(owner_keypair: &OwnerKeypair) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"mesh-client-node-id:v1");
    hasher.update(owner_keypair.signing_bytes());
    hasher.update(owner_keypair.encryption_bytes());
    let digest = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest[..32]);
    seed
}

async fn mesh_http_request(
    state: Arc<Mutex<ClientState>>,
    connect_timeout: Duration,
    request: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let mut last_error = None;

    for _ in 0..2 {
        let (endpoint, bootstrap_addr, existing_connection) = {
            let state = state.lock().unwrap();
            (
                state.endpoint.clone(),
                state
                    .bootstrap_addr
                    .clone()
                    .map(Ok)
                    .unwrap_or_else(|| decode_invite_token(&state.invite_token)),
                state.connection.clone(),
            )
        };

        let connection = if let Some(connection) = existing_connection {
            connection
        } else {
            let bootstrap_addr = bootstrap_addr?;
            let connection = tokio::time::timeout(
                connect_timeout,
                connect_mesh(endpoint.as_ref(), bootstrap_addr.clone()),
            )
            .await
            .map_err(|_| {
                format!(
                    "timed out after {}s while connecting to mesh",
                    connect_timeout.as_secs()
                )
            })?
            .map_err(|err| err.to_string())?;
            let mut guard = state.lock().unwrap();
            guard.bootstrap_addr = Some(bootstrap_addr);
            guard.connection = Some(connection.clone());
            connection
        };

        match perform_http_request(connection, request.as_slice()).await {
            Ok(response) => return Ok(response),
            Err(error) => {
                state.lock().unwrap().connection = None;
                last_error = Some(error);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| "mesh request failed".to_string()))
}

async fn perform_http_request(connection: Connection, request: &[u8]) -> Result<Vec<u8>, String> {
    let (mut send, mut recv) = tokio::time::timeout(Duration::from_secs(5), connection.open_bi())
        .await
        .map_err(|_| "timed out opening mesh HTTP tunnel".to_string())?
        .map_err(|err| format!("open mesh HTTP tunnel: {err}"))?;

    send.write_all(&[STREAM_TUNNEL_HTTP])
        .await
        .map_err(|err| format!("write tunnel type: {err}"))?;
    send.write_all(request)
        .await
        .map_err(|err| format!("write mesh HTTP request: {err}"))?;
    send.finish()
        .map_err(|err| format!("finish mesh HTTP request: {err}"))?;

    let mut response = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        match recv.read(&mut buffer).await {
            Ok(Some(0)) | Ok(None) => break,
            Ok(Some(bytes_read)) => response.extend_from_slice(&buffer[..bytes_read]),
            Err(err) => return Err(format!("read mesh HTTP response: {err}")),
        }
    }

    Ok(response)
}

fn parse_json_response<T: for<'de> Deserialize<'de>>(response: &[u8]) -> Result<T, String> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line_end = response
        .windows(2)
        .position(|window| window == b"\r\n")
        .ok_or_else(|| "missing HTTP status line".to_string())?;
    let status_line = std::str::from_utf8(&response[..status_line_end])
        .map_err(|err| format!("invalid HTTP status line: {err}"))?;
    if !status_line.contains(" 200 ") {
        let body = String::from_utf8_lossy(&response[header_end + 4..]).to_string();
        return Err(format!("HTTP request failed: {status_line}: {body}"));
    }
    serde_json::from_slice(&response[header_end + 4..]).map_err(|err| format!("decode JSON: {err}"))
}
