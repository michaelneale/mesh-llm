use crate::crypto::keys::OwnerKeypair;
use crate::protocol::{
    connect_mesh, connection_protocol, read_len_prefixed, write_len_prefixed, ControlProtocol,
    ValidateControlFrame, NODE_PROTOCOL_GENERATION, STREAM_GOSSIP, STREAM_TOPOLOGY_SUBSCRIBE,
    STREAM_TUNNEL_HTTP,
};
use crate::runtime::CoreRuntime;
use base64::Engine as _;
use iroh::endpoint::{Connection, QuicTransportConfig, RelayMode};
use iroh::{Endpoint, EndpointAddr, RelayConfig, RelayMap, SecretKey};
use prost::Message as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
type CancelFlagMap =
    Arc<Mutex<HashMap<String, (Arc<AtomicBool>, Arc<dyn crate::events::EventListener>)>>>;
type ListenerMap = Arc<Mutex<HashMap<String, Arc<dyn crate::events::EventListener>>>>;

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
                api_base_url: std::env::var("MESH_CLIENT_API_BASE").ok(),
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
        let endpoint = runtime_block_on(
            &runtime,
            async move { create_endpoint(&owner_keypair).await },
        )
        .map_err(ClientError::Endpoint)?;

        Ok(MeshClient {
            runtime,
            config: self.config,
            state: Arc::new(Mutex::new(ClientState {
                endpoint,
                invite_token,
                bootstrap_addr: None,
                connection: None,
                admitted: false,
            })),
            cancel_flags: Arc::new(Mutex::new(HashMap::new())),
            listeners: Arc::new(Mutex::new(HashMap::new())),
            event_stream_generation: Arc::new(AtomicU64::new(0)),
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
    admitted: bool,
}

pub struct MeshClient {
    runtime: CoreRuntime,
    pub(crate) config: ClientConfig,
    state: Arc<Mutex<ClientState>>,
    pub(crate) cancel_flags: CancelFlagMap,
    pub listeners: ListenerMap,
    event_stream_generation: Arc<AtomicU64>,
    pub reconnect_attempts: u32,
    pub user_disconnected: bool,
}

impl MeshClient {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.runtime.handle().block_on(future)
    }

    /// Join the mesh using the invite token.
    pub async fn join(&mut self) -> Result<(), ClientError> {
        if self.config.api_base_url.is_some() {
            self.emit_event(crate::events::Event::Connecting);
            {
                let mut state = self.state.lock().unwrap();
                state.admitted = true;
            }
            self.emit_event(crate::events::Event::Joined {
                node_id: self.config.invite_token.0.clone(),
            });
            return Ok(());
        }

        let _ = self.ensure_connected().await?;
        if !self.listeners.lock().unwrap().is_empty() {
            self.start_event_stream_observer();
        }
        Ok(())
    }

    pub fn join_blocking(&mut self) -> Result<(), ClientError> {
        let handle = self.runtime.handle().clone();
        handle.block_on(self.join())
    }

    /// List available models on the mesh.
    pub async fn list_models(&self) -> Result<Vec<Model>, ClientError> {
        if let Some(base_url) = self.config.api_base_url.as_deref() {
            let response =
                http_get_json::<ModelsResponse>(base_url, "/v1/models", &self.config.user_agent)
                    .await
                    .map_err(ClientError::Endpoint)?;

            return Ok(response
                .data
                .into_iter()
                .map(|model| Model {
                    id: model.id.clone(),
                    name: model.id,
                })
                .collect());
        }

        self.fetch_models().await
    }

    pub fn list_models_blocking(&self) -> Result<Vec<Model>, ClientError> {
        self.block_on(self.list_models())
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
        let api_base_url = self.config.api_base_url.clone();
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

            let result = match api_base_url {
                Some(base_url) => {
                    http_post_json::<ChatCompletionResponse>(
                        &base_url,
                        "/v1/chat/completions",
                        &user_agent,
                        body,
                    )
                    .await
                    .map(|response| response.message_content())
                }
                None => mesh_http_request(state, connect_timeout, request.into_bytes())
                    .await
                    .and_then(|response| {
                        parse_json_response::<ChatCompletionResponse>(&response)
                            .map(|response| response.message_content())
                    }),
            };

            match result {
                Ok(content) => {
                    let cancelled = cancel_flag.load(Ordering::Relaxed);
                    if !cancelled {
                        if let Some(content) = content {
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
        let connected = if self.config.api_base_url.is_some() {
            state.admitted
        } else {
            state.connection.is_some()
        };
        Status {
            connected,
            peer_count: usize::from(connected),
        }
    }

    pub fn status_blocking(&self) -> Status {
        self.block_on(self.status())
    }

    /// Subscribe to mesh lifecycle and model-availability events.
    ///
    /// Returns a listener id that can be passed to `remove_event_listener`.
    pub fn add_event_listener(&self, listener: Arc<dyn crate::events::EventListener>) -> String {
        let listener_id = uuid::Uuid::new_v4().to_string();
        let should_start = {
            let mut listeners = self.listeners.lock().unwrap();
            let was_empty = listeners.is_empty();
            listeners.insert(listener_id.clone(), listener);
            was_empty
        };

        if should_start && self.is_joined() {
            self.start_event_stream_observer();
        }

        listener_id
    }

    /// Remove a previously-registered mesh event listener.
    pub fn remove_event_listener(&self, listener_id: &str) {
        let should_stop = {
            let mut listeners = self.listeners.lock().unwrap();
            listeners.remove(listener_id);
            listeners.is_empty()
        };

        if should_stop {
            self.event_stream_generation.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub async fn disconnect(&mut self) {
        self.user_disconnected = true;
        self.event_stream_generation.fetch_add(1, Ordering::SeqCst);
        self.clear_connection();
        self.emit_event(crate::events::Event::Disconnected {
            reason: "disconnect_requested".to_string(),
        });
    }

    pub fn disconnect_blocking(&mut self) {
        let handle = self.runtime.handle().clone();
        handle.block_on(self.disconnect())
    }

    pub async fn reconnect(&mut self) -> Result<(), ClientError> {
        self.user_disconnected = false;
        self.reconnect_attempts = 0;
        self.event_stream_generation.fetch_add(1, Ordering::SeqCst);
        self.clear_connection();
        self.emit_event(crate::events::Event::Disconnected {
            reason: "reconnect_requested".to_string(),
        });
        self.join().await
    }

    pub fn reconnect_blocking(&mut self) -> Result<(), ClientError> {
        let handle = self.runtime.handle().clone();
        handle.block_on(self.reconnect())
    }

    fn fetch_models_request(&self) -> String {
        format!(
            "GET /v1/models HTTP/1.1\r\nHost: mesh\r\nUser-Agent: {}\r\nConnection: close\r\n\r\n",
            self.config.user_agent
        )
    }

    async fn fetch_models(&self) -> Result<Vec<Model>, ClientError> {
        let response = mesh_http_request(
            self.state.clone(),
            self.config.connect_timeout,
            self.fetch_models_request().into_bytes(),
        )
        .await
        .map_err(ClientError::Endpoint)?;
        let response: ModelsResponse =
            parse_json_response(&response).map_err(ClientError::Endpoint)?;
        Ok(response
            .data
            .into_iter()
            .map(|model| Model {
                id: model.id.clone(),
                name: model.id,
            })
            .collect())
    }

    fn start_event_stream_observer(&self) {
        let generation = self.event_stream_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let state = self.state.clone();
        let listeners = self.listeners.clone();
        let connect_timeout = self.config.connect_timeout;
        let generation_counter = self.event_stream_generation.clone();

        self.runtime.handle().spawn(async move {
            observe_mesh_events(
                state,
                listeners,
                connect_timeout,
                generation_counter,
                generation,
            )
            .await;
        });
    }

    fn is_joined(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.admitted && state.connection.is_some()
    }

    async fn ensure_connected(&self) -> Result<Connection, ClientError> {
        let (endpoint, bootstrap_addr, existing_connection, admitted) = {
            let state = self.state.lock().unwrap();
            (
                state.endpoint.clone(),
                state
                    .bootstrap_addr
                    .clone()
                    .map(Ok)
                    .unwrap_or_else(|| decode_invite_token(&state.invite_token)),
                state.connection.clone(),
                state.admitted,
            )
        };

        if let Some(connection) = existing_connection.clone() {
            if admitted {
                return Ok(connection);
            }
        }

        self.emit_event(crate::events::Event::Connecting);

        let bootstrap_addr = bootstrap_addr.map_err(ClientError::Join)?;

        let connection = if let Some(connection) = existing_connection {
            connection
        } else {
            tokio::time::timeout(
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
            .map_err(|err| ClientError::Join(err.to_string()))?
        };

        ensure_connection_admitted(connection.clone(), endpoint.as_ref())
            .await
            .map_err(ClientError::Join)?;

        {
            let mut state = self.state.lock().unwrap();
            if let Some(existing) = state.connection.clone() {
                if state.admitted {
                    return Ok(existing);
                }
            }
            state.bootstrap_addr = Some(bootstrap_addr.clone());
            state.connection = Some(connection.clone());
            state.admitted = true;
        }

        self.emit_event(crate::events::Event::Joined {
            node_id: hex::encode(bootstrap_addr.id.as_bytes()),
        });

        Ok(connection)
    }

    fn clear_connection(&self) {
        let mut state = self.state.lock().unwrap();
        state.connection = None;
        state.admitted = false;
    }

    fn emit_event(&self, event: crate::events::Event) {
        emit_event_to_listeners(&self.listeners, event);
    }
}

fn emit_event_to_listeners(listeners: &ListenerMap, event: crate::events::Event) {
    let snapshot = listeners
        .lock()
        .unwrap()
        .values()
        .cloned()
        .collect::<Vec<_>>();
    for listener in snapshot {
        listener.on_event(event.clone());
    }
}

async fn observe_mesh_events(
    state: Arc<Mutex<ClientState>>,
    listeners: ListenerMap,
    connect_timeout: Duration,
    generation_counter: Arc<AtomicU64>,
    generation: u64,
) {
    let mut last_models_signature = None::<String>;
    let mut last_disconnect_reason = None::<String>;

    loop {
        if generation_counter.load(Ordering::SeqCst) != generation {
            break;
        }
        if listeners.lock().unwrap().is_empty() {
            break;
        }

        match subscribe_to_topology_stream(
            state.clone(),
            connect_timeout,
            generation_counter.clone(),
            generation,
            &mut last_models_signature,
            &listeners,
        )
        .await
        {
            Ok(()) => break,
            Err(StreamOutcome::Disconnected(error)) => {
                clear_connection_state(&state);
                if last_disconnect_reason.as_ref() != Some(&error) {
                    last_disconnect_reason = Some(error.clone());
                    emit_event_to_listeners(
                        &listeners,
                        crate::events::Event::Disconnected { reason: error },
                    );
                }
                tokio::time::sleep(Duration::from_millis(750)).await;
            }
        }
    }
}

enum StreamOutcome {
    Disconnected(String),
}

fn model_inventory_signature(models: &[Model]) -> String {
    let mut ids = models
        .iter()
        .map(|model| model.id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids.join(",")
}

#[derive(Clone, PartialEq, prost::Message)]
struct TopologySubscribe {
    #[prost(bytes = "vec", tag = "1")]
    subscriber_id: Vec<u8>,
    #[prost(uint32, tag = "2")]
    r#gen: u32,
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

impl ChatCompletionResponse {
    fn message_content(self) -> Option<String> {
        self.choices
            .first()
            .map(|choice| choice.message.content.clone())
    }
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

fn clear_connection_state(state: &Arc<Mutex<ClientState>>) {
    let mut guard = state.lock().unwrap();
    guard.connection = None;
    guard.admitted = false;
}

async fn http_get_json<T: for<'de> Deserialize<'de>>(
    base_url: &str,
    path: &str,
    user_agent: &str,
) -> Result<T, String> {
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {user_agent}\r\nConnection: close\r\n\r\n",
        host_header(base_url)?
    );
    let response = http_request(base_url, request).await?;
    parse_json_response(&response)
}

async fn http_post_json<T: for<'de> Deserialize<'de>>(
    base_url: &str,
    path: &str,
    user_agent: &str,
    body: String,
) -> Result<T, String> {
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {user_agent}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        host_header(base_url)?,
        body.len(),
        body
    );
    let response = http_request(base_url, request).await?;
    parse_json_response(&response)
}

async fn http_request(base_url: &str, request: String) -> Result<Vec<u8>, String> {
    let address = socket_addr(base_url)?;
    let mut stream = TcpStream::connect(&address)
        .await
        .map_err(|err| format!("connect {address}: {err}"))?;
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|err| format!("write request: {err}"))?;
    stream
        .shutdown()
        .await
        .map_err(|err| format!("shutdown request: {err}"))?;

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .map_err(|err| format!("read response: {err}"))?;
    Ok(response)
}

async fn subscribe_to_topology_stream(
    state: Arc<Mutex<ClientState>>,
    connect_timeout: Duration,
    generation_counter: Arc<AtomicU64>,
    generation: u64,
    last_models_signature: &mut Option<String>,
    listeners: &ListenerMap,
) -> Result<(), StreamOutcome> {
    let connection = ensure_mesh_connection(state.clone(), connect_timeout)
        .await
        .map_err(StreamOutcome::Disconnected)?;
    let endpoint = {
        let guard = state.lock().unwrap();
        guard.endpoint.clone()
    };

    let (mut send, mut recv) = tokio::time::timeout(Duration::from_secs(5), connection.open_bi())
        .await
        .map_err(|_| StreamOutcome::Disconnected("timed out opening topology stream".to_string()))?
        .map_err(|err| StreamOutcome::Disconnected(format!("open topology stream: {err}")))?;

    send.write_all(&[STREAM_TOPOLOGY_SUBSCRIBE])
        .await
        .map_err(|err| StreamOutcome::Disconnected(format!("write topology stream type: {err}")))?;

    let request = TopologySubscribe {
        subscriber_id: endpoint.id().as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    write_len_prefixed(&mut send, &request.encode_to_vec())
        .await
        .map_err(|err| StreamOutcome::Disconnected(format!("write topology subscribe: {err}")))?;

    loop {
        if generation_counter.load(Ordering::SeqCst) != generation {
            return Ok(());
        }
        if listeners.lock().unwrap().is_empty() {
            return Ok(());
        }

        let buf = match tokio::time::timeout(Duration::from_secs(1), read_len_prefixed(&mut recv))
            .await
        {
            Ok(Ok(buf)) => buf,
            Ok(Err(err)) => {
                return Err(StreamOutcome::Disconnected(format!(
                    "read topology snapshot: {err}"
                )));
            }
            Err(_) => continue,
        };

        let frame = crate::proto::node::GossipFrame::decode(buf.as_slice()).map_err(|err| {
            StreamOutcome::Disconnected(format!("decode topology snapshot: {err}"))
        })?;
        frame.validate_frame().map_err(|err| {
            StreamOutcome::Disconnected(format!("validate topology snapshot: {err}"))
        })?;

        let models = topology_models_from_gossip_frame(&frame);
        let signature = model_inventory_signature(&models);
        if last_models_signature.as_ref() != Some(&signature) {
            *last_models_signature = Some(signature);
            emit_event_to_listeners(listeners, crate::events::Event::ModelsUpdated { models });
        }
    }
}

fn topology_models_from_gossip_frame(frame: &crate::proto::node::GossipFrame) -> Vec<Model> {
    let mut ids = BTreeSet::new();

    for peer in &frame.peers {
        let is_http_host = peer.role == crate::proto::node::NodeRole::Host as i32;
        if !is_http_host {
            continue;
        }

        let hosted_models_known = peer
            .hosted_models_known
            .unwrap_or(!peer.hosted_models.is_empty());
        let model_names = if hosted_models_known {
            &peer.hosted_models
        } else {
            &peer.serving_models
        };

        for model_name in model_names {
            let trimmed = model_name.trim();
            if !trimmed.is_empty() {
                ids.insert(trimmed.to_string());
            }
        }
    }

    ids.into_iter()
        .map(|id| Model {
            name: id.clone(),
            id,
        })
        .collect()
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
    let relay_map =
        RelayMap::from_iter(DEFAULT_RELAY_URLS.iter().map(|url| {
            RelayConfig::new(url.parse().expect("default relay URL must be valid"), None)
        }));

    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
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
        let connection = ensure_mesh_connection(state.clone(), connect_timeout).await?;

        match perform_http_request(connection, request.as_slice()).await {
            Ok(response) => return Ok(response),
            Err(error) => {
                let mut guard = state.lock().unwrap();
                guard.connection = None;
                guard.admitted = false;
                last_error = Some(error);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| "mesh request failed".to_string()))
}

async fn ensure_mesh_connection(
    state: Arc<Mutex<ClientState>>,
    connect_timeout: Duration,
) -> Result<Connection, String> {
    let (endpoint, bootstrap_addr, existing_connection, admitted) = {
        let state = state.lock().unwrap();
        (
            state.endpoint.clone(),
            state
                .bootstrap_addr
                .clone()
                .map(Ok)
                .unwrap_or_else(|| decode_invite_token(&state.invite_token)),
            state.connection.clone(),
            state.admitted,
        )
    };

    let bootstrap_addr = bootstrap_addr?;
    let connection = if let Some(connection) = existing_connection {
        connection
    } else {
        tokio::time::timeout(
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
        .map_err(|err| err.to_string())?
    };

    if !admitted {
        ensure_connection_admitted(connection.clone(), endpoint.as_ref()).await?;
    }

    let mut guard = state.lock().unwrap();
    guard.bootstrap_addr = Some(bootstrap_addr);
    guard.connection = Some(connection.clone());
    guard.admitted = true;
    Ok(connection)
}

async fn ensure_connection_admitted(
    connection: Connection,
    endpoint: &Endpoint,
) -> Result<(), String> {
    match connection_protocol(&connection) {
        ControlProtocol::ProtoV1 => perform_proto_gossip_admission(connection, endpoint).await,
        ControlProtocol::JsonV0 => Err("legacy JSON gossip admission is not supported".to_string()),
    }
}

async fn perform_proto_gossip_admission(
    connection: Connection,
    endpoint: &Endpoint,
) -> Result<(), String> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|err| format!("open gossip stream: {err}"))?;
    send.write_all(&[STREAM_GOSSIP])
        .await
        .map_err(|err| format!("write gossip stream type: {err}"))?;

    let self_announcement = crate::proto::node::PeerAnnouncement {
        endpoint_id: endpoint.id().as_bytes().to_vec(),
        role: crate::proto::node::NodeRole::Client as i32,
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        serialized_addr: serde_json::to_vec(&endpoint.addr())
            .map_err(|err| format!("serialize endpoint addr: {err}"))?,
        hosted_models_known: Some(true),
        ..Default::default()
    };
    let frame = crate::proto::node::GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        peers: vec![self_announcement],
        sender_id: endpoint.id().as_bytes().to_vec(),
    };

    write_len_prefixed(&mut send, &frame.encode_to_vec())
        .await
        .map_err(|err| format!("write gossip payload: {err}"))?;
    send.finish()
        .map_err(|err| format!("finish gossip payload: {err}"))?;

    let response = read_len_prefixed(&mut recv)
        .await
        .map_err(|err| format!("read gossip response: {err}"))?;
    let response_frame = crate::proto::node::GossipFrame::decode(response.as_slice())
        .map_err(|err| format!("decode gossip response: {err}"))?;
    response_frame
        .validate_frame()
        .map_err(|err| format!("validate gossip response: {err}"))?;
    let _ = recv.read_to_end(0).await;
    Ok(())
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
    let header_end = match response.windows(4).position(|window| window == b"\r\n\r\n") {
        Some(index) => index,
        None => {
            // Some mesh paths hand back the upstream JSON body directly instead of a fully
            // framed HTTP response. Accept that shape too so the embedded SDK stays tolerant
            // across tunnel/proxy implementations.
            return serde_json::from_slice(response).map_err(|err| {
                format!("malformed HTTP response: decode JSON without headers: {err}")
            });
        }
    };
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

fn host_header(base_url: &str) -> Result<String, String> {
    socket_addr(base_url)
}

fn socket_addr(base_url: &str) -> Result<String, String> {
    base_url
        .strip_prefix("http://")
        .or_else(|| base_url.strip_prefix("https://"))
        .unwrap_or(base_url)
        .trim_end_matches('/')
        .split('/')
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .ok_or_else(|| format!("invalid API base URL: {base_url}"))
}
