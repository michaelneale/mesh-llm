use mesh_api::events::{Event, EventListener as CoreEventListener};
use mesh_api::OwnerKeypair;
use mesh_api::{
    create_auto_client as sdk_create_auto_client,
    discover_public_meshes as sdk_discover_public_meshes, ChatMessage, ChatRequest, ClientBuilder,
    InviteToken, MeshClient, PublicMeshQuery, RequestId, ResponsesRequest,
};
use pollster::block_on;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

uniffi::setup_scaffolding!("mesh_ffi");

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("invalid invite token: {0}")]
    InvalidInviteToken(String),
    #[error("invalid owner keypair: {0}")]
    InvalidOwnerKeypair(String),
    #[error("client build failed: {0}")]
    BuildFailed(String),
    #[error("join failed: {0}")]
    JoinFailed(String),
    #[error("discovery failed: {0}")]
    DiscoveryFailed(String),
    #[error("stream failed: {0}")]
    StreamFailed(String),
    #[error("cancelled: {0}")]
    Cancelled(String),
    #[error("reconnect failed: {0}")]
    ReconnectFailed(String),
    #[error("host unavailable: {0}")]
    HostUnavailable(String),
}

#[derive(uniffi::Record)]
pub struct ModelDto {
    pub id: String,
    pub name: String,
}

#[derive(uniffi::Record)]
pub struct StatusDto {
    pub connected: bool,
    pub peer_count: u64,
}

#[derive(uniffi::Record)]
pub struct PublicMeshQueryDto {
    pub model: Option<String>,
    pub min_vram_gb: Option<f64>,
    pub region: Option<String>,
    pub target_name: Option<String>,
    pub relays: Vec<String>,
}

#[derive(uniffi::Record)]
pub struct PublicMeshDto {
    pub invite_token: String,
    pub serving: Vec<String>,
    pub wanted: Vec<String>,
    pub on_disk: Vec<String>,
    pub total_vram_bytes: u64,
    pub node_count: u64,
    pub client_count: u64,
    pub max_clients: u64,
    pub name: Option<String>,
    pub region: Option<String>,
    pub mesh_id: Option<String>,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

#[derive(uniffi::Record)]
pub struct ChatRequestDto {
    pub model: String,
    pub messages: Vec<ChatMessageDto>,
}

#[derive(uniffi::Record)]
pub struct ChatMessageDto {
    pub role: String,
    pub content: String,
}

#[derive(uniffi::Record)]
pub struct ResponsesRequestDto {
    pub model: String,
    pub input: String,
}

#[derive(uniffi::Enum)]
pub enum EventDto {
    Connecting,
    Joined { node_id: String },
    ModelsUpdated { models: Vec<ModelDto> },
    TokenDelta { request_id: String, delta: String },
    Completed { request_id: String },
    Failed { request_id: String, error: String },
    Disconnected { reason: String },
}

#[uniffi::export(callback_interface)]
pub trait EventListener: Send + Sync {
    fn on_event(&self, event: EventDto);
}

struct EventListenerBridge {
    inner: Box<dyn EventListener>,
}

impl CoreEventListener for EventListenerBridge {
    fn on_event(&self, event: Event) {
        let dto = match event {
            Event::Connecting => EventDto::Connecting,
            Event::Joined { node_id } => EventDto::Joined { node_id },
            Event::ModelsUpdated { models } => EventDto::ModelsUpdated {
                models: models
                    .into_iter()
                    .map(|m| ModelDto {
                        id: m.id,
                        name: m.name,
                    })
                    .collect(),
            },
            Event::TokenDelta { request_id, delta } => EventDto::TokenDelta { request_id, delta },
            Event::Completed { request_id } => EventDto::Completed { request_id },
            Event::Failed { request_id, error } => EventDto::Failed { request_id, error },
            Event::Disconnected { reason } => EventDto::Disconnected { reason },
        };
        self.inner.on_event(dto);
    }
}

#[derive(uniffi::Object)]
pub struct MeshClientHandle {
    command_tx: mpsc::Sender<ClientCommand>,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
}

enum ClientCommand {
    Join {
        response_tx: mpsc::SyncSender<Result<(), FfiError>>,
    },
    ListModels {
        response_tx: mpsc::SyncSender<Result<Vec<ModelDto>, FfiError>>,
    },
    Chat {
        request: ChatRequestDto,
        listener: Box<dyn EventListener>,
        response_tx: mpsc::SyncSender<String>,
    },
    Responses {
        request: ResponsesRequestDto,
        listener: Box<dyn EventListener>,
        response_tx: mpsc::SyncSender<String>,
    },
    AddEventListener {
        listener: Box<dyn EventListener>,
        response_tx: mpsc::SyncSender<String>,
    },
    RemoveEventListener {
        listener_id: String,
    },
    Cancel {
        request_id: String,
    },
    Status {
        response_tx: mpsc::SyncSender<StatusDto>,
    },
    Disconnect,
    Reconnect {
        response_tx: mpsc::SyncSender<Result<(), FfiError>>,
    },
    Shutdown,
}

/// Generate a fresh owner keypair, returning its hex-encoded form.
///
/// Callers should persist this value on first run and pass it back to
/// `create_client` on subsequent launches so the embedded client keeps a
/// stable identity. Generating a new keypair on every launch will make the
/// app look like a different owner to the mesh each time.
#[uniffi::export]
pub fn generate_owner_keypair_hex() -> String {
    OwnerKeypair::generate().to_hex()
}

#[uniffi::export]
pub fn create_client(
    owner_keypair_bytes_hex: String,
    invite_token: String,
) -> Result<Arc<MeshClientHandle>, FfiError> {
    let token = invite_token
        .parse::<InviteToken>()
        .map_err(FfiError::InvalidInviteToken)?;
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    let client = ClientBuilder::new(kp, token)
        .build()
        .map_err(|error| FfiError::BuildFailed(error.to_string()))?;
    Ok(spawn_client_worker(client)?)
}

#[uniffi::export]
pub fn create_auto_client(
    owner_keypair_bytes_hex: String,
    query: PublicMeshQueryDto,
) -> Result<Arc<MeshClientHandle>, FfiError> {
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    let client = block_on(sdk_create_auto_client(kp, query.into()))
        .map(|result| result.client)
        .map_err(map_mesh_api_error)?;
    Ok(spawn_client_worker(client)?)
}

#[uniffi::export]
pub fn discover_public_meshes(query: PublicMeshQueryDto) -> Result<Vec<PublicMeshDto>, FfiError> {
    let meshes = block_on(sdk_discover_public_meshes(query.into())).map_err(map_mesh_api_error)?;
    Ok(meshes.into_iter().map(PublicMeshDto::from).collect())
}

fn spawn_client_worker(client: MeshClient) -> Result<Arc<MeshClientHandle>, FfiError> {
    let (command_tx, command_rx) = mpsc::channel();
    let worker = thread::Builder::new()
        .name("mesh-ffi-client".to_string())
        .spawn(move || run_client_worker(client, command_rx))
        .map_err(|error| FfiError::BuildFailed(error.to_string()))?;
    Ok(Arc::new(MeshClientHandle {
        command_tx,
        worker: Mutex::new(Some(worker)),
    }))
}

fn parse_owner_keypair(owner_keypair_bytes_hex: &str) -> Result<OwnerKeypair, FfiError> {
    // An empty keypair is rejected rather than silently generating a fresh identity:
    // a caller that forgets to pass their persisted owner keypair would otherwise
    // get a brand-new identity every launch with no error. Callers that genuinely
    // want a new keypair should create one explicitly before calling create_client.
    let trimmed = owner_keypair_bytes_hex.trim();
    if trimmed.is_empty() {
        return Err(FfiError::InvalidOwnerKeypair(
            "owner keypair must not be empty".to_string(),
        ));
    }
    OwnerKeypair::from_hex(trimmed)
        .map_err(|error| FfiError::InvalidOwnerKeypair(error.to_string()))
}

fn map_mesh_api_error(error: mesh_api::MeshApiError) -> FfiError {
    match error {
        mesh_api::MeshApiError::Client(error) => FfiError::BuildFailed(error.to_string()),
        mesh_api::MeshApiError::Discovery(message) => FfiError::DiscoveryFailed(message),
        mesh_api::MeshApiError::NoPublicMeshFound => {
            FfiError::HostUnavailable("no public mesh matched the requested criteria".to_string())
        }
        mesh_api::MeshApiError::InvalidInviteToken(message) => {
            FfiError::InvalidInviteToken(message)
        }
    }
}

#[uniffi::export]
impl MeshClientHandle {
    pub fn join(&self) -> Result<(), FfiError> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::Join { response_tx })?;
        response_rx
            .recv()
            .map_err(|error| FfiError::JoinFailed(error.to_string()))?
    }

    pub fn list_models(&self) -> Result<Vec<ModelDto>, FfiError> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::ListModels { response_tx })?;
        response_rx
            .recv()
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))?
    }

    pub fn chat(&self, request: ChatRequestDto, listener: Box<dyn EventListener>) -> String {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::Chat {
            request,
            listener,
            response_tx,
        })
        .expect("mesh ffi client worker should accept chat commands");
        response_rx
            .recv()
            .expect("mesh ffi client worker should return chat request ids")
    }

    pub fn responses(
        &self,
        request: ResponsesRequestDto,
        listener: Box<dyn EventListener>,
    ) -> String {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::Responses {
            request,
            listener,
            response_tx,
        })
        .expect("mesh ffi client worker should accept responses commands");
        response_rx
            .recv()
            .expect("mesh ffi client worker should return responses request ids")
    }

    pub fn add_event_listener(&self, listener: Box<dyn EventListener>) -> String {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::AddEventListener {
            listener,
            response_tx,
        })
        .expect("mesh ffi client worker should accept mesh event listeners");
        response_rx
            .recv()
            .expect("mesh ffi client worker should return mesh event listener ids")
    }

    pub fn remove_event_listener(&self, listener_id: String) {
        let _ = self.send_command(ClientCommand::RemoveEventListener { listener_id });
    }

    pub fn cancel(&self, request_id: String) {
        let _ = self.send_command(ClientCommand::Cancel { request_id });
    }

    pub fn status(&self) -> StatusDto {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if self
            .send_command(ClientCommand::Status { response_tx })
            .is_err()
        {
            return StatusDto {
                connected: false,
                peer_count: 0,
            };
        }
        response_rx.recv().unwrap_or(StatusDto {
            connected: false,
            peer_count: 0,
        })
    }

    pub fn disconnect(&self) {
        let _ = self.send_command(ClientCommand::Disconnect);
    }

    pub fn reconnect(&self) -> Result<(), FfiError> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::Reconnect { response_tx })?;
        response_rx
            .recv()
            .map_err(|error| FfiError::ReconnectFailed(error.to_string()))?
    }
}

impl MeshClientHandle {
    fn send_command(&self, command: ClientCommand) -> Result<(), FfiError> {
        self.command_tx
            .send(command)
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))
    }
}

impl Drop for MeshClientHandle {
    fn drop(&mut self) {
        let _ = self.command_tx.send(ClientCommand::Shutdown);
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
    }
}

impl From<PublicMeshQueryDto> for PublicMeshQuery {
    fn from(value: PublicMeshQueryDto) -> Self {
        Self {
            model: value.model,
            min_vram_gb: value.min_vram_gb,
            region: value.region,
            target_name: value.target_name,
            relays: value.relays,
        }
    }
}

impl From<mesh_api::PublicMesh> for PublicMeshDto {
    fn from(value: mesh_api::PublicMesh) -> Self {
        Self {
            invite_token: value.invite_token,
            serving: value.serving,
            wanted: value.wanted,
            on_disk: value.on_disk,
            total_vram_bytes: value.total_vram_bytes,
            node_count: value.node_count as u64,
            client_count: value.client_count as u64,
            max_clients: value.max_clients as u64,
            name: value.name,
            region: value.region,
            mesh_id: value.mesh_id,
            publisher_npub: value.publisher_npub,
            published_at: value.published_at,
            expires_at: value.expires_at,
        }
    }
}

fn run_client_worker(mut client: MeshClient, command_rx: mpsc::Receiver<ClientCommand>) {
    while let Ok(command) = command_rx.recv() {
        match command {
            ClientCommand::Join { response_tx } => {
                let result = client
                    .join_blocking()
                    .map_err(|error| FfiError::JoinFailed(error.to_string()));
                let _ = response_tx.send(result);
            }
            ClientCommand::ListModels { response_tx } => {
                let result = client
                    .list_models_blocking()
                    .map(|models| {
                        models
                            .into_iter()
                            .map(|m| ModelDto {
                                id: m.id,
                                name: m.name,
                            })
                            .collect()
                    })
                    .map_err(|error| FfiError::DiscoveryFailed(error.to_string()));
                let _ = response_tx.send(result);
            }
            ClientCommand::Chat {
                request,
                listener,
                response_tx,
            } => {
                let bridge = Arc::new(EventListenerBridge { inner: listener });
                let req = ChatRequest {
                    model: request.model,
                    messages: request
                        .messages
                        .into_iter()
                        .map(|m| ChatMessage {
                            role: m.role,
                            content: m.content,
                        })
                        .collect(),
                };
                let _ = response_tx.send(client.chat(req, bridge).0);
            }
            ClientCommand::Responses {
                request,
                listener,
                response_tx,
            } => {
                let bridge = Arc::new(EventListenerBridge { inner: listener });
                let req = ResponsesRequest {
                    model: request.model,
                    input: request.input,
                };
                let _ = response_tx.send(client.responses(req, bridge).0);
            }
            ClientCommand::AddEventListener {
                listener,
                response_tx,
            } => {
                let bridge = Arc::new(EventListenerBridge { inner: listener });
                let _ = response_tx.send(client.add_event_listener(bridge));
            }
            ClientCommand::RemoveEventListener { listener_id } => {
                client.remove_event_listener(&listener_id);
            }
            ClientCommand::Cancel { request_id } => {
                client.cancel(RequestId(request_id));
            }
            ClientCommand::Status { response_tx } => {
                let status = client.status_blocking();
                let _ = response_tx.send(StatusDto {
                    connected: status.connected,
                    peer_count: status.peer_count as u64,
                });
            }
            ClientCommand::Disconnect => {
                client.disconnect_blocking();
            }
            ClientCommand::Reconnect { response_tx } => {
                let result = client
                    .reconnect_blocking()
                    .map_err(|error| FfiError::ReconnectFailed(error.to_string()));
                let _ = response_tx.send(result);
            }
            ClientCommand::Shutdown => break,
        }
    }
}
