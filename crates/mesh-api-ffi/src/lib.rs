use mesh_api::events::{Event, EventListener as CoreEventListener};
use mesh_api::OwnerKeypair;
use mesh_api::{
    ChatMessage, ChatRequest, ClientBuilder, InviteToken, MeshClient, RequestId, ResponsesRequest,
};
use pollster::block_on;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

uniffi::setup_scaffolding!("mesh_ffi");

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("invalid invite token")]
    InvalidInviteToken,
    #[error("invalid owner keypair")]
    InvalidOwnerKeypair,
    #[error("client build failed")]
    BuildFailed,
    #[error("join failed")]
    JoinFailed,
    #[error("discovery failed")]
    DiscoveryFailed,
    #[error("stream failed")]
    StreamFailed,
    #[error("cancelled")]
    Cancelled,
    #[error("reconnect failed")]
    ReconnectFailed,
    #[error("host unavailable")]
    HostUnavailable,
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
        .map_err(|_| FfiError::InvalidInviteToken)?;
    // An empty keypair is rejected rather than silently generating a fresh one:
    // a caller that forgets to pass their persisted owner keypair would otherwise
    // get a brand-new identity every launch with no error. Callers that genuinely
    // want a new keypair should create one explicitly before calling create_client.
    let trimmed = owner_keypair_bytes_hex.trim();
    if trimmed.is_empty() {
        return Err(FfiError::InvalidOwnerKeypair);
    }
    let kp = OwnerKeypair::from_hex(trimmed).map_err(|_| FfiError::InvalidOwnerKeypair)?;
    let client = ClientBuilder::new(kp, token)
        .build()
        .map_err(|_| FfiError::BuildFailed)?;
    let (command_tx, command_rx) = mpsc::channel();
    let worker = thread::Builder::new()
        .name("mesh-ffi-client".to_string())
        .spawn(move || run_client_worker(client, command_rx))
        .map_err(|_| FfiError::BuildFailed)?;
    Ok(Arc::new(MeshClientHandle {
        command_tx,
        worker: Mutex::new(Some(worker)),
    }))
}

#[uniffi::export]
impl MeshClientHandle {
    pub fn join(&self) -> Result<(), FfiError> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::Join { response_tx })?;
        response_rx.recv().map_err(|_| FfiError::JoinFailed)?
    }

    pub fn list_models(&self) -> Result<Vec<ModelDto>, FfiError> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        self.send_command(ClientCommand::ListModels { response_tx })?;
        response_rx.recv().map_err(|_| FfiError::DiscoveryFailed)?
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
        response_rx.recv().map_err(|_| FfiError::ReconnectFailed)?
    }
}

impl MeshClientHandle {
    fn send_command(&self, command: ClientCommand) -> Result<(), FfiError> {
        self.command_tx
            .send(command)
            .map_err(|_| FfiError::HostUnavailable)
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

fn run_client_worker(mut client: MeshClient, command_rx: mpsc::Receiver<ClientCommand>) {
    while let Ok(command) = command_rx.recv() {
        match command {
            ClientCommand::Join { response_tx } => {
                let result = block_on(client.join()).map_err(|_| FfiError::JoinFailed);
                let _ = response_tx.send(result);
            }
            ClientCommand::ListModels { response_tx } => {
                let result = block_on(client.list_models())
                    .map(|models| {
                        models
                            .into_iter()
                            .map(|m| ModelDto {
                                id: m.id,
                                name: m.name,
                            })
                            .collect()
                    })
                    .map_err(|_| FfiError::DiscoveryFailed);
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
            ClientCommand::Cancel { request_id } => {
                client.cancel(RequestId(request_id));
            }
            ClientCommand::Status { response_tx } => {
                let status = block_on(client.status());
                let _ = response_tx.send(StatusDto {
                    connected: status.connected,
                    peer_count: status.peer_count as u64,
                });
            }
            ClientCommand::Disconnect => {
                block_on(client.disconnect());
            }
            ClientCommand::Reconnect { response_tx } => {
                let result = block_on(client.reconnect()).map_err(|_| FfiError::ReconnectFailed);
                let _ = response_tx.send(result);
            }
            ClientCommand::Shutdown => break,
        }
    }
}
