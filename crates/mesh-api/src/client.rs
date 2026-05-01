use crate::events::{Event, EventListener};
use crate::{InviteToken, OwnerKeypair};
use mesh_client::ClientError;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

pub const MAX_RECONNECT_ATTEMPTS: u32 = mesh_client::client::builder::MAX_RECONNECT_ATTEMPTS;

#[derive(Debug, Error)]
pub enum MeshApiError {
    #[error(transparent)]
    Client(#[from] ClientError),
}

#[derive(Clone, Debug)]
pub struct ClientConfig {
    pub owner_keypair: OwnerKeypair,
    pub invite_token: InviteToken,
    pub user_agent: String,
    pub connect_timeout: Duration,
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
                user_agent: format!("mesh-api/{}", env!("CARGO_PKG_VERSION")),
                connect_timeout: Duration::from_secs(30),
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

    pub fn build(self) -> Result<MeshClient, MeshApiError> {
        let inner = mesh_client::ClientBuilder::new(
            self.config.owner_keypair.into_inner(),
            self.config.invite_token.into_inner(),
        )
        .with_user_agent(self.config.user_agent.clone())
        .with_connect_timeout(self.config.connect_timeout)
        .build()?;

        Ok(MeshClient { inner })
    }
}

pub struct MeshClient {
    inner: mesh_client::MeshClient,
}

impl MeshClient {
    pub async fn join(&mut self) -> Result<(), MeshApiError> {
        self.inner.join().await?;
        Ok(())
    }

    pub async fn list_models(&self) -> Result<Vec<Model>, MeshApiError> {
        Ok(self
            .inner
            .list_models()
            .await?
            .into_iter()
            .map(Model::from)
            .collect())
    }

    pub fn chat(&self, request: ChatRequest, listener: Arc<dyn EventListener>) -> RequestId {
        let request_id = self.inner.chat(
            mesh_client::ChatRequest::from(request),
            Arc::new(EventListenerAdapter { inner: listener }),
        );
        RequestId(request_id.0)
    }

    pub fn responses(
        &self,
        request: ResponsesRequest,
        listener: Arc<dyn EventListener>,
    ) -> RequestId {
        let request_id = self.inner.responses(
            mesh_client::ResponsesRequest::from(request),
            Arc::new(EventListenerAdapter { inner: listener }),
        );
        RequestId(request_id.0)
    }

    pub fn cancel(&self, request_id: RequestId) {
        self.inner.cancel(mesh_client::RequestId(request_id.0));
    }

    pub async fn status(&self) -> Status {
        Status::from(self.inner.status().await)
    }

    pub async fn disconnect(&mut self) {
        self.inner.disconnect().await;
    }

    pub async fn reconnect(&mut self) -> Result<(), MeshApiError> {
        self.inner.reconnect().await?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

impl From<ChatRequest> for mesh_client::ChatRequest {
    fn from(value: ChatRequest) -> Self {
        Self {
            model: value.model,
            messages: value.messages.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl From<ChatMessage> for mesh_client::ChatMessage {
    fn from(value: ChatMessage) -> Self {
        Self {
            role: value.role,
            content: value.content,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: String,
}

impl From<ResponsesRequest> for mesh_client::ResponsesRequest {
    fn from(value: ResponsesRequest) -> Self {
        Self {
            model: value.model,
            input: value.input,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub name: String,
}

impl From<mesh_client::Model> for Model {
    fn from(value: mesh_client::Model) -> Self {
        Self {
            id: value.id,
            name: value.name,
        }
    }
}

pub struct Status {
    pub connected: bool,
    pub peer_count: usize,
}

impl From<mesh_client::Status> for Status {
    fn from(value: mesh_client::Status) -> Self {
        Self {
            connected: value.connected,
            peer_count: value.peer_count,
        }
    }
}

pub struct RequestId(pub String);

impl RequestId {
    pub fn new() -> Self {
        Self(mesh_client::RequestId::new().0)
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

struct EventListenerAdapter {
    inner: Arc<dyn EventListener>,
}

impl mesh_client::events::EventListener for EventListenerAdapter {
    fn on_event(&self, event: mesh_client::events::Event) {
        self.inner.on_event(match event {
            mesh_client::events::Event::Connecting => Event::Connecting,
            mesh_client::events::Event::Joined { node_id } => Event::Joined { node_id },
            mesh_client::events::Event::ModelsUpdated { models } => Event::ModelsUpdated {
                models: models.into_iter().map(Model::from).collect(),
            },
            mesh_client::events::Event::TokenDelta { request_id, delta } => {
                Event::TokenDelta { request_id, delta }
            }
            mesh_client::events::Event::Completed { request_id } => Event::Completed { request_id },
            mesh_client::events::Event::Failed { request_id, error } => {
                Event::Failed { request_id, error }
            }
            mesh_client::events::Event::Disconnected { reason } => Event::Disconnected { reason },
        });
    }
}
