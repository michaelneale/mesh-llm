use mesh_client::client::builder::InviteToken;
use mesh_client::client::builder::{
    ChatMessage, ChatRequest, ClientBuilder, RequestId, ResponsesRequest,
};
use mesh_client::crypto::keys::OwnerKeypair;
use mesh_client::events::{Event, EventListener};
use std::sync::{Arc, Mutex};

mod support;

struct MockListener {
    events: Arc<Mutex<Vec<Event>>>,
}

impl EventListener for MockListener {
    fn on_event(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }
}

#[tokio::test]
async fn mesh_client_join_and_status() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken(support::spawn_mock_mesh(&["mesh-model-1"], "hello from mesh").await);
    let mut client = ClientBuilder::new(kp, token).build().unwrap();

    let status_before = client.status().await;
    assert!(!status_before.connected);

    client.join().await.unwrap();

    let status_after = client.status().await;
    assert!(status_after.connected);
}

#[tokio::test]
async fn api_backed_client_join_updates_status() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken("api-mode-token".to_string());
    let mut client = ClientBuilder::new(kp, token)
        .with_api_base_url("http://127.0.0.1:9347".to_string())
        .build()
        .unwrap();

    let status_before = client.status().await;
    assert!(!status_before.connected);

    client.join().await.unwrap();

    let status_after = client.status().await;
    assert!(status_after.connected);
    assert_eq!(status_after.peer_count, 1);

    client.disconnect().await;
    let status_disconnected = client.status().await;
    assert!(!status_disconnected.connected);
}

#[tokio::test]
async fn mesh_client_disconnect() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken(support::spawn_mock_mesh(&["mesh-model-1"], "hello from mesh").await);
    let mut client = ClientBuilder::new(kp, token).build().unwrap();

    client.join().await.unwrap();
    client.disconnect().await;

    let status = client.status().await;
    assert!(!status.connected);
}

#[tokio::test]
async fn mesh_client_reconnect() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken(support::spawn_mock_mesh(&["mesh-model-1"], "hello from mesh").await);
    let mut client = ClientBuilder::new(kp, token).build().unwrap();

    client.reconnect().await.unwrap();

    let status = client.status().await;
    assert!(status.connected);
}

#[tokio::test]
async fn mesh_client_cancel_idempotent() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken("test-token".to_string());
    let client = ClientBuilder::new(kp, token).build().unwrap();

    client.cancel(RequestId("unknown-id".to_string()));
}

#[tokio::test]
async fn mesh_client_list_models_returns_models_from_mesh() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken(
        support::spawn_mock_mesh(&["mesh-model-1", "mesh-model-2"], "hello from mesh").await,
    );
    let client = ClientBuilder::new(kp, token).build().unwrap();

    let models = client.list_models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "mesh-model-1");
    assert_eq!(models[1].id, "mesh-model-2");
}

#[tokio::test]
async fn mesh_client_chat_returns_request_id() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken(support::spawn_mock_mesh(&["test-model"], "hello from mesh").await);
    let client = ClientBuilder::new(kp, token).build().unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(MockListener {
        events: events.clone(),
    });

    let req = ChatRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }],
    };

    let id = client.chat(req, listener);
    assert!(!id.0.is_empty());
}

#[tokio::test]
async fn mesh_client_responses_returns_request_id() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken(support::spawn_mock_mesh(&["test-model"], "hello from mesh").await);
    let client = ClientBuilder::new(kp, token).build().unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(MockListener {
        events: events.clone(),
    });

    let req = ResponsesRequest {
        model: "test-model".to_string(),
        input: "hello".to_string(),
    };

    let id = client.responses(req, listener);
    assert!(!id.0.is_empty());
}
