use mesh_client::events::{Event, EventListener};
use mesh_client::{ChatMessage, ChatRequest, ClientBuilder, InviteToken, OwnerKeypair};
use std::sync::Arc;
use tokio::sync::mpsc;

mod support;

struct ChannelListener {
    sender: mpsc::UnboundedSender<Event>,
}

impl EventListener for ChannelListener {
    fn on_event(&self, event: Event) {
        let _ = self.sender.send(event);
    }
}

#[tokio::test]
async fn invite_token_supports_models_and_chat_over_mesh_transport() {
    let invite_token =
        support::spawn_mock_mesh(&["mesh-model-1", "mesh-model-2"], "hello from mesh").await;
    let owner_keypair = OwnerKeypair::generate();
    let mut client = ClientBuilder::new(owner_keypair, InviteToken(invite_token))
        .build()
        .expect("build client");

    client.join().await.expect("join mesh");

    let models = client.list_models().await.expect("list models");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "mesh-model-1");
    assert_eq!(models[1].id, "mesh-model-2");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let listener = Arc::new(ChannelListener { sender: tx });
    let _request_id = client.chat(
        ChatRequest {
            model: "mesh-model-1".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
        },
        listener,
    );

    let mut deltas = Vec::new();
    let mut saw_completed = false;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("event timeout")
    {
        match event {
            Event::TokenDelta { delta, .. } => deltas.push(delta),
            Event::Completed { .. } => {
                saw_completed = true;
                break;
            }
            Event::Failed { error, .. } => panic!("chat failed: {error}"),
            _ => {}
        }
    }

    assert!(saw_completed, "expected completed event");
    assert_eq!(deltas.join(""), "hello from mesh");
}
