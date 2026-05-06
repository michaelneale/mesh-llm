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

#[tokio::test]
async fn mesh_event_listener_receives_models_updated_from_topology_stream() {
    let mesh =
        support::spawn_mock_mesh_controlled(&["mesh-model-1", "mesh-model-2"], "hello from mesh")
            .await;
    let owner_keypair = OwnerKeypair::generate();
    let mut client = ClientBuilder::new(owner_keypair, InviteToken(mesh.invite_token))
        .build()
        .expect("build client");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let listener = Arc::new(ChannelListener { sender: tx });
    let _listener_id = client.add_event_listener(listener);

    client.join().await.expect("join mesh");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("event timeout")
        else {
            continue;
        };

        if let Event::ModelsUpdated { models } = event {
            let ids = models.into_iter().map(|model| model.id).collect::<Vec<_>>();
            assert_eq!(
                ids,
                vec!["mesh-model-1".to_string(), "mesh-model-2".to_string()]
            );
            return;
        }
    }

    panic!("expected ModelsUpdated event from topology stream");
}

#[tokio::test]
async fn mesh_event_listener_receives_models_updated_when_topology_changes() {
    let mesh = support::spawn_mock_mesh_controlled(&["mesh-model-1"], "hello from mesh").await;
    let owner_keypair = OwnerKeypair::generate();
    let mut client = ClientBuilder::new(owner_keypair, InviteToken(mesh.invite_token.clone()))
        .build()
        .expect("build client");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let listener = Arc::new(ChannelListener { sender: tx });
    let _listener_id = client.add_event_listener(listener);

    client.join().await.expect("join mesh");

    let initial = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if let Some(Event::ModelsUpdated { models }) = rx.recv().await {
                break models;
            }
        }
    })
    .await
    .expect("initial models timeout");
    let initial_ids = initial
        .into_iter()
        .map(|model| model.id)
        .collect::<Vec<_>>();
    assert_eq!(initial_ids, vec!["mesh-model-1".to_string()]);

    mesh.set_models(&["mesh-model-1", "mesh-model-2"]);

    let updated = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if let Some(Event::ModelsUpdated { models }) = rx.recv().await {
                break models;
            }
        }
    })
    .await
    .expect("updated models timeout");
    let updated_ids = updated
        .into_iter()
        .map(|model| model.id)
        .collect::<Vec<_>>();
    assert_eq!(
        updated_ids,
        vec!["mesh-model-1".to_string(), "mesh-model-2".to_string()]
    );
}

#[tokio::test]
async fn list_models_does_not_emit_models_updated_events() {
    let mesh =
        support::spawn_mock_mesh_controlled(&["mesh-model-1", "mesh-model-2"], "hello from mesh")
            .await;
    let owner_keypair = OwnerKeypair::generate();
    let mut client = ClientBuilder::new(owner_keypair, InviteToken(mesh.invite_token))
        .build()
        .expect("build client");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let listener = Arc::new(ChannelListener { sender: tx });
    let _listener_id = client.add_event_listener(listener);

    client.join().await.expect("join mesh");

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if let Some(Event::ModelsUpdated { .. }) = rx.recv().await {
                break;
            }
        }
    })
    .await
    .expect("initial models timeout");

    let models = client.list_models().await.expect("list models");
    assert_eq!(models.len(), 2);

    if let Ok(Some(Event::ModelsUpdated { models })) =
        tokio::time::timeout(std::time::Duration::from_millis(400), rx.recv()).await
    {
        panic!(
            "list_models must not emit ModelsUpdated, but received {:?}",
            models.into_iter().map(|model| model.id).collect::<Vec<_>>()
        );
    }
}
