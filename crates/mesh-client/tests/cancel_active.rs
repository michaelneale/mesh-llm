use mesh_client::client::builder::{ChatMessage, ChatRequest, ClientBuilder, InviteToken};
use mesh_client::crypto::keys::OwnerKeypair;
use mesh_client::events::{Event, EventListener};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

struct MockListener {
    events: Arc<Mutex<Vec<Event>>>,
}

impl EventListener for MockListener {
    fn on_event(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }
}

#[test]
fn cancel_active_request_emits_failed_cancelled() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(MockListener {
        events: events.clone(),
    });

    let client = ClientBuilder::new(kp, token).build().unwrap();

    let request = ChatRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }],
    };
    let request_id = client.chat(request, listener.clone());

    client.cancel(request_id);

    let received = events.lock().unwrap();
    let has_failed = received
        .iter()
        .any(|e| matches!(e, Event::Failed { error, .. } if error == "cancelled"));
    assert!(has_failed, "Expected Failed event with 'cancelled' error");
}
