use mesh_client::client::builder::{ClientBuilder, InviteToken};
use mesh_client::crypto::keys::OwnerKeypair;
use mesh_client::events::{Event, EventListener};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

struct MockListener {
    events: Arc<Mutex<Vec<String>>>,
}

impl EventListener for MockListener {
    fn on_event(&self, event: Event) {
        let name = match &event {
            Event::Connecting => "Connecting",
            Event::Joined { .. } => "Joined",
            Event::Disconnected { .. } => "Disconnected",
            _ => "Other",
        };
        self.events.lock().unwrap().push(name.to_string());
    }
}

#[tokio::test]
async fn reconnect_emits_disconnected_then_joined() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(MockListener {
        events: events.clone(),
    });

    let mut client = ClientBuilder::new(kp, token).build().unwrap();
    client.listeners.lock().unwrap().push(listener);

    client.join().await.unwrap();
    events.lock().unwrap().clear();

    client.reconnect().await.unwrap();

    let received = events.lock().unwrap().clone();
    assert!(
        received.contains(&"Disconnected".to_string()),
        "Expected Disconnected event, got: {received:?}"
    );
    assert!(
        received.contains(&"Joined".to_string()),
        "Expected Joined event, got: {received:?}"
    );

    let disconnected_pos = received.iter().position(|e| e == "Disconnected").unwrap();
    let joined_pos = received.iter().position(|e| e == "Joined").unwrap();
    assert!(
        disconnected_pos < joined_pos,
        "Disconnected must precede Joined, got order: {received:?}"
    );
}

#[tokio::test]
async fn reconnect_emits_reconnect_requested_reason() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();

    struct ReasonCapture {
        reasons: Arc<Mutex<Vec<String>>>,
    }
    impl EventListener for ReasonCapture {
        fn on_event(&self, event: Event) {
            if let Event::Disconnected { reason } = event {
                self.reasons.lock().unwrap().push(reason);
            }
        }
    }

    let reasons = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(ReasonCapture {
        reasons: reasons.clone(),
    });

    let mut client = ClientBuilder::new(kp, token).build().unwrap();
    client.listeners.lock().unwrap().push(listener);

    client.reconnect().await.unwrap();

    let captured = reasons.lock().unwrap().clone();
    assert_eq!(
        captured,
        vec!["reconnect_requested"],
        "reconnect() must emit reason 'reconnect_requested'"
    );
}

#[tokio::test]
async fn disconnect_emits_disconnect_requested_reason() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();

    struct ReasonCapture {
        reasons: Arc<Mutex<Vec<String>>>,
    }
    impl EventListener for ReasonCapture {
        fn on_event(&self, event: Event) {
            if let Event::Disconnected { reason } = event {
                self.reasons.lock().unwrap().push(reason);
            }
        }
    }

    let reasons = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(ReasonCapture {
        reasons: reasons.clone(),
    });

    let mut client = ClientBuilder::new(kp, token).build().unwrap();
    client.listeners.lock().unwrap().push(listener);

    client.join().await.unwrap();
    client.disconnect().await;

    let captured = reasons.lock().unwrap().clone();
    assert_eq!(
        captured,
        vec!["disconnect_requested"],
        "disconnect() must emit reason 'disconnect_requested'"
    );
}

#[tokio::test]
async fn reconnect_resets_attempt_counter() {
    let kp = OwnerKeypair::generate();
    let token = InviteToken::from_str("test-token").unwrap();
    let mut client = ClientBuilder::new(kp, token).build().unwrap();

    client.reconnect_attempts = 7;
    client.reconnect().await.unwrap();

    assert_eq!(
        client.reconnect_attempts, 0,
        "manual reconnect() must reset the attempt counter"
    );
}
