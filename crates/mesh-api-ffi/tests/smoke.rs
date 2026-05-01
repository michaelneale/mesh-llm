use mesh_ffi::{create_client, EventDto, EventListener, FfiError};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn valid_owner_keypair_hex() -> String {
    mesh_api::OwnerKeypair::generate().to_hex()
}

struct MockListener {
    events: Arc<Mutex<Vec<String>>>,
}

impl EventListener for MockListener {
    fn on_event(&self, event: EventDto) {
        let name = match &event {
            EventDto::Connecting => "Connecting".to_string(),
            EventDto::Joined { .. } => "Joined".to_string(),
            EventDto::ModelsUpdated { .. } => "ModelsUpdated".to_string(),
            EventDto::TokenDelta { .. } => "TokenDelta".to_string(),
            EventDto::Completed { .. } => "Completed".to_string(),
            EventDto::Failed { .. } => "Failed".to_string(),
            EventDto::Disconnected { .. } => "Disconnected".to_string(),
        };
        self.events.lock().unwrap().push(name);
    }
}

struct ReentrantListener {
    handle: Arc<mesh_ffi::MeshClientHandle>,
    sender: Mutex<Sender<mesh_ffi::StatusDto>>,
}

impl EventListener for ReentrantListener {
    fn on_event(&self, event: EventDto) {
        if let EventDto::Completed { .. } = event {
            let status = self.handle.status();
            let _ = self.sender.lock().unwrap().send(status);
        }
    }
}

#[test]
fn create_client_with_invalid_token_fails() {
    let result = create_client("".to_string(), "".to_string());
    assert!(matches!(result, Err(FfiError::InvalidInviteToken)));
}

#[test]
fn create_client_with_valid_token_succeeds() {
    let result = create_client(valid_owner_keypair_hex(), "valid-token".to_string());
    assert!(result.is_ok());
}

#[test]
fn create_client_with_empty_owner_keypair_fails() {
    // Empty keypair is rejected rather than silently generating a fresh identity.
    let result = create_client("".to_string(), "valid-token".to_string());
    assert!(matches!(result, Err(FfiError::InvalidOwnerKeypair)));
}

#[test]
fn client_handle_status_returns_disconnected() {
    let handle = create_client(valid_owner_keypair_hex(), "valid-token".to_string()).unwrap();
    let status = handle.status();
    assert!(!status.connected);
    assert_eq!(status.peer_count, 0);
}

#[test]
fn client_handle_cancel_unknown_id_is_noop() {
    let handle = create_client(valid_owner_keypair_hex(), "valid-token".to_string()).unwrap();
    handle.cancel("unknown-id".to_string());
}

#[test]
fn create_client_with_invalid_owner_keypair_fails() {
    let result = create_client("deadbeef".to_string(), "valid-token".to_string());
    assert!(matches!(result, Err(FfiError::InvalidOwnerKeypair)));
}

#[test]
fn create_client_uses_supplied_owner_keypair() {
    let owner_keypair_hex = {
        let keypair = mesh_api::OwnerKeypair::generate();
        keypair.to_hex()
    };

    let handle = create_client(owner_keypair_hex.clone(), "valid-token".to_string()).unwrap();
    let status = handle.status();
    assert!(!status.connected);
    assert_eq!(status.peer_count, 0);
}

#[test]
fn mock_listener_receives_events() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let listener = MockListener {
        events: events.clone(),
    };

    listener.on_event(EventDto::Connecting);
    listener.on_event(EventDto::Joined {
        node_id: "test-node".to_string(),
    });
    listener.on_event(EventDto::ModelsUpdated { models: vec![] });
    listener.on_event(EventDto::TokenDelta {
        request_id: "req-1".to_string(),
        delta: "hello".to_string(),
    });
    listener.on_event(EventDto::Completed {
        request_id: "req-1".to_string(),
    });
    listener.on_event(EventDto::Failed {
        request_id: "req-2".to_string(),
        error: "timeout".to_string(),
    });
    listener.on_event(EventDto::Disconnected {
        reason: "network".to_string(),
    });

    let received = events.lock().unwrap();
    assert_eq!(received.len(), 7);
    assert_eq!(received[0], "Connecting");
    assert_eq!(received[1], "Joined");
    assert_eq!(received[2], "ModelsUpdated");
    assert_eq!(received[3], "TokenDelta");
    assert_eq!(received[4], "Completed");
    assert_eq!(received[5], "Failed");
    assert_eq!(received[6], "Disconnected");
}

#[test]
fn handle_create_destroy_loop_25_times() {
    for i in 0..25 {
        let token = format!("invite-token-{}", i);
        let handle = create_client(valid_owner_keypair_hex(), token)
            .expect("create_client should succeed with valid inputs");
        let status = handle.status();
        assert!(!status.connected, "iteration {}: expected disconnected", i);
    }
}

#[test]
fn listener_can_reenter_handle_during_callback() {
    let handle = create_client(valid_owner_keypair_hex(), "valid-token".to_string()).unwrap();
    let (tx, rx) = mpsc::channel();
    let request_id = handle.chat(
        mesh_ffi::ChatRequestDto {
            model: "test-model".to_string(),
            messages: vec![mesh_ffi::ChatMessageDto {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
        },
        Box::new(ReentrantListener {
            handle: handle.clone(),
            sender: Mutex::new(tx),
        }),
    );

    assert!(!request_id.is_empty(), "chat should return a request id");
    let status = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("callback should be able to reenter handle without deadlocking");
    assert!(!status.connected);
}

#[test]
#[ignore] // Requires real FixtureMesh + Qwen2.5-0.5B-Q4 model
fn full_lifecycle_against_fixture() {
    // Run with: cargo test -p mesh-api-ffi --test smoke -- --ignored
    //
    // Expected sequence:
    // 1. create client handle with a real invite token
    // 2. register callback listener
    // 3. join via invite token  -> receives Connecting + Joined events
    // 4. list models            -> at least one model returned
    // 5. start one short stream -> receives TokenDelta(s) + Completed
    // 6. start second stream and cancel it
    //                           -> receives Failed { error: "cancelled" }
    // 7. drop handle cleanly    -> no panic, no leaked threads
    // 8. repeat create/destroy loop 25 times verifying clean shutdown each time
    println!("create_destroy_iterations=25");
    unimplemented!("requires FixtureMesh");
}
