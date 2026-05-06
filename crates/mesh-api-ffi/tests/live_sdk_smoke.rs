use mesh_ffi::{create_client, ChatMessageDto, ChatRequestDto, EventDto, EventListener};
use std::env;
use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct ChannelListener {
    sender: Mutex<Sender<EventDto>>,
}

impl EventListener for ChannelListener {
    fn on_event(&self, event: EventDto) {
        let _ = self.sender.lock().unwrap().send(event);
    }
}

#[test]
fn ffi_client_runs_against_live_mesh() {
    let Ok(invite_token) = env::var("MESH_SDK_INVITE_TOKEN") else {
        eprintln!("skipping live SDK smoke; MESH_SDK_INVITE_TOKEN is not set");
        return;
    };
    let expected_model = env::var("MESH_SDK_MODEL_ID").unwrap_or_default();

    let owner_keypair_hex = env::var("MESH_SDK_OWNER_KEYPAIR_HEX")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| mesh_api::OwnerKeypair::generate().to_hex());
    let handle = create_client(owner_keypair_hex, invite_token).expect("create_client");
    handle.join().expect("join");

    let status = handle.status();
    assert!(status.connected, "client should be connected after join");

    let models = wait_for_models(&handle);
    assert!(!models.is_empty(), "expected at least one model");
    if !expected_model.is_empty() {
        assert!(
            models.iter().any(|model| model.id == expected_model),
            "expected model {expected_model} in returned list"
        );
    }

    let model_id = if expected_model.is_empty() {
        models[0].id.clone()
    } else {
        expected_model
    };

    let (tx, rx) = mpsc::channel();
    let request_id = handle.chat(
        ChatRequestDto {
            model: model_id,
            messages: vec![ChatMessageDto {
                role: "user".to_string(),
                content: "Say hello in exactly three words.".to_string(),
            }],
        },
        Box::new(ChannelListener {
            sender: Mutex::new(tx),
        }),
    );

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut saw_token = false;
    let mut completed = false;

    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(EventDto::TokenDelta {
                request_id: event_request_id,
                ..
            }) if event_request_id == request_id => {
                saw_token = true;
            }
            Ok(EventDto::Completed {
                request_id: event_request_id,
            }) if event_request_id == request_id => {
                completed = true;
                break;
            }
            Ok(EventDto::Failed {
                request_id: event_request_id,
                error,
            }) if event_request_id == request_id => {
                panic!("chat request failed: {error}");
            }
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(err) => panic!("event channel closed unexpectedly: {err}"),
        }
    }

    assert!(saw_token, "expected at least one token delta event");
    assert!(completed, "expected completed event before timeout");

    handle.disconnect();

    let disconnect_deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < disconnect_deadline {
        if !handle.status().connected {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    panic!("client remained connected after disconnect");
}

fn wait_for_models(handle: &mesh_ffi::MeshClientHandle) -> Vec<mesh_ffi::ModelDto> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        let models = handle.list_models().expect("list_models");
        if !models.is_empty() {
            return models;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Vec::new()
}
