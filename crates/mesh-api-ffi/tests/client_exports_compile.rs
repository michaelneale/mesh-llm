use mesh_ffi::{create_client, EventDto, EventListener, FfiError};

struct MockListener;

impl EventListener for MockListener {
    fn on_event(&self, _event: EventDto) {}
}

#[test]
fn client_exports_compile() {
    let _listener: Box<dyn EventListener> = Box::new(MockListener);
    let result = create_client("deadbeef".to_string(), "".to_string());
    assert!(matches!(result, Err(FfiError::InvalidInviteToken)));
}
