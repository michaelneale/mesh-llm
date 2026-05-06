use mesh_ffi::{
    create_client, discover_public_meshes, EventDto, EventListener, FfiError, PublicMeshQueryDto,
};

struct MockListener;

impl EventListener for MockListener {
    fn on_event(&self, _event: EventDto) {}
}

#[test]
fn client_exports_compile() {
    let _listener: Box<dyn EventListener> = Box::new(MockListener);
    let result = create_client("deadbeef".to_string(), "".to_string());
    assert!(matches!(result, Err(FfiError::InvalidInviteToken(_))));
    let _query = PublicMeshQueryDto {
        model: None,
        min_vram_gb: None,
        region: None,
        target_name: None,
        relays: vec![],
    };
    let _discover_fn = discover_public_meshes;
}
