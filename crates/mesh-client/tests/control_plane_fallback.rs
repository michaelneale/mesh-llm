use mesh_client::proto::node::OwnerControlErrorCode;
use mesh_client::{
    ConfigTransportSelection, ControlPlaneBootstrapOptions, ControlPlaneRetryPolicy,
};

#[test]
fn control_plane_fallback_new_client_requires_explicit_endpoint_by_default() {
    let err = ControlPlaneBootstrapOptions::new()
        .select_transport()
        .expect_err("new config clients should require an explicit control endpoint by default");

    assert_eq!(err.code, OwnerControlErrorCode::ControlEndpointRequired);
    assert!(!err.legacy_retry_allowed);
}

#[test]
fn control_plane_fallback_new_client_uses_explicit_control_endpoint() {
    let selection = ControlPlaneBootstrapOptions::new()
        .with_control_endpoint("https://control.example.test")
        .select_transport()
        .expect("explicit control endpoint should select owner-control transport");

    assert_eq!(
        selection,
        ConfigTransportSelection::OwnerControl {
            endpoint: "https://control.example.test".to_string(),
            retry_policy: ControlPlaneRetryPolicy::NoSilentLegacyDowngrade,
        }
    );
}

#[test]
fn control_plane_fallback_no_silent_downgrade_on_unreachable_configured_endpoint() {
    let options =
        ControlPlaneBootstrapOptions::new().with_control_endpoint("https://control.example.test");

    let err = options.configured_endpoint_failure(
        OwnerControlErrorCode::ControlUnavailable,
        "dial tcp 127.0.0.1:7447: connection refused",
    );

    assert_eq!(err.code, OwnerControlErrorCode::ControlUnavailable);
    assert!(!err.legacy_retry_allowed);
}

#[test]
fn control_plane_fallback_no_silent_downgrade_on_alpn_mismatch() {
    let options =
        ControlPlaneBootstrapOptions::new().with_control_endpoint("https://control.example.test");

    let err = options.configured_endpoint_failure(
        OwnerControlErrorCode::ControlUnsupported,
        "remote endpoint did not negotiate mesh-llm-control/1",
    );

    assert_eq!(err.code, OwnerControlErrorCode::ControlUnsupported);
    assert!(!err.legacy_retry_allowed);
}
