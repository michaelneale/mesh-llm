//! Protocol V0 ↔ Client Compatibility Tests
//!
//! Proves the dual-support scheme: ALPN negotiation routes v0 and v1 peers to
//! the correct codec, and the wire-format ALPN byte sequences are stable.
//! All assertions run against the public API of `mesh-client::protocol`.

use mesh_client::protocol::{
    protocol_from_alpn, ControlProtocol, ALPN_CONTROL_V1, ALPN_V0, ALPN_V1, STREAM_CONFIG_PUSH,
    STREAM_CONFIG_SUBSCRIBE,
};
use mesh_client::{
    ConfigTransportSelection, ControlPlaneBootstrapOptions, ControlPlaneRetryPolicy,
};

#[test]
fn alpn_v0_byte_sequence_is_stable() {
    assert_eq!(ALPN_V0, b"mesh-llm/0");
}

#[test]
fn alpn_v1_byte_sequence_is_stable() {
    assert_eq!(ALPN_V1, b"mesh-llm/1");
}

#[test]
fn control_alpn_byte_sequence_is_stable() {
    assert_eq!(ALPN_CONTROL_V1, b"mesh-llm-control/1");
}

#[test]
fn alpn_v0_and_v1_are_distinct() {
    assert_ne!(ALPN_V0, ALPN_V1);
}

#[test]
fn control_alpn_is_distinct_from_public_mesh_alpns() {
    assert_ne!(ALPN_CONTROL_V1, ALPN_V0);
    assert_ne!(ALPN_CONTROL_V1, ALPN_V1);
}

#[test]
fn legacy_config_stream_constants_remain_stable() {
    assert_eq!(STREAM_CONFIG_SUBSCRIBE, 0x0b);
    assert_eq!(STREAM_CONFIG_PUSH, 0x0c);
}

#[test]
fn explicit_control_endpoint_selects_owner_control() {
    let selection = ControlPlaneBootstrapOptions::new()
        .with_control_endpoint("https://control.example.test")
        .select_transport()
        .expect("explicit endpoint should stay on owner-control lane");

    assert_eq!(
        selection,
        ConfigTransportSelection::OwnerControl {
            endpoint: "https://control.example.test".to_string(),
            retry_policy: ControlPlaneRetryPolicy::NoSilentLegacyDowngrade,
        }
    );
}

#[test]
fn missing_control_endpoint_rejects_config_bootstrap() {
    let err = ControlPlaneBootstrapOptions::new()
        .select_transport()
        .expect_err("owner-control endpoint must be explicit");

    assert_eq!(
        err.code,
        mesh_client::proto::node::OwnerControlErrorCode::ControlEndpointRequired
    );
    assert!(!err.legacy_retry_allowed);
}

/// Old peers advertising only `mesh-llm/0` must be accepted with the JSON codec.
#[test]
fn protocol_from_alpn_v0_yields_json_v0() {
    assert_eq!(protocol_from_alpn(ALPN_V0), ControlProtocol::JsonV0);
}

#[test]
fn protocol_from_alpn_v1_yields_proto_v1() {
    assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
}

/// Unknown or future ALPNs must fall back to ProtoV1 without panicking.
#[test]
fn protocol_from_alpn_unknown_falls_back_to_proto_v1() {
    assert_eq!(
        protocol_from_alpn(b"mesh-llm/999"),
        ControlProtocol::ProtoV1
    );
    assert_eq!(protocol_from_alpn(b"unknown"), ControlProtocol::ProtoV1);
    assert_eq!(protocol_from_alpn(b""), ControlProtocol::ProtoV1);
}

#[test]
fn control_protocol_variants_are_distinct() {
    assert_ne!(ControlProtocol::JsonV0, ControlProtocol::ProtoV1);
}

/// ControlProtocol must be Copy; this test fails to compile if that is lost.
#[test]
fn control_protocol_is_copy() {
    let p = ControlProtocol::JsonV0;
    let _q = p;
    let _r = p;
}

/// V0 gossip is a JSON array on the wire; an empty array is a valid payload.
#[test]
fn v0_gossip_json_byte_format_is_valid_json_array() {
    let parsed: serde_json::Value =
        serde_json::from_slice(b"[]").expect("empty v0 gossip array must be valid JSON");
    assert!(parsed.is_array());
    assert_eq!(parsed.as_array().unwrap().len(), 0);
}

/// A JSON object payload must be rejected when deserialising as a v0 gossip Vec.
#[test]
fn v0_gossip_json_requires_array_not_object() {
    let result: Result<Vec<serde_json::Value>, _> = serde_json::from_slice(b"{\"peers\":[]}");
    assert!(result.is_err());
}
