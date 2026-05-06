//! Protocol Conversion Matrix Tests
//!
//! Covers the v0↔v1 conversion paths exposed by `mesh-client::protocol`:
//! `canonical_config_hash`, frame encode/decode round-trips, frame validation,
//! and the v0 legacy tunnel-map decode path.

use mesh_client::proto::node::{
    GossipFrame, NodeConfigSnapshot, NodeGpuConfig, NodeModelEntry, PeerAnnouncement,
};
use mesh_client::protocol::convert::canonical_config_hash;
use mesh_client::protocol::{
    decode_control_frame, decode_legacy_tunnel_map_frame, encode_control_frame, ControlFrameError,
    MAX_CONTROL_FRAME_BYTES, NODE_PROTOCOL_GENERATION, STREAM_GOSSIP, STREAM_TUNNEL_MAP,
};

fn minimal_config() -> NodeConfigSnapshot {
    NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: mesh_client::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![NodeModelEntry {
            model: "Qwen3-8B".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            model_ref: None,
            mmproj_ref: None,
        }],
        plugins: vec![],
    }
}

fn valid_gossip_frame() -> GossipFrame {
    GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0xAB; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: mesh_client::proto::node::NodeRole::Worker as i32,
            ..Default::default()
        }],
    }
}

// ── canonical_config_hash ─────────────────────────────────────────────────────

#[test]
fn canonical_config_hash_output_is_32_bytes() {
    let hash = canonical_config_hash(&minimal_config());
    assert_eq!(hash.len(), 32);
}

#[test]
fn canonical_config_hash_is_deterministic() {
    let a = canonical_config_hash(&minimal_config());
    let b = canonical_config_hash(&minimal_config());
    assert_eq!(a, b);
}

#[test]
fn canonical_config_hash_differs_for_different_configs() {
    let config_a = minimal_config();
    let mut config_b = minimal_config();
    config_b.models.push(NodeModelEntry {
        model: "GLM-4.7-Flash".to_string(),
        mmproj: None,
        ctx_size: None,
        gpu_id: None,
        model_ref: None,
        mmproj_ref: None,
    });
    assert_ne!(
        canonical_config_hash(&config_a),
        canonical_config_hash(&config_b)
    );
}

// ── encode_control_frame + decode_control_frame round-trips ──────────────────

#[test]
fn gossip_frame_encodes_and_decodes_intact() {
    let frame = valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.sender_id, vec![0xAB; 32]);
    assert_eq!(decoded.peers.len(), 1);
}

/// decode_control_frame must reject a buffer whose stream-type byte does not match.
#[test]
fn decode_rejects_wrong_stream_type() {
    let encoded = encode_control_frame(STREAM_GOSSIP, &valid_gossip_frame());
    let err = decode_control_frame::<GossipFrame>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("mismatched stream type must be rejected");
    assert!(matches!(
        err,
        ControlFrameError::WrongStreamType {
            expected: STREAM_TUNNEL_MAP,
            got: STREAM_GOSSIP
        }
    ));
}

/// A frame whose embedded length field exceeds MAX_CONTROL_FRAME_BYTES must be
/// rejected before any allocation attempt.
#[test]
fn decode_rejects_oversize_frame() {
    let oversize_len = (MAX_CONTROL_FRAME_BYTES + 1) as u32;
    let mut fake_frame = vec![STREAM_GOSSIP];
    fake_frame.extend_from_slice(&oversize_len.to_le_bytes());
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &fake_frame)
        .expect_err("frame claiming to exceed MAX_CONTROL_FRAME_BYTES must be rejected");
    assert!(matches!(err, ControlFrameError::OversizeFrame { .. }));
}

#[test]
fn decode_rejects_truncated_frame() {
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &[0x01, 0x02])
        .expect_err("truncated frame must be rejected");
    assert!(matches!(err, ControlFrameError::DecodeError(_)));
}

// ── v0 → v1: legacy tunnel-map JSON decode ───────────────────────────────────

#[test]
fn v0_tunnel_map_json_decode_roundtrip() {
    let mut map = std::collections::HashMap::new();
    map.insert(hex::encode([0xBB; 32]), 9337u16);
    let json = serde_json::to_vec(&map).unwrap();
    let tunnel_map = decode_legacy_tunnel_map_frame(&json)
        .expect("well-formed v0 tunnel map JSON must decode successfully");
    assert_eq!(tunnel_map.entries.len(), 1);
    assert_eq!(tunnel_map.entries[0].tunnel_port, 9337);
    assert_eq!(tunnel_map.entries[0].target_peer_id, vec![0xBB; 32]);
}

#[test]
fn v0_tunnel_map_empty_json_object_yields_no_entries() {
    let tunnel_map = decode_legacy_tunnel_map_frame(b"{}")
        .expect("empty v0 tunnel map must decode to zero entries");
    assert!(tunnel_map.entries.is_empty());
}

#[test]
fn v0_tunnel_map_rejects_invalid_hex_peer_id() {
    let mut map = std::collections::HashMap::new();
    map.insert("not-valid-hex".to_string(), 9337u16);
    let json = serde_json::to_vec(&map).unwrap();
    let tunnel_map = decode_legacy_tunnel_map_frame(&json)
        .expect("invalid hex entries must be silently skipped, not panic");
    assert!(
        tunnel_map.entries.is_empty(),
        "invalid hex peer ids must be filtered out"
    );
}

// ── Frame validation rejects bad generation ───────────────────────────────────

#[test]
fn gossip_frame_with_wrong_generation_is_rejected() {
    let bad_frame = GossipFrame {
        gen: 0,
        sender_id: vec![0u8; 32],
        peers: vec![],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("gossip frame with gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));
}
