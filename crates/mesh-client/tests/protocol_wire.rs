// Integration tests for mesh-client::protocol wire types.
// These tests verify the portable protocol layer that is safe to use on mobile targets.

use mesh_client::proto::node::{
    GossipFrame, NodeRole, PeerAnnouncement, PeerDown, PeerLeaving, RouteTable, RouteTableRequest,
};
use mesh_client::protocol::{
    decode_control_frame, decode_legacy_tunnel_map_frame, encode_control_frame, ControlFrameError,
    ControlProtocol, ALPN_V0, ALPN_V1, MAX_CONTROL_FRAME_BYTES, NODE_PROTOCOL_GENERATION,
    STREAM_CONFIG_PUSH, STREAM_CONFIG_SUBSCRIBE, STREAM_GOSSIP, STREAM_PEER_DOWN,
    STREAM_PEER_LEAVING, STREAM_ROUTE_REQUEST, STREAM_TUNNEL_MAP,
};

// ── ALPN constants ──────────────────────────────────────────────────────────

#[test]
fn alpn_v0_is_correct() {
    assert_eq!(ALPN_V0, b"mesh-llm/0");
}

#[test]
fn alpn_v1_is_correct() {
    assert_eq!(ALPN_V1, b"mesh-llm/1");
}

// ── ControlProtocol ─────────────────────────────────────────────────────────

#[test]
fn protocol_from_alpn_v1() {
    use mesh_client::protocol::protocol_from_alpn;
    assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
}

#[test]
fn protocol_from_alpn_v0() {
    use mesh_client::protocol::protocol_from_alpn;
    assert_eq!(protocol_from_alpn(ALPN_V0), ControlProtocol::JsonV0);
}

#[test]
fn protocol_from_alpn_unknown_defaults_to_v1() {
    use mesh_client::protocol::protocol_from_alpn;
    assert_eq!(
        protocol_from_alpn(b"mesh-llm/999"),
        ControlProtocol::ProtoV1
    );
}

// ── Wire constants sanity ────────────────────────────────────────────────────

#[test]
fn stream_type_constants_are_distinct() {
    let types = [
        STREAM_GOSSIP,
        STREAM_TUNNEL_MAP,
        STREAM_ROUTE_REQUEST,
        STREAM_PEER_DOWN,
        STREAM_PEER_LEAVING,
        STREAM_CONFIG_SUBSCRIBE,
        STREAM_CONFIG_PUSH,
    ];
    let mut seen = std::collections::HashSet::new();
    for t in &types {
        assert!(seen.insert(t), "duplicate stream type constant: {:#04x}", t);
    }
}

#[test]
fn node_protocol_generation_is_one() {
    assert_eq!(NODE_PROTOCOL_GENERATION, 1u32);
}

#[test]
fn max_control_frame_bytes_is_eight_mib() {
    assert_eq!(MAX_CONTROL_FRAME_BYTES, 8 * 1024 * 1024);
}

// ── Control frame encode / decode roundtrip ──────────────────────────────────

fn make_valid_gossip_frame() -> GossipFrame {
    GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    }
}

#[test]
fn gossip_frame_roundtrip() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.sender_id, vec![0u8; 32]);
    assert_eq!(decoded.peers.len(), 1);
    assert_eq!(decoded.peers[0].endpoint_id, vec![0u8; 32]);
    assert_eq!(decoded.peers[0].role, NodeRole::Worker as i32);
}

#[test]
fn gossip_frame_bad_generation_rejected() {
    let mut frame = make_valid_gossip_frame();
    frame.gen = 0;
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("gen=0 gossip frame must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}}, got {err:?}"
    );
}

#[test]
fn gossip_frame_invalid_sender_id_rejected() {
    let mut frame = make_valid_gossip_frame();
    frame.sender_id = vec![0u8; 16]; // 16 bytes instead of 32
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("short sender_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidSenderId { got: 16 }),
        "expected InvalidSenderId{{got:16}}, got {err:?}"
    );
}

#[test]
fn wrong_stream_type_rejected() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("wrong stream type must be rejected");
    assert!(
        matches!(
            err,
            ControlFrameError::WrongStreamType {
                expected: STREAM_TUNNEL_MAP,
                got: STREAM_GOSSIP,
            }
        ),
        "expected WrongStreamType, got {err:?}"
    );
}

// ── PeerDown / PeerLeaving roundtrip ─────────────────────────────────────────

#[test]
fn peer_down_roundtrip() {
    let msg = PeerDown {
        peer_id: vec![0xAB; 32],
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &msg);
    let decoded: PeerDown =
        decode_control_frame(STREAM_PEER_DOWN, &encoded).expect("valid PeerDown must decode");
    assert_eq!(decoded.peer_id, vec![0xAB; 32]);
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
}

#[test]
fn peer_leaving_roundtrip() {
    let msg = PeerLeaving {
        peer_id: vec![0xCD; 32],
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &msg);
    let decoded: PeerLeaving =
        decode_control_frame(STREAM_PEER_LEAVING, &encoded).expect("valid PeerLeaving must decode");
    assert_eq!(decoded.peer_id, vec![0xCD; 32]);
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
}

#[test]
fn peer_down_bad_generation_rejected() {
    let msg = PeerDown {
        peer_id: vec![0x77; 32],
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &msg);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration, got {err:?}"
    );
}

// ── RouteTable roundtrip ─────────────────────────────────────────────────────

#[test]
fn route_table_request_bad_generation_rejected() {
    let req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("RouteTableRequest gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration, got {err:?}"
    );
}

#[test]
fn route_table_bad_generation_rejected() {
    let table = RouteTable {
        entries: vec![],
        mesh_id: None,
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &table);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("RouteTable gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration, got {err:?}"
    );
}

// ── v0 legacy compatibility ───────────────────────────────────────────────────

#[test]
fn decode_legacy_tunnel_map_from_json() {
    // JSON: { "<hex-peer-id>": <port> }
    let peer_bytes = [0x42u8; 32];
    let hex_id = hex::encode(peer_bytes);
    let json = format!("{{\"{hex_id}\": 9337}}");

    let frame = decode_legacy_tunnel_map_frame(json.as_bytes())
        .expect("valid legacy tunnel map JSON must decode");

    assert_eq!(frame.entries.len(), 1);
    assert_eq!(frame.entries[0].target_peer_id, peer_bytes.to_vec());
    assert_eq!(frame.entries[0].tunnel_port, 9337);
}

#[test]
fn decode_legacy_tunnel_map_invalid_hex_ignored() {
    let json = b"{\"notvalidhex\": 9337}";
    let frame = decode_legacy_tunnel_map_frame(json)
        .expect("invalid hex entries should be silently ignored");
    assert_eq!(frame.entries.len(), 0);
}

// ── ControlFrameError Display ────────────────────────────────────────────────

#[test]
fn control_frame_error_display_bad_generation() {
    let err = ControlFrameError::BadGeneration { got: 99 };
    let s = err.to_string();
    assert!(s.contains("99"), "Display must mention the bad gen value");
}

#[test]
fn control_frame_error_implements_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(ControlFrameError::BadGeneration { got: 0 });
    assert!(err.to_string().contains("0"));
}
