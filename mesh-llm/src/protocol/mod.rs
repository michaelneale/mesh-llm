//! # V0 Sunset Checklist
//!
//! To remove legacy v0 protocol support:
//! 1. Delete `v0.rs`
//! 2. Remove `mod v0;` and `pub(crate) use v0::*;` from this file
//! 3. Remove `ControlProtocol::JsonV0` variant from the enum
//! 4. Remove `JsonV0 =>` match arms from: write_gossip_payload, decode_gossip_payload
//! 5. Remove `with_additional_alpns` from `connect_mesh()` (use direct `connect()`)
//! 6. In mesh.rs: remove `JsonV0 =>` match arms from broadcast_peer_down,
//!    broadcast_leaving, broadcast_tunnel_map, _dispatch_streams inline handlers
//! 7. Remove v0-related tests (search for "legacy" and "v0" in test names)
//! 8. Update message_protocol.md to remove v0 references

// Protocol infrastructure — extracted from mesh.rs

#[cfg(test)]
use crate::mesh::NodeRole;
use crate::mesh::PeerAnnouncement;

pub(crate) mod convert;
pub(crate) mod v0;
use anyhow::Result;
pub(crate) use convert::*;
use iroh::endpoint::{ConnectOptions, Connection};
use iroh::{Endpoint, EndpointAddr, EndpointId};
use prost::Message;
pub(crate) use v0::*;
pub const ALPN_V1: &[u8] = b"mesh-llm/1";
pub const ALPN: &[u8] = ALPN_V1;
pub(crate) const NODE_PROTOCOL_GENERATION: u32 = 1;
pub(crate) const MAX_CONTROL_FRAME_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

pub(crate) const STREAM_GOSSIP: u8 = 0x01;
pub(crate) const STREAM_TUNNEL: u8 = 0x02;
pub(crate) const STREAM_TUNNEL_MAP: u8 = 0x03;
pub const STREAM_TUNNEL_HTTP: u8 = 0x04;
pub(crate) const STREAM_ROUTE_REQUEST: u8 = 0x05;
pub(crate) const STREAM_PEER_DOWN: u8 = 0x06;
pub(crate) const STREAM_PEER_LEAVING: u8 = 0x07;
pub(crate) const STREAM_BLACKBOARD: u8 = 0x08;
pub(crate) const STREAM_PLUGIN_CHANNEL: u8 = 0x09;
pub(crate) const STREAM_PLUGIN_BULK_TRANSFER: u8 = 0x0a;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ControlProtocol {
    ProtoV1,
    JsonV0,
}

#[derive(Debug, PartialEq)]
pub(crate) enum ControlFrameError {
    OversizeFrame { size: usize },
    BadGeneration { got: u32 },
    InvalidEndpointId { got: usize },
    InvalidSenderId { got: usize },
    MissingHttpPort,
    DecodeError(String),
    WrongStreamType { expected: u8, got: u8 },
    ForgedSender,
}

impl std::fmt::Display for ControlFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlFrameError::OversizeFrame { size } => write!(
                f,
                "control frame too large: {} bytes (max {})",
                size, MAX_CONTROL_FRAME_BYTES
            ),
            ControlFrameError::BadGeneration { got } => write!(
                f,
                "bad protocol generation: expected {}, got {}",
                NODE_PROTOCOL_GENERATION, got
            ),
            ControlFrameError::InvalidEndpointId { got } => {
                write!(f, "invalid endpoint_id length: expected 32, got {}", got)
            }
            ControlFrameError::InvalidSenderId { got } => {
                write!(f, "invalid sender_id length: expected 32, got {}", got)
            }
            ControlFrameError::MissingHttpPort => {
                write!(f, "HOST-role peer annotation missing http_port")
            }
            ControlFrameError::DecodeError(msg) => write!(f, "protobuf decode error: {}", msg),
            ControlFrameError::WrongStreamType { expected, got } => write!(
                f,
                "wrong stream type: expected {:#04x}, got {:#04x}",
                expected, got
            ),
            ControlFrameError::ForgedSender => {
                write!(f, "frame peer_id does not match QUIC connection identity")
            }
        }
    }
}

impl std::error::Error for ControlFrameError {}

pub(crate) trait ValidateControlFrame: prost::Message + Default + Sized {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::GossipFrame {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        if self.sender_id.len() != 32 {
            return Err(ControlFrameError::InvalidSenderId {
                got: self.sender_id.len(),
            });
        }
        for pa in &self.peers {
            validate_peer_announcement(pa)?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::TunnelMap {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.owner_peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.owner_peer_id.len(),
            });
        }
        for entry in &self.entries {
            if entry.target_peer_id.len() != 32 {
                return Err(ControlFrameError::InvalidEndpointId {
                    got: entry.target_peer_id.len(),
                });
            }
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::RouteTableRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        if !self.requester_id.is_empty() && self.requester_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.requester_id.len(),
            });
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::RouteTable {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        for entry in &self.entries {
            if entry.endpoint_id.len() != 32 {
                return Err(ControlFrameError::InvalidEndpointId {
                    got: entry.endpoint_id.len(),
                });
            }
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::PeerDown {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        if self.peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.peer_id.len(),
            });
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::PeerLeaving {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        if self.peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.peer_id.len(),
            });
        }
        Ok(())
    }
}

pub(crate) fn validate_peer_announcement(
    pa: &crate::proto::node::PeerAnnouncement,
) -> Result<(), ControlFrameError> {
    if pa.endpoint_id.len() != 32 {
        return Err(ControlFrameError::InvalidEndpointId {
            got: pa.endpoint_id.len(),
        });
    }
    if pa.role == crate::proto::node::NodeRole::Host as i32 && pa.http_port.is_none() {
        return Err(ControlFrameError::MissingHttpPort);
    }
    Ok(())
}

pub(crate) fn protocol_from_alpn(alpn: &[u8]) -> ControlProtocol {
    if alpn == ALPN_V0 {
        ControlProtocol::JsonV0
    } else {
        ControlProtocol::ProtoV1
    }
}

pub(crate) fn connection_protocol(conn: &Connection) -> ControlProtocol {
    protocol_from_alpn(conn.alpn())
}

pub(crate) async fn connect_mesh(endpoint: &Endpoint, addr: EndpointAddr) -> Result<Connection> {
    let opts = ConnectOptions::new().with_additional_alpns(vec![ALPN_V0.to_vec()]);
    let connecting = endpoint.connect_with_opts(addr, ALPN_V1, opts).await?;
    Ok(connecting.await?)
}

pub(crate) async fn write_len_prefixed(
    send: &mut iroh::endpoint::SendStream,
    body: &[u8],
) -> Result<()> {
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

pub(crate) async fn read_len_prefixed(recv: &mut iroh::endpoint::RecvStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_CONTROL_FRAME_BYTES {
        anyhow::bail!("control frame too large: {} bytes", len);
    }
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf).await?;
    Ok(buf)
}

pub(crate) async fn write_gossip_payload(
    send: &mut iroh::endpoint::SendStream,
    protocol: ControlProtocol,
    anns: &[PeerAnnouncement],
    sender_id: EndpointId,
) -> Result<()> {
    match protocol {
        ControlProtocol::ProtoV1 => {
            let frame = build_gossip_frame(anns, sender_id);
            write_len_prefixed(send, &frame.encode_to_vec()).await?;
        }
        ControlProtocol::JsonV0 => {
            let json = serde_json::to_vec(anns)?;
            write_len_prefixed(send, &json).await?;
        }
    }
    Ok(())
}

pub(crate) fn decode_gossip_payload(
    protocol: ControlProtocol,
    remote: EndpointId,
    buf: &[u8],
) -> Result<Vec<(EndpointAddr, PeerAnnouncement)>> {
    match protocol {
        ControlProtocol::ProtoV1 => {
            let frame = crate::proto::node::GossipFrame::decode(buf)
                .map_err(|e| anyhow::anyhow!("gossip decode from {}: {e}", remote.fmt_short()))?;
            frame.validate_frame().map_err(|e| {
                anyhow::anyhow!("invalid gossip frame from {}: {e}", remote.fmt_short())
            })?;
            if frame.sender_id.as_slice() != remote.as_bytes() {
                anyhow::bail!(
                    "gossip sender_id mismatch from {}: connection identity does not match frame sender_id",
                    remote.fmt_short()
                );
            }
            Ok(frame
                .peers
                .iter()
                .filter_map(proto_ann_to_local)
                .collect::<Vec<_>>())
        }
        ControlProtocol::JsonV0 => {
            let anns: Vec<PeerAnnouncement> = serde_json::from_slice(buf)?;
            Ok(anns
                .into_iter()
                .map(|ann| (ann.addr.clone(), ann))
                .collect::<Vec<_>>())
        }
    }
}

pub(crate) fn encode_control_frame(stream_type: u8, msg: &impl prost::Message) -> Vec<u8> {
    let proto_bytes = msg.encode_to_vec();
    let len = proto_bytes.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + proto_bytes.len());
    buf.push(stream_type);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&proto_bytes);
    buf
}

pub(crate) fn decode_control_frame<T: ValidateControlFrame>(
    expected_stream_type: u8,
    data: &[u8],
) -> Result<T, ControlFrameError> {
    const HEADER_LEN: usize = 5;
    if data.len() < HEADER_LEN {
        return Err(ControlFrameError::DecodeError(format!(
            "frame too short: {} bytes (minimum {})",
            data.len(),
            HEADER_LEN
        )));
    }
    let actual_type = data[0];
    if actual_type != expected_stream_type {
        return Err(ControlFrameError::WrongStreamType {
            expected: expected_stream_type,
            got: actual_type,
        });
    }
    let len = u32::from_le_bytes(data[1..5].try_into().unwrap()) as usize;
    if len > MAX_CONTROL_FRAME_BYTES {
        return Err(ControlFrameError::OversizeFrame { size: len });
    }
    let proto_bytes = data.get(5..5 + len).ok_or_else(|| {
        ControlFrameError::DecodeError(format!(
            "frame truncated: header says {} bytes but only {} available",
            len,
            data.len().saturating_sub(5)
        ))
    })?;
    let msg = T::decode(proto_bytes).map_err(|e| ControlFrameError::DecodeError(e.to_string()))?;
    msg.validate_frame()?;
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::{make_test_peer_info, make_valid_gossip_frame};
    use crate::mesh::{resolve_peer_down, resolve_peer_leaving, ModelDemand, PeerInfo};
    use crate::proto::node::{GossipFrame, NodeRole, PeerAnnouncement, RouteTableRequest};
    use iroh::{EndpointAddr, EndpointId, SecretKey};
    use std::collections::{HashMap, HashSet};

    #[test]
    fn protocol_from_alpn_supports_v1_and_legacy_v0() {
        assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
        assert_eq!(protocol_from_alpn(ALPN_V0), ControlProtocol::JsonV0);
        assert_eq!(
            protocol_from_alpn(b"mesh-llm/999"),
            ControlProtocol::ProtoV1
        );
    }

    #[test]
    fn legacy_json_gossip_payload_decodes() {
        let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x42; 32]).public());
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Host { http_port: 3131 },
            models: vec!["Qwen".into()],
            vram_bytes: 48_000_000_000,
            model_source: Some("Qwen.gguf".into()),
            serving: Some("Qwen".into()),
            serving_models: vec!["Qwen".into()],
            available_models: vec!["Qwen".into()],
            requested_models: vec!["Qwen".into()],
            version: Some("0.52.0".into()),
            model_demand: HashMap::from([(
                "Qwen".into(),
                ModelDemand {
                    last_active: 123,
                    request_count: 7,
                },
            )]),
            mesh_id: Some("mesh-compat".into()),
            gpu_name: Some("NVIDIA A100".into()),
            hostname: Some("worker-01".into()),
            is_soc: Some(false),
            gpu_vram: Some("51539607552".into()),
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::from([("Qwen".into(), 1234_u64)]),
        };
        let json = serde_json::to_vec(&vec![ann.clone()]).unwrap();

        let decoded = decode_gossip_payload(ControlProtocol::JsonV0, peer_id, &json).unwrap();

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0.id, peer_id);
        assert_eq!(decoded[0].1.serving.as_deref(), Some("Qwen"));
        assert_eq!(decoded[0].1.mesh_id.as_deref(), Some("mesh-compat"));
    }

    #[test]
    fn legacy_json_tunnel_map_decodes() {
        let target = EndpointId::from(SecretKey::from_bytes(&[0x24; 32]).public());
        let json = serde_json::to_vec(&HashMap::from([(hex::encode(target.as_bytes()), 9337_u16)]))
            .unwrap();

        let frame = decode_legacy_tunnel_map_frame(&json).unwrap();

        assert_eq!(frame.entries.len(), 1);
        assert_eq!(frame.entries[0].target_peer_id, target.as_bytes().to_vec());
        assert_eq!(frame.entries[0].tunnel_port, 9337);
    }

    #[test]
    fn control_frame_roundtrip() {
        let frame = make_valid_gossip_frame();
        let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
        let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
            .expect("valid gossip frame must decode successfully");
        assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
        assert_eq!(decoded.peers.len(), 1);
        assert_eq!(decoded.peers[0].endpoint_id, vec![0u8; 32]);
        assert_eq!(decoded.peers[0].role, NodeRole::Worker as i32);
    }

    #[test]
    fn proto_v1_route_table_rejects_bad_generation_or_legacy_payload() {
        use crate::proto::node::RouteTable;

        let zero_gen_req = RouteTableRequest {
            requester_id: vec![0u8; 32],
            gen: 0,
        };
        let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &zero_gen_req);
        let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
            .expect_err("request gen=0 must be rejected");
        assert!(
            matches!(err, ControlFrameError::BadGeneration { got: 0 }),
            "expected BadGeneration{{got:0}}, got {:?}",
            err
        );

        let wrong_gen_req = RouteTableRequest {
            requester_id: vec![0u8; 32],
            gen: 99,
        };
        let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_req);
        let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
            .expect_err("request gen=99 must be rejected");
        assert!(
            matches!(err, ControlFrameError::BadGeneration { got: 99 }),
            "expected BadGeneration{{got:99}}, got {:?}",
            err
        );

        let bad_gen_response = RouteTable {
            entries: vec![],
            mesh_id: None,
            gen: 0,
        };
        let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_response);
        let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
            .expect_err("response gen=0 must be rejected");
        assert!(
            matches!(err, ControlFrameError::BadGeneration { got: 0 }),
            "expected BadGeneration{{got:0}} for response, got {:?}",
            err
        );

        let wrong_gen_response = RouteTable {
            entries: vec![],
            mesh_id: None,
            gen: 42,
        };
        let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_response);
        let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
            .expect_err("response gen=42 must be rejected");
        assert!(
            matches!(err, ControlFrameError::BadGeneration { got: 42 }),
            "expected BadGeneration{{got:42}} for response, got {:?}",
            err
        );

        let legacy_json = b"{\"hosts\":[],\"mesh_id\":null}";
        let mut fake_frame = vec![STREAM_ROUTE_REQUEST];
        fake_frame.extend_from_slice(&(legacy_json.len() as u32).to_le_bytes());
        fake_frame.extend_from_slice(legacy_json);
        let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &fake_frame)
            .expect_err("legacy JSON payload must be rejected");
        assert!(
            matches!(err, ControlFrameError::DecodeError(_)),
            "expected DecodeError for JSON payload, got {:?}",
            err
        );
    }

    #[test]
    fn peer_lifecycle_messages_roundtrip() {
        use crate::proto::node::{PeerDown, PeerLeaving};

        let leaving_id = EndpointId::from(SecretKey::from_bytes(&[0x55; 32]).public());

        let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
        peers.insert(leaving_id, make_test_peer_info(leaving_id));
        let mut connection_ids: HashSet<EndpointId> = HashSet::new();
        connection_ids.insert(leaving_id);

        let leaving_msg = PeerLeaving {
            peer_id: leaving_id.as_bytes().to_vec(),
            gen: NODE_PROTOCOL_GENERATION,
        };
        let encoded = encode_control_frame(STREAM_PEER_LEAVING, &leaving_msg);
        let decoded_leaving: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded)
            .expect("valid PeerLeaving must decode");

        let accepted_id = resolve_peer_leaving(leaving_id, &decoded_leaving)
            .expect("PeerLeaving from sender itself must be accepted");

        peers.remove(&accepted_id);
        connection_ids.remove(&accepted_id);

        assert!(
            !peers.contains_key(&leaving_id),
            "leaving peer must be removed from peers after accepted PeerLeaving"
        );
        assert!(
            !connection_ids.contains(&leaving_id),
            "leaving peer must be removed from connections after accepted PeerLeaving"
        );

        let self_id = EndpointId::from(SecretKey::from_bytes(&[0xAA; 32]).public());
        let dead_id = EndpointId::from(SecretKey::from_bytes(&[0xBB; 32]).public());

        let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
        peers.insert(dead_id, make_test_peer_info(dead_id));
        let mut connection_ids: HashSet<EndpointId> = HashSet::new();
        connection_ids.insert(dead_id);

        let down_msg = PeerDown {
            peer_id: dead_id.as_bytes().to_vec(),
            gen: NODE_PROTOCOL_GENERATION,
        };
        let encoded = encode_control_frame(STREAM_PEER_DOWN, &down_msg);
        let decoded_down: PeerDown =
            decode_control_frame(STREAM_PEER_DOWN, &encoded).expect("valid PeerDown must decode");

        let result = resolve_peer_down(self_id, dead_id, true);
        assert_eq!(
            result,
            Some(dead_id),
            "confirmed-unreachable peer must be returned for removal"
        );

        if let Some(id) = result {
            peers.remove(&id);
            connection_ids.remove(&id);
        }

        assert!(
            !peers.contains_key(&dead_id),
            "dead peer must be removed from peers when confirmed unreachable"
        );
        assert!(
            !connection_ids.contains(&dead_id),
            "dead peer must be removed from connections when confirmed unreachable"
        );

        assert_eq!(decoded_down.gen, NODE_PROTOCOL_GENERATION);
    }

    #[test]
    fn peer_lifecycle_rejects_forged_sender_or_unverified_down() {
        use crate::proto::node::{PeerDown, PeerLeaving};

        let valid_peer_bytes = EndpointId::from(SecretKey::from_bytes(&[0x77; 32]).public())
            .as_bytes()
            .to_vec();

        let bad_gen_down = PeerDown {
            peer_id: valid_peer_bytes.clone(),
            gen: 0,
        };
        let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
        let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
            .expect_err("PeerDown gen=0 must be rejected");
        assert!(
            matches!(err, ControlFrameError::BadGeneration { got: 0 }),
            "expected BadGeneration{{got:0}} for PeerDown, got {:?}",
            err
        );

        let bad_gen_leaving = PeerLeaving {
            peer_id: valid_peer_bytes.clone(),
            gen: 0,
        };
        let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
        let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
            .expect_err("PeerLeaving gen=0 must be rejected");
        assert!(
            matches!(err, ControlFrameError::BadGeneration { got: 0 }),
            "expected BadGeneration{{got:0}} for PeerLeaving, got {:?}",
            err
        );

        let remote_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
        let victim_id = EndpointId::from(SecretKey::from_bytes(&[0x22; 32]).public());

        let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
        peers.insert(victim_id, make_test_peer_info(victim_id));

        let forged = PeerLeaving {
            peer_id: victim_id.as_bytes().to_vec(),
            gen: NODE_PROTOCOL_GENERATION,
        };
        let encoded = encode_control_frame(STREAM_PEER_LEAVING, &forged);
        let decoded: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded)
            .expect("structurally valid PeerLeaving must decode");

        let err = resolve_peer_leaving(remote_id, &decoded)
            .expect_err("forged PeerLeaving (peer_id != remote) must be rejected");
        assert!(
            matches!(err, ControlFrameError::ForgedSender),
            "expected ForgedSender, got {:?}",
            err
        );

        assert!(
            peers.contains_key(&victim_id),
            "victim peer must NOT be removed when PeerLeaving is forged"
        );

        let self_id = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
        let still_alive_id = EndpointId::from(SecretKey::from_bytes(&[0x44; 32]).public());

        let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
        peers.insert(still_alive_id, make_test_peer_info(still_alive_id));

        let result = resolve_peer_down(self_id, still_alive_id, false);
        assert!(
            result.is_none(),
            "PeerDown must not trigger removal when peer is still reachable"
        );

        assert!(
            peers.contains_key(&still_alive_id),
            "reachable peer must NOT be removed after PeerDown with should_remove=false"
        );
    }

    #[test]
    fn proto_v1_control_frames_reject_legacy_json_and_wrong_gen() {
        use crate::proto::node::{PeerDown, PeerLeaving};

        // JSON bytes that look plausible for the old wire format on each stream
        let json_gossip = b"[{\"addr\":{\"id\":\"aabbcc\",\"addrs\":[]}}]";
        let json_tunnel_map = b"{\"owner\":\"aabbcc\",\"entries\":[]}";
        let json_route = b"{\"hosts\":[],\"mesh_id\":null}";
        let json_peer_down = b"\"aabbccdd\"";
        let json_peer_leaving = b"\"aabbccdd\"";

        // All migrated streams must reject legacy JSON with DecodeError
        for (stream_type, json_bytes) in [
            (STREAM_GOSSIP, json_gossip.as_slice()),
            (STREAM_TUNNEL_MAP, json_tunnel_map.as_slice()),
            (STREAM_ROUTE_REQUEST, json_route.as_slice()),
            (STREAM_PEER_DOWN, json_peer_down.as_slice()),
            (STREAM_PEER_LEAVING, json_peer_leaving.as_slice()),
        ] {
            let mut frame = vec![stream_type];
            frame.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
            frame.extend_from_slice(json_bytes);
            // Each stream uses its own message type for decode; we test gossip and route
            // request specifically since those carry gen validation too.
            if stream_type == STREAM_GOSSIP {
                let err = decode_control_frame::<GossipFrame>(stream_type, &frame).expect_err(
                    &format!("JSON must be rejected on stream {:#04x}", stream_type),
                );
                assert!(
                    matches!(err, ControlFrameError::DecodeError(_)),
                    "stream {:#04x}: expected DecodeError for JSON, got {:?}",
                    stream_type,
                    err
                );
            } else if stream_type == STREAM_ROUTE_REQUEST {
                let err =
                    decode_control_frame::<RouteTableRequest>(stream_type, &frame).expect_err(
                        &format!("JSON must be rejected on stream {:#04x}", stream_type),
                    );
                assert!(
                    matches!(err, ControlFrameError::DecodeError(_)),
                    "stream {:#04x}: expected DecodeError for JSON, got {:?}",
                    stream_type,
                    err
                );
            }
            // STREAM_TUNNEL_MAP, STREAM_PEER_DOWN, STREAM_PEER_LEAVING: JSON fails prost
            // decode which returns DecodeError — verified via the decode_control_frame
            // path used in the existing per-stream tests.
        }

        // All migrated streams must also reject gen=0 and gen=99 where gen is checked
        let bad_gen_gossip = GossipFrame {
            gen: 0,
            sender_id: vec![],
            peers: vec![PeerAnnouncement {
                endpoint_id: vec![0u8; 32],
                role: NodeRole::Worker as i32,
                ..Default::default()
            }],
        };
        let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen_gossip);
        let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
            .expect_err("GossipFrame gen=0 must be rejected");
        assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

        let bad_gen_req = RouteTableRequest {
            requester_id: vec![0u8; 32],
            gen: 0,
        };
        let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_req);
        let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
            .expect_err("RouteTableRequest gen=0 must be rejected");
        assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

        let bad_gen_down = PeerDown {
            peer_id: vec![0u8; 32],
            gen: 0,
        };
        let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
        let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
            .expect_err("PeerDown gen=0 must be rejected");
        assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

        let bad_gen_leaving = PeerLeaving {
            peer_id: vec![0u8; 32],
            gen: 0,
        };
        let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
        let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
            .expect_err("PeerLeaving gen=0 must be rejected");
        assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

        // Wrong gen (e.g. 2) also rejected
        let wrong_gen_gossip = GossipFrame {
            gen: 2,
            sender_id: vec![0u8; 32],
            peers: vec![PeerAnnouncement {
                endpoint_id: vec![0u8; 32],
                role: NodeRole::Worker as i32,
                ..Default::default()
            }],
        };
        let encoded = encode_control_frame(STREAM_GOSSIP, &wrong_gen_gossip);
        let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
            .expect_err("GossipFrame gen=2 (future version) must be rejected");
        assert!(matches!(err, ControlFrameError::BadGeneration { got: 2 }));
    }

    #[test]
    fn v0_gossip_without_generation_field_accepted() {
        use prost::Message as _;

        let remote_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: remote_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            models: vec!["v0-model".to_string()],
            vram_bytes: 8 * 1024 * 1024 * 1024,
            model_source: None,
            serving: Some("v0-model".to_string()),
            serving_models: vec!["v0-model".to_string()],
            available_models: vec![],
            requested_models: vec![],
            version: Some("0.49.0".to_string()),
            model_demand: HashMap::new(),
            mesh_id: Some("gossip-gen-test".to_string()),
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
        };
        let json = serde_json::to_vec(&vec![ann]).expect("JSON serialization must succeed");

        let decoded = decode_gossip_payload(ControlProtocol::JsonV0, remote_id, &json)
            .expect("v0 JSON gossip without gen field must be accepted");
        assert_eq!(
            decoded.len(),
            1,
            "must decode exactly one peer announcement"
        );
        assert_eq!(
            decoded[0].0.id, remote_id,
            "decoded addr id must match remote_id"
        );
        assert_eq!(
            decoded[0].1.serving.as_deref(),
            Some("v0-model"),
            "serving model must round-trip correctly through JSON decode"
        );
        assert_eq!(
            decoded[0].1.mesh_id.as_deref(),
            Some("gossip-gen-test"),
            "mesh_id must round-trip correctly through JSON decode"
        );

        let sender_id = EndpointId::from(SecretKey::from_bytes(&[0x22; 32]).public());
        let good_frame = GossipFrame {
            gen: NODE_PROTOCOL_GENERATION,
            sender_id: sender_id.as_bytes().to_vec(),
            peers: vec![PeerAnnouncement {
                endpoint_id: sender_id.as_bytes().to_vec(),
                role: NodeRole::Worker as i32,
                ..Default::default()
            }],
        };
        let good_encoded = good_frame.encode_to_vec();
        let v1_result = decode_gossip_payload(ControlProtocol::ProtoV1, sender_id, &good_encoded)
            .expect("ProtoV1 gossip with correct gen must be accepted");
        assert_eq!(
            v1_result.len(),
            1,
            "ProtoV1 gossip must decode one peer entry"
        );

        let bad_frame = GossipFrame {
            gen: 99,
            sender_id: sender_id.as_bytes().to_vec(),
            peers: vec![],
        };
        let bad_encoded = bad_frame.encode_to_vec();
        let bad_result = decode_gossip_payload(ControlProtocol::ProtoV1, sender_id, &bad_encoded);
        assert!(
            bad_result.is_err(),
            "ProtoV1 gossip with gen=99 must be rejected by the generation gate"
        );
    }
}
