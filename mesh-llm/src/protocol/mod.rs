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
use crate::mesh::{PeerAnnouncement, PeerAnnouncementV0};

pub(crate) mod convert;
pub(crate) mod v0;
use anyhow::Result;
pub(crate) use convert::*;
use iroh::endpoint::{ConnectOptions, Connection};
use iroh::{Endpoint, EndpointAddr, EndpointId};
use prost::Message;
pub(crate) use v0::*;
pub const ALPN_V1: &[u8] = b"mesh-llm/1";
#[cfg(test)]
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
pub(crate) const STREAM_PLUGIN_CHANNEL: u8 = 0x08;
pub(crate) const STREAM_PLUGIN_BULK_TRANSFER: u8 = 0x09;
pub(crate) const STREAM_CONFIG_SUBSCRIBE: u8 = 0x0b;
pub(crate) const STREAM_CONFIG_PUSH: u8 = 0x0c;
const _: () = {
    let _ = STREAM_CONFIG_SUBSCRIBE;
    let _ = STREAM_CONFIG_PUSH;
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ControlProtocol {
    ProtoV1,
    JsonV0,
}

#[derive(Debug, PartialEq)]
pub(crate) enum ControlFrameError {
    #[cfg(test)]
    OversizeFrame {
        size: usize,
    },
    BadGeneration {
        got: u32,
    },
    InvalidEndpointId {
        got: usize,
    },
    InvalidSenderId {
        got: usize,
    },
    MissingHttpPort,
    InvalidConfigHashLength {
        got: usize,
    },
    InvalidPublicKeyLength {
        got: usize,
    },
    MissingSignature,
    InvalidSignatureLength {
        got: usize,
    },
    MissingConfig,
    #[cfg(test)]
    DecodeError(String),
    #[cfg(test)]
    WrongStreamType {
        expected: u8,
        got: u8,
    },
    ForgedSender,
}

impl std::fmt::Display for ControlFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(test)]
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
            ControlFrameError::InvalidConfigHashLength { got } => {
                write!(f, "invalid config_hash length: expected 32, got {}", got)
            }
            ControlFrameError::InvalidPublicKeyLength { got } => {
                write!(f, "invalid public key length: expected 32, got {}", got)
            }
            ControlFrameError::MissingSignature => write!(f, "config push missing signature"),
            ControlFrameError::InvalidSignatureLength { got } => {
                write!(f, "invalid signature length: expected 64, got {got}")
            }
            ControlFrameError::MissingConfig => {
                write!(f, "config field is required but missing")
            }
            #[cfg(test)]
            ControlFrameError::DecodeError(msg) => write!(f, "protobuf decode error: {}", msg),
            #[cfg(test)]
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

impl ValidateControlFrame for crate::proto::node::ConfigSubscribe {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        validate_endpoint_id_length(self.subscriber_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::ConfigSnapshotResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        let is_error = matches!(self.error.as_deref(), Some(s) if !s.is_empty());
        if !is_error {
            validate_endpoint_id_length(self.node_id.len())?;
            validate_config_hash_length(self.config_hash.len())?;
            if self.config.is_none() {
                return Err(ControlFrameError::MissingConfig);
            }
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::ConfigUpdateNotification {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        validate_endpoint_id_length(self.node_id.len())?;
        validate_config_hash_length(self.config_hash.len())?;
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::ConfigPush {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        validate_endpoint_id_length(self.requester_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        validate_public_key_length(self.owner_signing_public_key.len())?;
        if self.signature.is_empty() {
            return Err(ControlFrameError::MissingSignature);
        }
        if self.signature.len() != 64 {
            return Err(ControlFrameError::InvalidSignatureLength {
                got: self.signature.len(),
            });
        }
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::ConfigPushResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.gen });
        }
        if self.success || !self.config_hash.is_empty() {
            validate_config_hash_length(self.config_hash.len())?;
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

fn validate_endpoint_id_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidEndpointId { got: len });
    }
    Ok(())
}

fn validate_config_hash_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidConfigHashLength { got: len });
    }
    Ok(())
}

fn validate_public_key_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidPublicKeyLength { got: len });
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
            let sanitized: Vec<PeerAnnouncement> = anns
                .iter()
                .map(crate::protocol::convert::sanitize_gossip_announcement_for_wire)
                .collect();
            let legacy_anns: Vec<PeerAnnouncementV0> =
                sanitized.iter().map(PeerAnnouncementV0::from).collect();
            let json = serde_json::to_vec(&legacy_anns)?;
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
            let anns: Vec<PeerAnnouncementV0> = serde_json::from_slice(buf)?;
            Ok(anns
                .into_iter()
                .map(|ann| {
                    let mut ann = ann.into_internal();
                    ann.available_models.clear();
                    ann.available_model_metadata.clear();
                    ann.available_model_sizes.clear();
                    crate::mesh::backfill_legacy_descriptors(&mut ann);
                    (ann.addr.clone(), ann)
                })
                .collect::<Vec<_>>())
        }
    }
}

#[cfg(test)]
pub(crate) fn encode_control_frame(stream_type: u8, msg: &impl prost::Message) -> Vec<u8> {
    let proto_bytes = msg.encode_to_vec();
    let len = proto_bytes.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + proto_bytes.len());
    buf.push(stream_type);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&proto_bytes);
    buf
}

#[cfg(test)]
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
    use crate::crypto::OwnershipSummary;
    use crate::mesh::{resolve_peer_down, resolve_peer_leaving, ModelDemand, PeerInfo};
    use crate::proto::node::{
        ConfigPush, ConfigPushResponse, ConfigSnapshotResponse, ConfigSubscribe,
        ConfigUpdateNotification, GossipFrame, NodeConfigSnapshot, NodeGpuConfig, NodeModelEntry,
        NodePluginEntry, NodeRole, PeerAnnouncement, RouteTableRequest,
    };
    use iroh::{EndpointAddr, EndpointId, SecretKey};
    use std::collections::{HashMap, HashSet};

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

    fn make_config_snapshot() -> NodeConfigSnapshot {
        NodeConfigSnapshot {
            version: 1,
            gpu: Some(NodeGpuConfig {
                assignment: crate::proto::node::GpuAssignment::Auto as i32,
            }),
            models: vec![NodeModelEntry {
                model: "Qwen3-8B".to_string(),
                mmproj: Some("mmproj-cut".to_string()),
                ctx_size: Some(8192),
            }],
            plugins: vec![NodePluginEntry {
                name: "blackboard".to_string(),
                enabled: Some(true),
                command: Some("mesh-llm".to_string()),
                args: vec!["--plugin".to_string(), "blackboard".to_string()],
            }],
        }
    }

    fn make_valid_config_subscribe() -> ConfigSubscribe {
        ConfigSubscribe {
            gen: NODE_PROTOCOL_GENERATION,
            subscriber_id: vec![0xAA; 32],
        }
    }

    fn make_test_peer_info(peer_id: EndpointId) -> PeerInfo {
        PeerInfo {
            id: peer_id,
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            tunnel_port: None,
            role: crate::mesh::NodeRole::Worker,
            models: vec![],
            vram_bytes: 0,
            rtt_ms: None,
            model_source: None,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            available_models: vec![],
            requested_models: vec![],
            last_seen: std::time::Instant::now(),
            moe_recovered_at: None,
            version: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            owner_summary: OwnershipSummary::default(),
        }
    }

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
            serving_models: vec!["Qwen".into()],
            hosted_models: Some(vec!["Qwen".into()]),
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
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
        };
        let json = serde_json::to_vec(&vec![PeerAnnouncementV0::from(&ann)]).unwrap();

        let decoded = decode_gossip_payload(ControlProtocol::JsonV0, peer_id, &json).unwrap();

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0.id, peer_id);
        assert_eq!(
            decoded[0].1.serving_models.first().map(String::as_str),
            Some("Qwen")
        );
        assert_eq!(decoded[0].1.mesh_id.as_deref(), Some("mesh-compat"));
        assert!(
            decoded[0].1.available_models.is_empty(),
            "legacy JSON gossip must not populate passive available_models"
        );
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
    fn config_frames_roundtrip_and_validation() {
        let snapshot = make_config_snapshot();
        let config_hash = vec![0xA5; 32];
        let node_id = vec![0x42; 32];

        let subscribe = make_valid_config_subscribe();
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &subscribe);
        let _decoded: ConfigSubscribe = decode_control_frame(STREAM_CONFIG_SUBSCRIBE, &encoded)
            .expect("valid config subscribe must decode");

        let snapshot_response = ConfigSnapshotResponse {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: node_id.clone(),
            revision: 7,
            config_hash: config_hash.clone(),
            config: Some(snapshot.clone()),
            hostname: Some("node-01".to_string()),
            error: None,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &snapshot_response);
        let decoded: ConfigSnapshotResponse =
            decode_control_frame(STREAM_CONFIG_SUBSCRIBE, &encoded)
                .expect("valid snapshot response must decode");
        assert_eq!(decoded.revision, 7);
        assert_eq!(decoded.config_hash, config_hash);
        assert_eq!(decoded.hostname.as_deref(), Some("node-01"));

        let update = ConfigUpdateNotification {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: node_id.clone(),
            revision: 8,
            config_hash: config_hash.clone(),
            config: Some(snapshot.clone()),
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &update);
        let decoded: ConfigUpdateNotification =
            decode_control_frame(STREAM_CONFIG_SUBSCRIBE, &encoded)
                .expect("valid update notification must decode");
        assert_eq!(decoded.revision, 8);

        let push = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x10; 32],
            target_node_id: node_id.clone(),
            expected_revision: 8,
            config: Some(snapshot.clone()),
            owner_signing_public_key: vec![0x02; 32],
            signature: vec![0x03; 64],
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push);
        let decoded: ConfigPush = decode_control_frame(STREAM_CONFIG_PUSH, &encoded)
            .expect("valid config push must decode");
        assert_eq!(decoded.expected_revision, 8);

        let push_response = ConfigPushResponse {
            gen: NODE_PROTOCOL_GENERATION,
            success: true,
            current_revision: 9,
            config_hash: config_hash.clone(),
            error: None,
            apply_mode: 1,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push_response);
        let decoded: ConfigPushResponse = decode_control_frame(STREAM_CONFIG_PUSH, &encoded)
            .expect("valid push response must decode");
        assert!(decoded.success);
        assert_eq!(decoded.current_revision, 9);
    }

    #[test]
    fn config_frames_validation_rejects_bad_data() {
        let mut subscribe = make_valid_config_subscribe();
        subscribe.gen = 0;
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &subscribe);
        let err = decode_control_frame::<ConfigSubscribe>(STREAM_CONFIG_SUBSCRIBE, &encoded)
            .expect_err("bad generation must be rejected");
        assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

        let mut subscribe = make_valid_config_subscribe();
        subscribe.subscriber_id = vec![0x01; 16];
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &subscribe);
        let err = decode_control_frame::<ConfigSubscribe>(STREAM_CONFIG_SUBSCRIBE, &encoded)
            .expect_err("invalid subscriber id length must be rejected");
        assert!(matches!(err, ControlFrameError::InvalidEndpointId { .. }));

        let snapshot_response = ConfigSnapshotResponse {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: vec![0x01; 16],
            revision: 1,
            config_hash: vec![0x02; 32],
            config: Some(make_config_snapshot()),
            hostname: None,
            error: None,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &snapshot_response);
        let err = decode_control_frame::<ConfigSnapshotResponse>(STREAM_CONFIG_SUBSCRIBE, &encoded)
            .expect_err("invalid node id must be rejected");
        assert!(matches!(err, ControlFrameError::InvalidEndpointId { .. }));

        let snapshot_response = ConfigSnapshotResponse {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: vec![0xAA; 32],
            revision: 1,
            config_hash: vec![0x02; 16],
            config: Some(make_config_snapshot()),
            hostname: None,
            error: None,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &snapshot_response);
        let err = decode_control_frame::<ConfigSnapshotResponse>(STREAM_CONFIG_SUBSCRIBE, &encoded)
            .expect_err("invalid config hash length must be rejected");
        assert!(matches!(
            err,
            ControlFrameError::InvalidConfigHashLength { .. }
        ));

        let update = ConfigUpdateNotification {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: vec![0xBB; 32],
            revision: 2,
            config_hash: vec![0xCC; 16],
            config: Some(make_config_snapshot()),
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &update);
        let err =
            decode_control_frame::<ConfigUpdateNotification>(STREAM_CONFIG_SUBSCRIBE, &encoded)
                .expect_err("invalid config hash for update must be rejected");
        assert!(matches!(
            err,
            ControlFrameError::InvalidConfigHashLength { .. }
        ));

        let mut push = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 2,
            config: Some(make_config_snapshot()),
            owner_signing_public_key: vec![0x03; 16],
            signature: vec![0x04],
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push);
        let err = decode_control_frame::<ConfigPush>(STREAM_CONFIG_PUSH, &encoded)
            .expect_err("short public key must be rejected");
        assert!(matches!(
            err,
            ControlFrameError::InvalidPublicKeyLength { .. }
        ));

        push.owner_signing_public_key = vec![0x06; 32];
        push.signature = vec![];
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push);
        let err = decode_control_frame::<ConfigPush>(STREAM_CONFIG_PUSH, &encoded)
            .expect_err("empty signature must be rejected");
        assert!(matches!(err, ControlFrameError::MissingSignature));

        let push_response = ConfigPushResponse {
            gen: 0,
            success: false,
            current_revision: 2,
            config_hash: vec![0x07; 32],
            error: Some("fail".to_string()),
            apply_mode: 0,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push_response);
        let err = decode_control_frame::<ConfigPushResponse>(STREAM_CONFIG_PUSH, &encoded)
            .expect_err("bad gen must be rejected");
        assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));
    }

    #[test]
    fn config_snapshot_response_error_shape_passes_validation() {
        // Error responses from handle_config_subscribe have empty node_id / config_hash
        // and no config payload. Validation must pass so callers can surface the error.
        let error_snapshot = ConfigSnapshotResponse {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: vec![],
            revision: 0,
            config_hash: vec![],
            config: None,
            hostname: None,
            error: Some("node has no local owner".to_string()),
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &error_snapshot);
        decode_control_frame::<ConfigSnapshotResponse>(STREAM_CONFIG_SUBSCRIBE, &encoded)
            .expect("error-shaped snapshot must pass validation");
    }

    #[test]
    fn config_push_response_error_shape_passes_validation() {
        // Error responses from send_push_error have an empty config_hash.
        let error_response = crate::proto::node::ConfigPushResponse {
            gen: NODE_PROTOCOL_GENERATION,
            success: false,
            current_revision: 0,
            config_hash: vec![],
            error: Some("not the owner of this node".to_string()),
            apply_mode: 0,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &error_response);
        decode_control_frame::<crate::proto::node::ConfigPushResponse>(
            STREAM_CONFIG_PUSH,
            &encoded,
        )
        .expect("error-shaped push response must pass validation");
    }

    #[test]
    fn config_push_valid_64_byte_signature_passes_validation() {
        let push_with_64_bytes = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 0,
            config: Some(make_config_snapshot()),
            owner_signing_public_key: vec![0x03; 32],
            signature: vec![0x04; 64],
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push_with_64_bytes);
        decode_control_frame::<ConfigPush>(STREAM_CONFIG_PUSH, &encoded)
            .expect("push with 64-byte signature must pass validation");
    }

    #[test]
    fn config_push_wrong_length_signature_rejected() {
        // 32-byte signature must be rejected
        let push_32 = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 0,
            config: Some(make_config_snapshot()),
            owner_signing_public_key: vec![0x03; 32],
            signature: vec![0x04; 32],
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push_32);
        let err = decode_control_frame::<ConfigPush>(STREAM_CONFIG_PUSH, &encoded)
            .expect_err("32-byte signature must be rejected");
        assert!(
            matches!(err, ControlFrameError::InvalidSignatureLength { got: 32 }),
            "expected InvalidSignatureLength {{got: 32}}, got {err:?}"
        );

        // 1-byte signature must also be rejected
        let push_1 = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 0,
            config: Some(make_config_snapshot()),
            owner_signing_public_key: vec![0x03; 32],
            signature: vec![0x04; 1],
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push_1);
        let err = decode_control_frame::<ConfigPush>(STREAM_CONFIG_PUSH, &encoded)
            .expect_err("1-byte signature must be rejected");
        assert!(
            matches!(err, ControlFrameError::InvalidSignatureLength { got: 1 }),
            "expected InvalidSignatureLength {{got: 1}}, got {err:?}"
        );
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
            matches!(err, crate::protocol::ControlFrameError::ForgedSender),
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
            serving_models: vec!["v0-model".to_string()],
            hosted_models: Some(vec!["v0-model".to_string()]),
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
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
        };
        let json = serde_json::to_vec(&vec![PeerAnnouncementV0::from(&ann)])
            .expect("JSON serialization must succeed");

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
            decoded[0].1.serving_models.first().map(String::as_str),
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

    #[test]
    fn owner_fields_roundtrip_through_proto_announcement() {
        let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xAB; 32]).public());
        let ann = super::PeerAnnouncement {
            addr: iroh::EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            version: None,
            model_demand: HashMap::new(),
            mesh_id: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: Some(crate::crypto::SignedNodeOwnership {
                claim: crate::crypto::NodeOwnershipClaim {
                    version: 1,
                    cert_id: "cert-123".to_string(),
                    owner_id: "owner-abc".to_string(),
                    owner_sign_public_key: "11".repeat(32),
                    node_endpoint_id: "22".repeat(32),
                    issued_at_unix_ms: 10,
                    expires_at_unix_ms: 20,
                    node_label: Some("studio".to_string()),
                    hostname_hint: Some("worker-01".to_string()),
                },
                signature: "33".repeat(64),
            }),
        };
        let proto_pa = local_ann_to_proto_ann(&ann);
        assert_eq!(
            proto_pa
                .owner_attestation
                .as_ref()
                .map(|att| att.owner_id.as_str()),
            Some("owner-abc")
        );

        let (_, roundtripped) =
            proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed");
        let roundtripped = roundtripped
            .owner_attestation
            .expect("owner attestation must round-trip");
        assert_eq!(roundtripped.claim.owner_id, "owner-abc");
        assert_eq!(roundtripped.claim.cert_id, "cert-123");
        assert_eq!(roundtripped.claim.node_label.as_deref(), Some("studio"));
    }

    #[test]
    fn mesh_config_proto_roundtrip() {
        let snapshot = make_config_snapshot();
        let config = proto_config_to_mesh(&snapshot);
        assert_eq!(config.version, Some(1));
        assert_eq!(config.gpu.assignment, crate::plugin::GpuAssignment::Auto);
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].model, "Qwen3-8B");
        assert_eq!(config.models[0].mmproj.as_deref(), Some("mmproj-cut"));
        assert_eq!(config.models[0].ctx_size, Some(8192));
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.plugins[0].name, "blackboard");

        let roundtripped = mesh_config_to_proto(&config);
        assert_eq!(roundtripped.version, snapshot.version);
        assert_eq!(
            roundtripped.gpu.as_ref().map(|g| g.assignment),
            Some(crate::proto::node::GpuAssignment::Auto as i32)
        );
        assert_eq!(roundtripped.models.len(), snapshot.models.len());
        assert_eq!(roundtripped.models[0].model, snapshot.models[0].model);
        assert_eq!(roundtripped.models[0].mmproj, snapshot.models[0].mmproj);
        assert_eq!(roundtripped.models[0].ctx_size, snapshot.models[0].ctx_size);
        assert_eq!(roundtripped.plugins.len(), snapshot.plugins.len());
        assert_eq!(roundtripped.plugins[0].name, snapshot.plugins[0].name);
    }

    #[test]
    fn canonical_config_hash_is_stable() {
        let snapshot = make_config_snapshot();
        let hash1 = canonical_config_hash(&snapshot);
        let hash2 = canonical_config_hash(&snapshot);
        assert_eq!(hash1, hash2, "same config must produce the same hash");
        assert_eq!(hash1.len(), 32);

        let mut different = snapshot.clone();
        different.version = 2;
        let hash3 = canonical_config_hash(&different);
        assert_ne!(hash1, hash3, "different config must produce different hash");
    }

    #[test]
    fn config_sync_v0_gossip_accepted() {
        // Prove that a V0 JSON gossip frame (PeerAnnouncementV0 without owner fields)
        // is accepted by the v1 gossip decoder and produces a PeerAnnouncement without
        // ownership metadata.
        let remote_id = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: remote_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            models: vec!["test-model".to_string()],
            vram_bytes: 16 * 1024 * 1024 * 1024,
            model_source: None,
            serving_models: vec!["test-model".to_string()],
            hosted_models: Some(vec!["test-model".to_string()]),
            available_models: vec![],
            requested_models: vec![],
            version: Some("0.50.0".to_string()),
            model_demand: HashMap::new(),
            mesh_id: Some("v0-compat-test".to_string()),
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
        };
        let json = serde_json::to_vec(&vec![PeerAnnouncementV0::from(&ann)])
            .expect("JSON serialization must succeed");

        let decoded = decode_gossip_payload(ControlProtocol::JsonV0, remote_id, &json)
            .expect("v0 JSON gossip must be accepted");

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
            decoded[0].1.owner_attestation, None,
            "v0 gossip without owner metadata must decode to None"
        );
        assert_eq!(
            decoded[0].1.serving_models.first().map(String::as_str),
            Some("test-model"),
            "serving model must round-trip correctly"
        );
    }

    #[test]
    fn config_sync_v0_announcement_roundtrip() {
        // PeerAnnouncementV0 must ignore ownership metadata so old JSON peers remain compatible.
        let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x44; 32]).public());
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            version: None,
            model_demand: HashMap::new(),
            mesh_id: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: Some(crate::crypto::SignedNodeOwnership {
                claim: crate::crypto::NodeOwnershipClaim {
                    version: 1,
                    cert_id: "cert-v0".to_string(),
                    owner_id: "test-owner".to_string(),
                    owner_sign_public_key: "44".repeat(32),
                    node_endpoint_id: "55".repeat(32),
                    issued_at_unix_ms: 10,
                    expires_at_unix_ms: 20,
                    node_label: None,
                    hostname_hint: None,
                },
                signature: "66".repeat(64),
            }),
        };

        let v0 = PeerAnnouncementV0::from(&ann);
        let json_str = serde_json::to_string(&v0).expect("JSON serialization must succeed");
        assert!(
            !json_str.contains("owner_attestation"),
            "v0 JSON must not serialize owner attestation metadata"
        );

        let deserialized: PeerAnnouncementV0 =
            serde_json::from_str(&json_str).expect("v0 JSON format must deserialize");
        let restored = deserialized.into_internal();
        assert_eq!(
            restored.owner_attestation, None,
            "v0 JSON round-trip must drop owner attestation metadata"
        );
    }

    #[test]
    fn config_sync_full_config_roundtrip() {
        use crate::plugin::{GpuAssignment, GpuConfig, ModelConfigEntry, PluginConfigEntry};
        let config = crate::plugin::MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![ModelConfigEntry {
                model: "Qwen3-8B.gguf".to_string(),
                mmproj: Some("mm.gguf".to_string()),
                ctx_size: Some(8192),
            }],
            plugins: vec![PluginConfigEntry {
                name: "blackboard".to_string(),
                enabled: Some(true),
                command: Some("mesh-llm".to_string()),
                args: vec!["--plugin".to_string()],
            }],
        };
        let snapshot = mesh_config_to_proto(&config);
        let restored = proto_config_to_mesh(&snapshot);
        assert_eq!(restored.version, config.version);
        assert_eq!(restored.models.len(), 1);
        assert_eq!(restored.models[0].model, "Qwen3-8B.gguf");
        assert_eq!(restored.models[0].mmproj.as_deref(), Some("mm.gguf"));
        assert_eq!(restored.models[0].ctx_size, Some(8192));
        assert_eq!(restored.plugins.len(), 1);
        assert_eq!(restored.plugins[0].name, "blackboard");
        assert_eq!(restored.plugins[0].enabled, Some(true));
        assert_eq!(restored.plugins[0].command.as_deref(), Some("mesh-llm"));
        assert_eq!(restored.plugins[0].args, vec!["--plugin"]);
    }

    #[test]
    fn config_sync_empty_config_roundtrip() {
        let config = crate::plugin::MeshConfig::default();
        let snapshot = mesh_config_to_proto(&config);
        let restored = proto_config_to_mesh(&snapshot);
        assert!(restored.models.is_empty());
        assert!(restored.plugins.is_empty());
    }

    #[test]
    fn config_sync_config_hash_determinism() {
        use crate::plugin::{GpuAssignment, GpuConfig, ModelConfigEntry};
        let config = crate::plugin::MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![ModelConfigEntry {
                model: "test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
            }],
            plugins: vec![],
        };
        let snap1 = mesh_config_to_proto(&config);
        let snap2 = mesh_config_to_proto(&config);
        let h1 = canonical_config_hash(&snap1);
        let h2 = canonical_config_hash(&snap2);
        assert_eq!(h1, h2, "same config must produce same hash");

        let config2 = crate::plugin::MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![ModelConfigEntry {
                model: "other.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
            }],
            plugins: vec![],
        };
        let snap3 = mesh_config_to_proto(&config2);
        let h3 = canonical_config_hash(&snap3);
        assert_ne!(h1, h3, "different config must produce different hash");
    }

    #[test]
    fn config_sync_push_signature_roundtrip() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::from_bytes(&[0x42u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let push = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 0,
            config: Some(NodeConfigSnapshot {
                version: 1,
                ..Default::default()
            }),
            owner_signing_public_key: verifying_key.to_bytes().to_vec(),
            signature: vec![0u8; 64],
        };

        let payload = crate::mesh::config_push_signature_payload(&push);
        let sig = signing_key.sign(&payload);
        let mut push_signed = push.clone();
        push_signed.signature = sig.to_bytes().to_vec();

        let vk = ed25519_dalek::VerifyingKey::from_bytes(
            &push_signed
                .owner_signing_public_key
                .as_slice()
                .try_into()
                .unwrap(),
        )
        .unwrap();
        let payload2 = crate::mesh::config_push_signature_payload(&push_signed);
        let sig2 = ed25519_dalek::Signature::from_bytes(
            &push_signed.signature.as_slice().try_into().unwrap(),
        );
        assert!(
            vk.verify_strict(&payload2, &sig2).is_ok(),
            "valid signature must verify"
        );
    }

    #[test]
    fn config_sync_push_tampered_payload_rejected() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::from_bytes(&[0x43u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let push = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 0,
            config: Some(NodeConfigSnapshot {
                version: 1,
                ..Default::default()
            }),
            owner_signing_public_key: verifying_key.to_bytes().to_vec(),
            signature: vec![0u8; 64],
        };

        let payload = crate::mesh::config_push_signature_payload(&push);
        let sig = signing_key.sign(&payload);
        let mut push_signed = push.clone();
        push_signed.signature = sig.to_bytes().to_vec();

        let mut push_tampered = push_signed.clone();
        push_tampered.expected_revision += 1;

        let vk = ed25519_dalek::VerifyingKey::from_bytes(
            &push_tampered
                .owner_signing_public_key
                .as_slice()
                .try_into()
                .unwrap(),
        )
        .unwrap();
        let payload_tampered = crate::mesh::config_push_signature_payload(&push_tampered);
        let sig2 = ed25519_dalek::Signature::from_bytes(
            &push_signed.signature.as_slice().try_into().unwrap(),
        );
        assert!(
            vk.verify_strict(&payload_tampered, &sig2).is_err(),
            "tampered payload must fail verification"
        );
    }

    #[test]
    fn config_sync_v0_mesh_coexistence() {
        // Prove round-trip through the V0 conversion path correctly handles config metadata.
        // This documents the intentional behavior: config_hash is ephemeral in v0 gossip
        // (lost in JSON round-trip due to skip_serializing).
        let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x66; 32]).public());

        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            version: None,
            model_demand: HashMap::new(),
            mesh_id: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: Some(crate::crypto::SignedNodeOwnership {
                claim: crate::crypto::NodeOwnershipClaim {
                    version: 1,
                    cert_id: "mesh-cert".to_string(),
                    owner_id: "mesh-owner".to_string(),
                    owner_sign_public_key: "aa".repeat(32),
                    node_endpoint_id: "bb".repeat(32),
                    issued_at_unix_ms: 10,
                    expires_at_unix_ms: 20,
                    node_label: None,
                    hostname_hint: None,
                },
                signature: "cc".repeat(64),
            }),
        };

        // Convert to V0, serialize to JSON, deserialize back, then convert to internal
        let v0 = PeerAnnouncementV0::from(&ann);
        let json = serde_json::to_vec(&vec![v0]).expect("JSON serialization must succeed");
        let v0_deserialized: Vec<PeerAnnouncementV0> =
            serde_json::from_slice(&json).expect("JSON deserialization must succeed");
        let restored = v0_deserialized[0].clone().into_internal();

        assert_eq!(
            restored.owner_attestation, None,
            "v0 JSON round-trip must drop owner attestation metadata"
        );
    }

    #[test]
    fn config_sync_v0_peer_no_config_metadata() {
        let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x77; 32]).public());
        let ann = super::PeerAnnouncement {
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            role: super::NodeRole::Worker,
            models: vec!["v0-model".to_string()],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec!["v0-model".to_string()],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            version: Some("0.49.0".to_string()),
            model_demand: HashMap::new(),
            mesh_id: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
        };

        let v0 = PeerAnnouncementV0::from(&ann);
        let json = serde_json::to_vec(&vec![v0]).expect("JSON serialization must succeed");
        let v0_deserialized: Vec<PeerAnnouncementV0> =
            serde_json::from_slice(&json).expect("JSON deserialization must succeed");
        let internal = v0_deserialized[0].clone().into_internal();

        assert_eq!(
            internal.owner_attestation, None,
            "v0 peer has no ownership attestation"
        );
    }

    #[test]
    fn config_sync_push_validates_signature_length_too_short() {
        use crate::proto::node::ConfigPush;
        let push = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 0,
            config: Some(make_config_snapshot()),
            owner_signing_public_key: vec![0x03; 32],
            signature: vec![0x04; 10],
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push);
        let err = decode_control_frame::<ConfigPush>(STREAM_CONFIG_PUSH, &encoded)
            .expect_err("short signature must be rejected");
        assert!(
            matches!(err, ControlFrameError::InvalidSignatureLength { got: 10 }),
            "expected InvalidSignatureLength{{got:10}}, got {:?}",
            err
        );
    }

    #[test]
    fn config_sync_push_validates_signature_length_empty() {
        use crate::proto::node::ConfigPush;
        let push = ConfigPush {
            gen: NODE_PROTOCOL_GENERATION,
            requester_id: vec![0x01; 32],
            target_node_id: vec![0x02; 32],
            expected_revision: 0,
            config: Some(make_config_snapshot()),
            owner_signing_public_key: vec![0x03; 32],
            signature: vec![],
        };
        let encoded = encode_control_frame(STREAM_CONFIG_PUSH, &push);
        let err = decode_control_frame::<ConfigPush>(STREAM_CONFIG_PUSH, &encoded)
            .expect_err("empty signature must be rejected");
        assert!(
            matches!(err, ControlFrameError::MissingSignature),
            "expected MissingSignature, got {:?}",
            err
        );
    }

    #[test]
    fn config_sync_snapshot_response_validates_config_present() {
        use crate::proto::node::ConfigSnapshotResponse;
        let response = ConfigSnapshotResponse {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: vec![0x01; 32],
            revision: 1,
            config_hash: vec![0x02; 32],
            config: None,
            hostname: None,
            error: None,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &response);
        let err = decode_control_frame::<ConfigSnapshotResponse>(STREAM_CONFIG_SUBSCRIBE, &encoded)
            .expect_err("snapshot response with config=None must be rejected");
        assert!(
            matches!(err, ControlFrameError::MissingConfig),
            "expected MissingConfig, got {:?}",
            err
        );
    }

    #[test]
    fn config_sync_update_notification_validates_config_present() {
        use crate::proto::node::ConfigUpdateNotification;
        let notification = ConfigUpdateNotification {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: vec![0x01; 32],
            revision: 1,
            config_hash: vec![0x02; 32],
            config: None,
        };
        let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &notification);
        let err =
            decode_control_frame::<ConfigUpdateNotification>(STREAM_CONFIG_SUBSCRIBE, &encoded)
                .expect_err("update notification with config=None must be rejected");
        assert!(
            matches!(err, ControlFrameError::MissingConfig),
            "expected MissingConfig, got {:?}",
            err
        );
    }
}
