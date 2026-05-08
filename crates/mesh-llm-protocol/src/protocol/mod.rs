pub mod convert;
pub mod v0;
use anyhow::Result;
pub use convert::*;
use iroh::endpoint::{ConnectOptions, Connection};
use iroh::{Endpoint, EndpointAddr};
pub use v0::*;
pub const ALPN_V1: &[u8] = b"mesh-llm/1";
pub const NODE_PROTOCOL_GENERATION: u32 = 1;
pub const MAX_CONTROL_FRAME_BYTES: usize = 8 * 1024 * 1024;

pub const STREAM_GOSSIP: u8 = 0x01;
pub const STREAM_TUNNEL: u8 = 0x02;
pub const STREAM_TUNNEL_MAP: u8 = 0x03;
pub const STREAM_TUNNEL_HTTP: u8 = 0x04;
pub const STREAM_ROUTE_REQUEST: u8 = 0x05;
pub const STREAM_PEER_DOWN: u8 = 0x06;
pub const STREAM_PEER_LEAVING: u8 = 0x07;
pub const STREAM_PLUGIN_CHANNEL: u8 = 0x08;
pub const STREAM_PLUGIN_BULK_TRANSFER: u8 = 0x09;
pub const STREAM_CONFIG_SUBSCRIBE: u8 = 0x0b;
pub const STREAM_CONFIG_PUSH: u8 = 0x0c;
const _: () = {
    let _ = STREAM_CONFIG_SUBSCRIBE;
    let _ = STREAM_CONFIG_PUSH;
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlProtocol {
    ProtoV1,
    JsonV0,
}

#[derive(Debug, PartialEq)]
pub enum ControlFrameError {
    OversizeFrame { size: usize },
    BadGeneration { got: u32 },
    InvalidEndpointId { got: usize },
    InvalidSenderId { got: usize },
    MissingHttpPort,
    MissingOwnerId,
    InvalidConfigHashLength { got: usize },
    InvalidPublicKeyLength { got: usize },
    MissingSignature,
    InvalidSignatureLength { got: usize },
    MissingConfig,
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
            ControlFrameError::MissingOwnerId => write!(f, "config frame missing owner_id"),
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

pub trait ValidateControlFrame: prost::Message + Default + Sized {
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
        if self.owner_id.is_empty() {
            return Err(ControlFrameError::MissingOwnerId);
        }
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
            if self.owner_id.is_empty() {
                return Err(ControlFrameError::MissingOwnerId);
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
        if self.owner_id.is_empty() {
            return Err(ControlFrameError::MissingOwnerId);
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
        if self.owner_id.is_empty() {
            return Err(ControlFrameError::MissingOwnerId);
        }
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

pub fn validate_peer_announcement(
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

pub fn protocol_from_alpn(alpn: &[u8]) -> ControlProtocol {
    if alpn == ALPN_V0 {
        ControlProtocol::JsonV0
    } else {
        ControlProtocol::ProtoV1
    }
}

pub fn connection_protocol(conn: &Connection) -> ControlProtocol {
    protocol_from_alpn(conn.alpn())
}

pub async fn connect_mesh(endpoint: &Endpoint, addr: EndpointAddr) -> Result<Connection> {
    let opts = ConnectOptions::new().with_additional_alpns(vec![ALPN_V0.to_vec()]);
    let connecting = endpoint.connect_with_opts(addr, ALPN_V1, opts).await?;
    Ok(connecting.await?)
}

pub async fn write_len_prefixed(send: &mut iroh::endpoint::SendStream, body: &[u8]) -> Result<()> {
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

pub async fn read_len_prefixed(recv: &mut iroh::endpoint::RecvStream) -> Result<Vec<u8>> {
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

pub fn encode_control_frame(stream_type: u8, msg: &impl prost::Message) -> Vec<u8> {
    let proto_bytes = msg.encode_to_vec();
    let len = proto_bytes.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + proto_bytes.len());
    buf.push(stream_type);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&proto_bytes);
    buf
}

pub fn decode_control_frame<T: ValidateControlFrame>(
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
