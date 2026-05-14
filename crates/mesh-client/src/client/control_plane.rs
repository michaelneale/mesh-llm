use crate::client::builder::MeshClient;
use crate::crypto::OwnerKeypair;
use crate::proto::node::{
    NodeConfigSnapshot, OwnerControlApplyConfigRequest, OwnerControlApplyConfigResponse,
    OwnerControlConfigSnapshot, OwnerControlConfigUpdate, OwnerControlEnvelope, OwnerControlError,
    OwnerControlErrorCode, OwnerControlGetConfigRequest, OwnerControlHandshake,
    OwnerControlRefreshInventoryRequest, OwnerControlRequest, OwnerControlResponse,
    OwnerControlWatchAccepted, OwnerControlWatchConfigRequest, OwnerControlWatchConfigResponse,
    SignedNodeOwnership,
};
use crate::protocol::{
    decode_owner_control_envelope, write_len_prefixed, ALPN_CONTROL_V1, ALPN_V1,
    NODE_PROTOCOL_GENERATION,
};
use anyhow::Context;
use base64::Engine;
use iroh::{Endpoint, EndpointAddr};
use prost::Message;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

const DEFAULT_NODE_CERT_LIFETIME_SECS: u64 = 7 * 24 * 60 * 60;
const NODE_OWNERSHIP_VERSION: u32 = 1;
const SIGNING_DOMAIN_TAG: &[u8] = b"mesh-llm-node-ownership-v1:";
const OWNER_CONTROL_CONNECT_TIMEOUT_SECS: u64 = 8;

fn owner_control_client_bind_addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([0, 0, 0, 0], 0))
}

/// Explicit owner-control bootstrap policy for new config clients.
///
/// Negotiation matrix:
/// - new client + explicit control endpoint -> use `mesh-llm-control/1`; configured control
///   failures stay on the control lane and return structured errors.
/// - new client + no control endpoint -> return `ControlEndpointRequired`.
///
/// Config and inventory mutation is intentionally exclusive to `mesh-llm-control/1`.
/// The legacy mesh-plane config stream IDs remain reserved, but no client bootstrap path
/// falls back to them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ControlPlaneBootstrapOptions {
    control_endpoint: Option<String>,
}

impl ControlPlaneBootstrapOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_control_endpoint(mut self, control_endpoint: impl Into<String>) -> Self {
        self.control_endpoint = Some(control_endpoint.into());
        self
    }

    pub fn control_endpoint(&self) -> Option<&str> {
        self.control_endpoint.as_deref()
    }

    pub fn select_transport(
        &self,
    ) -> Result<ConfigTransportSelection, ControlPlaneNegotiationError> {
        match self.control_endpoint() {
            Some(endpoint) => Ok(ConfigTransportSelection::OwnerControl {
                endpoint: endpoint.to_string(),
                retry_policy: ControlPlaneRetryPolicy::NoSilentLegacyDowngrade,
            }),
            None => Err(ControlPlaneNegotiationError::endpoint_required()),
        }
    }

    pub fn configured_endpoint_failure(
        &self,
        code: OwnerControlErrorCode,
        message: impl Into<String>,
    ) -> ControlPlaneNegotiationError {
        debug_assert!(self.control_endpoint.is_some());
        ControlPlaneNegotiationError::structured(code, message, false)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigTransportSelection {
    OwnerControl {
        endpoint: String,
        retry_policy: ControlPlaneRetryPolicy,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlPlaneRetryPolicy {
    NoSilentLegacyDowngrade,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlPlaneNegotiationError {
    pub code: OwnerControlErrorCode,
    pub message: String,
    pub legacy_retry_allowed: bool,
}

impl fmt::Display for ControlPlaneNegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for ControlPlaneNegotiationError {}

impl ControlPlaneNegotiationError {
    pub fn endpoint_required() -> Self {
        Self {
            code: OwnerControlErrorCode::ControlEndpointRequired,
            message: "owner-control endpoint must be provided explicitly".to_string(),
            legacy_retry_allowed: false,
        }
    }

    pub fn structured(
        code: OwnerControlErrorCode,
        message: impl Into<String>,
        legacy_retry_allowed: bool,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            legacy_retry_allowed,
        }
    }
}

#[derive(Debug, Error)]
pub enum ControlPlaneClientError {
    #[error(transparent)]
    Negotiation(#[from] ControlPlaneNegotiationError),
    #[error(transparent)]
    Remote(#[from] OwnerControlRemoteError),
    #[error("control transport error: {0}")]
    Transport(String),
    #[error("control protocol error: {0}")]
    Protocol(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerControlRemoteError {
    pub code: OwnerControlErrorCode,
    pub message: String,
    pub request_id: Option<u64>,
    pub current_revision: Option<u64>,
}

impl fmt::Display for OwnerControlRemoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for OwnerControlRemoteError {}

impl From<OwnerControlError> for OwnerControlRemoteError {
    fn from(error: OwnerControlError) -> Self {
        Self {
            code: OwnerControlErrorCode::try_from(error.code)
                .unwrap_or(OwnerControlErrorCode::BadRequest),
            message: error.message,
            request_id: error.request_id,
            current_revision: error.current_revision,
        }
    }
}

/// Control-plane bootstrap is explicit and out-of-band.
///
/// Callers either receive an owner-control session bound to a configured endpoint,
/// or a structured error. The client never performs a silent downgrade.
pub enum ControlPlaneConnection {
    OwnerControl(Box<OwnerControlClient>),
}

pub struct OwnerControlClient {
    endpoint_token: String,
    endpoint: Endpoint,
    connection: iroh::endpoint::Connection,
    owner_keypair: OwnerKeypair,
    next_request_id: AtomicU64,
}

pub struct OwnerControlWatchStream {
    send: iroh::endpoint::SendStream,
    recv: iroh::endpoint::RecvStream,
    request_id: u64,
    closed: bool,
}

pub enum OwnerControlWatchEvent {
    Accepted(OwnerControlWatchAccepted),
    Snapshot(OwnerControlConfigSnapshot),
    Update(OwnerControlConfigUpdate),
}

impl MeshClient {
    /// Bootstrap config transport using the explicit owner-control endpoint policy.
    ///
    /// Owner-control endpoints are not discovered through gossip or status APIs;
    /// callers must provide them explicitly through out-of-band bootstrap.
    pub async fn connect_control_plane(
        &self,
        options: ControlPlaneBootstrapOptions,
    ) -> Result<ControlPlaneConnection, ControlPlaneClientError> {
        match options.select_transport()? {
            ConfigTransportSelection::OwnerControl { endpoint, .. } => {
                OwnerControlClient::connect(endpoint, self.config.owner_keypair.clone(), &options)
                    .await
                    .map(Box::new)
                    .map(ControlPlaneConnection::OwnerControl)
            }
        }
    }
}

impl OwnerControlClient {
    async fn connect(
        endpoint_token: String,
        owner_keypair: OwnerKeypair,
        options: &ControlPlaneBootstrapOptions,
    ) -> Result<Self, ControlPlaneClientError> {
        let control_addr = decode_endpoint_addr_token(&endpoint_token).map_err(|error| {
            ControlPlaneClientError::Negotiation(options.configured_endpoint_failure(
                OwnerControlErrorCode::ControlUnavailable,
                format!("invalid owner-control endpoint token: {error}"),
            ))
        })?;
        let mut builder = Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(iroh::SecretKey::generate())
            .alpns(vec![ALPN_CONTROL_V1.to_vec()])
            .bind_addr(owner_control_client_bind_addr())
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        builder = builder.relay_mode(relay_mode_from_endpoint_addr(&control_addr));
        let endpoint = builder
            .bind()
            .await
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        if control_addr.relay_urls().next().is_some() {
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(OWNER_CONTROL_CONNECT_TIMEOUT_SECS),
                endpoint.online(),
            )
            .await;
        }
        let connection = match tokio::time::timeout(
            std::time::Duration::from_secs(OWNER_CONTROL_CONNECT_TIMEOUT_SECS),
            endpoint.connect(control_addr.clone(), ALPN_CONTROL_V1),
        )
        .await
        {
            Ok(Ok(connection)) => connection,
            Ok(Err(error)) => {
                return Err(configured_endpoint_connect_error(
                    &endpoint,
                    control_addr,
                    options,
                    error,
                )
                .await)
            }
            Err(_) => {
                return Err(ControlPlaneClientError::Negotiation(options.configured_endpoint_failure(
                    OwnerControlErrorCode::ControlUnavailable,
                    format!(
                        "remote owner-control endpoint is unavailable or unreachable: connect timed out after {OWNER_CONTROL_CONNECT_TIMEOUT_SECS}s"
                    ),
                )));
            }
        };
        Ok(Self {
            endpoint_token,
            endpoint,
            connection,
            owner_keypair,
            next_request_id: AtomicU64::new(1),
        })
    }

    pub fn endpoint_token(&self) -> &str {
        &self.endpoint_token
    }

    pub fn local_node_id(&self) -> [u8; 32] {
        *self.endpoint.id().as_bytes()
    }

    pub fn target_node_id(&self) -> [u8; 32] {
        *self.connection.remote_id().as_bytes()
    }

    pub async fn get_config(&self) -> Result<OwnerControlConfigSnapshot, ControlPlaneClientError> {
        let response = self
            .send_unary_request(|request_id, requester_node_id, target_node_id| {
                OwnerControlRequest {
                    request_id,
                    get_config: Some(OwnerControlGetConfigRequest {
                        requester_node_id,
                        target_node_id,
                    }),
                    watch_config: None,
                    apply_config: None,
                    refresh_inventory: None,
                }
            })
            .await?;
        response
            .get_config
            .and_then(|response| response.snapshot)
            .ok_or_else(|| {
                ControlPlaneClientError::Protocol(
                    "owner-control get_config response missing snapshot payload".to_string(),
                )
            })
    }

    pub async fn apply_config(
        &self,
        expected_revision: u64,
        config: NodeConfigSnapshot,
    ) -> Result<OwnerControlApplyConfigResponse, ControlPlaneClientError> {
        let response = self
            .send_unary_request(|request_id, requester_node_id, target_node_id| {
                OwnerControlRequest {
                    request_id,
                    get_config: None,
                    watch_config: None,
                    apply_config: Some(OwnerControlApplyConfigRequest {
                        requester_node_id,
                        target_node_id,
                        expected_revision,
                        config: Some(config),
                    }),
                    refresh_inventory: None,
                }
            })
            .await?;
        response.apply_config.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control apply_config response missing apply payload".to_string(),
            )
        })
    }

    pub async fn refresh_inventory(
        &self,
    ) -> Result<OwnerControlConfigSnapshot, ControlPlaneClientError> {
        let response = self
            .send_unary_request(|request_id, requester_node_id, target_node_id| {
                OwnerControlRequest {
                    request_id,
                    get_config: None,
                    watch_config: None,
                    apply_config: None,
                    refresh_inventory: Some(OwnerControlRefreshInventoryRequest {
                        requester_node_id,
                        target_node_id,
                    }),
                }
            })
            .await?;
        response
            .refresh_inventory
            .and_then(|response| response.snapshot)
            .ok_or_else(|| {
                ControlPlaneClientError::Protocol(
                    "owner-control refresh_inventory response missing snapshot payload".to_string(),
                )
            })
    }

    pub async fn watch_config(
        &self,
        include_snapshot: bool,
    ) -> Result<OwnerControlWatchStream, ControlPlaneClientError> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (mut send, recv) = self.open_authenticated_stream().await?;
        let envelope = OwnerControlEnvelope {
            gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id,
                get_config: None,
                watch_config: Some(OwnerControlWatchConfigRequest {
                    requester_node_id: self.endpoint.id().as_bytes().to_vec(),
                    target_node_id: self.connection.remote_id().as_bytes().to_vec(),
                    include_snapshot,
                }),
                apply_config: None,
                refresh_inventory: None,
            }),
            response: None,
            error: None,
        };
        write_len_prefixed(&mut send, &envelope.encode_to_vec())
            .await
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        Ok(OwnerControlWatchStream {
            send,
            recv,
            request_id,
            closed: false,
        })
    }

    async fn send_unary_request<F>(
        &self,
        build_request: F,
    ) -> Result<OwnerControlResponse, ControlPlaneClientError>
    where
        F: FnOnce(u64, Vec<u8>, Vec<u8>) -> OwnerControlRequest,
    {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (mut send, mut recv) = self.open_authenticated_stream().await?;
        let envelope = OwnerControlEnvelope {
            gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(build_request(
                request_id,
                self.endpoint.id().as_bytes().to_vec(),
                self.connection.remote_id().as_bytes().to_vec(),
            )),
            response: None,
            error: None,
        };
        write_len_prefixed(&mut send, &envelope.encode_to_vec())
            .await
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        let envelope = read_owner_control_message(&mut recv).await?;
        let _ = send.finish();
        decode_response_envelope(request_id, envelope)
    }

    async fn open_authenticated_stream(
        &self,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream), ControlPlaneClientError>
    {
        let (mut send, recv) = self
            .connection
            .open_bi()
            .await
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        let handshake = OwnerControlEnvelope {
            gen: NODE_PROTOCOL_GENERATION,
            handshake: Some(OwnerControlHandshake {
                ownership: Some(sign_node_ownership_proto(
                    &self.owner_keypair,
                    self.endpoint.id().as_bytes(),
                )),
            }),
            request: None,
            response: None,
            error: None,
        };
        write_len_prefixed(&mut send, &handshake.encode_to_vec())
            .await
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        Ok((send, recv))
    }
}

impl OwnerControlWatchStream {
    pub fn request_id(&self) -> u64 {
        self.request_id
    }

    pub async fn next(&mut self) -> Result<OwnerControlWatchEvent, ControlPlaneClientError> {
        let envelope = read_owner_control_message(&mut self.recv).await?;
        let response = decode_response_envelope(self.request_id, envelope)?;
        let watch = response.watch_config.ok_or_else(|| {
            ControlPlaneClientError::Protocol(
                "owner-control watch response missing watch_config payload".to_string(),
            )
        })?;
        decode_watch_event(watch)
    }

    pub async fn close(&mut self) -> Result<(), ControlPlaneClientError> {
        if self.closed {
            return Ok(());
        }
        self.send
            .finish()
            .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
        self.closed = true;
        Ok(())
    }

    pub async fn cancel(&mut self) -> Result<(), ControlPlaneClientError> {
        self.close().await
    }
}

impl Drop for OwnerControlWatchStream {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.send.finish();
            self.closed = true;
        }
    }
}

fn decode_watch_event(
    watch: OwnerControlWatchConfigResponse,
) -> Result<OwnerControlWatchEvent, ControlPlaneClientError> {
    if let Some(accepted) = watch.accepted {
        return Ok(OwnerControlWatchEvent::Accepted(accepted));
    }
    if let Some(snapshot) = watch.snapshot {
        return Ok(OwnerControlWatchEvent::Snapshot(snapshot));
    }
    if let Some(update) = watch.update {
        return Ok(OwnerControlWatchEvent::Update(update));
    }
    Err(ControlPlaneClientError::Protocol(
        "owner-control watch response missing accepted/snapshot/update payload".to_string(),
    ))
}

fn decode_response_envelope(
    expected_request_id: u64,
    envelope: OwnerControlEnvelope,
) -> Result<OwnerControlResponse, ControlPlaneClientError> {
    if let Some(error) = envelope.error {
        return Err(ControlPlaneClientError::Remote(error.into()));
    }
    let response = envelope.response.ok_or_else(|| {
        ControlPlaneClientError::Protocol(
            "owner-control response envelope missing response payload".to_string(),
        )
    })?;
    if response.request_id != expected_request_id {
        return Err(ControlPlaneClientError::Protocol(format!(
            "owner-control response request_id mismatch: expected {expected_request_id}, got {}",
            response.request_id
        )));
    }
    Ok(response)
}

async fn read_owner_control_message(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<OwnerControlEnvelope, ControlPlaneClientError> {
    let bytes = crate::protocol::read_len_prefixed(recv)
        .await
        .map_err(|error| ControlPlaneClientError::Transport(error.to_string()))?;
    decode_owner_control_envelope(&bytes)
        .map_err(|error| ControlPlaneClientError::Protocol(error.to_string()))
}

async fn configured_endpoint_connect_error(
    endpoint: &Endpoint,
    control_addr: EndpointAddr,
    options: &ControlPlaneBootstrapOptions,
    error: iroh::endpoint::ConnectError,
) -> ControlPlaneClientError {
    let message = error.to_string();
    let legacy_mesh_reachable = legacy_mesh_probe(endpoint, control_addr).await;
    let (code, rendered) = if legacy_mesh_reachable || is_alpn_mismatch_message(&message) {
        (
            OwnerControlErrorCode::ControlUnsupported,
            format!("remote endpoint did not negotiate mesh-llm-control/1: {message}"),
        )
    } else {
        (
            OwnerControlErrorCode::ControlUnavailable,
            format!("remote owner-control endpoint is unavailable or unreachable: {message}"),
        )
    };
    ControlPlaneClientError::Negotiation(options.configured_endpoint_failure(code, rendered))
}

async fn legacy_mesh_probe(_endpoint: &Endpoint, control_addr: EndpointAddr) -> bool {
    let Ok(probe_endpoint) = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_V1.to_vec()])
        .relay_mode(relay_mode_from_endpoint_addr(&control_addr))
        .bind_addr(owner_control_client_bind_addr())
    else {
        return false;
    };
    let Ok(probe_endpoint) = probe_endpoint.bind().await else {
        return false;
    };
    if control_addr.relay_urls().next().is_some() {
        let _ =
            tokio::time::timeout(std::time::Duration::from_secs(3), probe_endpoint.online()).await;
    }
    match tokio::time::timeout(
        std::time::Duration::from_secs(3),
        probe_endpoint.connect(control_addr, ALPN_V1),
    )
    .await
    {
        Ok(Ok(connection)) => {
            drop(connection);
            true
        }
        _ => false,
    }
}

fn relay_mode_from_endpoint_addr(addr: &EndpointAddr) -> iroh::endpoint::RelayMode {
    match relay_map_from_endpoint_addr(addr) {
        Some(relay_map) => iroh::endpoint::RelayMode::Custom(relay_map),
        None => iroh::endpoint::RelayMode::Disabled,
    }
}

fn is_alpn_mismatch_message(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("alpn")
        || lowered.contains("application protocol")
        || lowered.contains("no application protocol")
}

fn decode_endpoint_addr_token(invite_token: &str) -> anyhow::Result<EndpointAddr> {
    let json = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(invite_token)
        .context("invalid endpoint encoding")?;
    serde_json::from_slice(&json).context("invalid endpoint JSON")
}

fn relay_map_from_endpoint_addr(addr: &EndpointAddr) -> Option<iroh::RelayMap> {
    let configs: Vec<_> = addr
        .relay_urls()
        .cloned()
        .map(|url| iroh::RelayConfig::new(url, None))
        .collect();
    if configs.is_empty() {
        None
    } else {
        Some(iroh::RelayMap::from_iter(configs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn owner_control_client_binds_wildcard_for_direct_remote_endpoints() {
        let bind_addr = owner_control_client_bind_addr();

        assert_eq!(bind_addr.port(), 0);
        assert!(
            bind_addr.ip().is_unspecified(),
            "owner-control clients must not be loopback-bound when dialing explicit remote endpoints"
        );
    }

    #[test]
    fn relay_mode_uses_custom_relays_from_endpoint_addr() {
        let addr = EndpointAddr::new(iroh::SecretKey::generate().public()).with_relay_url(
            iroh::RelayUrl::from_str("https://relay.example.com").expect("relay URL parses"),
        );

        assert!(matches!(
            relay_mode_from_endpoint_addr(&addr),
            iroh::endpoint::RelayMode::Custom(_)
        ));
    }

    #[test]
    fn relay_mode_is_disabled_without_endpoint_relays() {
        let addr = EndpointAddr::new(iroh::SecretKey::generate().public());

        assert!(matches!(
            relay_mode_from_endpoint_addr(&addr),
            iroh::endpoint::RelayMode::Disabled
        ));
    }
}

fn sign_node_ownership_proto(
    owner: &OwnerKeypair,
    node_endpoint_id: &[u8; 32],
) -> SignedNodeOwnership {
    let issued_at_unix_ms = current_time_unix_ms();
    let expires_at_unix_ms =
        issued_at_unix_ms + DEFAULT_NODE_CERT_LIFETIME_SECS.saturating_mul(1000);
    let cert_id = uuid::Uuid::new_v4().simple().to_string();
    let owner_sign_public_key = owner.verifying_key().as_bytes().to_vec();
    let owner_id = owner.owner_id();
    let signature_payload = canonical_claim_bytes(CanonicalClaim {
        version: NODE_OWNERSHIP_VERSION,
        cert_id: &cert_id,
        owner_id: &owner_id,
        owner_sign_public_key: &owner_sign_public_key,
        node_endpoint_id,
        issued_at_unix_ms,
        expires_at_unix_ms,
        node_label: None,
        hostname_hint: None,
    });
    SignedNodeOwnership {
        version: NODE_OWNERSHIP_VERSION,
        cert_id,
        owner_id,
        owner_sign_public_key,
        node_endpoint_id: node_endpoint_id.to_vec(),
        issued_at_unix_ms,
        expires_at_unix_ms,
        node_label: None,
        hostname_hint: None,
        signature: owner.sign_bytes(&signature_payload).to_vec(),
    }
}

struct CanonicalClaim<'a> {
    version: u32,
    cert_id: &'a str,
    owner_id: &'a str,
    owner_sign_public_key: &'a [u8],
    node_endpoint_id: &'a [u8; 32],
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    node_label: Option<&'a str>,
    hostname_hint: Option<&'a str>,
}

fn canonical_claim_bytes(claim: CanonicalClaim<'_>) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(SIGNING_DOMAIN_TAG);
    buf.extend_from_slice(&claim.version.to_le_bytes());
    write_string(&mut buf, claim.cert_id);
    write_string(&mut buf, claim.owner_id);
    buf.extend_from_slice(claim.owner_sign_public_key);
    buf.extend_from_slice(claim.node_endpoint_id);
    buf.extend_from_slice(&claim.issued_at_unix_ms.to_le_bytes());
    buf.extend_from_slice(&claim.expires_at_unix_ms.to_le_bytes());
    write_optional_string(&mut buf, claim.node_label);
    write_optional_string(&mut buf, claim.hostname_hint);
    buf
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_optional_string(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            buf.push(1);
            write_string(buf, value);
        }
        None => buf.push(0),
    }
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
