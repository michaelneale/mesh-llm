use std::path::{Path, PathBuf};

use clap::ValueEnum;
use ed25519_dalek::Signer;
use serde::{Deserialize, Serialize};

use super::error::CryptoError;
use super::keys::{owner_id_from_verifying_key, OwnerKeypair};
use super::keystore::write_keystore_bytes_atomically;

pub const NODE_OWNERSHIP_VERSION: u32 = 1;
pub const TRUST_STORE_VERSION: u32 = 1;
pub const DEFAULT_NODE_CERT_LIFETIME_SECS: u64 = 7 * 24 * 60 * 60;
pub const DEFAULT_NODE_CERT_RENEW_WINDOW_SECS: u64 = 36 * 60 * 60;
const SIGNING_DOMAIN_TAG: &[u8] = b"mesh-llm-node-ownership-v1:";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum, Default)]
#[serde(rename_all = "kebab-case")]
pub enum TrustPolicy {
    #[default]
    Off,
    PreferOwned,
    RequireOwned,
    Allowlist,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeOwnershipClaim {
    pub version: u32,
    pub cert_id: String,
    pub owner_id: String,
    pub owner_sign_public_key: String,
    pub node_endpoint_id: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedNodeOwnership {
    pub claim: NodeOwnershipClaim,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrustedOwner {
    pub owner_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RevokedOwner {
    pub owner_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RevokedNodeCert {
    pub cert_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RevokedNodeId {
    pub node_endpoint_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrustStore {
    pub version: u32,
    pub policy: TrustPolicy,
    #[serde(default)]
    pub trusted_owners: Vec<TrustedOwner>,
    #[serde(default)]
    pub revoked_owners: Vec<RevokedOwner>,
    #[serde(default)]
    pub revoked_node_certs: Vec<RevokedNodeCert>,
    #[serde(default)]
    pub revoked_node_ids: Vec<RevokedNodeId>,
}

impl Default for TrustStore {
    fn default() -> Self {
        Self {
            version: TRUST_STORE_VERSION,
            policy: TrustPolicy::Off,
            trusted_owners: Vec::new(),
            revoked_owners: Vec::new(),
            revoked_node_certs: Vec::new(),
            revoked_node_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OwnershipStatus {
    Verified,
    #[default]
    Unsigned,
    Expired,
    InvalidSignature,
    MismatchedNodeId,
    RevokedOwner,
    RevokedCert,
    RevokedNodeId,
    UnsupportedProtocol,
    UntrustedOwner,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct OwnershipSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cert_id: Option<String>,
    pub status: OwnershipStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname_hint: Option<String>,
    pub verified: bool,
}

fn mesh_dir() -> Result<PathBuf, CryptoError> {
    let home = dirs::home_dir().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine home directory",
        ))
    })?;
    Ok(home.join(".mesh-llm"))
}

pub fn default_node_ownership_path() -> Result<PathBuf, CryptoError> {
    Ok(mesh_dir()?.join("node-ownership.json"))
}

pub fn default_trust_store_path() -> Result<PathBuf, CryptoError> {
    Ok(mesh_dir()?.join("trusted-owners.json"))
}

pub fn load_node_ownership(path: &Path) -> Result<SignedNodeOwnership, CryptoError> {
    let raw = std::fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(Into::into)
}

pub fn save_node_ownership(
    path: &Path,
    ownership: &SignedNodeOwnership,
) -> Result<(), CryptoError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(ownership)?;
    write_keystore_bytes_atomically(path, &bytes)?;
    Ok(())
}

pub fn load_trust_store(path: &Path) -> Result<TrustStore, CryptoError> {
    if !path.exists() {
        return Ok(TrustStore::default());
    }
    let raw = std::fs::read_to_string(path)?;
    let store: TrustStore = serde_json::from_str(&raw)?;
    if store.version != TRUST_STORE_VERSION {
        return Err(CryptoError::UnsupportedVersion {
            version: store.version,
        });
    }
    Ok(store)
}

pub fn save_trust_store(path: &Path, store: &TrustStore) -> Result<(), CryptoError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(store)?;
    write_keystore_bytes_atomically(path, &bytes)?;
    Ok(())
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn decode_hex_32(label: &str, value: &str) -> Result<[u8; 32], CryptoError> {
    let decoded = hex::decode(value).map_err(|e| CryptoError::InvalidKeyMaterial {
        reason: format!("bad {label} hex: {e}"),
    })?;
    decoded
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: format!("{label} must be 32 bytes"),
        })
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

fn canonical_claim_bytes(claim: &NodeOwnershipClaim) -> Result<Vec<u8>, CryptoError> {
    let owner_sign_public_key =
        decode_hex_32("owner_sign_public_key", &claim.owner_sign_public_key)?;
    let node_endpoint_id = decode_hex_32("node_endpoint_id", &claim.node_endpoint_id)?;
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(SIGNING_DOMAIN_TAG);
    buf.extend_from_slice(&claim.version.to_le_bytes());
    write_string(&mut buf, &claim.cert_id);
    write_string(&mut buf, &claim.owner_id);
    buf.extend_from_slice(&owner_sign_public_key);
    buf.extend_from_slice(&node_endpoint_id);
    buf.extend_from_slice(&claim.issued_at_unix_ms.to_le_bytes());
    buf.extend_from_slice(&claim.expires_at_unix_ms.to_le_bytes());
    write_optional_string(&mut buf, claim.node_label.as_deref());
    write_optional_string(&mut buf, claim.hostname_hint.as_deref());
    Ok(buf)
}

fn random_cert_id() -> String {
    let bytes: [u8; 16] = rand::random();
    hex::encode(bytes)
}

pub fn sign_node_ownership(
    owner: &OwnerKeypair,
    node_endpoint_id: &[u8; 32],
    expires_at_unix_ms: u64,
    node_label: Option<String>,
    hostname_hint: Option<String>,
) -> Result<SignedNodeOwnership, CryptoError> {
    let claim = NodeOwnershipClaim {
        version: NODE_OWNERSHIP_VERSION,
        cert_id: random_cert_id(),
        owner_id: owner.owner_id(),
        owner_sign_public_key: hex::encode(owner.verifying_key().as_bytes()),
        node_endpoint_id: hex::encode(node_endpoint_id),
        issued_at_unix_ms: current_time_unix_ms(),
        expires_at_unix_ms,
        node_label,
        hostname_hint,
    };
    let bytes = canonical_claim_bytes(&claim)?;
    let signature = owner.signing.sign(&bytes);
    Ok(SignedNodeOwnership {
        claim,
        signature: hex::encode(signature.to_bytes()),
    })
}

pub fn certificate_needs_renewal(
    ownership: &SignedNodeOwnership,
    renew_window_secs: u64,
    now_unix_ms: u64,
) -> bool {
    if ownership.claim.expires_at_unix_ms <= now_unix_ms {
        return true;
    }
    let remaining_ms = ownership
        .claim
        .expires_at_unix_ms
        .saturating_sub(now_unix_ms);
    remaining_ms <= renew_window_secs.saturating_mul(1000)
}

pub fn verify_node_ownership(
    ownership: Option<&SignedNodeOwnership>,
    actual_node_endpoint_id: &[u8; 32],
    trust_store: &TrustStore,
    policy: TrustPolicy,
    now_unix_ms: u64,
) -> OwnershipSummary {
    let Some(ownership) = ownership else {
        return OwnershipSummary::default();
    };

    let mut summary = OwnershipSummary {
        owner_id: Some(ownership.claim.owner_id.clone()),
        cert_id: Some(ownership.claim.cert_id.clone()),
        expires_at_unix_ms: Some(ownership.claim.expires_at_unix_ms),
        node_label: ownership.claim.node_label.clone(),
        hostname_hint: ownership.claim.hostname_hint.clone(),
        ..OwnershipSummary::default()
    };
    if ownership.claim.version != NODE_OWNERSHIP_VERSION {
        summary.status = OwnershipStatus::UnsupportedProtocol;
        return summary;
    }

    let owner_sign_public_key = match decode_hex_32(
        "owner_sign_public_key",
        &ownership.claim.owner_sign_public_key,
    ) {
        Ok(bytes) => bytes,
        Err(_) => {
            summary.status = OwnershipStatus::InvalidSignature;
            return summary;
        }
    };
    let node_endpoint_id =
        match decode_hex_32("node_endpoint_id", &ownership.claim.node_endpoint_id) {
            Ok(bytes) => bytes,
            Err(_) => {
                summary.status = OwnershipStatus::InvalidSignature;
                return summary;
            }
        };

    if node_endpoint_id != *actual_node_endpoint_id {
        summary.status = OwnershipStatus::MismatchedNodeId;
        return summary;
    }

    let verifying_key = match ed25519_dalek::VerifyingKey::from_bytes(&owner_sign_public_key) {
        Ok(value) => value,
        Err(_) => {
            summary.status = OwnershipStatus::InvalidSignature;
            return summary;
        }
    };
    let expected_owner_id = owner_id_from_verifying_key(&verifying_key);
    if expected_owner_id != ownership.claim.owner_id {
        summary.status = OwnershipStatus::InvalidSignature;
        return summary;
    }

    let canonical = match canonical_claim_bytes(&ownership.claim) {
        Ok(value) => value,
        Err(_) => {
            summary.status = OwnershipStatus::InvalidSignature;
            return summary;
        }
    };
    let sig_bytes = match hex::decode(&ownership.signature) {
        Ok(value) => value,
        Err(_) => {
            summary.status = OwnershipStatus::InvalidSignature;
            return summary;
        }
    };
    let sig_bytes: [u8; 64] = match sig_bytes.try_into() {
        Ok(value) => value,
        Err(_) => {
            summary.status = OwnershipStatus::InvalidSignature;
            return summary;
        }
    };
    let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    if verifying_key.verify_strict(&canonical, &signature).is_err() {
        summary.status = OwnershipStatus::InvalidSignature;
        return summary;
    }

    if ownership.claim.expires_at_unix_ms <= now_unix_ms {
        summary.status = OwnershipStatus::Expired;
        return summary;
    }

    if trust_store
        .revoked_owners
        .iter()
        .any(|entry| entry.owner_id == ownership.claim.owner_id)
    {
        summary.status = OwnershipStatus::RevokedOwner;
        return summary;
    }

    if trust_store
        .revoked_node_certs
        .iter()
        .any(|entry| entry.cert_id == ownership.claim.cert_id)
    {
        summary.status = OwnershipStatus::RevokedCert;
        return summary;
    }

    if trust_store.revoked_node_ids.iter().any(|entry| {
        entry
            .node_endpoint_id
            .eq_ignore_ascii_case(&ownership.claim.node_endpoint_id)
    }) {
        summary.status = OwnershipStatus::RevokedNodeId;
        return summary;
    }

    if matches!(policy, TrustPolicy::Allowlist)
        && !trust_store
            .trusted_owners
            .iter()
            .any(|entry| entry.owner_id == ownership.claim.owner_id)
    {
        summary.status = OwnershipStatus::UntrustedOwner;
        return summary;
    }

    summary.status = OwnershipStatus::Verified;
    summary.verified = true;
    summary
}

impl TrustStore {
    pub fn merged_with_trusted_owners(mut self, owners: &[String]) -> Self {
        for owner_id in owners {
            self.add_trusted_owner(owner_id.clone(), None);
        }
        self
    }

    pub fn add_trusted_owner(&mut self, owner_id: String, label: Option<String>) {
        if let Some(existing) = self
            .trusted_owners
            .iter_mut()
            .find(|entry| entry.owner_id == owner_id)
        {
            if label.is_some() {
                existing.label = label;
            }
            return;
        }
        self.trusted_owners.push(TrustedOwner { owner_id, label });
        self.trusted_owners
            .sort_by(|a, b| a.owner_id.cmp(&b.owner_id));
    }

    pub fn remove_trusted_owner(&mut self, owner_id: &str) -> bool {
        let before = self.trusted_owners.len();
        self.trusted_owners
            .retain(|entry| entry.owner_id != owner_id);
        before != self.trusted_owners.len()
    }

    pub fn revoke_owner(&mut self, owner_id: String, reason: Option<String>) {
        if self
            .revoked_owners
            .iter()
            .any(|entry| entry.owner_id == owner_id)
        {
            return;
        }
        self.revoked_owners.push(RevokedOwner { owner_id, reason });
        self.revoked_owners
            .sort_by(|a, b| a.owner_id.cmp(&b.owner_id));
    }

    pub fn revoke_node_cert(&mut self, cert_id: String, reason: Option<String>) {
        if self
            .revoked_node_certs
            .iter()
            .any(|entry| entry.cert_id == cert_id)
        {
            return;
        }
        self.revoked_node_certs
            .push(RevokedNodeCert { cert_id, reason });
        self.revoked_node_certs
            .sort_by(|a, b| a.cert_id.cmp(&b.cert_id));
    }

    pub fn revoke_node_id(&mut self, node_endpoint_id: String, reason: Option<String>) {
        if self.revoked_node_ids.iter().any(|entry| {
            entry
                .node_endpoint_id
                .eq_ignore_ascii_case(&node_endpoint_id)
        }) {
            return;
        }
        self.revoked_node_ids.push(RevokedNodeId {
            node_endpoint_id,
            reason,
        });
        self.revoked_node_ids
            .sort_by(|a, b| a.node_endpoint_id.cmp(&b.node_endpoint_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_ownership_verifies() {
        let owner = OwnerKeypair::generate();
        let node_endpoint_id = [0x42; 32];
        let ownership = sign_node_ownership(
            &owner,
            &node_endpoint_id,
            current_time_unix_ms() + 60_000,
            Some("studio".into()),
            Some("studio-host".into()),
        )
        .unwrap();

        let summary = verify_node_ownership(
            Some(&ownership),
            &node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        );

        assert_eq!(summary.status, OwnershipStatus::Verified);
        assert!(summary.verified);
        assert_eq!(summary.node_label.as_deref(), Some("studio"));
    }

    #[test]
    fn mismatched_node_id_is_rejected() {
        let owner = OwnerKeypair::generate();
        let node_endpoint_id = [0x42; 32];
        let wrong_node_endpoint_id = [0x24; 32];
        let ownership = sign_node_ownership(
            &owner,
            &node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();

        let summary = verify_node_ownership(
            Some(&ownership),
            &wrong_node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        );

        assert_eq!(summary.status, OwnershipStatus::MismatchedNodeId);
        assert!(!summary.verified);
    }

    #[test]
    fn revoked_owner_is_rejected() {
        let owner = OwnerKeypair::generate();
        let node_endpoint_id = [0x42; 32];
        let ownership = sign_node_ownership(
            &owner,
            &node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();
        let mut trust_store = TrustStore::default();
        trust_store.revoke_owner(owner.owner_id(), Some("compromised".into()));

        let summary = verify_node_ownership(
            Some(&ownership),
            &node_endpoint_id,
            &trust_store,
            TrustPolicy::Off,
            current_time_unix_ms(),
        );

        assert_eq!(summary.status, OwnershipStatus::RevokedOwner);
    }

    #[test]
    fn allowlist_requires_trusted_owner() {
        let owner = OwnerKeypair::generate();
        let node_endpoint_id = [0x42; 32];
        let ownership = sign_node_ownership(
            &owner,
            &node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();

        let summary = verify_node_ownership(
            Some(&ownership),
            &node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Allowlist,
            current_time_unix_ms(),
        );

        assert_eq!(summary.status, OwnershipStatus::UntrustedOwner);
    }

    #[test]
    fn unsupported_claim_version_is_rejected() {
        let owner = OwnerKeypair::generate();
        let node_endpoint_id = [0x42; 32];
        let mut ownership = sign_node_ownership(
            &owner,
            &node_endpoint_id,
            current_time_unix_ms() + 60_000,
            None,
            None,
        )
        .unwrap();
        ownership.claim.version = NODE_OWNERSHIP_VERSION + 1;

        let summary = verify_node_ownership(
            Some(&ownership),
            &node_endpoint_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
        );

        assert_eq!(summary.status, OwnershipStatus::UnsupportedProtocol);
        assert!(!summary.verified);
    }
}
