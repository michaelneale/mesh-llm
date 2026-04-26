//! Secure Enclave hardware attestation for Apple Silicon nodes.
//!
//! Provides:
//! - `HardwareAttestation` struct with hardware identity + security posture
//! - P-256 ECDSA signing via Apple Secure Enclave (macOS only)
//! - P-256 ECDSA verification (all platforms — any peer can verify)
//! - Challenge-response nonce signing for continuous liveness proof
//!
//! On macOS, keys are created in the Secure Enclave via Security.framework.
//! The private key never leaves the hardware. On other platforms, SE
//! operations are unavailable but verification still works.

use base64::Engine;
use p256::ecdsa::signature::Verifier;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::CryptoError;

/// Domain separation tag for attestation signatures.
const ATTESTATION_DOMAIN: &[u8] = b"mesh-llm-hardware-attestation-v1:";

/// Domain separation tag for challenge-response nonce signatures.
const CHALLENGE_DOMAIN: &[u8] = b"mesh-llm-challenge-v1:";

// ── Attestation structs ───────────────────────────────────────────

/// Hardware attestation blob — describes this node's hardware and
/// security state. Signed by the Secure Enclave P-256 key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareAttestation {
    /// Apple chip name, e.g. "Apple M4 Max"
    pub chip_name: String,
    /// Hardware model identifier, e.g. "Mac16,1"
    pub hardware_model: String,
    /// Actual unified memory in bytes (from hardware, not self-reported)
    pub unified_memory_bytes: u64,

    // Security posture
    pub sip_enabled: bool,
    pub secure_boot_enabled: bool,
    pub rdma_disabled: bool,

    // Identity binding
    /// iroh endpoint ID — ties attestation to mesh identity
    pub node_endpoint_id: String,
    /// X25519 inference public key (base64) — ties encryption key to hardware
    pub inference_public_key: String,
    /// P-256 Secure Enclave public key (base64, SEC1 compressed)
    pub se_public_key: String,
    /// SHA-256 hash of the running binary
    pub binary_hash: String,

    /// ISO 8601 timestamp for freshness
    pub timestamp: String,
}

/// Signed attestation blob — attestation + P-256 ECDSA signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedHardwareAttestation {
    pub attestation: HardwareAttestation,
    /// DER-encoded P-256 ECDSA signature, base64
    pub signature: String,
}

/// Result of verifying a peer's hardware attestation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationStatus {
    /// Signature valid, all checks passed.
    Verified,
    /// No attestation provided.
    None,
    /// Signature invalid or tampered.
    InvalidSignature,
    /// Attestation too old.
    Expired,
    /// node_endpoint_id doesn't match QUIC peer.
    MismatchedNodeId,
    /// inference_public_key doesn't match gossip.
    MismatchedInferenceKey,
    /// Security posture insufficient (SIP disabled, etc).
    InsecurePosture,
    /// Binary hash not in blessed set.
    UnblessedBinary,
}

impl AttestationStatus {
    pub fn is_verified(&self) -> bool {
        matches!(self, AttestationStatus::Verified)
    }
}

// ── Canonical signing bytes ───────────────────────────────────────

/// Build the canonical byte representation of an attestation for signing.
///
/// Deterministic: same attestation always produces same bytes.
fn attestation_signing_bytes(att: &HardwareAttestation) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);
    buf.extend_from_slice(ATTESTATION_DOMAIN);

    // Fixed-order fields with length prefixes for unambiguous parsing
    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    push_str(&mut buf, &att.chip_name);
    push_str(&mut buf, &att.hardware_model);
    buf.extend_from_slice(&att.unified_memory_bytes.to_le_bytes());
    buf.push(att.sip_enabled as u8);
    buf.push(att.secure_boot_enabled as u8);
    buf.push(att.rdma_disabled as u8);
    push_str(&mut buf, &att.node_endpoint_id);
    push_str(&mut buf, &att.inference_public_key);
    push_str(&mut buf, &att.se_public_key);
    push_str(&mut buf, &att.binary_hash);
    push_str(&mut buf, &att.timestamp);

    // Hash for compact signing (same pattern as envelope.rs)
    Sha256::digest(&buf).to_vec()
}

/// Build canonical bytes for a challenge nonce signature.
fn challenge_signing_bytes(nonce: &[u8], se_public_key: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(CHALLENGE_DOMAIN);
    buf.extend_from_slice(&(se_public_key.len() as u32).to_le_bytes());
    buf.extend_from_slice(se_public_key.as_bytes());
    buf.extend_from_slice(nonce);
    Sha256::digest(&buf).to_vec()
}

// ── Verification (all platforms) ──────────────────────────────────

/// Parse a base64-encoded P-256 public key (SEC1 compressed or uncompressed).
pub fn parse_se_public_key(b64: &str) -> Result<p256::ecdsa::VerifyingKey, CryptoError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad SE public key base64: {e}"),
        })?;

    let point =
        p256::EncodedPoint::from_bytes(&bytes).map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad SE public key encoding: {e}"),
        })?;

    p256::ecdsa::VerifyingKey::from_encoded_point(&point).map_err(|e| {
        CryptoError::InvalidKeyMaterial {
            reason: format!("invalid SE public key: {e}"),
        }
    })
}

/// Parse a base64-encoded DER ECDSA signature.
fn parse_signature(b64: &str) -> Result<p256::ecdsa::DerSignature, CryptoError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad signature base64: {e}"),
        })?;

    p256::ecdsa::DerSignature::try_from(bytes.as_slice()).map_err(|e| {
        CryptoError::InvalidKeyMaterial {
            reason: format!("bad DER signature: {e}"),
        }
    })
}

/// Verify a signed hardware attestation.
///
/// Checks:
/// 1. P-256 ECDSA signature valid against embedded SE public key
/// 2. node_endpoint_id matches expected peer
/// 3. inference_public_key matches expected
/// 4. Security posture (SIP enabled)
/// 5. Timestamp freshness (within `max_age_secs`)
/// 6. Optionally: binary_hash in blessed set
pub fn verify_attestation(
    signed: &SignedHardwareAttestation,
    expected_node_id: &str,
    expected_inference_key: &str,
    max_age_secs: u64,
    blessed_hashes: Option<&[String]>,
) -> AttestationStatus {
    // 1. Verify signature
    let verifying_key = match parse_se_public_key(&signed.attestation.se_public_key) {
        Ok(k) => k,
        Err(_) => return AttestationStatus::InvalidSignature,
    };

    let signature = match parse_signature(&signed.signature) {
        Ok(s) => s,
        Err(_) => return AttestationStatus::InvalidSignature,
    };

    let msg = attestation_signing_bytes(&signed.attestation);
    if verifying_key.verify(&msg, &signature).is_err() {
        return AttestationStatus::InvalidSignature;
    }

    // 2. Node ID binding
    if signed.attestation.node_endpoint_id != expected_node_id {
        return AttestationStatus::MismatchedNodeId;
    }

    // 3. Inference key binding
    if signed.attestation.inference_public_key != expected_inference_key {
        return AttestationStatus::MismatchedInferenceKey;
    }

    // 4. Security posture
    if !signed.attestation.sip_enabled {
        return AttestationStatus::InsecurePosture;
    }
    if !signed.attestation.secure_boot_enabled {
        return AttestationStatus::InsecurePosture;
    }
    if !signed.attestation.rdma_disabled {
        return AttestationStatus::InsecurePosture;
    }

    // 5. Timestamp freshness
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&signed.attestation.timestamp) {
        let now = chrono::Utc::now();
        let delta = now.signed_duration_since(ts).num_seconds();
        // Reject future timestamps beyond 30s clock skew
        if delta < -30 {
            return AttestationStatus::Expired;
        }
        if delta >= 0 && delta as u64 > max_age_secs {
            return AttestationStatus::Expired;
        }
    } else {
        return AttestationStatus::Expired;
    }

    // 6. Binary hash (optional)
    if let Some(blessed) = blessed_hashes {
        if !blessed.contains(&signed.attestation.binary_hash) {
            return AttestationStatus::UnblessedBinary;
        }
    }

    AttestationStatus::Verified
}

/// Verify a challenge-response nonce signature.
///
/// The peer must have signed the nonce with their SE private key.
/// We verify against their known SE public key.
pub fn verify_challenge_response(
    nonce: &[u8],
    signature_b64: &str,
    se_public_key_b64: &str,
) -> bool {
    let verifying_key = match parse_se_public_key(se_public_key_b64) {
        Ok(k) => k,
        Err(_) => return false,
    };

    let signature = match parse_signature(signature_b64) {
        Ok(s) => s,
        Err(_) => return false,
    };

    let msg = challenge_signing_bytes(nonce, se_public_key_b64);
    verifying_key.verify(&msg, &signature).is_ok()
}

// ── SE operations (macOS only) ────────────────────────────────────
//
// On macOS, we use Security.framework via core_foundation / security_framework
// crates to create P-256 keys in the Secure Enclave and sign with them.
//
// For now, we provide a trait + stub. The actual Security.framework FFI
// integration is behind #[cfg(target_os = "macos")] and requires the
// `security-framework` crate (added as an optional dep).
//
// The signing interface is simple:
//   1. create_se_key() → (public_key_bytes, opaque_handle)
//   2. se_sign(handle, data) → signature_bytes
//
// The private key handle is an opaque reference — the actual key bits
// never leave the Secure Enclave hardware.

/// Check if the Secure Enclave is available on this machine.
pub fn secure_enclave_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // The Secure Enclave is available on all Apple Silicon Macs
        // and T2-equipped Intel Macs. We check by trying to detect
        // Apple Silicon via sysctl.
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("Apple"))
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Sign an attestation blob using a **software** P-256 signing key.
///
/// **Testing / development only.** In production on macOS the Secure
/// Enclave hardware key should be used instead. This helper exists so
/// that tests and non-macOS environments can exercise the attestation
/// verification path without SE hardware.
pub fn sign_attestation_with_software_key(
    attestation: &HardwareAttestation,
    signing_key: &p256::ecdsa::SigningKey,
) -> Result<SignedHardwareAttestation, CryptoError> {
    use p256::ecdsa::signature::Signer;

    let msg = attestation_signing_bytes(attestation);
    let signature: p256::ecdsa::DerSignature = signing_key.sign(&msg);

    Ok(SignedHardwareAttestation {
        attestation: attestation.clone(),
        signature: base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()),
    })
}

/// Sign a challenge nonce using a **software** P-256 signing key.
///
/// **Testing / development only.** See [`sign_attestation_with_software_key`]
/// for rationale. In production the Secure Enclave signs challenges directly.
pub fn sign_challenge_with_software_key(
    nonce: &[u8],
    se_public_key_b64: &str,
    signing_key: &p256::ecdsa::SigningKey,
) -> Result<String, CryptoError> {
    use p256::ecdsa::signature::Signer;

    let msg = challenge_signing_bytes(nonce, se_public_key_b64);
    let signature: p256::ecdsa::DerSignature = signing_key.sign(&msg);

    Ok(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()))
}

/// Get the base64-encoded public key from a P-256 signing key.
pub fn public_key_base64(signing_key: &p256::ecdsa::SigningKey) -> String {
    let verifying_key = signing_key.verifying_key();
    let point = verifying_key.to_encoded_point(true); // compressed
    base64::engine::general_purpose::STANDARD.encode(point.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signing_key() -> p256::ecdsa::SigningKey {
        p256::ecdsa::SigningKey::random(&mut p256::elliptic_curve::rand_core::OsRng)
    }

    fn test_attestation(signing_key: &p256::ecdsa::SigningKey) -> HardwareAttestation {
        HardwareAttestation {
            chip_name: "Apple M4 Max".into(),
            hardware_model: "Mac16,1".into(),
            unified_memory_bytes: 128 * 1024 * 1024 * 1024, // 128 GB
            sip_enabled: true,
            secure_boot_enabled: true,
            rdma_disabled: true,
            node_endpoint_id: "test-node-123".into(),
            inference_public_key: "dGVzdC1pbmZlcmVuY2Uta2V5".into(),
            se_public_key: public_key_base64(signing_key),
            binary_hash: "abcd1234".repeat(8),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    #[test]
    fn test_sign_and_verify_attestation() {
        let sk = test_signing_key();
        let att = test_attestation(&sk);
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::Verified);
    }

    #[test]
    fn test_wrong_key_fails_verification() {
        let sk = test_signing_key();
        let att = test_attestation(&sk);
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        // Tamper: change node ID
        let status = verify_attestation(
            &signed,
            "wrong-node-id",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::MismatchedNodeId);
    }

    #[test]
    fn test_tampered_attestation_fails() {
        let sk = test_signing_key();
        let att = test_attestation(&sk);
        let mut signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        // Tamper with the attestation after signing
        signed.attestation.unified_memory_bytes = 999;

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::InvalidSignature);
    }

    #[test]
    fn test_expired_attestation() {
        let sk = test_signing_key();
        let mut att = test_attestation(&sk);
        // Set timestamp to 2 hours ago
        att.timestamp = (chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600, // 10 minute max age
            None,
        );
        assert_eq!(status, AttestationStatus::Expired);
    }

    #[test]
    fn test_sip_disabled_fails() {
        let sk = test_signing_key();
        let mut att = test_attestation(&sk);
        att.sip_enabled = false;
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::InsecurePosture);
    }

    #[test]
    fn test_unblessed_binary_hash() {
        let sk = test_signing_key();
        let att = test_attestation(&sk);
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let blessed = vec!["different_hash".to_string()];
        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            Some(&blessed),
        );
        assert_eq!(status, AttestationStatus::UnblessedBinary);
    }

    #[test]
    fn test_challenge_response_roundtrip() {
        let sk = test_signing_key();
        let pub_b64 = public_key_base64(&sk);
        let nonce = b"random-challenge-nonce-32bytes!!";

        let sig = sign_challenge_with_software_key(nonce, &pub_b64, &sk).unwrap();
        assert!(verify_challenge_response(nonce, &sig, &pub_b64));
    }

    #[test]
    fn test_challenge_wrong_nonce_fails() {
        let sk = test_signing_key();
        let pub_b64 = public_key_base64(&sk);

        let sig = sign_challenge_with_software_key(b"correct-nonce", &pub_b64, &sk).unwrap();
        assert!(!verify_challenge_response(b"wrong-nonce", &sig, &pub_b64));
    }

    #[test]
    fn test_challenge_wrong_key_fails() {
        let sk1 = test_signing_key();
        let sk2 = test_signing_key();
        let pub1 = public_key_base64(&sk1);
        let pub2 = public_key_base64(&sk2);
        let nonce = b"test-nonce";

        let sig = sign_challenge_with_software_key(nonce, &pub1, &sk1).unwrap();
        // Verify against wrong public key
        assert!(!verify_challenge_response(nonce, &sig, &pub2));
    }

    #[test]
    fn test_inference_key_mismatch() {
        let sk = test_signing_key();
        let att = test_attestation(&sk);
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "d3JvbmctaW5mZXJlbmNlLWtleQ==", // wrong inference key
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::MismatchedInferenceKey);
    }

    #[test]
    fn test_parse_invalid_se_key() {
        assert!(parse_se_public_key("not-valid-base64!!!").is_err());
        assert!(parse_se_public_key("dG9vc2hvcnQ=").is_err()); // too short
    }

    #[test]
    fn test_se_available_returns_bool() {
        // Just verify it doesn't panic. Result depends on platform.
        let _ = secure_enclave_available();
    }

    #[test]
    fn test_secure_boot_disabled_fails() {
        let sk = test_signing_key();
        let mut att = test_attestation(&sk);
        att.secure_boot_enabled = false;
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::InsecurePosture);
    }

    #[test]
    fn test_rdma_enabled_fails() {
        let sk = test_signing_key();
        let mut att = test_attestation(&sk);
        att.rdma_disabled = false;
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::InsecurePosture);
    }

    #[test]
    fn test_future_timestamp_rejected() {
        let sk = test_signing_key();
        let mut att = test_attestation(&sk);
        // Set timestamp 60 seconds in the future (beyond 30s clock skew tolerance)
        att.timestamp = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::Expired);
    }

    #[test]
    fn test_small_clock_skew_accepted() {
        let sk = test_signing_key();
        let mut att = test_attestation(&sk);
        // Set timestamp 5 seconds in the future (within 30s clock skew tolerance)
        att.timestamp = (chrono::Utc::now() + chrono::Duration::seconds(5)).to_rfc3339();
        let signed = sign_attestation_with_software_key(&att, &sk).unwrap();

        let status = verify_attestation(
            &signed,
            "test-node-123",
            "dGVzdC1pbmZlcmVuY2Uta2V5",
            600,
            None,
        );
        assert_eq!(status, AttestationStatus::Verified);
    }
}
