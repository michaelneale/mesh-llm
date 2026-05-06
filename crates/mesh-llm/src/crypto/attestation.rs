//! Hardware attestation for Apple Silicon nodes.
//!
//! - `HardwareAttestation` struct with hardware identity + security posture
//! - P-256 ECDSA signing via Apple Secure Enclave (macOS only)
//! - P-256 ECDSA verification (all platforms)
//! - Challenge-response nonce signing for liveness proof

use base64::Engine;
use p256::ecdsa::signature::Verifier;
use serde::{Deserialize, Serialize};

use super::error::CryptoError;

const ATTESTATION_DOMAIN: &[u8] = b"mesh-llm-hardware-attestation-v1:";
const CHALLENGE_DOMAIN: &[u8] = b"mesh-llm-challenge-v1:";

// ── Attestation structs ───────────────────────────────────────────

/// Hardware attestation blob — signed by the Secure Enclave P-256 key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareAttestation {
    pub chip_name: String,
    pub hardware_model: String,
    pub unified_memory_bytes: u64,
    pub sip_enabled: bool,
    pub secure_boot_enabled: bool,
    pub rdma_disabled: bool,
    /// iroh endpoint ID (hex)
    pub node_endpoint_id: String,
    /// X25519 inference public key (base64)
    pub inference_public_key: String,
    /// P-256 SE public key (base64, SEC1)
    pub se_public_key: String,
    /// SHA-256 of the running binary
    pub binary_hash: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
}

/// Signed attestation = attestation + P-256 ECDSA signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedHardwareAttestation {
    pub attestation: HardwareAttestation,
    /// DER-encoded P-256 ECDSA signature, base64
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationStatus {
    Verified,
    None,
    InvalidSignature,
    Expired,
    MismatchedNodeId,
    MismatchedInferenceKey,
    InsecurePosture,
    UnblessedBinary,
}

impl AttestationStatus {
    pub fn is_verified(&self) -> bool {
        matches!(self, AttestationStatus::Verified)
    }
}

// ── Canonical signing bytes ───────────────────────────────────────

fn attestation_signing_bytes(att: &HardwareAttestation) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);
    buf.extend_from_slice(ATTESTATION_DOMAIN);

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

    buf
}

fn challenge_signing_bytes(nonce: &[u8], se_public_key: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(CHALLENGE_DOMAIN);
    buf.extend_from_slice(&(se_public_key.len() as u32).to_le_bytes());
    buf.extend_from_slice(se_public_key.as_bytes());
    buf.extend_from_slice(nonce);
    buf
}

// ── Verification (all platforms) ──────────────────────────────────

fn parse_se_public_key(b64: &str) -> Result<p256::ecdsa::VerifyingKey, CryptoError> {
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
/// **Trust model limitation:** This verifies cryptographic self-consistency
/// (the blob was signed by the embedded SE key) but does NOT verify a
/// hardware root of trust chain. Any node can forge an attestation using a
/// software P-256 key. To make this meaningful, callers should combine
/// attestation with one of:
///   - TOFU: pin the SE public key on first observation
///   - Operator-managed blessed SE key fingerprints
///   - Apple DeviceCheck attestation chain (future)
pub fn verify_attestation(
    signed: &SignedHardwareAttestation,
    expected_node_id: &str,
    expected_inference_key: &str,
    max_age_secs: u64,
    blessed_hashes: Option<&[String]>,
) -> AttestationStatus {
    // 1. Signature
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
    if !signed.attestation.sip_enabled
        || !signed.attestation.secure_boot_enabled
        || !signed.attestation.rdma_disabled
    {
        return AttestationStatus::InsecurePosture;
    }

    // 5. Timestamp freshness
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&signed.attestation.timestamp) {
        let now = chrono::Utc::now();
        let delta = now.signed_duration_since(ts).num_seconds();
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

// ── SE operations (macOS) ─────────────────────────────────────────

/// Check if the Secure Enclave is available.
pub fn secure_enclave_available() -> bool {
    #[cfg(target_os = "macos")]
    {
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

/// A P-256 key held in the Apple Secure Enclave.
/// The private key never leaves hardware.
#[cfg(target_os = "macos")]
pub struct SecureEnclaveIdentity {
    key: security_framework::key::SecKey,
    public_key_base64: String,
}

#[cfg(target_os = "macos")]
impl SecureEnclaveIdentity {
    /// Create a new ephemeral P-256 key in the Secure Enclave.
    pub fn create() -> Result<Self, CryptoError> {
        use core_foundation::base::TCFType;
        use core_foundation::number::CFNumber;

        unsafe {
            let keys: Vec<_> = vec![
                security_framework_sys::item::kSecAttrKeyType,
                security_framework_sys::item::kSecAttrKeySizeInBits,
                security_framework_sys::item::kSecAttrTokenID,
                security_framework_sys::item::kSecAttrIsPermanent,
            ];
            let k_type = security_framework_sys::item::kSecAttrKeyTypeECSECPrimeRandom;
            let size = CFNumber::from(256i32);
            let k_token = security_framework_sys::item::kSecAttrTokenIDSecureEnclave;
            let values: Vec<_> = vec![
                k_type as *const _,
                size.as_concrete_TypeRef() as *const _,
                k_token as *const _,
                core_foundation::boolean::kCFBooleanFalse as *const _,
            ];

            let dict = core_foundation_sys::dictionary::CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr() as *const *const _,
                values.as_ptr() as *const *const _,
                keys.len() as _,
                &core_foundation_sys::dictionary::kCFTypeDictionaryKeyCallBacks,
                &core_foundation_sys::dictionary::kCFTypeDictionaryValueCallBacks,
            );

            let mut error = std::ptr::null_mut();
            let key_ref = security_framework_sys::key::SecKeyCreateRandomKey(dict, &mut error);
            core_foundation_sys::base::CFRelease(dict as *const _);

            if key_ref.is_null() {
                let err_desc = if !error.is_null() {
                    let cf_err = core_foundation::error::CFError::wrap_under_create_rule(error);
                    cf_err.description().to_string()
                } else {
                    "unknown error".to_string()
                };
                return Err(CryptoError::InvalidKeyMaterial {
                    reason: format!("SE key creation failed: {err_desc}"),
                });
            }

            let key = security_framework::key::SecKey::wrap_under_create_rule(key_ref);
            let pub_key = key.public_key().ok_or(CryptoError::InvalidKeyMaterial {
                reason: "failed to get SE public key".into(),
            })?;
            let pub_data =
                pub_key
                    .external_representation()
                    .ok_or(CryptoError::InvalidKeyMaterial {
                        reason: "failed to export SE public key".into(),
                    })?;

            let pub_b64 = base64::engine::general_purpose::STANDARD.encode(pub_data.bytes());

            Ok(Self {
                key,
                public_key_base64: pub_b64,
            })
        }
    }

    pub fn public_key_base64(&self) -> &str {
        &self.public_key_base64
    }

    /// Sign raw bytes with the SE private key (ECDSA-SHA256).
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>, CryptoError> {
        use security_framework::key::Algorithm;
        self.key
            .create_signature(Algorithm::ECDSASignatureMessageX962SHA256, data)
            .map_err(|e| CryptoError::InvalidKeyMaterial {
                reason: format!("SE signing failed: {e}"),
            })
    }

    /// Sign a hardware attestation blob.
    pub fn sign_attestation(
        &self,
        attestation: &HardwareAttestation,
    ) -> Result<SignedHardwareAttestation, CryptoError> {
        let msg = attestation_signing_bytes(attestation);
        let signature = self.sign(&msg)?;
        Ok(SignedHardwareAttestation {
            attestation: attestation.clone(),
            signature: base64::engine::general_purpose::STANDARD.encode(&signature),
        })
    }

    /// Sign a challenge nonce.
    pub fn sign_challenge(&self, nonce: &[u8]) -> Result<String, CryptoError> {
        let msg = challenge_signing_bytes(nonce, &self.public_key_base64);
        let signature = self.sign(&msg)?;
        Ok(base64::engine::general_purpose::STANDARD.encode(&signature))
    }
}

// ── Software signing (tests + non-macOS) ──────────────────────────

/// Sign an attestation with a software P-256 key. Testing/dev only.
pub fn sign_attestation_software(
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

/// Sign a challenge nonce with a software key. Testing/dev only.
pub fn sign_challenge_software(
    nonce: &[u8],
    se_public_key_b64: &str,
    signing_key: &p256::ecdsa::SigningKey,
) -> Result<String, CryptoError> {
    use p256::ecdsa::signature::Signer;
    let msg = challenge_signing_bytes(nonce, se_public_key_b64);
    let signature: p256::ecdsa::DerSignature = signing_key.sign(&msg);
    Ok(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()))
}

/// Get base64 public key from a P-256 signing key.
pub fn public_key_base64_from_signing_key(signing_key: &p256::ecdsa::SigningKey) -> String {
    let vk = signing_key.verifying_key();
    let point = vk.to_encoded_point(false); // uncompressed for SE compat
    base64::engine::general_purpose::STANDARD.encode(point.as_bytes())
}

// ── Persistent SE handle ──────────────────────────────────────────

/// Opaque handle to a Secure Enclave identity that can be stored in a `Node`
/// and reused across attestation refreshes. This avoids creating a new
/// ephemeral SE key every refresh cycle, which would prevent peers from
/// pinning (TOFU) the SE public key.
///
/// On non-macOS platforms this is a zero-sized type that is always empty.
#[derive(Clone)]
pub struct SeIdentityHandle {
    #[cfg(target_os = "macos")]
    inner: Option<std::sync::Arc<SecureEnclaveIdentity>>,
}

impl SeIdentityHandle {
    /// Create an empty handle (no SE key yet).
    pub fn empty() -> Self {
        Self {
            #[cfg(target_os = "macos")]
            inner: None,
        }
    }
}

// ── Convenience: attempt attestation at startup ───────────────────

/// Attempt to create a signed hardware attestation, **reusing** the SE
/// identity from `handle` when available. If the handle is empty (first
/// call), a new SE key is created and the handle is populated so that
/// subsequent calls reuse the same key — preserving SE public-key
/// continuity for TOFU pinning.
///
/// Returns `None` if SE is unavailable (non-macOS, Intel Mac, or SE access
/// denied).
pub fn try_create_attestation(
    node_endpoint_id: &str,
    inference_public_key: &str,
    posture: &crate::system::hardening::SecurityPosture,
    handle: &mut SeIdentityHandle,
) -> Option<SignedHardwareAttestation> {
    if !secure_enclave_available() {
        tracing::debug!("Secure Enclave not available, skipping hardware attestation");
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        // Reuse the existing SE identity or create one on first call.
        let se = match &handle.inner {
            Some(existing) => existing.clone(),
            None => {
                let new_se = match SecureEnclaveIdentity::create() {
                    Ok(se) => std::sync::Arc::new(se),
                    Err(e) => {
                        tracing::warn!("Failed to create SE identity: {e}");
                        return None;
                    }
                };
                handle.inner = Some(new_se.clone());
                new_se
            }
        };

        // Gather hardware info
        let chip_name = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        let hardware_model = std::process::Command::new("sysctl")
            .args(["-n", "hw.model"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        let unified_memory_bytes = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
            .unwrap_or(0);

        let attestation = HardwareAttestation {
            node_endpoint_id: node_endpoint_id.to_string(),
            inference_public_key: inference_public_key.to_string(),
            se_public_key: se.public_key_base64().to_string(),
            binary_hash: posture.binary_hash.clone().unwrap_or_default(),
            chip_name,
            hardware_model,
            unified_memory_bytes,
            sip_enabled: posture.sip_enabled,
            secure_boot_enabled: true, // assume if SE works
            rdma_disabled: posture.rdma_disabled,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        match se.sign_attestation(&attestation) {
            Ok(signed) => Some(signed),
            Err(e) => {
                tracing::warn!("Failed to sign attestation: {e}");
                None
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (node_endpoint_id, inference_public_key, posture, handle);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> p256::ecdsa::SigningKey {
        p256::ecdsa::SigningKey::random(&mut p256::elliptic_curve::rand_core::OsRng)
    }

    fn test_attestation(sk: &p256::ecdsa::SigningKey) -> HardwareAttestation {
        HardwareAttestation {
            chip_name: "Apple M4 Max".into(),
            hardware_model: "Mac16,1".into(),
            unified_memory_bytes: 128 * 1024 * 1024 * 1024,
            sip_enabled: true,
            secure_boot_enabled: true,
            rdma_disabled: true,
            node_endpoint_id: "test-node-123".into(),
            inference_public_key: "dGVzdC1pbmZlcmVuY2Uta2V5".into(),
            se_public_key: public_key_base64_from_signing_key(sk),
            binary_hash: "abcd1234".repeat(8),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    #[test]
    fn sign_and_verify() {
        let sk = test_key();
        let att = test_attestation(&sk);
        let signed = sign_attestation_software(&att, &sk).unwrap();
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
    fn tampered_fails() {
        let sk = test_key();
        let att = test_attestation(&sk);
        let mut signed = sign_attestation_software(&att, &sk).unwrap();
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
    fn expired_fails() {
        let sk = test_key();
        let mut att = test_attestation(&sk);
        att.timestamp = (chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        let signed = sign_attestation_software(&att, &sk).unwrap();
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
    fn challenge_roundtrip() {
        let sk = test_key();
        let pub_b64 = public_key_base64_from_signing_key(&sk);
        let nonce = b"random-challenge-nonce-32bytes!!";
        let sig = sign_challenge_software(nonce, &pub_b64, &sk).unwrap();
        assert!(verify_challenge_response(nonce, &sig, &pub_b64));
    }

    #[test]
    fn challenge_wrong_nonce_fails() {
        let sk = test_key();
        let pub_b64 = public_key_base64_from_signing_key(&sk);
        let sig = sign_challenge_software(b"correct", &pub_b64, &sk).unwrap();
        assert!(!verify_challenge_response(b"wrong", &sig, &pub_b64));
    }
}
