//! E2E inference encryption using X25519 + XSalsa20-Poly1305 (NaCl box).
//!
//! Encrypts inference payloads (prompts, responses) from the API entry
//! point to the host node. Intermediate tunnel relays see only ciphertext.
//!
//! Each node generates an X25519 keypair at startup. The public key is
//! advertised in gossip. Per-request, the sender generates an ephemeral
//! keypair for forward secrecy.
//!
//! Wire format:
//! ```json
//! {
//!   "model": "...",                     // cleartext for routing
//!   "ephemeral_public_key": "<base64>", // one-time sender key
//!   "ciphertext": "<base64>"            // NaCl box of raw JSON body
//! }
//! ```

use base64::Engine;
use crypto_box::aead::{Aead, AeadCore, OsRng};
use crypto_box::SalsaBox;
use serde::{Deserialize, Serialize};

use super::error::CryptoError;

// ── Inference keypair ─────────────────────────────────────────────

/// X25519 keypair used for inference payload encryption.
///
/// The private key stays on this node. The public key is advertised in
/// gossip so peers can encrypt payloads to this node.
pub struct InferenceKeypair {
    secret: crypto_box::SecretKey,
    public: crypto_box::PublicKey,
}

impl InferenceKeypair {
    /// Generate a new random inference keypair.
    pub fn generate() -> Self {
        let secret = crypto_box::SecretKey::generate(&mut OsRng);
        let public = secret.public_key();
        Self { secret, public }
    }

    /// Reconstruct from raw secret key bytes (e.g. from sealed storage).
    pub fn from_secret_bytes(bytes: &[u8; 32]) -> Self {
        let secret = crypto_box::SecretKey::from(*bytes);
        let public = secret.public_key();
        Self { secret, public }
    }

    /// The public key as base64 for gossip advertisement.
    pub fn public_key_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.public.as_bytes())
    }

    /// Raw secret key bytes (for sealing to Secure Enclave).
    pub fn secret_bytes(&self) -> [u8; 32] {
        // crypto_box 0.9: SecretKey doesn't expose to_bytes() directly,
        // but we can serialize through the underlying representation.
        let mut out = [0u8; 32];
        out.copy_from_slice(&self.secret.to_bytes());
        out
    }

    /// The public key reference for encryption operations.
    pub fn public_key(&self) -> &crypto_box::PublicKey {
        &self.public
    }

    /// The secret key reference for decryption operations.
    pub fn secret_key(&self) -> &crypto_box::SecretKey {
        &self.secret
    }
}

// ── Wire types ────────────────────────────────────────────────────

/// Encrypted inference request as sent over the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedInferencePayload {
    /// Model name — cleartext for routing.
    pub model: String,
    /// Sender's ephemeral X25519 public key (base64).
    pub ephemeral_public_key: String,
    /// NaCl box ciphertext: nonce (24 bytes) || encrypted body (base64).
    pub ciphertext: String,
    /// Marks this as an encrypted payload.
    #[serde(default = "default_true")]
    pub encrypted: bool,
}

fn default_true() -> bool {
    true
}

/// Encrypted response chunk from host back to consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedResponseChunk {
    /// NaCl box ciphertext: nonce (24 bytes) || encrypted chunk (base64).
    pub ciphertext: String,
    /// Host's inference public key (base64) — so consumer can verify sender.
    pub sender_public_key: String,
}

// ── Ephemeral session ─────────────────────────────────────────────

/// Ephemeral keypair generated per-request for forward secrecy.
pub struct EphemeralSession {
    pub secret: crypto_box::SecretKey,
    pub public: crypto_box::PublicKey,
}

impl EphemeralSession {
    /// Generate a new ephemeral session for one request.
    pub fn new() -> Self {
        let secret = crypto_box::SecretKey::generate(&mut OsRng);
        let public = secret.public_key();
        Self { secret, public }
    }

    /// Public key as base64 (sent in the wire payload).
    pub fn public_key_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.public.as_bytes())
    }
}

// ── Encrypt / Decrypt ─────────────────────────────────────────────

/// Parse a base64-encoded X25519 public key.
pub fn parse_public_key(b64: &str) -> Result<crypto_box::PublicKey, CryptoError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad base64: {e}"),
        })?;
    let arr: [u8; 32] = bytes
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "public key must be 32 bytes".into(),
        })?;
    Ok(crypto_box::PublicKey::from(arr))
}

/// Encrypt an inference payload to a recipient's public key.
///
/// Returns the encrypted wire payload with the model extracted for routing.
/// The ephemeral session is returned so the caller can decrypt responses.
pub fn encrypt_inference_request(
    body: &[u8],
    model: &str,
    recipient_pub: &crypto_box::PublicKey,
) -> Result<(EncryptedInferencePayload, EphemeralSession), CryptoError> {
    let session = EphemeralSession::new();
    let salsa_box = SalsaBox::new(recipient_pub, &session.secret);
    let nonce = crypto_box::SalsaBox::generate_nonce(&mut OsRng);
    let ciphertext = salsa_box
        .encrypt(&nonce, body)
        .map_err(|_| CryptoError::EncryptionFailed)?;

    // Wire format: nonce || ciphertext, base64 encoded.
    let mut wire = Vec::with_capacity(24 + ciphertext.len());
    wire.extend_from_slice(&nonce);
    wire.extend_from_slice(&ciphertext);

    let payload = EncryptedInferencePayload {
        model: model.to_string(),
        ephemeral_public_key: session.public_key_base64(),
        ciphertext: base64::engine::general_purpose::STANDARD.encode(&wire),
        encrypted: true,
    };

    Ok((payload, session))
}

/// Decrypt an inference request received by this node.
///
/// `own_key` is this node's inference private key.
/// Returns the decrypted JSON body.
pub fn decrypt_inference_request(
    payload: &EncryptedInferencePayload,
    own_key: &crypto_box::SecretKey,
) -> Result<Vec<u8>, CryptoError> {
    let sender_pub = parse_public_key(&payload.ephemeral_public_key)?;
    let wire = base64::engine::general_purpose::STANDARD
        .decode(&payload.ciphertext)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad ciphertext base64: {e}"),
        })?;

    if wire.len() < 24 {
        return Err(CryptoError::DecryptionFailed);
    }
    let (nonce_bytes, ciphertext) = wire.split_at(24);
    let nonce = crypto_box::Nonce::from_slice(nonce_bytes);

    let salsa_box = SalsaBox::new(&sender_pub, own_key);
    salsa_box
        .decrypt(nonce, ciphertext)
        .map_err(|_| CryptoError::DecryptionFailed)
}

/// Encrypt a response chunk from the host back to the consumer.
///
/// `own_key` is the host's inference private key.
/// `recipient_pub` is the consumer's ephemeral public key from the request.
pub fn encrypt_response_chunk(
    chunk: &[u8],
    own_key: &InferenceKeypair,
    recipient_pub: &crypto_box::PublicKey,
) -> Result<EncryptedResponseChunk, CryptoError> {
    let salsa_box = SalsaBox::new(recipient_pub, own_key.secret_key());
    let nonce = crypto_box::SalsaBox::generate_nonce(&mut OsRng);
    let ciphertext = salsa_box
        .encrypt(&nonce, chunk)
        .map_err(|_| CryptoError::EncryptionFailed)?;

    let mut wire = Vec::with_capacity(24 + ciphertext.len());
    wire.extend_from_slice(&nonce);
    wire.extend_from_slice(&ciphertext);

    Ok(EncryptedResponseChunk {
        ciphertext: base64::engine::general_purpose::STANDARD.encode(&wire),
        sender_public_key: own_key.public_key_base64(),
    })
}

/// Decrypt a response chunk received by the consumer.
///
/// `session` is the consumer's ephemeral session from the request.
/// `sender_pub` is the host's inference public key.
pub fn decrypt_response_chunk(
    chunk: &EncryptedResponseChunk,
    session: &EphemeralSession,
    expected_sender_pub: &crypto_box::PublicKey,
) -> Result<Vec<u8>, CryptoError> {
    // Verify sender matches expected host.
    let sender_pub = parse_public_key(&chunk.sender_public_key)?;
    if sender_pub.as_bytes() != expected_sender_pub.as_bytes() {
        return Err(CryptoError::InvalidKeyMaterial {
            reason: "response sender key mismatch".into(),
        });
    }

    let wire = base64::engine::general_purpose::STANDARD
        .decode(&chunk.ciphertext)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad chunk ciphertext base64: {e}"),
        })?;

    if wire.len() < 24 {
        return Err(CryptoError::DecryptionFailed);
    }
    let (nonce_bytes, ciphertext) = wire.split_at(24);
    let nonce = crypto_box::Nonce::from_slice(nonce_bytes);

    let salsa_box = SalsaBox::new(&sender_pub, &session.secret);
    salsa_box
        .decrypt(nonce, ciphertext)
        .map_err(|_| CryptoError::DecryptionFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation_and_base64() {
        let kp = InferenceKeypair::generate();
        let b64 = kp.public_key_base64();
        assert!(!b64.is_empty());
        // Round-trip parse
        let parsed = parse_public_key(&b64).unwrap();
        assert_eq!(parsed.as_bytes(), kp.public_key().as_bytes());
    }

    #[test]
    fn test_keypair_from_secret_bytes() {
        let kp1 = InferenceKeypair::generate();
        let bytes = kp1.secret_bytes();
        let kp2 = InferenceKeypair::from_secret_bytes(&bytes);
        assert_eq!(kp1.public_key().as_bytes(), kp2.public_key().as_bytes());
    }

    #[test]
    fn test_encrypt_decrypt_request_roundtrip() {
        let host = InferenceKeypair::generate();
        let body = br#"{"messages":[{"role":"user","content":"hello"}]}"#;

        let (payload, _session) =
            encrypt_inference_request(body, "test-model", host.public_key()).unwrap();

        assert_eq!(payload.model, "test-model");
        assert!(payload.encrypted);

        let decrypted = decrypt_inference_request(&payload, host.secret_key()).unwrap();
        assert_eq!(decrypted, body);
    }

    #[test]
    fn test_encrypt_decrypt_response_roundtrip() {
        let host = InferenceKeypair::generate();
        let session = EphemeralSession::new();
        let chunk = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n";

        let encrypted = encrypt_response_chunk(chunk, &host, &session.public).unwrap();

        let decrypted = decrypt_response_chunk(&encrypted, &session, host.public_key()).unwrap();
        assert_eq!(decrypted, chunk);
    }

    #[test]
    fn test_wrong_key_fails_decryption() {
        let host = InferenceKeypair::generate();
        let wrong_host = InferenceKeypair::generate();
        let body = b"secret prompt";

        let (payload, _session) =
            encrypt_inference_request(body, "model", host.public_key()).unwrap();

        let result = decrypt_inference_request(&payload, wrong_host.secret_key());
        assert!(result.is_err());
    }

    #[test]
    fn test_response_sender_mismatch_rejected() {
        let host = InferenceKeypair::generate();
        let fake_host = InferenceKeypair::generate();
        let session = EphemeralSession::new();
        let chunk = b"response data";

        // Encrypt with fake host
        let encrypted = encrypt_response_chunk(chunk, &fake_host, &session.public).unwrap();

        // Try to decrypt expecting the real host's key
        let result = decrypt_response_chunk(&encrypted, &session, host.public_key());
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_secrecy_different_sessions() {
        let host = InferenceKeypair::generate();
        let body = b"same body";

        let (payload1, _s1) = encrypt_inference_request(body, "m", host.public_key()).unwrap();
        let (payload2, _s2) = encrypt_inference_request(body, "m", host.public_key()).unwrap();

        // Same plaintext → different ciphertexts (different ephemeral keys + nonces)
        assert_ne!(payload1.ciphertext, payload2.ciphertext);
        assert_ne!(payload1.ephemeral_public_key, payload2.ephemeral_public_key);
    }

    #[test]
    fn test_parse_invalid_public_key() {
        assert!(parse_public_key("not-base64!!!").is_err());
        assert!(parse_public_key("dG9vc2hvcnQ=").is_err()); // "tooshort" — not 32 bytes
    }

    #[test]
    fn test_truncated_ciphertext_fails() {
        let payload = EncryptedInferencePayload {
            model: "m".into(),
            ephemeral_public_key: InferenceKeypair::generate().public_key_base64(),
            ciphertext: base64::engine::general_purpose::STANDARD.encode(&[0u8; 10]), // too short
            encrypted: true,
        };
        let host = InferenceKeypair::generate();
        assert!(decrypt_inference_request(&payload, host.secret_key()).is_err());
    }
}
