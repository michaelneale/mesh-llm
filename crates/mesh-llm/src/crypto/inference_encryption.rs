//! E2E inference encryption using X25519 + XSalsa20-Poly1305 (NaCl box).
//!
//! Encrypts inference payloads (prompts, responses) from the API entry
//! point to the host node. Intermediate tunnel relays see only ciphertext.
//!
//! Each node generates an X25519 keypair at startup. The public key is
//! advertised in gossip. Per-request, the sender generates an ephemeral
//! keypair for forward secrecy.

use base64::Engine;
use crypto_box::aead::{Aead, AeadCore, OsRng};
use crypto_box::SalsaBox;
use serde::{Deserialize, Serialize};

use super::error::CryptoError;

/// Magic byte prefix for E2E encrypted tunnel payloads.
pub const ENCRYPTED_TUNNEL_MAGIC: u8 = 0xE1;

// ── Inference keypair ─────────────────────────────────────────────

/// X25519 keypair used for inference payload encryption.
/// The private key stays on this node. The public key is advertised in gossip.
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

    /// Reconstruct from raw secret key bytes.
    pub fn from_secret_bytes(bytes: &[u8; 32]) -> Self {
        let secret = crypto_box::SecretKey::from(*bytes);
        let public = secret.public_key();
        Self { secret, public }
    }

    /// The public key as base64 for gossip advertisement.
    pub fn public_key_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.public.as_bytes())
    }

    /// Raw secret key bytes.
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.secret.to_bytes()
    }

    pub fn public_key(&self) -> &crypto_box::PublicKey {
        &self.public
    }

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
}

/// Encrypted response from host back to consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedResponse {
    /// NaCl box ciphertext: nonce (24 bytes) || encrypted body (base64).
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
    pub fn new() -> Self {
        let secret = crypto_box::SecretKey::generate(&mut OsRng);
        let public = secret.public_key();
        Self { secret, public }
    }

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
/// Returns the encrypted wire payload and the ephemeral session (needed to decrypt response).
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

    // Wire: nonce (24) || ciphertext
    let mut wire = Vec::with_capacity(24 + ciphertext.len());
    wire.extend_from_slice(&nonce);
    wire.extend_from_slice(&ciphertext);

    let payload = EncryptedInferencePayload {
        model: model.to_string(),
        ephemeral_public_key: session.public_key_base64(),
        ciphertext: base64::engine::general_purpose::STANDARD.encode(&wire),
    };

    Ok((payload, session))
}

/// Decrypt an inference request received by this node.
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

/// Encrypt a response from the host back to the consumer.
pub fn encrypt_response(
    data: &[u8],
    own_key: &InferenceKeypair,
    recipient_pub: &crypto_box::PublicKey,
) -> Result<EncryptedResponse, CryptoError> {
    let salsa_box = SalsaBox::new(recipient_pub, own_key.secret_key());
    let nonce = crypto_box::SalsaBox::generate_nonce(&mut OsRng);
    let ciphertext = salsa_box
        .encrypt(&nonce, data)
        .map_err(|_| CryptoError::EncryptionFailed)?;

    let mut wire = Vec::with_capacity(24 + ciphertext.len());
    wire.extend_from_slice(&nonce);
    wire.extend_from_slice(&ciphertext);

    Ok(EncryptedResponse {
        ciphertext: base64::engine::general_purpose::STANDARD.encode(&wire),
        sender_public_key: own_key.public_key_base64(),
    })
}

/// Decrypt a response received by the consumer.
pub fn decrypt_response(
    resp: &EncryptedResponse,
    session: &EphemeralSession,
    expected_sender_pub: &crypto_box::PublicKey,
) -> Result<Vec<u8>, CryptoError> {
    let sender_pub = parse_public_key(&resp.sender_public_key)?;
    if sender_pub.as_bytes() != expected_sender_pub.as_bytes() {
        return Err(CryptoError::InvalidKeyMaterial {
            reason: "response sender key mismatch".into(),
        });
    }

    let wire = base64::engine::general_purpose::STANDARD
        .decode(&resp.ciphertext)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad response ciphertext base64: {e}"),
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
    fn roundtrip_request() {
        let host = InferenceKeypair::generate();
        let body = br#"{"messages":[{"role":"user","content":"hello"}]}"#;

        let (payload, _session) =
            encrypt_inference_request(body, "test-model", host.public_key()).unwrap();
        assert_eq!(payload.model, "test-model");

        let decrypted = decrypt_inference_request(&payload, host.secret_key()).unwrap();
        assert_eq!(decrypted, body);
    }

    #[test]
    fn roundtrip_response() {
        let host = InferenceKeypair::generate();
        let session = EphemeralSession::new();
        let data = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n";

        let encrypted = encrypt_response(data, &host, &session.public).unwrap();
        let decrypted = decrypt_response(&encrypted, &session, host.public_key()).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn wrong_key_fails() {
        let host = InferenceKeypair::generate();
        let wrong = InferenceKeypair::generate();
        let body = b"secret";

        let (payload, _) = encrypt_inference_request(body, "m", host.public_key()).unwrap();
        assert!(decrypt_inference_request(&payload, wrong.secret_key()).is_err());
    }

    #[test]
    fn forward_secrecy() {
        let host = InferenceKeypair::generate();
        let body = b"same";

        let (p1, _) = encrypt_inference_request(body, "m", host.public_key()).unwrap();
        let (p2, _) = encrypt_inference_request(body, "m", host.public_key()).unwrap();
        assert_ne!(p1.ciphertext, p2.ciphertext);
        assert_ne!(p1.ephemeral_public_key, p2.ephemeral_public_key);
    }

    #[test]
    fn response_sender_mismatch_rejected() {
        let host = InferenceKeypair::generate();
        let fake = InferenceKeypair::generate();
        let session = EphemeralSession::new();

        let encrypted = encrypt_response(b"data", &fake, &session.public).unwrap();
        assert!(decrypt_response(&encrypted, &session, host.public_key()).is_err());
    }
}
