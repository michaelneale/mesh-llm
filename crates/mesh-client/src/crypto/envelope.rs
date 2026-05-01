use crypto_box::aead::{Aead, AeadCore, OsRng as CryptoOsRng};
use crypto_box::SalsaBox;
use ed25519_dalek::{Signer, Verifier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::CryptoError;
use super::keys::OwnerKeypair;

/// A signed-then-encrypted envelope for confidential control messages.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SignedEncryptedEnvelope {
    pub version: u32,
    pub sender_owner_id: String,
    pub sender_sign_public_key: String,
    pub sender_box_public_key: String,
    pub recipient_box_public_key: String,
    pub message_type: String,
    pub timestamp_unix_ms: u64,
    pub nonce: String,
    pub ciphertext: String,
}

/// The decrypted and verified message contents.
#[derive(Debug)]
pub struct OpenedMessage {
    pub sender_owner_id: String,
    pub sender_sign_public_key: [u8; 32],
    pub sender_box_public_key: [u8; 32],
    pub message_type: String,
    pub timestamp_unix_ms: u64,
    pub payload: Vec<u8>,
}

/// Inner plaintext: payload + detached signature.
#[derive(Serialize, Deserialize)]
struct InnerPayload {
    payload: Vec<u8>,
    signature: Vec<u8>,
}

/// Build the canonical bytes that get signed.
///
/// Includes all metadata fields + a hash of the payload to bind the signature
/// to both the envelope context and the message content.
fn canonical_signed_bytes(
    version: u32,
    sender_owner_id: &str,
    sender_box_public_key: &[u8],
    recipient_box_public_key: &[u8],
    message_type: &str,
    timestamp_unix_ms: u64,
    payload: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::new();
    let sender_owner_id_bytes = sender_owner_id.as_bytes();
    let message_type_bytes = message_type.as_bytes();

    // Domain separation tag to prevent cross-protocol signature reuse.
    buf.extend_from_slice(b"mesh-llm-envelope-v1:");
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&(sender_owner_id_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(sender_owner_id_bytes);
    buf.extend_from_slice(sender_box_public_key);
    buf.extend_from_slice(recipient_box_public_key);
    buf.extend_from_slice(&(message_type_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(message_type_bytes);
    buf.extend_from_slice(&timestamp_unix_ms.to_le_bytes());
    // Include a hash of the payload rather than the raw payload to keep
    // the signed data compact for large payloads.
    let payload_hash = Sha256::digest(payload);
    buf.extend_from_slice(&payload_hash);
    buf
}

/// Sign and encrypt a message for a specific recipient.
pub fn seal_message(
    sender: &OwnerKeypair,
    recipient_box_public_key: &crypto_box::PublicKey,
    message_type: &str,
    payload: &[u8],
    timestamp_unix_ms: u64,
) -> Result<SignedEncryptedEnvelope, CryptoError> {
    let version = 1u32;
    let sender_owner_id = sender.owner_id();
    let sender_box_pk = sender.encryption_public_key();

    // 1. Build canonical bytes and sign.
    let signed_bytes = canonical_signed_bytes(
        version,
        &sender_owner_id,
        sender_box_pk.as_bytes(),
        recipient_box_public_key.as_bytes(),
        message_type,
        timestamp_unix_ms,
        payload,
    );
    let signature = sender.signing.sign(&signed_bytes);

    // 2. Build inner payload with detached signature.
    let inner = InnerPayload {
        payload: payload.to_vec(),
        signature: signature.to_bytes().to_vec(),
    };
    let inner_bytes = serde_json::to_vec(&inner)?;

    // 3. Encrypt with crypto_box (XSalsa20Poly1305).
    let salsa_box = SalsaBox::new(recipient_box_public_key, &sender.encryption);
    let nonce = SalsaBox::generate_nonce(&mut CryptoOsRng);
    let ct = salsa_box
        .encrypt(&nonce, inner_bytes.as_ref())
        .map_err(|_| CryptoError::VerificationFailed {
            reason: "encryption failed".into(),
        })?;

    Ok(SignedEncryptedEnvelope {
        version,
        sender_owner_id,
        sender_sign_public_key: hex::encode(sender.verifying_key().as_bytes()),
        sender_box_public_key: hex::encode(sender_box_pk.as_bytes()),
        recipient_box_public_key: hex::encode(recipient_box_public_key.as_bytes()),
        message_type: message_type.to_string(),
        timestamp_unix_ms,
        nonce: hex::encode(nonce),
        ciphertext: hex::encode(ct),
    })
}

/// Decrypt and verify an envelope addressed to this recipient.
pub fn open_message(
    recipient: &OwnerKeypair,
    envelope: &SignedEncryptedEnvelope,
) -> Result<OpenedMessage, CryptoError> {
    // 0. Reject unknown envelope versions.
    if envelope.version != 1 {
        return Err(CryptoError::VerificationFailed {
            reason: format!("unsupported envelope version: {}", envelope.version),
        });
    }

    // 1. Parse sender public keys.
    let sender_sign_pk_bytes: [u8; 32] = hex::decode(&envelope.sender_sign_public_key)
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "bad sender signing key hex".into(),
        })?
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "sender signing key must be 32 bytes".into(),
        })?;

    let sender_box_pk_bytes: [u8; 32] = hex::decode(&envelope.sender_box_public_key)
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "bad sender box key hex".into(),
        })?
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "sender box key must be 32 bytes".into(),
        })?;

    let recipient_box_pk_bytes: [u8; 32] = hex::decode(&envelope.recipient_box_public_key)
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "bad recipient box key hex".into(),
        })?
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "recipient box key must be 32 bytes".into(),
        })?;

    let sender_box_pk = crypto_box::PublicKey::from(sender_box_pk_bytes);

    // 2. Verify that the envelope's claimed recipient key matches the actual recipient.
    // This prevents an attacker from encrypting to the correct recipient while claiming
    // a different recipient in the signed metadata.
    let actual_recipient_box_pk_bytes = *recipient.encryption_public_key().as_bytes();
    if recipient_box_pk_bytes != actual_recipient_box_pk_bytes {
        return Err(CryptoError::VerificationFailed {
            reason: "recipient_box_public_key does not match recipient encryption public key"
                .into(),
        });
    }

    // 3. Verify sender_owner_id matches the signing key (prevents identity spoofing).
    let sender_verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&sender_sign_pk_bytes)
        .map_err(|_| CryptoError::InvalidSignature)?;
    let expected_owner_id = super::keys::owner_id_from_verifying_key(&sender_verifying_key);
    if envelope.sender_owner_id != expected_owner_id {
        return Err(CryptoError::VerificationFailed {
            reason: "sender_owner_id does not match signing public key".into(),
        });
    }

    // 4. Decrypt.
    let nonce_bytes = hex::decode(&envelope.nonce).map_err(|_| CryptoError::DecryptionFailed)?;
    if nonce_bytes.len() != 24 {
        return Err(CryptoError::DecryptionFailed);
    }
    let nonce = crypto_box::Nonce::from_slice(&nonce_bytes);
    let ct = hex::decode(&envelope.ciphertext).map_err(|_| CryptoError::DecryptionFailed)?;

    let salsa_box = SalsaBox::new(&sender_box_pk, &recipient.encryption);
    let inner_bytes = salsa_box
        .decrypt(nonce, ct.as_ref())
        .map_err(|_| CryptoError::DecryptionFailed)?;

    // 5. Parse inner payload.
    let inner: InnerPayload =
        serde_json::from_slice(&inner_bytes).map_err(|_| CryptoError::DecryptionFailed)?;

    // 6. Verify signature.
    let signed_bytes = canonical_signed_bytes(
        envelope.version,
        &envelope.sender_owner_id,
        &sender_box_pk_bytes,
        &recipient_box_pk_bytes,
        &envelope.message_type,
        envelope.timestamp_unix_ms,
        &inner.payload,
    );

    let sig_bytes: [u8; 64] = inner
        .signature
        .try_into()
        .map_err(|_| CryptoError::InvalidSignature)?;
    let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);

    sender_verifying_key
        .verify(&signed_bytes, &signature)
        .map_err(|_| CryptoError::InvalidSignature)?;

    Ok(OpenedMessage {
        sender_owner_id: expected_owner_id,
        sender_sign_public_key: sender_sign_pk_bytes,
        sender_box_public_key: sender_box_pk_bytes,
        message_type: envelope.message_type.clone(),
        timestamp_unix_ms: envelope.timestamp_unix_ms,
        payload: inner.payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_open_round_trip() {
        let sender = OwnerKeypair::generate();
        let recipient = OwnerKeypair::generate();

        let payload = b"hello, mesh-llm!";
        let timestamp = 1_700_000_000_000u64;

        let envelope = seal_message(
            &sender,
            &recipient.encryption_public_key(),
            "test.message",
            payload,
            timestamp,
        )
        .unwrap();

        let opened = open_message(&recipient, &envelope).unwrap();
        assert_eq!(opened.payload, payload);
        assert_eq!(opened.message_type, "test.message");
        assert_eq!(opened.timestamp_unix_ms, timestamp);
        assert_eq!(opened.sender_owner_id, sender.owner_id());
    }

    #[test]
    fn wrong_recipient_cannot_decrypt() {
        let sender = OwnerKeypair::generate();
        let recipient = OwnerKeypair::generate();
        let wrong_recipient = OwnerKeypair::generate();

        let envelope = seal_message(
            &sender,
            &recipient.encryption_public_key(),
            "secret",
            b"classified",
            0,
        )
        .unwrap();

        let result = open_message(&wrong_recipient, &envelope);
        assert!(result.is_err(), "wrong recipient should fail to decrypt");
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let sender = OwnerKeypair::generate();
        let recipient = OwnerKeypair::generate();

        let mut envelope = seal_message(
            &sender,
            &recipient.encryption_public_key(),
            "test",
            b"payload",
            0,
        )
        .unwrap();

        // Flip a byte in the ciphertext.
        let mut ct_bytes = hex::decode(&envelope.ciphertext).unwrap();
        if let Some(byte) = ct_bytes.last_mut() {
            *byte ^= 0xff;
        }
        envelope.ciphertext = hex::encode(&ct_bytes);

        let result = open_message(&recipient, &envelope);
        assert!(result.is_err(), "tampered ciphertext should fail");
    }

    #[test]
    fn spoofed_owner_id_rejected() {
        let sender = OwnerKeypair::generate();
        let recipient = OwnerKeypair::generate();

        let mut envelope = seal_message(
            &sender,
            &recipient.encryption_public_key(),
            "test",
            b"payload",
            0,
        )
        .unwrap();

        // Spoof the owner_id to a different value.
        envelope.sender_owner_id =
            "0000000000000000000000000000000000000000000000000000000000000000".into();

        let result = open_message(&recipient, &envelope);
        assert!(
            matches!(result, Err(CryptoError::VerificationFailed { .. })),
            "spoofed owner_id should be rejected"
        );
    }

    #[test]
    fn unknown_envelope_version_rejected() {
        let sender = OwnerKeypair::generate();
        let recipient = OwnerKeypair::generate();

        let mut envelope = seal_message(
            &sender,
            &recipient.encryption_public_key(),
            "test",
            b"payload",
            0,
        )
        .unwrap();

        envelope.version = 99;

        let result = open_message(&recipient, &envelope);
        assert!(
            matches!(result, Err(CryptoError::VerificationFailed { .. })),
            "unknown version should be rejected"
        );
    }

    #[test]
    fn mismatched_recipient_key_rejected() {
        let sender = OwnerKeypair::generate();
        let recipient = OwnerKeypair::generate();

        let mut envelope = seal_message(
            &sender,
            &recipient.encryption_public_key(),
            "test",
            b"payload",
            0,
        )
        .unwrap();

        // Claim a different recipient key in the envelope metadata.
        let other = OwnerKeypair::generate();
        envelope.recipient_box_public_key = hex::encode(other.encryption_public_key().as_bytes());

        let result = open_message(&recipient, &envelope);
        assert!(
            matches!(result, Err(CryptoError::VerificationFailed { .. })),
            "mismatched recipient key should be rejected"
        );
    }

    #[test]
    fn canonical_bytes_length_prefix_variable_fields() {
        let sender_box_key = [7u8; 32];
        let recipient_box_key = [9u8; 32];

        let left = canonical_signed_bytes(
            1,
            "ab",
            &sender_box_key,
            &recipient_box_key,
            "c",
            42,
            b"payload",
        );
        let right = canonical_signed_bytes(
            1,
            "a",
            &sender_box_key,
            &recipient_box_key,
            "bc",
            42,
            b"payload",
        );

        assert_ne!(left, right, "variable-length fields must be unambiguous");
    }
}
