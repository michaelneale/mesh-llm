#![forbid(unsafe_code)]

pub mod envelope;
pub mod error;
pub mod keys;
pub mod provider;

pub use envelope::{open_message, seal_message, OpenedMessage, SignedEncryptedEnvelope};
pub use error::CryptoError;
pub use keys::{owner_id_from_verifying_key, OwnerKeypair};
pub use provider::{InMemoryKeyProvider, KeyProvider, KeyProviderError};
