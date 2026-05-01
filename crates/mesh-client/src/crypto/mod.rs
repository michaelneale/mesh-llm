mod envelope;
pub mod error;
pub mod keys;
pub mod provider;

pub use self::envelope::{open_message, seal_message, OpenedMessage, SignedEncryptedEnvelope};
pub use self::error::CryptoError;
pub use self::keys::{owner_id_from_verifying_key, OwnerKeypair};
pub use self::provider::{InMemoryKeyProvider, KeyProvider, KeyProviderError};
