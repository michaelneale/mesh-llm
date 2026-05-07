use thiserror::Error;

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("keystore not found at {path}")]
    KeystoreNotFound { path: String },

    #[error("keystore already exists at {path} (use --force to overwrite)")]
    KeystoreAlreadyExists { path: String },

    #[error("wrong passphrase or corrupted keystore")]
    DecryptionFailed,

    #[error("invalid signature")]
    InvalidSignature,

    #[error("unsupported keystore version: {version}")]
    UnsupportedVersion { version: u32 },

    #[error("envelope verification failed: {reason}")]
    VerificationFailed { reason: String },

    #[error("invalid key material: {reason}")]
    InvalidKeyMaterial { reason: String },

    #[error("missing owner passphrase and no interactive terminal is available")]
    MissingPassphrase,

    #[error("OS keychain unavailable: {reason}")]
    KeychainUnavailable { reason: String },

    #[error("OS keychain access denied: {reason}")]
    KeychainAccessDenied { reason: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
