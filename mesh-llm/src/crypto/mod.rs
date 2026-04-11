mod envelope;
mod error;
mod keychain;
mod keys;
mod keystore;
mod ownership;

pub use self::envelope::{open_message, seal_message, OpenedMessage, SignedEncryptedEnvelope};
pub use self::error::CryptoError;
pub use self::keychain::{
    delete_secret as keychain_delete, get_secret as keychain_get,
    is_available as keychain_available, load_owner_keypair_from_keychain,
    owner_account_for_path as owner_keychain_account_for_path, save_keystore_with_keychain,
    set_secret as keychain_set, OwnerKeychainLoadError, DEFAULT_OWNER_ACCOUNT, KEYCHAIN_SERVICE,
};
pub use self::keys::{owner_id_from_verifying_key, OwnerKeypair};
pub(crate) use self::keystore::write_keystore_bytes_atomically;
pub use self::keystore::{
    default_keystore_path, keystore_exists, keystore_metadata, load_keystore, save_keystore,
    KeystoreInfo,
};
pub use self::ownership::{
    certificate_needs_renewal, default_node_ownership_path, default_trust_store_path,
    load_node_ownership, load_trust_store, save_node_ownership, save_trust_store,
    sign_node_ownership, verify_node_ownership, NodeOwnershipClaim, OwnershipStatus,
    OwnershipSummary, SignedNodeOwnership, TrustPolicy, TrustStore,
    DEFAULT_NODE_CERT_LIFETIME_SECS, DEFAULT_NODE_CERT_RENEW_WINDOW_SECS,
};
