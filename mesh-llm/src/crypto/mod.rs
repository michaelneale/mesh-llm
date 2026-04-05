mod envelope;
mod error;
mod keychain;
mod keys;
mod keystore;

pub use self::envelope::{open_message, seal_message, OpenedMessage, SignedEncryptedEnvelope};
pub use self::error::CryptoError;
pub use self::keychain::{
    delete_secret as keychain_delete, get_secret as keychain_get, is_available as keychain_available,
    set_secret as keychain_set, DEFAULT_OWNER_ACCOUNT, KEYCHAIN_SERVICE,
};
pub use self::keys::{owner_id_from_verifying_key, OwnerKeypair};
pub use self::keystore::{
    default_keystore_path, keystore_exists, keystore_metadata, load_keystore, save_keystore,
    KeystoreInfo,
};
