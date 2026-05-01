//! Thin wrapper around `keyring` for storing the owner keystore unlock secret
//! in the OS-native credential store (macOS Keychain, Windows Credential
//! Manager, Linux Secret Service).
//!
//! This module is backend-neutral: it stores and retrieves opaque UTF-8
//! strings by (service, account) key. Callers decide whether the stored value
//! is a passphrase, hex-encoded key bytes, or something else.

use std::path::Path;

use base64::Engine as _;
use keyring::Entry;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use super::error::CryptoError;
use super::keys::OwnerKeypair;
use super::keystore::{load_keystore, save_keystore, write_keystore_bytes_atomically};

/// Service name used for all mesh-llm keychain entries.
pub const KEYCHAIN_SERVICE: &str = "mesh-llm";

/// Default account name for the owner keystore unlock secret.
pub const DEFAULT_OWNER_ACCOUNT: &str = "owner-keystore";

#[derive(Debug)]
pub enum OwnerKeychainLoadError {
    NoEntry,
    Crypto(CryptoError),
}

/// Store a secret in the OS keychain under (service, account).
///
/// Overwrites any existing entry with the same (service, account) pair.
pub fn set_secret(service: &str, account: &str, secret: &str) -> Result<(), CryptoError> {
    let entry = Entry::new(service, account).map_err(map_err)?;
    entry.set_password(secret).map_err(map_err)
}

/// Retrieve a secret from the OS keychain by (service, account).
///
/// Returns `Ok(None)` when no entry exists for the given pair. Returns
/// `Err(CryptoError)` when the keychain backend is unavailable or errored.
pub fn get_secret(service: &str, account: &str) -> Result<Option<String>, CryptoError> {
    let entry = Entry::new(service, account).map_err(map_err)?;
    match entry.get_password() {
        Ok(s) => Ok(Some(s)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(map_err(e)),
    }
}

/// Delete a secret from the OS keychain by (service, account).
///
/// Returns `Ok(false)` when no entry existed to delete, `Ok(true)` when an
/// entry was removed.
pub fn delete_secret(service: &str, account: &str) -> Result<bool, CryptoError> {
    let entry = Entry::new(service, account).map_err(map_err)?;
    match entry.delete_credential() {
        Ok(()) => Ok(true),
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(e) => Err(map_err(e)),
    }
}

/// Probe whether a native keychain backend is reachable on this host.
///
/// On Linux this typically means a Secret Service daemon (gnome-keyring,
/// KWallet) is running and reachable over D-Bus. On macOS and Windows the
/// backend is effectively always available.
///
/// Implementation: attempts a read on a probe account. `NoEntry` counts as
/// available (the backend answered). A `PlatformFailure` or similar counts as
/// unavailable.
pub fn is_available() -> bool {
    const PROBE_ACCOUNT: &str = "__availability-probe__";
    let entry = match Entry::new(KEYCHAIN_SERVICE, PROBE_ACCOUNT) {
        Ok(e) => e,
        Err(_) => return false,
    };
    match entry.get_password() {
        Ok(_) => true,
        Err(keyring::Error::NoEntry) => true,
        Err(_) => false,
    }
}

/// Derive a stable keychain account name from a keystore path.
///
/// Form: `owner-keystore:<16 hex chars of sha256(normalized_absolute_path)>`.
/// The prefix matches `DEFAULT_OWNER_ACCOUNT` so users can identify mesh-llm
/// entries. The hash bounds the length and avoids leaking full paths into
/// keychain account fields, while still giving every keystore path its own
/// slot.
///
/// Uses `std::path::absolute` rather than `canonicalize` so the result is
/// consistent whether the file exists yet (during `auth init`) or already
/// exists (during status/load). `canonicalize` would resolve symlinks and
/// only succeed on existing files, producing different account names for
/// the same logical path at different lifecycle points.
///
/// On Windows the path is lowercased before hashing, because the default
/// filesystem (NTFS) is case-insensitive; without this, `Owner-Keystore.json`
/// and `owner-keystore.json` would map to different keychain entries despite
/// being the same file.
pub fn owner_account_for_path(path: &Path) -> String {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let path_str = absolute.to_string_lossy();
    #[cfg(windows)]
    let normalized = path_str.to_lowercase();
    #[cfg(not(windows))]
    let normalized = path_str.into_owned();
    let hash = Sha256::digest(normalized.as_bytes());
    format!("{}:{}", DEFAULT_OWNER_ACCOUNT, &hex::encode(hash)[..16])
}

/// Save an owner keystore encrypted with a random passphrase and bind that
/// passphrase to the OS keychain account derived from `path`.
///
/// The keystore is written before the keychain is updated. If the keychain
/// write fails, the keystore is restored to its previous bytes (or removed if
/// it did not exist), so ordinary failures do not strand the file and keychain
/// in different states.
pub fn save_keystore_with_keychain(
    path: &Path,
    keypair: &OwnerKeypair,
    overwrite: bool,
) -> Result<String, CryptoError> {
    let account = owner_account_for_path(path);
    let previous_keystore = if path.exists() {
        Some(std::fs::read(path)?)
    } else {
        None
    };
    // Snapshot any existing keychain entry so rollback is symmetric: if
    // set_secret partially modifies or clears the entry before failing, we
    // restore the original secret (or delete the entry if there was none).
    let previous_keychain = get_secret(KEYCHAIN_SERVICE, &account)?;
    let generated = Zeroizing::new(generate_random_passphrase());

    save_keystore(path, keypair, Some(generated.as_str()), overwrite)?;

    if let Err(err) = set_secret(KEYCHAIN_SERVICE, &account, generated.as_str()) {
        let _ = rollback_keystore(path, previous_keystore.as_deref());
        let _ = rollback_keychain_secret(KEYCHAIN_SERVICE, &account, previous_keychain.as_deref());
        return Err(err);
    }

    Ok(account)
}

/// Try to load an encrypted keystore using a passphrase stored in the OS
/// keychain under the account derived from the keystore path.
pub fn load_owner_keypair_from_keychain(
    path: &Path,
) -> Result<OwnerKeypair, OwnerKeychainLoadError> {
    if !is_available() {
        return Err(OwnerKeychainLoadError::Crypto(
            CryptoError::KeychainUnavailable {
                reason: "no backend reachable on this host".into(),
            },
        ));
    }
    let account = owner_account_for_path(path);
    let pass = get_secret(KEYCHAIN_SERVICE, &account).map_err(OwnerKeychainLoadError::Crypto)?;
    let pass = match pass {
        Some(p) => Zeroizing::new(p),
        None => return Err(OwnerKeychainLoadError::NoEntry),
    };
    load_keystore(path, Some(pass.as_str())).map_err(OwnerKeychainLoadError::Crypto)
}

fn map_err(e: keyring::Error) -> CryptoError {
    match e {
        keyring::Error::NoStorageAccess(err) => CryptoError::KeychainAccessDenied {
            reason: err.to_string(),
        },
        other => CryptoError::KeychainUnavailable {
            reason: other.to_string(),
        },
    }
}

fn generate_random_passphrase() -> String {
    let bytes: [u8; 32] = rand::random();
    base64::engine::general_purpose::STANDARD_NO_PAD.encode(bytes)
}

fn rollback_keystore(path: &Path, previous_keystore: Option<&[u8]>) -> Result<(), CryptoError> {
    match previous_keystore {
        Some(previous) => write_keystore_bytes_atomically(path, previous),
        None => {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
            Ok(())
        }
    }
}

/// Restore a keychain entry to its previous state.
///
/// If `previous` is `Some`, the secret is written back; if `None`, the entry
/// is deleted so no orphan is left behind.
fn rollback_keychain_secret(
    service: &str,
    account: &str,
    previous: Option<&str>,
) -> Result<(), CryptoError> {
    match previous {
        Some(secret) => set_secret(service, account, secret),
        None => {
            delete_secret(service, account)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // Keychain tests hit the real OS credential store, so use a unique account
    // per test run and clean up afterward. They run serially because some
    // backends (e.g. Windows Credential Manager) don't handle concurrent
    // mutations from the same process cleanly. Tests skip themselves when
    // no keychain backend is reachable (e.g. CI without a Secret Service).
    fn test_account(tag: &str) -> String {
        format!("test-{}-{}", tag, rand::random::<u64>())
    }

    #[test]
    #[serial]
    fn round_trip_set_get_delete() {
        if !is_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }
        let account = test_account("round-trip");
        let secret = "correct horse battery staple";

        set_secret(KEYCHAIN_SERVICE, &account, secret).unwrap();
        let got = get_secret(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(got.as_deref(), Some(secret));

        let removed = delete_secret(KEYCHAIN_SERVICE, &account).unwrap();
        assert!(removed);

        let after = get_secret(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(after, None);
    }

    #[test]
    #[serial]
    fn get_missing_entry_returns_none() {
        if !is_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }
        let account = test_account("missing");
        let got = get_secret(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(got, None);
    }

    #[test]
    #[serial]
    fn delete_missing_entry_returns_false() {
        if !is_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }
        let account = test_account("delete-missing");
        let removed = delete_secret(KEYCHAIN_SERVICE, &account).unwrap();
        assert!(!removed);
    }

    #[test]
    #[serial]
    fn overwrite_existing_entry() {
        if !is_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }
        let account = test_account("overwrite");
        set_secret(KEYCHAIN_SERVICE, &account, "first").unwrap();
        set_secret(KEYCHAIN_SERVICE, &account, "second").unwrap();

        let got = get_secret(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(got.as_deref(), Some("second"));

        delete_secret(KEYCHAIN_SERVICE, &account).ok();
    }

    #[test]
    fn generated_passphrase_has_expected_entropy() {
        let p1 = generate_random_passphrase();
        let p2 = generate_random_passphrase();
        assert_ne!(p1, p2, "two passphrases should not collide");
        assert_eq!(p1.len(), 43);
        assert_eq!(p2.len(), 43);
    }

    #[test]
    fn account_name_is_deterministic_for_same_path() {
        let a1 = owner_account_for_path(Path::new("/tmp/does-not-exist-1.json"));
        let a2 = owner_account_for_path(Path::new("/tmp/does-not-exist-1.json"));
        assert_eq!(a1, a2);
    }

    #[test]
    fn account_names_differ_for_different_paths() {
        let a1 = owner_account_for_path(Path::new("/tmp/keystore-a.json"));
        let a2 = owner_account_for_path(Path::new("/tmp/keystore-b.json"));
        assert_ne!(a1, a2);
    }

    #[test]
    fn account_name_shape() {
        let account = owner_account_for_path(Path::new("/tmp/x.json"));
        assert!(account.starts_with(DEFAULT_OWNER_ACCOUNT));
        assert!(account.contains(':'));
        assert_eq!(account.len(), DEFAULT_OWNER_ACCOUNT.len() + 1 + 16);
    }

    #[cfg(windows)]
    #[test]
    fn account_name_is_case_insensitive_on_windows() {
        let lower = owner_account_for_path(Path::new(r"C:\tmp\Owner\keystore.json"));
        let upper = owner_account_for_path(Path::new(r"C:\TMP\OWNER\KEYSTORE.JSON"));
        assert_eq!(
            lower, upper,
            "NTFS paths differing only in case must hash the same"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn account_name_is_case_sensitive_on_unix() {
        let lower = owner_account_for_path(Path::new("/tmp/owner/keystore.json"));
        let upper = owner_account_for_path(Path::new("/tmp/OWNER/KEYSTORE.JSON"));
        assert_ne!(
            lower, upper,
            "POSIX paths differing in case must not collide"
        );
    }
}
