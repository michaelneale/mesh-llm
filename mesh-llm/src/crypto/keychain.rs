//! Thin wrapper around `keyring` for storing the owner keystore unlock secret
//! in the OS-native credential store (macOS Keychain, Windows Credential
//! Manager, Linux Secret Service).
//!
//! This module is backend-neutral: it stores and retrieves opaque UTF-8
//! strings by (service, account) key. Callers decide whether the stored value
//! is a passphrase, hex-encoded key bytes, or something else.

use keyring::Entry;

use super::error::CryptoError;

/// Service name used for all mesh-llm keychain entries.
pub const KEYCHAIN_SERVICE: &str = "mesh-llm";

/// Default account name for the owner keystore unlock secret.
pub const DEFAULT_OWNER_ACCOUNT: &str = "owner-keystore";

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

fn map_err(e: keyring::Error) -> CryptoError {
    CryptoError::KeychainUnavailable {
        reason: e.to_string(),
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
}
