use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use base64::Engine as _;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::crypto::{
    default_keystore_path, keystore_exists, keystore_metadata, load_keystore, save_keystore,
    OwnerKeypair, DEFAULT_OWNER_ACCOUNT, KEYCHAIN_SERVICE,
};

/// Run `mesh-llm auth init`.
pub(crate) fn run_init(
    owner_key: Option<PathBuf>,
    force: bool,
    no_passphrase: bool,
    keychain: bool,
) -> Result<()> {
    let path = match owner_key {
        Some(p) => p,
        None => default_keystore_path()?,
    };

    if keystore_exists(&path) && !force {
        bail!(
            "Owner keystore already exists at {}\nUse --force to overwrite.",
            path.display()
        );
    }

    let (passphrase, source): (Option<Zeroizing<String>>, PassphraseSource) = if no_passphrase {
        (None, PassphraseSource::None)
    } else if keychain {
        if !crate::crypto::keychain_available() {
            bail!(
                "No OS keychain backend is available on this host.\n\
                 Retry without --keychain to set a passphrase, or with --no-passphrase \
                 to store keys unencrypted."
            );
        }
        let account = keychain_account_for_path(&path);
        // Capture any existing keychain entry so we can roll back cleanly if
        // the keystore write fails after we overwrite it.
        let previous = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account)
            .ok()
            .flatten()
            .map(Zeroizing::new);
        let generated = Zeroizing::new(generate_random_passphrase());
        crate::crypto::keychain_set(KEYCHAIN_SERVICE, &account, generated.as_str())?;
        (
            Some(generated),
            PassphraseSource::Keychain { account, previous },
        )
    } else {
        let pass = Zeroizing::new(rpassword::prompt_password_stderr(
            "Enter passphrase (empty for none): ",
        )?);
        if pass.is_empty() {
            (None, PassphraseSource::None)
        } else {
            let confirm =
                Zeroizing::new(rpassword::prompt_password_stderr("Confirm passphrase: ")?);
            if pass.as_str() != confirm.as_str() {
                bail!("Passphrases do not match.");
            }
            (Some(pass), PassphraseSource::Prompt)
        }
    };
    let encrypted = passphrase.is_some();

    let keypair = OwnerKeypair::generate();
    let owner_id = keypair.owner_id();
    let sign_pk = hex::encode(keypair.verifying_key().as_bytes());
    let enc_pk = hex::encode(keypair.encryption_public_key().as_bytes());

    // If save fails after we've written to the keychain, roll back so we
    // don't orphan a new entry (no previous) or strand an existing keystore
    // whose previous unlock secret we just overwrote.
    if let Err(e) = save_keystore(
        &path,
        &keypair,
        passphrase.as_ref().map(|pass| pass.as_str()),
        force,
    ) {
        if let PassphraseSource::Keychain { account, previous } = &source {
            match previous {
                Some(prev) => {
                    let _ = crate::crypto::keychain_set(KEYCHAIN_SERVICE, account, prev.as_str());
                }
                None => {
                    let _ = crate::crypto::keychain_delete(KEYCHAIN_SERVICE, account);
                }
            }
        }
        return Err(e.into());
    }

    eprintln!();
    eprintln!("Owner keystore created.");
    eprintln!("Owner ID:        {owner_id}");
    eprintln!("Signing key:     {sign_pk}");
    eprintln!("Encryption key:  {enc_pk}");
    eprintln!("Path:            {}", path.display());
    eprintln!("Encrypted:       {}", if encrypted { "yes" } else { "no" });
    match &source {
        PassphraseSource::Keychain { account, .. } => {
            eprintln!("Unlock:          OS keychain (service={KEYCHAIN_SERVICE}, account={account})");
        }
        PassphraseSource::Prompt => {
            eprintln!("Unlock:          passphrase prompt");
        }
        PassphraseSource::None => {}
    }
    eprintln!();
    eprintln!("Next steps:");
    match &source {
        PassphraseSource::Keychain { account, .. } => {
            eprintln!(
                "- This keystore is unlock-bound to this machine's keychain. To share the same \
                 owner identity on another node, retrieve the passphrase from your OS keychain \
                 (service={KEYCHAIN_SERVICE}, account={account}) and enter it there, or re-run \
                 `auth init` with a manual passphrase so the same passphrase can be used \
                 everywhere."
            );
        }
        PassphraseSource::Prompt | PassphraseSource::None => {
            eprintln!(
                "- Copy this keystore to other trusted nodes that should share the same owner identity."
            );
        }
    }
    eprintln!("- Start mesh-llm with --owner-key {}", path.display());

    Ok(())
}

/// Run `mesh-llm auth status`.
pub(crate) fn run_status(owner_key: Option<PathBuf>) -> Result<()> {
    let path = match owner_key {
        Some(p) => p,
        None => default_keystore_path()?,
    };

    if !keystore_exists(&path) {
        eprintln!("No owner keystore found at {}", path.display());
        eprintln!("Run `mesh-llm auth init` to create one.");
        return Ok(());
    }

    let info = keystore_metadata(&path)?;

    eprintln!("Owner keystore:  {}", path.display());
    eprintln!("Status:          present");
    eprintln!(
        "Encrypted:       {}",
        if info.encrypted { "yes" } else { "no" }
    );
    eprintln!("Owner ID:        {}", info.owner_id);
    if let Some(ref spk) = info.signing_public_key {
        eprintln!("Signing key:     {spk}");
    }
    if let Some(ref epk) = info.encryption_public_key {
        eprintln!("Encryption key:  {epk}");
    }
    eprintln!("Created:         {}", info.created_at);

    // Attempt to load so we can report whether this node has the unlock
    // secret (keychain or plaintext) ready for use.
    let load_result = if info.encrypted {
        load_with_keychain(&path)
    } else {
        load_keystore(&path, None)
            .map(|kp| (kp, UnlockSource::Plain))
            .map_err(KeychainLoadError::Crypto)
    };

    match load_result {
        Ok((_, UnlockSource::Plain)) => {
            eprintln!("Keystore:        valid (keys loaded successfully)");
        }
        Ok((_, UnlockSource::Keychain)) => {
            eprintln!("Keystore:        valid (unlocked from OS keychain)");
        }
        Err(KeychainLoadError::NoEntry) => {
            eprintln!(
                "Keystore:        encrypted (no keychain entry for this path; \
                 provide the passphrase when the owner keystore is consumed)"
            );
        }
        Err(KeychainLoadError::Crypto(crate::crypto::CryptoError::KeychainUnavailable {
            reason,
        })) => {
            eprintln!("Keystore:        encrypted (keychain unavailable: {reason})");
        }
        Err(KeychainLoadError::Crypto(e)) => {
            eprintln!("Keystore:        ERROR loading keys: {e}");
        }
    }

    Ok(())
}

// ── Internal helpers ────────────────────────────────────────────────

#[derive(Debug)]
enum PassphraseSource {
    None,
    Prompt,
    Keychain {
        account: String,
        previous: Option<Zeroizing<String>>,
    },
}

#[derive(Debug)]
enum UnlockSource {
    Plain,
    Keychain,
}

#[derive(Debug)]
enum KeychainLoadError {
    NoEntry,
    Crypto(crate::crypto::CryptoError),
}

/// Try to load an encrypted keystore using a passphrase stored in the OS
/// keychain under the account derived from the keystore path.
fn load_with_keychain(path: &Path) -> Result<(OwnerKeypair, UnlockSource), KeychainLoadError> {
    if !crate::crypto::keychain_available() {
        return Err(KeychainLoadError::Crypto(
            crate::crypto::CryptoError::KeychainUnavailable {
                reason: "no backend reachable on this host".into(),
            },
        ));
    }
    let account = keychain_account_for_path(path);
    let pass = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account)
        .map_err(KeychainLoadError::Crypto)?;
    let pass = match pass {
        Some(p) => Zeroizing::new(p),
        None => return Err(KeychainLoadError::NoEntry),
    };
    let kp = load_keystore(path, Some(pass.as_str())).map_err(KeychainLoadError::Crypto)?;
    Ok((kp, UnlockSource::Keychain))
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
/// filesystem (NTFS) is case-insensitive — without this, `Owner-Keystore.json`
/// and `owner-keystore.json` would map to different keychain entries despite
/// being the same file.
fn keychain_account_for_path(path: &Path) -> String {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let path_str = absolute.to_string_lossy();
    #[cfg(windows)]
    let normalized = path_str.to_lowercase();
    #[cfg(not(windows))]
    let normalized = path_str.into_owned();
    let hash = Sha256::digest(normalized.as_bytes());
    format!("{}:{}", DEFAULT_OWNER_ACCOUNT, &hex::encode(hash)[..16])
}

/// Generate a cryptographically random passphrase with 256 bits of entropy,
/// encoded as base64 (no padding) for storage in the keychain.
fn generate_random_passphrase() -> String {
    let bytes: [u8; 32] = rand::random();
    base64::engine::general_purpose::STANDARD_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn generated_passphrase_has_expected_entropy() {
        let p1 = generate_random_passphrase();
        let p2 = generate_random_passphrase();
        assert_ne!(p1, p2, "two passphrases should not collide");
        // 32 bytes → 43 chars base64 no-pad.
        assert_eq!(p1.len(), 43);
        assert_eq!(p2.len(), 43);
    }

    #[test]
    fn account_name_is_deterministic_for_same_path() {
        let a1 = keychain_account_for_path(Path::new("/tmp/does-not-exist-1.json"));
        let a2 = keychain_account_for_path(Path::new("/tmp/does-not-exist-1.json"));
        assert_eq!(a1, a2);
    }

    #[test]
    fn account_names_differ_for_different_paths() {
        let a1 = keychain_account_for_path(Path::new("/tmp/keystore-a.json"));
        let a2 = keychain_account_for_path(Path::new("/tmp/keystore-b.json"));
        assert_ne!(a1, a2);
    }

    #[test]
    fn account_name_shape() {
        let a = keychain_account_for_path(Path::new("/tmp/x.json"));
        assert!(a.starts_with(DEFAULT_OWNER_ACCOUNT));
        assert!(a.contains(':'));
        // prefix + ':' + 16 hex chars
        assert_eq!(a.len(), DEFAULT_OWNER_ACCOUNT.len() + 1 + 16);
    }

    #[cfg(windows)]
    #[test]
    fn account_name_is_case_insensitive_on_windows() {
        let lower = keychain_account_for_path(Path::new(r"C:\tmp\Owner\keystore.json"));
        let upper = keychain_account_for_path(Path::new(r"C:\TMP\OWNER\KEYSTORE.JSON"));
        assert_eq!(lower, upper, "NTFS paths differing only in case must hash the same");
    }

    #[cfg(not(windows))]
    #[test]
    fn account_name_is_case_sensitive_on_unix() {
        let lower = keychain_account_for_path(Path::new("/tmp/owner/keystore.json"));
        let upper = keychain_account_for_path(Path::new("/tmp/OWNER/KEYSTORE.JSON"));
        assert_ne!(lower, upper, "POSIX paths differing in case must not collide");
    }

    #[test]
    #[serial]
    fn force_keychain_save_failure_restores_previous_secret() {
        if !crate::crypto::keychain_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }

        // Use a path whose parent is a FILE rather than a directory. That
        // makes the save_keystore call fail at create_dir_all, triggering
        // our rollback logic.
        let tmp_dir = std::env::temp_dir().join(format!(
            "mesh-llm-force-rollback-{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let blocking_file = tmp_dir.join("blocker");
        std::fs::write(&blocking_file, b"not a directory").unwrap();
        // This path uses `blocker` (a regular file) as if it were a parent
        // directory, so any attempt to create_dir_all against it fails.
        let bad_path = blocking_file.join("owner-keystore.json");

        // Seed the keychain with a "previous" unlock secret at the account
        // that run_init will compute for bad_path.
        let account = keychain_account_for_path(&bad_path);
        let previous_secret = "previous-unlock-secret-do-not-lose";
        crate::crypto::keychain_set(KEYCHAIN_SERVICE, &account, previous_secret).unwrap();

        // Force-init with keychain. Expected to fail because the parent of
        // bad_path is a regular file. Our rollback should restore the
        // previous secret we seeded.
        let result = run_init(Some(bad_path.clone()), true, false, true);
        assert!(result.is_err(), "run_init must fail when save cannot succeed");

        let restored = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(
            restored.as_deref(),
            Some(previous_secret),
            "previous keychain secret must be restored after failed force-init"
        );

        // Cleanup.
        crate::crypto::keychain_delete(KEYCHAIN_SERVICE, &account).ok();
        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    #[serial]
    fn fresh_keychain_save_failure_leaves_no_orphan() {
        if !crate::crypto::keychain_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }

        let tmp_dir = std::env::temp_dir().join(format!(
            "mesh-llm-fresh-rollback-{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let blocking_file = tmp_dir.join("blocker");
        std::fs::write(&blocking_file, b"not a directory").unwrap();
        let bad_path = blocking_file.join("owner-keystore.json");

        let account = keychain_account_for_path(&bad_path);

        // Make sure no entry exists at that account before we start.
        crate::crypto::keychain_delete(KEYCHAIN_SERVICE, &account).ok();

        let result = run_init(Some(bad_path.clone()), false, false, true);
        assert!(result.is_err(), "run_init must fail when save cannot succeed");

        let residual = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(
            residual, None,
            "a fresh init failure must leave no keychain entry behind"
        );

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    #[serial]
    fn init_with_keychain_then_load_round_trip() {
        if !crate::crypto::keychain_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }

        // Use a unique temp path per run so tests can be repeated / run in
        // parallel without stomping each other.
        let dir = std::env::temp_dir().join(format!(
            "mesh-llm-keychain-rt-{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("owner-keystore.json");

        run_init(Some(path.clone()), false, false, true)
            .expect("auth init --keychain should succeed");

        // Keystore file exists and is encrypted.
        assert!(path.exists(), "keystore file should exist");
        let info = keystore_metadata(&path).unwrap();
        assert!(info.encrypted, "keystore should be encrypted when using keychain");

        // Keychain has an entry for this path's account.
        let account = keychain_account_for_path(&path);
        let stored = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
        assert!(
            stored.is_some(),
            "keychain must have a passphrase entry for this keystore path"
        );

        // Loading through the keychain helper succeeds and yields the same
        // owner id the keystore metadata advertises.
        let (kp, src) = load_with_keychain(&path).expect("load via keychain must succeed");
        assert_eq!(kp.owner_id(), info.owner_id);
        assert!(matches!(src, UnlockSource::Keychain));

        // Cleanup: remove keychain entry + temp dir.
        crate::crypto::keychain_delete(KEYCHAIN_SERVICE, &account).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}
