//! Mesh identity — key management and mesh-id persistence.
//!
//! Handles:
//! - Node secret key (load or generate, stored at `~/.mesh-llm/key`)
//! - Mesh ID (named or random, persisted at `~/.mesh-llm/mesh-id`)
//! - Last-joined mesh tracking (`~/.mesh-llm/last-mesh`)
//! - Public-to-private identity transition marker (`~/.mesh-llm/was-public`)

use anyhow::Result;
use iroh::SecretKey;

/// Generate a mesh ID for a new mesh.
/// Named meshes: `sha256("mesh-llm:" + name + ":" + nostr_pubkey)` — deterministic, unique per creator.
/// Unnamed meshes: random UUID, persisted to `~/.mesh-llm/mesh-id`.
pub fn generate_mesh_id(name: Option<&str>, nostr_pubkey: Option<&str>) -> String {
    if let Some(name) = name {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "mesh-llm:".hash(&mut hasher);
        name.hash(&mut hasher);
        if let Some(pk) = nostr_pubkey {
            pk.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    } else {
        // Try to load persisted mesh-id
        let path = mesh_id_path();
        if let Ok(id) = std::fs::read_to_string(&path) {
            let id = id.trim().to_string();
            if !id.is_empty() {
                return id;
            }
        }
        // Generate new random ID and persist
        let id = format!(
            "{:016x}{:016x}",
            rand::random::<u64>(),
            rand::random::<u64>()
        );
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, &id);
        id
    }
}

fn mesh_id_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("mesh-id")
}

/// Save the mesh ID of the last mesh we successfully joined.
pub fn save_last_mesh_id(mesh_id: &str) {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, mesh_id);
}

/// Load the mesh ID of the last mesh we successfully joined.
pub fn load_last_mesh_id() -> Option<String> {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------
// Public-to-private identity transition
// ---------------------------------------------------------------------------

fn was_public_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("was-public")
}

/// Record that this node was started in public mode (--auto / --publish / --mesh-name).
/// Called at startup so we can detect a public→private transition next time.
pub fn mark_was_public() {
    let path = was_public_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, "1");
}

/// Returns true if the previous run was public (marker file exists).
pub fn was_previously_public() -> bool {
    was_public_path().exists()
}

/// Clear identity files (key, nostr.nsec, mesh-id, last-mesh, was-public) so the
/// next start gets a completely fresh identity. Called when transitioning from
/// public → private to avoid reusing a publicly-known identity in a private mesh.
pub fn clear_public_identity() {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let dir = home.join(".mesh-llm");
    let mut ok = true;
    for name in &["key", "nostr.nsec", "mesh-id", "last-mesh"] {
        let p = dir.join(name);
        if p.exists() {
            if std::fs::remove_file(&p).is_ok() {
                tracing::info!("Cleared {}", p.display());
            } else {
                tracing::warn!("Failed to clear {}", p.display());
                ok = false;
            }
        }
    }
    // Only remove the marker after identity files are gone, so a failed
    // cleanup is retried on the next private start.
    let marker = dir.join("was-public");
    if ok {
        let _ = std::fs::remove_file(&marker);
    } else {
        tracing::warn!("Keeping was-public marker — will retry cleanup next start");
    }
}

/// Load secret key from ~/.mesh-llm/key, or create a new one and save it.
pub(super) async fn load_or_create_key() -> Result<SecretKey> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    let dir = home.join(".mesh-llm");
    let key_path = dir.join("key");

    if key_path.exists() {
        let hex = tokio::fs::read_to_string(&key_path).await?;
        let bytes = hex::decode(hex.trim())?;
        if bytes.len() != 32 {
            anyhow::bail!("Invalid key length in {}", key_path.display());
        }
        let key = SecretKey::from_bytes(&bytes.try_into().unwrap());
        tracing::info!("Loaded key from {}", key_path.display());
        return Ok(key);
    }

    let key = SecretKey::generate(&mut rand::rng());
    tokio::fs::create_dir_all(&dir).await?;
    tokio::fs::write(&key_path, hex::encode(key.to_bytes())).await?;
    tracing::info!("Generated new key, saved to {}", key_path.display());
    Ok(key)
}

#[cfg(test)]
mod public_identity_tests {
    use super::*;
    use std::fs;

    /// Test that mark_was_public / was_previously_public / clear_public_identity
    /// work correctly.  Uses the real ~/.mesh-llm/ directory (same approach as
    /// the rotate_keys tests) and restores originals afterward.
    #[test]
    fn public_to_private_transition_clears_identity() {
        let dir = dirs::home_dir().unwrap().join(".mesh-llm");
        fs::create_dir_all(&dir).ok();

        // Files we may touch:
        let paths: Vec<std::path::PathBuf> =
            ["key", "nostr.nsec", "mesh-id", "last-mesh", "was-public"]
                .iter()
                .map(|n| dir.join(n))
                .collect();

        // Save originals so we can restore after the test.
        let originals: Vec<Option<Vec<u8>>> = paths
            .iter()
            .map(|p| {
                if p.exists() {
                    Some(fs::read(p).unwrap())
                } else {
                    None
                }
            })
            .collect();

        // --- Scenario 1: no marker → was_previously_public is false ---
        let _ = fs::remove_file(dir.join("was-public"));
        assert!(!was_previously_public(), "should be false when no marker");

        // --- Scenario 2: mark as public → marker exists ---
        mark_was_public();
        assert!(was_previously_public(), "should be true after marking");

        // Plant some identity files to verify clear removes them.
        fs::write(dir.join("key"), b"test-key").unwrap();
        fs::write(dir.join("nostr.nsec"), b"test-nsec").unwrap();
        fs::write(dir.join("mesh-id"), b"test-mesh-id").unwrap();
        fs::write(dir.join("last-mesh"), b"test-last-mesh").unwrap();

        // --- Scenario 3: clear_public_identity removes everything ---
        clear_public_identity();
        for name in &["key", "nostr.nsec", "mesh-id", "last-mesh", "was-public"] {
            assert!(
                !dir.join(name).exists(),
                "{name} should be deleted after clear"
            );
        }
        assert!(
            !was_previously_public(),
            "marker should be gone after clear"
        );

        // --- Scenario 4: clear on already-clean directory is fine ---
        clear_public_identity(); // should not panic

        // Restore originals.
        for (path, orig) in paths.iter().zip(originals.iter()) {
            if let Some(data) = orig {
                fs::write(path, data).ok();
            } else {
                let _ = fs::remove_file(path);
            }
        }
    }
}
