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
