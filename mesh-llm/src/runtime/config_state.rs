use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::plugin::{load_config, validate_config, MeshConfig};
use crate::protocol::convert::{canonical_config_hash, mesh_config_to_proto};

/// Mirrors the `ConfigApplyMode` proto enum; kept in the domain layer so
/// `config_state` does not depend on the generated proto crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigApplyMode {
    /// Config written to disk and revision counter advanced.
    Staged,
    /// No-op: the incoming config was identical to the current one.
    Noop,
}

#[derive(Debug)]
pub(crate) enum ApplyResult {
    Applied {
        revision: u64,
        hash: [u8; 32],
        apply_mode: ConfigApplyMode,
    },
    RevisionConflict {
        current_revision: u64,
    },
    PersistedWithRevisionTrackingError {
        revision: u64,
        hash: [u8; 32],
        error: String,
    },
    ValidationError(String),
    PersistError(String),
}

pub(crate) struct ConfigState {
    revision: u64,
    config_hash: [u8; 32],
    config: MeshConfig,
    config_path: PathBuf,
    last_write_hash: [u8; 32],
}

fn revision_sidecar_path(config_path: &Path) -> PathBuf {
    let parent = config_path.parent().unwrap_or(Path::new("."));
    if let Some(file_name) = config_path.file_name() {
        let mut sidecar_name = std::ffi::OsString::from(file_name);
        sidecar_name.push(".revision");
        parent.join(sidecar_name)
    } else {
        parent.join("config-revision")
    }
}

fn read_revision(sidecar: &Path) -> u64 {
    let rev = std::fs::read_to_string(sidecar)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    if let Some(rev) = rev {
        return rev;
    }
    let legacy = sidecar
        .parent()
        .unwrap_or(Path::new("."))
        .join("config-revision");
    std::fs::read_to_string(&legacy)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn atomic_write(target: &Path, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file_name = target
        .file_name()
        .unwrap_or(target.as_os_str())
        .to_string_lossy();
    let parent = target.parent().unwrap_or(Path::new("."));
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let tmp = parent.join(format!(".{}.{}.{}.tmp", file_name, pid, nanos));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    file.write_all(contents)?;
    file.sync_all()?;
    drop(file);
    // TODO(windows): this remove+rename sequence is not truly atomic on Windows.
    // Replace with MoveFileExW(MOVEFILE_REPLACE_EXISTING) or tempfile::persist_noclobber-like behavior.
    #[cfg(windows)]
    if target.exists() {
        std::fs::remove_file(target)?;
    }
    if let Err(e) = std::fs::rename(&tmp, target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

impl Default for ConfigState {
    fn default() -> Self {
        let config = crate::plugin::MeshConfig::default();
        let proto = mesh_config_to_proto(&config);
        let config_hash = canonical_config_hash(&proto);
        Self {
            revision: 0,
            config_hash,
            config,
            config_path: std::path::PathBuf::from("config.toml"),
            last_write_hash: [0xFF; 32],
        }
    }
}

impl ConfigState {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let config = load_config(Some(path))?;
        let revision = read_revision(&revision_sidecar_path(path));
        let proto = mesh_config_to_proto(&config);
        let config_hash = canonical_config_hash(&proto);
        let last_write_hash = if path.exists() {
            config_hash
        } else {
            [0xFF; 32]
        };
        Ok(Self {
            revision,
            config_hash,
            config,
            config_path: path.to_path_buf(),
            last_write_hash,
        })
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn config_hash(&self) -> &[u8; 32] {
        &self.config_hash
    }

    pub(crate) fn config(&self) -> &MeshConfig {
        &self.config
    }

    pub(crate) fn apply(&mut self, new_config: MeshConfig, expected_revision: u64) -> ApplyResult {
        if let Err(e) = validate_config(&new_config) {
            return ApplyResult::ValidationError(e.to_string());
        }

        if expected_revision != self.revision {
            return ApplyResult::RevisionConflict {
                current_revision: self.revision,
            };
        }

        let proto = mesh_config_to_proto(&new_config);
        let new_hash = canonical_config_hash(&proto);

        if new_hash == self.last_write_hash {
            return ApplyResult::Applied {
                revision: self.revision,
                hash: self.config_hash,
                apply_mode: ConfigApplyMode::Noop,
            };
        }

        let toml_str = match toml::to_string(&new_config) {
            Ok(s) => s,
            Err(e) => return ApplyResult::PersistError(format!("toml serialization failed: {e}")),
        };

        if let Err(e) = atomic_write(&self.config_path, toml_str.as_bytes()) {
            return ApplyResult::PersistError(format!("failed to write config: {e}"));
        }

        let new_revision = self.revision + 1;
        let sidecar = revision_sidecar_path(&self.config_path);
        if let Err(e) = atomic_write(&sidecar, new_revision.to_string().as_bytes()) {
            self.config = new_config;
            self.config_hash = new_hash;
            self.revision = new_revision;
            return ApplyResult::PersistedWithRevisionTrackingError {
                revision: self.revision,
                hash: self.config_hash,
                error: format!(
                    "failed to write revision sidecar: {e}; config persisted and in-memory revision advanced, but on-disk revision tracking may be stale"
                ),
            };
        }

        self.config = new_config;
        self.config_hash = new_hash;
        self.last_write_hash = new_hash;
        self.revision = new_revision;

        ApplyResult::Applied {
            revision: self.revision,
            hash: self.config_hash,
            apply_mode: ConfigApplyMode::Staged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{GpuAssignment, GpuConfig, MeshConfig};

    fn test_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mesh-llm-config-state-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    fn minimal_valid_config() -> MeshConfig {
        MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![],
            plugins: vec![],
        }
    }

    #[test]
    fn config_sync_state_load() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        std::fs::write(
            &config_path,
            "version = 1\n\n[gpu]\nassignment = \"auto\"\n",
        )
        .expect("write config");

        let state = ConfigState::load(&config_path).expect("load");
        assert_eq!(state.revision(), 0);
        assert_eq!(state.config().version, Some(1));
        assert_eq!(state.config().gpu.assignment, GpuAssignment::Auto);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_apply_success() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        let mut state = ConfigState::load(&config_path).expect("load");
        assert_eq!(state.revision(), 0);

        let result = state.apply(minimal_valid_config(), 0);
        match result {
            ApplyResult::Applied {
                revision,
                hash: _,
                apply_mode,
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
            }
            other => panic!("expected Applied, got {other:?}"),
        }

        assert!(config_path.exists(), "config file not written");

        let sidecar = revision_sidecar_path(&config_path);
        let sidecar_contents = std::fs::read_to_string(&sidecar).expect("read sidecar");
        assert_eq!(sidecar_contents.trim(), "1");

        assert_eq!(state.revision(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_conflict() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        let mut state = ConfigState::load(&config_path).expect("load");

        let result = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(result, ApplyResult::Applied { revision: 1, .. }),
            "first apply failed: {result:?}"
        );

        let result2 = state.apply(minimal_valid_config(), 0);
        match result2 {
            ApplyResult::RevisionConflict { current_revision } => {
                assert_eq!(current_revision, 1);
            }
            other => panic!("expected RevisionConflict, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_concurrent_applies() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();

        let r1 = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(r1, ApplyResult::Applied { revision: 1, .. }),
            "first apply must succeed: {r1:?}"
        );

        let r2 = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(
                r2,
                ApplyResult::RevisionConflict {
                    current_revision: 1
                }
            ),
            "second apply with stale revision must conflict: {r2:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_revision_monotonic() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();

        let make_config = |model: &str| MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![crate::plugin::ModelConfigEntry {
                model: model.to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
            }],
            plugins: vec![],
        };

        assert_eq!(state.revision(), 0);
        state.apply(make_config("model-a.gguf"), 0);
        assert_eq!(state.revision(), 1);
        state.apply(make_config("model-b.gguf"), 1);
        assert_eq!(state.revision(), 2);
        state.apply(make_config("model-c.gguf"), 2);
        assert_eq!(state.revision(), 3);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_hash_changes_on_different_config() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();
        let initial_hash = *state.config_hash();

        let config_with_model = MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![crate::plugin::ModelConfigEntry {
                model: "test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
            }],
            plugins: vec![],
        };
        state.apply(config_with_model, 0);
        let new_hash = *state.config_hash();
        assert_ne!(
            initial_hash, new_hash,
            "hash must change when config changes"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_load_propagates_invalid_toml_error() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, "this is [not valid toml !!!\n").expect("write bad toml");
        let result = ConfigState::load(&config_path);
        assert!(result.is_err(), "load must return Err on malformed TOML");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_noop_apply_skips_disk_write() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        let config_with_model = MeshConfig {
            version: Some(1),
            gpu: crate::plugin::GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![crate::plugin::ModelConfigEntry {
                model: "noop-test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
            }],
            plugins: vec![],
        };

        let r1 = state.apply(config_with_model.clone(), 0);
        let rev_after_first = match r1 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(
                    apply_mode,
                    ConfigApplyMode::Staged,
                    "first apply must save to disk"
                );
                revision
            }
            other => panic!("expected Applied, got {other:?}"),
        };

        let r2 = state.apply(config_with_model.clone(), rev_after_first);
        match r2 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(
                    apply_mode,
                    ConfigApplyMode::Noop,
                    "no-op apply must not save to disk"
                );
                assert_eq!(
                    revision, rev_after_first,
                    "revision must not change on no-op"
                );
            }
            other => panic!("expected Applied with Noop apply_mode, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_sidecar_path_derived_from_filename() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let sidecar = revision_sidecar_path(&config_path);
        let expected = dir.join("config.toml.revision");
        assert_eq!(
            sidecar, expected,
            "sidecar path must be config filename + .revision suffix"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_sidecar_migration_fallback() {
        let dir = test_dir();
        let legacy_path = dir.join("config-revision");
        std::fs::write(&legacy_path, "42\n").expect("write legacy revision");

        let config_path = dir.join("config.toml");
        let new_sidecar = revision_sidecar_path(&config_path);
        assert_ne!(
            new_sidecar, legacy_path,
            "new sidecar must differ from legacy"
        );

        let revision = read_revision(&new_sidecar);
        assert_eq!(
            revision, 42,
            "must fall back to legacy config-revision file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
