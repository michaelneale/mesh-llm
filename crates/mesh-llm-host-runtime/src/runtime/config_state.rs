use anyhow::Result;
use sha2::{Digest, Sha256};
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
    last_write_config_hash: [u8; 32],
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

fn local_config_write_hash(config: &MeshConfig) -> [u8; 32] {
    let bytes = serde_json::to_vec(config)
        .or_else(|_| toml::to_string(config).map(String::into_bytes))
        .unwrap_or_default();
    let digest = Sha256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
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
            last_write_config_hash: [0xFF; 32],
        }
    }
}

impl ConfigState {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let config = load_config(Some(path))?;
        let revision = read_revision(&revision_sidecar_path(path));
        let proto = mesh_config_to_proto(&config);
        let config_hash = canonical_config_hash(&proto);
        let last_write_config_hash = if path.exists() {
            local_config_write_hash(&config)
        } else {
            [0xFF; 32]
        };
        Ok(Self {
            revision,
            config_hash,
            config,
            config_path: path.to_path_buf(),
            last_write_config_hash,
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
        let new_write_hash = local_config_write_hash(&new_config);

        if new_write_hash == self.last_write_config_hash {
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
            self.last_write_config_hash = new_write_hash;
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
        self.last_write_config_hash = new_write_hash;
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

    const FULL_SURFACE_VALID_FIXTURE: &str =
        include_str!("../../tests/fixtures/skippy_full_surface_valid.toml");

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
                parallel: None,
            },
            owner_control: Default::default(),
            telemetry: Default::default(),
            defaults: None,
            models: vec![],
            plugins: vec![],
        }
    }

    fn representative_nested_config() -> MeshConfig {
        toml::from_str(
            r#"version = 1

[gpu]
assignment = "auto"
parallel = 2

[defaults.model_fit]
ctx_size = 8192
kv_unified = "auto"

[defaults.hardware]
gpu_layers = "auto"
tensor_split = []

[defaults.throughput]
parallel = 3

[defaults.skippy]
activation_wire_dtype = "auto"

[defaults.speculative]
mode = "auto"
pairing_fault = "warn_disable"

[defaults.request_defaults]
reasoning_budget = "auto"
reasoning_format = "auto"

[defaults.multimodal]
mmproj = "defaults-projector.gguf"

[defaults.advanced.server]
alias = "defaults-alias"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.model_fit]
ctx_size = 16384
cache_type_k = "q8_0"

[models.hardware]
gpu_layers = 99
tensor_split = [0.7, 0.3]

[models.throughput]
parallel = 4

[models.skippy]
binary_stage_transport = "auto"

[models.speculative]
mode = "auto"
draft_selection_policy = "auto"

[models.request_defaults]
top_p = 0.95
reasoning_budget = "auto"

[models.multimodal]
mmproj = "model-projector.gguf"

[models.advanced.server]
alias = "model-alias"
"#,
        )
        .expect("representative nested config should parse")
    }

    fn assert_representative_nested_fields(config: &MeshConfig) {
        let json = serde_json::to_value(config).expect("config should serialize");
        assert_eq!(json["defaults"]["model_fit"]["kv_unified"], "auto");
        assert_eq!(json["defaults"]["hardware"]["gpu_layers"], "auto");
        assert_eq!(json["defaults"]["throughput"]["parallel"], 3);
        assert_eq!(json["defaults"]["skippy"]["activation_wire_dtype"], "auto");
        assert_eq!(json["defaults"]["speculative"]["mode"], "auto");
        assert_eq!(
            json["defaults"]["request_defaults"]["reasoning_budget"],
            "auto"
        );
        assert_eq!(
            json["defaults"]["multimodal"]["mmproj"],
            "defaults-projector.gguf"
        );
        assert_eq!(
            json["defaults"]["advanced"]["server"]["alias"],
            "defaults-alias"
        );

        assert_eq!(json["models"][0]["model_fit"]["ctx_size"], 16384);
        assert_eq!(json["models"][0]["hardware"]["gpu_layers"], 99);
        assert_eq!(json["models"][0]["throughput"]["parallel"], 4);
        assert_eq!(
            json["models"][0]["skippy"]["binary_stage_transport"],
            "auto"
        );
        assert_eq!(
            json["models"][0]["speculative"]["draft_selection_policy"],
            "auto"
        );
        assert_eq!(json["models"][0]["request_defaults"]["top_p"], 0.95);
        assert_eq!(
            json["models"][0]["multimodal"]["mmproj"],
            "model-projector.gguf"
        );
        assert_eq!(
            json["models"][0]["advanced"]["server"]["alias"],
            "model-alias"
        );
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
                parallel: None,
            },
            owner_control: Default::default(),
            telemetry: Default::default(),
            defaults: None,
            models: vec![crate::plugin::ModelConfigEntry {
                model: model.to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
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
                parallel: None,
            },
            owner_control: Default::default(),
            telemetry: Default::default(),
            defaults: None,
            models: vec![crate::plugin::ModelConfigEntry {
                model: "test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
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
    fn config_sync_state_apply_preserves_nested_sections_and_updates_hash() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        let first = representative_nested_config();
        let first_result = state.apply(first.clone(), 0);
        let first_hash = match first_result {
            ApplyResult::Applied {
                revision,
                hash,
                apply_mode,
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
                hash
            }
            other => panic!("expected Applied, got {other:?}"),
        };
        assert_representative_nested_fields(state.config());

        let persisted = ConfigState::load(&config_path).expect("reload persisted config");
        assert_representative_nested_fields(persisted.config());

        let mut changed = first;
        changed
            .models
            .first_mut()
            .expect("model")
            .advanced
            .get_or_insert_with(Default::default)
            .server
            .get_or_insert_with(Default::default)
            .alias = Some("model-alias-updated".to_string());

        let second_result = state.apply(changed, 1);
        match second_result {
            ApplyResult::Applied { revision, hash, .. } => {
                assert_eq!(revision, 2);
                assert_ne!(first_hash, hash, "nested field change must change hash");
            }
            other => panic!("expected Applied, got {other:?}"),
        }

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
    fn config_sync_load_nested_validation_error_is_stable() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        std::fs::write(
            &config_path,
            r#"version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
        )
        .expect("write invalid config");

        let error = match ConfigState::load(&config_path) {
            Ok(_) => panic!("load must fail"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        assert!(
            message.contains(
                "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
            ),
            "unexpected error: {message}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_load_malformed_nested_toml_still_errors() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        std::fs::write(
            &config_path,
            r#"version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults
temperature = 0.2
"#,
        )
        .expect("write malformed config");

        let result = ConfigState::load(&config_path);
        assert!(
            result.is_err(),
            "load must return Err on malformed nested TOML"
        );

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
                parallel: None,
            },
            owner_control: Default::default(),
            telemetry: Default::default(),
            defaults: None,
            models: vec![crate::plugin::ModelConfigEntry {
                model: "noop-test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
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
    fn config_sync_telemetry_only_change_is_persisted_locally() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        let base = minimal_valid_config();
        let r1 = state.apply(base.clone(), 0);
        let rev_after_first = match r1 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
                revision
            }
            other => panic!("expected Applied, got {other:?}"),
        };

        let mut telemetry_only = base;
        telemetry_only.telemetry.enabled = Some(true);
        telemetry_only.telemetry.endpoint = Some("https://otel.example.com".to_string());

        let r2 = state.apply(telemetry_only, rev_after_first);
        match r2 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(
                    apply_mode,
                    ConfigApplyMode::Staged,
                    "local-only telemetry changes must still be written to config.toml"
                );
                assert_eq!(revision, rev_after_first + 1);
            }
            other => panic!("expected Applied with Staged apply_mode, got {other:?}"),
        }

        let persisted = std::fs::read_to_string(&config_path).expect("persisted config");
        assert!(persisted.contains("https://otel.example.com"));

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

    #[test]
    fn config_sync_state_apply_persists_integrated_fixture_sections_and_hashes_changes() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");
        let config: MeshConfig =
            toml::from_str(FULL_SURFACE_VALID_FIXTURE).expect("fixture parses");

        let first = state.apply(config.clone(), 0);
        let first_hash = match first {
            ApplyResult::Applied {
                revision,
                hash,
                apply_mode,
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
                hash
            }
            other => panic!("expected Applied, got {other:?}"),
        };

        let persisted = std::fs::read_to_string(&config_path).expect("persisted config");
        assert!(persisted.contains("[models.skippy]"));
        assert!(persisted.contains("prefill_chunk_schedule = \"128,256,384\""));
        assert!(persisted.contains("reasoning_budget = 256"));

        let reloaded = ConfigState::load(&config_path).expect("reload config");
        assert_eq!(reloaded.config().models.len(), 2);
        assert_eq!(
            reloaded.config().models[0]
                .advanced
                .as_ref()
                .and_then(|advanced| advanced.server.as_ref())
                .and_then(|server| server.alias.as_deref()),
            Some("model-alias")
        );

        let mut changed = config;
        changed
            .defaults
            .as_mut()
            .and_then(|defaults| defaults.request_defaults.as_mut())
            .expect("request defaults")
            .temperature = Some(0.6);
        let second = state.apply(changed, 1);
        match second {
            ApplyResult::Applied { revision, hash, .. } => {
                assert_eq!(revision, 2);
                assert_ne!(first_hash, hash, "request-default change must update hash");
            }
            other => panic!("expected Applied, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
