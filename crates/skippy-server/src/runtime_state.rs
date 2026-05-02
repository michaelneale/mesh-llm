use std::{
    collections::BTreeMap,
    env,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use skippy_protocol::{LoadMode, StageConfig, StageDevice};
use skippy_runtime::{
    parse_cache_type, ActivationFrame, GenerationSignalWindow, RuntimeConfig, RuntimeKvPage,
    RuntimeKvPageDesc, RuntimeLoadMode, SamplingConfig, StageModel, StageSession,
    StageSessionCheckpoint, TokenSignal,
};

use crate::package::materialize_layer_package;

pub struct RuntimeState {
    pub model: StageModel,
    layer_start: u32,
    layer_end: u32,
    sessions: BTreeMap<String, StageSession>,
    idle_sessions: Vec<StageSession>,
    session_token_counts: BTreeMap<String, u64>,
    session_checkpoints: BTreeMap<String, StageSessionCheckpoint>,
    warm_kv_sessions: BTreeMap<String, WarmKvSession>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeSessionStats {
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub warm_kv_sessions: usize,
    pub tracked_token_counts: usize,
    pub checkpoints: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RuntimeSessionDropStats {
    pub reset_session: bool,
    pub reset_ms: f64,
    pub stats_after: RuntimeSessionStats,
}

struct WarmKvSession {
    session: StageSession,
    token_end: u64,
}

impl RuntimeState {
    pub fn prefill(&mut self, session_id: &str, token_ids: &[i32]) -> Result<()> {
        let session = self.session(session_id)?;
        session
            .prefill_chunk_frame(token_ids, None, 0)
            .map(|_| ())?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(())
    }

    pub fn decode(&mut self, session_id: &str, token_id: i32) -> Result<i32> {
        self.decode_sampled(session_id, token_id, None)
    }

    pub fn decode_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        let session = self.session(session_id)?;
        let token = session
            .decode_step_frame_sampled(token_id, sampling, None, 0)
            .map(|(predicted_token, _)| predicted_token)?;
        self.add_session_tokens(session_id, 1);
        Ok(token)
    }

    pub fn last_token_signal(&mut self, session_id: &str) -> Result<TokenSignal> {
        self.session(session_id)?.last_token_signal()
    }

    pub fn signal_window(
        &mut self,
        session_id: &str,
        window_tokens: u32,
    ) -> Result<GenerationSignalWindow> {
        self.session(session_id)?.signal_window(window_tokens)
    }

    pub fn prefill_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<ActivationFrame> {
        let session = self.session(session_id)?;
        let frame = session.prefill_chunk_frame(token_ids, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(frame)
    }

    #[allow(dead_code)]
    pub fn decode_frame(
        &mut self,
        session_id: &str,
        token_id: i32,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        self.decode_frame_sampled(session_id, token_id, None, input)
    }

    pub fn decode_frame_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.decode_step_frame_sampled(token_id, sampling, input, 0)?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn verify_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<(Vec<i32>, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.verify_tokens_frame(token_ids, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(output)
    }

    pub fn checkpoint_session(&mut self, session_id: &str) -> Result<()> {
        let checkpoint = self.session(session_id)?.checkpoint()?;
        self.session_checkpoints
            .insert(session_id.to_string(), checkpoint);
        Ok(())
    }

    pub fn restore_session(&mut self, session_id: &str) -> Result<()> {
        let checkpoint = self
            .session_checkpoints
            .get(session_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing checkpoint for session {session_id}"))?;
        let token_count = {
            let session = self.session(session_id)?;
            session.restore_checkpoint(&checkpoint)?;
            session.token_count()
        };
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
        Ok(())
    }

    fn session(&mut self, session_id: &str) -> Result<&mut StageSession> {
        if !self.sessions.contains_key(session_id) {
            let session = self
                .idle_sessions
                .pop()
                .map(Ok)
                .unwrap_or_else(|| self.model.create_session())?;
            self.sessions.insert(session_id.to_string(), session);
        }
        Ok(self
            .sessions
            .get_mut(session_id)
            .expect("session inserted above"))
    }

    pub fn drop_session_timed(&mut self, session_id: &str) -> Result<RuntimeSessionDropStats> {
        let reset_started = Instant::now();
        let mut reset_session = false;
        if let Some(mut session) = self.sessions.remove(session_id) {
            reset_session = true;
            session.reset()?;
            self.idle_sessions.push(session);
        }
        self.session_token_counts.remove(session_id);
        self.session_checkpoints.remove(session_id);
        Ok(RuntimeSessionDropStats {
            reset_session,
            reset_ms: reset_started.elapsed().as_secs_f64() * 1000.0,
            stats_after: self.session_stats(),
        })
    }

    pub fn session_stats(&self) -> RuntimeSessionStats {
        RuntimeSessionStats {
            active_sessions: self.sessions.len(),
            idle_sessions: self.idle_sessions.len(),
            warm_kv_sessions: self.warm_kv_sessions.len(),
            tracked_token_counts: self.session_token_counts.len(),
            checkpoints: self.session_checkpoints.len(),
        }
    }

    pub fn take_warm_kv_session(
        &mut self,
        session_id: &str,
        page_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> bool {
        if self.sessions.contains_key(session_id) {
            return false;
        }
        let Some(expected_token_end) = token_start.checked_add(token_count) else {
            return false;
        };
        let Some(warm) = self.warm_kv_sessions.remove(page_id) else {
            return false;
        };
        if warm.token_end != expected_token_end {
            self.warm_kv_sessions.insert(page_id.to_string(), warm);
            return false;
        }
        self.sessions.insert(session_id.to_string(), warm.session);
        self.session_token_counts
            .insert(session_id.to_string(), warm.token_end);
        true
    }

    pub fn warm_kv_session(
        &mut self,
        page_id: String,
        manifest: &crate::kv_proto::KvPageManifest,
        bytes: &[u8],
    ) -> Result<()> {
        if self.warm_kv_sessions.contains_key(&page_id) {
            return Ok(());
        }
        let desc = kv_desc_from_manifest(manifest)?;
        let mut session = self
            .idle_sessions
            .pop()
            .map(Ok)
            .unwrap_or_else(|| self.model.create_session())?;
        session.import_kv_page(&desc, bytes)?;
        let token_end = desc
            .token_start
            .checked_add(desc.token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        if self.warm_kv_sessions.len() >= 8 {
            if let Some(evict_key) = self.warm_kv_sessions.keys().next().cloned() {
                if let Some(mut evicted) = self.warm_kv_sessions.remove(&evict_key) {
                    evicted.session.reset()?;
                    self.idle_sessions.push(evicted.session);
                }
            }
        }
        self.warm_kv_sessions
            .insert(page_id, WarmKvSession { session, token_end });
        Ok(())
    }

    #[allow(dead_code)]
    pub fn export_kv_page(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPage> {
        let token_end = token_start
            .checked_add(token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        let known_tokens = self
            .session_token_counts
            .get(session_id)
            .copied()
            .unwrap_or_default();
        if token_end > known_tokens {
            bail!(
                "cannot export KV page [{token_start}, {token_end}) from session with {known_tokens} known tokens"
            );
        }
        let layer_start = i32::try_from(self.layer_start)?;
        let layer_end = i32::try_from(self.layer_end)?;
        let session = self.session(session_id)?;
        session.export_kv_page(layer_start, layer_end, token_start, token_count)
    }

    #[allow(dead_code)]
    pub fn probe_kv_page(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPageDesc> {
        self.validate_export_range(session_id, token_start, token_count)?;
        let layer_start = i32::try_from(self.layer_start)?;
        let layer_end = i32::try_from(self.layer_end)?;
        let session = self.session(session_id)?;
        session.probe_kv_page(layer_start, layer_end, token_start, token_count)
    }

    #[allow(dead_code)]
    pub fn export_kv_page_into(
        &mut self,
        session_id: &str,
        desc: &RuntimeKvPageDesc,
        output: &mut [u8],
    ) -> Result<()> {
        self.validate_export_range(session_id, desc.token_start, desc.token_count)?;
        let expected_len = usize::try_from(desc.payload_bytes)?;
        if output.len() != expected_len {
            bail!(
                "KV page output buffer has {} bytes, descriptor requires {expected_len}",
                output.len()
            );
        }
        let layer_start = i32::try_from(self.layer_start)?;
        let layer_end = i32::try_from(self.layer_end)?;
        let session = self.session(session_id)?;
        let exported = session.export_kv_page_into(
            layer_start,
            layer_end,
            desc.token_start,
            desc.token_count,
            output,
        )?;
        if &exported != desc {
            bail!("KV page descriptor changed between probe and export");
        }
        Ok(())
    }

    pub fn import_kv_page(
        &mut self,
        session_id: &str,
        manifest: &crate::kv_proto::KvPageManifest,
        bytes: &[u8],
    ) -> Result<()> {
        let desc = kv_desc_from_manifest(manifest)?;
        let session = self.session(session_id)?;
        session.import_kv_page(&desc, bytes)?;
        let token_end = desc
            .token_start
            .checked_add(desc.token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = (*current).max(token_end))
            .or_insert(token_end);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn export_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.layer_start)?;
        let layer_end = i32::try_from(self.layer_end)?;
        let session = self.session(session_id)?;
        session.export_state(layer_start, layer_end)
    }

    pub fn import_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.layer_start)?;
        let layer_end = i32::try_from(self.layer_end)?;
        let session = self.session(session_id)?;
        session.import_state(layer_start, layer_end, bytes)
    }

    pub fn export_full_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.layer_start)?;
        let layer_end = i32::try_from(self.layer_end)?;
        let session = self.session(session_id)?;
        session.export_full_state(layer_start, layer_end)
    }

    pub fn import_full_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.layer_start)?;
        let layer_end = i32::try_from(self.layer_end)?;
        let session = self.session(session_id)?;
        session.import_full_state(layer_start, layer_end, bytes)
    }

    pub fn has_session_range(&self, session_id: &str, token_start: u64, token_count: u64) -> bool {
        let Some(token_end) = token_start.checked_add(token_count) else {
            return false;
        };
        self.session_token_counts
            .get(session_id)
            .copied()
            .is_some_and(|known_tokens| token_end <= known_tokens)
    }

    fn add_session_tokens(&mut self, session_id: &str, count: u64) {
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = current.saturating_add(count))
            .or_insert(count);
    }

    fn validate_export_range(
        &self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<()> {
        let token_end = token_start
            .checked_add(token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        let known_tokens = self
            .session_token_counts
            .get(session_id)
            .copied()
            .unwrap_or_default();
        if token_end > known_tokens {
            bail!(
                "cannot export KV page [{token_start}, {token_end}) from session with {known_tokens} known tokens"
            );
        }
        Ok(())
    }
}

pub fn kv_desc_annotations(desc: &RuntimeKvPageDesc) -> BTreeMap<String, String> {
    [
        ("runtime.kv_desc.version", desc.version.to_string()),
        ("runtime.kv_desc.layer_start", desc.layer_start.to_string()),
        ("runtime.kv_desc.layer_end", desc.layer_end.to_string()),
        ("runtime.kv_desc.token_start", desc.token_start.to_string()),
        ("runtime.kv_desc.token_count", desc.token_count.to_string()),
        ("runtime.kv_desc.layer_count", desc.layer_count.to_string()),
        ("runtime.kv_desc.k_type", desc.k_type.to_string()),
        ("runtime.kv_desc.v_type", desc.v_type.to_string()),
        ("runtime.kv_desc.k_row_bytes", desc.k_row_bytes.to_string()),
        ("runtime.kv_desc.v_row_bytes", desc.v_row_bytes.to_string()),
        (
            "runtime.kv_desc.v_element_bytes",
            desc.v_element_bytes.to_string(),
        ),
        (
            "runtime.kv_desc.payload_bytes",
            desc.payload_bytes.to_string(),
        ),
        ("runtime.kv_desc.flags", desc.flags.to_string()),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

fn kv_desc_from_manifest(manifest: &crate::kv_proto::KvPageManifest) -> Result<RuntimeKvPageDesc> {
    let annotation = |key: &str| -> Result<&str> {
        manifest
            .annotations
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| anyhow::anyhow!("KV page manifest missing annotation {key}"))
    };
    Ok(RuntimeKvPageDesc {
        version: annotation("runtime.kv_desc.version")?.parse()?,
        layer_start: annotation("runtime.kv_desc.layer_start")?.parse()?,
        layer_end: annotation("runtime.kv_desc.layer_end")?.parse()?,
        token_start: annotation("runtime.kv_desc.token_start")?.parse()?,
        token_count: annotation("runtime.kv_desc.token_count")?.parse()?,
        layer_count: annotation("runtime.kv_desc.layer_count")?.parse()?,
        k_type: annotation("runtime.kv_desc.k_type")?.parse()?,
        v_type: annotation("runtime.kv_desc.v_type")?.parse()?,
        k_row_bytes: annotation("runtime.kv_desc.k_row_bytes")?.parse()?,
        v_row_bytes: annotation("runtime.kv_desc.v_row_bytes")?.parse()?,
        v_element_bytes: annotation("runtime.kv_desc.v_element_bytes")?.parse()?,
        payload_bytes: annotation("runtime.kv_desc.payload_bytes")?.parse()?,
        flags: annotation("runtime.kv_desc.flags")?.parse()?,
    })
}

pub fn load_runtime(config: &StageConfig) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    let cache_type_k = parse_cache_type(&config.cache_type_k)
        .with_context(|| format!("parse cache_type_k for {}", config.stage_id))?;
    let cache_type_v = parse_cache_type(&config.cache_type_v)
        .with_context(|| format!("parse cache_type_v for {}", config.stage_id))?;
    let runtime_config = RuntimeConfig {
        stage_index: config.stage_index,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        ctx_size: config.ctx_size,
        n_gpu_layers: config.n_gpu_layers,
        cache_type_k,
        cache_type_v,
        load_mode: match config.load_mode {
            LoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
            LoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
            LoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
        },
        include_embeddings: config.stage_index == 0,
        include_output: config.downstream.is_none(),
        filter_tensors_on_load: config.filter_tensors_on_load,
    };

    let model = match config.load_mode {
        LoadMode::LayerPackage => {
            let materialized_path =
                materialize_layer_package(config).context("materialize layer package for stage")?;
            open_stage_model(
                &materialized_path,
                &runtime_config,
                config.selected_device.as_ref(),
            )?
        }
        _ => {
            let Some(model_path) = config.model_path.as_ref().map(std::path::Path::new) else {
                return Ok(None);
            };
            open_stage_model(model_path, &runtime_config, config.selected_device.as_ref())?
        }
    };

    Ok(Some(Arc::new(Mutex::new(RuntimeState {
        model,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        sessions: BTreeMap::new(),
        idle_sessions: Vec::new(),
        session_token_counts: BTreeMap::new(),
        session_checkpoints: BTreeMap::new(),
        warm_kv_sessions: BTreeMap::new(),
    }))))
}

fn open_stage_model(
    path: &std::path::Path,
    runtime_config: &RuntimeConfig,
    selected_device: Option<&StageDevice>,
) -> Result<StageModel> {
    let Some(scope) = DeviceEnvScope::new(selected_device) else {
        return StageModel::open(path, runtime_config);
    };

    let _guard = DEVICE_ENV_LOCK.lock().expect("device env lock poisoned");
    let saved = scope.apply();
    let result = StageModel::open(path, runtime_config);
    saved.restore();
    result
}

static DEVICE_ENV_LOCK: Mutex<()> = Mutex::new(());

struct DeviceEnvScope {
    vars: &'static [&'static str],
    visible_index: String,
}

impl DeviceEnvScope {
    fn new(selected_device: Option<&StageDevice>) -> Option<Self> {
        let backend_device = selected_device?.backend_device.as_str();
        let (prefix, vars): (&str, &'static [&'static str]) = if backend_device.starts_with("CUDA")
        {
            ("CUDA", &["CUDA_VISIBLE_DEVICES"])
        } else if backend_device.starts_with("HIP") {
            ("HIP", &["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
        } else if backend_device.starts_with("ROCm") {
            ("ROCm", &["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
        } else {
            return None;
        };
        let visible_index = backend_device.trim_start_matches(prefix);
        if visible_index.is_empty() || !visible_index.chars().all(|ch| ch.is_ascii_digit()) {
            return None;
        }
        Some(Self {
            vars,
            visible_index: visible_index.to_string(),
        })
    }

    fn apply(&self) -> SavedDeviceEnv {
        let values = self
            .vars
            .iter()
            .map(|var| (*var, env::var_os(var)))
            .collect::<Vec<_>>();
        for var in self.vars {
            env::set_var(var, &self.visible_index);
        }
        SavedDeviceEnv { values }
    }
}

struct SavedDeviceEnv {
    values: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl SavedDeviceEnv {
    fn restore(self) {
        for (var, value) in self.values {
            match value {
                Some(value) => env::set_var(var, value),
                None => env::remove_var(var),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kv_proto::KvPageManifest;
    use skippy_protocol::StageDevice;
    use skippy_runtime::RuntimeKvPageDesc;

    use super::{kv_desc_annotations, kv_desc_from_manifest, DeviceEnvScope};

    #[test]
    fn kv_desc_annotations_round_trip() {
        let desc = RuntimeKvPageDesc {
            version: 1,
            layer_start: 0,
            layer_end: 4,
            token_start: 16,
            token_count: 32,
            layer_count: 4,
            k_type: 1,
            v_type: 1,
            k_row_bytes: 256,
            v_row_bytes: 256,
            v_element_bytes: 0,
            payload_bytes: 65_536,
            flags: 0,
        };
        let manifest = KvPageManifest {
            annotations: kv_desc_annotations(&desc).into_iter().collect(),
            ..Default::default()
        };

        assert_eq!(kv_desc_from_manifest(&manifest).expect("desc"), desc);
    }

    #[test]
    fn kv_desc_annotations_are_required_for_import() {
        let manifest = KvPageManifest::default();
        let error = kv_desc_from_manifest(&manifest)
            .expect_err("missing descriptor annotations should fail")
            .to_string();
        assert!(error.contains("runtime.kv_desc.version"));
    }

    #[test]
    fn device_env_scope_maps_cuda_backend_device_to_visible_index() {
        let device = StageDevice {
            backend_device: "CUDA3".into(),
            stable_id: Some("uuid:GPU-123".into()),
            index: Some(3),
            vram_bytes: Some(24_000_000_000),
        };

        let scope = DeviceEnvScope::new(Some(&device)).expect("CUDA scope");

        assert_eq!(scope.visible_index, "3");
        assert_eq!(scope.vars, ["CUDA_VISIBLE_DEVICES"]);
    }

    #[test]
    fn device_env_scope_maps_amd_aliases_to_hip_visible_index() {
        for backend_device in ["HIP1", "ROCm1"] {
            let device = StageDevice {
                backend_device: backend_device.into(),
                stable_id: None,
                index: Some(1),
                vram_bytes: None,
            };

            let scope = DeviceEnvScope::new(Some(&device)).expect("AMD scope");

            assert_eq!(scope.visible_index, "1");
            assert_eq!(scope.vars, ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]);
        }
    }

    #[test]
    fn device_env_scope_ignores_backend_without_visibility_env() {
        let device = StageDevice {
            backend_device: "Vulkan0".into(),
            stable_id: None,
            index: Some(0),
            vram_bytes: None,
        };

        assert!(DeviceEnvScope::new(Some(&device)).is_none());
    }
}
