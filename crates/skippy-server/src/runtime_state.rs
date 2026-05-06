use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};
use skippy_runtime::{
    parse_cache_type, ActivationFrame, FlashAttentionType as RuntimeFlashAttentionType,
    GenerationSignalWindow, MediaInput, MediaPrefill, MediaPrefillFrame, RuntimeConfig,
    RuntimeLoadMode, SamplingConfig, StageModel, StageSession, StageSessionCheckpoint, TokenSignal,
};

use crate::package::select_package_parts;

pub struct RuntimeState {
    pub model: StageModel,
    lane_count: u32,
    next_lane_index: usize,
    sessions: BTreeMap<String, RuntimeLaneSession>,
    idle_sessions: Vec<RuntimeLaneSession>,
    session_token_counts: BTreeMap<String, u64>,
    session_checkpoints: BTreeMap<String, StageSessionCheckpoint>,
}

struct RuntimeLaneSession {
    index: usize,
    session: StageSession,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionLaneStats {
    pub index: usize,
    pub active: bool,
    pub session_id: Option<String>,
    pub token_count: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionStats {
    pub lane_count: usize,
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub tracked_token_counts: usize,
    pub max_session_tokens: u64,
    pub total_session_tokens: u64,
    pub checkpoints: usize,
    pub lanes: Vec<RuntimeSessionLaneStats>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeSessionDropStats {
    pub reset_session: bool,
    pub reset_ms: f64,
    pub stats_after: RuntimeSessionStats,
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

    pub fn media_marker(&self) -> String {
        self.model.media_marker()
    }

    pub fn has_media_projector(&self) -> bool {
        self.model.has_media_projector()
    }

    pub fn prefill_media(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
        sampling: Option<&SamplingConfig>,
    ) -> Result<MediaPrefill> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media(session, prompt, media, sampling) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
    }

    pub fn prefill_media_frame(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaPrefillFrame> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media_frame(session, prompt, media) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
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

    pub fn configure_chat_sampling(
        &mut self,
        session_id: &str,
        metadata_json: &str,
        prompt_token_count: u64,
        sampling: Option<&SamplingConfig>,
    ) -> Result<()> {
        self.session(session_id)?.configure_chat_sampling(
            metadata_json,
            prompt_token_count,
            sampling,
        )
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

    pub fn prefill_final_frame_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let frame = session.prefill_chunk_frame(token_ids, input, 0)?;
        let predicted = session.sample_current(sampling)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok((predicted, frame))
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
            let lane_session = self.idle_sessions.pop().map(Ok).unwrap_or_else(|| {
                if self.sessions.len() >= self.lane_count as usize {
                    bail!("all execution lanes are busy");
                }
                self.create_lane_session()
            })?;
            self.sessions.insert(session_id.to_string(), lane_session);
        }
        Ok(&mut self
            .sessions
            .get_mut(session_id)
            .expect("session inserted above")
            .session)
    }

    pub fn prewarm_idle_sessions(
        &mut self,
        target_idle_sessions: usize,
    ) -> Result<RuntimeSessionStats> {
        while self.idle_sessions.len() < target_idle_sessions {
            if self.sessions.len() + self.idle_sessions.len() >= self.lane_count as usize {
                break;
            }
            let lane_session = self.create_lane_session()?;
            self.idle_sessions.push(lane_session);
        }
        Ok(self.session_stats())
    }

    pub fn drop_session_timed(&mut self, session_id: &str) -> Result<RuntimeSessionDropStats> {
        let reset_started = Instant::now();
        let mut reset_session = false;
        if let Some(mut lane_session) = self.sessions.remove(session_id) {
            reset_session = true;
            lane_session.session.reset()?;
            self.idle_sessions.push(lane_session);
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
        let mut max_session_tokens = 0u64;
        let mut total_session_tokens = 0u64;
        let mut lanes = (0..self.lane_count as usize)
            .map(|index| RuntimeSessionLaneStats {
                index,
                active: false,
                session_id: None,
                token_count: None,
            })
            .collect::<Vec<_>>();

        for (session_id, lane_session) in &self.sessions {
            if let Some(token_count) = self.session_token_counts.get(session_id).copied() {
                max_session_tokens = max_session_tokens.max(token_count);
                total_session_tokens = total_session_tokens.saturating_add(token_count);
            }
            if let Some(lane) = lanes.get_mut(lane_session.index) {
                lane.active = true;
                lane.session_id = Some(session_id.clone());
                lane.token_count = self.session_token_counts.get(session_id).copied();
            }
        }

        RuntimeSessionStats {
            lane_count: self.lane_count as usize,
            active_sessions: self.sessions.len(),
            idle_sessions: self.idle_sessions.len(),
            tracked_token_counts: self.session_token_counts.len(),
            max_session_tokens,
            total_session_tokens,
            checkpoints: self.session_checkpoints.len(),
            lanes,
        }
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

    fn create_lane_session(&mut self) -> Result<RuntimeLaneSession> {
        let (index, session) =
            create_indexed_lane_resource(&mut self.next_lane_index, self.lane_count, || {
                self.model.create_session()
            })?;
        Ok(RuntimeLaneSession { index, session })
    }
}

fn create_indexed_lane_resource<T>(
    next_lane_index: &mut usize,
    lane_count: u32,
    create: impl FnOnce() -> Result<T>,
) -> Result<(usize, T)> {
    if *next_lane_index >= lane_count as usize {
        bail!("all execution lanes are busy");
    }
    let index = *next_lane_index;
    let resource = create()?;
    *next_lane_index = index + 1;
    Ok((index, resource))
}

impl Drop for RuntimeState {
    fn drop(&mut self) {
        self.sessions.clear();
        self.idle_sessions.clear();
    }
}

pub fn load_runtime(config: &StageConfig) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    let runtime_config = runtime_config_from_stage_config(config)?;

    let model = match config.load_mode {
        LoadMode::LayerPackage => {
            let selected =
                select_package_parts(config).context("select layer package parts for stage")?;
            open_stage_model_from_parts(&selected.absolute_paths, &runtime_config)?
        }
        _ => {
            let Some(model_path) = config.model_path.as_ref().map(std::path::Path::new) else {
                return Ok(None);
            };
            open_stage_model(model_path, &runtime_config)?
        }
    };

    Ok(Some(Arc::new(Mutex::new(RuntimeState {
        model,
        lane_count: config.lane_count,
        next_lane_index: 0,
        sessions: BTreeMap::new(),
        idle_sessions: Vec::new(),
        session_token_counts: BTreeMap::new(),
        session_checkpoints: BTreeMap::new(),
    }))))
}

fn runtime_config_from_stage_config(config: &StageConfig) -> Result<RuntimeConfig> {
    let cache_type_k = parse_cache_type(&config.cache_type_k)
        .with_context(|| format!("parse cache_type_k for {}", config.stage_id))?;
    let cache_type_v = parse_cache_type(&config.cache_type_v)
        .with_context(|| format!("parse cache_type_v for {}", config.stage_id))?;
    Ok(RuntimeConfig {
        stage_index: config.stage_index,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        ctx_size: config.ctx_size,
        lane_count: config.lane_count,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_gpu_layers: config.n_gpu_layers,
        selected_backend_device: config
            .selected_device
            .as_ref()
            .map(|device| device.backend_device.clone()),
        cache_type_k,
        cache_type_v,
        flash_attn_type: match config.flash_attn_type {
            FlashAttentionType::Auto => RuntimeFlashAttentionType::Auto,
            FlashAttentionType::Disabled => RuntimeFlashAttentionType::Disabled,
            FlashAttentionType::Enabled => RuntimeFlashAttentionType::Enabled,
        },
        load_mode: match config.load_mode {
            LoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
            LoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
            LoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
        },
        projector_path: config.projector_path.clone(),
        include_embeddings: config.stage_index == 0 || config.downstream.is_none(),
        include_output: config.downstream.is_none(),
        filter_tensors_on_load: config.filter_tensors_on_load,
    })
}

fn open_stage_model(path: &std::path::Path, runtime_config: &RuntimeConfig) -> Result<StageModel> {
    StageModel::open(path, runtime_config)
}

fn open_stage_model_from_parts(
    paths: &[std::path::PathBuf],
    runtime_config: &RuntimeConfig,
) -> Result<StageModel> {
    StageModel::open_from_parts(paths, runtime_config)
}

#[cfg(test)]
mod tests {
    use anyhow::{bail, Result};
    use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig, StageDevice};
    use skippy_runtime::FlashAttentionType as RuntimeFlashAttentionType;

    use super::{create_indexed_lane_resource, runtime_config_from_stage_config};

    #[test]
    fn create_indexed_lane_resource_keeps_index_available_when_creation_fails() {
        let mut next_lane_index = 0;

        let error = create_indexed_lane_resource(&mut next_lane_index, 2, || -> Result<()> {
            bail!("transient session creation failure")
        })
        .expect_err("failed creation should propagate the original error");

        assert_eq!(error.to_string(), "transient session creation failure");
        assert_eq!(next_lane_index, 0);

        let (index, resource) =
            create_indexed_lane_resource(&mut next_lane_index, 2, || Ok("lane"))
                .expect("successful retry should reuse the unconsumed lane index");

        assert_eq!(index, 0);
        assert_eq!(resource, "lane");
        assert_eq!(next_lane_index, 1);
    }

    #[test]
    fn runtime_config_preserves_selected_backend_device() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: Some("/tmp/mmproj.gguf".to_string()),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 2,
            n_batch: Some(1024),
            n_ubatch: Some(256),
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Enabled,
            filter_tensors_on_load: true,
            selected_device: Some(StageDevice {
                backend_device: "Vulkan1".into(),
                stable_id: Some("pci:0000:65:00.0".into()),
                index: Some(1),
                vram_bytes: Some(16_000_000_000),
            }),
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        };

        let runtime_config = runtime_config_from_stage_config(&config).unwrap();

        assert_eq!(
            runtime_config.selected_backend_device.as_deref(),
            Some("Vulkan1")
        );
        assert_eq!(runtime_config.lane_count, 2);
        assert_eq!(runtime_config.n_batch, Some(1024));
        assert_eq!(runtime_config.n_ubatch, Some(256));
        assert_eq!(
            runtime_config.flash_attn_type,
            RuntimeFlashAttentionType::Enabled
        );
    }

    #[test]
    fn runtime_config_keeps_embeddings_for_final_layer_package_stage() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: Some("/tmp/package".to_string()),
            manifest_sha256: Some("manifest".to_string()),
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/package".to_string()),
            projector_path: None,
            stage_id: "stage-2".to_string(),
            stage_index: 2,
            layer_start: 20,
            layer_end: 30,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None,
            load_mode: LoadMode::LayerPackage,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: "tcp://127.0.0.1:19001".to_string(),
            }),
            downstream: None,
        };

        let runtime_config = runtime_config_from_stage_config(&config).unwrap();

        assert!(runtime_config.include_embeddings);
        assert!(runtime_config.include_output);
    }
}
