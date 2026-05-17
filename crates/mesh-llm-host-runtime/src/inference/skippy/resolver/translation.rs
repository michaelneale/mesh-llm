use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{bail, Result};
use openai_frontend::OpenAiHookPolicy;
use skippy_protocol::{LoadMode, StageConfig, StageKvCacheConfig, StageKvCachePayload};
use skippy_server::{
    telemetry::Telemetry, EmbeddedOpenAiArgs, EmbeddedOpenAiRequestDefaults, EmbeddedRuntimeOptions,
};

use super::super::{
    family_policy_for_model_path, single_stage_config, synthetic_direct_gguf_package,
    SkippyDeviceDescriptor, SkippyModelLoadOptions, SkippyPackageIdentity, SkippyTelemetryOptions,
};
use super::request_defaults::{
    resolve_reasoning_budget, resolve_reasoning_enabled, resolve_reasoning_format,
    resolve_request_logit_bias, resolve_request_repeat_last_n, resolve_request_seed,
    resolve_request_top_k,
};
use super::support::resolve_prefill_chunk_policy;
use super::types::{
    ResolvedEmbeddedOpenAiArgs, ResolvedSkippyConfig, ResolvedStageKvCache,
    BUILTIN_PREFILL_ADAPTIVE_MAX, BUILTIN_PREFILL_ADAPTIVE_START, BUILTIN_PREFILL_ADAPTIVE_STEP,
    BUILTIN_PREFILL_CHUNK_SIZE,
};

impl ResolvedSkippyConfig {
    pub(crate) fn to_model_load_options(
        &self,
        telemetry: SkippyTelemetryOptions,
    ) -> Result<SkippyModelLoadOptions> {
        self.build_model_load_options(telemetry, false)
    }

    fn build_model_load_options(
        &self,
        telemetry: SkippyTelemetryOptions,
        allow_staged_range: bool,
    ) -> Result<SkippyModelLoadOptions> {
        if !allow_staged_range {
            self.ensure_single_stage_safe()?;
        } else {
            self.ensure_embedded_openai_safe(true)?;
        }
        let mut options = SkippyModelLoadOptions::for_direct_gguf(
            self.model_id.clone(),
            self.hardware.resolved_model_path.clone(),
        )
        .with_ctx_size(self.model_fit.ctx_size)
        .with_generation_concurrency(self.throughput.parallel)
        .with_cache_types(&self.model_fit.cache_type_k, &self.model_fit.cache_type_v)
        .with_batch_sizes(Some(self.model_fit.batch), Some(self.model_fit.ubatch))
        .with_thread_counts(self.throughput.threads, self.throughput.threads_batch)
        .with_flash_attn_type(self.model_fit.flash_attention)
        .with_telemetry(telemetry);

        options.default_max_tokens = self.request_defaults.max_tokens;
        options.n_gpu_layers = self.hardware.gpu_layers;
        if let Some(projector_path) = self.hardware.projector_path.clone() {
            options = options.with_projector_path(projector_path);
        }
        if let (Some(layer_start), Some(layer_end)) = (
            self.hardware.stage_layer_start,
            self.hardware.stage_layer_end,
        ) {
            options = options.with_layer_range(layer_start, layer_end);
        }
        if let Some(device) = self.hardware.device.clone() {
            options = options.with_selected_device(SkippyDeviceDescriptor {
                backend_device: device,
                stable_id: None,
                index: None,
                vram_bytes: None,
            });
        }
        let stage_config = single_stage_config(&options)?;
        let family_policy =
            family_policy_for_model_path(&self.hardware.resolved_model_path, Some(&self.model_id));
        let kv_cache = self
            .resolve_stage_kv_cache(family_policy.stage_kv_cache_config_for_stage(&stage_config))?;
        Ok(options.with_kv_cache(kv_cache))
    }

    pub(crate) fn to_stage_config(
        &self,
        package_identity: Option<SkippyPackageIdentity>,
        load_mode: LoadMode,
    ) -> Result<StageConfig> {
        let mut load_options =
            self.build_model_load_options(SkippyTelemetryOptions::off(), true)?;
        if let Some(package_identity) = package_identity {
            load_options.package_identity = Some(package_identity.clone());
            if self.hardware.stage_layer_end.is_none() {
                load_options.layer_end = Some(package_identity.layer_count);
            }
            load_options.model_path = match &load_mode {
                LoadMode::LayerPackage => PathBuf::from(package_identity.package_ref),
                LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => {
                    self.hardware.resolved_model_path.clone()
                }
            };
        }
        let mut stage_config = single_stage_config(&load_options)?;
        stage_config.load_mode = load_mode;
        stage_config.filter_tensors_on_load =
            !matches!(stage_config.load_mode, LoadMode::RuntimeSlice)
                || stage_config.layer_start > 0;
        if matches!(stage_config.load_mode, LoadMode::LayerPackage)
            && load_options.package_identity.is_none()
        {
            let synthetic = synthetic_direct_gguf_package(&self.model_id, &self.model_path)?;
            stage_config.package_ref = Some(synthetic.package_ref.clone());
            stage_config.manifest_sha256 = Some(synthetic.manifest_sha256.clone());
        }
        let family_policy =
            family_policy_for_model_path(&self.hardware.resolved_model_path, Some(&self.model_id));
        stage_config.kv_cache = self
            .resolve_stage_kv_cache(family_policy.stage_kv_cache_config_for_stage(&stage_config))?;
        Ok(stage_config)
    }

    pub(crate) fn to_embedded_runtime_options(
        &self,
        telemetry: &SkippyTelemetryOptions,
        package_identity: Option<SkippyPackageIdentity>,
        load_mode: LoadMode,
    ) -> Result<EmbeddedRuntimeOptions> {
        Ok(EmbeddedRuntimeOptions {
            config: self.to_stage_config(package_identity, load_mode.clone())?,
            topology: None,
            n_threads: self.throughput.threads,
            n_threads_batch: self.throughput.threads_batch,
            metrics_otlp_grpc: telemetry.metrics_otlp_grpc.clone(),
            telemetry_queue_capacity: telemetry.queue_capacity,
            telemetry_level: telemetry.level,
        })
    }

    pub(crate) fn to_embedded_openai_args(
        &self,
        activation_width: i32,
        staged: bool,
    ) -> Result<ResolvedEmbeddedOpenAiArgs> {
        self.ensure_embedded_openai_safe(staged)?;
        let mode = self.speculative_mode_for_embedded(staged);
        Ok(ResolvedEmbeddedOpenAiArgs {
            model_id: Some(self.model_id.clone()),
            default_max_tokens: self.request_defaults.max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults {
                stop: self.request_defaults.stop.clone(),
                temperature: self.request_defaults.temperature.map(|value| value as f32),
                top_p: self.request_defaults.top_p.map(|value| value as f32),
                presence_penalty: self
                    .request_defaults
                    .presence_penalty
                    .map(|value| value as f32),
                frequency_penalty: self
                    .request_defaults
                    .frequency_penalty
                    .map(|value| value as f32),
                seed: self
                    .request_defaults
                    .seed
                    .map(resolve_request_seed)
                    .transpose()?,
                logit_bias: self
                    .request_defaults
                    .logit_bias
                    .as_ref()
                    .map(resolve_request_logit_bias)
                    .transpose()?,
                top_k: self
                    .request_defaults
                    .top_k
                    .map(resolve_request_top_k)
                    .transpose()?,
                min_p: self.request_defaults.min_p.map(|value| value as f32),
                repeat_penalty: self
                    .request_defaults
                    .repeat_penalty
                    .map(|value| value as f32),
                repeat_last_n: self
                    .request_defaults
                    .repeat_last_n
                    .map(resolve_request_repeat_last_n)
                    .transpose()?,
                reasoning_format: self
                    .request_defaults
                    .reasoning_format
                    .as_deref()
                    .and_then(resolve_reasoning_format),
                reasoning_enabled: self
                    .request_defaults
                    .reasoning_enabled
                    .as_ref()
                    .and_then(resolve_reasoning_enabled),
                reasoning_budget: self
                    .request_defaults
                    .reasoning_budget
                    .as_ref()
                    .and_then(resolve_reasoning_budget),
            },
            generation_concurrency: self.throughput.parallel,
            prefill_chunk_size: self.skippy.prefill_chunk_size,
            prefill_chunk_policy: resolve_prefill_chunk_policy(&self.skippy.prefill_chunking),
            prefill_chunk_schedule: self.skippy.prefill_chunk_schedule.clone(),
            prefill_adaptive_start: BUILTIN_PREFILL_ADAPTIVE_START,
            prefill_adaptive_step: BUILTIN_PREFILL_ADAPTIVE_STEP,
            prefill_adaptive_max: BUILTIN_PREFILL_ADAPTIVE_MAX,
            draft_model_path: if mode == "draft" {
                self.speculative.draft_model_path.clone()
            } else {
                None
            },
            speculative_window: if mode == "draft" {
                self.speculative.draft_max_tokens as usize
            } else {
                0
            },
            adaptive_speculative_window: false,
            draft_n_gpu_layers: if mode == "draft" {
                self.speculative.draft_n_gpu_layers
            } else {
                None
            },
            activation_width,
            wire_dtype: self.skippy.activation_wire_dtype.into(),
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        })
    }

    fn ensure_single_stage_safe(&self) -> Result<()> {
        if self.hardware.stage_layer_start.is_some() || self.hardware.stage_layer_end.is_some() {
            bail!("skippy hardware.stage_layer_start/stage_layer_end are staged-only controls");
        }
        self.ensure_embedded_openai_safe(false)
    }

    fn ensure_embedded_openai_safe(&self, staged: bool) -> Result<()> {
        if !staged {
            if self.skippy.activation_wire_dtype_explicit {
                bail!("skippy.activation_wire_dtype requires staged serving");
            }
            if self.skippy.prefill_controls_explicit {
                bail!("skippy prefill chunk controls require staged serving");
            }
            if self.speculative.explicit {
                bail!("speculative draft controls require staged serving");
            }
        }
        Ok(())
    }

    fn speculative_mode_for_embedded(&self, staged: bool) -> &'static str {
        if !staged {
            return "disabled";
        }
        if self.speculative.mode == "draft" && self.speculative.draft_model_path.is_some() {
            "draft"
        } else {
            "disabled"
        }
    }

    fn resolve_stage_kv_cache(
        &self,
        family_default: Option<StageKvCacheConfig>,
    ) -> Result<Option<StageKvCacheConfig>> {
        match &self.model_fit.prefix_cache {
            ResolvedStageKvCache::FamilyDefault => Ok(family_default),
            ResolvedStageKvCache::Disabled => Ok(None),
            ResolvedStageKvCache::Explicit(template) => {
                let mut cache = family_default.unwrap_or(StageKvCacheConfig {
                    mode: template.mode.clone(),
                    payload: StageKvCachePayload::Auto,
                    max_entries: 128,
                    max_bytes: 0,
                    min_tokens: 256,
                    shared_prefix_stride_tokens: 128,
                    shared_prefix_record_limit: 2,
                });
                cache.mode = template.mode.clone();
                cache.payload = template.payload;
                if let Some(value) = template.max_entries {
                    cache.max_entries = value;
                }
                if let Some(value) = template.max_bytes {
                    cache.max_bytes = value;
                }
                if let Some(value) = template.min_tokens {
                    cache.min_tokens = value;
                }
                if let Some(value) = template.shared_prefix_stride_tokens {
                    cache.shared_prefix_stride_tokens = value;
                }
                if let Some(value) = template.shared_prefix_record_limit {
                    cache.shared_prefix_record_limit = value as u64;
                }
                Ok(Some(cache))
            }
        }
    }
}

impl ResolvedEmbeddedOpenAiArgs {
    pub(crate) fn direct_single_stage_defaults(
        model_id: String,
        default_max_tokens: u32,
        generation_concurrency: usize,
        wire_dtype: skippy_protocol::binary::WireActivationDType,
    ) -> Self {
        Self {
            model_id: Some(model_id),
            default_max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults::default(),
            generation_concurrency,
            prefill_chunk_size: BUILTIN_PREFILL_CHUNK_SIZE,
            prefill_chunk_policy: "fixed".to_string(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: BUILTIN_PREFILL_ADAPTIVE_START,
            prefill_adaptive_step: BUILTIN_PREFILL_ADAPTIVE_STEP,
            prefill_adaptive_max: BUILTIN_PREFILL_ADAPTIVE_MAX,
            draft_model_path: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            draft_n_gpu_layers: None,
            activation_width: 0,
            wire_dtype,
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        }
    }

    pub(crate) fn embedded_stage_defaults(
        model_id: Option<String>,
        default_max_tokens: u32,
        generation_concurrency: usize,
        activation_width: i32,
        wire_dtype: skippy_protocol::binary::WireActivationDType,
    ) -> Self {
        Self {
            model_id,
            default_max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults::default(),
            generation_concurrency,
            prefill_chunk_size: BUILTIN_PREFILL_CHUNK_SIZE,
            prefill_chunk_policy: "fixed".to_string(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: BUILTIN_PREFILL_ADAPTIVE_START,
            prefill_adaptive_step: BUILTIN_PREFILL_ADAPTIVE_STEP,
            prefill_adaptive_max: BUILTIN_PREFILL_ADAPTIVE_MAX,
            draft_model_path: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            draft_n_gpu_layers: None,
            activation_width,
            wire_dtype,
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        }
    }

    pub(crate) fn build(
        self,
        bind_addr: SocketAddr,
        config: StageConfig,
        runtime: Arc<Mutex<skippy_server::runtime_state::RuntimeState>>,
        telemetry: Telemetry,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    ) -> EmbeddedOpenAiArgs {
        EmbeddedOpenAiArgs {
            bind_addr,
            config,
            runtime,
            model_id: self.model_id,
            default_max_tokens: self.default_max_tokens,
            request_defaults: self.request_defaults,
            generation_concurrency: self.generation_concurrency,
            prefill_chunk_size: self.prefill_chunk_size,
            prefill_chunk_policy: self.prefill_chunk_policy,
            prefill_chunk_schedule: self.prefill_chunk_schedule,
            prefill_adaptive_start: self.prefill_adaptive_start,
            prefill_adaptive_step: self.prefill_adaptive_step,
            prefill_adaptive_max: self.prefill_adaptive_max,
            draft_model_path: self.draft_model_path,
            speculative_window: self.speculative_window,
            adaptive_speculative_window: self.adaptive_speculative_window,
            draft_n_gpu_layers: self.draft_n_gpu_layers,
            activation_width: self.activation_width,
            wire_dtype: self.wire_dtype,
            reply_credit_limit: self.reply_credit_limit,
            downstream_connect_timeout_secs: self.downstream_connect_timeout_secs,
            downstream_wire_condition: skippy_server::binary_transport::WireCondition::new(
                0.0, None,
            )
            .expect("static downstream wire condition should construct"),
            telemetry,
            hook_policy,
        }
    }
}
