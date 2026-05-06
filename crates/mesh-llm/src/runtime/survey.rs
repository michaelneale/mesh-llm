use crate::plugin;
use crate::system::hardware;
use anyhow::{Context, Result};
use opentelemetry::metrics::{Counter, Gauge, Histogram, MeterProvider as _};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::Resource;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

const DEFAULT_SERVICE_NAME: &str = "mesh-llm";
const DEFAULT_EXPORT_INTERVAL_SECS: u64 = 15;
const DEFAULT_QUEUE_SIZE: usize = 2048;
const OTLP_ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
const OTLP_METRICS_ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";

#[derive(Clone)]
pub(super) struct SurveyTelemetry {
    inner: Option<Arc<SurveyTelemetryInner>>,
}

struct SurveyTelemetryInner {
    queue: Arc<SurveyEventQueue>,
    hardware: hardware::HardwareSurvey,
}

#[derive(Clone, Debug)]
pub(super) struct SurveyLoadedModel {
    attrs: SurveyAttributes,
    loaded_at: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SurveyLaunchKind {
    Startup,
    RuntimeLoad,
    MultiModel,
    MoeFallback,
    MoeShard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SurveyFailureReason {
    SpawnFailed,
    HealthTimeout,
    ExitedBeforeHealthy,
    BackendProxyFailed,
    CapacityRejected,
    KnownKvCacheCrash,
    MmprojMissing,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SurveyModelSpec<'a> {
    pub(super) model: &'a str,
    pub(super) model_path: Option<&'a Path>,
    pub(super) launch_kind: SurveyLaunchKind,
    pub(super) pinned_gpu: Option<&'a super::StartupPinnedGpuTarget>,
    pub(super) backend: Option<&'a str>,
    pub(super) context_length: Option<u64>,
}

#[derive(Clone, Debug)]
struct SurveySettings {
    service_name: String,
    endpoint: String,
    headers: std::collections::HashMap<String, String>,
    export_interval: Duration,
    queue_size: usize,
}

impl SurveySettings {
    fn from_config(config: &plugin::MeshConfig) -> Option<Self> {
        Self::from_config_with_env(config, |key| std::env::var(key).ok())
    }

    fn from_config_with_env<F>(config: &plugin::MeshConfig, env: F) -> Option<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        if !plugin::telemetry_plugin_enabled(config) {
            return None;
        }
        if config.telemetry.enabled == Some(false) {
            return None;
        }
        let endpoint = resolve_metrics_endpoint(&config.telemetry, env)?;
        let service_name = config
            .telemetry
            .service_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_SERVICE_NAME)
            .to_string();
        let headers = config
            .telemetry
            .headers
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let export_interval = Duration::from_secs(
            config
                .telemetry
                .export_interval_secs
                .unwrap_or(DEFAULT_EXPORT_INTERVAL_SECS),
        );
        let queue_size = config.telemetry.queue_size.unwrap_or(DEFAULT_QUEUE_SIZE);
        Some(Self {
            service_name,
            endpoint,
            headers,
            export_interval,
            queue_size,
        })
    }
}

impl SurveyTelemetry {
    pub(super) fn disabled() -> Self {
        Self { inner: None }
    }

    pub(super) fn start(config: &plugin::MeshConfig, hardware: hardware::HardwareSurvey) -> Self {
        let Some(settings) = SurveySettings::from_config(config) else {
            return Self::disabled();
        };
        let queue = Arc::new(SurveyEventQueue::new(settings.queue_size));
        let recorder = match SurveyRecorder::otlp(&settings) {
            Ok(recorder) => recorder,
            Err(err) => {
                tracing::warn!("disabling telemetry OTLP metrics exporter: {err:#}");
                return Self::disabled();
            }
        };
        spawn_survey_worker(queue.clone(), recorder);
        Self {
            inner: Some(Arc::new(SurveyTelemetryInner { queue, hardware })),
        }
    }

    pub(super) fn model(&self, spec: SurveyModelSpec<'_>) -> SurveyLoadedModel {
        let attrs = if let Some(inner) = self.inner.as_ref() {
            SurveyAttributes::from_spec(spec, &inner.hardware)
        } else {
            SurveyAttributes::from_disabled_spec(spec)
        };
        SurveyLoadedModel {
            attrs,
            loaded_at: Instant::now(),
        }
    }

    pub(super) fn record_launch_success(&self, model: &SurveyLoadedModel, duration: Duration) {
        self.emit(SurveyEvent::LaunchSuccess {
            attrs: model.attrs.clone(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        });
    }

    pub(super) fn record_launch_failure(
        &self,
        spec: SurveyModelSpec<'_>,
        duration: Duration,
        reason: SurveyFailureReason,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.emit(SurveyEvent::LaunchFailure {
            attrs: SurveyAttributes::from_spec(spec, &inner.hardware),
            duration_ms: duration.as_secs_f64() * 1000.0,
            reason,
        });
    }

    pub(super) fn record_unload(&self, model: &SurveyLoadedModel) {
        self.emit(SurveyEvent::Unload {
            attrs: model.attrs.clone(),
            uptime_s: model.loaded_at.elapsed().as_secs_f64(),
        });
    }

    pub(super) fn record_unexpected_exit(&self, model: &SurveyLoadedModel) {
        self.emit(SurveyEvent::UnexpectedExit {
            attrs: model.attrs.clone(),
            uptime_s: model.loaded_at.elapsed().as_secs_f64(),
        });
    }

    fn emit(&self, event: SurveyEvent) {
        if let Some(inner) = self.inner.as_ref() {
            inner.queue.push(event);
        }
    }
}

pub(super) fn classify_launch_failure(err: &anyhow::Error) -> SurveyFailureReason {
    let message = format!("{err:#}").to_ascii_lowercase();
    if message.contains("capacity")
        || message.contains("fit locally")
        || message.contains("requires")
    {
        SurveyFailureReason::CapacityRejected
    } else if message.contains("mmproj") {
        SurveyFailureReason::MmprojMissing
    } else if message.contains("health") || message.contains("timeout") {
        SurveyFailureReason::HealthTimeout
    } else if message.contains("kv cache") {
        SurveyFailureReason::KnownKvCacheCrash
    } else if message.contains("proxy") {
        SurveyFailureReason::BackendProxyFailed
    } else if message.contains("exit") || message.contains("exited") {
        SurveyFailureReason::ExitedBeforeHealthy
    } else if message.contains("spawn") || message.contains("start") || message.contains("launch") {
        SurveyFailureReason::SpawnFailed
    } else {
        SurveyFailureReason::Other
    }
}

fn spawn_survey_worker(queue: Arc<SurveyEventQueue>, mut recorder: SurveyRecorder) {
    tokio::spawn(async move {
        loop {
            let events = queue.drain();
            if events.is_empty() {
                queue.notified().await;
                continue;
            }
            for event in events {
                recorder.record(event);
            }
        }
    });
}

fn resolve_metrics_endpoint<F>(config: &plugin::TelemetryConfig, env: F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    trimmed_nonempty(config.metrics.endpoint.as_deref())
        .map(ToOwned::to_owned)
        .or_else(|| trimmed_nonempty(config.endpoint.as_deref()).map(metrics_endpoint_from_base))
        .or_else(|| {
            trimmed_nonempty(env(OTLP_METRICS_ENDPOINT_ENV).as_deref()).map(ToOwned::to_owned)
        })
        .or_else(|| {
            trimmed_nonempty(env(OTLP_ENDPOINT_ENV).as_deref()).map(metrics_endpoint_from_base)
        })
}

fn metrics_endpoint_from_base(endpoint: &str) -> String {
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.ends_with("/v1/metrics") {
        endpoint.to_string()
    } else {
        format!("{endpoint}/v1/metrics")
    }
}

fn trimmed_nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[derive(Clone, Debug)]
struct SurveyAttributes {
    model: String,
    architecture: Option<String>,
    quantization: Option<String>,
    launch_kind: SurveyLaunchKind,
    gpu_name: Option<String>,
    gpu_stable_id: Option<String>,
    backend_device: Option<String>,
    gpu_count: u64,
    is_soc: bool,
    backend: Option<String>,
    context_length: Option<u64>,
}

impl SurveyAttributes {
    fn from_disabled_spec(spec: SurveyModelSpec<'_>) -> Self {
        Self {
            model: model_metric_value(spec.model),
            architecture: None,
            quantization: None,
            launch_kind: spec.launch_kind,
            gpu_name: None,
            gpu_stable_id: None,
            backend_device: None,
            gpu_count: 0,
            is_soc: false,
            backend: spec
                .backend
                .and_then(|value| trimmed_nonempty(Some(value)))
                .map(ToOwned::to_owned),
            context_length: spec.context_length,
        }
    }

    fn from_spec(spec: SurveyModelSpec<'_>, hardware: &hardware::HardwareSurvey) -> Self {
        let gpu = spec
            .pinned_gpu
            .and_then(|pinned| hardware.gpus.iter().find(|gpu| gpu.index == pinned.index))
            .or_else(|| hardware.gpus.first());
        let gpu_name = gpu
            .map(|gpu| gpu.display_name.as_str())
            .or(hardware.gpu_name.as_deref())
            .and_then(|value| trimmed_nonempty(Some(value)))
            .map(ToOwned::to_owned);
        let stable_id = spec
            .pinned_gpu
            .map(|gpu| gpu.stable_id.as_str())
            .or_else(|| gpu.and_then(|gpu| gpu.stable_id.as_deref()));
        let backend_device = spec
            .pinned_gpu
            .map(|gpu| gpu.backend_device.as_str())
            .or_else(|| gpu.and_then(|gpu| gpu.backend_device.as_deref()))
            .and_then(|value| trimmed_nonempty(Some(value)))
            .map(ToOwned::to_owned);
        let architecture = spec
            .model_path
            .and_then(crate::models::gguf::scan_gguf_compact_meta)
            .and_then(|meta| {
                trimmed_nonempty(Some(meta.architecture.as_str())).map(ToOwned::to_owned)
            });
        let quantization = spec
            .model_path
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .map(crate::models::inventory::derive_quantization_type)
            .and_then(|value| trimmed_nonempty(Some(value.as_str())).map(ToOwned::to_owned))
            .or_else(|| super::dashboard_quantization_from_model_name(spec.model));
        Self {
            model: model_metric_value(spec.model),
            architecture,
            quantization,
            launch_kind: spec.launch_kind,
            gpu_name,
            gpu_stable_id: stable_id.and_then(redact_stable_id),
            backend_device,
            gpu_count: u64::from(hardware.gpu_count).max(hardware.gpus.len() as u64),
            is_soc: hardware.is_soc,
            backend: spec
                .backend
                .and_then(|value| trimmed_nonempty(Some(value)))
                .map(ToOwned::to_owned),
            context_length: spec.context_length,
        }
    }

    fn key_values(&self, failure_reason: Option<SurveyFailureReason>) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("mesh_llm.model", self.model.clone()),
            KeyValue::new("mesh_llm.launch_kind", self.launch_kind.as_str()),
            KeyValue::new("mesh_llm.gpu_count", self.gpu_count as i64),
            KeyValue::new("mesh_llm.is_soc", self.is_soc),
            KeyValue::new("mesh_llm.service_version", crate::VERSION),
        ];
        if let Some(value) = &self.architecture {
            attrs.push(KeyValue::new("mesh_llm.architecture", value.clone()));
        }
        if let Some(value) = &self.quantization {
            attrs.push(KeyValue::new("mesh_llm.quantization", value.clone()));
        }
        if let Some(value) = &self.gpu_name {
            attrs.push(KeyValue::new("mesh_llm.gpu_name", value.clone()));
        }
        if let Some(value) = &self.gpu_stable_id {
            attrs.push(KeyValue::new("mesh_llm.gpu_stable_id", value.clone()));
        }
        if let Some(value) = &self.backend_device {
            attrs.push(KeyValue::new("mesh_llm.backend_device", value.clone()));
        }
        if let Some(value) = &self.backend {
            attrs.push(KeyValue::new("mesh_llm.backend", value.clone()));
        }
        if let Some(context_length) = self.context_length {
            attrs.push(KeyValue::new(
                "mesh_llm.context_bucket",
                context_bucket(context_length),
            ));
        }
        if let Some(reason) = failure_reason {
            attrs.push(KeyValue::new("mesh_llm.failure_reason", reason.as_str()));
        }
        attrs
    }
}

fn model_metric_value(model: &str) -> String {
    let path = Path::new(model);
    if path.is_absolute() || (path.components().count() > 1 && path.extension().is_some()) {
        return path
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or(model)
            .to_string();
    }
    model.to_string()
}

fn redact_stable_id(stable_id: &str) -> Option<String> {
    let stable_id = stable_id.trim();
    if stable_id.is_empty() {
        return None;
    }
    let digest = Sha256::digest(stable_id.as_bytes());
    Some(format!("sha256:{}", hex::encode(&digest[..8])))
}

fn context_bucket(context_length: u64) -> &'static str {
    match context_length {
        0..=8192 => "<=8k",
        8193..=16_384 => "8k_16k",
        16_385..=32_768 => "16k_32k",
        32_769..=65_536 => "32k_64k",
        65_537..=131_072 => "64k_128k",
        _ => ">128k",
    }
}

impl SurveyLaunchKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::RuntimeLoad => "runtime_load",
            Self::MultiModel => "multi_model",
            Self::MoeFallback => "moe_fallback",
            Self::MoeShard => "moe_shard",
        }
    }
}

impl SurveyFailureReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::SpawnFailed => "spawn_failed",
            Self::HealthTimeout => "health_timeout",
            Self::ExitedBeforeHealthy => "exited_before_healthy",
            Self::BackendProxyFailed => "backend_proxy_failed",
            Self::CapacityRejected => "capacity_rejected",
            Self::KnownKvCacheCrash => "known_kv_cache_crash",
            Self::MmprojMissing => "mmproj_missing",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug)]
enum SurveyEvent {
    LaunchSuccess {
        attrs: SurveyAttributes,
        duration_ms: f64,
    },
    LaunchFailure {
        attrs: SurveyAttributes,
        duration_ms: f64,
        reason: SurveyFailureReason,
    },
    Unload {
        attrs: SurveyAttributes,
        uptime_s: f64,
    },
    UnexpectedExit {
        attrs: SurveyAttributes,
        uptime_s: f64,
    },
}

#[derive(Debug)]
struct SurveyEventQueue {
    capacity: usize,
    events: Mutex<VecDeque<SurveyEvent>>,
    notify: Notify,
}

impl SurveyEventQueue {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            events: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            notify: Notify::new(),
        }
    }

    fn push(&self, event: SurveyEvent) {
        let mut events = self
            .events
            .lock()
            .expect("telemetry event queue lock poisoned");
        if events.len() == self.capacity {
            events.pop_front();
        }
        events.push_back(event);
        drop(events);
        self.notify.notify_one();
    }

    fn drain(&self) -> Vec<SurveyEvent> {
        let mut events = self
            .events
            .lock()
            .expect("telemetry event queue lock poisoned");
        events.drain(..).collect()
    }

    async fn notified(&self) {
        self.notify.notified().await;
    }
}

struct SurveyRecorder {
    _provider: SdkMeterProvider,
    launch_total: Counter<u64>,
    launch_success_total: Counter<u64>,
    launch_failure_total: Counter<u64>,
    unload_total: Counter<u64>,
    unexpected_exit_total: Counter<u64>,
    loaded_models: Gauge<u64>,
    model_loaded: Gauge<u64>,
    model_context_length: Gauge<u64>,
    launch_duration_ms: Histogram<f64>,
    uptime_s: Histogram<f64>,
    loaded_count: u64,
}

impl SurveyRecorder {
    fn otlp(settings: &SurveySettings) -> Result<Self> {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(settings.endpoint.clone())
            .with_timeout(Duration::from_secs(10))
            .with_headers(settings.headers.clone())
            .build()
            .context("build OTLP metrics exporter")?;
        let reader = PeriodicReader::builder(exporter)
            .with_interval(settings.export_interval)
            .build();
        let provider = SdkMeterProvider::builder()
            .with_resource(
                Resource::builder()
                    .with_service_name(settings.service_name.clone())
                    .with_attribute(KeyValue::new("service.version", crate::VERSION))
                    .build(),
            )
            .with_reader(reader)
            .build();
        Ok(Self::new(provider))
    }

    fn new(provider: SdkMeterProvider) -> Self {
        let meter = provider.meter("mesh-llm.telemetry");
        Self {
            _provider: provider,
            launch_total: meter
                .u64_counter("mesh_llm_model_launch_total")
                .with_description("Total local model launch attempts.")
                .build(),
            launch_success_total: meter
                .u64_counter("mesh_llm_model_launch_success_total")
                .with_description("Successful local model launches.")
                .build(),
            launch_failure_total: meter
                .u64_counter("mesh_llm_model_launch_failure_total")
                .with_description("Failed local model launches.")
                .build(),
            unload_total: meter
                .u64_counter("mesh_llm_model_unload_total")
                .with_description("Intentional local model unloads.")
                .build(),
            unexpected_exit_total: meter
                .u64_counter("mesh_llm_model_exit_unexpected_total")
                .with_description("Unexpected local model exits.")
                .build(),
            loaded_models: meter
                .u64_gauge("mesh_llm_loaded_models")
                .with_description("Current number of locally loaded models.")
                .build(),
            model_loaded: meter
                .u64_gauge("mesh_llm_model_loaded")
                .with_description("Whether a local model is currently loaded.")
                .build(),
            model_context_length: meter
                .u64_gauge("mesh_llm_model_context_length")
                .with_description("Effective context length for a loaded local model.")
                .with_unit("{token}")
                .build(),
            launch_duration_ms: meter
                .f64_histogram("mesh_llm_model_launch_duration_ms")
                .with_description("Local model launch duration.")
                .with_unit("ms")
                .build(),
            uptime_s: meter
                .f64_histogram("mesh_llm_model_uptime_s")
                .with_description("Local model uptime before unload or unexpected exit.")
                .with_unit("s")
                .build(),
            loaded_count: 0,
        }
    }

    fn record(&mut self, event: SurveyEvent) {
        match event {
            SurveyEvent::LaunchSuccess { attrs, duration_ms } => {
                let kv = attrs.key_values(None);
                self.launch_total.add(1, &kv);
                self.launch_success_total.add(1, &kv);
                self.launch_duration_ms.record(duration_ms, &kv);
                self.loaded_count = self.loaded_count.saturating_add(1);
                self.loaded_models
                    .record(self.loaded_count, &service_version_attrs());
                self.model_loaded.record(1, &kv);
                if let Some(context_length) = attrs.context_length {
                    self.model_context_length.record(context_length, &kv);
                }
            }
            SurveyEvent::LaunchFailure {
                attrs,
                duration_ms,
                reason,
            } => {
                let kv = attrs.key_values(Some(reason));
                self.launch_total.add(1, &kv);
                self.launch_failure_total.add(1, &kv);
                self.launch_duration_ms.record(duration_ms, &kv);
            }
            SurveyEvent::Unload { attrs, uptime_s } => {
                let kv = attrs.key_values(None);
                self.unload_total.add(1, &kv);
                self.uptime_s.record(uptime_s, &kv);
                self.loaded_count = self.loaded_count.saturating_sub(1);
                self.loaded_models
                    .record(self.loaded_count, &service_version_attrs());
                self.model_loaded.record(0, &kv);
            }
            SurveyEvent::UnexpectedExit { attrs, uptime_s } => {
                let kv = attrs.key_values(None);
                self.unexpected_exit_total.add(1, &kv);
                self.uptime_s.record(uptime_s, &kv);
                self.loaded_count = self.loaded_count.saturating_sub(1);
                self.loaded_models
                    .record(self.loaded_count, &service_version_attrs());
                self.model_loaded.record(0, &kv);
            }
        }
    }
}

fn service_version_attrs() -> Vec<KeyValue> {
    vec![KeyValue::new("mesh_llm.service_version", crate::VERSION)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{MeshConfig, PluginConfigEntry, TelemetryConfig, TelemetryMetricsConfig};
    use std::collections::{BTreeMap, HashMap};

    fn survey_config() -> MeshConfig {
        MeshConfig {
            telemetry: TelemetryConfig {
                enabled: Some(true),
                service_name: Some("mesh-llm-test".into()),
                endpoint: Some("https://config.example.com".into()),
                export_interval_secs: Some(5),
                queue_size: Some(2),
                ..Default::default()
            },
            plugins: vec![PluginConfigEntry {
                name: plugin::TELEMETRY_PLUGIN_ID.into(),
                enabled: Some(true),
                command: None,
                args: Vec::new(),
                url: None,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn settings_require_survey_plugin_opt_in() {
        let mut config = survey_config();
        config.plugins.clear();
        assert!(SurveySettings::from_config_with_env(&config, |_| {
            Some("https://env.example.com".into())
        })
        .is_none());
    }

    #[test]
    fn metrics_endpoint_prefers_config_metrics_endpoint_over_base_and_env() {
        let mut config = survey_config();
        config.telemetry.endpoint = Some("https://base.example.com".into());
        config.telemetry.metrics = TelemetryMetricsConfig {
            endpoint: Some("https://metrics.example.com/custom".into()),
        };

        let settings = SurveySettings::from_config_with_env(&config, |key| match key {
            OTLP_METRICS_ENDPOINT_ENV => Some("https://env-metrics.example.com/v1/metrics".into()),
            OTLP_ENDPOINT_ENV => Some("https://env-base.example.com".into()),
            _ => None,
        })
        .expect("settings");

        assert_eq!(settings.endpoint, "https://metrics.example.com/custom");
        assert_eq!(settings.queue_size, 2);
        assert_eq!(settings.export_interval, Duration::from_secs(5));
    }

    #[test]
    fn metrics_endpoint_normalizes_base_endpoint_from_env() {
        let mut config = survey_config();
        config.telemetry.endpoint = None;
        config.telemetry.metrics.endpoint = None;

        let settings = SurveySettings::from_config_with_env(&config, |key| match key {
            OTLP_ENDPOINT_ENV => Some("https://collector.example.com/".into()),
            _ => None,
        })
        .expect("settings");

        assert_eq!(
            settings.endpoint,
            "https://collector.example.com/v1/metrics"
        );
    }

    #[test]
    fn event_queue_drops_oldest_when_full() {
        let queue = SurveyEventQueue::new(2);
        for model in ["first", "second", "third"] {
            let attrs = SurveyAttributes {
                model: model.into(),
                architecture: None,
                quantization: None,
                launch_kind: SurveyLaunchKind::Startup,
                gpu_name: None,
                gpu_stable_id: None,
                backend_device: None,
                gpu_count: 0,
                is_soc: false,
                backend: None,
                context_length: None,
            };
            queue.push(SurveyEvent::LaunchSuccess {
                attrs,
                duration_ms: 1.0,
            });
        }

        let drained = queue.drain();
        let models: Vec<_> = drained
            .iter()
            .filter_map(|event| match event {
                SurveyEvent::LaunchSuccess { attrs, .. } => Some(attrs.model.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(models, vec!["second", "third"]);
    }

    #[test]
    fn attributes_hash_gpu_stable_id_and_bucket_context() {
        let hardware = hardware::HardwareSurvey {
            gpu_count: 1,
            is_soc: true,
            gpus: vec![hardware::GpuFacts {
                index: 0,
                display_name: "NVIDIA Test".into(),
                backend_device: Some("CUDA0".into()),
                stable_id: Some("uuid:SECRET-GPU".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let attrs = SurveyAttributes::from_spec(
            SurveyModelSpec {
                model: "/private/models/Qwen3-8B-Q4_K_M.gguf",
                model_path: None,
                launch_kind: SurveyLaunchKind::RuntimeLoad,
                pinned_gpu: None,
                backend: Some("skippy"),
                context_length: Some(32_768),
            },
            &hardware,
        );
        let kv: HashMap<_, _> = attrs
            .key_values(None)
            .into_iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();

        assert_eq!(
            kv.get("mesh_llm.model").map(String::as_str),
            Some("Qwen3-8B-Q4_K_M.gguf")
        );
        assert_eq!(
            kv.get("mesh_llm.context_bucket").map(String::as_str),
            Some("16k_32k")
        );
        let stable_id = kv.get("mesh_llm.gpu_stable_id").expect("stable id");
        assert!(stable_id.starts_with("sha256:"));
        assert!(!stable_id.contains("SECRET-GPU"));
        assert_eq!(
            kv.get("mesh_llm.backend").map(String::as_str),
            Some("skippy")
        );
        assert_eq!(kv.get("mesh_llm.is_soc").map(String::as_str), Some("true"));
    }

    #[test]
    fn model_metric_keeps_huggingface_refs_but_strips_absolute_paths() {
        assert_eq!(
            model_metric_value("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
            "Qwen/Qwen3-8B-GGUF:Q4_K_M"
        );
        assert_eq!(
            model_metric_value("/private/models/Qwen3-8B-Q4_K_M.gguf"),
            "Qwen3-8B-Q4_K_M.gguf"
        );
        assert_eq!(
            model_metric_value("models/Qwen3-8B-Q4_K_M.gguf"),
            "Qwen3-8B-Q4_K_M.gguf"
        );
    }

    #[test]
    fn telemetry_headers_are_copied_from_config() {
        let mut config = survey_config();
        config.telemetry.headers = BTreeMap::from([("authorization".into(), "Bearer abc".into())]);

        let settings = SurveySettings::from_config_with_env(&config, |_| None).expect("settings");

        assert_eq!(
            settings.headers.get("authorization").map(String::as_str),
            Some("Bearer abc")
        );
    }
}
