//! Bounded in-memory routing outcome and local routing pressure metrics for
//! operator/API surfaces.

use serde::Serialize;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const METRICS_TTL: Duration = Duration::from_secs(60 * 60);
const MAX_TRACKED_MODELS: usize = 128;
const MAX_TARGETS_PER_MODEL: usize = 16;
const DEFAULT_MODEL_SHARDS: usize = 32;
const THROUGHPUT_SCALE_MILLI: u64 = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetricLayer {
    Runtime,
    Information,
    Strategy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetricScope {
    LocalOnly,
    PeerAdvertised,
    MeshDerived,
}

const METRIC_LAYER_VOCAB: [MetricLayer; 3] = [
    MetricLayer::Runtime,
    MetricLayer::Information,
    MetricLayer::Strategy,
];
const METRIC_SCOPE_VOCAB: [MetricScope; 3] = [
    MetricScope::LocalOnly,
    MetricScope::PeerAdvertised,
    MetricScope::MeshDerived,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MetricGroupMetadata {
    pub(crate) name: &'static str,
    pub(crate) layer: MetricLayer,
    pub(crate) scope: MetricScope,
    pub(crate) api_surface: &'static str,
    pub(crate) description: &'static str,
}

pub(crate) const ROUTING_METRIC_GROUPS: [MetricGroupMetadata; 5] = [
    MetricGroupMetadata {
        name: "routing_metrics",
        layer: MetricLayer::Information,
        scope: MetricScope::LocalOnly,
        api_surface: "/api/status",
        description: "Current-node routing outcome summary for operator/API inspection.",
    },
    MetricGroupMetadata {
        name: "routing_metrics.local_node",
        layer: MetricLayer::Runtime,
        scope: MetricScope::LocalOnly,
        api_surface: "/api/status",
        description: "Current-node routing pressure and lightweight utilization proxies.",
    },
    MetricGroupMetadata {
        name: "routing_metrics.pressure",
        layer: MetricLayer::Information,
        scope: MetricScope::LocalOnly,
        api_surface: "/api/status",
        description: "Current-node service mix summary for locally fronted traffic.",
    },
    MetricGroupMetadata {
        name: "mesh_models[].routing_metrics",
        layer: MetricLayer::Information,
        scope: MetricScope::LocalOnly,
        api_surface: "/api/models",
        description: "Per-model routing outcome summary observed on the current node.",
    },
    MetricGroupMetadata {
        name: "mesh_models[].routing_metrics.targets[]",
        layer: MetricLayer::Runtime,
        scope: MetricScope::LocalOnly,
        api_surface: "/api/models",
        description: "Per-target routing outcome memory observed on the current node.",
    },
];

fn metric_group(name: &str) -> &'static MetricGroupMetadata {
    ROUTING_METRIC_GROUPS
        .iter()
        .find(|group| group.name == name)
        .expect("routing metric group metadata must stay in sync with exported API groups")
}

fn metric_vocabulary_is_complete() -> bool {
    METRIC_LAYER_VOCAB.len() == 3 && METRIC_SCOPE_VOCAB.len() == 3
}

/// Local-only current-node routing outcome summary exposed on `/api/status`.
///
/// These counters are measured on the current node only and do not represent a
/// mesh-wide aggregate.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RoutingMetricsStatusSnapshot {
    pub request_count: u64,
    pub successful_requests: u64,
    pub success_rate: f64,
    pub retry_count: u64,
    pub failover_count: u64,
    pub attempt_timeout_count: u64,
    pub attempt_unavailable_count: u64,
    pub attempt_context_overflow_count: u64,
    pub attempt_reject_count: u64,
    pub avg_queue_wait_ms: f64,
    pub avg_attempt_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_tokens_per_second: Option<f64>,
    pub completion_tokens_observed: u64,
    pub throughput_samples: u64,
    /// Current-node routing pressure and lightweight utilization proxies.
    pub local_node: LocalNodePressureSnapshot,
    /// Current-node service mix for requests fronted by this node.
    pub pressure: RoutingPressureSnapshot,
}

impl Default for RoutingMetricsStatusSnapshot {
    fn default() -> Self {
        Self {
            request_count: 0,
            successful_requests: 0,
            success_rate: 0.0,
            retry_count: 0,
            failover_count: 0,
            attempt_timeout_count: 0,
            attempt_unavailable_count: 0,
            attempt_context_overflow_count: 0,
            attempt_reject_count: 0,
            avg_queue_wait_ms: 0.0,
            avg_attempt_ms: 0.0,
            avg_tokens_per_second: None,
            completion_tokens_observed: 0,
            throughput_samples: 0,
            local_node: LocalNodePressureSnapshot::default(),
            pressure: RoutingPressureSnapshot::default(),
        }
    }
}

/// Current-node routing pressure and lightweight utilization proxies.
///
/// These values are measured locally and intentionally avoid claiming to be a
/// complete node utilization model.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct LocalNodePressureSnapshot {
    pub current_inflight_requests: u64,
    pub peak_inflight_requests: u64,
    pub local_attempt_count: u64,
    pub remote_attempt_count: u64,
    pub endpoint_attempt_count: u64,
    pub avg_queue_wait_ms: f64,
    pub avg_attempt_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_tokens_per_second: Option<f64>,
    pub completion_tokens_observed: u64,
    pub throughput_samples: u64,
}

/// Current-node service mix summary for requests fronted by this node.
///
/// These shares are derived from local routing outcomes and are not mesh-wide
/// demand or serving totals.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct RoutingPressureSnapshot {
    pub fronted_request_count: u64,
    pub locally_served_request_count: u64,
    pub remotely_served_request_count: u64,
    pub endpoint_request_count: u64,
    pub local_service_share: f64,
    pub remote_service_share: f64,
    pub endpoint_service_share: f64,
}

/// Local-only per-model routing outcome summary exposed on `/api/models`.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct ModelRoutingMetricsSnapshot {
    pub request_count: u64,
    pub successful_requests: u64,
    pub success_rate: f64,
    pub retry_count: u64,
    pub failover_count: u64,
    pub attempt_timeout_count: u64,
    pub attempt_unavailable_count: u64,
    pub attempt_context_overflow_count: u64,
    pub attempt_reject_count: u64,
    pub avg_queue_wait_ms: f64,
    pub avg_attempt_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_tokens_per_second: Option<f64>,
    pub completion_tokens_observed: u64,
    pub throughput_samples: u64,
    /// Local-only per-target routing outcome memory for this model.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub targets: Vec<TargetRoutingMetricsSnapshot>,
}

/// Local-only per-target routing outcome memory exposed on `/api/models`.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct TargetRoutingMetricsSnapshot {
    pub target: String,
    pub kind: String,
    pub attempt_count: u64,
    pub success_count: u64,
    pub success_rate: f64,
    pub timeout_rate: f64,
    pub timeout_count: u64,
    pub unavailable_count: u64,
    pub context_overflow_count: u64,
    pub reject_count: u64,
    pub avg_queue_wait_ms: f64,
    pub avg_attempt_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_tokens_per_second: Option<f64>,
    pub completion_tokens_observed: u64,
    pub throughput_samples: u64,
    pub last_updated_secs_ago: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AttemptTarget {
    Local(String),
    Remote(String),
    Endpoint(String),
}

impl AttemptTarget {
    fn key(&self) -> TargetKey {
        match self {
            Self::Local(label) => TargetKey {
                kind: TargetKind::Local,
                label: label.clone(),
            },
            Self::Remote(label) => TargetKey {
                kind: TargetKind::Remote,
                label: label.clone(),
            },
            Self::Endpoint(label) => TargetKey {
                kind: TargetKind::Endpoint,
                label: label.clone(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttemptOutcome {
    Success,
    Timeout,
    Unavailable,
    ContextOverflow,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RequestService {
    Local,
    Remote,
    Endpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RequestOutcome {
    Success(RequestService),
    Rejected(RequestService),
    Unavailable,
}

#[derive(Clone)]
pub struct RoutingMetrics {
    globals: Arc<GlobalMetrics>,
    shards: Arc<Vec<Mutex<ModelShard>>>,
    config: MetricsConfig,
}

impl RoutingMetrics {
    pub fn new() -> Self {
        Self::with_metrics_config(MetricsConfig::default())
    }

    fn with_metrics_config(config: MetricsConfig) -> Self {
        let shard_count = config.shard_count.max(1);
        let mut shards = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            shards.push(Mutex::new(ModelShard::default()));
        }
        Self {
            globals: Arc::new(GlobalMetrics::default()),
            shards: Arc::new(shards),
            config,
        }
    }

    #[cfg(test)]
    fn with_config(ttl: Duration, max_models: usize, max_targets_per_model: usize) -> Self {
        Self::with_config_and_shards(ttl, max_models, max_targets_per_model, 1)
    }

    #[cfg(test)]
    fn with_config_and_shards(
        ttl: Duration,
        max_models: usize,
        max_targets_per_model: usize,
        shard_count: usize,
    ) -> Self {
        Self::with_metrics_config(MetricsConfig::new(
            ttl,
            max_models,
            max_targets_per_model,
            shard_count,
        ))
    }

    pub fn observe_inflight(&self, current: u64) {
        self.globals.observe_inflight(current);
    }

    pub fn record_attempt(
        &self,
        model: Option<&str>,
        target: AttemptTarget,
        queue_wait: Duration,
        attempt_time: Duration,
        outcome: AttemptOutcome,
        completion_tokens: Option<u64>,
    ) {
        let queue_wait_ms = duration_millis(queue_wait);
        let attempt_ms = duration_millis(attempt_time);
        let target_key = target.key();
        let target_kind = target_key.kind;
        self.globals.record_attempt(
            target_kind,
            queue_wait_ms,
            attempt_ms,
            outcome,
            completion_tokens,
            attempt_time,
        );

        if let Some(model) = normalized_model_name(model) {
            let now = Instant::now();
            let shard_index = self.shard_index(model);
            let mut shard = self.shards[shard_index].lock().unwrap();
            shard.record_attempt(
                model,
                now,
                target_key,
                queue_wait_ms,
                attempt_ms,
                outcome,
                completion_tokens,
                &self.config,
            );
        }
    }

    pub fn record_request(&self, model: Option<&str>, attempts: usize, outcome: RequestOutcome) {
        self.globals.record_request(attempts, outcome);
        if let Some(model) = normalized_model_name(model) {
            let now = Instant::now();
            let shard_index = self.shard_index(model);
            let mut shard = self.shards[shard_index].lock().unwrap();
            shard.record_request(model, now, attempts, outcome, &self.config);
        }
    }

    pub fn status_snapshot(&self, current_inflight_requests: u64) -> RoutingMetricsStatusSnapshot {
        self.globals.status_snapshot(current_inflight_requests)
    }

    pub fn model_snapshots(&self) -> HashMap<String, ModelRoutingMetricsSnapshot> {
        let now = Instant::now();
        let mut snapshots = HashMap::new();
        for shard in self.shards.iter() {
            let mut shard = shard.lock().unwrap();
            shard.compact(now, &self.config);
            snapshots.extend(
                shard
                    .models
                    .iter()
                    .map(|(name, metrics)| (name.clone(), metrics.snapshot(now))),
            );
        }
        snapshots
    }

    fn shard_index(&self, model: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        model.hash(&mut hasher);
        (hasher.finish() as usize) % self.config.shard_count
    }

    #[cfg(test)]
    fn age_model_for_test(&self, model: &str, age: Duration) {
        let shard_index = self.shard_index(model);
        let mut shard = self.shards[shard_index].lock().unwrap();
        if let Some(metrics) = shard.models.get_mut(model) {
            metrics.last_updated = Instant::now() - age;
        }
    }
}

impl Default for RoutingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
struct MetricsConfig {
    ttl: Duration,
    max_targets_per_model: usize,
    shard_count: usize,
    max_models_per_shard: usize,
}

impl MetricsConfig {
    fn new(
        ttl: Duration,
        max_models: usize,
        max_targets_per_model: usize,
        shard_count: usize,
    ) -> Self {
        let shard_count = shard_count.max(1);
        let max_models = max_models.max(1);
        let max_targets_per_model = max_targets_per_model.max(1);
        let max_models_per_shard = max_models.div_ceil(shard_count).max(1);
        Self {
            ttl,
            max_targets_per_model,
            shard_count,
            max_models_per_shard,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self::new(
            METRICS_TTL,
            MAX_TRACKED_MODELS,
            MAX_TARGETS_PER_MODEL,
            DEFAULT_MODEL_SHARDS,
        )
    }
}

#[derive(Default)]
struct GlobalMetrics {
    request_count: AtomicU64,
    successful_requests: AtomicU64,
    retry_count: AtomicU64,
    failover_count: AtomicU64,
    attempt_count: AtomicU64,
    attempt_timeout_count: AtomicU64,
    attempt_unavailable_count: AtomicU64,
    attempt_context_overflow_count: AtomicU64,
    attempt_reject_count: AtomicU64,
    queue_wait_ms_total: AtomicU64,
    attempt_ms_total: AtomicU64,
    completion_tokens_observed: AtomicU64,
    throughput_tps_milli_sum: AtomicU64,
    throughput_samples: AtomicU64,
    locally_served_request_count: AtomicU64,
    remotely_served_request_count: AtomicU64,
    endpoint_request_count: AtomicU64,
    local_attempt_count: AtomicU64,
    remote_attempt_count: AtomicU64,
    endpoint_attempt_count: AtomicU64,
    peak_inflight_requests: AtomicU64,
}

impl GlobalMetrics {
    fn observe_inflight(&self, current: u64) {
        self.peak_inflight_requests
            .fetch_max(current, Ordering::Relaxed);
    }

    fn record_attempt(
        &self,
        target_kind: TargetKind,
        queue_wait_ms: u64,
        attempt_ms: u64,
        outcome: AttemptOutcome,
        completion_tokens: Option<u64>,
        attempt_time: Duration,
    ) {
        self.attempt_count.fetch_add(1, Ordering::Relaxed);
        self.queue_wait_ms_total
            .fetch_add(queue_wait_ms, Ordering::Relaxed);
        self.attempt_ms_total
            .fetch_add(attempt_ms, Ordering::Relaxed);
        match target_kind {
            TargetKind::Local => {
                self.local_attempt_count.fetch_add(1, Ordering::Relaxed);
            }
            TargetKind::Remote => {
                self.remote_attempt_count.fetch_add(1, Ordering::Relaxed);
            }
            TargetKind::Endpoint => {
                self.endpoint_attempt_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        match outcome {
            AttemptOutcome::Success => {
                if let Some(tokens) = completion_tokens {
                    self.completion_tokens_observed
                        .fetch_add(tokens, Ordering::Relaxed);
                    if let Some(tps_milli) = tokens_per_second_milli(tokens, attempt_time) {
                        self.throughput_tps_milli_sum
                            .fetch_add(tps_milli, Ordering::Relaxed);
                        self.throughput_samples.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            AttemptOutcome::Timeout => {
                self.attempt_timeout_count.fetch_add(1, Ordering::Relaxed);
            }
            AttemptOutcome::Unavailable => {
                self.attempt_unavailable_count
                    .fetch_add(1, Ordering::Relaxed);
            }
            AttemptOutcome::ContextOverflow => {
                self.attempt_context_overflow_count
                    .fetch_add(1, Ordering::Relaxed);
            }
            AttemptOutcome::Rejected => {
                self.attempt_reject_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn record_request(&self, attempts: usize, outcome: RequestOutcome) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.retry_count
            .fetch_add(attempts.saturating_sub(1) as u64, Ordering::Relaxed);
        if attempts > 1 {
            self.failover_count.fetch_add(1, Ordering::Relaxed);
        }
        match outcome {
            RequestOutcome::Success(service) => {
                self.successful_requests.fetch_add(1, Ordering::Relaxed);
                self.record_service_request(service);
            }
            RequestOutcome::Rejected(service) => {
                self.record_service_request(service);
            }
            RequestOutcome::Unavailable => {}
        }
    }

    fn record_service_request(&self, service: RequestService) {
        match service {
            RequestService::Local => {
                self.locally_served_request_count
                    .fetch_add(1, Ordering::Relaxed);
            }
            RequestService::Remote => {
                self.remotely_served_request_count
                    .fetch_add(1, Ordering::Relaxed);
            }
            RequestService::Endpoint => {
                self.endpoint_request_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn status_snapshot(&self, current_inflight_requests: u64) -> RoutingMetricsStatusSnapshot {
        debug_assert!(metric_vocabulary_is_complete());
        debug_assert_eq!(
            metric_group("routing_metrics").scope,
            MetricScope::LocalOnly
        );
        debug_assert_eq!(
            metric_group("routing_metrics.local_node").scope,
            MetricScope::LocalOnly
        );
        debug_assert_eq!(
            metric_group("routing_metrics.pressure").scope,
            MetricScope::LocalOnly
        );
        let request_count = load_u64(&self.request_count);
        let successful_requests = load_u64(&self.successful_requests);
        let attempt_count = load_u64(&self.attempt_count);
        let completion_tokens_observed = load_u64(&self.completion_tokens_observed);
        let throughput_samples = load_u64(&self.throughput_samples);
        let avg_queue_wait_ms = average(load_u64(&self.queue_wait_ms_total), attempt_count);
        let avg_attempt_ms = average(load_u64(&self.attempt_ms_total), attempt_count);
        let avg_tokens_per_second =
            average_milli(load_u64(&self.throughput_tps_milli_sum), throughput_samples);
        let local_node = LocalNodePressureSnapshot {
            current_inflight_requests,
            peak_inflight_requests: load_u64(&self.peak_inflight_requests),
            local_attempt_count: load_u64(&self.local_attempt_count),
            remote_attempt_count: load_u64(&self.remote_attempt_count),
            endpoint_attempt_count: load_u64(&self.endpoint_attempt_count),
            avg_queue_wait_ms,
            avg_attempt_ms,
            avg_tokens_per_second,
            completion_tokens_observed,
            throughput_samples,
        };
        let fronted_request_count = request_count;
        let pressure = RoutingPressureSnapshot {
            fronted_request_count,
            locally_served_request_count: load_u64(&self.locally_served_request_count),
            remotely_served_request_count: load_u64(&self.remotely_served_request_count),
            endpoint_request_count: load_u64(&self.endpoint_request_count),
            local_service_share: ratio(
                load_u64(&self.locally_served_request_count),
                fronted_request_count,
            ),
            remote_service_share: ratio(
                load_u64(&self.remotely_served_request_count),
                fronted_request_count,
            ),
            endpoint_service_share: ratio(
                load_u64(&self.endpoint_request_count),
                fronted_request_count,
            ),
        };

        RoutingMetricsStatusSnapshot {
            request_count,
            successful_requests,
            success_rate: ratio(successful_requests, request_count),
            retry_count: load_u64(&self.retry_count),
            failover_count: load_u64(&self.failover_count),
            attempt_timeout_count: load_u64(&self.attempt_timeout_count),
            attempt_unavailable_count: load_u64(&self.attempt_unavailable_count),
            attempt_context_overflow_count: load_u64(&self.attempt_context_overflow_count),
            attempt_reject_count: load_u64(&self.attempt_reject_count),
            avg_queue_wait_ms,
            avg_attempt_ms,
            avg_tokens_per_second,
            completion_tokens_observed,
            throughput_samples,
            local_node,
            pressure,
        }
    }
}

#[derive(Default)]
struct ModelShard {
    models: HashMap<String, ModelMetrics>,
}

impl ModelShard {
    fn record_attempt(
        &mut self,
        model: &str,
        now: Instant,
        target: TargetKey,
        queue_wait_ms: u64,
        attempt_ms: u64,
        outcome: AttemptOutcome,
        completion_tokens: Option<u64>,
        config: &MetricsConfig,
    ) {
        let inserted = !self.models.contains_key(model);
        if inserted && self.models.len() >= config.max_models_per_shard {
            self.compact(now, config);
        }
        let metrics = self
            .models
            .entry(model.to_string())
            .or_insert_with(ModelMetrics::default);
        metrics.last_updated = now;
        metrics.record_attempt(
            target,
            now,
            queue_wait_ms,
            attempt_ms,
            outcome,
            completion_tokens,
            config,
        );
    }

    fn record_request(
        &mut self,
        model: &str,
        now: Instant,
        attempts: usize,
        outcome: RequestOutcome,
        config: &MetricsConfig,
    ) {
        let inserted = !self.models.contains_key(model);
        if inserted && self.models.len() >= config.max_models_per_shard {
            self.compact(now, config);
        }
        let metrics = self
            .models
            .entry(model.to_string())
            .or_insert_with(ModelMetrics::default);
        metrics.last_updated = now;
        metrics.record_request(attempts, outcome);
    }

    fn compact(&mut self, now: Instant, config: &MetricsConfig) {
        self.models
            .retain(|_, metrics| now.duration_since(metrics.last_updated) <= config.ttl);
        while self.models.len() > config.max_models_per_shard {
            let Some(oldest_key) = self
                .models
                .iter()
                .min_by_key(|(_, metrics)| metrics.last_updated)
                .map(|(name, _)| name.clone())
            else {
                break;
            };
            self.models.remove(&oldest_key);
        }
    }
}

struct ModelMetrics {
    last_updated: Instant,
    request_count: u64,
    successful_requests: u64,
    retry_count: u64,
    failover_count: u64,
    attempt_count: u64,
    attempt_timeout_count: u64,
    attempt_unavailable_count: u64,
    attempt_context_overflow_count: u64,
    attempt_reject_count: u64,
    queue_wait_ms_total: u64,
    attempt_ms_total: u64,
    completion_tokens_observed: u64,
    throughput_tps_milli_sum: u64,
    throughput_samples: u64,
    targets: HashMap<TargetKey, TargetMetrics>,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            last_updated: Instant::now(),
            request_count: 0,
            successful_requests: 0,
            retry_count: 0,
            failover_count: 0,
            attempt_count: 0,
            attempt_timeout_count: 0,
            attempt_unavailable_count: 0,
            attempt_context_overflow_count: 0,
            attempt_reject_count: 0,
            queue_wait_ms_total: 0,
            attempt_ms_total: 0,
            completion_tokens_observed: 0,
            throughput_tps_milli_sum: 0,
            throughput_samples: 0,
            targets: HashMap::new(),
        }
    }
}

impl ModelMetrics {
    fn record_attempt(
        &mut self,
        target: TargetKey,
        now: Instant,
        queue_wait_ms: u64,
        attempt_ms: u64,
        outcome: AttemptOutcome,
        completion_tokens: Option<u64>,
        config: &MetricsConfig,
    ) {
        self.last_updated = now;
        self.attempt_count += 1;
        self.queue_wait_ms_total = self.queue_wait_ms_total.saturating_add(queue_wait_ms);
        self.attempt_ms_total = self.attempt_ms_total.saturating_add(attempt_ms);
        match outcome {
            AttemptOutcome::Success => {
                if let Some(tokens) = completion_tokens {
                    self.completion_tokens_observed =
                        self.completion_tokens_observed.saturating_add(tokens);
                    if let Some(tps_milli) =
                        tokens_per_second_milli(tokens, Duration::from_millis(attempt_ms))
                    {
                        self.throughput_tps_milli_sum =
                            self.throughput_tps_milli_sum.saturating_add(tps_milli);
                        self.throughput_samples += 1;
                    }
                }
            }
            AttemptOutcome::Timeout => self.attempt_timeout_count += 1,
            AttemptOutcome::Unavailable => self.attempt_unavailable_count += 1,
            AttemptOutcome::ContextOverflow => self.attempt_context_overflow_count += 1,
            AttemptOutcome::Rejected => self.attempt_reject_count += 1,
        }

        let inserted = !self.targets.contains_key(&target);
        if inserted && self.targets.len() >= config.max_targets_per_model {
            self.compact_targets(now, config);
        }
        let metrics = self
            .targets
            .entry(target)
            .or_insert_with(TargetMetrics::default);
        metrics.last_updated = now;
        metrics.record(queue_wait_ms, attempt_ms, outcome, completion_tokens);
    }

    fn record_request(&mut self, attempts: usize, outcome: RequestOutcome) {
        self.request_count += 1;
        self.retry_count += attempts.saturating_sub(1) as u64;
        if attempts > 1 {
            self.failover_count += 1;
        }
        if matches!(outcome, RequestOutcome::Success(_)) {
            self.successful_requests += 1;
        }
    }

    fn compact_targets(&mut self, now: Instant, config: &MetricsConfig) {
        self.targets
            .retain(|_, metrics| now.duration_since(metrics.last_updated) <= config.ttl);
        while self.targets.len() > config.max_targets_per_model {
            let Some(oldest_key) = self
                .targets
                .iter()
                .min_by_key(|(_, metrics)| metrics.last_updated)
                .map(|(target, _)| target.clone())
            else {
                break;
            };
            self.targets.remove(&oldest_key);
        }
    }

    fn snapshot(&self, now: Instant) -> ModelRoutingMetricsSnapshot {
        debug_assert!(metric_vocabulary_is_complete());
        debug_assert_eq!(
            metric_group("mesh_models[].routing_metrics").scope,
            MetricScope::LocalOnly
        );
        debug_assert_eq!(
            metric_group("mesh_models[].routing_metrics.targets[]").scope,
            MetricScope::LocalOnly
        );
        let mut targets = self
            .targets
            .iter()
            .map(|(target, metrics)| TargetRoutingMetricsSnapshot {
                target: target.label.clone(),
                kind: target.kind.label().to_string(),
                attempt_count: metrics.attempt_count,
                success_count: metrics.success_count,
                success_rate: ratio(metrics.success_count, metrics.attempt_count),
                timeout_rate: ratio(metrics.timeout_count, metrics.attempt_count),
                timeout_count: metrics.timeout_count,
                unavailable_count: metrics.unavailable_count,
                context_overflow_count: metrics.context_overflow_count,
                reject_count: metrics.reject_count,
                avg_queue_wait_ms: average(metrics.queue_wait_ms_total, metrics.attempt_count),
                avg_attempt_ms: average(metrics.attempt_ms_total, metrics.attempt_count),
                avg_tokens_per_second: average_milli(
                    metrics.throughput_tps_milli_sum,
                    metrics.throughput_samples,
                ),
                completion_tokens_observed: metrics.completion_tokens_observed,
                throughput_samples: metrics.throughput_samples,
                last_updated_secs_ago: now.duration_since(metrics.last_updated).as_secs(),
            })
            .collect::<Vec<_>>();
        targets.sort_by(|a, b| {
            b.attempt_count
                .cmp(&a.attempt_count)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.target.cmp(&b.target))
        });

        ModelRoutingMetricsSnapshot {
            request_count: self.request_count,
            successful_requests: self.successful_requests,
            success_rate: ratio(self.successful_requests, self.request_count),
            retry_count: self.retry_count,
            failover_count: self.failover_count,
            attempt_timeout_count: self.attempt_timeout_count,
            attempt_unavailable_count: self.attempt_unavailable_count,
            attempt_context_overflow_count: self.attempt_context_overflow_count,
            attempt_reject_count: self.attempt_reject_count,
            avg_queue_wait_ms: average(self.queue_wait_ms_total, self.attempt_count),
            avg_attempt_ms: average(self.attempt_ms_total, self.attempt_count),
            avg_tokens_per_second: average_milli(
                self.throughput_tps_milli_sum,
                self.throughput_samples,
            ),
            completion_tokens_observed: self.completion_tokens_observed,
            throughput_samples: self.throughput_samples,
            targets,
        }
    }
}

struct TargetMetrics {
    last_updated: Instant,
    attempt_count: u64,
    success_count: u64,
    timeout_count: u64,
    unavailable_count: u64,
    context_overflow_count: u64,
    reject_count: u64,
    queue_wait_ms_total: u64,
    attempt_ms_total: u64,
    completion_tokens_observed: u64,
    throughput_tps_milli_sum: u64,
    throughput_samples: u64,
}

impl Default for TargetMetrics {
    fn default() -> Self {
        Self {
            last_updated: Instant::now(),
            attempt_count: 0,
            success_count: 0,
            timeout_count: 0,
            unavailable_count: 0,
            context_overflow_count: 0,
            reject_count: 0,
            queue_wait_ms_total: 0,
            attempt_ms_total: 0,
            completion_tokens_observed: 0,
            throughput_tps_milli_sum: 0,
            throughput_samples: 0,
        }
    }
}

impl TargetMetrics {
    fn record(
        &mut self,
        queue_wait_ms: u64,
        attempt_ms: u64,
        outcome: AttemptOutcome,
        completion_tokens: Option<u64>,
    ) {
        self.attempt_count += 1;
        self.queue_wait_ms_total = self.queue_wait_ms_total.saturating_add(queue_wait_ms);
        self.attempt_ms_total = self.attempt_ms_total.saturating_add(attempt_ms);
        match outcome {
            AttemptOutcome::Success => {
                self.success_count += 1;
                if let Some(tokens) = completion_tokens {
                    self.completion_tokens_observed =
                        self.completion_tokens_observed.saturating_add(tokens);
                    if let Some(tps_milli) =
                        tokens_per_second_milli(tokens, Duration::from_millis(attempt_ms))
                    {
                        self.throughput_tps_milli_sum =
                            self.throughput_tps_milli_sum.saturating_add(tps_milli);
                        self.throughput_samples += 1;
                    }
                }
            }
            AttemptOutcome::Timeout => self.timeout_count += 1,
            AttemptOutcome::Unavailable => self.unavailable_count += 1,
            AttemptOutcome::ContextOverflow => self.context_overflow_count += 1,
            AttemptOutcome::Rejected => self.reject_count += 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum TargetKind {
    Local,
    Remote,
    Endpoint,
}

impl TargetKind {
    fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
            Self::Endpoint => "endpoint",
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TargetKey {
    kind: TargetKind,
    label: String,
}

fn normalized_model_name(model: Option<&str>) -> Option<&str> {
    model.filter(|model| !model.is_empty() && *model != "auto")
}

fn load_u64(value: &AtomicU64) -> u64 {
    value.load(Ordering::Relaxed)
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn average(total: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn average_milli(total_milli: u64, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(total_milli as f64 / THROUGHPUT_SCALE_MILLI as f64 / count as f64)
    }
}

fn tokens_per_second_milli(tokens: u64, elapsed: Duration) -> Option<u64> {
    let secs = elapsed.as_secs_f64();
    if tokens == 0 || secs <= 0.0 {
        None
    } else {
        let scaled = (tokens as f64 / secs) * THROUGHPUT_SCALE_MILLI as f64;
        Some(scaled.max(0.0).min(u64::MAX as f64) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn routing_metric_groups_declare_explicit_layer_and_scope() {
        let mut names = ROUTING_METRIC_GROUPS
            .iter()
            .map(|group| group.name)
            .collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "mesh_models[].routing_metrics",
                "mesh_models[].routing_metrics.targets[]",
                "routing_metrics",
                "routing_metrics.local_node",
                "routing_metrics.pressure",
            ]
        );
        assert!(ROUTING_METRIC_GROUPS
            .iter()
            .all(|group| !group.description.is_empty()));
        assert!(ROUTING_METRIC_GROUPS
            .iter()
            .all(|group| !group.api_surface.is_empty()));
        assert!(ROUTING_METRIC_GROUPS.iter().all(|group| matches!(
            group.layer,
            MetricLayer::Runtime | MetricLayer::Information | MetricLayer::Strategy
        )));
        assert!(ROUTING_METRIC_GROUPS.iter().all(|group| matches!(
            group.scope,
            MetricScope::LocalOnly | MetricScope::PeerAdvertised | MetricScope::MeshDerived
        )));
        assert!(ROUTING_METRIC_GROUPS
            .iter()
            .all(|group| group.scope == MetricScope::LocalOnly));
    }

    #[test]
    fn routing_metrics_enforces_model_and_target_bounds() {
        let metrics = RoutingMetrics::with_config(Duration::from_secs(3600), 2, 2);
        metrics.record_attempt(
            Some("alpha"),
            AttemptTarget::Remote("peer-a".into()),
            Duration::from_millis(1),
            Duration::from_millis(10),
            AttemptOutcome::Success,
            Some(8),
        );
        metrics.record_attempt(
            Some("alpha"),
            AttemptTarget::Remote("peer-b".into()),
            Duration::from_millis(2),
            Duration::from_millis(12),
            AttemptOutcome::Success,
            Some(9),
        );
        metrics.record_attempt(
            Some("alpha"),
            AttemptTarget::Remote("peer-c".into()),
            Duration::from_millis(3),
            Duration::from_millis(15),
            AttemptOutcome::Timeout,
            None,
        );
        metrics.record_attempt(
            Some("beta"),
            AttemptTarget::Local("127.0.0.1:9001".into()),
            Duration::from_millis(1),
            Duration::from_millis(11),
            AttemptOutcome::Success,
            Some(7),
        );
        metrics.record_attempt(
            Some("gamma"),
            AttemptTarget::Endpoint("http://example.com".into()),
            Duration::from_millis(4),
            Duration::from_millis(20),
            AttemptOutcome::Unavailable,
            None,
        );

        let model_snapshots = metrics.model_snapshots();
        assert_eq!(model_snapshots.len(), 2);
        assert!(model_snapshots.contains_key("beta"));
        assert!(model_snapshots.contains_key("gamma"));
        assert_eq!(model_snapshots["beta"].targets.len(), 1);
    }

    #[test]
    fn routing_metrics_prunes_stale_entries_on_snapshot() {
        let metrics = RoutingMetrics::with_config(Duration::from_secs(1), 8, 8);
        metrics.record_attempt(
            Some("stale"),
            AttemptTarget::Remote("peer-a".into()),
            Duration::from_millis(1),
            Duration::from_millis(10),
            AttemptOutcome::Success,
            Some(3),
        );
        metrics.age_model_for_test("stale", Duration::from_secs(2));

        let snapshots = metrics.model_snapshots();
        assert!(snapshots.is_empty());
    }

    #[test]
    fn routing_metrics_aggregates_success_retry_and_pressure() {
        let metrics = RoutingMetrics::new();
        metrics.observe_inflight(3);
        metrics.record_attempt(
            Some("glm"),
            AttemptTarget::Remote("peer-a".into()),
            Duration::from_millis(5),
            Duration::from_millis(20),
            AttemptOutcome::Timeout,
            None,
        );
        metrics.record_attempt(
            Some("glm"),
            AttemptTarget::Remote("peer-b".into()),
            Duration::from_millis(25),
            Duration::from_millis(40),
            AttemptOutcome::Success,
            Some(12),
        );
        metrics.record_request(
            Some("glm"),
            2,
            RequestOutcome::Success(RequestService::Remote),
        );

        metrics.record_attempt(
            Some("qwen"),
            AttemptTarget::Local("127.0.0.1:9338".into()),
            Duration::from_millis(2),
            Duration::from_millis(16),
            AttemptOutcome::Rejected,
            None,
        );
        metrics.record_request(
            Some("qwen"),
            1,
            RequestOutcome::Rejected(RequestService::Local),
        );

        let status = metrics.status_snapshot(1);
        assert_eq!(status.request_count, 2);
        assert_eq!(status.successful_requests, 1);
        assert_eq!(status.retry_count, 1);
        assert_eq!(status.failover_count, 1);
        assert_eq!(status.attempt_timeout_count, 1);
        assert_eq!(status.attempt_reject_count, 1);
        assert_eq!(status.local_node.peak_inflight_requests, 3);
        assert_eq!(status.pressure.fronted_request_count, 2);
        assert_eq!(status.pressure.remotely_served_request_count, 1);
        assert_eq!(status.pressure.locally_served_request_count, 1);

        let model = metrics.model_snapshots().remove("glm").unwrap();
        assert_eq!(model.request_count, 1);
        assert_eq!(model.successful_requests, 1);
        assert_eq!(model.retry_count, 1);
        assert_eq!(model.failover_count, 1);
        assert_eq!(model.attempt_timeout_count, 1);
        assert_eq!(model.targets.len(), 2);
        assert!(model.avg_tokens_per_second.is_some());
    }

    #[test]
    fn routing_metrics_tracks_unattributed_requests_in_global_status_only() {
        let metrics = RoutingMetrics::new();
        metrics.record_attempt(
            None,
            AttemptTarget::Remote("peer-a".into()),
            Duration::from_millis(3),
            Duration::from_millis(14),
            AttemptOutcome::Unavailable,
            None,
        );
        metrics.record_request(None, 1, RequestOutcome::Unavailable);

        let status = metrics.status_snapshot(0);
        let model_snapshots = metrics.model_snapshots();
        assert_eq!(status.request_count, 1);
        assert_eq!(status.attempt_unavailable_count, 1);
        assert_eq!(status.local_node.remote_attempt_count, 1);
        assert!(model_snapshots.is_empty());
    }

    #[test]
    fn routing_metrics_ignores_auto_model_for_per_model_state() {
        let metrics = RoutingMetrics::new();
        metrics.record_attempt(
            Some("auto"),
            AttemptTarget::Local("127.0.0.1:9337".into()),
            Duration::from_millis(1),
            Duration::from_millis(5),
            AttemptOutcome::Success,
            Some(2),
        );
        metrics.record_request(
            Some("auto"),
            1,
            RequestOutcome::Success(RequestService::Local),
        );

        let status = metrics.status_snapshot(0);
        assert_eq!(status.request_count, 1);
        assert_eq!(status.successful_requests, 1);
        assert!(metrics.model_snapshots().is_empty());
    }

    #[test]
    fn observe_inflight_tracks_peak_monotonically() {
        let metrics = RoutingMetrics::new();
        metrics.observe_inflight(3);
        metrics.observe_inflight(1);
        metrics.observe_inflight(5);
        metrics.observe_inflight(2);

        let status = metrics.status_snapshot(0);
        assert_eq!(status.local_node.peak_inflight_requests, 5);
    }

    #[test]
    fn routing_metrics_concurrent_updates_preserve_totals() {
        let metrics = RoutingMetrics::with_config_and_shards(Duration::from_secs(3600), 64, 8, 8);
        let metrics = Arc::new(metrics);
        let threads = 8usize;
        let per_thread = 250usize;
        let barrier = Arc::new(Barrier::new(threads));
        let mut handles = Vec::new();

        for thread_idx in 0..threads {
            let metrics = metrics.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                let model = format!("model-{}", thread_idx % 4);
                barrier.wait();
                for _ in 0..per_thread {
                    metrics.observe_inflight((thread_idx + 1) as u64);
                    metrics.record_attempt(
                        Some(&model),
                        AttemptTarget::Remote(format!("peer-{thread_idx}")),
                        Duration::from_millis(2),
                        Duration::from_millis(10),
                        AttemptOutcome::Success,
                        Some(4),
                    );
                    metrics.record_request(
                        Some(&model),
                        1,
                        RequestOutcome::Success(RequestService::Remote),
                    );
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let status = metrics.status_snapshot(0);
        assert_eq!(status.request_count, (threads * per_thread) as u64);
        assert_eq!(status.successful_requests, (threads * per_thread) as u64);
        assert_eq!(
            status.local_node.remote_attempt_count,
            (threads * per_thread) as u64
        );

        let total_model_requests: u64 = metrics
            .model_snapshots()
            .values()
            .map(|snapshot| snapshot.request_count)
            .sum();
        assert_eq!(total_model_requests, (threads * per_thread) as u64);
    }
}
