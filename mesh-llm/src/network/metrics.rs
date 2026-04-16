//! Bounded in-memory routing and utilization metrics for operator/API surfaces.

use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const METRICS_TTL: Duration = Duration::from_secs(60 * 60);
const MAX_TRACKED_MODELS: usize = 128;
const MAX_TARGETS_PER_MODEL: usize = 16;

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
    pub local_node: LocalNodeUtilizationSnapshot,
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
            local_node: LocalNodeUtilizationSnapshot::default(),
            pressure: RoutingPressureSnapshot::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct LocalNodeUtilizationSnapshot {
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
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub targets: Vec<TargetRoutingMetricsSnapshot>,
}

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
    inner: Arc<Mutex<RoutingMetricsState>>,
    config: MetricsConfig,
}

impl RoutingMetrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RoutingMetricsState::default())),
            config: MetricsConfig::default(),
        }
    }

    #[cfg(test)]
    fn with_config(ttl: Duration, max_models: usize, max_targets_per_model: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RoutingMetricsState::default())),
            config: MetricsConfig {
                ttl,
                max_models,
                max_targets_per_model,
            },
        }
    }

    pub fn observe_inflight(&self, current: u64) {
        let mut state = self.inner.lock().unwrap();
        state.peak_inflight_requests = state.peak_inflight_requests.max(current);
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
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune(now, &self.config);

        let queue_wait_ms = duration_millis(queue_wait);
        let attempt_ms = duration_millis(attempt_time);
        let target_key = target.key();
        let target_kind = target_key.kind;

        if let Some(model) = model.filter(|model| !model.is_empty() && *model != "auto") {
            let model_metrics = state.touch_model(model, now, &self.config);
            model_metrics.record_attempt(
                target_key,
                now,
                queue_wait_ms,
                attempt_ms,
                outcome,
                completion_tokens,
                &self.config,
            );
        }
        state.record_attempt_totals(
            target_kind,
            queue_wait_ms,
            attempt_ms,
            outcome,
            completion_tokens,
            attempt_time,
        );
    }

    pub fn record_request(&self, model: Option<&str>, attempts: usize, outcome: RequestOutcome) {
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune(now, &self.config);
        if let Some(model) = model.filter(|model| !model.is_empty() && *model != "auto") {
            let model_metrics = state.touch_model(model, now, &self.config);
            model_metrics.record_request(attempts, outcome);
        }
        state.record_request_totals(attempts, outcome);
    }

    pub fn status_snapshot(&self, current_inflight_requests: u64) -> RoutingMetricsStatusSnapshot {
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune(now, &self.config);
        let request_count = state.request_count;
        let successful_requests = state.successful_requests;
        let success_rate = ratio(successful_requests, request_count);
        let avg_queue_wait_ms = average(state.queue_wait_ms_total, state.attempt_count);
        let avg_attempt_ms = average(state.attempt_ms_total, state.attempt_count);
        let avg_tokens_per_second = average_f64(state.throughput_tps_sum, state.throughput_samples);
        let local_node = LocalNodeUtilizationSnapshot {
            current_inflight_requests,
            peak_inflight_requests: state.peak_inflight_requests,
            local_attempt_count: state.local_attempt_count,
            remote_attempt_count: state.remote_attempt_count,
            endpoint_attempt_count: state.endpoint_attempt_count,
            avg_queue_wait_ms,
            avg_attempt_ms,
            avg_tokens_per_second,
            completion_tokens_observed: state.completion_tokens_observed,
            throughput_samples: state.throughput_samples,
        };
        let fronted_request_count = state.request_count;
        let local_service_share = ratio(state.locally_served_request_count, fronted_request_count);
        let remote_service_share =
            ratio(state.remotely_served_request_count, fronted_request_count);
        let endpoint_service_share = ratio(state.endpoint_request_count, fronted_request_count);
        let pressure = RoutingPressureSnapshot {
            fronted_request_count,
            locally_served_request_count: state.locally_served_request_count,
            remotely_served_request_count: state.remotely_served_request_count,
            endpoint_request_count: state.endpoint_request_count,
            local_service_share,
            remote_service_share,
            endpoint_service_share,
        };
        RoutingMetricsStatusSnapshot {
            request_count,
            successful_requests,
            success_rate,
            retry_count: state.retry_count,
            failover_count: state.failover_count,
            attempt_timeout_count: state.attempt_timeout_count,
            attempt_unavailable_count: state.attempt_unavailable_count,
            attempt_context_overflow_count: state.attempt_context_overflow_count,
            attempt_reject_count: state.attempt_reject_count,
            avg_queue_wait_ms,
            avg_attempt_ms,
            avg_tokens_per_second,
            completion_tokens_observed: state.completion_tokens_observed,
            throughput_samples: state.throughput_samples,
            local_node,
            pressure,
        }
    }

    pub fn model_snapshots(&self) -> HashMap<String, ModelRoutingMetricsSnapshot> {
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune(now, &self.config);
        state
            .models
            .iter()
            .map(|(name, metrics)| (name.clone(), metrics.snapshot(now)))
            .collect()
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
    max_models: usize,
    max_targets_per_model: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            ttl: METRICS_TTL,
            max_models: MAX_TRACKED_MODELS,
            max_targets_per_model: MAX_TARGETS_PER_MODEL,
        }
    }
}

#[derive(Default)]
struct RoutingMetricsState {
    models: HashMap<String, ModelMetrics>,
    lru: VecDeque<String>,
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
    throughput_tps_sum: f64,
    throughput_samples: u64,
    locally_served_request_count: u64,
    remotely_served_request_count: u64,
    endpoint_request_count: u64,
    local_attempt_count: u64,
    remote_attempt_count: u64,
    endpoint_attempt_count: u64,
    peak_inflight_requests: u64,
}

impl RoutingMetricsState {
    fn prune(&mut self, now: Instant, config: &MetricsConfig) {
        while let Some(model_name) = self.lru.front().cloned() {
            match self.models.get(&model_name) {
                Some(metrics) if now.duration_since(metrics.last_updated) > config.ttl => {
                    self.lru.pop_front();
                    self.models.remove(&model_name);
                }
                Some(_) => break,
                None => {
                    self.lru.pop_front();
                }
            }
        }
    }

    fn touch_model<'a>(
        &'a mut self,
        model: &str,
        now: Instant,
        config: &MetricsConfig,
    ) -> &'a mut ModelMetrics {
        if let Some(pos) = self.lru.iter().position(|existing| existing == model) {
            self.lru.remove(pos);
        }
        if !self.models.contains_key(model) && self.models.len() >= config.max_models {
            while self.models.len() >= config.max_models {
                if let Some(oldest) = self.lru.pop_front() {
                    if oldest != model && self.models.remove(&oldest).is_some() {
                        continue;
                    }
                    if oldest == model {
                        self.lru.push_back(oldest);
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        self.lru.push_back(model.to_string());
        let metrics = self
            .models
            .entry(model.to_string())
            .or_insert_with(ModelMetrics::default);
        metrics.last_updated = now;
        metrics
    }

    fn record_attempt_totals(
        &mut self,
        target_kind: TargetKind,
        queue_wait_ms: u64,
        attempt_ms: u64,
        outcome: AttemptOutcome,
        completion_tokens: Option<u64>,
        attempt_time: Duration,
    ) {
        self.attempt_count += 1;
        self.queue_wait_ms_total = self.queue_wait_ms_total.saturating_add(queue_wait_ms);
        self.attempt_ms_total = self.attempt_ms_total.saturating_add(attempt_ms);
        match target_kind {
            TargetKind::Local => self.local_attempt_count += 1,
            TargetKind::Remote => self.remote_attempt_count += 1,
            TargetKind::Endpoint => self.endpoint_attempt_count += 1,
        }
        match outcome {
            AttemptOutcome::Success => {
                if let Some(tokens) = completion_tokens {
                    self.completion_tokens_observed =
                        self.completion_tokens_observed.saturating_add(tokens);
                    if let Some(tps) = tokens_per_second(tokens, attempt_time) {
                        self.throughput_tps_sum += tps;
                        self.throughput_samples += 1;
                    }
                }
            }
            AttemptOutcome::Timeout => self.attempt_timeout_count += 1,
            AttemptOutcome::Unavailable => self.attempt_unavailable_count += 1,
            AttemptOutcome::ContextOverflow => self.attempt_context_overflow_count += 1,
            AttemptOutcome::Rejected => self.attempt_reject_count += 1,
        }
    }

    fn record_request_totals(&mut self, attempts: usize, outcome: RequestOutcome) {
        self.request_count += 1;
        self.retry_count += attempts.saturating_sub(1) as u64;
        if attempts > 1 {
            self.failover_count += 1;
        }
        match outcome {
            RequestOutcome::Success(service) => {
                self.successful_requests += 1;
                self.record_service_request(service);
            }
            RequestOutcome::Rejected(service) => {
                self.record_service_request(service);
            }
            RequestOutcome::Unavailable => {}
        }
    }

    fn record_service_request(&mut self, service: RequestService) {
        match service {
            RequestService::Local => self.locally_served_request_count += 1,
            RequestService::Remote => self.remotely_served_request_count += 1,
            RequestService::Endpoint => self.endpoint_request_count += 1,
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
    throughput_tps_sum: f64,
    throughput_samples: u64,
    targets: HashMap<TargetKey, TargetMetrics>,
    target_lru: VecDeque<TargetKey>,
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
            throughput_tps_sum: 0.0,
            throughput_samples: 0,
            targets: HashMap::new(),
            target_lru: VecDeque::new(),
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
                    if let Some(tps) = tokens_per_second(tokens, Duration::from_millis(attempt_ms))
                    {
                        self.throughput_tps_sum += tps;
                        self.throughput_samples += 1;
                    }
                }
            }
            AttemptOutcome::Timeout => self.attempt_timeout_count += 1,
            AttemptOutcome::Unavailable => self.attempt_unavailable_count += 1,
            AttemptOutcome::ContextOverflow => self.attempt_context_overflow_count += 1,
            AttemptOutcome::Rejected => self.attempt_reject_count += 1,
        }

        if let Some(pos) = self
            .target_lru
            .iter()
            .position(|existing| existing == &target)
        {
            self.target_lru.remove(pos);
        }
        self.target_lru.push_back(target.clone());
        let metrics = self
            .targets
            .entry(target.clone())
            .or_insert_with(TargetMetrics::default);
        metrics.last_updated = now;
        metrics.record(queue_wait_ms, attempt_ms, outcome, completion_tokens);
        self.prune_targets(now, config);
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

    fn prune_targets(&mut self, now: Instant, config: &MetricsConfig) {
        while let Some(target_key) = self.target_lru.front().cloned() {
            match self.targets.get(&target_key) {
                Some(metrics) if now.duration_since(metrics.last_updated) > config.ttl => {
                    self.target_lru.pop_front();
                    self.targets.remove(&target_key);
                }
                Some(_) => break,
                None => {
                    self.target_lru.pop_front();
                }
            }
        }
        while self.targets.len() > config.max_targets_per_model {
            if let Some(oldest) = self.target_lru.pop_front() {
                if self.targets.remove(&oldest).is_some() {
                    continue;
                }
            } else {
                break;
            }
        }
    }

    fn snapshot(&self, now: Instant) -> ModelRoutingMetricsSnapshot {
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
                avg_tokens_per_second: average_f64(
                    metrics.throughput_tps_sum,
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
            avg_tokens_per_second: average_f64(self.throughput_tps_sum, self.throughput_samples),
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
    throughput_tps_sum: f64,
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
            throughput_tps_sum: 0.0,
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
                    if let Some(tps) = tokens_per_second(tokens, Duration::from_millis(attempt_ms))
                    {
                        self.throughput_tps_sum += tps;
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

fn average_f64(total: f64, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

fn tokens_per_second(tokens: u64, elapsed: Duration) -> Option<f64> {
    let secs = elapsed.as_secs_f64();
    if tokens == 0 || secs <= 0.0 {
        None
    } else {
        Some(tokens as f64 / secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        {
            let mut state = metrics.inner.lock().unwrap();
            state.models.get_mut("stale").unwrap().last_updated =
                Instant::now() - Duration::from_secs(2);
        }

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
}
