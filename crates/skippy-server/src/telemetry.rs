use std::{
    collections::{BTreeMap, VecDeque},
    env,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use clap::ValueEnum;
use opentelemetry_proto::tonic::{
    collector::trace::v1::{trace_service_client::TraceServiceClient, ExportTraceServiceRequest},
    common::v1::{any_value, AnyValue, InstrumentationScope, KeyValue},
    resource::v1::Resource,
    trace::v1::{span, ResourceSpans, ScopeSpans, Span},
};
use serde::Serialize;
use serde_json::{json, Value};
use skippy_metrics::{attr, metric};
use skippy_protocol::StageConfig;
use tokio::{
    sync::mpsc,
    time::{sleep, timeout},
};

static EVENT_COUNTER: AtomicU64 = AtomicU64::new(1);
const EXPORT_TIMEOUT: Duration = Duration::from_secs(10);
const EXPORT_BATCH_MAX: usize = 128;
const RETRY_BUFFER_CAPACITY: usize = 8192;
const RETRY_BACKOFF: Duration = Duration::from_millis(100);

#[derive(Clone)]
pub struct Telemetry {
    tx: Option<mpsc::Sender<TelemetryEvent>>,
    stats: Arc<TelemetryCounters>,
    config: Arc<StageConfig>,
    level: TelemetryLevel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TelemetryLevel {
    Off,
    Summary,
    Debug,
}

#[derive(Default)]
struct TelemetryCounters {
    queued: AtomicU64,
    sent: AtomicU64,
    dropped: AtomicU64,
    export_errors: AtomicU64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TelemetryStats {
    pub queued: u64,
    pub sent: u64,
    pub dropped: u64,
    pub export_errors: u64,
}

#[derive(Clone)]
struct TelemetryEvent {
    id: u64,
    name: String,
    attributes: BTreeMap<String, Value>,
    start_time_unix_nanos: u64,
    end_time_unix_nanos: u64,
}

impl Telemetry {
    pub fn new(
        endpoint: Option<String>,
        capacity: usize,
        config: StageConfig,
        level: TelemetryLevel,
    ) -> Self {
        let stats = Arc::new(TelemetryCounters::default());
        let config = Arc::new(config);
        let tx = (level != TelemetryLevel::Off)
            .then_some(endpoint)
            .flatten()
            .map(|endpoint| {
                let (tx, rx) = mpsc::channel(capacity);
                tokio::spawn(telemetry_export_loop(
                    endpoint,
                    rx,
                    stats.clone(),
                    config.clone(),
                ));
                tx
            });

        Self {
            tx,
            stats,
            config,
            level,
        }
    }

    pub fn emit(&self, name: &str, mut attributes: BTreeMap<String, Value>) {
        if self.level == TelemetryLevel::Off {
            return;
        }
        self.add_common_attributes(&mut attributes);

        let now = now_unix_nanos() as u64;
        self.enqueue(name, attributes, now, now + 1);
    }

    pub fn emit_span(
        &self,
        name: &str,
        mut attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if self.level == TelemetryLevel::Off {
            return;
        }
        self.add_common_attributes(&mut attributes);
        self.enqueue(
            name,
            attributes,
            start_time_unix_nanos,
            end_time_unix_nanos.max(start_time_unix_nanos + 1),
        );
    }

    fn add_common_attributes(&self, attributes: &mut BTreeMap<String, Value>) {
        attributes
            .entry(attr::RUN_ID.to_string())
            .or_insert_with(|| json!(self.config.run_id));
        attributes
            .entry(attr::STAGE_ID.to_string())
            .or_insert_with(|| json!(self.config.stage_id));
        attributes.insert(
            metric::OTEL_DROPPED_EVENTS.to_string(),
            json!(self.stats.dropped.load(Ordering::Relaxed)),
        );
        attributes.insert(
            metric::OTEL_EXPORT_ERRORS.to_string(),
            json!(self.stats.export_errors.load(Ordering::Relaxed)),
        );
    }

    fn enqueue(
        &self,
        name: &str,
        attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if env::var_os("SKIPPY_TELEMETRY_STDERR").is_some() {
            let line = json!({
                "event": name,
                "attributes": attributes,
                "start_time_unix_nanos": start_time_unix_nanos,
                "end_time_unix_nanos": end_time_unix_nanos,
            });
            eprintln!("{line}");
        }
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        let event = TelemetryEvent {
            id: EVENT_COUNTER.fetch_add(1, Ordering::Relaxed),
            name: name.to_string(),
            attributes,
            start_time_unix_nanos,
            end_time_unix_nanos,
        };
        match tx.try_send(event) {
            Ok(()) => {
                self.stats.queued.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn emit_debug(&self, name: &str, attributes: BTreeMap<String, Value>) {
        if self.level == TelemetryLevel::Debug {
            self.emit(name, attributes);
        }
    }

    pub fn emit_debug_span(
        &self,
        name: &str,
        attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if self.level == TelemetryLevel::Debug {
            self.emit_span(name, attributes, start_time_unix_nanos, end_time_unix_nanos);
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.level != TelemetryLevel::Off
    }

    pub fn is_debug_enabled(&self) -> bool {
        self.level == TelemetryLevel::Debug
    }

    pub fn stats(&self) -> TelemetryStats {
        TelemetryStats {
            queued: self.stats.queued.load(Ordering::Relaxed),
            sent: self.stats.sent.load(Ordering::Relaxed),
            dropped: self.stats.dropped.load(Ordering::Relaxed),
            export_errors: self.stats.export_errors.load(Ordering::Relaxed),
        }
    }
}

pub fn lifecycle_attrs(config: &StageConfig) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (attr::RUN_ID.to_string(), json!(config.run_id)),
        (attr::MODEL_ID.to_string(), json!(config.model_id)),
        (attr::TOPOLOGY_ID.to_string(), json!(config.topology_id)),
        (attr::STAGE_ID.to_string(), json!(config.stage_id)),
        (attr::STAGE_INDEX.to_string(), json!(config.stage_index)),
        (attr::LAYER_START.to_string(), json!(config.layer_start)),
        (attr::LAYER_END.to_string(), json!(config.layer_end)),
        (
            attr::LOAD_MODE.to_string(),
            serde_json::to_value(&config.load_mode).unwrap_or(Value::Null),
        ),
    ])
}

pub fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos() as i64
}

async fn telemetry_export_loop(
    endpoint: String,
    mut rx: mpsc::Receiver<TelemetryEvent>,
    stats: Arc<TelemetryCounters>,
    config: Arc<StageConfig>,
) {
    let mut client = None;
    let mut retry_buffer = VecDeque::new();

    loop {
        let Some(first_event) = next_export_event(&mut rx, &mut retry_buffer).await else {
            break;
        };
        let mut batch = Vec::with_capacity(EXPORT_BATCH_MAX);
        batch.push(first_event);
        fill_export_batch(&mut rx, &mut retry_buffer, &mut batch);

        let event_count = batch.len() as u64;
        if export_batch(&endpoint, &mut client, &config, &stats, &batch).await {
            stats.sent.fetch_add(event_count, Ordering::Relaxed);
        } else {
            stats
                .export_errors
                .fetch_add(event_count, Ordering::Relaxed);
            let dropped = requeue_failed_batch(&mut retry_buffer, batch);
            if dropped > 0 {
                stats.dropped.fetch_add(dropped as u64, Ordering::Relaxed);
            }
            sleep(RETRY_BACKOFF).await;
        }
    }
}

async fn next_export_event(
    rx: &mut mpsc::Receiver<TelemetryEvent>,
    retry_buffer: &mut VecDeque<TelemetryEvent>,
) -> Option<TelemetryEvent> {
    match retry_buffer.pop_front() {
        Some(event) => Some(event),
        None => rx.recv().await,
    }
}

fn fill_export_batch(
    rx: &mut mpsc::Receiver<TelemetryEvent>,
    retry_buffer: &mut VecDeque<TelemetryEvent>,
    batch: &mut Vec<TelemetryEvent>,
) {
    while batch.len() < EXPORT_BATCH_MAX {
        if let Some(event) = retry_buffer.pop_front() {
            batch.push(event);
            continue;
        }
        match rx.try_recv() {
            Ok(event) => batch.push(event),
            Err(_) => break,
        }
    }
}

async fn export_batch(
    endpoint: &str,
    client: &mut Option<TraceServiceClient<tonic::transport::Channel>>,
    config: &StageConfig,
    stats: &TelemetryCounters,
    events: &[TelemetryEvent],
) -> bool {
    if client.is_none() {
        match timeout(
            EXPORT_TIMEOUT,
            TraceServiceClient::connect(endpoint.to_string()),
        )
        .await
        {
            Ok(Ok(connected)) => {
                *client = Some(connected);
            }
            Ok(Err(_)) | Err(_) => return false,
        }
    }

    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: Some(Resource {
                attributes: resource_attributes(config),
                dropped_attributes_count: 0,
                entity_refs: vec![],
            }),
            scope_spans: vec![ScopeSpans {
                scope: Some(InstrumentationScope {
                    name: "skippy-server".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    attributes: vec![],
                    dropped_attributes_count: 0,
                }),
                spans: events
                    .iter()
                    .map(|event| event_to_span(event, stats))
                    .collect(),
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    };

    let export = client
        .as_mut()
        .expect("client is connected above")
        .export(request);
    match timeout(EXPORT_TIMEOUT, export).await {
        Ok(Ok(_)) => true,
        Ok(Err(_)) | Err(_) => {
            *client = None;
            false
        }
    }
}

fn requeue_failed_batch(
    retry_buffer: &mut VecDeque<TelemetryEvent>,
    batch: Vec<TelemetryEvent>,
) -> usize {
    let overflow = retry_buffer
        .len()
        .saturating_add(batch.len())
        .saturating_sub(RETRY_BUFFER_CAPACITY);
    for _ in 0..overflow {
        retry_buffer.pop_back();
    }
    for event in batch.into_iter().rev() {
        retry_buffer.push_front(event);
    }
    overflow
}

fn event_to_span(event: &TelemetryEvent, stats: &TelemetryCounters) -> Span {
    let mut attributes = event.attributes.clone();
    attributes.insert(
        metric::OTEL_DROPPED_EVENTS.to_string(),
        json!(stats.dropped.load(Ordering::Relaxed)),
    );
    attributes.insert(
        metric::OTEL_EXPORT_ERRORS.to_string(),
        json!(stats.export_errors.load(Ordering::Relaxed)),
    );
    Span {
        trace_id: trace_id(event.start_time_unix_nanos, event.id),
        span_id: span_id(event.id),
        trace_state: String::new(),
        parent_span_id: vec![],
        flags: 1,
        name: event.name.clone(),
        kind: span::SpanKind::Internal as i32,
        start_time_unix_nano: event.start_time_unix_nanos,
        end_time_unix_nano: event.end_time_unix_nanos,
        attributes: attributes
            .into_iter()
            .map(|(key, value)| KeyValue {
                key,
                value: Some(json_to_any_value(value)),
            })
            .collect(),
        dropped_attributes_count: 0,
        events: vec![],
        dropped_events_count: 0,
        links: vec![],
        dropped_links_count: 0,
        status: None,
    }
}

fn resource_attributes(config: &StageConfig) -> Vec<KeyValue> {
    lifecycle_attrs(config)
        .into_iter()
        .map(|(key, value)| KeyValue {
            key,
            value: Some(json_to_any_value(value)),
        })
        .collect()
}

fn json_to_any_value(value: Value) -> AnyValue {
    let value = match value {
        Value::String(value) => any_value::Value::StringValue(value),
        Value::Bool(value) => any_value::Value::BoolValue(value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                any_value::Value::IntValue(value)
            } else {
                any_value::Value::DoubleValue(value.as_f64().unwrap_or_default())
            }
        }
        Value::Array(values) => {
            any_value::Value::ArrayValue(opentelemetry_proto::tonic::common::v1::ArrayValue {
                values: values.into_iter().map(json_to_any_value).collect(),
            })
        }
        Value::Object(values) => {
            any_value::Value::KvlistValue(opentelemetry_proto::tonic::common::v1::KeyValueList {
                values: values
                    .into_iter()
                    .map(|(key, value)| KeyValue {
                        key,
                        value: Some(json_to_any_value(value)),
                    })
                    .collect(),
            })
        }
        Value::Null => any_value::Value::StringValue(String::new()),
    };
    AnyValue { value: Some(value) }
}

fn trace_id(start_time_unix_nanos: u64, event_id: u64) -> Vec<u8> {
    let mut id = [0_u8; 16];
    id[..8].copy_from_slice(&start_time_unix_nanos.to_be_bytes());
    id[8..].copy_from_slice(&event_id.to_be_bytes());
    id.to_vec()
}

fn span_id(counter: u64) -> Vec<u8> {
    counter.to_be_bytes().to_vec()
}
