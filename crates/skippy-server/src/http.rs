use std::{
    collections::BTreeMap,
    future::Future,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use skippy_metrics::attr;
use skippy_protocol::{
    AckMessage, MessageBase, StageConfig, StageMessage, StageTopology, TokenReplyMessage,
    SCHEMA_VERSION,
};
use tokio::net::TcpListener;

use crate::{
    cli::ServeArgs,
    config::{load_json, validate_config},
    kv_integration::KvStageIntegration,
    runtime_state::{kv_desc_annotations, load_runtime, RuntimeState},
    telemetry::{lifecycle_attrs, now_unix_nanos, Telemetry, TelemetryLevel, TelemetryStats},
};

struct KvRecordCandidate {
    attrs: BTreeMap<String, Value>,
    session_id: String,
    page_id: String,
    identity: crate::kv_proto::PageIdentity,
    token_start: u64,
    token_count: u64,
}

#[derive(Clone)]
struct AppState {
    config: Arc<StageConfig>,
    topology: Option<Arc<StageTopology>>,
    runtime: Option<Arc<Mutex<RuntimeState>>>,
    kv: Option<Arc<KvStageIntegration>>,
    lifecycle: Arc<Mutex<LifecycleState>>,
    telemetry: Telemetry,
}

#[derive(Default)]
struct LifecycleState {
    started_at_unix_nanos: i64,
    ready: bool,
    peer_ready: BTreeMap<String, ReadyPeer>,
    received_messages: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReadyPeer {
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct StatusBody {
    pub status: &'static str,
    pub run_id: String,
    pub topology_id: String,
    pub model_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub topology_stage_count: Option<usize>,
    pub runtime_loaded: bool,
    pub kv_mode: Option<String>,
    pub ready: bool,
    pub started_at_unix_nanos: i64,
    pub received_messages: u64,
    pub peer_ready: Vec<ReadyPeer>,
    pub telemetry: TelemetryStats,
}

#[derive(Deserialize)]
struct TextRequest {
    request_id: String,
    session_id: String,
    prompt: String,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default = "default_add_special")]
    add_special: bool,
}

#[derive(Serialize)]
struct TextResponse {
    request_id: String,
    session_id: String,
    prompt_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
    generated_text: String,
}

fn default_max_new_tokens() -> usize {
    1
}

fn default_add_special() -> bool {
    true
}

#[derive(Debug)]
struct AppError(anyhow::Error);

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}

#[derive(Clone)]
pub struct StageHttpOptions {
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub bind_addr: SocketAddr,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: TelemetryLevel,
}

impl StageHttpOptions {
    pub fn from_cli_args(args: ServeArgs) -> Result<Self> {
        let config = load_json::<StageConfig>(&args.config)
            .with_context(|| format!("load stage config {}", args.config.display()))?;
        let topology = match args.topology.as_ref() {
            Some(path) => Some(
                load_json::<StageTopology>(path)
                    .with_context(|| format!("load topology {}", path.display()))?,
            ),
            None => None,
        };
        let bind_addr = args.bind_addr.unwrap_or(config.bind_addr.parse()?);
        Ok(Self {
            config,
            topology,
            bind_addr,
            metrics_otlp_grpc: args.metrics_otlp_grpc,
            telemetry_queue_capacity: args.telemetry_queue_capacity,
            telemetry_level: args.telemetry_level,
        })
    }
}

pub async fn serve(args: ServeArgs) -> Result<()> {
    serve_stage_http(StageHttpOptions::from_cli_args(args)?).await
}

pub async fn serve_stage_http(options: StageHttpOptions) -> Result<()> {
    serve_stage_http_with_shutdown(options, std::future::pending::<()>()).await
}

pub async fn serve_stage_http_with_shutdown(
    options: StageHttpOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let bind_addr = options.bind_addr;
    let stage_id = options.config.stage_id.clone();
    let layer_start = options.config.layer_start;
    let layer_end = options.config.layer_end;
    let load_mode = options.config.load_mode.clone();
    let app = stage_http_router(options)?;

    println!(
        "skippy-server listening: http={} stage_id={} layer_range={}..{} load_mode={:?}",
        bind_addr, stage_id, layer_start, layer_end, load_mode,
    );

    let listener = TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

pub fn stage_http_router(options: StageHttpOptions) -> Result<Router> {
    let StageHttpOptions {
        config,
        topology,
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        telemetry_level,
        ..
    } = options;
    validate_config(&config, topology.as_ref())?;
    let telemetry = Telemetry::new(
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        config.clone(),
        telemetry_level,
    );
    telemetry.emit("stage.server_start", lifecycle_attrs(&config));
    let runtime = load_runtime(&config)?;
    let kv = KvStageIntegration::from_config(&config)?.map(Arc::new);

    let state = AppState {
        config: Arc::new(config),
        topology: topology.map(Arc::new),
        runtime,
        kv,
        lifecycle: Arc::new(Mutex::new(LifecycleState {
            started_at_unix_nanos: now_unix_nanos(),
            ready: true,
            peer_ready: BTreeMap::new(),
            received_messages: 0,
        })),
        telemetry,
    };

    Ok(Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/status", get(status))
        .route("/v1/ready", post(peer_ready))
        .route("/v1/messages", post(message))
        .route("/v1/text", post(text_entrypoint))
        .with_state(state))
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn ready(State(state): State<AppState>) -> Json<StageMessage> {
    state
        .telemetry
        .emit("stage.ready_status", lifecycle_attrs(&state.config));
    Json(state.config.ready_message())
}

async fn status(State(state): State<AppState>) -> Json<StatusBody> {
    Json(status_body(&state))
}

async fn peer_ready(
    State(state): State<AppState>,
    Json(message): Json<StageMessage>,
) -> Result<Json<StageMessage>, AppError> {
    validate_message(&state.config, &message)?;
    let StageMessage::Ready(ready) = message else {
        return Err(AppError(anyhow!("expected ready message")));
    };

    {
        let mut lifecycle = state.lifecycle.lock().expect("lifecycle lock poisoned");
        lifecycle.peer_ready.insert(
            ready.base.stage_id.clone(),
            ReadyPeer {
                stage_id: ready.base.stage_id.clone(),
                stage_index: ready.base.stage_index,
                layer_start: ready.layer_start,
                layer_end: ready.layer_end,
            },
        );
    }

    let ack = StageMessage::Ack(AckMessage {
        base: local_reply_base(&state.config, &ready.base),
        acked_seq: ready.base.seq.unwrap_or(0),
    });
    state
        .telemetry
        .emit("stage.ready_handshake", lifecycle_attrs(&state.config));
    Ok(Json(ack))
}

async fn message(
    State(state): State<AppState>,
    Json(message): Json<StageMessage>,
) -> Result<Json<StageMessage>, AppError> {
    validate_message(&state.config, &message)?;
    {
        let mut lifecycle = state.lifecycle.lock().expect("lifecycle lock poisoned");
        lifecycle.received_messages += 1;
    }

    let mut attrs = lifecycle_attrs(&state.config);
    attrs.insert(
        "skippy.message_kind".to_string(),
        json!(format!("{:?}", message.kind())),
    );
    attrs.insert(
        attr::REQUEST_ID.to_string(),
        json!(message.base().request_id.clone()),
    );
    attrs.insert(
        attr::SESSION_ID.to_string(),
        json!(message.base().session_id.clone()),
    );
    state.telemetry.emit("stage.recv", attrs);

    let response = match message {
        StageMessage::PrefillChunk(prefill) => {
            let restored_tokens = maybe_lookup_prefill(
                &state,
                state.runtime.as_ref(),
                &prefill.base,
                prefill.prompt_token_start,
                &prefill.token_ids,
            )
            .await;
            if let Some(runtime) = state.runtime.as_ref() {
                if restored_tokens < prefill.token_ids.len() {
                    let records = {
                        let mut runtime = runtime.lock().expect("runtime lock poisoned");
                        runtime.prefill(
                            &prefill.base.session_id,
                            &prefill.token_ids[restored_tokens..],
                        )?;
                        let records = maybe_plan_record_prefill(
                            &state,
                            &prefill.base,
                            prefill.prompt_token_start,
                            &prefill.token_ids,
                            restored_tokens as u64,
                        );
                        state
                            .telemetry
                            .emit("stage.llama_decode", lifecycle_attrs(&state.config));
                        records
                    };
                    spawn_record_prefill(state.clone(), records);
                }
            }
            StageMessage::PrefillChunk(prefill).ack_for(&state.config)
        }
        StageMessage::FinalPrefillChunk(prefill) => {
            let restored_tokens = maybe_lookup_prefill(
                &state,
                state.runtime.as_ref(),
                &prefill.base,
                prefill.prompt_token_start,
                &prefill.token_ids,
            )
            .await;
            if let Some(runtime) = state.runtime.as_ref() {
                if restored_tokens < prefill.token_ids.len() {
                    let records = {
                        let mut runtime = runtime.lock().expect("runtime lock poisoned");
                        runtime.prefill(
                            &prefill.base.session_id,
                            &prefill.token_ids[restored_tokens..],
                        )?;
                        let records = maybe_plan_record_prefill(
                            &state,
                            &prefill.base,
                            prefill.prompt_token_start,
                            &prefill.token_ids,
                            restored_tokens as u64,
                        );
                        state
                            .telemetry
                            .emit("stage.llama_decode", lifecycle_attrs(&state.config));
                        records
                    };
                    spawn_record_prefill(state.clone(), records);
                }
            }
            StageMessage::FinalPrefillChunk(prefill).ack_for(&state.config)
        }
        StageMessage::DecodeToken(decode) if state.config.downstream.is_none() => {
            let token_id = if let Some(runtime) = state.runtime.as_ref() {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                let token = runtime.decode(&decode.base.session_id, decode.token_id)?;
                state
                    .telemetry
                    .emit("stage.llama_decode", lifecycle_attrs(&state.config));
                token
            } else {
                decode.token_id
            };
            StageMessage::TokenReply(TokenReplyMessage {
                base: local_reply_base(&state.config, &decode.base),
                token_id,
                decode_index: Some(decode.decode_index),
            })
        }
        StageMessage::Stop(stop) => {
            maybe_drop_kv_session(&state, &stop.base.session_id).await;
            StageMessage::Ack(AckMessage {
                base: local_reply_base(&state.config, &stop.base),
                acked_seq: stop.base.seq.unwrap_or(0),
            })
        }
        other => other.ack_for(&state.config),
    };

    Ok(Json(response))
}

async fn text_entrypoint(
    State(state): State<AppState>,
    Json(request): Json<TextRequest>,
) -> Result<Json<TextResponse>, AppError> {
    let Some(runtime) = state.runtime.as_ref() else {
        return Err(AppError(anyhow!(
            "stage config does not include model_path"
        )));
    };

    let prompt_token_ids = {
        let runtime = runtime.lock().expect("runtime lock poisoned");
        runtime
            .model
            .tokenize(&request.prompt, request.add_special)?
    };
    if prompt_token_ids.is_empty() {
        return Err(AppError(anyhow!("prompt produced no tokens")));
    }

    let mut pending_kv_records = Vec::new();
    if prompt_token_ids.len() > 1 {
        let base = text_message_base(&state.config, &request);
        let restored_tokens = maybe_lookup_prefill(
            &state,
            state.runtime.as_ref(),
            &base,
            0,
            &prompt_token_ids[..prompt_token_ids.len() - 1],
        )
        .await;
        if restored_tokens < prompt_token_ids.len() - 1 {
            pending_kv_records = {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                runtime.prefill(
                    &request.session_id,
                    &prompt_token_ids[restored_tokens..prompt_token_ids.len() - 1],
                )?;
                maybe_plan_record_prefill(
                    &state,
                    &base,
                    0,
                    &prompt_token_ids[..prompt_token_ids.len() - 1],
                    restored_tokens as u64,
                )
            };
        }
    }

    let mut current = *prompt_token_ids.last().expect("checked non-empty prompt");
    let mut generated_token_ids = Vec::new();
    {
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        for _ in 0..request.max_new_tokens {
            current = runtime.decode(&request.session_id, current)?;
            generated_token_ids.push(current);
        }
    }
    let generated_text = {
        let runtime = runtime.lock().expect("runtime lock poisoned");
        runtime.model.detokenize(&generated_token_ids)?
    };
    spawn_record_prefill(state.clone(), pending_kv_records);

    let mut attrs = lifecycle_attrs(&state.config);
    attrs.insert(attr::REQUEST_ID.to_string(), json!(request.request_id));
    attrs.insert(attr::SESSION_ID.to_string(), json!(request.session_id));
    state.telemetry.emit("stage.text_entrypoint", attrs);

    Ok(Json(TextResponse {
        request_id: request.request_id,
        session_id: request.session_id,
        prompt_token_ids,
        generated_token_ids,
        generated_text,
    }))
}

fn status_body(state: &AppState) -> StatusBody {
    let lifecycle = state.lifecycle.lock().expect("lifecycle lock poisoned");
    StatusBody {
        status: "ok",
        run_id: state.config.run_id.clone(),
        topology_id: state.config.topology_id.clone(),
        model_id: state.config.model_id.clone(),
        stage_id: state.config.stage_id.clone(),
        stage_index: state.config.stage_index,
        layer_start: state.config.layer_start,
        layer_end: state.config.layer_end,
        topology_stage_count: state
            .topology
            .as_ref()
            .map(|topology| topology.stages.len()),
        runtime_loaded: state.runtime.is_some(),
        kv_mode: state.kv.as_ref().map(|kv| format!("{:?}", kv.mode())),
        ready: lifecycle.ready,
        started_at_unix_nanos: lifecycle.started_at_unix_nanos,
        received_messages: lifecycle.received_messages,
        peer_ready: lifecycle.peer_ready.values().cloned().collect(),
        telemetry: state.telemetry.stats(),
    }
}

async fn maybe_lookup_prefill(
    state: &AppState,
    runtime: Option<&Arc<Mutex<RuntimeState>>>,
    base: &MessageBase,
    token_start: u32,
    token_ids: &[i32],
) -> usize {
    let Some(kv) = state.kv.as_ref() else {
        return 0;
    };
    let Some(runtime) = runtime else {
        return 0;
    };
    if !kv.should_lookup() || token_ids.is_empty() {
        return 0;
    }
    let identities = kv.lookup_identities(&state.config, base, token_start as u64, token_ids);
    let mut attrs = kv_attrs(&state.config, kv);
    attrs.insert(attr::REQUEST_ID.to_string(), json!(base.request_id.clone()));
    attrs.insert(attr::SESSION_ID.to_string(), json!(base.session_id.clone()));
    attrs.insert(
        "skippy.kv.lookup_candidates".to_string(),
        json!(identities.len()),
    );
    attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
    let started = Instant::now();
    let identity_count = identities.len();
    let lookup = match kv
        .lookup_prefixes(
            identities
                .into_iter()
                .map(|candidate| candidate.identity)
                .collect(),
        )
        .await
    {
        Ok(lookup) => lookup,
        Err(error) => {
            let lookup_ms = started.elapsed().as_secs_f64() * 1000.0;
            attrs.insert("skippy.kv.lookup_ms".to_string(), json!(lookup_ms));
            attrs.insert("skippy.kv.decision".to_string(), json!("error"));
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            state.telemetry.emit("stage.kv_lookup_decision", attrs);
            return 0;
        }
    };
    if !lookup.errors.is_empty() && lookup.pages.is_empty() {
        let lookup_ms = started.elapsed().as_secs_f64() * 1000.0;
        attrs.insert("skippy.kv.lookup_ms".to_string(), json!(lookup_ms));
        attrs.insert("skippy.kv.decision".to_string(), json!("error"));
        attrs.insert(
            "skippy.kv.error".to_string(),
            json!(lookup.errors.join("; ")),
        );
        state.telemetry.emit("stage.kv_lookup_decision", attrs);
        return 0;
    }
    let lookup_ms = started.elapsed().as_secs_f64() * 1000.0;
    let hit_count = lookup.pages.len();
    attrs.insert("skippy.kv.lookup_ms".to_string(), json!(lookup_ms));
    attrs.insert("skippy.kv.lookup_hits".to_string(), json!(hit_count));
    attrs.insert(
        "skippy.kv.lookup_batches".to_string(),
        json!(u8::from(identity_count > 1)),
    );
    if let Some(page) = lookup
        .pages
        .into_iter()
        .max_by_key(|page| page.identity.as_ref().map(|identity| identity.token_count))
    {
        let restored_tokens = page
            .identity
            .as_ref()
            .map(|identity| identity.token_count as usize)
            .unwrap_or(0)
            .min(token_ids.len());
        attrs.insert(
            "skippy.kv.hit_page_id".to_string(),
            json!(page.page_id.clone()),
        );
        attrs.insert(
            "skippy.kv.restored_tokens".to_string(),
            json!(restored_tokens),
        );
        let already_loaded = {
            let runtime = runtime.lock().expect("runtime lock poisoned");
            runtime.has_session_range(&base.session_id, token_start as u64, restored_tokens as u64)
        };
        if already_loaded {
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("hit_already_loaded"),
            );
            state.telemetry.emit("stage.kv_lookup_decision", attrs);
            return restored_tokens;
        }
        let warmed = {
            let mut runtime = runtime.lock().expect("runtime lock poisoned");
            runtime.take_warm_kv_session(
                &base.session_id,
                &page.page_id,
                token_start as u64,
                restored_tokens as u64,
            )
        };
        if warmed {
            attrs.insert("skippy.kv.decision".to_string(), json!("hit_warm_session"));
            state.telemetry.emit("stage.kv_lookup_decision", attrs);
            return restored_tokens;
        }
        let attach_started = Instant::now();
        match kv.attach_page(&page.page_id).await {
            Ok(attached) => {
                attrs.insert(
                    "skippy.kv.attach_ms".to_string(),
                    json!(attach_started.elapsed().as_secs_f64() * 1000.0),
                );
                attrs.insert(
                    "skippy.kv.hit_bytes".to_string(),
                    json!(attached.bytes().len()),
                );
                let import_started = Instant::now();
                let import_result = {
                    let mut runtime = runtime.lock().expect("runtime lock poisoned");
                    runtime.import_kv_page(&base.session_id, &attached.manifest, attached.bytes())
                };
                attrs.insert(
                    "skippy.kv.runtime_import_ms".to_string(),
                    json!(import_started.elapsed().as_secs_f64() * 1000.0),
                );
                match import_result {
                    Ok(()) => {
                        attrs.insert("skippy.kv.decision".to_string(), json!("hit_imported"));
                        state.telemetry.emit("stage.kv_lookup_decision", attrs);
                        return restored_tokens;
                    }
                    Err(error) => {
                        attrs.insert("skippy.kv.decision".to_string(), json!("hit_import_error"));
                        attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                    }
                }
            }
            Err(error) => {
                attrs.insert(
                    "skippy.kv.attach_ms".to_string(),
                    json!(attach_started.elapsed().as_secs_f64() * 1000.0),
                );
                attrs.insert("skippy.kv.decision".to_string(), json!("hit_attach_error"));
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            }
        }
    } else {
        attrs.insert("skippy.kv.decision".to_string(), json!("miss"));
    }
    state.telemetry.emit("stage.kv_lookup_decision", attrs);
    0
}

fn maybe_plan_record_prefill(
    state: &AppState,
    base: &MessageBase,
    token_start: u32,
    token_ids: &[i32],
    min_record_tokens: u64,
) -> Vec<KvRecordCandidate> {
    let Some(kv) = state.kv.as_ref() else {
        return Vec::new();
    };
    if !kv.should_record() || token_ids.is_empty() {
        return Vec::new();
    }
    let identities = kv
        .record_identities(&state.config, base, token_start as u64, token_ids)
        .into_iter()
        .filter(|identity| identity.identity.token_count > min_record_tokens)
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(identities.len());
    for identity in identities {
        let token_count = identity.identity.token_count;
        let mut attrs = kv_attrs(&state.config, kv);
        attrs.insert(attr::REQUEST_ID.to_string(), json!(base.request_id.clone()));
        attrs.insert(attr::SESSION_ID.to_string(), json!(base.session_id.clone()));
        attrs.insert(
            "skippy.kv.page_id".to_string(),
            json!(identity.page_id.clone()),
        );
        attrs.insert("skippy.kv.token_count".to_string(), json!(token_count));
        if !kv.try_begin_record(&identity.page_id) {
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("record_skipped_inflight"),
            );
            state.telemetry.emit("stage.kv_record_commit", attrs);
            continue;
        }
        attrs.insert("skippy.kv.decision".to_string(), json!("record_queued"));
        records.push(KvRecordCandidate {
            attrs,
            session_id: base.session_id.clone(),
            page_id: identity.page_id,
            identity: identity.identity,
            token_start: token_start as u64,
            token_count,
        });
    }
    records
}

fn spawn_record_prefill(state: AppState, records: Vec<KvRecordCandidate>) {
    if records.is_empty() {
        return;
    }
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2));
        let Ok(tokio) = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
        else {
            return;
        };
        tokio.block_on(async move {
            maybe_commit_record_prefill(&state, records).await;
        });
    });
}

async fn maybe_commit_record_prefill(state: &AppState, records: Vec<KvRecordCandidate>) {
    if records.is_empty() {
        return;
    }
    let Some(kv) = state.kv.as_ref() else {
        return;
    };
    for record in records {
        let KvRecordCandidate {
            mut attrs,
            session_id,
            page_id,
            identity,
            token_start,
            token_count,
        } = record;
        let page_id_for_finish = page_id.clone();
        let export_started = Instant::now();
        let page =
            match export_kv_record_payload(state, &session_id, token_start, token_count).await {
                Ok(page) => page,
                Err(error) => {
                    let error = error.to_string();
                    let decision = if error.contains("not implemented") {
                        "unsupported"
                    } else if error.contains("busy") {
                        "record_skipped_runtime_busy"
                    } else {
                        "record_error"
                    };
                    attrs.insert(
                        "skippy.kv.runtime_export_ms".to_string(),
                        json!(export_started.elapsed().as_secs_f64() * 1000.0),
                    );
                    attrs.insert("skippy.kv.decision".to_string(), json!(decision));
                    attrs.insert("skippy.kv.error".to_string(), json!(error));
                    state.telemetry.emit("stage.kv_runtime_export", attrs);
                    kv.finish_record(&page_id_for_finish);
                    continue;
                }
            };
        let desc = page.desc;
        let payload = page.payload;
        attrs.insert(
            "skippy.kv.runtime_export_ms".to_string(),
            json!(export_started.elapsed().as_secs_f64() * 1000.0),
        );
        attrs.insert(
            "skippy.kv.export_bytes".to_string(),
            json!(desc.payload_bytes),
        );
        attrs.insert(
            "skippy.kv.export_layers".to_string(),
            json!(desc.layer_count),
        );
        let byte_size = payload.len();
        let annotations = kv_desc_annotations(&desc);
        let started = Instant::now();
        match kv
            .record_page_into(
                page_id.clone(),
                identity,
                byte_size,
                annotations,
                |output| {
                    output.copy_from_slice(&payload);
                    Ok(())
                },
            )
            .await
        {
            Ok(outcome) => {
                attrs.insert(
                    "skippy.kv.record_ms".to_string(),
                    json!(started.elapsed().as_secs_f64() * 1000.0),
                );
                attrs.insert(
                    "skippy.kv.record_write_ms".to_string(),
                    json!(outcome.write_ms),
                );
                attrs.insert(
                    "skippy.kv.checksum_ms".to_string(),
                    json!(outcome.checksum_ms),
                );
                attrs.insert(
                    "skippy.kv.recorded_bytes".to_string(),
                    json!(outcome.manifest.byte_size),
                );
                attrs.insert("skippy.kv.decision".to_string(), json!("committed"));
                state.telemetry.emit("stage.kv_record_commit", attrs);
                maybe_warm_kv_session(state, page_id, &outcome.manifest, &payload).await;
            }
            Err(error) => {
                attrs.insert(
                    "skippy.kv.record_ms".to_string(),
                    json!(started.elapsed().as_secs_f64() * 1000.0),
                );
                attrs.insert("skippy.kv.decision".to_string(), json!("record_error"));
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                state.telemetry.emit("stage.kv_record_commit", attrs);
            }
        }
        kv.finish_record(&page_id_for_finish);
    }
}

async fn export_kv_record_payload(
    state: &AppState,
    session_id: &str,
    token_start: u64,
    token_count: u64,
) -> Result<skippy_runtime::RuntimeKvPage> {
    let runtime = state
        .runtime
        .as_ref()
        .ok_or_else(|| anyhow!("runtime unavailable for KV export"))?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?;
    runtime.export_kv_page(session_id, token_start, token_count)
}

async fn maybe_warm_kv_session(
    state: &AppState,
    page_id: String,
    manifest: &crate::kv_proto::KvPageManifest,
    payload: &[u8],
) {
    let Some(runtime) = state.runtime.as_ref() else {
        return;
    };
    let mut attrs = lifecycle_attrs(&state.config);
    attrs.insert("skippy.kv.page_id".to_string(), json!(page_id.clone()));
    let started = Instant::now();
    for _ in 0..2 {
        match runtime.try_lock() {
            Ok(mut runtime) => {
                match runtime.warm_kv_session(page_id.clone(), manifest, payload) {
                    Ok(()) => {
                        attrs.insert(
                            "skippy.kv.warm_ms".to_string(),
                            json!(started.elapsed().as_secs_f64() * 1000.0),
                        );
                        attrs.insert("skippy.kv.decision".to_string(), json!("warmed"));
                        state.telemetry.emit("stage.kv_warm_session", attrs);
                    }
                    Err(error) => {
                        attrs.insert(
                            "skippy.kv.warm_ms".to_string(),
                            json!(started.elapsed().as_secs_f64() * 1000.0),
                        );
                        attrs.insert("skippy.kv.decision".to_string(), json!("warm_error"));
                        attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                        state.telemetry.emit("stage.kv_warm_session", attrs);
                    }
                }
                return;
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                attrs.insert("skippy.kv.decision".to_string(), json!("warm_error"));
                attrs.insert(
                    "skippy.kv.error".to_string(),
                    json!("runtime lock poisoned"),
                );
                state.telemetry.emit("stage.kv_warm_session", attrs);
                return;
            }
        }
    }
    attrs.insert(
        "skippy.kv.warm_ms".to_string(),
        json!(started.elapsed().as_secs_f64() * 1000.0),
    );
    attrs.insert(
        "skippy.kv.decision".to_string(),
        json!("warm_skipped_runtime_busy"),
    );
    state.telemetry.emit("stage.kv_warm_session", attrs);
}

async fn maybe_drop_kv_session(state: &AppState, session_id: &str) {
    let Some(kv) = state.kv.as_ref() else {
        return;
    };
    let mut attrs = kv_attrs(&state.config, kv);
    attrs.insert(attr::SESSION_ID.to_string(), json!(session_id));
    match kv.drop_session(session_id).await {
        Ok(dropped) => {
            attrs.insert("skippy.kv.dropped_pages".to_string(), json!(dropped));
            state.telemetry.emit("stage.kv_drop_session", attrs);
        }
        Err(error) => {
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            state.telemetry.emit("stage.kv_drop_session_failed", attrs);
        }
    }
}

fn kv_attrs(config: &StageConfig, kv: &KvStageIntegration) -> BTreeMap<String, Value> {
    let mut attrs = lifecycle_attrs(config);
    for (key, value) in kv.attrs() {
        attrs.insert(key.to_string(), value);
    }
    attrs
}

fn text_message_base(config: &StageConfig, request: &TextRequest) -> MessageBase {
    MessageBase {
        schema_version: SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        request_id: request.request_id.clone(),
        session_id: request.session_id.clone(),
        stage_id: config.stage_id.clone(),
        stage_index: config.stage_index,
        topology_id: config.topology_id.clone(),
        model_id: Some(config.model_id.clone()),
        tokenizer_id: None,
        chat_template_id: None,
        seq: None,
    }
}

fn local_reply_base(config: &StageConfig, incoming: &MessageBase) -> MessageBase {
    MessageBase {
        schema_version: SCHEMA_VERSION,
        run_id: incoming.run_id.clone(),
        request_id: incoming.request_id.clone(),
        session_id: incoming.session_id.clone(),
        stage_id: config.stage_id.clone(),
        stage_index: config.stage_index,
        topology_id: config.topology_id.clone(),
        model_id: Some(config.model_id.clone()),
        tokenizer_id: incoming.tokenizer_id.clone(),
        chat_template_id: incoming.chat_template_id.clone(),
        seq: incoming.seq,
    }
}

fn validate_message(config: &StageConfig, message: &StageMessage) -> Result<()> {
    let base = message.base();
    if base.schema_version != SCHEMA_VERSION {
        bail!(
            "unsupported schema_version {}, expected {}",
            base.schema_version,
            SCHEMA_VERSION
        );
    }
    if base.run_id != config.run_id {
        bail!("message run_id does not match server run_id");
    }
    if base.topology_id != config.topology_id {
        bail!("message topology_id does not match server topology_id");
    }
    if message_has_activation_payload(message) {
        bail!("activation frame payload handling is not implemented yet");
    }
    Ok(())
}

fn message_has_activation_payload(message: &StageMessage) -> bool {
    match message {
        StageMessage::PrefillChunk(message) => {
            message.activation_ref.is_some()
                || message
                    .activation
                    .as_ref()
                    .is_some_and(|activation| activation.payload_bytes > 0)
        }
        StageMessage::FinalPrefillChunk(message) => {
            message.activation_ref.is_some()
                || message
                    .activation
                    .as_ref()
                    .is_some_and(|activation| activation.payload_bytes > 0)
        }
        StageMessage::DecodeToken(message) => {
            message.activation_ref.is_some()
                || message
                    .activation
                    .as_ref()
                    .is_some_and(|activation| activation.payload_bytes > 0)
        }
        _ => false,
    }
}
