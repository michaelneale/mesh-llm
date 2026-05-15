use std::{
    collections::BTreeMap,
    future::Future,
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::Response,
    Router,
};
use base64::Engine;
use futures_util::{stream, StreamExt};
use openai_frontend::{
    chat_mesh_hooks_enabled, inject_text_into_chat_messages, normalize_reasoning_template_options,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream,
    ChatHookAction, ChatHookOutcome, CompletionChunk, CompletionRequest, CompletionResponse,
    CompletionStream, FinishReason, GenerationHookSignals, MessageContent, MessageContentPart,
    ModelId, ModelObject, OpenAiBackend, OpenAiError, OpenAiErrorKind, OpenAiHookPolicy,
    OpenAiRequestContext, OpenAiResult, PrefillHookSignals, Usage,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use skippy_metrics::attr as attr_key;
use skippy_protocol::binary::{
    recv_ready, recv_reply, state_flags, write_stage_message, StageLogitBias as WireLogitBias,
    StageReply, StageReplyStats, StageSamplingConfig as WireSamplingConfig, StageStateHeader,
    StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind, LLAMA_TOKEN_NULL,
    MAX_STAGE_LOGIT_BIAS,
};
use skippy_protocol::{MessageBase, StageConfig, StageTopology, SCHEMA_VERSION};
use skippy_runtime::{
    ActivationFrame, ChatTemplateJsonOptions, ChatTemplateOptions,
    FlashAttentionType as RuntimeFlashAttentionType, GenerationSignalWindow,
    LogitBias as RuntimeLogitBias, MediaInput, ModelInfo, RuntimeConfig, RuntimeLoadMode,
    SamplingConfig, StageModel, StageSession, TokenSignal, MAX_LOGIT_BIAS,
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, OwnedSemaphorePermit, Semaphore, TryAcquireError},
    task,
};

use crate::{
    binary_transport::{
        connect_binary_downstream, forwarded_stage_message, forwarded_stage_message_timed,
        run_binary_stage_message, write_stage_message_conditioned, WireCondition,
    },
    cli::ServeOpenAiArgs,
    config::{load_json, validate_config},
    kv_integration::KvStageIntegration,
    runtime_state::{load_runtime, RuntimeSessionStats, RuntimeState},
    telemetry::{lifecycle_attrs, now_unix_nanos, Telemetry},
};

mod backend;
mod binary_chain_generation;
mod embedded_execution;
mod embedded_generation;
mod generation_flow;
mod local_generation;
mod prefill;
mod prefix_cache;
mod prompting;
mod request;
mod speculative;
mod util;
mod wire_messages;

use self::{prefill::*, request::*, speculative::*, util::*, wire_messages::*};

static OPENAI_GENERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub const CONTEXT_BUDGET_MAX_TOKENS: u32 = u32::MAX;
const GENERATION_ADMISSION_TIMEOUT: Duration = Duration::from_secs(10);
const GENERATION_RETRY_AFTER_SECS: u64 = 1;

pub async fn serve_openai(args: ServeOpenAiArgs) -> Result<()> {
    let config = load_json::<StageConfig>(&args.config)
        .with_context(|| format!("load stage config {}", args.config.display()))?;
    let topology = match args.topology.as_ref() {
        Some(path) => Some(
            load_json::<StageTopology>(path)
                .with_context(|| format!("load topology {}", path.display()))?,
        ),
        None => None,
    };
    validate_config(&config, topology.as_ref())?;
    if args.first_stage_addr.is_none() && config.downstream.is_some() {
        bail!("serve-openai local backend requires a final/single-stage config with no downstream");
    }
    if args.prefill_chunk_size == 0 {
        bail!("--prefill-chunk-size must be greater than zero");
    }
    if args.generation_concurrency == 0 {
        bail!("--generation-concurrency must be greater than zero");
    }

    let runtime = load_runtime(&config)?.ok_or_else(|| {
        anyhow!("serve-openai requires a stage config with model_path for tokenization and decode")
    })?;
    let model_id = ModelId::new(args.model_id.unwrap_or_else(|| config.model_id.clone()))
        .map_err(|error| anyhow!("invalid OpenAI model id: {error}"))?
        .into_string();
    let mode = match args.first_stage_addr {
        Some(first_stage_addr) => OpenAiBackendMode::BinaryChain {
            first_stage_addr,
            wire_dtype: parse_wire_dtype(&args.activation_wire_dtype)?,
            prefill_chunk_policy: PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
                policy: &args.prefill_chunk_policy,
                schedule: args.prefill_chunk_schedule.as_deref(),
                fixed_chunk_size: args.prefill_chunk_size,
                adaptive_start: args.prefill_adaptive_start,
                adaptive_step: args.prefill_adaptive_step,
                adaptive_max: args.prefill_adaptive_max,
                schedule_arg: "--prefill-chunk-schedule",
                policy_arg: "--prefill-chunk-policy",
            })?,
            startup_timeout_secs: args.startup_timeout_secs,
        },
        None => OpenAiBackendMode::LocalRuntime,
    };
    let mode_label = mode.label();
    let telemetry = Telemetry::new(
        args.metrics_otlp_grpc,
        args.telemetry_queue_capacity,
        config.clone(),
        args.telemetry_level,
    );
    telemetry.emit("stage.openai_server_start", lifecycle_attrs(&config));
    if matches!(&mode, OpenAiBackendMode::LocalRuntime) {
        ensure_generation_concurrency_fits_lanes(
            args.generation_concurrency,
            config.lane_count,
            "--generation-concurrency",
        )?;
        prewarm_generation_sessions(
            &runtime,
            args.generation_concurrency,
            &telemetry,
            &config,
            "stage.openai_runtime_prewarm",
        )
        .context("prewarm OpenAI runtime sessions")?;
    }
    let kv = KvStageIntegration::from_config(&config)?.map(Arc::new);
    let ctx_size = usize::try_from(config.ctx_size).unwrap_or(usize::MAX);
    let backend = Arc::new(StageOpenAiBackend {
        runtime,
        config,
        telemetry: telemetry.clone(),
        model_id: model_id.clone(),
        default_max_tokens: args.default_max_tokens,
        ctx_size,
        mode,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        generation_limit: Arc::new(Semaphore::new(args.generation_concurrency)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: args.generation_concurrency,
        hook_policy: None,
        kv,
    });
    let app: Router = instrumented_openai_router(backend, telemetry.clone());

    println!(
        "skippy-server listening: openai={} model_id={} backend={} generation_concurrency={}",
        args.bind_addr, model_id, mode_label, args.generation_concurrency,
    );

    let listener = TcpListener::bind(args.bind_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Clone)]
pub struct EmbeddedOpenAiArgs {
    pub bind_addr: SocketAddr,
    pub config: StageConfig,
    pub runtime: Arc<Mutex<RuntimeState>>,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub generation_concurrency: usize,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: usize,
    pub adaptive_speculative_window: bool,
    pub draft_n_gpu_layers: Option<i32>,
    pub activation_width: i32,
    pub wire_dtype: WireActivationDType,
    pub reply_credit_limit: Option<usize>,
    pub downstream_connect_timeout_secs: u64,
    pub downstream_wire_condition: WireCondition,
    pub telemetry: Telemetry,
    pub hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
}

pub async fn serve_embedded_openai(args: EmbeddedOpenAiArgs) -> Result<()> {
    serve_embedded_openai_with_shutdown(args, std::future::pending::<()>()).await
}

pub async fn serve_embedded_openai_with_shutdown(
    args: EmbeddedOpenAiArgs,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let bind_addr = args.bind_addr;
    let binding = embedded_openai_router(args)?;

    println!(
        "skippy-server listening: openai={} model_id={} backend=embedded-stage0 generation_concurrency={}",
        bind_addr, binding.model_id, binding.generation_concurrency,
    );

    let listener = TcpListener::bind(bind_addr).await?;
    axum::serve(listener, binding.router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

pub struct EmbeddedOpenAiRouter {
    pub router: Router,
    pub model_id: String,
    pub generation_concurrency: usize,
}

#[derive(Clone)]
pub struct EmbeddedOpenAiBackend {
    pub backend: Arc<dyn OpenAiBackend>,
    pub model_id: String,
    pub generation_concurrency: usize,
}

pub fn embedded_openai_router(args: EmbeddedOpenAiArgs) -> Result<EmbeddedOpenAiRouter> {
    let telemetry = args.telemetry.clone();
    let binding = embedded_openai_backend(args)?;
    let router = instrumented_openai_router(binding.backend.clone(), telemetry);

    Ok(EmbeddedOpenAiRouter {
        router,
        model_id: binding.model_id,
        generation_concurrency: binding.generation_concurrency,
    })
}

pub fn embedded_openai_backend(args: EmbeddedOpenAiArgs) -> Result<EmbeddedOpenAiBackend> {
    if args.prefill_chunk_size == 0 {
        bail!("--openai-prefill-chunk-size must be greater than zero");
    }
    if args.generation_concurrency == 0 {
        bail!("--openai-generation-concurrency must be greater than zero");
    }
    ensure_generation_concurrency_fits_lanes(
        args.generation_concurrency,
        args.config.lane_count,
        "--openai-generation-concurrency",
    )?;
    if args.draft_model_path.is_some() && args.speculative_window == 0 {
        bail!("--openai-speculative-window must be greater than zero when a draft model is set");
    }
    if args.config.stage_index != 0 || args.config.layer_start != 0 {
        bail!("embedded OpenAI serving is only supported on stage 0");
    }
    let draft = open_draft_runner(
        args.draft_model_path.as_deref(),
        &args.config,
        args.draft_n_gpu_layers,
        args.speculative_window,
    )?;
    let model_id = ModelId::new(
        args.model_id
            .unwrap_or_else(|| args.config.model_id.clone()),
    )
    .map_err(|error| anyhow!("invalid OpenAI model id: {error}"))?
    .into_string();
    let lane_pool = PersistentStageLanePool::new(
        &args.config,
        args.generation_concurrency,
        args.downstream_connect_timeout_secs,
        args.telemetry.clone(),
    )
    .context("create embedded OpenAI persistent downstream lanes")?;
    let prefill_reply_credit_limit = args.reply_credit_limit.unwrap_or(3);
    let mode = OpenAiBackendMode::EmbeddedStageZero {
        config: args.config.clone(),
        wire_dtype: args.wire_dtype,
        prefill_chunk_policy: PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
            policy: &args.prefill_chunk_policy,
            schedule: args.prefill_chunk_schedule.as_deref(),
            fixed_chunk_size: args.prefill_chunk_size,
            adaptive_start: args.prefill_adaptive_start,
            adaptive_step: args.prefill_adaptive_step,
            adaptive_max: args.prefill_adaptive_max,
            schedule_arg: "--openai-prefill-chunk-schedule",
            policy_arg: "--openai-prefill-chunk-policy",
        })?,
        activation_width: args.activation_width,
        downstream_wire_condition: args.downstream_wire_condition,
        prefill_reply_credit_limit,
        lane_pool,
    };
    args.telemetry
        .emit("stage.openai_server_start", lifecycle_attrs(&args.config));
    prewarm_generation_sessions(
        &args.runtime,
        args.generation_concurrency,
        &args.telemetry,
        &args.config,
        "stage.openai_runtime_prewarm",
    )
    .context("prewarm embedded OpenAI runtime sessions")?;
    let kv = KvStageIntegration::from_config(&args.config)?.map(Arc::new);
    let ctx_size = usize::try_from(args.config.ctx_size).unwrap_or(usize::MAX);
    let backend = Arc::new(StageOpenAiBackend {
        runtime: args.runtime,
        config: args.config.clone(),
        telemetry: args.telemetry.clone(),
        model_id: model_id.clone(),
        default_max_tokens: args.default_max_tokens,
        ctx_size,
        mode,
        draft,
        speculative_window: args.speculative_window,
        adaptive_speculative_window: args.adaptive_speculative_window,
        generation_limit: Arc::new(Semaphore::new(args.generation_concurrency)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: args.generation_concurrency,
        hook_policy: args.hook_policy,
        kv,
    });

    Ok(EmbeddedOpenAiBackend {
        backend,
        model_id,
        generation_concurrency: args.generation_concurrency,
    })
}

#[derive(Clone)]
struct StageOpenAiBackend {
    runtime: Arc<Mutex<RuntimeState>>,
    config: StageConfig,
    telemetry: Telemetry,
    model_id: String,
    default_max_tokens: u32,
    ctx_size: usize,
    mode: OpenAiBackendMode,
    draft: Option<Arc<Mutex<DraftRunner>>>,
    speculative_window: usize,
    adaptive_speculative_window: bool,
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    kv: Option<Arc<KvStageIntegration>>,
}

struct GenerationQueueReservation {
    depth: Arc<AtomicUsize>,
}

impl Drop for GenerationQueueReservation {
    fn drop(&mut self) {
        self.depth.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn acquire_generation_permit_with_queue(
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    admission_timeout: Duration,
) -> OpenAiResult<OwnedSemaphorePermit> {
    match generation_limit.clone().try_acquire_owned() {
        Ok(permit) => return Ok(permit),
        Err(TryAcquireError::Closed) => return Err(generation_lanes_busy_error()),
        Err(TryAcquireError::NoPermits) => {}
    }

    let _queue_reservation =
        reserve_generation_queue(generation_queue_depth, generation_queue_limit)
            .ok_or_else(generation_queue_full_error)?;
    tokio::time::timeout(admission_timeout, generation_limit.acquire_owned())
        .await
        .map_err(|_| generation_queue_timeout_error(admission_timeout))?
        .map_err(|_| generation_lanes_busy_error())
}

fn reserve_generation_queue(
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
) -> Option<GenerationQueueReservation> {
    let mut current = generation_queue_depth.load(Ordering::Acquire);
    loop {
        if current >= generation_queue_limit {
            return None;
        }
        match generation_queue_depth.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                return Some(GenerationQueueReservation {
                    depth: generation_queue_depth,
                });
            }
            Err(next) => current = next,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerationTokenLimit {
    Explicit(u32),
    ContextBudget,
}

impl GenerationTokenLimit {
    fn from_request(requested: Option<u32>, default_max_tokens: u32) -> Self {
        match requested {
            Some(max_tokens) => Self::Explicit(max_tokens),
            None if default_max_tokens == CONTEXT_BUDGET_MAX_TOKENS => Self::ContextBudget,
            None => Self::Explicit(default_max_tokens),
        }
    }

    fn resolve(self, prompt_token_count: usize, ctx_size: usize) -> OpenAiResult<u32> {
        match self {
            Self::Explicit(max_tokens) => {
                ensure_context_capacity(prompt_token_count, max_tokens, ctx_size)?;
                Ok(max_tokens)
            }
            Self::ContextBudget => context_budget_completion_tokens(prompt_token_count, ctx_size),
        }
    }
}

fn instrumented_openai_router(backend: Arc<dyn OpenAiBackend>, telemetry: Telemetry) -> Router {
    openai_frontend::router_for(backend).layer(middleware::from_fn_with_state(
        telemetry,
        openai_http_telemetry,
    ))
}

fn prewarm_generation_sessions(
    runtime: &Arc<Mutex<RuntimeState>>,
    generation_concurrency: usize,
    telemetry: &Telemetry,
    config: &StageConfig,
    event_name: &'static str,
) -> Result<()> {
    let timer = PhaseTimer::start();
    let sessions = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?
        .prewarm_idle_sessions(generation_concurrency)?;
    let mut attrs = lifecycle_attrs(config);
    attrs.insert(
        "llama_stage.generation_concurrency".to_string(),
        json!(generation_concurrency),
    );
    attrs.insert(
        "llama_stage.lane_count".to_string(),
        json!(sessions.lane_count),
    );
    attrs.insert(
        "llama_stage.runtime_sessions_active".to_string(),
        json!(sessions.active_sessions),
    );
    attrs.insert(
        "llama_stage.runtime_sessions_idle".to_string(),
        json!(sessions.idle_sessions),
    );
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(timer.elapsed_ms()),
    );
    telemetry.emit_span(
        event_name,
        attrs,
        timer.start_unix_nanos,
        now_unix_nanos() as u64,
    );
    Ok(())
}

fn ensure_generation_concurrency_fits_lanes(
    generation_concurrency: usize,
    lane_count: u32,
    flag_name: &str,
) -> Result<()> {
    let lane_count = usize::try_from(lane_count).unwrap_or(usize::MAX);
    if generation_concurrency > lane_count {
        bail!(
            "{flag_name} ({generation_concurrency}) cannot exceed configured lane_count ({lane_count})"
        );
    }
    Ok(())
}

fn generation_lanes_busy_error() -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        "all execution lanes are busy",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

fn generation_queue_full_error() -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        "generation queue is full; retry later",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

fn generation_queue_timeout_error(timeout: Duration) -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        format!(
            "timed out waiting for an execution lane after {} seconds",
            timeout.as_secs()
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

async fn openai_http_telemetry(
    State(telemetry): State<Telemetry>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let timer = PhaseTimer::start();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    let status = response.status().as_u16();
    let mut attrs = BTreeMap::from([
        ("llama_stage.http_method".to_string(), json!(method)),
        ("llama_stage.http_path".to_string(), json!(path)),
        ("llama_stage.http_status".to_string(), json!(status)),
    ]);
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(timer.elapsed_ms()),
    );
    telemetry.emit_span(
        "stage.openai_http_request",
        attrs,
        timer.start_unix_nanos,
        now_unix_nanos() as u64,
    );
    response
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum OpenAiBackendMode {
    LocalRuntime,
    BinaryChain {
        first_stage_addr: String,
        wire_dtype: WireActivationDType,
        prefill_chunk_policy: PrefillChunkPolicy,
        startup_timeout_secs: u64,
    },
    EmbeddedStageZero {
        config: StageConfig,
        wire_dtype: WireActivationDType,
        prefill_chunk_policy: PrefillChunkPolicy,
        activation_width: i32,
        downstream_wire_condition: WireCondition,
        prefill_reply_credit_limit: usize,
        lane_pool: Option<Arc<PersistentStageLanePool>>,
    },
}

struct PersistentStageLanePool {
    config: StageConfig,
    timeout_secs: u64,
    telemetry: Telemetry,
    lanes: Mutex<Vec<PersistentStageLane>>,
    next_lane_id: AtomicU64,
    capacity: usize,
}

struct PersistentStageLane {
    id: u64,
    stream: TcpStream,
}

impl PersistentStageLanePool {
    fn new(
        config: &StageConfig,
        capacity: usize,
        timeout_secs: u64,
        telemetry: Telemetry,
    ) -> Result<Option<Arc<Self>>> {
        if config.downstream.is_none() {
            return Ok(None);
        }
        let pool = Arc::new(Self {
            config: config.clone(),
            timeout_secs,
            telemetry,
            lanes: Mutex::new(Vec::with_capacity(capacity)),
            next_lane_id: AtomicU64::new(0),
            capacity,
        });
        let timer = PhaseTimer::start();
        for _ in 0..capacity {
            let lane = pool.connect_lane()?;
            pool.return_lane(lane);
        }
        let mut attrs = lifecycle_attrs(config);
        attrs.insert(
            "llama_stage.openai_downstream_pool_capacity".to_string(),
            json!(capacity),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        pool.telemetry.emit_span(
            "stage.openai_downstream_pool_ready",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(Some(pool))
    }

    fn checkout(&self, ids: &OpenAiGenerationIds) -> OpenAiResult<PersistentStageLane> {
        let timer = PhaseTimer::start();
        let lane = {
            let mut lanes = self
                .lanes
                .lock()
                .map_err(|_| OpenAiError::backend("persistent lane pool lock poisoned"))?;
            lanes.pop()
        };
        let lane = match lane {
            Some(lane) => lane,
            None => self.connect_lane().map_err(openai_backend_error)?,
        };
        let mut attrs = BTreeMap::from([
            (
                "llama_stage.openai_downstream_persistent".to_string(),
                json!(true),
            ),
            (
                "llama_stage.openai_downstream_lane_id".to_string(),
                json!(lane.id),
            ),
            (
                "llama_stage.openai_downstream_pool_capacity".to_string(),
                json!(self.capacity),
            ),
            (
                "llama_stage.request_id".to_string(),
                json!(ids.request_id_string()),
            ),
            (
                "llama_stage.session_id".to_string(),
                json!(ids.session_id_string()),
            ),
        ]);
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        self.telemetry.emit_span(
            "stage.openai_downstream_connect",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(lane)
    }

    fn return_lane(&self, lane: PersistentStageLane) {
        match self.lanes.lock() {
            Ok(mut lanes) => lanes.push(lane),
            Err(_) => {
                let mut attrs = lifecycle_attrs(&self.config);
                attrs.insert(
                    "llama_stage.error".to_string(),
                    json!("persistent lane pool lock poisoned"),
                );
                self.telemetry
                    .emit("stage.openai_downstream_lane_return_failed", attrs);
            }
        }
    }

    fn replace_lane(&self, retired_lane_id: u64) {
        let timer = PhaseTimer::start();
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.openai_downstream_retired_lane_id".to_string(),
            json!(retired_lane_id),
        );
        match self.connect_lane() {
            Ok(lane) => {
                attrs.insert(
                    "llama_stage.openai_downstream_lane_id".to_string(),
                    json!(lane.id),
                );
                attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(timer.elapsed_ms()),
                );
                self.return_lane(lane);
                self.telemetry.emit_span(
                    "stage.openai_downstream_lane_replaced",
                    attrs,
                    timer.start_unix_nanos,
                    now_unix_nanos() as u64,
                );
            }
            Err(error) => {
                attrs.insert("llama_stage.error".to_string(), json!(error.to_string()));
                attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(timer.elapsed_ms()),
                );
                self.telemetry.emit_span(
                    "stage.openai_downstream_lane_replace_failed",
                    attrs,
                    timer.start_unix_nanos,
                    now_unix_nanos() as u64,
                );
            }
        }
    }

    fn connect_lane(&self) -> Result<PersistentStageLane> {
        let lane_id = self.next_lane_id.fetch_add(1, Ordering::Relaxed);
        let timer = PhaseTimer::start();
        let mut stream = connect_binary_downstream(&self.config, self.timeout_secs)?
            .ok_or_else(|| anyhow!("embedded stage0 has no downstream"))?;
        recv_ready(&mut stream).context("persistent downstream lane did not become ready")?;
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.openai_downstream_lane_id".to_string(),
            json!(lane_id),
        );
        attrs.insert(
            "llama_stage.openai_downstream_pool_capacity".to_string(),
            json!(self.capacity),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        self.telemetry.emit_span(
            "stage.openai_downstream_persistent_connect",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(PersistentStageLane {
            id: lane_id,
            stream,
        })
    }
}

#[derive(Clone)]
struct OpenAiGenerationIds {
    session_label: String,
    session_id: u64,
    request_id: u64,
    cache: OpenAiCacheHints,
}

impl OpenAiGenerationIds {
    fn new(cache: OpenAiCacheHints) -> Self {
        let sequence = OPENAI_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let session_label = format!("openai-session-{}-{sequence}", now_unix_millis());
        Self {
            session_id: stable_wire_id(&[session_label.as_bytes()]),
            request_id: stable_wire_id(&[session_label.as_bytes(), b"request"]),
            session_label,
            cache,
        }
    }

    fn session_id_string(&self) -> String {
        self.session_id.to_string()
    }

    fn request_id_string(&self) -> String {
        self.request_id.to_string()
    }
}

#[derive(Clone, Default)]
struct OpenAiCacheHints {
    prompt_cache_key: Option<String>,
    prompt_cache_retention: Option<String>,
}

impl OpenAiCacheHints {
    fn from_chat_request(request: &ChatCompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    fn from_completion_request(request: &CompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    fn namespace(&self) -> Option<String> {
        self.prompt_cache_key
            .as_ref()
            .map(|key| format!("openai:prompt_cache_key:{key}"))
    }
}

fn prompt_cache_retention_label(retention: openai_frontend::PromptCacheRetention) -> &'static str {
    match retention {
        openai_frontend::PromptCacheRetention::InMemory => "in_memory",
        openai_frontend::PromptCacheRetention::TwentyFourHours => "24h",
    }
}

#[derive(Clone, Copy, Default)]
struct GenerationCacheStats {
    cached_prompt_tokens: u32,
    matched_prefix_tokens: u32,
    suffix_prefill_tokens: u32,
    hit_kind: Option<&'static str>,
}

struct ChainPrefixRestore {
    restored_tokens: usize,
    stats: StageReplyStats,
}

struct PhaseTimer {
    start_unix_nanos: u64,
    start_instant: Instant,
}

impl PhaseTimer {
    fn start() -> Self {
        Self {
            start_unix_nanos: now_unix_nanos() as u64,
            start_instant: Instant::now(),
        }
    }

    fn elapsed_ms(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64() * 1000.0
    }
}

fn decode_token_phase(decode_step: u32) -> &'static str {
    match decode_step {
        0 => "cold",
        1..=7 => "warmup",
        _ => "steady",
    }
}

#[derive(Default)]
struct GenerationMetrics {
    detokenize_ms: f64,
    text_emit_ms: f64,
    eog_check_ms: f64,
}

struct DraftRunner {
    path: PathBuf,
    window: usize,
    _model: StageModel,
    session: StageSession,
}

impl DraftRunner {
    fn open(
        path: &Path,
        config: &StageConfig,
        n_gpu_layers: Option<i32>,
        window: usize,
    ) -> Result<Self> {
        if !path.is_file() {
            bail!("draft model does not exist: {}", path.display());
        }
        let layer_count = model_layer_count(path)?;
        let model = StageModel::open(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size: config.ctx_size,
                lane_count: 1,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
                selected_backend_device: config
                    .selected_device
                    .as_ref()
                    .map(|device| device.backend_device.clone()),
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: RuntimeFlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                filter_tensors_on_load: false,
            },
        )
        .with_context(|| format!("open draft model {}", path.display()))?;
        let session = model.create_session().context("create draft session")?;
        Ok(Self {
            path: path.to_path_buf(),
            window,
            _model: model,
            session,
        })
    }

    fn reset_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        self.session.reset().context("reset draft session")?;
        if context_tokens.len() > 1 {
            self.session
                .prefill_chunk(&context_tokens[..context_tokens.len() - 1])
                .context("prefill draft context")?;
        }
        Ok(())
    }

    fn propose(&mut self, mut current: i32, max_tokens: usize) -> Result<Vec<i32>> {
        let mut tokens = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            current = self
                .session
                .decode_step(current)
                .context("draft decode step")?;
            tokens.push(current);
        }
        Ok(tokens)
    }
}

fn open_draft_runner(
    path: Option<&Path>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
    window: usize,
) -> Result<Option<Arc<Mutex<DraftRunner>>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    Ok(Some(Arc::new(Mutex::new(DraftRunner::open(
        path,
        config,
        n_gpu_layers,
        window,
    )?))))
}

fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let layer_count = info
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .ok_or_else(|| anyhow!("could not infer layer count for {}", path.display()))?;
    Ok(layer_count)
}

impl OpenAiBackendMode {
    fn label(&self) -> &'static str {
        match self {
            Self::LocalRuntime => "local-runtime",
            Self::BinaryChain { .. } => "binary-chain",
            Self::EmbeddedStageZero { .. } => "embedded-stage0",
        }
    }
}

fn ensure_requested_model(advertised_model_id: &str, requested: &str) -> OpenAiResult<()> {
    if requested == advertised_model_id
        || strip_default_revision(requested) == strip_default_revision(advertised_model_id)
    {
        Ok(())
    } else {
        Err(OpenAiError::model_not_found(requested))
    }
}

/// Strip `@main` so `org/repo@main:Q4` and `org/repo:Q4` compare equal.
///
/// Only removes `@main` when it sits at a revision boundary — followed by
/// `:` (quant separator) or end-of-string.  This avoids corrupting repo
/// names that happen to contain `@main` as a prefix of a longer segment
/// (e.g. `@mainland`).
fn strip_default_revision(id: &str) -> String {
    if let Some(pos) = id.find("@main") {
        let after = pos + "@main".len();
        if after == id.len() || id.as_bytes()[after] == b':' {
            let mut s = id[..pos].to_string();
            s.push_str(&id[after..]);
            return s;
        }
    }
    id.to_string()
}

fn apply_chat_hook_outcome(request: &mut ChatCompletionRequest, outcome: &ChatHookOutcome) {
    for action in &outcome.actions {
        match action {
            ChatHookAction::InjectText { text } => {
                inject_text_into_chat_messages(&mut request.messages, text.clone());
            }
            ChatHookAction::None => {}
        }
    }
}

fn hook_injected_text(outcome: &ChatHookOutcome) -> Option<String> {
    let text = outcome
        .actions
        .iter()
        .filter_map(|action| match action {
            ChatHookAction::InjectText { text } if !text.is_empty() => Some(text.as_str()),
            ChatHookAction::InjectText { .. } | ChatHookAction::None => None,
        })
        .collect::<Vec<_>>()
        .join("");
    (!text.is_empty()).then_some(text)
}

fn mid_generation_window_should_fire(
    decoded_tokens: usize,
    last_hook_at: &Option<usize>,
    window: &GenerationSignalWindow,
) -> bool {
    const MIN_DECODED_TOKENS: usize = 12;
    const COOLDOWN_TOKENS: usize = 32;
    const REPETITION_TRIGGER_COUNT: u32 = 3;

    if decoded_tokens < MIN_DECODED_TOKENS || window.token_count == 0 {
        return false;
    }
    if last_hook_at.is_some_and(|last| decoded_tokens.saturating_sub(last) < COOLDOWN_TOKENS) {
        return false;
    }
    let sustained_entropy =
        window.high_entropy_count.saturating_mul(4) >= window.token_count.saturating_mul(3);
    sustained_entropy || window.repetition_count >= REPETITION_TRIGGER_COUNT
}

fn attrs_insert_prefill_chunk_policy(
    attrs: &mut BTreeMap<String, Value>,
    policy: &PrefillChunkPolicy,
    min_chunk_size: usize,
    max_chunk_size: usize,
) {
    attrs.insert(
        "llama_stage.prefill_chunk_size".to_string(),
        json!(policy.fixed_chunk_size()),
    );
    attrs.insert(
        "llama_stage.prefill_chunk_policy".to_string(),
        json!(policy.policy_label()),
    );
    if let Some(schedule) = policy.schedule() {
        attrs.insert(
            "llama_stage.prefill_chunk_schedule".to_string(),
            json!(schedule.label()),
        );
    }
    if let Some((start, step, max)) = policy.adaptive_params() {
        attrs.insert(
            "llama_stage.prefill_adaptive_start".to_string(),
            json!(start),
        );
        attrs.insert("llama_stage.prefill_adaptive_step".to_string(), json!(step));
        attrs.insert("llama_stage.prefill_adaptive_max".to_string(), json!(max));
    }
    if min_chunk_size != usize::MAX {
        attrs.insert(
            "llama_stage.prefill_min_chunk_size".to_string(),
            json!(min_chunk_size),
        );
        attrs.insert(
            "llama_stage.prefill_max_chunk_size".to_string(),
            json!(max_chunk_size),
        );
    }
}

#[derive(Debug, Clone)]
struct PreparedGenerationPrompt {
    text: String,
    media: Vec<MediaInput>,
    chat_parse_metadata: Option<String>,
}

impl PreparedGenerationPrompt {
    fn text(text: String) -> Self {
        Self {
            text,
            media: Vec::new(),
            chat_parse_metadata: None,
        }
    }

    fn has_media(&self) -> bool {
        !self.media.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedToolCalls {
    content: Option<String>,
    tool_calls: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedChatMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Value>,
}

fn tool_calls_requested(request: &ChatCompletionRequest) -> bool {
    request.tools.as_ref().is_some_and(has_requested_tools)
        && !request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| matches!(choice.as_str(), Some("none")))
}

fn chat_response_from_generated_text(
    model: String,
    output: &GeneratedText,
    parsed_message: Option<ParsedChatMessage>,
) -> ChatCompletionResponse {
    if let Some(parsed) = parsed_message {
        let finish_reason = if parsed.tool_calls.is_some() {
            FinishReason::ToolCalls
        } else {
            output.finish_reason
        };
        return ChatCompletionResponse {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion",
            created: openai_frontend::now_unix_secs(),
            model,
            choices: vec![openai_frontend::ChatCompletionChoice {
                index: 0,
                message: openai_frontend::AssistantMessage {
                    role: "assistant",
                    content: parsed.content,
                    reasoning_content: parsed.reasoning_content,
                    tool_calls: parsed.tool_calls,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage: output.usage(),
        };
    }

    ChatCompletionResponse::new_with_reason(
        model,
        output.text.clone(),
        output.usage(),
        output.finish_reason,
    )
}

fn parsed_chat_message_from_json(
    message_json: &str,
    request: &ChatCompletionRequest,
) -> Option<ParsedChatMessage> {
    let value = serde_json::from_str::<Value>(message_json).ok()?;
    let tool_calls =
        parsed_tool_calls_from_message_value(&value, request).map(|parsed| parsed.tool_calls);
    Some(ParsedChatMessage {
        content: string_field(&value, "content"),
        reasoning_content: string_field(&value, "reasoning_content"),
        tool_calls,
    })
}

#[cfg(test)]
fn parsed_tool_calls_from_message_json(
    message_json: &str,
    request: &ChatCompletionRequest,
) -> Option<ParsedToolCalls> {
    let value = serde_json::from_str::<Value>(message_json).ok()?;
    parsed_tool_calls_from_message_value(&value, request)
}

fn parsed_tool_calls_from_message_value(
    value: &Value,
    request: &ChatCompletionRequest,
) -> Option<ParsedToolCalls> {
    if !tool_calls_requested(request) {
        return None;
    }
    let allowed_names = request_allowed_tool_names(request);
    let mut tool_calls = value
        .get("tool_calls")
        .and_then(Value::as_array)?
        .iter()
        .filter(|call| tool_call_allowed(call, &allowed_names))
        .cloned()
        .collect::<Vec<_>>();
    if request.parallel_tool_calls == Some(false) {
        tool_calls.truncate(1);
    }
    if tool_calls.is_empty() {
        return None;
    }
    Some(ParsedToolCalls {
        content: string_field(value, "content"),
        tool_calls: Value::Array(tool_calls),
    })
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .map(ToString::to_string)
}

fn request_allowed_tool_names(request: &ChatCompletionRequest) -> Vec<String> {
    if let Some(choice_name) = request
        .tool_choice
        .as_ref()
        .and_then(tool_choice_function_name)
    {
        return vec![choice_name];
    }
    request_tool_names(request)
}

fn tool_choice_function_name(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|object| {
            object
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| object.get("name"))
        })
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .as_str()
                .filter(|choice| !matches!(*choice, "auto" | "none" | "required"))
        })
        .map(ToString::to_string)
}

fn request_tool_names(request: &ChatCompletionRequest) -> Vec<String> {
    request
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| tool.get("name"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect()
}

fn tool_call_allowed(value: &Value, allowed_names: &[String]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let function = object.get("function").and_then(Value::as_object);
    let Some(name) = function
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    allowed_names.is_empty() || allowed_names.iter().any(|allowed| allowed == name)
}

fn tool_calls_stream_delta(tool_calls: Value) -> Value {
    match tool_calls {
        Value::Array(calls) => Value::Array(
            calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| match call {
                    Value::Object(mut object) => {
                        object
                            .entry("index")
                            .or_insert_with(|| Value::from(index as u64));
                        Value::Object(object)
                    }
                    other => other,
                })
                .collect(),
        ),
        other => other,
    }
}

fn chat_message_generation_value(
    message: &openai_frontend::ChatMessage,
    marker: &str,
    media: &mut Vec<MediaInput>,
) -> OpenAiResult<Value> {
    let mut value = serde_json::to_value(message)
        .map_err(|error| OpenAiError::invalid_request(format!("serialize message: {error}")))?;
    let content = message
        .content
        .as_ref()
        .map(|content| message_content_to_generation_text(content, marker, media))
        .transpose()?;
    if let Some(object) = value.as_object_mut() {
        match content {
            Some(content) => {
                object.insert("content".to_string(), Value::String(content));
            }
            None => {
                object.insert("content".to_string(), Value::Null);
            }
        }
    }
    Ok(value)
}

struct LocalGeneration<'a> {
    prompt_token_ids: &'a [i32],
    max_tokens: u32,
    sampling: &'a SamplingConfig,
    chat_sampling_metadata: Option<&'a str>,
    hook_request: Option<ChatCompletionRequest>,
    hook_runtime: Option<tokio::runtime::Handle>,
    cancellation: Option<&'a openai_frontend::CancellationToken>,
    ids: &'a OpenAiGenerationIds,
}

struct BinaryChainGeneration<'a> {
    first_stage_addr: &'a str,
    wire_dtype: WireActivationDType,
    prefill_chunk_policy: &'a PrefillChunkPolicy,
    startup_timeout_secs: u64,
    prompt_token_ids: &'a [i32],
    max_tokens: u32,
    sampling: &'a SamplingConfig,
    chat_sampling_metadata: Option<&'a str>,
    cancellation: Option<&'a openai_frontend::CancellationToken>,
    ids: &'a OpenAiGenerationIds,
}

struct EmbeddedStageZeroGeneration<'a> {
    config: &'a StageConfig,
    wire_dtype: WireActivationDType,
    prefill_chunk_policy: &'a PrefillChunkPolicy,
    activation_width: i32,
    downstream_wire_condition: WireCondition,
    prefill_reply_credit_limit: usize,
    lane_pool: Option<Arc<PersistentStageLanePool>>,
    draft: Option<Arc<Mutex<DraftRunner>>>,
    speculative_window: usize,
    adaptive_speculative_window: bool,
    prompt_token_ids: &'a [i32],
    max_tokens: u32,
    sampling: &'a SamplingConfig,
    chat_sampling_metadata: Option<&'a str>,
    hook_request: Option<ChatCompletionRequest>,
    hook_runtime: Option<tokio::runtime::Handle>,
    cancellation: Option<&'a openai_frontend::CancellationToken>,
    ids: &'a OpenAiGenerationIds,
}

struct SplitMultimodalGeneration<'a> {
    prompt: PreparedGenerationPrompt,
    max_tokens: GenerationTokenLimit,
    stop: Option<&'a openai_frontend::StopSequence>,
    sampling: SamplingConfig,
    cancellation: Option<&'a openai_frontend::CancellationToken>,
    ids: OpenAiGenerationIds,
    config: StageConfig,
    wire_dtype: WireActivationDType,
    activation_width: i32,
    downstream_wire_condition: WireCondition,
    lane_pool: Arc<PersistentStageLanePool>,
}

struct EmbeddedLocalOutput {
    output: skippy_runtime::ActivationFrame,
    runtime_lock_wait_ms: f64,
    runtime_lock_hold_ms: f64,
}

struct EmbeddedExecutionStats {
    stage0_compute_ms: f64,
    runtime_lock_wait_ms: f64,
    runtime_lock_hold_ms: f64,
    activation_encode_ms: f64,
    output_activation_bytes: usize,
    forward_activation_bytes: usize,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
}

struct EmbeddedStageExecution {
    reply: StageReply,
    stats: EmbeddedExecutionStats,
    elapsed_ms: f64,
}

struct EmbeddedFusedFirstDecode {
    predicted: i32,
    reply_stats: StageReplyStats,
    execution: EmbeddedExecutionStats,
    elapsed_ms: f64,
}

struct EmbeddedSessionControl {
    elapsed_ms: f64,
    local_ms: f64,
    downstream_write_ms: f64,
    downstream_wait_ms: f64,
}

type GenerationStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = OpenAiResult<GenerationStreamEvent>> + Send>>;

enum GenerationStreamEvent {
    Delta(String),
    ReasoningDelta(String),
    ToolCalls(Value),
    Usage(Usage),
    Done(FinishReason),
}

struct ChatOutputStreamParser {
    backend: StageOpenAiBackend,
    request: ChatCompletionRequest,
    metadata: String,
    text: String,
    emitted_content: String,
    emitted_reasoning_content: String,
    emitted_tool_calls: bool,
}

impl ChatOutputStreamParser {
    fn new(backend: StageOpenAiBackend, request: ChatCompletionRequest, metadata: String) -> Self {
        Self {
            backend,
            request,
            metadata,
            text: String::new(),
            emitted_content: String::new(),
            emitted_reasoning_content: String::new(),
            emitted_tool_calls: false,
        }
    }

    fn push_delta(&mut self, delta: &str) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        self.text.push_str(delta);
        self.events_for_text(true)
    }

    fn finish(&mut self, text: &str) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        if self.text != text {
            self.text = text.to_string();
        }
        self.events_for_text(false)
    }

    fn events_for_text(&mut self, is_partial: bool) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        let Some(parsed) = self.backend.parse_chat_output(
            &self.text,
            &self.request,
            Some(&self.metadata),
            is_partial,
        )?
        else {
            return Ok(Vec::new());
        };
        let mut events = Vec::new();
        if let Some(delta) = suffix_delta(
            parsed.reasoning_content.as_deref(),
            &mut self.emitted_reasoning_content,
        ) {
            events.push(GenerationStreamEvent::ReasoningDelta(delta));
        }
        if let Some(delta) = suffix_delta(parsed.content.as_deref(), &mut self.emitted_content) {
            events.push(GenerationStreamEvent::Delta(delta));
        }
        if !is_partial && !self.emitted_tool_calls {
            if let Some(tool_calls) = parsed.tool_calls {
                self.emitted_tool_calls = true;
                events.push(GenerationStreamEvent::ToolCalls(tool_calls));
            }
        }
        Ok(events)
    }

    fn finish_reason(&self, fallback: FinishReason) -> FinishReason {
        if self.emitted_tool_calls {
            FinishReason::ToolCalls
        } else {
            fallback
        }
    }
}

fn suffix_delta(current: Option<&str>, emitted: &mut String) -> Option<String> {
    let current = current?;
    let delta = current.strip_prefix(emitted.as_str())?;
    if delta.is_empty() {
        return None;
    }
    emitted.push_str(delta);
    Some(delta.to_string())
}

fn generation_event_to_chat_chunk(
    event: OpenAiResult<GenerationStreamEvent>,
    model: &str,
) -> OpenAiResult<ChatCompletionChunk> {
    match event? {
        GenerationStreamEvent::Delta(delta) => {
            Ok(ChatCompletionChunk::delta(model.to_string(), delta))
        }
        GenerationStreamEvent::ReasoningDelta(delta) => Ok(ChatCompletionChunk {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: openai_frontend::now_unix_secs(),
            model: model.to_string(),
            choices: vec![openai_frontend::ChatCompletionChunkChoice {
                index: 0,
                delta: openai_frontend::ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some(delta),
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }),
        GenerationStreamEvent::ToolCalls(tool_calls) => Ok(ChatCompletionChunk {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: openai_frontend::now_unix_secs(),
            model: model.to_string(),
            choices: vec![openai_frontend::ChatCompletionChunkChoice {
                index: 0,
                delta: openai_frontend::ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(tool_calls_stream_delta(tool_calls)),
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }),
        GenerationStreamEvent::Usage(usage) => {
            Ok(ChatCompletionChunk::usage(model.to_string(), usage))
        }
        GenerationStreamEvent::Done(reason) => Ok(ChatCompletionChunk::done_with_reason(
            model.to_string(),
            reason,
        )),
    }
}

fn generation_event_to_completion_chunk(
    event: OpenAiResult<GenerationStreamEvent>,
    model: &str,
) -> OpenAiResult<CompletionChunk> {
    match event? {
        GenerationStreamEvent::Delta(delta) => Ok(CompletionChunk::delta(model.to_string(), delta)),
        GenerationStreamEvent::ReasoningDelta(_) => {
            Ok(CompletionChunk::delta(model.to_string(), ""))
        }
        GenerationStreamEvent::ToolCalls(_) => Ok(CompletionChunk::delta(model.to_string(), "")),
        GenerationStreamEvent::Usage(usage) => Ok(CompletionChunk::usage(model.to_string(), usage)),
        GenerationStreamEvent::Done(reason) => {
            Ok(CompletionChunk::done_with_reason(model.to_string(), reason))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokenControl {
    Continue,
    Stop,
}

struct TextGenerationCollector<'a, F>
where
    F: FnMut(&str) -> OpenAiResult<()>,
{
    runtime: Arc<Mutex<RuntimeState>>,
    stop_values: Vec<&'a str>,
    on_text_chunk: F,
    text: String,
    streamed_text_len: usize,
    max_stop_bytes: usize,
    generated_text_tokens: Vec<i32>,
    completion_tokens: usize,
    finish_reason: FinishReason,
    metrics: GenerationMetrics,
}

impl<'a, F> TextGenerationCollector<'a, F>
where
    F: FnMut(&str) -> OpenAiResult<()>,
{
    fn new(runtime: Arc<Mutex<RuntimeState>>, stop_values: Vec<&'a str>, on_text_chunk: F) -> Self {
        let max_stop_bytes = stop_values
            .iter()
            .map(|value| value.len())
            .max()
            .unwrap_or(0);
        Self {
            runtime,
            stop_values,
            on_text_chunk,
            text: String::new(),
            streamed_text_len: 0,
            max_stop_bytes,
            generated_text_tokens: Vec::new(),
            completion_tokens: 0,
            finish_reason: finish_reason_for_generation(true),
            metrics: GenerationMetrics::default(),
        }
    }

    fn push_token(&mut self, token: i32) -> OpenAiResult<TokenControl> {
        let eog_timer = Instant::now();
        if token_is_eog_with_runtime(&self.runtime, token)? {
            self.metrics.eog_check_ms += eog_timer.elapsed().as_secs_f64() * 1000.0;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        self.metrics.eog_check_ms += eog_timer.elapsed().as_secs_f64() * 1000.0;
        self.completion_tokens += 1;
        self.generated_text_tokens.push(token);
        let detokenize_timer = Instant::now();
        let candidate_bytes =
            detokenize_bytes_with_runtime(&self.runtime, &self.generated_text_tokens)?;
        self.metrics.detokenize_ms += detokenize_timer.elapsed().as_secs_f64() * 1000.0;
        let valid_len = valid_utf8_prefix_len(&candidate_bytes);
        if valid_len > 0 {
            let candidate = std::str::from_utf8(&candidate_bytes[..valid_len])
                .map_err(|error| OpenAiError::backend(error.to_string()))?;
            if let Some(delta) = candidate.strip_prefix(&self.text) {
                if !delta.is_empty() {
                    self.text = candidate.to_string();
                }
            } else if candidate != self.text {
                self.text = candidate.to_string();
            }
        }
        if self
            .stop_values
            .iter()
            .any(|stop| !stop.is_empty() && self.text.contains(stop))
        {
            self.text = trim_at_stop(&self.text, &self.stop_values).to_string();
            self.emit_safe_delta(true)?;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        self.emit_safe_delta(false)?;
        Ok(TokenControl::Continue)
    }

    fn emit_safe_delta(&mut self, flush_all: bool) -> OpenAiResult<()> {
        let mut target_len = if flush_all || self.max_stop_bytes == 0 {
            self.text.len()
        } else {
            self.text
                .len()
                .saturating_sub(self.max_stop_bytes.saturating_sub(1))
        };
        while target_len > self.streamed_text_len && !self.text.is_char_boundary(target_len) {
            target_len -= 1;
        }
        if target_len < self.streamed_text_len {
            self.streamed_text_len = target_len;
            return Ok(());
        }
        if target_len > self.streamed_text_len {
            let delta = &self.text[self.streamed_text_len..target_len];
            let emit_timer = Instant::now();
            (self.on_text_chunk)(delta)?;
            self.metrics.text_emit_ms += emit_timer.elapsed().as_secs_f64() * 1000.0;
            self.streamed_text_len = target_len;
        }
        Ok(())
    }

    fn finish(
        mut self,
        prompt_token_count: usize,
        cache_stats: GenerationCacheStats,
    ) -> OpenAiResult<GeneratedText> {
        self.emit_safe_delta(true)?;
        Ok(GeneratedText {
            prompt_tokens: saturating_u32(prompt_token_count),
            completion_tokens: saturating_u32(self.completion_tokens),
            cached_prompt_tokens: cache_stats.cached_prompt_tokens,
            matched_prefix_tokens: cache_stats.matched_prefix_tokens,
            suffix_prefill_tokens: cache_stats.suffix_prefill_tokens,
            cache_hit_kind: cache_stats.hit_kind,
            text: self.text,
            finish_reason: self.finish_reason,
            detokenize_ms: self.metrics.detokenize_ms,
            text_emit_ms: self.metrics.text_emit_ms,
            eog_check_ms: self.metrics.eog_check_ms,
        })
    }
}

struct GeneratedText {
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_prompt_tokens: u32,
    matched_prefix_tokens: u32,
    suffix_prefill_tokens: u32,
    cache_hit_kind: Option<&'static str>,
    text: String,
    finish_reason: FinishReason,
    detokenize_ms: f64,
    text_emit_ms: f64,
    eog_check_ms: f64,
}

impl GeneratedText {
    fn usage(&self) -> Usage {
        Usage::new(self.prompt_tokens, self.completion_tokens)
            .with_cached_tokens(self.cached_prompt_tokens)
    }
}

fn finish_reason_for_generation(exhausted_max_tokens: bool) -> FinishReason {
    if exhausted_max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

fn ensure_context_capacity(
    prompt_token_count: usize,
    max_tokens: u32,
    ctx_size: usize,
) -> OpenAiResult<()> {
    let requested_tokens = prompt_token_count.saturating_add(max_tokens as usize);
    if requested_tokens > ctx_size {
        return Err(OpenAiError::context_length_exceeded(format!(
            "requested prompt plus completion tokens ({requested_tokens}) exceed context window ({ctx_size})"
        )));
    }
    Ok(())
}

fn context_budget_completion_tokens(
    prompt_token_count: usize,
    ctx_size: usize,
) -> OpenAiResult<u32> {
    if prompt_token_count > ctx_size {
        return Err(OpenAiError::context_length_exceeded(format!(
            "requested prompt tokens ({prompt_token_count}) exceed context window ({ctx_size})"
        )));
    }
    Ok(ctx_size
        .saturating_sub(prompt_token_count)
        .min(u32::MAX as usize) as u32)
}

fn detokenize_bytes_with_runtime(
    runtime: &Arc<Mutex<RuntimeState>>,
    token_ids: &[i32],
) -> OpenAiResult<Vec<u8>> {
    let runtime = runtime
        .lock()
        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
    runtime
        .model
        .detokenize_bytes(token_ids)
        .map_err(openai_backend_error)
}

fn token_is_eog_with_runtime(
    runtime: &Arc<Mutex<RuntimeState>>,
    token_id: i32,
) -> OpenAiResult<bool> {
    let runtime = runtime
        .lock()
        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
    runtime
        .model
        .token_is_eog(token_id)
        .map_err(openai_backend_error)
}

fn ms_to_us(ms: f64) -> i64 {
    (ms * 1000.0).round() as i64
}

fn us_to_ms(us: i64) -> f64 {
    us as f64 / 1000.0
}

fn openai_backend_error(error: anyhow::Error) -> OpenAiError {
    OpenAiError::backend(error.to_string())
}

fn openai_io_error(error: std::io::Error) -> OpenAiError {
    OpenAiError::backend(error.to_string())
}

fn parse_wire_dtype(value: &str) -> Result<WireActivationDType> {
    match value {
        "fp32" | "f32" => Ok(WireActivationDType::F32),
        "fp16" | "f16" => Ok(WireActivationDType::F16),
        "q8" | "int8" | "i8" => Ok(WireActivationDType::Q8),
        _ => bail!("unsupported activation wire dtype {value}"),
    }
}

fn connect_endpoint_ready(endpoint: &str, timeout_secs: u64) -> Result<TcpStream> {
    let endpoint = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match TcpStream::connect(endpoint) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                match recv_ready(&mut stream) {
                    Ok(()) => return Ok(stream),
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

#[cfg(test)]
mod tests;
