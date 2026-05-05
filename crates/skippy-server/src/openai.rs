use std::{
    collections::BTreeMap,
    future::Future,
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
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
    ChatTemplateJsonOptions, ChatTemplateOptions, FlashAttentionType as RuntimeFlashAttentionType,
    GenerationSignalWindow, LogitBias as RuntimeLogitBias, MediaInput, ModelInfo, RuntimeConfig,
    RuntimeLoadMode, SamplingConfig, StageModel, StageSession, TokenSignal, MAX_LOGIT_BIAS,
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, Semaphore},
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

mod request;
mod util;

use self::{request::*, util::*};

static OPENAI_GENERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub const CONTEXT_BUDGET_MAX_TOKENS: u32 = u32::MAX;

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
    hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    kv: Option<Arc<KvStageIntegration>>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct PrefillChunkSchedule {
    sizes: Vec<usize>,
}

impl PrefillChunkSchedule {
    fn parse(spec: Option<&str>) -> Result<Option<Self>> {
        let Some(spec) = spec else {
            return Ok(None);
        };
        let spec = spec.trim();
        if spec.is_empty() {
            return Ok(None);
        }
        let mut sizes = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                bail!("empty chunk size in schedule");
            }
            let size = part
                .parse::<usize>()
                .with_context(|| format!("invalid chunk size '{part}'"))?;
            if size == 0 {
                bail!("chunk sizes must be greater than zero");
            }
            sizes.push(size);
        }
        Ok(Some(Self { sizes }))
    }

    fn chunk_size_for(&self, chunk_index: usize) -> usize {
        self.sizes
            .get(chunk_index)
            .copied()
            .or_else(|| self.sizes.last().copied())
            .expect("schedule has at least one size")
    }

    fn label(&self) -> String {
        self.sizes
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PrefillChunkPolicy {
    Fixed {
        chunk_size: usize,
    },
    Schedule {
        fixed_chunk_size: usize,
        schedule: PrefillChunkSchedule,
    },
    AdaptiveRamp {
        fixed_chunk_size: usize,
        start: usize,
        step: usize,
        max: usize,
    },
}

struct PrefillChunkPolicyArgs<'a> {
    policy: &'a str,
    schedule: Option<&'a str>,
    fixed_chunk_size: usize,
    adaptive_start: usize,
    adaptive_step: usize,
    adaptive_max: usize,
    schedule_arg: &'static str,
    policy_arg: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct PrefillChunkObservation {
    compute_ms: f64,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
}

#[derive(Clone, Debug)]
struct PrefillChunkPlanner {
    policy: PrefillChunkPolicy,
    next_adaptive_size: usize,
}

impl PrefillChunkPolicy {
    fn parse(args: PrefillChunkPolicyArgs<'_>) -> Result<Self> {
        if args.fixed_chunk_size == 0 {
            bail!("prefill chunk size must be greater than zero");
        }
        let normalized = args.policy.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "fixed" => {
                if let Some(schedule) = PrefillChunkSchedule::parse(args.schedule)
                    .with_context(|| format!("invalid {} value", args.schedule_arg))?
                {
                    return Ok(Self::Schedule {
                        fixed_chunk_size: args.fixed_chunk_size,
                        schedule,
                    });
                }
                Ok(Self::Fixed {
                    chunk_size: args.fixed_chunk_size,
                })
            }
            "schedule" => {
                let schedule = PrefillChunkSchedule::parse(args.schedule)
                    .with_context(|| format!("invalid {} value", args.schedule_arg))?
                    .ok_or_else(|| anyhow!("{} requires {}", args.policy_arg, args.schedule_arg))?;
                Ok(Self::Schedule {
                    fixed_chunk_size: args.fixed_chunk_size,
                    schedule,
                })
            }
            "adaptive" | "adaptive-ramp" => {
                if args.adaptive_start == 0
                    || args.adaptive_step == 0
                    || args.adaptive_max == 0
                    || args.adaptive_start > args.adaptive_max
                {
                    bail!(
                        "{} adaptive-ramp requires positive start/step/max with start <= max",
                        args.policy_arg
                    );
                }
                Ok(Self::AdaptiveRamp {
                    fixed_chunk_size: args.fixed_chunk_size,
                    start: args.adaptive_start,
                    step: args.adaptive_step,
                    max: args.adaptive_max,
                })
            }
            other => bail!(
                "invalid {} '{}'; expected fixed, schedule, or adaptive-ramp",
                args.policy_arg,
                other
            ),
        }
    }

    fn planner(&self) -> PrefillChunkPlanner {
        let next_adaptive_size = match self {
            Self::AdaptiveRamp { start, .. } => *start,
            _ => 0,
        };
        PrefillChunkPlanner {
            policy: self.clone(),
            next_adaptive_size,
        }
    }

    fn policy_label(&self) -> &'static str {
        match self {
            Self::Fixed { .. } => "fixed",
            Self::Schedule { .. } => "schedule",
            Self::AdaptiveRamp { .. } => "adaptive-ramp",
        }
    }

    fn fixed_chunk_size(&self) -> usize {
        match self {
            Self::Fixed { chunk_size } => *chunk_size,
            Self::Schedule {
                fixed_chunk_size, ..
            }
            | Self::AdaptiveRamp {
                fixed_chunk_size, ..
            } => *fixed_chunk_size,
        }
    }

    fn schedule(&self) -> Option<&PrefillChunkSchedule> {
        match self {
            Self::Schedule { schedule, .. } => Some(schedule),
            _ => None,
        }
    }

    fn adaptive_params(&self) -> Option<(usize, usize, usize)> {
        match self {
            Self::AdaptiveRamp {
                start, step, max, ..
            } => Some((*start, *step, *max)),
            _ => None,
        }
    }
}

impl PrefillChunkPlanner {
    fn chunk_size_for(&mut self, chunk_index: usize) -> usize {
        match &self.policy {
            PrefillChunkPolicy::Fixed { chunk_size } => *chunk_size,
            PrefillChunkPolicy::Schedule { schedule, .. } => schedule.chunk_size_for(chunk_index),
            PrefillChunkPolicy::AdaptiveRamp { .. } => self.next_adaptive_size,
        }
    }

    fn observe(&mut self, observation: PrefillChunkObservation) {
        let PrefillChunkPolicy::AdaptiveRamp {
            start, step, max, ..
        } = &self.policy
        else {
            return;
        };
        let compute_ms = observation.compute_ms.max(0.001);
        let downstream_hidden = observation.downstream_wait_ms <= compute_ms * 0.75
            && observation.forward_write_ms <= compute_ms * 0.25;
        let downstream_exposed = observation.downstream_wait_ms > compute_ms * 1.25;
        if downstream_hidden {
            self.next_adaptive_size = self.next_adaptive_size.saturating_add(*step).min(*max);
        } else if downstream_exposed {
            self.next_adaptive_size = self.next_adaptive_size.saturating_sub(*step).max(*start);
        }
    }

    fn advance_without_observation(&mut self) {
        let PrefillChunkPolicy::AdaptiveRamp { step, max, .. } = &self.policy else {
            return;
        };
        self.next_adaptive_size = self.next_adaptive_size.saturating_add(*step).min(*max);
    }
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
}

impl OpenAiGenerationIds {
    fn new() -> Self {
        let sequence = OPENAI_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let session_label = format!("openai-session-{}-{sequence}", now_unix_millis());
        Self {
            session_id: stable_wire_id(&[session_label.as_bytes()]),
            request_id: stable_wire_id(&[session_label.as_bytes(), b"request"]),
            session_label,
        }
    }

    fn session_id_string(&self) -> String {
        self.session_id.to_string()
    }

    fn request_id_string(&self) -> String {
        self.request_id.to_string()
    }
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

#[async_trait]
impl OpenAiBackend for StageOpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let ids = OpenAiGenerationIds::new();
        let request_timer = PhaseTimer::start();
        self.apply_before_chat_hooks(&mut request).await?;
        self.ensure_model(&request.model)?;
        ensure_chat_runtime_features_supported(&request)?;
        let sampling = chat_sampling_config(&request)?;
        let template_options = chat_template_options(&request)?;
        let template_timer = PhaseTimer::start();
        let prompt = self.prepare_chat_prompt(&request, template_options)?;
        let mut template_attrs = self.openai_attrs(&ids);
        template_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        template_attrs.insert(
            "llama_stage.chat_message_count".to_string(),
            json!(request.messages.len()),
        );
        template_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        template_attrs.insert(
            "llama_stage.media_item_count".to_string(),
            json!(prompt.media.len()),
        );
        self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
        let max_tokens = GenerationTokenLimit::from_request(
            request.effective_max_tokens(),
            self.default_max_tokens,
        );
        let chat_parse_metadata = prompt.chat_parse_metadata.clone();
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                Some(request.clone()),
                ids.clone(),
            )
            .await?;
        let response_timer = PhaseTimer::start();
        let parsed_tool_calls =
            self.parse_tool_call_output(&output.text, &request, chat_parse_metadata.as_deref())?;
        let response =
            chat_response_from_generated_text(request.model.clone(), &output, parsed_tool_calls);
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let ids = OpenAiGenerationIds::new();
        self.apply_before_chat_hooks(&mut request).await?;
        self.ensure_model(&request.model)?;
        ensure_chat_runtime_features_supported(&request)?;
        let sampling = chat_sampling_config(&request)?;
        let include_usage = request.include_usage();
        let template_options = chat_template_options(&request)?;
        let template_timer = PhaseTimer::start();
        let prompt = self.prepare_chat_prompt(&request, template_options)?;
        let mut template_attrs = self.openai_attrs(&ids);
        template_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion_stream"),
        );
        template_attrs.insert(
            "llama_stage.chat_message_count".to_string(),
            json!(request.messages.len()),
        );
        template_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        template_attrs.insert(
            "llama_stage.media_item_count".to_string(),
            json!(prompt.media.len()),
        );
        self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
        let max_tokens = GenerationTokenLimit::from_request(
            request.effective_max_tokens(),
            self.default_max_tokens,
        );
        let model = request.model.clone();
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                Some(request.clone()),
                context,
                ids,
            )
            .await?;
        Ok(Box::pin(stream.map(move |event| {
            generation_event_to_chat_chunk(event, &model)
        })))
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        let ids = OpenAiGenerationIds::new();
        let request_timer = PhaseTimer::start();
        self.ensure_model(&request.model)?;
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                None,
                ids.clone(),
            )
            .await?;
        let response_timer = PhaseTimer::start();
        let response = CompletionResponse::new_with_reason(
            request.model,
            output.text,
            Usage::new(output.prompt_tokens, output.completion_tokens),
            output.finish_reason,
        );
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        let ids = OpenAiGenerationIds::new();
        self.ensure_model(&request.model)?;
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let include_usage = request.include_usage();
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let model = request.model.clone();
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion_stream"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                None,
                context,
                ids,
            )
            .await?;
        Ok(Box::pin(stream.map(move |event| {
            generation_event_to_completion_chunk(event, &model)
        })))
    }
}

impl StageOpenAiBackend {
    fn openai_attrs(&self, ids: &OpenAiGenerationIds) -> BTreeMap<String, Value> {
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            attr_key::SESSION_ID.to_string(),
            json!(ids.session_id_string()),
        );
        attrs.insert(
            attr_key::REQUEST_ID.to_string(),
            json!(ids.request_id_string()),
        );
        attrs.insert(
            "llama_stage.openai_backend".to_string(),
            json!(self.mode.label()),
        );
        attrs
    }

    fn local_kv_message_base(&self, session_id: &str, ids: &OpenAiGenerationIds) -> MessageBase {
        MessageBase {
            schema_version: SCHEMA_VERSION,
            run_id: self.config.run_id.clone(),
            request_id: ids.request_id_string(),
            session_id: session_id.to_string(),
            stage_id: "openai-local".to_string(),
            stage_index: self.config.stage_index,
            topology_id: self.config.topology_id.clone(),
            model_id: Some(self.config.model_id.clone()),
            tokenizer_id: None,
            chat_template_id: None,
            seq: Some(ids.session_id),
        }
    }

    fn insert_runtime_session_stats(
        attrs: &mut BTreeMap<String, Value>,
        prefix: &str,
        stats: RuntimeSessionStats,
    ) {
        attrs.insert(
            format!("{prefix}.active_sessions"),
            json!(stats.active_sessions),
        );
        attrs.insert(
            format!("{prefix}.idle_sessions"),
            json!(stats.idle_sessions),
        );
        attrs.insert(
            format!("{prefix}.tracked_token_counts"),
            json!(stats.tracked_token_counts),
        );
        attrs.insert(format!("{prefix}.checkpoints"), json!(stats.checkpoints));
    }

    fn emit_openai_phase(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_debug_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    fn emit_openai_summary(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    fn ensure_model(&self, requested: &str) -> OpenAiResult<()> {
        ensure_requested_model(&self.model_id, requested)
    }

    async fn apply_before_chat_hooks(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<()> {
        let Some(hooks) = self.hook_policy.as_ref() else {
            return Ok(());
        };
        if !chat_mesh_hooks_enabled(request) {
            return Ok(());
        }
        let outcome = hooks.before_chat_completion(request).await?;
        apply_chat_hook_outcome(request, &outcome);
        Ok(())
    }

    async fn run_generation(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GeneratedText> {
        let admit_timer = PhaseTimer::start();
        let permit = self
            .generation_limit
            .clone()
            .try_acquire_owned()
            .map_err(|_| generation_lanes_busy_error())?;
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let backend = self.clone();
        let hook_runtime = Some(tokio::runtime::Handle::current());
        task::spawn_blocking(move || {
            let _permit = permit;
            backend.generate_text(
                prompt,
                max_tokens,
                stop.as_ref(),
                sampling,
                hook_request,
                hook_runtime,
                None,
                ids,
                |_| Ok(()),
            )
        })
        .await
        .map_err(|error| OpenAiError::backend(format!("generation task failed: {error}")))?
    }

    async fn run_generation_stream(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        include_usage: bool,
        hook_request: Option<ChatCompletionRequest>,
        context: OpenAiRequestContext,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GenerationStream> {
        let admit_timer = PhaseTimer::start();
        let permit = self
            .generation_limit
            .clone()
            .try_acquire_owned()
            .map_err(|_| generation_lanes_busy_error())?;
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let backend = self.clone();
        let tool_call_stream = hook_request.as_ref().is_some_and(tool_calls_requested)
            && prompt.chat_parse_metadata.is_some();
        let chat_parse_metadata = prompt.chat_parse_metadata.clone();
        let (tx, rx) = mpsc::channel(16);
        let hook_runtime = Some(tokio::runtime::Handle::current());
        let stream_tool_request = hook_request.clone();
        task::spawn_blocking(move || {
            let _permit = permit;
            let result = backend.generate_text(
                prompt,
                max_tokens,
                stop.as_ref(),
                sampling,
                hook_request,
                hook_runtime,
                Some(&context.cancellation_token()),
                ids,
                |chunk| {
                    if tool_call_stream {
                        return Ok(());
                    }
                    if context.is_cancelled() {
                        return Err(OpenAiError::backend("stream receiver cancelled"));
                    }
                    tx.blocking_send(Ok(GenerationStreamEvent::Delta(chunk.to_string())))
                        .map_err(|_| {
                            context.cancel();
                            OpenAiError::backend("stream receiver dropped")
                        })
                },
            );
            if context.is_cancelled() {
                return;
            }
            match result {
                Ok(output) => {
                    if tool_call_stream {
                        if let (Some(request), Some(metadata)) =
                            (stream_tool_request.as_ref(), chat_parse_metadata.as_deref())
                        {
                            match backend.parse_tool_call_output(
                                &output.text,
                                request,
                                Some(metadata),
                            ) {
                                Ok(Some(tool_output)) => {
                                    if tx
                                        .blocking_send(Ok(GenerationStreamEvent::ToolCalls(
                                            tool_output.tool_calls,
                                        )))
                                        .is_err()
                                    {
                                        context.cancel();
                                        return;
                                    }
                                    if include_usage
                                        && tx
                                            .blocking_send(Ok(GenerationStreamEvent::Usage(
                                                output.usage(),
                                            )))
                                            .is_err()
                                    {
                                        context.cancel();
                                        return;
                                    }
                                    let _ = tx.blocking_send(Ok(GenerationStreamEvent::Done(
                                        FinishReason::ToolCalls,
                                    )));
                                    return;
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    let _ = tx.blocking_send(Err(error));
                                    return;
                                }
                            }
                        }
                        if !output.text.is_empty()
                            && tx
                                .blocking_send(Ok(GenerationStreamEvent::Delta(
                                    output.text.clone(),
                                )))
                                .is_err()
                        {
                            context.cancel();
                            return;
                        }
                    }
                    if include_usage
                        && tx
                            .blocking_send(Ok(GenerationStreamEvent::Usage(output.usage())))
                            .is_err()
                    {
                        context.cancel();
                        return;
                    }
                    let _ = tx.blocking_send(Ok(GenerationStreamEvent::Done(output.finish_reason)));
                }
                Err(error) => {
                    let _ = tx.blocking_send(Err(error));
                }
            }
        });
        Ok(Box::pin(stream::unfold(rx, |mut rx| async {
            rx.recv().await.map(|item| (item, rx))
        })))
    }

    fn prepare_chat_prompt(
        &self,
        request: &ChatCompletionRequest,
        options: ChatTemplateOptions,
    ) -> OpenAiResult<PreparedGenerationPrompt> {
        let marker = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime.media_marker()
        };
        let mut media = Vec::new();
        let template_messages = request
            .messages
            .iter()
            .map(|message| chat_message_generation_value(message, &marker, &mut media))
            .collect::<OpenAiResult<Vec<_>>>()?;
        let messages_json = serde_json::to_string(&template_messages).map_err(|error| {
            OpenAiError::invalid_request(format!("serialize messages: {error}"))
        })?;
        let tools_json = request
            .tools
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| OpenAiError::invalid_request(format!("serialize tools: {error}")))?;
        let tool_choice_json = request
            .tool_choice
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| {
                OpenAiError::invalid_request(format!("serialize tool_choice: {error}"))
            })?;
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let result = runtime
            .model
            .apply_chat_template_json(
                &messages_json,
                ChatTemplateJsonOptions {
                    add_assistant: options.add_assistant,
                    enable_thinking: options.enable_thinking,
                    tools_json,
                    tool_choice_json,
                    parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
                },
            )
            .map_err(openai_backend_error)?;
        Ok(PreparedGenerationPrompt {
            text: result.prompt,
            media,
            chat_parse_metadata: Some(result.metadata_json),
        })
    }

    fn parse_tool_call_output(
        &self,
        text: &str,
        request: &ChatCompletionRequest,
        metadata: Option<&str>,
    ) -> OpenAiResult<Option<ParsedToolCalls>> {
        if !tool_calls_requested(request) {
            return Ok(None);
        }
        let Some(metadata) = metadata else {
            return Ok(None);
        };
        let parsed_json = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .model
                .parse_chat_response_json(text, metadata, false)
                .map_err(openai_backend_error)?
        };
        Ok(parsed_tool_calls_from_message_json(&parsed_json, request))
    }

    fn generate_text(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<&openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        hook_runtime: Option<tokio::runtime::Handle>,
        cancellation: Option<&openai_frontend::CancellationToken>,
        ids: OpenAiGenerationIds,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        let generation_timer = PhaseTimer::start();
        if prompt.text.is_empty() {
            return Err(OpenAiError::invalid_request(
                "request prompt/messages produced no text",
            ));
        }
        if prompt.has_media() {
            return self.generate_multimodal_text(
                prompt,
                max_tokens,
                stop,
                sampling,
                hook_request,
                hook_runtime,
                cancellation,
                ids,
                on_text_chunk,
            );
        }
        let stop_values = stop.map(|stop| stop.values()).unwrap_or_default();
        let tokenize_timer = PhaseTimer::start();
        let prompt_token_ids = self.tokenize(&prompt.text)?;
        let mut tokenize_attrs = self.openai_attrs(&ids);
        tokenize_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        tokenize_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(prompt_token_ids.len()),
        );
        self.emit_openai_phase("stage.openai_tokenize", tokenize_timer, tokenize_attrs);
        if prompt_token_ids.is_empty() {
            return Err(OpenAiError::invalid_request("prompt produced no tokens"));
        }
        let max_tokens = max_tokens.resolve(prompt_token_ids.len(), self.ctx_size)?;
        let chat_sampling_metadata = prompt.chat_parse_metadata.as_deref();

        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk);
        match self.mode.clone() {
            OpenAiBackendMode::LocalRuntime => self.generate_local_tokens(
                LocalGeneration {
                    prompt_token_ids: &prompt_token_ids,
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    hook_request: hook_request.clone(),
                    hook_runtime: hook_runtime.clone(),
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
            OpenAiBackendMode::BinaryChain {
                first_stage_addr,
                wire_dtype,
                prefill_chunk_policy,
                startup_timeout_secs,
            } => self.generate_binary_chain_tokens(
                BinaryChainGeneration {
                    first_stage_addr: &first_stage_addr,
                    wire_dtype,
                    prefill_chunk_policy: &prefill_chunk_policy,
                    startup_timeout_secs,
                    prompt_token_ids: &prompt_token_ids,
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
            OpenAiBackendMode::EmbeddedStageZero {
                config,
                wire_dtype,
                prefill_chunk_policy,
                activation_width,
                downstream_wire_condition,
                lane_pool,
            } => self.generate_embedded_stage_zero_tokens(
                EmbeddedStageZeroGeneration {
                    config: &config,
                    wire_dtype,
                    prefill_chunk_policy: &prefill_chunk_policy,
                    activation_width,
                    downstream_wire_condition,
                    lane_pool,
                    draft: self.draft.clone(),
                    speculative_window: self.speculative_window,
                    adaptive_speculative_window: self.adaptive_speculative_window,
                    prompt_token_ids: &prompt_token_ids,
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    hook_request,
                    hook_runtime,
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
        };

        let output = collector.finish(prompt_token_ids.len())?;
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        summary_attrs.insert(
            "llama_stage.detokenize_ms".to_string(),
            json!(output.detokenize_ms),
        );
        summary_attrs.insert(
            "llama_stage.text_emit_ms".to_string(),
            json!(output.text_emit_ms),
        );
        summary_attrs.insert(
            "llama_stage.eog_check_ms".to_string(),
            json!(output.eog_check_ms),
        );
        self.emit_openai_summary(
            "stage.openai_generation_summary",
            generation_timer,
            summary_attrs,
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_multimodal_text(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<&openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        hook_runtime: Option<tokio::runtime::Handle>,
        cancellation: Option<&openai_frontend::CancellationToken>,
        ids: OpenAiGenerationIds,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        if let OpenAiBackendMode::EmbeddedStageZero {
            config,
            wire_dtype,
            activation_width,
            downstream_wire_condition,
            lane_pool,
            ..
        } = self.mode.clone()
        {
            if config.downstream.is_some() {
                let lane_pool = lane_pool.ok_or_else(|| {
                    OpenAiError::backend("embedded stage 0 has no downstream lane pool")
                })?;
                return self.generate_split_multimodal_text(
                    SplitMultimodalGeneration {
                        prompt,
                        max_tokens,
                        stop,
                        sampling,
                        cancellation,
                        ids,
                        config,
                        wire_dtype,
                        activation_width,
                        downstream_wire_condition,
                        lane_pool,
                    },
                    on_text_chunk,
                );
            }
        }

        match &self.mode {
            OpenAiBackendMode::LocalRuntime => {}
            OpenAiBackendMode::EmbeddedStageZero { config, .. } if config.downstream.is_none() => {}
            OpenAiBackendMode::EmbeddedStageZero { .. } | OpenAiBackendMode::BinaryChain { .. } => {
                return Err(OpenAiError::unsupported(
                    "multimodal requests require an embedded stage-0 runtime",
                ));
            }
        }

        let stop_values = stop.map(|stop| stop.values()).unwrap_or_default();
        let session_id = ids.session_label.clone();
        let prefill_timer = PhaseTimer::start();
        let (prefill, mut token_signal, mut signal_window) = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            if !runtime.has_media_projector() {
                return Err(OpenAiError::invalid_request(
                    "multimodal request requires a configured projector",
                ));
            }
            let runtime_sessions_before = runtime.session_stats();
            let lock_hold_timer = PhaseTimer::start();
            let prefill = runtime
                .prefill_media(
                    &session_id,
                    &prompt.text,
                    &prompt.media,
                    sampling.enabled.then_some(&sampling),
                )
                .map_err(openai_backend_error)?;
            let token_signal = runtime.last_token_signal(&session_id).ok();
            let signal_window = runtime.signal_window(&session_id, 16).ok();
            let runtime_sessions_after = runtime.session_stats();
            let runtime_lock_hold_ms = lock_hold_timer.elapsed_ms();
            let mut attrs = self.openai_attrs(&ids);
            attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill.token_count),
            );
            attrs.insert(
                "llama_stage.prefill_position".to_string(),
                json!(prefill.position),
            );
            attrs.insert(
                "llama_stage.media_item_count".to_string(),
                json!(prompt.media.len()),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_before",
                runtime_sessions_before,
            );
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_after",
                runtime_sessions_after,
            );
            self.emit_openai_phase("stage.openai_media_prefill", prefill_timer, attrs);
            (prefill, token_signal, signal_window)
        };
        let max_tokens = max_tokens.resolve(prefill.position as usize, self.ctx_size)?;

        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk);
        let result = (|| {
            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut current = prefill.first_token;
            let mut runtime_lock_wait_ms = 0.0;
            let mut runtime_lock_wait_max_ms = 0.0_f64;
            let mut runtime_lock_hold_ms = 0.0;
            let mut runtime_lock_hold_max_ms = 0.0_f64;
            let mut runtime_lock_acquires = 0usize;
            let mut runtime_sessions_before = None;
            let mut runtime_sessions_after = None;
            let mut hook_request = hook_request;
            let hook_runtime = hook_runtime;
            let mut post_prefill_hook_checked = false;
            let mut last_mid_generation_hook_at = None;

            while decoded_tokens < max_tokens as usize {
                if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
                    break;
                }
                if let Some(injected_current) = self.maybe_run_generation_hooks(
                    &session_id,
                    &mut hook_request,
                    hook_runtime.as_ref(),
                    decoded_tokens,
                    &mut post_prefill_hook_checked,
                    &mut last_mid_generation_hook_at,
                    token_signal.take(),
                    signal_window.take(),
                )? {
                    current = injected_current;
                    continue;
                }
                if collector.push_token(current)? == TokenControl::Stop {
                    decoded_tokens += 1;
                    break;
                }
                decoded_tokens += 1;
                if decoded_tokens >= max_tokens as usize {
                    break;
                }

                let token_timer = PhaseTimer::start();
                let token_runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms;
                let token_signal_next;
                let signal_window_next;
                let decode_step = decoded_tokens;
                current = {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    let lock_wait_ms = lock_timer.elapsed_ms();
                    token_runtime_lock_wait_ms = lock_wait_ms;
                    runtime_lock_wait_ms += lock_wait_ms;
                    runtime_lock_wait_max_ms = runtime_lock_wait_max_ms.max(lock_wait_ms);
                    runtime_lock_acquires += 1;
                    let hold_timer = PhaseTimer::start();
                    runtime_sessions_before.get_or_insert_with(|| runtime.session_stats());
                    let predicted = runtime
                        .decode_sampled(&session_id, current, sampling.enabled.then_some(&sampling))
                        .map_err(openai_backend_error)?;
                    token_signal_next = runtime.last_token_signal(&session_id).ok();
                    signal_window_next = runtime.signal_window(&session_id, 16).ok();
                    runtime_sessions_after = Some(runtime.session_stats());
                    token_runtime_lock_hold_ms = hold_timer.elapsed_ms();
                    runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                    runtime_lock_hold_max_ms =
                        runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                    predicted
                };
                token_signal = token_signal_next;
                signal_window = signal_window_next;
                let mut token_attrs = self.openai_attrs(&ids);
                token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(token_timer.elapsed_ms()),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(token_runtime_lock_wait_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(token_runtime_lock_hold_ms),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeToken"));
                self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
            }
            let mut attrs = self.openai_attrs(&ids);
            attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(runtime_lock_wait_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(runtime_lock_hold_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            if let Some(stats) = runtime_sessions_before {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = runtime_sessions_after {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            self.emit_openai_phase("stage.openai_decode", decode_timer, attrs);
            Ok(())
        })();
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_id) {
                let mut attrs = self.openai_attrs(&ids);
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset_ms".to_string(),
                    json!(drop_stats.reset_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset".to_string(),
                    json!(drop_stats.reset_session),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    drop_stats.stats_after,
                );
                self.telemetry
                    .emit_debug("stage.openai_session_stop", attrs);
            }
        }
        result?;
        collector.finish(prefill.token_count)
    }

    fn generate_split_multimodal_text(
        &self,
        request: SplitMultimodalGeneration<'_>,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        let stop_values = request.stop.map(|stop| stop.values()).unwrap_or_default();
        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk);
        let wire_sampling = wire_sampling_config(&request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let mut lane = request.lane_pool.checkout(&request.ids)?;

        let mut prompt_tokens = 0usize;
        let result = (|| {
            let prefill_timer = PhaseTimer::start();
            let prefill = {
                let lock_timer = PhaseTimer::start();
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                let lock_wait_ms = lock_timer.elapsed_ms();
                if !runtime.has_media_projector() {
                    return Err(OpenAiError::invalid_request(
                        "multimodal request requires a configured projector",
                    ));
                }
                let runtime_sessions_before = runtime.session_stats();
                let lock_hold_timer = PhaseTimer::start();
                let prefill = runtime
                    .prefill_media_frame(&session_key, &request.prompt.text, &request.prompt.media)
                    .map_err(openai_backend_error)?;
                let runtime_sessions_after = runtime.session_stats();
                let runtime_lock_hold_ms = lock_hold_timer.elapsed_ms();
                let mut attrs = self.openai_attrs(&request.ids);
                attrs.insert(
                    "llama_stage.prefill_token_count".to_string(),
                    json!(prefill.token_count),
                );
                attrs.insert(
                    "llama_stage.prefill_position".to_string(),
                    json!(prefill.position),
                );
                attrs.insert(
                    "llama_stage.media_item_count".to_string(),
                    json!(request.prompt.media.len()),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(runtime_lock_hold_ms),
                );
                attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
                attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(prefill.output.payload.len()),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    runtime_sessions_before,
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    runtime_sessions_after,
                );
                self.emit_openai_phase("stage.openai_media_prefill", prefill_timer, attrs);
                prefill
            };
            prompt_tokens = prefill.token_count;
            let max_tokens = request
                .max_tokens
                .resolve(prefill.position as usize, self.ctx_size)?;

            if let Some(message) = generation_config_message(
                request.wire_dtype,
                request_id,
                session_id,
                prefill.token_count,
                wire_sampling.clone(),
                request.prompt.chat_parse_metadata.as_deref(),
            )? {
                write_stage_message_conditioned(
                    &mut lane.stream,
                    &message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::Ack {
                    return Err(OpenAiError::backend(format!(
                        "expected multimodal generation config ACK from downstream, got {:?}",
                        reply.kind
                    )));
                }
            }

            let final_prefill = multimodal_final_prefill_message(
                request.wire_dtype,
                MultimodalFinalPrefillArgs {
                    request_id,
                    session_id,
                    prompt_token_count: prefill.token_count,
                    sampling: wire_sampling.clone(),
                },
            )?;
            let forwarded = forwarded_stage_message_timed(
                &request.config,
                &final_prefill,
                &prefill.output,
                request.wire_dtype,
                request.activation_width,
            )
            .map_err(openai_backend_error)?;
            let write_timer = PhaseTimer::start();
            write_stage_message_conditioned(
                &mut lane.stream,
                &forwarded.message,
                request.wire_dtype,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
            let forward_write_ms = write_timer.elapsed_ms();
            let wait_timer = PhaseTimer::start();
            let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
            let downstream_wait_ms = wait_timer.elapsed_ms();
            if reply.kind != WireReplyKind::PredictedToken {
                return Err(OpenAiError::backend(format!(
                    "expected multimodal prefill predicted-token reply from downstream, got {:?}",
                    reply.kind
                )));
            }
            let mut attrs = self.openai_attrs(&request.ids);
            attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(forwarded.message.activation.len()),
            );
            attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(forwarded.activation_encode_ms),
            );
            attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(forward_write_ms),
            );
            attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(downstream_wait_ms),
            );
            self.emit_openai_phase("stage.openai_media_prefill_forward", write_timer, attrs);

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut current = reply.predicted;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_downstream_wait_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;

            while decoded_tokens < max_tokens as usize {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                if collector.push_token(current)? == TokenControl::Stop {
                    decoded_tokens += 1;
                    break;
                }
                decoded_tokens += 1;
                if decoded_tokens >= max_tokens as usize {
                    break;
                }

                let decode_input_index = decoded_tokens - 1;
                let message = embedded_decode_message(
                    request.wire_dtype,
                    DecodeMessageArgs {
                        request_id,
                        session_id,
                        prompt_token_count: prefill.token_count,
                        pos_start: prefill.token_count + decode_input_index,
                        decode_step: decode_input_index,
                        current,
                        sampling: wire_sampling.clone(),
                    },
                )?;
                let token_timer = PhaseTimer::start();
                let stage0_timer = PhaseTimer::start();
                let output = {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    let lock_wait_ms = lock_timer.elapsed_ms();
                    decode_runtime_lock_wait_ms += lock_wait_ms;
                    decode_runtime_lock_acquires += 1;
                    let hold_timer = PhaseTimer::start();
                    let output = run_binary_stage_message(
                        &mut runtime,
                        &session_key,
                        &message,
                        &[current],
                        None,
                        false,
                    )
                    .map_err(openai_backend_error)?
                    .2;
                    decode_runtime_lock_hold_ms += hold_timer.elapsed_ms();
                    output
                };
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                let forwarded = forwarded_stage_message_timed(
                    &request.config,
                    &message,
                    &output,
                    request.wire_dtype,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut lane.stream,
                    &forwarded.message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                if reply.kind != WireReplyKind::PredictedToken {
                    return Err(OpenAiError::backend(format!(
                        "expected multimodal decode predicted-token reply from downstream, got {:?}",
                        reply.kind
                    )));
                }
                current = reply.predicted;
                let mut token_attrs = self.openai_attrs(&request.ids);
                token_attrs.insert(
                    "llama_stage.decode_step".to_string(),
                    json!(decode_input_index),
                );
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(stage0_compute_ms),
                );
                token_attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(forward_write_ms),
                );
                token_attrs.insert(
                    "llama_stage.downstream_wait_ms".to_string(),
                    json!(downstream_wait_ms),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeEmbd"));
                self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
            }

            let mut decode_attrs = self.openai_attrs(&request.ids);
            decode_attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            decode_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(decode_stage0_compute_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(decode_runtime_lock_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(decode_runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(decode_runtime_lock_acquires),
            );
            decode_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(decode_forward_write_ms),
            );
            decode_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(decode_downstream_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(decode_output_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(decode_forward_activation_bytes),
            );
            self.emit_openai_phase("stage.openai_decode", decode_timer, decode_attrs);
            Ok(())
        })();

        let stop_result = write_stage_message(
            &mut lane.stream,
            &StageWireMessage::stop_with_identity(request.wire_dtype, request_id, session_id),
            request.wire_dtype,
        )
        .and_then(|_| recv_reply(&mut lane.stream).map(|reply| reply.kind))
        .and_then(|kind| {
            if kind == WireReplyKind::Ack {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected stop ACK, got {kind:?}"),
                ))
            }
        });
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_key) {
                let mut attrs = self.openai_attrs(&request.ids);
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset_ms".to_string(),
                    json!(drop_stats.reset_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset".to_string(),
                    json!(drop_stats.reset_session),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    drop_stats.stats_after,
                );
                self.telemetry
                    .emit_debug("stage.openai_session_stop", attrs);
            }
        }
        let lane_id = lane.id;
        let stop_result = stop_result.map_err(openai_io_error);
        match (&result, &stop_result) {
            (Ok(_), Ok(_)) => request.lane_pool.return_lane(lane),
            _ => request.lane_pool.replace_lane(lane_id),
        }
        if result.is_ok() {
            stop_result?;
        }
        result?;
        collector.finish(prompt_tokens)
    }

    fn tokenize(&self, prompt: &str) -> OpenAiResult<Vec<i32>> {
        self.tokenize_with_options(prompt, true)
    }

    fn tokenize_continuation(&self, text: &str) -> OpenAiResult<Vec<i32>> {
        self.tokenize_with_options(text, false)
    }

    fn tokenize_with_options(&self, text: &str, add_special: bool) -> OpenAiResult<Vec<i32>> {
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        runtime
            .model
            .tokenize(text, add_special)
            .map_err(openai_backend_error)
    }

    fn inject_hook_text_into_session(
        &self,
        session_id: &str,
        text: &str,
    ) -> OpenAiResult<Option<i32>> {
        let token_ids = self.tokenize_continuation(text)?;
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() > 1 {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .prefill(session_id, &token_ids[..token_ids.len() - 1])
                .map_err(openai_backend_error)?;
        }
        Ok(token_ids.last().copied())
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_run_generation_hooks(
        &self,
        session_id: &str,
        hook_request: &mut Option<ChatCompletionRequest>,
        hook_runtime: Option<&tokio::runtime::Handle>,
        decoded_tokens: usize,
        post_prefill_hook_checked: &mut bool,
        last_mid_generation_hook_at: &mut Option<usize>,
        token_signal: Option<TokenSignal>,
        signal_window: Option<GenerationSignalWindow>,
    ) -> OpenAiResult<Option<i32>> {
        let Some(hooks) = self.hook_policy.as_ref() else {
            return Ok(None);
        };
        let Some(handle) = hook_runtime else {
            return Ok(None);
        };
        let Some(request) = hook_request.as_mut() else {
            return Ok(None);
        };
        if !chat_mesh_hooks_enabled(request) {
            return Ok(None);
        }

        if !*post_prefill_hook_checked {
            *post_prefill_hook_checked = true;
            if let Some(signal) = token_signal {
                let signals = PrefillHookSignals {
                    first_token_entropy: f64::from(signal.entropy),
                    first_token_margin: f64::from(signal.margin),
                };
                let outcome = handle.block_on(hooks.after_prefill(request, signals))?;
                apply_chat_hook_outcome(request, &outcome);
                if let Some(text) = hook_injected_text(&outcome) {
                    return self.inject_hook_text_into_session(session_id, &text);
                }
            }
        }

        let Some(window) = signal_window else {
            return Ok(None);
        };
        if !mid_generation_window_should_fire(decoded_tokens, last_mid_generation_hook_at, &window)
        {
            return Ok(None);
        }

        let signals = GenerationHookSignals {
            n_decoded: i64::try_from(decoded_tokens).unwrap_or(i64::MAX),
            window_tokens: window.token_count,
            mean_entropy: f64::from(window.mean_entropy),
            max_entropy: f64::from(window.max_entropy),
            mean_margin: f64::from(window.mean_margin),
            min_margin: f64::from(window.min_margin),
            high_entropy_count: window.high_entropy_count,
            repetition_count: window.repetition_count,
        };
        let outcome = handle.block_on(hooks.mid_generation(request, signals))?;
        *last_mid_generation_hook_at = Some(decoded_tokens);
        apply_chat_hook_outcome(request, &outcome);
        if let Some(text) = hook_injected_text(&outcome) {
            return self.inject_hook_text_into_session(session_id, &text);
        }
        Ok(None)
    }

    fn generate_local_tokens(
        &self,
        request: LocalGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<()> {
        let session_id = request.ids.session_label.clone();
        let result = (|| {
            if request.prompt_token_ids.len() > 1 {
                let prefill_timer = PhaseTimer::start();
                let prefill_tokens =
                    &request.prompt_token_ids[..request.prompt_token_ids.len() - 1];
                let mut restored_prefill = false;
                let mut resident_recorded_pages = 0usize;
                let lock_timer = PhaseTimer::start();
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                let runtime_lock_wait_ms = lock_timer.elapsed_ms();
                let runtime_lock_hold_timer = PhaseTimer::start();
                let runtime_sessions_before = runtime.session_stats();
                if let Some(kv) = self.kv.as_ref() {
                    let base = self.local_kv_message_base(&session_id, request.ids);
                    let identities = kv.lookup_identities(&self.config, &base, 0, prefill_tokens);
                    match kv.restore_resident_prefix(
                        &mut runtime,
                        &session_id,
                        &identities,
                        prefill_tokens,
                    ) {
                        Ok(Some(restored)) => {
                            restored_prefill = true;
                            let mut attrs = self.openai_attrs(request.ids);
                            attrs.insert("skippy.kv.decision".to_string(), json!("resident_hit"));
                            attrs.insert(
                                "skippy.kv.hit_page_id".to_string(),
                                json!(restored.page_id),
                            );
                            attrs.insert(
                                "skippy.kv.restored_tokens".to_string(),
                                json!(restored.token_count),
                            );
                            attrs.insert(
                                "skippy.kv.resident_seq_id".to_string(),
                                json!(restored.seq_id),
                            );
                            self.telemetry
                                .emit("stage.openai_kv_lookup_decision", attrs);
                        }
                        Ok(None) => {
                            self.telemetry.emit(
                                "stage.openai_kv_lookup_decision",
                                BTreeMap::from([
                                    ("skippy.kv.decision".to_string(), json!("miss")),
                                    (
                                        "llama_stage.request_id".to_string(),
                                        json!(request.ids.request_id_string()),
                                    ),
                                ]),
                            );
                        }
                        Err(error) => {
                            let mut attrs = self.openai_attrs(request.ids);
                            attrs.insert("skippy.kv.decision".to_string(), json!("resident_error"));
                            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                            self.telemetry
                                .emit("stage.openai_kv_lookup_decision", attrs);
                        }
                    }
                }
                if !restored_prefill {
                    runtime
                        .prefill(&session_id, prefill_tokens)
                        .map_err(openai_backend_error)?;
                    if let Some(kv) = self.kv.as_ref() {
                        let base = self.local_kv_message_base(&session_id, request.ids);
                        for identity in kv.record_identities(&self.config, &base, 0, prefill_tokens)
                        {
                            if let Ok(Some(record)) = kv.record_resident_prefix(
                                &mut runtime,
                                &session_id,
                                &identity,
                                prefill_tokens,
                            ) {
                                resident_recorded_pages = resident_recorded_pages.saturating_add(1);
                                let mut attrs = self.openai_attrs(request.ids);
                                attrs.insert(
                                    "skippy.kv.recorded_page_id".to_string(),
                                    json!(record.page_id),
                                );
                                attrs.insert(
                                    "skippy.kv.recorded_tokens".to_string(),
                                    json!(record.token_count),
                                );
                                attrs.insert(
                                    "skippy.kv.resident_seq_id".to_string(),
                                    json!(record.seq_id),
                                );
                                attrs.insert(
                                    "skippy.kv.resident_entries".to_string(),
                                    json!(record.entries),
                                );
                                attrs.insert(
                                    "skippy.kv.evicted_entries".to_string(),
                                    json!(record.evicted_entries),
                                );
                                self.telemetry
                                    .emit("stage.openai_kv_record_decision", attrs);
                            }
                        }
                    }
                }
                let runtime_sessions_after = runtime.session_stats();
                let runtime_lock_hold_ms = runtime_lock_hold_timer.elapsed_ms();
                let mut attrs = self.openai_attrs(&request.ids);
                attrs.insert(
                    "llama_stage.prefill_token_count".to_string(),
                    json!(prefill_tokens.len()),
                );
                attrs.insert("llama_stage.prefill_chunk_count".to_string(), json!(1));
                attrs.insert(
                    "skippy.kv.restored_prefill".to_string(),
                    json!(restored_prefill),
                );
                attrs.insert(
                    "skippy.kv.recorded_pages".to_string(),
                    json!(resident_recorded_pages),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(runtime_lock_hold_ms),
                );
                attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    runtime_sessions_before,
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    runtime_sessions_after,
                );
                self.emit_openai_phase("stage.openai_prefill", prefill_timer, attrs);
            }
            if let Some(metadata) = request.chat_sampling_metadata {
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                runtime
                    .configure_chat_sampling(
                        &session_id,
                        metadata,
                        request.prompt_token_ids.len() as u64,
                        request.sampling.enabled.then_some(request.sampling),
                    )
                    .map_err(openai_backend_error)?;
            }
            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut runtime_lock_wait_ms = 0.0;
            let mut runtime_lock_wait_max_ms = 0.0_f64;
            let mut runtime_lock_hold_ms = 0.0;
            let mut runtime_lock_hold_max_ms = 0.0_f64;
            let mut runtime_lock_acquires = 0usize;
            let mut runtime_sessions_before = None;
            let mut runtime_sessions_after = None;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            let mut hook_request = request.hook_request;
            let hook_runtime = request.hook_runtime;
            let mut post_prefill_hook_checked = false;
            let mut last_mid_generation_hook_at = None;
            while decoded_tokens < request.max_tokens as usize {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let decode_step = decoded_tokens;
                let token_timer = PhaseTimer::start();
                let token_runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms;
                let token_signal;
                let signal_window;
                current = {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    let lock_wait_ms = lock_timer.elapsed_ms();
                    token_runtime_lock_wait_ms = lock_wait_ms;
                    runtime_lock_wait_ms += lock_wait_ms;
                    runtime_lock_wait_max_ms = runtime_lock_wait_max_ms.max(lock_wait_ms);
                    runtime_lock_acquires += 1;
                    let hold_timer = PhaseTimer::start();
                    runtime_sessions_before.get_or_insert_with(|| runtime.session_stats());
                    let predicted = runtime
                        .decode_sampled(
                            &session_id,
                            current,
                            request.sampling.enabled.then_some(request.sampling),
                        )
                        .map_err(openai_backend_error)?;
                    token_signal = runtime.last_token_signal(&session_id).ok();
                    signal_window = runtime.signal_window(&session_id, 16).ok();
                    runtime_sessions_after = Some(runtime.session_stats());
                    token_runtime_lock_hold_ms = hold_timer.elapsed_ms();
                    runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                    runtime_lock_hold_max_ms =
                        runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                    predicted
                };
                if let Some(injected_current) = self.maybe_run_generation_hooks(
                    &session_id,
                    &mut hook_request,
                    hook_runtime.as_ref(),
                    decoded_tokens,
                    &mut post_prefill_hook_checked,
                    &mut last_mid_generation_hook_at,
                    token_signal,
                    signal_window,
                )? {
                    current = injected_current;
                    continue;
                }
                decoded_tokens += 1;
                let mut token_attrs = self.openai_attrs(&request.ids);
                token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                token_attrs.insert(
                    "llama_stage.decode_token_phase".to_string(),
                    json!(decode_token_phase(
                        u32::try_from(decode_step).unwrap_or(u32::MAX)
                    )),
                );
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(token_timer.elapsed_ms()),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(token_runtime_lock_wait_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(token_runtime_lock_hold_ms),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeToken"));
                self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            let mut attrs = self.openai_attrs(&request.ids);
            attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(runtime_lock_wait_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(runtime_lock_hold_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            if let Some(stats) = runtime_sessions_before {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = runtime_sessions_after {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            self.emit_openai_phase("stage.openai_decode", decode_timer, attrs);
            Ok(())
        })();
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_id) {
                let mut attrs = self.openai_attrs(&request.ids);
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset_ms".to_string(),
                    json!(drop_stats.reset_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset".to_string(),
                    json!(drop_stats.reset_session),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    drop_stats.stats_after,
                );
                self.telemetry
                    .emit_debug("stage.openai_session_stop", attrs);
            }
        }
        result
    }

    fn generate_binary_chain_tokens(
        &self,
        request: BinaryChainGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<()> {
        let wire_sampling = wire_sampling_config(request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let connect_timer = PhaseTimer::start();
        let mut stream =
            connect_endpoint_ready(request.first_stage_addr, request.startup_timeout_secs)
                .map_err(openai_backend_error)?;
        let mut connect_attrs = self.openai_attrs(&request.ids);
        connect_attrs.insert(
            "llama_stage.first_stage_addr".to_string(),
            json!(request.first_stage_addr),
        );
        self.emit_openai_phase(
            "stage.openai_downstream_connect",
            connect_timer,
            connect_attrs,
        );
        let result = (|| {
            let prefill_token_count = request.prompt_token_ids.len().saturating_sub(1);
            let prefill_timer = PhaseTimer::start();
            let mut prefill_chunks = 0usize;
            let mut prefill_min_chunk_size = usize::MAX;
            let mut prefill_max_chunk_size = 0usize;
            let mut prefill_planner = request.prefill_chunk_policy.planner();
            if prefill_token_count > 0 {
                let prefill_tokens = &request.prompt_token_ids[..prefill_token_count];
                let mut pos_start = 0usize;
                let mut chunk_index = 0usize;
                while pos_start < prefill_tokens.len() {
                    if request
                        .cancellation
                        .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                    {
                        return Ok(());
                    }
                    let chunk_size = prefill_planner.chunk_size_for(chunk_index);
                    let end = pos_start
                        .saturating_add(chunk_size)
                        .min(prefill_tokens.len());
                    let chunk = &prefill_tokens[pos_start..end];
                    prefill_min_chunk_size = prefill_min_chunk_size.min(chunk.len());
                    prefill_max_chunk_size = prefill_max_chunk_size.max(chunk.len());
                    send_prefill_chunk(
                        &mut stream,
                        request.wire_dtype,
                        OpenAiPrefillChunk {
                            seq_id: chunk_index,
                            pos_start,
                            prefill_token_count,
                            tokens: chunk,
                            request_id,
                            session_id,
                        },
                    )
                    .map_err(openai_backend_error)?;
                    prefill_planner.advance_without_observation();
                    prefill_chunks += 1;
                    pos_start = end;
                    chunk_index += 1;
                }
            }
            let mut prefill_attrs = self.openai_attrs(&request.ids);
            prefill_attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill_token_count),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunks),
            );
            attrs_insert_prefill_chunk_policy(
                &mut prefill_attrs,
                request.prefill_chunk_policy,
                prefill_min_chunk_size,
                prefill_max_chunk_size,
            );
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, prefill_attrs);

            if let Some(message) = generation_config_message(
                request.wire_dtype,
                request_id,
                session_id,
                request.prompt_token_ids.len(),
                wire_sampling.clone(),
                request.chat_sampling_metadata,
            )? {
                write_stage_message(&mut stream, &message, request.wire_dtype)
                    .map_err(openai_io_error)?;
                let reply = recv_reply(&mut stream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::Ack {
                    return Err(OpenAiError::backend(format!(
                        "expected generation config ACK, got {:?}",
                        reply.kind
                    )));
                }
            }

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            for decode_step in 0..request.max_tokens {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let mut state =
                    StageStateHeader::new(WireMessageKind::DecodeEmbd, request.wire_dtype);
                state.seq_id = 0;
                state.prompt_token_count = i32::try_from(request.prompt_token_ids.len())
                    .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
                state.decode_step = i32::try_from(decode_step)
                    .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
                state.current_token = current;
                state.source_stage_index = -1;
                let message = StageWireMessage {
                    kind: WireMessageKind::DecodeEmbd,
                    pos_start: i32::try_from(prefill_token_count + decode_step as usize)
                        .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
                    token_count: 1,
                    state,
                    request_id,
                    session_id,
                    sampling: wire_sampling.clone(),
                    chat_sampling_metadata: None,
                    tokens: vec![current],
                    activation: Vec::new(),
                    raw_bytes: Vec::new(),
                };
                write_stage_message(&mut stream, &message, request.wire_dtype)
                    .map_err(openai_io_error)?;
                let reply = recv_reply(&mut stream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::PredictedToken {
                    return Err(OpenAiError::backend(format!(
                        "expected predicted-token reply, got {:?}",
                        reply.kind
                    )));
                }
                current = reply.predicted;
                decoded_tokens += 1;
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            let mut decode_attrs = self.openai_attrs(&request.ids);
            decode_attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            self.emit_openai_phase("stage.openai_decode", decode_timer, decode_attrs);
            Ok(())
        })();
        let stop_result = write_stage_message(
            &mut stream,
            &StageWireMessage::stop_with_identity(request.wire_dtype, request_id, session_id),
            request.wire_dtype,
        )
        .and_then(|_| recv_reply(&mut stream).map(|reply| reply.kind))
        .and_then(|kind| {
            if kind == WireReplyKind::Ack {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected stop ACK, got {kind:?}"),
                ))
            }
        });
        if result.is_ok() {
            stop_result.map_err(openai_io_error)?;
        }
        result
    }

    fn generate_embedded_stage_zero_tokens(
        &self,
        request: EmbeddedStageZeroGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<()> {
        if request.config.downstream.is_none() {
            return self.generate_local_tokens(
                LocalGeneration {
                    prompt_token_ids: request.prompt_token_ids,
                    max_tokens: request.max_tokens,
                    sampling: request.sampling,
                    chat_sampling_metadata: request.chat_sampling_metadata,
                    hook_request: request.hook_request,
                    hook_runtime: request.hook_runtime,
                    cancellation: request.cancellation,
                    ids: request.ids,
                },
                on_token,
            );
        }

        let wire_sampling = wire_sampling_config(request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let lane_pool = request
            .lane_pool
            .as_ref()
            .ok_or_else(|| OpenAiError::backend("embedded stage 0 has no downstream lane pool"))?;
        let mut lane = lane_pool.checkout(&request.ids)?;

        let result = (|| {
            let downstream = &mut lane.stream;
            let prefill_token_count = request.prompt_token_ids.len().saturating_sub(1);
            let prefill_timer = PhaseTimer::start();
            let mut prefill_chunks = 0usize;
            let mut prefill_min_chunk_size = usize::MAX;
            let mut prefill_max_chunk_size = 0usize;
            let mut prefill_stage0_compute_ms = 0.0;
            let mut prefill_runtime_lock_wait_ms = 0.0;
            let mut prefill_runtime_lock_wait_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_hold_ms = 0.0;
            let mut prefill_runtime_lock_hold_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_acquires = 0usize;
            let mut prefill_runtime_sessions_before = None;
            let mut prefill_runtime_sessions_after = None;
            let mut prefill_forward_write_ms = 0.0;
            let mut prefill_output_activation_bytes = 0usize;
            let mut prefill_forward_activation_bytes = 0usize;
            let mut prefill_downstream_wait_ms = 0.0;
            let mut prefill_planner = request.prefill_chunk_policy.planner();
            if prefill_token_count > 0 {
                let prefill_tokens = &request.prompt_token_ids[..prefill_token_count];
                let mut pos_start = 0usize;
                let mut chunk_index = 0usize;
                while pos_start < prefill_tokens.len() {
                    if request
                        .cancellation
                        .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                    {
                        return Ok(());
                    }
                    let chunk_size = prefill_planner.chunk_size_for(chunk_index);
                    let end = pos_start
                        .saturating_add(chunk_size)
                        .min(prefill_tokens.len());
                    let chunk = &prefill_tokens[pos_start..end];
                    prefill_min_chunk_size = prefill_min_chunk_size.min(chunk.len());
                    prefill_max_chunk_size = prefill_max_chunk_size.max(chunk.len());
                    let message = embedded_prefill_message(
                        request.wire_dtype,
                        OpenAiPrefillChunk {
                            seq_id: chunk_index,
                            pos_start,
                            prefill_token_count,
                            tokens: chunk,
                            request_id,
                            session_id,
                        },
                    )?;
                    let stage0_timer = PhaseTimer::start();
                    let output = {
                        let lock_timer = PhaseTimer::start();
                        let mut runtime = self
                            .runtime
                            .lock()
                            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                        let lock_wait_ms = lock_timer.elapsed_ms();
                        prefill_runtime_lock_wait_ms += lock_wait_ms;
                        prefill_runtime_lock_wait_max_ms =
                            prefill_runtime_lock_wait_max_ms.max(lock_wait_ms);
                        prefill_runtime_lock_acquires += 1;
                        let lock_hold_timer = PhaseTimer::start();
                        prefill_runtime_sessions_before
                            .get_or_insert_with(|| runtime.session_stats());
                        let output = run_binary_stage_message(
                            &mut runtime,
                            &session_key,
                            &message,
                            chunk,
                            None,
                            false,
                        )
                        .map_err(openai_backend_error)?
                        .2;
                        prefill_runtime_sessions_after = Some(runtime.session_stats());
                        let lock_hold_ms = lock_hold_timer.elapsed_ms();
                        prefill_runtime_lock_hold_ms += lock_hold_ms;
                        prefill_runtime_lock_hold_max_ms =
                            prefill_runtime_lock_hold_max_ms.max(lock_hold_ms);
                        output
                    };
                    let chunk_stage0_compute_ms = stage0_timer.elapsed_ms();
                    prefill_stage0_compute_ms += chunk_stage0_compute_ms;
                    let forwarded = forwarded_stage_message(
                        request.config,
                        &message,
                        &output,
                        request.wire_dtype,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?;
                    prefill_output_activation_bytes =
                        prefill_output_activation_bytes.saturating_add(output.payload.len());
                    prefill_forward_activation_bytes =
                        prefill_forward_activation_bytes.saturating_add(forwarded.activation.len());
                    let write_timer = PhaseTimer::start();
                    write_stage_message_conditioned(
                        &mut *downstream,
                        &forwarded,
                        request.wire_dtype,
                        request.downstream_wire_condition,
                    )
                    .map_err(openai_io_error)?;
                    let chunk_forward_write_ms = write_timer.elapsed_ms();
                    prefill_forward_write_ms += chunk_forward_write_ms;
                    let wait_timer = PhaseTimer::start();
                    let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                    let chunk_downstream_wait_ms = wait_timer.elapsed_ms();
                    prefill_downstream_wait_ms += chunk_downstream_wait_ms;
                    if reply.kind != WireReplyKind::Ack {
                        return Err(OpenAiError::backend(format!(
                            "expected prefill ACK from downstream, got {:?}",
                            reply.kind
                        )));
                    }
                    prefill_planner.observe(PrefillChunkObservation {
                        compute_ms: chunk_stage0_compute_ms,
                        forward_write_ms: chunk_forward_write_ms,
                        downstream_wait_ms: chunk_downstream_wait_ms,
                    });
                    prefill_chunks += 1;
                    pos_start = end;
                    chunk_index += 1;
                }
            }
            let mut prefill_attrs = self.openai_attrs(&request.ids);
            prefill_attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill_token_count),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunks),
            );
            attrs_insert_prefill_chunk_policy(
                &mut prefill_attrs,
                request.prefill_chunk_policy,
                prefill_min_chunk_size,
                prefill_max_chunk_size,
            );
            prefill_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(prefill_stage0_compute_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(prefill_runtime_lock_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(prefill_runtime_lock_wait_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(prefill_runtime_lock_hold_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(prefill_runtime_lock_hold_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(prefill_runtime_lock_acquires),
            );
            if let Some(stats) = prefill_runtime_sessions_before {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = prefill_runtime_sessions_after {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            prefill_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(prefill_forward_write_ms),
            );
            prefill_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(prefill_output_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(prefill_forward_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(prefill_downstream_wait_ms),
            );
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, prefill_attrs);

            if let Some(message) = generation_config_message(
                request.wire_dtype,
                request_id,
                session_id,
                request.prompt_token_ids.len(),
                wire_sampling.clone(),
                request.chat_sampling_metadata,
            )? {
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::Ack {
                    return Err(OpenAiError::backend(format!(
                        "expected generation config ACK from downstream, got {:?}",
                        reply.kind
                    )));
                }
            }

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_wait_max_ms = 0.0_f64;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_hold_max_ms = 0.0_f64;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_runtime_sessions_before = None;
            let mut decode_runtime_sessions_after = None;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_forward_activation_encode_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;
            let mut decode_downstream_wait_ms = 0.0;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            let mut context_tokens = request.prompt_token_ids.to_vec();
            let max_speculative_window = request.speculative_window.max(1);
            let mut adaptive_window = if request.adaptive_speculative_window {
                max_speculative_window.min(4)
            } else {
                max_speculative_window
            };
            let mut speculative_stats = OpenAiSpeculativeStats {
                adaptive_window_start: adaptive_window,
                adaptive_window_final: adaptive_window,
                adaptive_window_max: max_speculative_window,
                adaptive_window_min: if request.draft.is_some() {
                    adaptive_window
                } else {
                    0
                },
                adaptive_window_max_seen: adaptive_window,
                adaptive_window_enabled: request.adaptive_speculative_window,
                ..OpenAiSpeculativeStats::default()
            };
            let mut draft_guard = match request.draft.as_ref() {
                Some(draft) if request.speculative_window > 0 => {
                    let draft_reset_timer = PhaseTimer::start();
                    let mut draft = draft
                        .lock()
                        .map_err(|_| OpenAiError::backend("draft model lock poisoned"))?;
                    draft
                        .reset_to_context(&context_tokens)
                        .map_err(openai_backend_error)?;
                    speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                    let mut attrs = self.openai_attrs(&request.ids);
                    attrs.insert(
                        "llama_stage.draft_model_path".to_string(),
                        json!(draft.path.display().to_string()),
                    );
                    attrs.insert(
                        "llama_stage.speculative_window".to_string(),
                        json!(draft.window),
                    );
                    attrs.insert(
                        "llama_stage.adaptive_speculative_window".to_string(),
                        json!(request.adaptive_speculative_window),
                    );
                    self.emit_openai_phase("stage.openai_draft_reset", draft_reset_timer, attrs);
                    Some(draft)
                }
                _ => None,
            };
            for decode_step in 0..request.max_tokens {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let token_timer = PhaseTimer::start();
                if draft_guard.is_some() {
                    let remaining = request.max_tokens as usize - decoded_tokens;
                    if remaining == 0 {
                        break;
                    }
                    let mut proposal_source = "none";
                    let proposal_limit = remaining.min(adaptive_window);
                    let propose_timer = PhaseTimer::start();
                    let mut draft_tokens = Vec::new();
                    if draft_tokens.is_empty() {
                        if let Some(draft) = draft_guard.as_deref_mut() {
                            let proposal_limit = proposal_limit.min(draft.window);
                            draft_tokens = draft
                                .propose(current, proposal_limit)
                                .map_err(openai_backend_error)?;
                            if !draft_tokens.is_empty() {
                                proposal_source = "draft-model";
                            }
                        }
                    }
                    let draft_propose_ms = propose_timer.elapsed_ms();
                    speculative_stats.draft_propose_ms += draft_propose_ms;
                    if !draft_tokens.is_empty() {
                        let verify_inputs = verify_inputs_for_proposals(current, &draft_tokens);
                        let message = embedded_verify_message(
                            request.wire_dtype,
                            VerifySpanMessageArgs {
                                request_id,
                                session_id,
                                prompt_token_count: request.prompt_token_ids.len(),
                                pos_start: prefill_token_count + decoded_tokens,
                                decode_step: decoded_tokens,
                                tokens: &verify_inputs,
                                checkpoint: true,
                            },
                        )?;
                        let verify = self.execute_embedded_stage_message(
                            &request,
                            downstream,
                            &session_key,
                            &message,
                            &verify_inputs,
                            WireReplyKind::PredictedTokens,
                        )?;
                        speculative_stats.windows += 1;
                        speculative_stats.draft_tokens += draft_tokens.len();
                        speculative_stats.primary_verify_requests += 1;
                        speculative_stats.primary_verify_tokens += verify_inputs.len();
                        speculative_stats.primary_verify_elapsed_ms += verify.elapsed_ms;
                        speculative_stats.primary_verify_stage0_compute_ms +=
                            verify.stats.stage0_compute_ms;
                        speculative_stats.primary_verify_runtime_lock_wait_ms +=
                            verify.stats.runtime_lock_wait_ms;
                        speculative_stats.primary_verify_runtime_lock_hold_ms +=
                            verify.stats.runtime_lock_hold_ms;
                        speculative_stats.primary_verify_activation_encode_ms +=
                            verify.stats.activation_encode_ms;
                        speculative_stats.primary_verify_forward_write_ms +=
                            verify.stats.forward_write_ms;
                        speculative_stats.primary_verify_downstream_wait_ms +=
                            verify.stats.downstream_wait_ms;
                        speculative_stats.primary_verify_output_activation_bytes =
                            speculative_stats
                                .primary_verify_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                        speculative_stats.primary_verify_forward_activation_bytes =
                            speculative_stats
                                .primary_verify_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                        decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                        decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                        decode_runtime_lock_wait_max_ms =
                            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
                        decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                        decode_runtime_lock_hold_max_ms =
                            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
                        decode_runtime_lock_acquires += 1;
                        decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
                        decode_output_activation_bytes = decode_output_activation_bytes
                            .saturating_add(verify.stats.output_activation_bytes);
                        decode_forward_activation_bytes = decode_forward_activation_bytes
                            .saturating_add(verify.stats.forward_activation_bytes);
                        decode_forward_write_ms += verify.stats.forward_write_ms;
                        decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                        speculative_stats.checkpoint_ms +=
                            us_to_ms(verify.reply.stats.checkpoint_total_us);
                        let decision = classify_verify_span(
                            &draft_tokens,
                            &verify.reply.predicted_tokens,
                            decoded_tokens,
                            request.max_tokens as usize,
                            |token| token_is_eog_with_runtime(&self.runtime, token),
                        )?;
                        speculative_stats.observe_verify_decision(
                            decision,
                            &mut adaptive_window,
                            request.adaptive_speculative_window,
                            max_speculative_window,
                        );
                        let mut commit_tokens =
                            verify.reply.predicted_tokens[..decision.commit_count].to_vec();
                        if decision.requires_repair() {
                            speculative_stats.recovery_restores += 1;
                            let restore = self.restore_embedded_stage_session(
                                &request,
                                downstream,
                                &session_key,
                                request_id,
                                session_id,
                            )?;
                            speculative_stats.recovery_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_local_ms += restore.local_ms;
                            speculative_stats.recovery_restore_downstream_write_ms +=
                                restore.downstream_write_ms;
                            speculative_stats.recovery_restore_downstream_wait_ms +=
                                restore.downstream_wait_ms;
                            let repair_input_count = decision
                                .repair_input_count
                                .ok_or_else(|| OpenAiError::backend("missing repair count"))?;
                            if repair_input_count == 1 {
                                let repair_message = embedded_decode_message(
                                    request.wire_dtype,
                                    DecodeMessageArgs {
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        current,
                                        sampling: wire_sampling.clone(),
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    &[current],
                                    WireReplyKind::PredictedToken,
                                )?;
                                commit_tokens = vec![repair.reply.predicted];
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_decode_repairs += 1;
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_decode_elapsed_ms += repair.elapsed_ms;
                            } else {
                                let repair_inputs = &verify_inputs[..repair_input_count];
                                let repair_message = embedded_verify_message(
                                    request.wire_dtype,
                                    VerifySpanMessageArgs {
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        tokens: repair_inputs,
                                        checkpoint: false,
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    repair_inputs,
                                    WireReplyKind::PredictedTokens,
                                )?;
                                commit_tokens = repaired_commit_tokens(
                                    &draft_tokens,
                                    decision.accepted_before_reject,
                                    repair_input_count,
                                    &repair.reply.predicted_tokens,
                                )?;
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_reverify_tokens += repair_inputs.len();
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_reverify_elapsed_ms += repair.elapsed_ms;
                            }
                        }

                        let mut reached_stop = false;
                        for token in commit_tokens {
                            current = token;
                            decoded_tokens += 1;
                            context_tokens.push(current);
                            if on_token(current)? == TokenControl::Stop {
                                reached_stop = true;
                            }
                            if reached_stop || decoded_tokens >= request.max_tokens as usize {
                                break;
                            }
                        }
                        speculative_stats.adaptive_window_final = adaptive_window;
                        if proposal_source == "draft-model" && (decision.rejected() || reached_stop)
                        {
                            let draft_reset_timer = PhaseTimer::start();
                            if let Some(draft) = draft_guard.as_deref_mut() {
                                draft
                                    .reset_to_context(&context_tokens)
                                    .map_err(openai_backend_error)?;
                                speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                            }
                        }
                        let mut token_attrs = self.openai_attrs(&request.ids);
                        token_attrs
                            .insert("llama_stage.decode_step".to_string(), json!(decode_step));
                        token_attrs
                            .insert("llama_stage.message_kind".to_string(), json!("VerifySpan"));
                        token_attrs.insert(
                            "llama_stage.spec.windows".to_string(),
                            json!(speculative_stats.windows),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposed".to_string(),
                            json!(draft_tokens.len()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.accepted".to_string(),
                            json!(decision.accepted_before_reject),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.rejected".to_string(),
                            json!(decision.rejected()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.draft_propose_ms".to_string(),
                            json!(draft_propose_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_source".to_string(),
                            json!(proposal_source),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_limit".to_string(),
                            json!(proposal_limit),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(verify.stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(verify.stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(verify.stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(verify.stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(verify.stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(verify.stats.downstream_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(verify.stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(verify.stats.forward_activation_bytes),
                        );
                        self.emit_openai_phase(
                            "stage.openai_decode_verify_window",
                            token_timer,
                            token_attrs,
                        );
                        if reached_stop {
                            break;
                        }
                        continue;
                    }
                }
                let mut state =
                    StageStateHeader::new(WireMessageKind::DecodeEmbd, request.wire_dtype);
                state.seq_id = 0;
                state.prompt_token_count = i32::try_from(request.prompt_token_ids.len())
                    .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
                state.decode_step = i32::try_from(decode_step)
                    .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
                state.current_token = current;
                state.source_stage_index = -1;
                let message = StageWireMessage {
                    kind: WireMessageKind::DecodeEmbd,
                    pos_start: i32::try_from(prefill_token_count + decode_step as usize)
                        .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
                    token_count: 1,
                    state,
                    request_id,
                    session_id,
                    sampling: wire_sampling.clone(),
                    chat_sampling_metadata: None,
                    tokens: vec![current],
                    activation: Vec::new(),
                    raw_bytes: Vec::new(),
                };
                let stage0_timer = PhaseTimer::start();
                let token_runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms;
                let output = {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    let lock_wait_ms = lock_timer.elapsed_ms();
                    token_runtime_lock_wait_ms = lock_wait_ms;
                    decode_runtime_lock_wait_ms += lock_wait_ms;
                    decode_runtime_lock_wait_max_ms =
                        decode_runtime_lock_wait_max_ms.max(lock_wait_ms);
                    decode_runtime_lock_acquires += 1;
                    let lock_hold_timer = PhaseTimer::start();
                    decode_runtime_sessions_before.get_or_insert_with(|| runtime.session_stats());
                    let output = run_binary_stage_message(
                        &mut runtime,
                        &session_key,
                        &message,
                        &[current],
                        None,
                        false,
                    )
                    .map_err(openai_backend_error)?
                    .2;
                    decode_runtime_sessions_after = Some(runtime.session_stats());
                    token_runtime_lock_hold_ms = lock_hold_timer.elapsed_ms();
                    decode_runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                    decode_runtime_lock_hold_max_ms =
                        decode_runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                    output
                };
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                let forwarded = forwarded_stage_message_timed(
                    request.config,
                    &message,
                    &output,
                    request.wire_dtype,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                decode_forward_activation_encode_ms += forwarded.activation_encode_ms;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                if reply.kind != WireReplyKind::PredictedToken {
                    return Err(OpenAiError::backend(format!(
                        "expected predicted-token reply from downstream, got {:?}",
                        reply.kind
                    )));
                }
                current = reply.predicted;
                decoded_tokens += 1;
                context_tokens.push(current);
                let mut token_attrs = self.openai_attrs(&request.ids);
                token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                token_attrs.insert(
                    "llama_stage.decode_token_phase".to_string(),
                    json!(decode_token_phase(decode_step)),
                );
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(stage0_compute_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(token_runtime_lock_wait_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(token_runtime_lock_hold_ms),
                );
                token_attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(output.payload.len()),
                );
                token_attrs.insert(
                    "llama_stage.forward_activation_bytes".to_string(),
                    json!(forwarded.message.activation.len()),
                );
                token_attrs.insert(
                    "llama_stage.activation_encode_ms".to_string(),
                    json!(forwarded.activation_encode_ms),
                );
                token_attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(forward_write_ms),
                );
                token_attrs.insert(
                    "llama_stage.downstream_wait_ms".to_string(),
                    json!(downstream_wait_ms),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeEmbd"));
                self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            let mut decode_attrs = self.openai_attrs(&request.ids);
            decode_attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            decode_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(decode_stage0_compute_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(decode_runtime_lock_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(decode_runtime_lock_wait_max_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(decode_runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(decode_runtime_lock_hold_max_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(decode_runtime_lock_acquires),
            );
            if let Some(stats) = decode_runtime_sessions_before {
                Self::insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = decode_runtime_sessions_after {
                Self::insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            decode_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(decode_forward_write_ms),
            );
            decode_attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(decode_forward_activation_encode_ms),
            );
            decode_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(decode_output_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(decode_forward_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(decode_downstream_wait_ms),
            );
            speculative_stats.insert_attrs(&mut decode_attrs);
            self.emit_openai_phase("stage.openai_decode", decode_timer, decode_attrs);
            Ok(())
        })();

        let stop_result = write_stage_message(
            &mut lane.stream,
            &StageWireMessage::stop_with_identity(request.wire_dtype, request_id, session_id),
            request.wire_dtype,
        )
        .and_then(|_| recv_reply(&mut lane.stream).map(|reply| reply.kind))
        .and_then(|kind| {
            if kind == WireReplyKind::Ack {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected stop ACK, got {kind:?}"),
                ))
            }
        });
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_key) {
                let mut attrs = self.openai_attrs(&request.ids);
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset_ms".to_string(),
                    json!(drop_stats.reset_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset".to_string(),
                    json!(drop_stats.reset_session),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    drop_stats.stats_after,
                );
                self.telemetry
                    .emit_debug("stage.openai_session_stop", attrs);
            }
        }
        let lane_id = lane.id;
        let stop_result = stop_result.map_err(openai_io_error);
        match (&result, &stop_result) {
            (Ok(_), Ok(_)) => lane_pool.return_lane(lane),
            _ => lane_pool.replace_lane(lane_id),
        }
        if result.is_ok() {
            stop_result?;
        }
        result
    }

    fn execute_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let timer = PhaseTimer::start();
        let mut stats = StageReplyStats::default();
        let stage0_timer = PhaseTimer::start();
        let output = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            let hold_timer = PhaseTimer::start();
            if message.kind == WireMessageKind::VerifySpan
                && (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0
            {
                let checkpoint_timer = PhaseTimer::start();
                runtime
                    .checkpoint_session(session_key)
                    .map_err(openai_backend_error)?;
                let checkpoint_us = ms_to_us(checkpoint_timer.elapsed_ms());
                stats.checkpoint_local_us += checkpoint_us;
                stats.checkpoint_total_us += checkpoint_us;
                stats.verify_span_checkpointed_requests += 1;
            } else if message.kind == WireMessageKind::VerifySpan {
                stats.verify_span_skip_checkpoint_requests += 1;
            }
            let output = run_binary_stage_message(
                &mut runtime,
                session_key,
                message,
                token_ids,
                None,
                false,
            )
            .map_err(openai_backend_error)?
            .2;
            let hold_ms = hold_timer.elapsed_ms();
            EmbeddedLocalOutput {
                output,
                runtime_lock_wait_ms: lock_wait_ms,
                runtime_lock_hold_ms: hold_ms,
            }
        };
        let stage0_compute_ms = stage0_timer.elapsed_ms();
        let forwarded = forwarded_stage_message_timed(
            request.config,
            message,
            &output.output,
            request.wire_dtype,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &forwarded.message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let forward_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != expected_reply {
            return Err(OpenAiError::backend(format!(
                "expected {expected_reply:?} reply from downstream, got {:?}",
                reply.kind
            )));
        }
        stats.merge(reply.stats);
        if message.kind == WireMessageKind::VerifySpan {
            stats.verify_span_compute_us += ms_to_us(stage0_compute_ms);
            stats.verify_span_forward_write_us += ms_to_us(forward_write_ms);
            stats.verify_span_downstream_wait_us += ms_to_us(downstream_wait_ms);
            stats.verify_span_total_us += ms_to_us(timer.elapsed_ms());
            stats.verify_span_stage_count += 1;
            stats.verify_span_request_count += 1;
            stats.verify_span_token_count += i64::from(message.token_count.max(0));
            stats.verify_span_max_tokens = stats
                .verify_span_max_tokens
                .max(i64::from(message.token_count.max(0)));
        }
        Ok(EmbeddedStageExecution {
            reply: StageReply { stats, ..reply },
            stats: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: output.runtime_lock_wait_ms,
                runtime_lock_hold_ms: output.runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.output.payload.len(),
                forward_activation_bytes: forwarded.message.activation.len(),
                forward_write_ms,
                downstream_wait_ms,
            },
            elapsed_ms: timer.elapsed_ms(),
        })
    }

    fn restore_embedded_stage_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .restore_session(session_key)
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message = embedded_session_control_message(
            request.wire_dtype,
            WireMessageKind::RestoreSession,
            request_id,
            session_id,
        );
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "restore expected ACK from downstream, got {:?}",
                reply.kind
            )));
        }
        Ok(EmbeddedSessionControl {
            elapsed_ms: timer.elapsed_ms(),
            local_ms,
            downstream_write_ms,
            downstream_wait_ms,
        })
    }
}

fn ensure_requested_model(advertised_model_id: &str, requested: &str) -> OpenAiResult<()> {
    if requested == advertised_model_id {
        Ok(())
    } else {
        Err(OpenAiError::model_not_found(requested))
    }
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
    parsed_tool_calls: Option<ParsedToolCalls>,
) -> ChatCompletionResponse {
    if let Some(parsed) = parsed_tool_calls {
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
                    tool_calls: Some(parsed.tool_calls),
                },
                logprobs: None,
                finish_reason: Some(FinishReason::ToolCalls),
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

fn parsed_tool_calls_from_message_json(
    message_json: &str,
    request: &ChatCompletionRequest,
) -> Option<ParsedToolCalls> {
    let value = serde_json::from_str::<Value>(message_json).ok()?;
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
        content: value
            .get("content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
            .map(ToString::to_string),
        tool_calls: Value::Array(tool_calls),
    })
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

#[derive(Default)]
struct OpenAiSpeculativeStats {
    windows: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
    rejected_tokens: usize,
    full_accept_windows: usize,
    accepted_stop_windows: usize,
    rejected_windows: usize,
    early_reject_windows: usize,
    tail_reject_windows: usize,
    early_reject_stop_windows: usize,
    repair_required_windows: usize,
    first_reject_position_sum: usize,
    primary_verify_requests: usize,
    primary_verify_tokens: usize,
    primary_verify_elapsed_ms: f64,
    primary_verify_stage0_compute_ms: f64,
    primary_verify_runtime_lock_wait_ms: f64,
    primary_verify_runtime_lock_hold_ms: f64,
    primary_verify_activation_encode_ms: f64,
    primary_verify_forward_write_ms: f64,
    primary_verify_downstream_wait_ms: f64,
    primary_verify_output_activation_bytes: usize,
    primary_verify_forward_activation_bytes: usize,
    checkpoint_ms: f64,
    draft_reset_ms: f64,
    draft_propose_ms: f64,
    recovery_restores: usize,
    recovery_decode_repairs: usize,
    recovery_decode_elapsed_ms: f64,
    recovery_reverify_tokens: usize,
    recovery_ms: f64,
    recovery_restore_ms: f64,
    recovery_restore_local_ms: f64,
    recovery_restore_downstream_write_ms: f64,
    recovery_restore_downstream_wait_ms: f64,
    recovery_reverify_elapsed_ms: f64,
    adaptive_window_start: usize,
    adaptive_window_final: usize,
    adaptive_window_max: usize,
    adaptive_window_min: usize,
    adaptive_window_max_seen: usize,
    adaptive_window_sum: usize,
    adaptive_window_grows: usize,
    adaptive_window_shrinks: usize,
    adaptive_window_enabled: bool,
}

impl OpenAiSpeculativeStats {
    fn observe_verify_decision(
        &mut self,
        decision: VerifySpanDecision,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        self.accepted_tokens += decision.accepted_before_reject;
        if decision.rejected() {
            self.rejected_tokens += 1;
        }
        self.adaptive_window_sum += *adaptive_window;
        self.adaptive_window_min = nonzero_min(self.adaptive_window_min, *adaptive_window);
        self.adaptive_window_max_seen = self.adaptive_window_max_seen.max(*adaptive_window);
        match decision.kind {
            VerifySpanDecisionKind::FullAccept => {
                self.full_accept_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifySpanDecisionKind::AcceptedStop => {
                self.accepted_stop_windows += 1;
            }
            VerifySpanDecisionKind::TailReject => {
                self.observe_reject(decision);
                self.tail_reject_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifySpanDecisionKind::EarlyReject => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.repair_required_windows += 1;
                self.shrink_adaptive_window(adaptive_window, adaptive_enabled, decision);
            }
            VerifySpanDecisionKind::EarlyRejectStop => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.early_reject_stop_windows += 1;
            }
        }
    }

    fn observe_reject(&mut self, decision: VerifySpanDecision) {
        if let Some(repair_input_count) = decision.repair_input_count {
            self.rejected_windows += 1;
            self.first_reject_position_sum += repair_input_count;
        }
    }

    fn grow_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        if adaptive_enabled && *adaptive_window < max_speculative_window {
            *adaptive_window += 1;
            self.adaptive_window_grows += 1;
        }
    }

    fn shrink_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        decision: VerifySpanDecision,
    ) {
        if !adaptive_enabled {
            return;
        }
        let Some(repair_input_count) = decision.repair_input_count else {
            return;
        };
        let next_window = (*adaptive_window)
            .saturating_sub(1)
            .max(repair_input_count)
            .max(1);
        if next_window < *adaptive_window {
            *adaptive_window = next_window;
            self.adaptive_window_shrinks += 1;
        }
    }

    fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        if self.windows == 0 {
            attrs.insert("llama_stage.spec.enabled".to_string(), json!(false));
            return;
        }
        attrs.insert("llama_stage.spec.enabled".to_string(), json!(true));
        attrs.insert("llama_stage.spec.windows".to_string(), json!(self.windows));
        attrs.insert(
            "llama_stage.spec.proposed".to_string(),
            json!(self.draft_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accepted".to_string(),
            json!(self.accepted_tokens),
        );
        attrs.insert(
            "llama_stage.spec.rejected".to_string(),
            json!(self.rejected_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accept_rate".to_string(),
            json!(if self.draft_tokens == 0 {
                0.0
            } else {
                self.accepted_tokens as f64 / self.draft_tokens as f64
            }),
        );
        attrs.insert(
            "llama_stage.spec.full_accept_windows".to_string(),
            json!(self.full_accept_windows),
        );
        attrs.insert(
            "llama_stage.spec.accepted_stop_windows".to_string(),
            json!(self.accepted_stop_windows),
        );
        attrs.insert(
            "llama_stage.spec.rejected_windows".to_string(),
            json!(self.rejected_windows),
        );
        attrs.insert(
            "llama_stage.spec.early_reject_windows".to_string(),
            json!(self.early_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.tail_reject_windows".to_string(),
            json!(self.tail_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.repair_required_windows".to_string(),
            json!(self.repair_required_windows),
        );
        attrs.insert(
            "llama_stage.spec.draft_reset_ms".to_string(),
            json!(self.draft_reset_ms),
        );
        attrs.insert(
            "llama_stage.spec.draft_propose_ms".to_string(),
            json!(self.draft_propose_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_elapsed_ms".to_string(),
            json!(self.primary_verify_elapsed_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_stage0_compute_ms".to_string(),
            json!(self.primary_verify_stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_wait_ms".to_string(),
            json!(self.primary_verify_runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_hold_ms".to_string(),
            json!(self.primary_verify_runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_activation_encode_ms".to_string(),
            json!(self.primary_verify_activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_write_ms".to_string(),
            json!(self.primary_verify_forward_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_downstream_wait_ms".to_string(),
            json!(self.primary_verify_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_output_activation_bytes".to_string(),
            json!(self.primary_verify_output_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_activation_bytes".to_string(),
            json!(self.primary_verify_forward_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.checkpoint_ms".to_string(),
            json!(self.checkpoint_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restores".to_string(),
            json!(self.recovery_restores),
        );
        attrs.insert(
            "llama_stage.spec.recovery_ms".to_string(),
            json!(self.recovery_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_local_ms".to_string(),
            json!(self.recovery_restore_local_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_write_ms".to_string(),
            json!(self.recovery_restore_downstream_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_wait_ms".to_string(),
            json!(self.recovery_restore_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.adaptive_enabled".to_string(),
            json!(self.adaptive_window_enabled),
        );
        attrs.insert(
            "llama_stage.spec.window_start".to_string(),
            json!(self.adaptive_window_start),
        );
        attrs.insert(
            "llama_stage.spec.window_final".to_string(),
            json!(self.adaptive_window_final),
        );
        attrs.insert(
            "llama_stage.spec.window_max".to_string(),
            json!(self.adaptive_window_max),
        );
        attrs.insert(
            "llama_stage.spec.window_min".to_string(),
            json!(self.adaptive_window_min),
        );
        attrs.insert(
            "llama_stage.spec.window_max_seen".to_string(),
            json!(self.adaptive_window_max_seen),
        );
        attrs.insert(
            "llama_stage.spec.window_grows".to_string(),
            json!(self.adaptive_window_grows),
        );
        attrs.insert(
            "llama_stage.spec.window_shrinks".to_string(),
            json!(self.adaptive_window_shrinks),
        );
    }
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
    ToolCalls(Value),
    Usage(Usage),
    Done(FinishReason),
}

fn generation_event_to_chat_chunk(
    event: OpenAiResult<GenerationStreamEvent>,
    model: &str,
) -> OpenAiResult<ChatCompletionChunk> {
    match event? {
        GenerationStreamEvent::Delta(delta) => {
            Ok(ChatCompletionChunk::delta(model.to_string(), delta))
        }
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

    fn finish(mut self, prompt_token_count: usize) -> OpenAiResult<GeneratedText> {
        self.emit_safe_delta(true)?;
        Ok(GeneratedText {
            prompt_tokens: saturating_u32(prompt_token_count),
            completion_tokens: saturating_u32(self.completion_tokens),
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
    text: String,
    finish_reason: FinishReason,
    detokenize_ms: f64,
    text_emit_ms: f64,
    eog_check_ms: f64,
}

impl GeneratedText {
    fn usage(&self) -> Usage {
        Usage::new(self.prompt_tokens, self.completion_tokens)
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

struct DecodeMessageArgs {
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    pos_start: usize,
    decode_step: usize,
    current: i32,
    sampling: Option<WireSamplingConfig>,
}

fn embedded_decode_message(
    wire_dtype: WireActivationDType,
    args: DecodeMessageArgs,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    state.decode_step = i32::try_from(args.decode_step)
        .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
    state.current_token = args.current;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
        token_count: 1,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: None,
        tokens: vec![args.current],
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

struct VerifySpanMessageArgs<'a> {
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    pos_start: usize,
    decode_step: usize,
    tokens: &'a [i32],
    checkpoint: bool,
}

fn embedded_verify_message(
    wire_dtype: WireActivationDType,
    args: VerifySpanMessageArgs<'_>,
) -> OpenAiResult<StageWireMessage> {
    if args.tokens.is_empty() {
        return Err(OpenAiError::backend(
            "verify span requires at least one token",
        ));
    }
    let mut state = StageStateHeader::new(WireMessageKind::VerifySpan, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    state.decode_step = i32::try_from(args.decode_step)
        .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
    state.current_token = args.tokens[0];
    state.source_stage_index = -1;
    if !args.checkpoint {
        state.flags |= state_flags::SKIP_VERIFY_CHECKPOINT;
    }
    Ok(StageWireMessage {
        kind: WireMessageKind::VerifySpan,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("verify span position exceeds i32"))?,
        token_count: i32::try_from(args.tokens.len())
            .map_err(|_| OpenAiError::backend("verify span exceeds i32"))?,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: args.tokens.to_vec(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

fn embedded_session_control_message(
    wire_dtype: WireActivationDType,
    kind: WireMessageKind,
    request_id: u64,
    session_id: u64,
) -> StageWireMessage {
    StageWireMessage {
        kind,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(kind, wire_dtype),
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

fn verify_inputs_for_proposals(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len());
    if proposals.is_empty() {
        return tokens;
    }
    tokens.push(current);
    tokens.extend(proposals.iter().take(proposals.len().saturating_sub(1)));
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerifySpanDecisionKind {
    FullAccept,
    AcceptedStop,
    TailReject,
    EarlyReject,
    EarlyRejectStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifySpanDecision {
    kind: VerifySpanDecisionKind,
    accepted_before_reject: usize,
    repair_input_count: Option<usize>,
    commit_count: usize,
}

impl VerifySpanDecision {
    fn rejected(self) -> bool {
        matches!(
            self.kind,
            VerifySpanDecisionKind::TailReject
                | VerifySpanDecisionKind::EarlyReject
                | VerifySpanDecisionKind::EarlyRejectStop
        )
    }

    fn requires_repair(self) -> bool {
        self.kind == VerifySpanDecisionKind::EarlyReject
    }
}

fn classify_verify_span<F>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<VerifySpanDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    if predicted_tokens.len() < draft_tokens.len() {
        return Err(OpenAiError::backend(format!(
            "verify span returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            draft_tokens.len()
        )));
    }

    let mut accepted_before_reject = 0usize;
    let mut commit_count = 0usize;
    for (draft_token, predicted) in draft_tokens.iter().zip(predicted_tokens.iter()) {
        commit_count += 1;
        let accepted = *predicted == *draft_token;
        let reached_eog = token_is_eog(*predicted)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if accepted {
            accepted_before_reject += 1;
            if (reached_eog || reached_limit) && commit_count < draft_tokens.len() {
                return Ok(VerifySpanDecision {
                    kind: VerifySpanDecisionKind::AcceptedStop,
                    accepted_before_reject,
                    repair_input_count: None,
                    commit_count,
                });
            }
            continue;
        }

        let repair_input_count = accepted_before_reject + 1;
        let kind = if repair_input_count == draft_tokens.len() {
            VerifySpanDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifySpanDecisionKind::EarlyRejectStop
        } else {
            VerifySpanDecisionKind::EarlyReject
        };
        return Ok(VerifySpanDecision {
            kind,
            accepted_before_reject,
            repair_input_count: Some(repair_input_count),
            commit_count,
        });
    }

    Ok(VerifySpanDecision {
        kind: VerifySpanDecisionKind::FullAccept,
        accepted_before_reject,
        repair_input_count: None,
        commit_count,
    })
}

fn repaired_commit_tokens(
    draft_tokens: &[i32],
    accepted_before_reject: usize,
    repair_input_count: usize,
    repaired_predictions: &[i32],
) -> OpenAiResult<Vec<i32>> {
    if repaired_predictions.len() < repair_input_count {
        return Err(OpenAiError::backend(format!(
            "recovery verify returned too few tokens: expected {} got {:?}",
            repair_input_count, repaired_predictions
        )));
    }
    if accepted_before_reject > 0
        && repaired_predictions[..accepted_before_reject] != draft_tokens[..accepted_before_reject]
    {
        eprintln!(
            "recovery verify changed accepted prefix; committing restored target tokens: accepted {:?}, repaired {:?}",
            &draft_tokens[..accepted_before_reject],
            &repaired_predictions[..accepted_before_reject]
        );
    }
    Ok(repaired_predictions[..repair_input_count].to_vec())
}

fn nonzero_min(current: usize, candidate: usize) -> usize {
    if current == 0 {
        candidate
    } else {
        current.min(candidate)
    }
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

struct OpenAiPrefillChunk<'a> {
    seq_id: usize,
    pos_start: usize,
    prefill_token_count: usize,
    tokens: &'a [i32],
    request_id: u64,
    session_id: u64,
}

fn send_prefill_chunk(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    chunk: OpenAiPrefillChunk<'_>,
) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id = i32::try_from(chunk.seq_id).context("prefill seq exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(chunk.prefill_token_count).context("prefill token count exceeds i32")?;
    state.current_token = *chunk.tokens.last().context("prefill chunk is empty")?;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: i32::try_from(chunk.pos_start).context("prefill chunk position exceeds i32")?,
        token_count: i32::try_from(chunk.tokens.len())
            .context("prefill token count exceeds i32")?,
        state,
        request_id: chunk.request_id,
        session_id: chunk.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: chunk.tokens.to_vec(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype).context("send prefill chunk")?;
    let reply = recv_reply(&mut *stream).context("receive prefill chunk ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

fn generation_config_message(
    wire_dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    sampling: Option<WireSamplingConfig>,
    chat_sampling_metadata: Option<&str>,
) -> OpenAiResult<Option<StageWireMessage>> {
    let Some(metadata) = chat_sampling_metadata else {
        return Ok(None);
    };
    let prompt_token_count = i32::try_from(prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    Ok(Some(StageWireMessage::configure_generation(
        wire_dtype,
        request_id,
        session_id,
        prompt_token_count,
        sampling,
        Some(metadata.to_string()),
    )))
}

fn embedded_prefill_message(
    wire_dtype: WireActivationDType,
    chunk: OpenAiPrefillChunk<'_>,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id =
        i32::try_from(chunk.seq_id).map_err(|_| OpenAiError::backend("prefill seq exceeds i32"))?;
    state.prompt_token_count = i32::try_from(chunk.prefill_token_count)
        .map_err(|_| OpenAiError::backend("prefill token count exceeds i32"))?;
    state.current_token = *chunk
        .tokens
        .last()
        .ok_or_else(|| OpenAiError::backend("prefill chunk is empty"))?;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: i32::try_from(chunk.pos_start)
            .map_err(|_| OpenAiError::backend("prefill chunk position exceeds i32"))?,
        token_count: i32::try_from(chunk.tokens.len())
            .map_err(|_| OpenAiError::backend("prefill token count exceeds i32"))?,
        state,
        request_id: chunk.request_id,
        session_id: chunk.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: chunk.tokens.to_vec(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

struct MultimodalFinalPrefillArgs {
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    sampling: Option<WireSamplingConfig>,
}

fn multimodal_final_prefill_message(
    wire_dtype: WireActivationDType,
    args: MultimodalFinalPrefillArgs,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillFinalEmbd, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("multimodal prefill token count exceeds i32"))?;
    state.current_token = LLAMA_TOKEN_NULL;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind: WireMessageKind::PrefillFinalEmbd,
        pos_start: 0,
        token_count: i32::try_from(args.prompt_token_count)
            .map_err(|_| OpenAiError::backend("multimodal prefill token count exceeds i32"))?,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

#[cfg(test)]
mod tests;
