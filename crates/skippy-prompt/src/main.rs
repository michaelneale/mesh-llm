use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet, VecDeque},
    fs,
    hash::{Hash, Hasher},
    io::{self, BufRead, BufReader, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use openai_frontend::{normalize_reasoning_template_options, ReasoningConfig};
use rustyline::{error::ReadlineError, DefaultEditor};
use serde_json::{json, Value};
use skippy_protocol::binary::{
    recv_reply, state_flags, write_stage_message, StageReplyStats, StageStateHeader,
    StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind, READY_MAGIC,
};
use skippy_runtime::{
    package::{materialize_layer_package, PackageStageRequest},
    restore_native_logs, suppress_native_logs, ChatTemplateMessage, ChatTemplateOptions, ModelInfo,
    RuntimeConfig, RuntimeLoadMode, StageModel, StageSession, GGML_TYPE_F16,
};
use skippy_topology::{
    dense_attention_layers, infer_family_capability, plan_contiguous_with_splits, BoundaryDecision,
    NodeSpec, PlannerPolicy, TopologyPlanRequest, WireValidation,
};

const DEFAULT_MIN_WINNER_COUNT: u32 = 2;
const DEFAULT_MIN_CONFIDENCE: f32 = 0.55;
const DEFAULT_MIN_MARGIN: u32 = 1;
const DEFAULT_CONFIDENCE_STEP: f32 = 0.0;
const DEFAULT_CONFIDENCE_STEP_TOKENS: usize = usize::MAX;
const DEFAULT_MAX_CONFIDENCE: f32 = 0.95;
const DEFAULT_COUNT_STEP_TOKENS: usize = usize::MAX;
const DEFAULT_MARGIN_STEP_TOKENS: usize = usize::MAX;

#[derive(Parser)]
#[command(about = "Prompt CLI for skippy binary servers")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand)]
enum CommandKind {
    #[command(name = "prompt")]
    Prompt(Box<PromptArgs>),
    Binary(Box<BinaryReplArgs>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ReplLoadMode {
    RuntimeSlice,
    ArtifactSlice,
    LayerPackage,
}

impl From<ReplLoadMode> for RuntimeLoadMode {
    fn from(value: ReplLoadMode) -> Self {
        match value {
            ReplLoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
            ReplLoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
            ReplLoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum NgramProposalMode {
    TransitionPool,
    HistoryMatch,
}

#[derive(Parser)]
pub struct PromptArgs {
    #[arg(long, default_value = "target/debug/metrics-server")]
    pub metrics_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/kv-server")]
    pub kv_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/ngram-pool-server")]
    pub ngram_pool_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/llama-model-slice")]
    pub model_slice_bin: PathBuf,
    #[arg(long, value_delimiter = ',')]
    pub hosts: Vec<String>,
    #[arg(long, default_value = "/tmp/skippy-remote-prompt")]
    pub remote_root: String,
    #[arg(long, default_value = "0.0.0.0")]
    pub remote_bind_host: String,
    #[arg(long)]
    pub metrics_otlp_grpc_url: Option<String>,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(
        long,
        default_value = "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M"
    )]
    pub model_id: String,
    #[arg(long, default_value = "/tmp/skippy-prompt")]
    pub run_root: PathBuf,
    #[arg(long)]
    pub splits: Option<String>,
    #[arg(
        long,
        help = "Run the full model in one stage. This infers the model layer count and conflicts with --splits."
    )]
    pub single_stage: bool,
    #[arg(long, default_value_t = 40)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 128)]
    pub prefill_chunk_size: usize,
    #[arg(long, default_value_t = 64)]
    pub max_new_tokens: usize,
    #[arg(long)]
    pub draft_model_path: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    pub speculative_window: usize,
    #[arg(long)]
    pub adaptive_speculative_window: bool,
    #[arg(long)]
    pub ngram_speculative: bool,
    #[arg(long, value_enum, default_value = "transition-pool")]
    pub ngram_proposal_mode: NgramProposalMode,
    #[arg(long, default_value_t = 24)]
    pub spec_ngram_size_n: usize,
    #[arg(long, default_value_t = 1)]
    pub ngram_history_min_hits: u32,
    #[arg(long, default_value_t = 12)]
    pub draft_min: usize,
    #[arg(long, default_value_t = 48)]
    pub draft_max: usize,
    #[arg(long, default_value_t = DEFAULT_MIN_WINNER_COUNT)]
    pub ngram_min_winner_count: u32,
    #[arg(long, default_value_t = DEFAULT_MIN_CONFIDENCE)]
    pub ngram_min_confidence: f32,
    #[arg(long, default_value_t = DEFAULT_MIN_MARGIN)]
    pub ngram_min_margin: u32,
    #[arg(long, default_value_t = DEFAULT_CONFIDENCE_STEP)]
    pub ngram_confidence_step: f32,
    #[arg(long, default_value_t = DEFAULT_CONFIDENCE_STEP_TOKENS)]
    pub ngram_confidence_step_tokens: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_CONFIDENCE)]
    pub ngram_max_confidence: f32,
    #[arg(long, default_value_t = DEFAULT_COUNT_STEP_TOKENS)]
    pub ngram_count_step_tokens: usize,
    #[arg(long, default_value_t = DEFAULT_MARGIN_STEP_TOKENS)]
    pub ngram_margin_step_tokens: usize,
    #[arg(long, default_value = "lookup-record")]
    pub kv_mode: String,
    #[arg(long, default_value_t = 512)]
    pub kv_page_size_tokens: u64,
    #[arg(long, default_value = "127.0.0.1:18080")]
    pub metrics_http_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:14317")]
    pub metrics_otlp_grpc_addr: SocketAddr,
    #[arg(long, default_value_t = 19031)]
    pub first_stage_port: u16,
    #[arg(long, default_value_t = 2)]
    pub stage_max_inflight: usize,
    #[arg(long, default_value_t = 1)]
    pub stage_reply_credit_limit: usize,
    #[arg(long)]
    pub no_stage_async_prefill_forward: bool,
    #[arg(long, default_value_t = 8192)]
    pub stage_telemetry_queue_capacity: usize,
    #[arg(long, default_value = "summary")]
    pub stage_telemetry_level: String,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long, default_value_t = 30)]
    pub decode_timeout_secs: u64,
    #[arg(long)]
    pub history_path: Option<PathBuf>,
    #[arg(long)]
    pub session_id: Option<String>,
    #[arg(long)]
    pub ngram_pool_uds_path: Option<PathBuf>,
    #[arg(long, default_value_t = 80)]
    pub log_tail_lines: usize,
    #[arg(long)]
    pub native_logs: bool,
    #[arg(
        long,
        help = "Send REPL input to the model as raw completion text instead of rendering a user chat turn."
    )]
    pub raw_prompt: bool,
    #[arg(long)]
    pub no_think: bool,
    #[arg(long)]
    pub thinking_token_budget: Option<usize>,
    #[arg(long)]
    pub show_thinking: bool,
}

#[derive(Parser)]
pub struct BinaryReplArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long)]
    pub tokenizer_model_path: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "runtime-slice")]
    pub tokenizer_load_mode: ReplLoadMode,
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
    pub tokenizer_n_gpu_layers: i32,
    #[arg(long, default_value = "127.0.0.1:19031")]
    pub first_stage_addr: String,
    #[arg(long, default_value_t = 0)]
    pub tokenizer_layer_start: u32,
    #[arg(long, default_value_t = 10)]
    pub tokenizer_layer_end: u32,
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 128)]
    pub prefill_chunk_size: usize,
    #[arg(long, default_value_t = 64)]
    pub max_new_tokens: usize,
    #[arg(long)]
    pub draft_model_path: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    pub speculative_window: usize,
    #[arg(long)]
    pub adaptive_speculative_window: bool,
    #[arg(long)]
    pub ngram_speculative: bool,
    #[arg(long, value_enum, default_value = "transition-pool")]
    pub ngram_proposal_mode: NgramProposalMode,
    #[arg(long, default_value_t = 24)]
    pub spec_ngram_size_n: usize,
    #[arg(long, default_value_t = 1)]
    pub ngram_history_min_hits: u32,
    #[arg(long, default_value_t = 12)]
    pub draft_min: usize,
    #[arg(long, default_value_t = 48)]
    pub draft_max: usize,
    #[arg(long, default_value_t = DEFAULT_MIN_WINNER_COUNT)]
    pub ngram_min_winner_count: u32,
    #[arg(long, default_value_t = DEFAULT_MIN_CONFIDENCE)]
    pub ngram_min_confidence: f32,
    #[arg(long, default_value_t = DEFAULT_MIN_MARGIN)]
    pub ngram_min_margin: u32,
    #[arg(long, default_value_t = DEFAULT_CONFIDENCE_STEP)]
    pub ngram_confidence_step: f32,
    #[arg(long, default_value_t = DEFAULT_CONFIDENCE_STEP_TOKENS)]
    pub ngram_confidence_step_tokens: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_CONFIDENCE)]
    pub ngram_max_confidence: f32,
    #[arg(long, default_value_t = DEFAULT_COUNT_STEP_TOKENS)]
    pub ngram_count_step_tokens: usize,
    #[arg(long, default_value_t = DEFAULT_MARGIN_STEP_TOKENS)]
    pub ngram_margin_step_tokens: usize,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long, default_value_t = 30)]
    pub decode_timeout_secs: u64,
    #[arg(long)]
    pub history_path: Option<PathBuf>,
    #[arg(long)]
    pub session_id: Option<String>,
    #[arg(long)]
    pub ngram_pool_uds_path: Option<PathBuf>,
    #[arg(long)]
    pub native_logs: bool,
    #[arg(
        long,
        help = "Send REPL input to the model as raw completion text instead of rendering a user chat turn."
    )]
    pub raw_prompt: bool,
    #[arg(long)]
    pub no_think: bool,
    #[arg(long)]
    pub thinking_token_budget: Option<usize>,
    #[arg(long)]
    pub show_thinking: bool,
    #[arg(skip)]
    pub diagnostics_hint: Option<String>,
    #[arg(skip)]
    log_context: Option<PromptLogContext>,
}

struct ChildGuard {
    child: Child,
}

impl ChildGuard {
    fn spawn(mut command: Command) -> Result<Self> {
        command.stdin(Stdio::null());
        let child = command
            .spawn()
            .with_context(|| format!("failed to spawn {:?}", command))?;
        Ok(Self { child })
    }
}

struct PromptInterruptState {
    requested: AtomicBool,
    active_stream: Mutex<Option<TcpStream>>,
}

struct ActivePromptStream {
    state: Arc<PromptInterruptState>,
}

impl PromptInterruptState {
    fn begin_request(&self) {
        self.requested.store(false, Ordering::SeqCst);
    }

    fn interrupt_requested(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }

    fn take_interrupt(&self) -> bool {
        self.requested.swap(false, Ordering::SeqCst)
    }

    fn activate(self: &Arc<Self>, stream: &TcpStream) -> Result<ActivePromptStream> {
        let clone = stream
            .try_clone()
            .context("clone prompt stream for interrupt handling")?;
        *self
            .active_stream
            .lock()
            .map_err(|_| anyhow!("prompt interrupt stream lock poisoned"))? = Some(clone);
        Ok(ActivePromptStream {
            state: Arc::clone(self),
        })
    }

    fn clear_active_stream(&self) {
        if let Ok(mut active) = self.active_stream.lock() {
            *active = None;
        }
    }

    fn request_interrupt(&self) {
        let stream = self
            .active_stream
            .lock()
            .ok()
            .and_then(|active| active.as_ref().and_then(|stream| stream.try_clone().ok()));
        if let Some(stream) = stream {
            self.requested.store(true, Ordering::SeqCst);
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

impl Drop for ActivePromptStream {
    fn drop(&mut self) {
        self.state.clear_active_stream();
    }
}

fn install_prompt_interrupt_handler() -> Result<Arc<PromptInterruptState>> {
    static INTERRUPT_STATE: OnceLock<Arc<PromptInterruptState>> = OnceLock::new();
    if let Some(state) = INTERRUPT_STATE.get() {
        return Ok(Arc::clone(state));
    }

    let state = Arc::new(PromptInterruptState {
        requested: AtomicBool::new(false),
        active_stream: Mutex::new(None),
    });
    let handler_state = Arc::clone(&state);
    ctrlc::set_handler(move || {
        handler_state.request_interrupt();
    })
    .context("install Ctrl-C prompt interrupt handler")?;

    let _ = INTERRUPT_STATE.set(Arc::clone(&state));
    Ok(INTERRUPT_STATE.get().map(Arc::clone).unwrap_or(state))
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        CommandKind::Prompt(args) => prompt_repl(*args),
        CommandKind::Binary(args) => binary_repl(*args),
    }
}

struct LocalStage {
    stage_id: String,
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    port: u16,
    bind_addr: String,
    endpoint_addr: String,
    config_path: PathBuf,
    kv_config_path: PathBuf,
    kv_uds_path: PathBuf,
    model_path: PathBuf,
    remote: Option<RemoteStage>,
}

struct RemoteStage {
    host: String,
    stage_dir: String,
    stage_server_bin: String,
    kv_server_bin: String,
    model_path: String,
    config_path: String,
    kv_config_path: String,
    kv_uds_path: String,
    kv_page_root: String,
    stage_log_path: String,
    kv_log_path: String,
    stage_exit_path: String,
}

struct PromptLogContext {
    entries: Vec<PromptLogEntry>,
    default_lines: usize,
}

struct PromptLogEntry {
    label: String,
    target: PromptLogTarget,
}

enum PromptLogTarget {
    Local(PathBuf),
    Remote { host: String, path: String },
}

pub fn prompt_repl(_args: PromptArgs) -> Result<()> {
    bail!(
        "skippy-prompt topology launch is disabled in mesh-llm; use `skippy-prompt binary` against a mesh-managed first stage"
    )
}

#[allow(dead_code)]
fn prompt_repl_launch(args: PromptArgs) -> Result<()> {
    let remote = !args.hosts.is_empty();
    if args.single_stage && args.splits.is_some() {
        bail!("--single-stage conflicts with --splits");
    }
    let hf_package_ref = args
        .model_path
        .to_str()
        .is_some_and(|s| s.starts_with("hf://"))
        || args.model_path.join("model-package.json").is_file();
    if !hf_package_ref && !args.model_path.is_file() {
        bail!("model does not exist: {}", args.model_path.display());
    }
    let effective_layer_end = if args.single_stage {
        if hf_package_ref {
            bail!("--single-stage requires a local GGUF file, not an hf:// package ref");
        }
        model_layer_count(&args.model_path).context("infer full layer count for --single-stage")?
    } else {
        args.layer_end
    };
    let default_stage_count = if args.single_stage {
        1
    } else if remote {
        args.hosts.len()
    } else {
        4
    };
    let ranges = resolve_stage_ranges(
        args.single_stage,
        args.splits.as_deref(),
        default_stage_count,
        effective_layer_end,
    )?;
    validate_prompt_topology_plan(&args, effective_layer_end, &ranges)?;
    if remote && args.hosts.len() != ranges.len() {
        bail!(
            "--hosts count must match the stage count; got {} hosts for {} stages",
            args.hosts.len(),
            ranges.len()
        );
    }
    let kv_mode = match args.kv_mode.as_str() {
        "disabled" | "record" | "lookup-record" | "correctness" => args.kv_mode.as_str(),
        other => bail!("unsupported kv mode {other}"),
    };

    let run_id = format!("prompt-{}", unix_millis());
    let run_dir = args.run_root.join(&run_id);
    let config_dir = run_dir.join("configs");
    fs::create_dir_all(&config_dir)
        .with_context(|| format!("create run config dir {}", config_dir.display()))?;
    let model_cache_key = if hf_package_ref {
        let mut hasher = DefaultHasher::new();
        args.model_path.to_str().unwrap_or("hf").hash(&mut hasher);
        format!("hf-{:016x}", hasher.finish())
    } else {
        model_cache_key(&args.model_path, &ranges)?
    };
    let model_cache_dir = args.run_root.join("model-cache").join(&model_cache_key);
    let model_package_cache_key = if hf_package_ref {
        model_cache_key.clone()
    } else {
        model_package_cache_key(&args.model_path)?
    };
    let model_package_dir = args
        .run_root
        .join("model-package-cache")
        .join(&model_package_cache_key);
    let binary_cache_key = binary_cache_key(&args.stage_server_bin, &args.kv_server_bin)?;

    let mut stages = Vec::with_capacity(ranges.len());
    for (index, (layer_start, layer_end)) in ranges.iter().copied().enumerate() {
        let port = args
            .first_stage_port
            .checked_add(u16::try_from(index).context("stage index exceeds u16")?)
            .context("stage port overflow")?;
        let (bind_addr, endpoint_addr, remote_stage) = if remote {
            let host = args.hosts[index].trim().to_string();
            if host.is_empty() {
                bail!("--hosts contains an empty host at position {}", index + 1);
            }
            let stage_dir = format!("{}/runs/{run_id}/stage-{index}", args.remote_root);
            let remote_model_package_dir = if hf_package_ref {
                args.model_path.to_str().unwrap_or("").to_string()
            } else {
                format!(
                    "{}/model-package-cache/{model_package_cache_key}",
                    args.remote_root
                )
            };
            let remote_binary_cache_dir =
                format!("{}/binary-cache/{binary_cache_key}", args.remote_root);
            (
                format!("{}:{port}", args.remote_bind_host),
                format!("{host}:{port}"),
                Some(RemoteStage {
                    host,
                    stage_server_bin: format!("{remote_binary_cache_dir}/skippy-server"),
                    kv_server_bin: format!("{remote_binary_cache_dir}/kv-server"),
                    model_path: remote_model_package_dir,
                    config_path: format!("{stage_dir}/stage-{index}.json"),
                    kv_config_path: format!("{stage_dir}/kv-stage-{index}.json"),
                    kv_uds_path: format!("{stage_dir}/kv-stage-{index}.sock"),
                    kv_page_root: format!("{stage_dir}/kv-pages"),
                    stage_log_path: format!("{stage_dir}/stage-{index}.log"),
                    kv_log_path: format!("{stage_dir}/kv-stage-{index}.log"),
                    stage_exit_path: format!("{stage_dir}/stage.exit"),
                    stage_dir,
                }),
            )
        } else {
            (
                format!("127.0.0.1:{port}"),
                format!("127.0.0.1:{port}"),
                None,
            )
        };
        stages.push(LocalStage {
            stage_id: format!("stage-{index}"),
            stage_index: index,
            layer_start,
            layer_end,
            port,
            bind_addr,
            endpoint_addr,
            config_path: config_dir.join(format!("stage-{index}.json")),
            kv_config_path: config_dir.join(format!("kv-stage-{index}.json")),
            kv_uds_path: run_dir.join(format!("kv-stage-{index}.sock")),
            model_path: model_cache_dir.join(format!("stage-{index}.gguf")),
            remote: remote_stage,
        });
    }

    let metrics_otlp_url = metrics_otlp_url(&args, &stages)?;
    if hf_package_ref {
        eprintln!(
            "launch: using HF layer package {} (each stage downloads its own layers)",
            args.model_path.display()
        );
    } else if remote {
        eprintln!(
            "launch: materializing remote model package cache at {}",
            model_package_dir.display()
        );
        materialize_model_package(&args, &model_package_dir)?;
    } else {
        eprintln!(
            "launch: materializing {} local GGUF stage shards",
            stages.len()
        );
        materialize_stage_artifacts(&args, &stages)?;
    }
    eprintln!(
        "launch: writing stage and KV configs under {}",
        run_dir.display()
    );
    write_local_configs(
        &args,
        &run_id,
        &run_dir,
        kv_mode,
        &stages,
        &metrics_otlp_url,
        hf_package_ref,
    )?;

    let mut children = Vec::new();
    let metrics_db = run_dir.join("metrics.sqlite");
    let metrics_otlp_bind_addr = metrics_otlp_bind_addr(&args, remote);
    let mut metrics = Command::new(&args.metrics_server_bin);
    metrics.args([
        "serve",
        "--db",
        path_str(&metrics_db)?,
        "--http-addr",
        &args.metrics_http_addr.to_string(),
        "--otlp-grpc-addr",
        &metrics_otlp_bind_addr,
    ]);
    configure_process_log(&mut metrics, &run_dir.join("metrics-server.log"))?;
    eprintln!(
        "launch: starting metrics-server http={} otlp_bind={} log={}",
        args.metrics_http_addr,
        metrics_otlp_bind_addr,
        run_dir.join("metrics-server.log").display()
    );
    children.push(ChildGuard::spawn(metrics)?);

    let ngram_pool_uds_path = if args.ngram_speculative {
        let socket_path = args
            .ngram_pool_uds_path
            .clone()
            .unwrap_or_else(|| run_dir.join("ngram-pool.sock"));
        let mut ngram_pool = Command::new(&args.ngram_pool_server_bin);
        ngram_pool.args(["serve", "--uds-path", path_str(&socket_path)?]);
        configure_process_log(&mut ngram_pool, &run_dir.join("ngram-pool-server.log"))?;
        eprintln!(
            "launch: starting ngram-pool-server socket={} log={}",
            socket_path.display(),
            run_dir.join("ngram-pool-server.log").display()
        );
        children.push(ChildGuard::spawn(ngram_pool)?);
        eprintln!(
            "launch: waiting for ngram pool socket {}",
            socket_path.display()
        );
        wait_for_socket(&socket_path, args.startup_timeout_secs)?;
        Some(socket_path)
    } else {
        None
    };

    if remote {
        rsync_remote_stage_inputs(&args, &stages, &model_package_dir, hf_package_ref)?;
        eprintln!(
            "launch: starting {} remote stages; first_stage_addr={}",
            stages.len(),
            stages[0].endpoint_addr
        );
        start_remote_stages(&args, &stages, &metrics_otlp_url, &mut children)?;
    } else {
        eprintln!("launch: starting {} local KV sidecars", stages.len());
        for stage in &stages {
            let mut kv = Command::new(&args.kv_server_bin);
            kv.args(["serve", "--config", path_str(&stage.kv_config_path)?]);
            configure_process_log(
                &mut kv,
                &run_dir.join(format!("kv-stage-{}.log", stage.stage_index)),
            )?;
            children.push(ChildGuard::spawn(kv)?);
        }
        eprintln!("launch: waiting for local KV sockets");
        wait_for_kv_sockets(&stages, args.startup_timeout_secs)?;

        eprintln!("launch: starting {} local stage servers", stages.len());
        for stage in stages.iter().rev() {
            let mut server = Command::new(&args.stage_server_bin);
            add_stage_server_args(&mut server, &args, stage, &metrics_otlp_url)?;
            configure_process_log(
                &mut server,
                &run_dir.join(format!("stage-{}.log", stage.stage_index)),
            )?;
            children.push(ChildGuard::spawn(server)?);
        }
    }

    eprintln!("prompt run dir: {}", run_dir.display());
    eprintln!(
        "metrics: otlp_bind={} otlp_url={}",
        metrics_otlp_bind_addr, metrics_otlp_url
    );
    eprintln!(
        "topology: {}",
        stages
            .iter()
            .map(|stage| format!(
                "{}:{}..{}@{} model={}",
                stage.stage_id,
                stage.layer_start,
                stage.layer_end,
                stage.endpoint_addr,
                stage_model_location(stage)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("logs: use :logs [name] [lines] inside the REPL");

    let log_context = Some(prompt_log_context(&run_dir, &stages, args.log_tail_lines));

    let tokenizer_model_path = if hf_package_ref {
        let hf_ref = args.model_path.to_str().context("model_path not utf-8")?;
        eprintln!("launch: materializing tokenizer from package (layers 0..1)");
        let tokenizer_gguf = materialize_layer_package(&PackageStageRequest {
            model_id: args.model_id.clone(),
            topology_id: "tokenizer".to_string(),
            package_ref: hf_ref.to_string(),
            stage_id: "tokenizer".to_string(),
            layer_start: 0,
            layer_end: 1,
            include_embeddings: true,
            include_output: true,
        })
        .context("materialize tokenizer from layer package")?;
        Some(tokenizer_gguf)
    } else if remote {
        None
    } else {
        stages.first().map(|stage| stage.model_path.clone())
    };
    let tokenizer_load_mode = if hf_package_ref {
        ReplLoadMode::LayerPackage
    } else if tokenizer_model_path.is_some() {
        ReplLoadMode::ArtifactSlice
    } else {
        ReplLoadMode::RuntimeSlice
    };
    let repl_result = binary_repl(BinaryReplArgs {
        model_path: args.model_path,
        tokenizer_model_path,
        tokenizer_load_mode,
        tokenizer_n_gpu_layers: 0,
        first_stage_addr: stages[0].endpoint_addr.clone(),
        tokenizer_layer_start: stages[0].layer_start as u32,
        tokenizer_layer_end: stages[0].layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        prefill_chunk_size: args.prefill_chunk_size,
        max_new_tokens: args.max_new_tokens,
        draft_model_path: args.draft_model_path,
        speculative_window: args.speculative_window,
        adaptive_speculative_window: args.adaptive_speculative_window,
        ngram_speculative: args.ngram_speculative,
        ngram_proposal_mode: args.ngram_proposal_mode,
        spec_ngram_size_n: args.spec_ngram_size_n,
        ngram_history_min_hits: args.ngram_history_min_hits,
        draft_min: args.draft_min,
        draft_max: args.draft_max,
        ngram_min_winner_count: args.ngram_min_winner_count,
        ngram_min_confidence: args.ngram_min_confidence,
        ngram_min_margin: args.ngram_min_margin,
        ngram_confidence_step: args.ngram_confidence_step,
        ngram_confidence_step_tokens: args.ngram_confidence_step_tokens,
        ngram_max_confidence: args.ngram_max_confidence,
        ngram_count_step_tokens: args.ngram_count_step_tokens,
        ngram_margin_step_tokens: args.ngram_margin_step_tokens,
        startup_timeout_secs: args.startup_timeout_secs,
        decode_timeout_secs: args.decode_timeout_secs,
        history_path: Some(
            args.history_path
                .unwrap_or_else(|| args.run_root.join("prompt-history.txt")),
        ),
        session_id: args.session_id,
        ngram_pool_uds_path,
        native_logs: args.native_logs,
        raw_prompt: args.raw_prompt,
        no_think: args.no_think,
        thinking_token_budget: args.thinking_token_budget,
        show_thinking: args.show_thinking,
        diagnostics_hint: Some(stage_diagnostics_hint(&run_dir, &stages)),
        log_context,
    });

    if remote {
        for stage in &stages {
            let _ = stop_remote_stage_listener(stage);
        }
    }
    drop(children);
    repl_result
}

pub fn binary_repl(args: BinaryReplArgs) -> Result<()> {
    if args.native_logs {
        restore_native_logs();
    } else {
        suppress_native_logs();
    }
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let tokenizer_path = args
        .tokenizer_model_path
        .as_deref()
        .unwrap_or(args.model_path.as_path());
    let tokenizer = StageModel::open(
        tokenizer_path,
        &RuntimeConfig {
            stage_index: 0,
            layer_start: args.tokenizer_layer_start,
            layer_end: args.tokenizer_layer_end,
            ctx_size: args.ctx_size,
            n_gpu_layers: args.tokenizer_n_gpu_layers,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            load_mode: args.tokenizer_load_mode.into(),
            projector_path: None,
            include_embeddings: true,
            include_output: false,
            filter_tensors_on_load: true,
        },
    )
    .with_context(|| format!("open tokenizer model {}", tokenizer_path.display()))?;
    eprintln!(
        "tokenizer model: {} load_mode={:?} n_gpu_layers={}",
        tokenizer_path.display(),
        args.tokenizer_load_mode,
        args.tokenizer_n_gpu_layers
    );
    let thinking_override = prompt_thinking_override(&args)?;
    let hf_repl = args
        .model_path
        .to_str()
        .is_some_and(|s| s.starts_with("hf://"));
    let chat_template_model = if !args.raw_prompt
        && !hf_repl
        && args
            .tokenizer_model_path
            .as_deref()
            .is_some_and(|path| path != args.model_path.as_path())
    {
        let model = StageModel::open(
            &args.model_path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: 1,
                ctx_size: args.ctx_size,
                n_gpu_layers: 0,
                selected_backend_device: None,
                cache_type_k: GGML_TYPE_F16,
                cache_type_v: GGML_TYPE_F16,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: false,
                filter_tensors_on_load: true,
            },
        )
        .with_context(|| format!("open chat template model {}", args.model_path.display()))?;
        eprintln!(
            "chat template model: {} load_mode={:?} n_gpu_layers=0",
            args.model_path.display(),
            ReplLoadMode::RuntimeSlice
        );
        Some(model)
    } else {
        None
    };
    let mut draft = match args.draft_model_path.as_ref() {
        Some(path) => Some(DraftRunner::open(
            path,
            args.ctx_size,
            args.n_gpu_layers,
            args.speculative_window,
        )?),
        None => None,
    };

    eprintln!(
        "binary REPL connected to first stage at {}; max_new_tokens={} prefill_chunk_size={}",
        args.first_stage_addr, args.max_new_tokens, args.prefill_chunk_size
    );
    if let Some(draft) = draft.as_ref() {
        eprintln!(
            "draft model enabled: {} speculative_window={}",
            draft.path.display(),
            draft.window
        );
    }
    if args.adaptive_speculative_window {
        eprintln!(
            "adaptive speculative window enabled: start={} max={}",
            args.speculative_window.clamp(1, 4),
            args.speculative_window.max(1)
        );
    }
    if args.raw_prompt {
        eprintln!("prompt mode: raw completion");
    } else {
        eprintln!("prompt mode: chat template");
        match thinking_override {
            Some(false) => eprintln!("thinking: disabled at chat-template render"),
            Some(true) => eprintln!(
                "thinking: enabled at chat-template render; requested budget={}",
                args.thinking_token_budget
                    .map(|budget| budget.to_string())
                    .unwrap_or_else(|| "default".to_string())
            ),
            None => eprintln!("thinking: model template default"),
        }
    }
    eprintln!(
        "thinking output: {}",
        if args.show_thinking {
            "showing <think> blocks"
        } else {
            "hiding visible <think> blocks"
        }
    );
    let default_session_id = args.session_id.clone().unwrap_or_else(default_session_id);
    let default_wire_session_id = stable_wire_id(&[default_session_id.as_bytes()]);
    eprintln!("session_id={default_session_id} wire_session_id={default_wire_session_id}");
    let mut ngram = if args.ngram_speculative {
        eprintln!(
            "ngram speculative enabled: mode={:?} n={} history_min_hits={} draft_min={} draft_max={} min_count={} min_confidence={:.2} min_margin={} confidence_step={:.2}/{} max_confidence={:.2} count_step={} margin_step={}",
            args.ngram_proposal_mode,
            args.spec_ngram_size_n,
            args.ngram_history_min_hits,
            args.draft_min,
            args.draft_max,
            args.ngram_min_winner_count,
            args.ngram_min_confidence,
            args.ngram_min_margin,
            args.ngram_confidence_step,
            args.ngram_confidence_step_tokens,
            args.ngram_max_confidence,
            args.ngram_count_step_tokens,
            args.ngram_margin_step_tokens
        );
        Some(NgramSource::open(&args, &default_session_id)?)
    } else {
        None
    };
    let interrupt = install_prompt_interrupt_handler()?;
    let mut history = PromptHistory::load(args.history_path.as_deref())?;
    let mut editor = prompt_editor(&history)?;
    if args.log_context.is_some() {
        eprintln!(
            "Type a prompt, use Up/Down for history, Ctrl-C to interrupt generation, :history, :logs [name] [lines], :rerun N, or :quit."
        );
    } else {
        eprintln!("Type a prompt, use Up/Down for history, Ctrl-C to interrupt generation, :history, :rerun N, or :quit.");
    }

    let mut prompt_index = 0usize;
    loop {
        let Some(input) = read_history_prompt(&mut editor, "> ")? else {
            break;
        };
        let raw_input = input.trim_end_matches(['\r', '\n']);
        if raw_input.trim().is_empty() {
            continue;
        }
        if let Some((prompt_session_id, prompt)) = parse_prompt_json_command(raw_input)? {
            let prompt_session_id = if prompt_session_id.is_empty() {
                default_session_id.clone()
            } else {
                prompt_session_id
            };
            let wire_session_id = stable_wire_id(&[prompt_session_id.as_bytes()]);
            run_prompt(PromptRun {
                args: &args,
                tokenizer: &tokenizer,
                chat_template_model: chat_template_model.as_ref(),
                draft: draft.as_mut(),
                ngram: ngram.as_mut(),
                interrupt: &interrupt,
                wire_dtype,
                session_id: &prompt_session_id,
                wire_session_id,
                prompt_index,
                prompt: &prompt,
            })
            .or_else(|error| handle_prompt_error(error, &interrupt, prompt_index))?;
            prompt_index += 1;
            continue;
        }
        let input = raw_input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, ":q" | ":quit" | ":exit") {
            break;
        }
        if input == ":history" {
            history.print();
            continue;
        }
        if input == ":logs" || input.starts_with(":logs ") {
            show_prompt_logs(
                args.log_context.as_ref(),
                input.trim_start_matches(":logs").trim(),
            )?;
            continue;
        }
        if let Some(index) = input.strip_prefix(":rerun ") {
            let index = index
                .trim()
                .parse::<usize>()
                .context("parse :rerun index")?;
            let Some(prompt) = history.get(index).map(str::to_string) else {
                eprintln!("history entry {index} does not exist");
                continue;
            };
            eprintln!("rerun {index}: {prompt}");
            run_prompt(PromptRun {
                args: &args,
                tokenizer: &tokenizer,
                chat_template_model: chat_template_model.as_ref(),
                draft: draft.as_mut(),
                ngram: ngram.as_mut(),
                interrupt: &interrupt,
                wire_dtype,
                session_id: &default_session_id,
                wire_session_id: default_wire_session_id,
                prompt_index,
                prompt: &prompt,
            })
            .or_else(|error| handle_prompt_error(error, &interrupt, prompt_index))?;
            prompt_index += 1;
            continue;
        }
        history.push(input)?;
        let _ = editor.add_history_entry(input);
        run_prompt(PromptRun {
            args: &args,
            tokenizer: &tokenizer,
            chat_template_model: chat_template_model.as_ref(),
            draft: draft.as_mut(),
            ngram: ngram.as_mut(),
            interrupt: &interrupt,
            wire_dtype,
            session_id: &default_session_id,
            wire_session_id: default_wire_session_id,
            prompt_index,
            prompt: input,
        })
        .or_else(|error| handle_prompt_error(error, &interrupt, prompt_index))?;
        prompt_index += 1;
    }

    Ok(())
}

fn handle_prompt_error(
    error: anyhow::Error,
    interrupt: &Arc<PromptInterruptState>,
    prompt_index: usize,
) -> Result<()> {
    if interrupt.take_interrupt() {
        eprintln!();
        eprintln!("request {prompt_index}: interrupted");
        Ok(())
    } else {
        Err(error)
    }
}

fn stage_diagnostics_hint(run_dir: &Path, stages: &[LocalStage]) -> String {
    let mut lines = vec![
        format!("run_dir={}", run_dir.display()),
        "stage diagnostics:".to_string(),
    ];
    for stage in stages {
        if let Some(remote) = stage.remote.as_ref() {
            lines.push(format!(
                "  {} {}:{}..{} addr={} log={}:{} exit={}:{}",
                stage.stage_id,
                remote.host,
                stage.layer_start,
                stage.layer_end,
                stage.endpoint_addr,
                remote.host,
                remote.stage_log_path,
                remote.host,
                remote.stage_exit_path
            ));
        } else {
            lines.push(format!(
                "  {} layers {}..{} addr={} log={}",
                stage.stage_id,
                stage.layer_start,
                stage.layer_end,
                stage.endpoint_addr,
                run_dir
                    .join(format!("stage-{}.log", stage.stage_index))
                    .display()
            ));
        }
    }
    lines.join("\n")
}

fn prompt_log_context(
    run_dir: &Path,
    stages: &[LocalStage],
    default_lines: usize,
) -> PromptLogContext {
    let mut entries = vec![PromptLogEntry {
        label: "metrics-server".to_string(),
        target: PromptLogTarget::Local(run_dir.join("metrics-server.log")),
    }];
    for stage in stages {
        if let Some(remote) = stage.remote.as_ref() {
            entries.push(PromptLogEntry {
                label: format!("kv-stage-{}", stage.stage_index),
                target: PromptLogTarget::Remote {
                    host: remote.host.clone(),
                    path: remote.kv_log_path.clone(),
                },
            });
            entries.push(PromptLogEntry {
                label: format!("stage-{}", stage.stage_index),
                target: PromptLogTarget::Remote {
                    host: remote.host.clone(),
                    path: remote.stage_log_path.clone(),
                },
            });
        } else {
            entries.push(PromptLogEntry {
                label: format!("kv-stage-{}", stage.stage_index),
                target: PromptLogTarget::Local(
                    run_dir.join(format!("kv-stage-{}.log", stage.stage_index)),
                ),
            });
            entries.push(PromptLogEntry {
                label: format!("stage-{}", stage.stage_index),
                target: PromptLogTarget::Local(
                    run_dir.join(format!("stage-{}.log", stage.stage_index)),
                ),
            });
        }
    }
    PromptLogContext {
        entries,
        default_lines,
    }
}

fn show_prompt_logs(context: Option<&PromptLogContext>, spec: &str) -> Result<()> {
    let Some(context) = context else {
        eprintln!("no prompt-managed logs are attached to this REPL");
        return Ok(());
    };
    let mut lines = context.default_lines.max(1);
    let mut filters = Vec::new();
    for part in spec.split_whitespace() {
        if let Ok(value) = part.parse::<usize>() {
            lines = value.max(1);
        } else {
            filters.push(part);
        }
    }
    let exact_entries = context
        .entries
        .iter()
        .filter(|entry| filters.iter().all(|filter| entry.label == *filter))
        .collect::<Vec<_>>();
    let entries = if exact_entries.is_empty() && !filters.is_empty() {
        context
            .entries
            .iter()
            .filter(|entry| filters.iter().all(|filter| entry.label.contains(filter)))
            .collect::<Vec<_>>()
    } else {
        exact_entries
    };
    if entries.is_empty() {
        let available = context
            .entries
            .iter()
            .map(|entry| entry.label.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("no logs matched; available logs: {available}");
        return Ok(());
    }

    for entry in entries {
        eprintln!(
            "==> {} ({}) <==",
            entry.label,
            describe_log_target(&entry.target)
        );
        match tail_prompt_log(&entry.target, lines) {
            Ok(tail) if tail.is_empty() => eprintln!("  <empty>"),
            Ok(tail) => {
                for line in tail {
                    eprintln!("{line}");
                }
            }
            Err(error) => eprintln!("  <failed to read log: {error:#}>"),
        }
    }
    Ok(())
}

fn describe_log_target(target: &PromptLogTarget) -> String {
    match target {
        PromptLogTarget::Local(path) => path.display().to_string(),
        PromptLogTarget::Remote { host, path } => format!("{host}:{path}"),
    }
}

fn tail_prompt_log(target: &PromptLogTarget, lines: usize) -> Result<Vec<String>> {
    match target {
        PromptLogTarget::Local(path) => tail_local_log(path, lines),
        PromptLogTarget::Remote { host, path } => tail_remote_log(host, path, lines),
    }
}

fn tail_local_log(path: &Path, lines: usize) -> Result<Vec<String>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut tail = VecDeque::with_capacity(lines);
    for line in reader.lines() {
        if tail.len() == lines {
            tail.pop_front();
        }
        tail.push_back(line.with_context(|| format!("read {}", path.display()))?);
    }
    Ok(tail.into_iter().collect())
}

fn tail_remote_log(host: &str, path: &str, lines: usize) -> Result<Vec<String>> {
    let output = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(format!("tail -n {lines} {}", shell_quote(path)))
        .output()
        .with_context(|| format!("tail remote log {host}:{path}"))?;
    if !output.status.success() {
        bail!(
            "tail remote log {host}:{path} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect())
}

struct ThinkingOutputFilter {
    enabled: bool,
    hidden: bool,
    suppress_leading_ws: bool,
    pending: String,
}

struct FilteredOutput {
    text: String,
    suppressed_thinking: bool,
}

impl ThinkingOutputFilter {
    const OPEN: &'static str = "<think>";
    const CLOSE: &'static str = "</think>";
    const KEEP_BYTES: usize = 7;

    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            hidden: false,
            suppress_leading_ws: false,
            pending: String::new(),
        }
    }

    fn push(&mut self, piece: &str) -> FilteredOutput {
        if !self.enabled {
            return FilteredOutput {
                text: piece.to_string(),
                suppressed_thinking: false,
            };
        }
        self.pending.push_str(piece);
        let mut output = String::new();
        let mut suppressed_thinking = false;
        loop {
            if self.hidden {
                let Some(end) = find_ascii_case_insensitive(&self.pending, Self::CLOSE) else {
                    let keep_bytes = self.pending.len().min(Self::KEEP_BYTES);
                    let drain_to =
                        previous_char_boundary(&self.pending, self.pending.len() - keep_bytes);
                    suppressed_thinking |= drain_to > 0;
                    self.pending.drain(..drain_to);
                    break;
                };
                let drain_to = end + Self::CLOSE.len();
                suppressed_thinking |= drain_to > 0;
                self.pending.drain(..drain_to);
                self.hidden = false;
                self.suppress_leading_ws = true;
                continue;
            }

            if self.suppress_leading_ws {
                let trimmed = self.pending.trim_start().len();
                let trim_bytes = self.pending.len().saturating_sub(trimmed);
                if trim_bytes > 0 {
                    self.pending.drain(..trim_bytes);
                }
                self.suppress_leading_ws = false;
            }

            if let Some(start) = find_ascii_case_insensitive(&self.pending, Self::OPEN) {
                output.push_str(&self.pending[..start]);
                let drain_to = start + Self::OPEN.len();
                suppressed_thinking = true;
                self.pending.drain(..drain_to);
                self.hidden = true;
                continue;
            }

            let emit_bytes = self.pending.len().saturating_sub(Self::KEEP_BYTES);
            if emit_bytes == 0 {
                break;
            }
            let emit_bytes = previous_char_boundary(&self.pending, emit_bytes);
            output.push_str(&self.pending[..emit_bytes]);
            self.pending.drain(..emit_bytes);
            break;
        }
        FilteredOutput {
            text: output,
            suppressed_thinking,
        }
    }

    fn finish(&mut self) -> FilteredOutput {
        if !self.enabled {
            return FilteredOutput {
                text: String::new(),
                suppressed_thinking: false,
            };
        }
        if self.hidden {
            self.pending.clear();
            self.hidden = false;
            self.suppress_leading_ws = false;
            return FilteredOutput {
                text: String::new(),
                suppressed_thinking: true,
            };
        }
        let mut output = std::mem::take(&mut self.pending);
        if self.suppress_leading_ws {
            output = output.trim_start().to_string();
            self.suppress_leading_ws = false;
        }
        FilteredOutput {
            text: output,
            suppressed_thinking: false,
        }
    }
}

fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

fn previous_char_boundary(value: &str, index: usize) -> usize {
    let mut index = index.min(value.len());
    while index > 0 && !value.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn format_prompt_for_model(
    tokenizer: &StageModel,
    chat_template_model: Option<&StageModel>,
    prompt: &str,
    args: &BinaryReplArgs,
) -> Result<String> {
    if args.raw_prompt {
        return Ok(prompt.to_string());
    }
    let enable_thinking = prompt_thinking_override(args)?;

    chat_template_model
        .unwrap_or(tokenizer)
        .apply_chat_template_with_options(
            &[ChatTemplateMessage::new("user", prompt)],
            ChatTemplateOptions {
                add_assistant: true,
                enable_thinking,
            },
        )
        .with_context(|| {
            let mode = match enable_thinking {
                Some(true) => "enabled",
                Some(false) => "disabled",
                None => "default",
            };
            format!("apply chat template with thinking {mode}")
        })
}

fn prompt_thinking_override(args: &BinaryReplArgs) -> Result<Option<bool>> {
    let reasoning = prompt_openai_reasoning_config(args.no_think, args.thinking_token_budget)?;
    let extra = BTreeMap::new();
    normalize_reasoning_template_options(reasoning.as_ref(), None, &extra)
        .map(|options| options.enable_thinking)
        .map_err(|error| {
            anyhow!(
                "normalize OpenAI reasoning controls: {}",
                error.body().error.message
            )
        })
}

fn prompt_openai_reasoning_config(
    no_think: bool,
    thinking_token_budget: Option<usize>,
) -> Result<Option<ReasoningConfig>> {
    if no_think || thinking_token_budget.unwrap_or(0) == 0 {
        return Ok(Some(ReasoningConfig {
            enabled: Some(false),
            max_tokens: Some(0),
            ..ReasoningConfig::default()
        }));
    }

    let Some(budget) = thinking_token_budget else {
        return Ok(None);
    };
    Ok(Some(ReasoningConfig {
        enabled: Some(true),
        max_tokens: Some(
            budget
                .try_into()
                .context("--thinking-token-budget exceeds u32 range")?,
        ),
        ..ReasoningConfig::default()
    }))
}

struct PromptRun<'a> {
    args: &'a BinaryReplArgs,
    tokenizer: &'a StageModel,
    chat_template_model: Option<&'a StageModel>,
    draft: Option<&'a mut DraftRunner>,
    ngram: Option<&'a mut NgramSource>,
    interrupt: &'a Arc<PromptInterruptState>,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    session_id: &'a str,
    wire_session_id: u64,
    prompt_index: usize,
    prompt: &'a str,
}

fn run_prompt(run: PromptRun<'_>) -> Result<()> {
    let PromptRun {
        args,
        tokenizer,
        chat_template_model,
        mut draft,
        mut ngram,
        interrupt,
        wire_dtype,
        session_id,
        wire_session_id,
        prompt_index,
        prompt,
    } = run;

    if args.prefill_chunk_size == 0 {
        bail!("prefill_chunk_size must be greater than zero");
    }
    interrupt.begin_request();
    let wall_started = Instant::now();
    let tokenize_started = Instant::now();
    let prompt_for_model = format_prompt_for_model(tokenizer, chat_template_model, prompt, args)?;
    let token_ids = tokenizer
        .tokenize(&prompt_for_model, true)
        .with_context(|| format!("tokenize prompt {prompt_for_model:?}"))?;
    let tokenize_ms = elapsed_ms(tokenize_started);
    if token_ids.is_empty() {
        bail!("prompt produced no tokens");
    }

    let prefill_token_count = if token_ids.len() == 1 {
        1
    } else {
        token_ids.len().saturating_sub(1)
    };
    eprintln!(
        "request {prompt_index}: prompt_tokens={} prefill_tokens={} max_new_tokens={}",
        token_ids.len(),
        prefill_token_count,
        args.max_new_tokens
    );
    let prompt_index_bytes = prompt_index.to_le_bytes();
    let request_id = stable_wire_id(&[session_id.as_bytes(), &prompt_index_bytes]);

    eprintln!(
        "request {prompt_index}: connecting to {}",
        args.first_stage_addr
    );
    let mut stream = connect_ready(&args.first_stage_addr, args.startup_timeout_secs)
        .context("first binary stage did not become ready")?;
    let _interrupt_guard = interrupt.activate(&stream)?;
    let io_timeout = Duration::from_secs(args.decode_timeout_secs.max(1));
    stream.set_read_timeout(Some(io_timeout)).ok();
    stream.set_write_timeout(Some(io_timeout)).ok();
    eprintln!("request {prompt_index}: connected");
    let prefill_started = Instant::now();
    let mut prefill_chunk_count = 0usize;
    let prefill_tokens = &token_ids[..prefill_token_count];
    if !prefill_tokens.is_empty() {
        for (chunk_index, chunk) in prefill_tokens.chunks(args.prefill_chunk_size).enumerate() {
            if interrupt.interrupt_requested() {
                bail!("prompt interrupted");
            }
            prefill_chunk_count += 1;
            eprintln!(
                "request {prompt_index}: prefill chunk {} tokens={} pos={}",
                chunk_index,
                chunk.len(),
                chunk_index * args.prefill_chunk_size
            );
            send_prefill_chunk(
                &mut stream,
                wire_dtype,
                ReplPrefillChunk {
                    prompt_index,
                    request_id,
                    session_id: wire_session_id,
                    pos_start: chunk_index * args.prefill_chunk_size,
                    prefill_token_count,
                    tokens: chunk,
                },
            )
            .with_context(|| stage_chain_error_context(args))?;
        }
    }
    let prefill_ms = elapsed_ms(prefill_started);
    eprintln!(
        "request {prompt_index}: prefill complete chunks={} elapsed_ms={:.2}",
        prefill_chunk_count, prefill_ms
    );

    let mut current = *token_ids.last().expect("checked non-empty tokens");
    let mut generated = Vec::with_capacity(args.max_new_tokens);
    let mut decode_ms = 0.0;
    let mut first_decode_ms = None;
    let mut first_time_to_token_ms = None;
    let mut saw_visible_output = false;
    let mut suppressed_thinking_tokens = 0usize;
    let mut reply_stats = StageReplyStats::default();
    let mut speculative_stats = SpeculativeStats::default();
    let mut output_filter = ThinkingOutputFilter::new(!args.show_thinking);
    let mut context_tokens = token_ids.clone();
    if let Some(draft) = draft.as_deref_mut() {
        draft.reset_to_context(&context_tokens)?;
    }
    if let Some(ngram) = ngram.as_deref_mut() {
        ngram.observe_sequence(session_id, &context_tokens)?;
    }
    let max_speculative_window = args.speculative_window.max(1);
    let mut adaptive_window = if args.adaptive_speculative_window {
        max_speculative_window.min(4)
    } else {
        max_speculative_window
    };
    if draft.is_some() || ngram.is_some() {
        speculative_stats.adaptive_window_max = max_speculative_window;
        speculative_stats.adaptive_window_start = adaptive_window;
        speculative_stats.adaptive_window_enabled = args.adaptive_speculative_window;
    }

    while generated.len() < args.max_new_tokens {
        if interrupt.interrupt_requested() {
            bail!("prompt interrupted");
        }
        if generated.is_empty() {
            eprintln!("request {prompt_index}: waiting for first decode token");
        }

        let remaining = args.max_new_tokens - generated.len();
        let proposal_limit = remaining.min(adaptive_window);
        let draft_tokens = match ngram.as_deref_mut() {
            Some(ngram) => ngram.propose(session_id, &context_tokens, proposal_limit)?,
            None => Vec::new(),
        };
        let draft_tokens = if draft_tokens.is_empty() {
            match draft.as_deref_mut() {
                Some(draft) if draft.window > 0 => {
                    draft.propose(current, proposal_limit.min(draft.window))?
                }
                _ => Vec::new(),
            }
        } else {
            draft_tokens
        };

        if draft_tokens.is_empty() {
            let decode_index = generated.len();
            let reply = send_decode_step(
                &mut stream,
                wire_dtype,
                prompt_index,
                request_id,
                wire_session_id,
                token_ids.len(),
                prefill_token_count,
                decode_index,
                current,
            )
            .with_context(|| stage_chain_error_context(args))?;
            decode_ms += reply.elapsed_ms;
            first_decode_ms.get_or_insert(reply.elapsed_ms);
            reply_stats.merge(reply.stats);
            current = reply.predicted;
            generated.push(current);
            context_tokens.push(current);
            if let Some(ngram) = ngram.as_deref_mut() {
                ngram.observe_accepted(session_id, &context_tokens)?;
            }
            first_time_to_token_ms.get_or_insert_with(|| elapsed_ms(wall_started));
            if tokenizer.token_is_eog(current)? {
                break;
            }
            let piece = tokenizer.detokenize(&[current])?;
            let filtered = output_filter.push(&piece);
            if filtered.suppressed_thinking {
                suppressed_thinking_tokens += 1;
            }
            if filtered.text.chars().any(|ch| !ch.is_whitespace()) {
                saw_visible_output = true;
            }
            print!("{}", filtered.text);
            io::stdout().flush().ok();
            continue;
        }

        speculative_stats.windows += 1;
        speculative_stats.draft_tokens += draft_tokens.len();
        speculative_stats.adaptive_window_sum += adaptive_window;
        speculative_stats.adaptive_window_min =
            nonzero_min(speculative_stats.adaptive_window_min, adaptive_window);
        speculative_stats.adaptive_window_max_seen = speculative_stats
            .adaptive_window_max_seen
            .max(adaptive_window);
        let decode_index = generated.len();
        let verify_inputs = verify_inputs_for_proposals(current, &draft_tokens);
        let reply = send_verify_span(
            &mut stream,
            wire_dtype,
            prompt_index,
            request_id,
            wire_session_id,
            token_ids.len(),
            prefill_token_count + decode_index,
            decode_index,
            &verify_inputs,
            true,
        )
        .with_context(|| stage_chain_error_context(args))?;
        decode_ms += reply.elapsed_ms;
        first_decode_ms.get_or_insert(reply.elapsed_ms);
        speculative_stats.observe_primary_verify(&reply, verify_inputs.len());
        reply_stats.merge(reply.stats);
        first_time_to_token_ms.get_or_insert_with(|| elapsed_ms(wall_started));
        let decision = classify_verify_span(
            &draft_tokens,
            &reply.predicted_tokens,
            generated.len(),
            args.max_new_tokens,
            |token| tokenizer.token_is_eog(token),
        )?;
        speculative_stats.observe_verify_decision(
            decision,
            &mut adaptive_window,
            args.adaptive_speculative_window,
            max_speculative_window,
        );

        let mut commit_tokens = reply.predicted_tokens[..decision.commit_count].to_vec();

        if decision.requires_repair() {
            let repair_input_count = decision
                .repair_input_count
                .context("missing rejected span index")?;
            speculative_stats.recovery_restores += 1;
            let restore = send_session_control(
                &mut stream,
                wire_dtype,
                prompt_index,
                request_id,
                wire_session_id,
                WireMessageKind::RestoreSession,
            )
            .with_context(|| stage_chain_error_context(args))?;
            decode_ms += restore.elapsed_ms;
            speculative_stats.recovery_ms += restore.elapsed_ms;
            speculative_stats.recovery_restore_ms += restore.elapsed_ms;
            reply_stats.merge(restore.stats);

            if repair_input_count == 1 {
                let repair = send_decode_step(
                    &mut stream,
                    wire_dtype,
                    prompt_index,
                    request_id,
                    wire_session_id,
                    token_ids.len(),
                    prefill_token_count,
                    decode_index,
                    current,
                )
                .with_context(|| stage_chain_error_context(args))?;
                commit_tokens = vec![repair.predicted];
                reply_stats.merge(repair.stats);
                decode_ms += repair.elapsed_ms;
                speculative_stats.recovery_decode_repairs += 1;
                speculative_stats.recovery_ms += repair.elapsed_ms;
                speculative_stats.recovery_decode_elapsed_ms += repair.elapsed_ms;
            } else {
                let repair_inputs = &verify_inputs[..repair_input_count];
                let repair = send_verify_span(
                    &mut stream,
                    wire_dtype,
                    prompt_index,
                    request_id,
                    wire_session_id,
                    token_ids.len(),
                    prefill_token_count + decode_index,
                    decode_index,
                    repair_inputs,
                    false,
                )
                .with_context(|| stage_chain_error_context(args))?;
                commit_tokens = repaired_commit_tokens(
                    &draft_tokens,
                    decision.accepted_before_reject,
                    repair_input_count,
                    &repair.predicted_tokens,
                )?;
                reply_stats.merge(repair.stats);
                decode_ms += repair.elapsed_ms;
                speculative_stats.recovery_reverify_tokens += repair_inputs.len();
                speculative_stats.recovery_ms += repair.elapsed_ms;
                speculative_stats.recovery_reverify_elapsed_ms += repair.elapsed_ms;
                speculative_stats.recovery_reverify_write_ms += repair.write_ms;
                speculative_stats.recovery_reverify_wait_ms += repair.wait_ms;
                speculative_stats.recovery_reverify_compute_us +=
                    repair.stats.verify_span_compute_us;
                speculative_stats.recovery_reverify_forward_write_us +=
                    repair.stats.verify_span_forward_write_us;
                speculative_stats.recovery_reverify_downstream_wait_us +=
                    repair.stats.verify_span_downstream_wait_us;
                speculative_stats.recovery_reverify_stage_count +=
                    repair.stats.verify_span_stage_count;
            }
        }
        let mut reached_eog = false;
        for predicted in commit_tokens {
            current = predicted;
            generated.push(current);
            context_tokens.push(current);
            if let Some(ngram) = ngram.as_deref_mut() {
                ngram.observe_accepted(session_id, &context_tokens)?;
            }
            if tokenizer.token_is_eog(current)? {
                reached_eog = true;
            }
            let piece = tokenizer.detokenize(&[current])?;
            let filtered = output_filter.push(&piece);
            if filtered.suppressed_thinking {
                suppressed_thinking_tokens += 1;
            }
            if filtered.text.chars().any(|ch| !ch.is_whitespace()) {
                saw_visible_output = true;
            }
            print!("{}", filtered.text);
            io::stdout().flush().ok();
            if reached_eog || generated.len() >= args.max_new_tokens {
                break;
            }
        }
        speculative_stats.adaptive_window_final = adaptive_window;
        if decision.rejected() || reached_eog {
            if let Some(draft) = draft.as_deref_mut() {
                draft.reset_to_context(&context_tokens)?;
            }
        }
        if reached_eog {
            break;
        }
    }
    let tail = output_filter.finish();
    if tail.suppressed_thinking {
        suppressed_thinking_tokens += 1;
    }
    if tail.text.chars().any(|ch| !ch.is_whitespace()) {
        saw_visible_output = true;
    }
    print!("{}", tail.text);
    println!();

    if !saw_visible_output {
        eprintln!(
            "warning: generated no visible non-whitespace text; first generated token ids: {:?}",
            generated.iter().take(16).collect::<Vec<_>>()
        );
    }
    if suppressed_thinking_tokens > 0 {
        eprintln!(
            "warning: suppressed {suppressed_thinking_tokens} generated tokens inside <think> blocks"
        );
    }

    write_stage_message(
        &mut stream,
        &StageWireMessage::stop_with_identity(wire_dtype, request_id, wire_session_id),
        wire_dtype,
    )
    .with_context(|| stage_chain_error_context(args))?;
    let stop_reply = recv_reply(&mut stream).with_context(|| stage_chain_error_context(args))?;
    if stop_reply.kind != WireReplyKind::Ack {
        bail!("expected stop ACK, got {:?}", stop_reply.kind);
    }

    let wallblock_ms = elapsed_ms(wall_started);
    let generated_tokens = generated.len();
    let tpot_ms = if generated_tokens == 0 {
        0.0
    } else {
        decode_ms / generated_tokens as f64
    };
    let tpot_after_first_ms = if generated_tokens <= 1 {
        0.0
    } else {
        (decode_ms - first_decode_ms.unwrap_or(0.0)) / (generated_tokens - 1) as f64
    };
    print_stats(Stats {
        prompt_tokens: token_ids.len(),
        prefill_tokens: prefill_token_count,
        prefill_chunks: prefill_chunk_count,
        generated_tokens,
        suppressed_thinking_tokens,
        tokenize_ms,
        prefill_ms,
        decode_ms,
        wallblock_ms,
        first_time_to_token_ms: first_time_to_token_ms.unwrap_or(0.0),
        tpot_ms,
        tpot_after_first_ms,
        reply_stats,
        speculative_stats,
    });

    Ok(())
}

struct Stats {
    prompt_tokens: usize,
    prefill_tokens: usize,
    prefill_chunks: usize,
    generated_tokens: usize,
    suppressed_thinking_tokens: usize,
    tokenize_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    wallblock_ms: f64,
    first_time_to_token_ms: f64,
    tpot_ms: f64,
    tpot_after_first_ms: f64,
    reply_stats: StageReplyStats,
    speculative_stats: SpeculativeStats,
}

fn stage_chain_error_context(args: &BinaryReplArgs) -> String {
    let mut message = format!(
        "target stage chain did not respond within {}s; a remote stage may have exited or stopped forwarding",
        args.decode_timeout_secs.max(1)
    );
    if let Some(hint) = args.diagnostics_hint.as_deref() {
        message.push('\n');
        message.push_str(hint);
    }
    message
}

#[derive(Default)]
struct SpeculativeStats {
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
    primary_verify_write_ms: f64,
    primary_verify_wait_ms: f64,
    primary_verify_compute_us: i64,
    primary_verify_forward_write_us: i64,
    primary_verify_downstream_wait_us: i64,
    primary_verify_total_us: i64,
    primary_verify_stage_count: i64,
    checkpoint_ms: f64,
    recovery_restores: usize,
    recovery_decode_repairs: usize,
    recovery_decode_elapsed_ms: f64,
    recovery_reverify_tokens: usize,
    recovery_ms: f64,
    recovery_restore_ms: f64,
    recovery_reverify_elapsed_ms: f64,
    recovery_reverify_write_ms: f64,
    recovery_reverify_wait_ms: f64,
    recovery_reverify_compute_us: i64,
    recovery_reverify_forward_write_us: i64,
    recovery_reverify_downstream_wait_us: i64,
    recovery_reverify_stage_count: i64,
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

impl SpeculativeStats {
    fn observe_primary_verify(&mut self, reply: &VerifySpanReply, token_count: usize) {
        self.primary_verify_requests += 1;
        self.primary_verify_tokens += token_count;
        self.primary_verify_elapsed_ms += reply.elapsed_ms;
        self.primary_verify_write_ms += reply.write_ms;
        self.primary_verify_wait_ms += reply.wait_ms;
        self.primary_verify_compute_us += reply.stats.verify_span_compute_us;
        self.primary_verify_forward_write_us += reply.stats.verify_span_forward_write_us;
        self.primary_verify_downstream_wait_us += reply.stats.verify_span_downstream_wait_us;
        self.primary_verify_total_us += reply.stats.verify_span_total_us;
        self.primary_verify_stage_count += reply.stats.verify_span_stage_count;
        self.checkpoint_ms += us_to_ms(reply.stats.checkpoint_total_us);
    }

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
}

struct DecodeStepReply {
    predicted: i32,
    stats: StageReplyStats,
    elapsed_ms: f64,
}

struct VerifySpanReply {
    predicted_tokens: Vec<i32>,
    stats: StageReplyStats,
    write_ms: f64,
    wait_ms: f64,
    elapsed_ms: f64,
}

struct SessionControlReply {
    stats: StageReplyStats,
    elapsed_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn send_decode_step(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    prefill_token_count: usize,
    decode_index: usize,
    current: i32,
) -> Result<DecodeStepReply> {
    let decode_started = Instant::now();
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?;
    state.decode_step = i32::try_from(decode_index).context("decode step exceeds i32")?;
    state.current_token = current;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: i32::try_from(prefill_token_count + decode_index)
            .context("decode position exceeds i32")?,
        token_count: 1,
        state,
        request_id,
        session_id,
        sampling: None,
        tokens: vec![current],
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send decode step {decode_index}"))?;
    let reply = recv_reply(&mut *stream)
        .with_context(|| format!("receive decode step {decode_index} reply"))?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    Ok(DecodeStepReply {
        predicted: reply.predicted,
        stats: reply.stats,
        elapsed_ms: elapsed_ms(decode_started),
    })
}

#[allow(clippy::too_many_arguments)]
fn send_verify_span(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    pos_start: usize,
    decode_index: usize,
    tokens: &[i32],
    checkpoint: bool,
) -> Result<VerifySpanReply> {
    if tokens.is_empty() {
        bail!("verify span requires at least one token");
    }
    let verify_started = Instant::now();
    let mut state = StageStateHeader::new(WireMessageKind::VerifySpan, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?;
    state.decode_step = i32::try_from(decode_index).context("decode step exceeds i32")?;
    state.current_token = tokens[0];
    state.source_stage_index = -1;
    if !checkpoint {
        state.flags |= state_flags::SKIP_VERIFY_CHECKPOINT;
    }
    let message = StageWireMessage {
        kind: WireMessageKind::VerifySpan,
        pos_start: i32::try_from(pos_start).context("verify span position exceeds i32")?,
        token_count: i32::try_from(tokens.len()).context("verify span exceeds i32")?,
        state,
        request_id,
        session_id,
        sampling: None,
        tokens: tokens.to_vec(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    let write_started = Instant::now();
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send verify span at decode step {decode_index}"))?;
    let write_ms = elapsed_ms(write_started);
    let wait_started = Instant::now();
    let reply = recv_reply(&mut *stream)
        .with_context(|| format!("receive verify span {decode_index} reply"))?;
    let wait_ms = elapsed_ms(wait_started);
    if reply.kind != WireReplyKind::PredictedTokens {
        bail!("expected predicted-tokens reply, got {:?}", reply.kind);
    }
    Ok(VerifySpanReply {
        predicted_tokens: reply.predicted_tokens,
        stats: reply.stats,
        write_ms,
        wait_ms,
        elapsed_ms: elapsed_ms(verify_started),
    })
}

fn print_stats(stats: Stats) {
    let prompt_tps = if stats.prefill_ms > 0.0 {
        stats.prefill_tokens as f64 / (stats.prefill_ms / 1000.0)
    } else {
        0.0
    };
    let decode_tps = if stats.tpot_ms > 0.0 {
        1000.0 / stats.tpot_ms
    } else {
        0.0
    };

    let chunk_label = if stats.prefill_chunks == 1 {
        "chunk"
    } else {
        "chunks"
    };
    eprintln!("stats:");
    eprintln!(
        "  tokens   prompt={} prefill={} ({} {}) generated={} hidden_think={}",
        stats.prompt_tokens,
        stats.prefill_tokens,
        stats.prefill_chunks,
        chunk_label,
        stats.generated_tokens,
        stats.suppressed_thinking_tokens
    );
    eprintln!(
        "  time     tokenize={:.2}ms prefill={:.2}ms decode={:.2}ms wallblock={:.2}ms",
        stats.tokenize_ms, stats.prefill_ms, stats.decode_ms, stats.wallblock_ms
    );
    eprintln!(
        "  speed    prefill={:.2} tok/s decode={:.2} tok/s tpot={:.2}ms steady_tpot={:.2}ms",
        prompt_tps, decode_tps, stats.tpot_ms, stats.tpot_after_first_ms
    );
    eprintln!("  latency  ttft={:.2}ms", stats.first_time_to_token_ms);
    if !stats.reply_stats.is_empty() {
        eprintln!(
            "  kv       lookup hit={} miss={} error={} imported_pages={} imported_tokens={}",
            stats.reply_stats.kv_lookup_hits,
            stats.reply_stats.kv_lookup_misses,
            stats.reply_stats.kv_lookup_errors,
            stats.reply_stats.kv_imported_pages,
            stats.reply_stats.kv_imported_tokens
        );
        eprintln!(
            "  kv       recorded_pages={} recorded_bytes={} hit_stages={} record_stages={}",
            stats.reply_stats.kv_recorded_pages,
            format_bytes(stats.reply_stats.kv_recorded_bytes.max(0) as u64),
            format_stage_mask(stats.reply_stats.kv_hit_stage_mask),
            format_stage_mask(stats.reply_stats.kv_record_stage_mask)
        );
    } else {
        eprintln!("  kv       no lookup/record events reported");
    }
    if stats.speculative_stats.windows > 0 {
        let acceptance = if stats.speculative_stats.draft_tokens == 0 {
            0.0
        } else {
            100.0 * stats.speculative_stats.accepted_tokens as f64
                / stats.speculative_stats.draft_tokens as f64
        };
        eprintln!(
            "  spec     windows={} proposed={} accepted={} rejected={} accept_rate={:.1}%",
            stats.speculative_stats.windows,
            stats.speculative_stats.draft_tokens,
            stats.speculative_stats.accepted_tokens,
            stats.speculative_stats.rejected_tokens,
            acceptance
        );
        let avg_reject_pos = if stats.speculative_stats.rejected_windows == 0 {
            0.0
        } else {
            stats.speculative_stats.first_reject_position_sum as f64
                / stats.speculative_stats.rejected_windows as f64
        };
        eprintln!(
            "  spec     full_accept_windows={} accepted_stop_windows={} rejected_windows={} early_reject_windows={} tail_reject_windows={} early_reject_stop_windows={} repair_required_windows={} avg_reject_pos={:.2}",
            stats.speculative_stats.full_accept_windows,
            stats.speculative_stats.accepted_stop_windows,
            stats.speculative_stats.rejected_windows,
            stats.speculative_stats.early_reject_windows,
            stats.speculative_stats.tail_reject_windows,
            stats.speculative_stats.early_reject_stop_windows,
            stats.speculative_stats.repair_required_windows,
            avg_reject_pos
        );
        if stats.speculative_stats.adaptive_window_max > 0 {
            let avg_window = stats.speculative_stats.adaptive_window_sum as f64
                / stats.speculative_stats.windows.max(1) as f64;
            eprintln!(
                "  spec     window_policy={} start={} final={} max={} avg={:.2} min={} max_seen={} grows={} shrinks={}",
                if stats.speculative_stats.adaptive_window_enabled {
                    "adaptive"
                } else {
                    "fixed"
                },
                stats.speculative_stats.adaptive_window_start,
                stats.speculative_stats.adaptive_window_final,
                stats.speculative_stats.adaptive_window_max,
                avg_window,
                stats.speculative_stats.adaptive_window_min,
                stats.speculative_stats.adaptive_window_max_seen,
                stats.speculative_stats.adaptive_window_grows,
                stats.speculative_stats.adaptive_window_shrinks
            );
        }
        if stats.speculative_stats.primary_verify_requests > 0 {
            let avg_span_ms = stats.speculative_stats.primary_verify_elapsed_ms
                / stats.speculative_stats.primary_verify_requests as f64;
            let ms_per_token = stats.speculative_stats.primary_verify_elapsed_ms
                / stats.speculative_stats.primary_verify_tokens.max(1) as f64;
            let primary_stage_total_ms = us_to_ms(stats.speculative_stats.primary_verify_total_us);
            let client_unaccounted_ms = (stats.speculative_stats.primary_verify_elapsed_ms
                - primary_stage_total_ms)
                .max(0.0);
            eprintln!(
                "  spec     verify_wall_ms requests={} tokens={} elapsed={:.2} write={:.2} wait={:.2} avg_span={:.2} ms_per_token={:.2} client_unaccounted={:.2}",
                stats.speculative_stats.primary_verify_requests,
                stats.speculative_stats.primary_verify_tokens,
                stats.speculative_stats.primary_verify_elapsed_ms,
                stats.speculative_stats.primary_verify_write_ms,
                stats.speculative_stats.primary_verify_wait_ms,
                avg_span_ms,
                ms_per_token,
                client_unaccounted_ms
            );
            let primary_verify_tok_s = if stats.speculative_stats.primary_verify_elapsed_ms > 0.0 {
                1000.0 * stats.speculative_stats.primary_verify_tokens as f64
                    / stats.speculative_stats.primary_verify_elapsed_ms
            } else {
                0.0
            };
            eprintln!(
                "  spec     verify_primary_breakdown_ms total={:.2} compute={:.2} forward={:.2} downstream_wait={:.2} stages={} verify_tok_s={:.2}",
                primary_stage_total_ms,
                us_to_ms(stats.speculative_stats.primary_verify_compute_us),
                us_to_ms(stats.speculative_stats.primary_verify_forward_write_us),
                us_to_ms(stats.speculative_stats.primary_verify_downstream_wait_us),
                stats.speculative_stats.primary_verify_stage_count,
                primary_verify_tok_s
            );
        }
        if stats.speculative_stats.recovery_restores > 0 {
            if stats.speculative_stats.checkpoint_ms > 0.0 {
                eprintln!(
                    "  spec     checkpoint_ms={:.2} recovery_restores={} recovery_decode_repairs={} recovery_decode_ms={:.2} recovery_reverify_tokens={} recovery_ms={:.2}",
                    stats.speculative_stats.checkpoint_ms,
                    stats.speculative_stats.recovery_restores,
                    stats.speculative_stats.recovery_decode_repairs,
                    stats.speculative_stats.recovery_decode_elapsed_ms,
                    stats.speculative_stats.recovery_reverify_tokens,
                    stats.speculative_stats.recovery_ms
                );
            } else {
                eprintln!(
                    "  spec     recovery_restores={} recovery_decode_repairs={} recovery_decode_ms={:.2} recovery_reverify_tokens={} recovery_ms={:.2}",
                    stats.speculative_stats.recovery_restores,
                    stats.speculative_stats.recovery_decode_repairs,
                    stats.speculative_stats.recovery_decode_elapsed_ms,
                    stats.speculative_stats.recovery_reverify_tokens,
                    stats.speculative_stats.recovery_ms
                );
            }
        } else if stats.speculative_stats.checkpoint_ms > 0.0 {
            eprintln!(
                "  spec     checkpoint_ms={:.2}",
                stats.speculative_stats.checkpoint_ms
            );
        }
        if stats.reply_stats.checkpoint_total_us > 0 {
            eprintln!(
                "  spec     checkpoint_breakdown_ms total={:.2} flush={:.2} prefill_drain={:.2} local={:.2} downstream_write={:.2} downstream_wait={:.2} drained_replies={}",
                us_to_ms(stats.reply_stats.checkpoint_total_us),
                us_to_ms(stats.reply_stats.checkpoint_flush_us),
                us_to_ms(stats.reply_stats.checkpoint_prefill_drain_us),
                us_to_ms(stats.reply_stats.checkpoint_local_us),
                us_to_ms(stats.reply_stats.checkpoint_downstream_write_us),
                us_to_ms(stats.reply_stats.checkpoint_downstream_wait_us),
                stats.reply_stats.checkpoint_prefill_drained_replies
            );
        }
        if stats.reply_stats.restore_total_us > 0 {
            eprintln!(
                "  spec     restore_breakdown_ms total={:.2} flush={:.2} prefill_drain={:.2} local={:.2} downstream_write={:.2} downstream_wait={:.2} drained_replies={}",
                us_to_ms(stats.reply_stats.restore_total_us),
                us_to_ms(stats.reply_stats.restore_flush_us),
                us_to_ms(stats.reply_stats.restore_prefill_drain_us),
                us_to_ms(stats.reply_stats.restore_local_us),
                us_to_ms(stats.reply_stats.restore_downstream_write_us),
                us_to_ms(stats.reply_stats.restore_downstream_wait_us),
                stats.reply_stats.restore_prefill_drained_replies
            );
        }
        if stats.reply_stats.verify_span_total_us > 0 {
            let verify_total_ms = us_to_ms(stats.reply_stats.verify_span_total_us);
            let verify_tok_s = if verify_total_ms > 0.0 {
                1000.0 * stats.speculative_stats.draft_tokens as f64 / verify_total_ms
            } else {
                0.0
            };
            eprintln!(
                "  spec     verify_breakdown_ms total={:.2} compute={:.2} forward={:.2} downstream_wait={:.2} stages={} proposed_tok_s={:.2}",
                verify_total_ms,
                us_to_ms(stats.reply_stats.verify_span_compute_us),
                us_to_ms(stats.reply_stats.verify_span_forward_write_us),
                us_to_ms(stats.reply_stats.verify_span_downstream_wait_us),
                stats.reply_stats.verify_span_stage_count,
                verify_tok_s
            );
            let protocol_avg_span = if stats.reply_stats.verify_span_request_count > 0 {
                stats.reply_stats.verify_span_token_count as f64
                    / stats.reply_stats.verify_span_request_count as f64
            } else {
                0.0
            };
            eprintln!(
                "  spec     verify_batch_stats protocol_requests={} protocol_tokens={} max_span={} avg_span={:.2} checkpointed_requests={} skip_checkpoint_requests={}",
                stats.reply_stats.verify_span_request_count,
                stats.reply_stats.verify_span_token_count,
                stats.reply_stats.verify_span_max_tokens,
                protocol_avg_span,
                stats.reply_stats.verify_span_checkpointed_requests,
                stats.reply_stats.verify_span_skip_checkpoint_requests
            );
        }
        if stats.speculative_stats.recovery_reverify_elapsed_ms > 0.0 {
            eprintln!(
                "  spec     recovery_reverify_breakdown_ms elapsed={:.2} write={:.2} wait={:.2} stage_compute={:.2} stage_forward={:.2} stage_downstream_wait={:.2} stages={}",
                stats.speculative_stats.recovery_reverify_elapsed_ms,
                stats.speculative_stats.recovery_reverify_write_ms,
                stats.speculative_stats.recovery_reverify_wait_ms,
                us_to_ms(stats.speculative_stats.recovery_reverify_compute_us),
                us_to_ms(stats.speculative_stats.recovery_reverify_forward_write_us),
                us_to_ms(stats.speculative_stats.recovery_reverify_downstream_wait_us),
                stats.speculative_stats.recovery_reverify_stage_count
            );
        }
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

    #[cfg(test)]
    fn tail_reject(self) -> bool {
        self.kind == VerifySpanDecisionKind::TailReject
    }
}

fn classify_verify_span<F>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> Result<VerifySpanDecision>
where
    F: FnMut(i32) -> Result<bool>,
{
    if predicted_tokens.len() < draft_tokens.len() {
        bail!(
            "verify span returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            draft_tokens.len()
        );
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
) -> Result<Vec<i32>> {
    if repaired_predictions.len() < repair_input_count {
        bail!(
            "recovery verify returned too few tokens: expected {} got {:?}",
            repair_input_count,
            repaired_predictions
        );
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

fn send_session_control(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    kind: WireMessageKind,
) -> Result<SessionControlReply> {
    if !kind.is_session_control() {
        bail!("session control requires a session-control message kind");
    }
    let started = Instant::now();
    let mut state = StageStateHeader::new(kind, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind,
        pos_start: 0,
        token_count: 0,
        state,
        request_id,
        session_id,
        sampling: None,
        tokens: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send session control {kind:?}"))?;
    let reply =
        recv_reply(&mut *stream).with_context(|| format!("receive session control {kind:?}"))?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected session-control ACK, got {:?}", reply.kind);
    }
    Ok(SessionControlReply {
        stats: reply.stats,
        elapsed_ms: elapsed_ms(started),
    })
}

struct ReplPrefillChunk<'a> {
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    pos_start: usize,
    prefill_token_count: usize,
    tokens: &'a [i32],
}

fn send_prefill_chunk(
    stream: &mut std::net::TcpStream,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    chunk: ReplPrefillChunk<'_>,
) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id = i32::try_from(chunk.prompt_index).context("prompt index exceeds i32")?;
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
        tokens: chunk.tokens.to_vec(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send prefill chunk at {}", chunk.pos_start))?;
    let reply = recv_reply(&mut *stream)
        .with_context(|| format!("receive prefill chunk ACK at {}", chunk.pos_start))?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn us_to_ms(us: i64) -> f64 {
    us as f64 / 1000.0
}

fn nonzero_min(current: usize, value: usize) -> usize {
    if current == 0 {
        value
    } else {
        current.min(value)
    }
}

fn default_session_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("prompt-{}-{millis}", std::process::id())
}

fn stable_wire_id(parts: &[&[u8]]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let id = u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("8-byte digest prefix"),
    );
    if id == 0 {
        1
    } else {
        id
    }
}

struct PromptHistory {
    prompts: Vec<String>,
    path: Option<PathBuf>,
}

struct NgramSource;

impl NgramSource {
    fn open(_args: &BinaryReplArgs, _session_id: &str) -> Result<Self> {
        bail!("ngram speculative prompting is not imported into mesh-llm")
    }

    fn observe_sequence(&mut self, _session_id: &str, _tokens: &[i32]) -> Result<()> {
        Ok(())
    }

    fn observe_accepted(&mut self, _session_id: &str, _context_tokens: &[i32]) -> Result<()> {
        Ok(())
    }

    fn propose(
        &mut self,
        _session_id: &str,
        _context_tokens: &[i32],
        _remaining: usize,
    ) -> Result<Vec<i32>> {
        Ok(Vec::new())
    }
}

struct DraftRunner {
    path: PathBuf,
    window: usize,
    _model: StageModel,
    session: StageSession,
}

impl DraftRunner {
    fn open(path: &Path, ctx_size: u32, n_gpu_layers: i32, window: usize) -> Result<Self> {
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
                ctx_size,
                n_gpu_layers,
                selected_backend_device: None,
                cache_type_k: GGML_TYPE_F16,
                cache_type_v: GGML_TYPE_F16,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                filter_tensors_on_load: true,
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

fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let layer_count = info
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .context("model has no layer-indexed tensors")?;
    Ok(layer_count)
}

impl PromptHistory {
    fn load(path: Option<&Path>) -> Result<Self> {
        let prompts = match path {
            Some(path) if path.exists() => {
                let file = fs::File::open(path)
                    .with_context(|| format!("open prompt history {}", path.display()))?;
                io::BufReader::new(file)
                    .lines()
                    .collect::<io::Result<Vec<_>>>()
                    .with_context(|| format!("read prompt history {}", path.display()))?
                    .into_iter()
                    .filter(|line| !line.trim().is_empty())
                    .collect()
            }
            _ => Vec::new(),
        };
        Ok(Self {
            prompts,
            path: path.map(Path::to_path_buf),
        })
    }

    fn push(&mut self, prompt: &str) -> Result<()> {
        if self.prompts.last().is_some_and(|last| last == prompt) {
            return Ok(());
        }
        self.prompts.push(prompt.to_string());
        if let Some(path) = self.path.as_ref() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create prompt history dir {}", parent.display()))?;
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("open prompt history {}", path.display()))?;
            writeln!(file, "{prompt}")
                .with_context(|| format!("append prompt history {}", path.display()))?;
        }
        Ok(())
    }

    fn get(&self, one_based_index: usize) -> Option<&str> {
        one_based_index
            .checked_sub(1)
            .and_then(|index| self.prompts.get(index))
            .map(String::as_str)
    }

    fn print(&self) {
        for (index, prompt) in self.prompts.iter().enumerate() {
            println!("{:>4}: {}", index + 1, prompt);
        }
    }
}

fn prompt_editor(history: &PromptHistory) -> Result<DefaultEditor> {
    let mut editor = DefaultEditor::new().context("create prompt line editor")?;
    for prompt in &history.prompts {
        let _ = editor.add_history_entry(prompt);
    }
    Ok(editor)
}

fn read_history_prompt(editor: &mut DefaultEditor, prompt: &str) -> Result<Option<String>> {
    match editor.readline(prompt) {
        Ok(input) => Ok(Some(input)),
        Err(ReadlineError::Interrupted) => {
            println!("^C");
            Ok(Some(String::new()))
        }
        Err(ReadlineError::Eof) => {
            println!();
            Ok(None)
        }
        Err(error) => Err(anyhow!(error)).context("read prompt line"),
    }
}

fn parse_prompt_json_command(input: &str) -> Result<Option<(String, String)>> {
    const PREFIX: &str = ":prompt-json\t";
    let Some(rest) = input.strip_prefix(PREFIX) else {
        return Ok(None);
    };
    let (session_json, prompt_json) = rest
        .split_once('\t')
        .context(":prompt-json requires session and prompt JSON strings")?;
    let session_id: String =
        serde_json::from_str(session_json).context("parse :prompt-json session")?;
    let prompt: String = serde_json::from_str(prompt_json).context("parse :prompt-json prompt")?;
    Ok(Some((session_id, prompt)))
}

#[cfg(test)]
mod prompt_json_tests {
    use super::*;

    #[test]
    fn parse_prompt_json_preserves_multiline_prompt() -> Result<()> {
        let command = concat!(
            ":prompt-json\t",
            "\"trajectory-7\"\t",
            "\"fn main() {\\n    println!(\\\"hi\\\");\\n}\""
        );

        let parsed = parse_prompt_json_command(command)?.expect("prompt-json command");
        assert_eq!(parsed.0, "trajectory-7");
        assert_eq!(parsed.1, "fn main() {\n    println!(\"hi\");\n}");
        Ok(())
    }

    #[test]
    fn parse_prompt_json_ignores_regular_prompt() -> Result<()> {
        assert!(parse_prompt_json_command("plain prompt")?.is_none());
        Ok(())
    }
}

fn write_local_configs(
    args: &PromptArgs,
    run_id: &str,
    run_dir: &Path,
    kv_mode: &str,
    stages: &[LocalStage],
    metrics_otlp_url: &str,
    hf_package_ref: bool,
) -> Result<()> {
    for stage in stages {
        let local_kv_page_root = run_dir.join(format!("kv-pages-stage-{}", stage.stage_index));
        let kv_page_root = stage
            .remote
            .as_ref()
            .map(|remote| json!(remote.kv_page_root.clone()))
            .unwrap_or_else(|| json!(local_kv_page_root));
        let kv_uds_path = stage
            .remote
            .as_ref()
            .map(|remote| json!(remote.kv_uds_path.clone()))
            .unwrap_or_else(|| json!(stage.kv_uds_path));
        if stage.remote.is_none() {
            fs::create_dir_all(&local_kv_page_root)
                .with_context(|| format!("create {}", local_kv_page_root.display()))?;
        }
        let kv_config = json!({
            "node_id": format!("local-stage-{}", stage.stage_index),
            "cluster_id": "local-binary-repl",
            "run_id": run_id,
            "page_root": kv_page_root,
            "uds_path": kv_uds_path,
            "quic_bind_addr": format!("127.0.0.1:{}", 19443 + stage.stage_index),
            "peers": {},
            "cache": { "max_local_bytes": null },
            "metrics": {
                "otlp_grpc_endpoint": metrics_otlp_url,
                "export_interval_ms": 5000
            },
            "compression": {
                "warm_tier_codec": "lz4-block",
                "compress_remote_only_pages": false
            },
            "security": { "mtls_enabled": false }
        });
        write_json(&stage.kv_config_path, &kv_config)?;

        let upstream = if stage.stage_index == 0 {
            json!({
                "stage_id": "driver",
                "stage_index": 0,
                "endpoint": "driver"
            })
        } else {
            let previous = &stages[stage.stage_index - 1];
            json!({
                "stage_id": previous.stage_id,
                "stage_index": previous.stage_index,
                "endpoint": format!("tcp://{}", previous.endpoint_addr)
            })
        };
        let downstream = stages
            .get(stage.stage_index + 1)
            .map(|next| {
                json!({
                    "stage_id": next.stage_id,
                    "stage_index": next.stage_index,
                    "endpoint": format!("tcp://{}", next.endpoint_addr)
                })
            })
            .unwrap_or_else(|| json!(null));
        let stage_model_path = if hf_package_ref {
            json!(args.model_path.to_str().unwrap_or(""))
        } else {
            stage
                .remote
                .as_ref()
                .map(|remote| json!(remote.model_path.clone()))
                .unwrap_or_else(|| json!(stage.model_path))
        };
        let stage_kv_uds_path = stage
            .remote
            .as_ref()
            .map(|remote| json!(remote.kv_uds_path.clone()))
            .unwrap_or_else(|| json!(stage.kv_uds_path));
        let stage_config = json!({
            "run_id": run_id,
            "topology_id": "local-binary-kv-repl",
            "model_id": args.model_id,
            "model_path": stage_model_path,
            "stage_id": stage.stage_id,
            "stage_index": stage.stage_index,
            "layer_start": stage.layer_start,
            "layer_end": stage.layer_end,
            "ctx_size": args.ctx_size,
            "n_gpu_layers": args.n_gpu_layers,
            "filter_tensors_on_load": true,
            "load_mode": if hf_package_ref || stage.remote.is_some() { "layer-package" } else { "artifact-slice" },
            "bind_addr": stage.bind_addr,
            "upstream": upstream,
            "downstream": downstream,
            "kv_server": {
                "uds_path": stage_kv_uds_path,
                "mode": kv_mode,
                "page_size_tokens": args.kv_page_size_tokens,
                "correctness_mode": kv_mode == "correctness"
            }
        });
        write_json(&stage.config_path, &stage_config)?;
    }
    Ok(())
}

fn materialize_stage_artifacts(args: &PromptArgs, stages: &[LocalStage]) -> Result<()> {
    let Some(first_stage) = stages.first() else {
        bail!("no stages to materialize");
    };
    let model_cache_dir = first_stage
        .model_path
        .parent()
        .context("stage model path has no parent")?;
    fs::create_dir_all(model_cache_dir)
        .with_context(|| format!("create model cache dir {}", model_cache_dir.display()))?;

    let total = stages.len();
    for (done, stage) in stages.iter().enumerate() {
        if stage_artifact_available(&stage.model_path)? {
            print_progress(
                "materializing GGUF shards",
                done + 1,
                total,
                &format!(
                    "cached {} layers {}..{}",
                    stage.stage_id, stage.layer_start, stage.layer_end
                ),
                Duration::ZERO,
            );
            eprintln!();
            continue;
        }
        let mut command = Command::new(&args.model_slice_bin);
        command.args([
            "write",
            path_str(&args.model_path)?,
            "--layers",
            &format!("{}..{}", stage.layer_start, stage.layer_end),
            "--out",
            path_str(&stage.model_path)?,
            "--stage-index",
            &stage.stage_index.to_string(),
        ]);
        if stage.layer_start == 0 {
            command.arg("--include-embeddings");
        }
        if stage.stage_index + 1 == total {
            command.arg("--include-output");
        }
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let label = format!(
            "{} layers {}..{}",
            stage.stage_id, stage.layer_start, stage.layer_end
        );
        let status = run_with_progress(command, "materializing GGUF shards", done, total, &label)
            .with_context(|| format!("run llama-model-slice for {}", stage.stage_id))?;
        if !status.success() {
            bail!(
                "llama-model-slice failed for {} with status {status}",
                stage.stage_id
            );
        }
    }

    Ok(())
}

fn materialize_model_package(args: &PromptArgs, package_dir: &Path) -> Result<()> {
    if package_artifact_available(package_dir)? {
        eprintln!(
            "materializing GGUF package: cached {}",
            package_dir.display()
        );
        return Ok(());
    }

    if package_dir.exists() {
        fs::remove_dir_all(package_dir)
            .with_context(|| format!("remove incomplete package {}", package_dir.display()))?;
    }
    let parent = package_dir
        .parent()
        .context("model package cache path has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create model package cache dir {}", parent.display()))?;

    let mut command = Command::new(&args.model_slice_bin);
    command.args([
        "write-package",
        path_str(&args.model_path)?,
        "--out-dir",
        path_str(package_dir)?,
        "--model-id",
        &args.model_id,
    ]);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let status = run_with_progress(
        command,
        "materializing GGUF package",
        0,
        1,
        &format!("{} -> {}", args.model_path.display(), package_dir.display()),
    )
    .with_context(|| "run llama-model-slice write-package")?;
    if !status.success() {
        bail!("llama-model-slice write-package failed with status {status}");
    }

    fs::write(
        package_dir.join(".complete"),
        format!(
            "model_id={}\nsource={}\n",
            args.model_id,
            args.model_path.display()
        ),
    )
    .with_context(|| format!("write package completion marker {}", package_dir.display()))?;
    Ok(())
}

fn run_with_progress(
    mut command: Command,
    title: &str,
    completed: usize,
    total: usize,
    label: &str,
) -> Result<std::process::ExitStatus> {
    let started = Instant::now();
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    loop {
        print_progress(title, completed, total, label, started.elapsed());
        if let Some(status) = child.try_wait()? {
            print_progress(title, completed + 1, total, label, started.elapsed());
            eprintln!();
            return Ok(status);
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn print_progress(title: &str, completed: usize, total: usize, label: &str, elapsed: Duration) {
    let width = 24usize;
    let filled = completed
        .saturating_mul(width)
        .checked_div(total)
        .unwrap_or(0);
    let empty = width.saturating_sub(filled);
    eprint!(
        "\r\x1b[2K{} [{}{}] {}/{} {:>5.1}s {}",
        title,
        "#".repeat(filled),
        "-".repeat(empty),
        completed,
        total,
        elapsed.as_secs_f64(),
        truncate_label(label, 96)
    );
    io::stderr().flush().ok();
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    let char_count = label.chars().count();
    if char_count <= max_chars {
        return label.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    format!("{}...", label.chars().take(keep).collect::<String>())
}

fn stage_artifact_available(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() > 0),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("stat {}", path.display())),
    }
}

fn package_artifact_available(path: &Path) -> Result<bool> {
    match fs::metadata(path.join("model-package.json")) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "stat package manifest {}",
                    path.join("model-package.json").display()
                )
            })
        }
    }
    match fs::metadata(path.join(".complete")) {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() > 0),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("stat package marker {}", path.display())),
    }
}

enum RemoteSyncEvent {
    HostStarted {
        host: String,
        stages: Vec<String>,
    },
    StepStarted {
        host: String,
        label: String,
    },
    StepProgress {
        host: String,
        label: String,
        detail: String,
        elapsed: Duration,
    },
    StepFinished {
        host: String,
        label: String,
        detail: String,
        elapsed: Duration,
    },
    HostFinished {
        host: String,
        elapsed: Duration,
    },
    HostFailed {
        host: String,
        error: String,
    },
}

fn rsync_remote_stage_inputs(
    args: &PromptArgs,
    stages: &[LocalStage],
    model_package_dir: &Path,
    hf_package_ref: bool,
) -> Result<()> {
    let mut stages_by_host: BTreeMap<String, Vec<&LocalStage>> = BTreeMap::new();
    for stage in stages {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        stages_by_host
            .entry(remote.host.clone())
            .or_default()
            .push(stage);
    }

    eprintln!(
        "remote sync: preparing inputs for {} stages on {} hosts in parallel",
        stages.len(),
        stages_by_host.len()
    );
    let (tx, rx) = mpsc::channel::<RemoteSyncEvent>();
    thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for (host, host_stages) in stages_by_host {
            let tx = tx.clone();
            handles.push(scope.spawn(move || {
                let started = Instant::now();
                send_remote_sync_event(
                    &tx,
                    RemoteSyncEvent::HostStarted {
                        host: host.clone(),
                        stages: host_stages
                            .iter()
                            .map(|stage| {
                                format!(
                                    "{}:{}..{}",
                                    stage.stage_id, stage.layer_start, stage.layer_end
                                )
                            })
                            .collect(),
                    },
                );
                let result = sync_remote_host_inputs(
                    args,
                    &host,
                    &host_stages,
                    model_package_dir,
                    hf_package_ref,
                    &tx,
                );
                match &result {
                    Ok(()) => send_remote_sync_event(
                        &tx,
                        RemoteSyncEvent::HostFinished {
                            host,
                            elapsed: started.elapsed(),
                        },
                    ),
                    Err(error) => send_remote_sync_event(
                        &tx,
                        RemoteSyncEvent::HostFailed {
                            host,
                            error: format!("{error:#}"),
                        },
                    ),
                }
                result
            }));
        }
        drop(tx);

        for event in rx {
            log_remote_sync_event(event);
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow!("remote sync worker panicked"))??;
        }
        Ok(())
    })?;
    eprintln!("remote sync: all remote inputs are ready");
    Ok(())
}

fn sync_remote_host_inputs(
    args: &PromptArgs,
    host: &str,
    stages: &[&LocalStage],
    model_package_dir: &Path,
    hf_package_ref: bool,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let first_remote = stages
        .first()
        .and_then(|stage| stage.remote.as_ref())
        .context("remote host has no stages")?;
    let mut mkdir_paths = BTreeSet::new();
    mkdir_paths.insert(first_remote.stage_dir.clone());
    mkdir_paths.insert(
        Path::new(&first_remote.stage_server_bin)
            .parent()
            .and_then(Path::to_str)
            .context("remote stage binary path has no parent")?
            .to_string(),
    );
    mkdir_paths.insert(
        Path::new(&first_remote.model_path)
            .parent()
            .and_then(Path::to_str)
            .context("remote model package path has no parent")?
            .to_string(),
    );
    for stage in stages.iter().skip(1) {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        mkdir_paths.insert(remote.stage_dir.clone());
    }

    let label = format!("create {} remote directories", mkdir_paths.len());
    let started = Instant::now();
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepStarted {
            host: host.to_string(),
            label: label.clone(),
        },
    );
    let mkdir_command = format!(
        "mkdir -p {}",
        mkdir_paths
            .iter()
            .map(|path| shell_quote(path))
            .collect::<Vec<_>>()
            .join(" ")
    );
    run_status(
        Command::new("ssh").arg("-n").arg(host).arg(mkdir_command),
        &format!("create remote dirs on {host}"),
    )?;
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepFinished {
            host: host.to_string(),
            label,
            detail: "done".to_string(),
            elapsed: started.elapsed(),
        },
    );

    rsync_to_host_cached(
        &args.stage_server_bin,
        host,
        &first_remote.stage_server_bin,
        &args.remote_root,
        tx,
    )?;
    rsync_to_host_cached(
        &args.kv_server_bin,
        host,
        &first_remote.kv_server_bin,
        &args.remote_root,
        tx,
    )?;
    if !hf_package_ref {
        rsync_dir_to_host_cached(model_package_dir, host, &first_remote.model_path, tx)?;
    }

    for stage in stages {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        rsync_to_host_with_progress(
            &stage.config_path,
            host,
            &remote.config_path,
            &format!("{} config", stage.stage_id),
            tx,
        )?;
        rsync_to_host_with_progress(
            &stage.kv_config_path,
            host,
            &remote.kv_config_path,
            &format!("kv-stage-{} config", stage.stage_index),
            tx,
        )?;
    }

    Ok(())
}

fn send_remote_sync_event(tx: &mpsc::Sender<RemoteSyncEvent>, event: RemoteSyncEvent) {
    let _ = tx.send(event);
}

fn log_remote_sync_event(event: RemoteSyncEvent) {
    match event {
        RemoteSyncEvent::HostStarted { host, stages } => {
            eprintln!("remote sync [{host}]: start {}", stages.join(", "));
        }
        RemoteSyncEvent::StepStarted { host, label } => {
            eprintln!("remote sync [{host}]: {label} ...");
        }
        RemoteSyncEvent::StepProgress {
            host,
            label,
            detail,
            elapsed,
        } => {
            eprintln!(
                "remote sync [{host}]: {label} still running ({:.1}s) {}",
                elapsed.as_secs_f64(),
                truncate_label(&detail, 96)
            );
        }
        RemoteSyncEvent::StepFinished {
            host,
            label,
            detail,
            elapsed,
        } => {
            eprintln!(
                "remote sync [{host}]: {label} {} ({:.1}s)",
                detail,
                elapsed.as_secs_f64()
            );
        }
        RemoteSyncEvent::HostFinished { host, elapsed } => {
            eprintln!(
                "remote sync [{host}]: complete ({:.1}s)",
                elapsed.as_secs_f64()
            );
        }
        RemoteSyncEvent::HostFailed { host, error } => {
            eprintln!("remote sync [{host}]: failed: {error}");
        }
    }
}

fn start_remote_stages(
    args: &PromptArgs,
    stages: &[LocalStage],
    metrics_otlp_url: &str,
    children: &mut Vec<ChildGuard>,
) -> Result<()> {
    for stage in stages {
        stop_remote_stage_listener(stage)?;
    }

    for stage in stages.iter().rev() {
        let remote = stage
            .remote
            .as_ref()
            .context("remote stage missing placement")?;
        let command_text = remote_stage_command(args, remote, metrics_otlp_url)?;
        let mut ssh = Command::new("ssh");
        ssh.arg("-n").arg(&remote.host).arg(command_text);
        ssh.stdin(Stdio::null());
        ssh.stdout(Stdio::null()).stderr(Stdio::null());
        children.push(ChildGuard::spawn(ssh)?);
    }

    Ok(())
}

fn stop_remote_stage_listener(stage: &LocalStage) -> Result<()> {
    let Some(remote) = stage.remote.as_ref() else {
        return Ok(());
    };
    let command = format!(
        concat!(
            "pids=$(lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null || true); ",
            "if [ -n \"$pids\" ]; then ",
            "echo stopping stale listener on :{port} >&2; ",
            "kill $pids 2>/dev/null || true; ",
            "sleep 0.5; ",
            "kill -9 $pids 2>/dev/null || true; ",
            "fi"
        ),
        port = stage.port
    );
    run_status(
        Command::new("ssh").arg("-n").arg(&remote.host).arg(command),
        &format!("stop stale stage listener {}:{}", remote.host, stage.port),
    )
}

fn add_stage_server_args(
    command: &mut Command,
    args: &PromptArgs,
    stage: &LocalStage,
    metrics_otlp_url: &str,
) -> Result<()> {
    command.args([
        "serve-binary",
        "--config",
        path_str(&stage.config_path)?,
        "--activation-width",
        &args.activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
        "--metrics-otlp-grpc",
        metrics_otlp_url,
        "--telemetry-queue-capacity",
        &args.stage_telemetry_queue_capacity.to_string(),
        "--telemetry-level",
        &args.stage_telemetry_level,
        "--max-inflight",
        &args.stage_max_inflight.to_string(),
        "--reply-credit-limit",
        &args.stage_reply_credit_limit.to_string(),
    ]);
    if !args.no_stage_async_prefill_forward {
        command.arg("--async-prefill-forward");
    }
    Ok(())
}

fn remote_stage_command(
    args: &PromptArgs,
    remote: &RemoteStage,
    metrics_otlp_url: &str,
) -> Result<String> {
    let wait_attempts = args.startup_timeout_secs.saturating_mul(5).max(1);
    let mut stage_args = vec![
        shell_quote(&remote.stage_server_bin),
        "serve-binary".to_string(),
        "--config".to_string(),
        shell_quote(&remote.config_path),
        "--activation-width".to_string(),
        shell_quote(&args.activation_width.to_string()),
        "--activation-wire-dtype".to_string(),
        shell_quote(&args.activation_wire_dtype),
        "--metrics-otlp-grpc".to_string(),
        shell_quote(metrics_otlp_url),
        "--telemetry-queue-capacity".to_string(),
        shell_quote(&args.stage_telemetry_queue_capacity.to_string()),
        "--telemetry-level".to_string(),
        shell_quote(&args.stage_telemetry_level),
        "--max-inflight".to_string(),
        shell_quote(&args.stage_max_inflight.to_string()),
        "--reply-credit-limit".to_string(),
        shell_quote(&args.stage_reply_credit_limit.to_string()),
    ];
    if !args.no_stage_async_prefill_forward {
        stage_args.push("--async-prefill-forward".to_string());
    }

    Ok(format!(
        concat!(
            "set -e; ",
            "cd {stage_dir}; ",
            "chmod +x {kv_bin} {stage_bin}; ",
            "rm -f {kv_sock} kv.pid stage.pid {stage_exit}; ",
            "{kv_bin} serve --config {kv_config} > {kv_log} 2>&1 & ",
            "kv_pid=$!; echo $kv_pid > kv.pid; ",
            "i=0; while [ ! -S {kv_sock} ] && [ $i -lt {wait_attempts} ]; do sleep 0.2; i=$((i + 1)); done; ",
            "[ -S {kv_sock} ]; ",
            "{stage_command} > {stage_log} 2>&1 & ",
            "stage_pid=$!; echo $stage_pid > stage.pid; ",
            "trap 'kill $stage_pid $kv_pid 2>/dev/null || true; wait $stage_pid 2>/dev/null || true; wait $kv_pid 2>/dev/null || true' INT TERM HUP EXIT; ",
            "set +e; ",
            "wait $stage_pid; stage_status=$?; ",
            "echo $stage_status > {stage_exit}; ",
            "kill $kv_pid 2>/dev/null || true; wait $kv_pid 2>/dev/null || true; ",
            "exit $stage_status"
        ),
        stage_dir = shell_quote(&remote.stage_dir),
        kv_bin = shell_quote(&remote.kv_server_bin),
        stage_bin = shell_quote(&remote.stage_server_bin),
        kv_sock = shell_quote(&remote.kv_uds_path),
        kv_config = shell_quote(&remote.kv_config_path),
        kv_log = shell_quote(&remote.kv_log_path),
        stage_command = stage_args.join(" "),
        stage_log = shell_quote(&remote.stage_log_path),
        stage_exit = shell_quote(&remote.stage_exit_path),
        wait_attempts = wait_attempts,
    ))
}

fn rsync_to_host_with_progress(
    local: &Path,
    host: &str,
    remote_path: &str,
    label: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let file_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input");
    let size = fs::metadata(local).ok().map(|metadata| metadata.len());
    let label = match size {
        Some(size) => format!("{label} ({file_name}, {})", format_bytes(size)),
        None => format!("{label} ({file_name})"),
    };
    let started = Instant::now();
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepStarted {
            host: host.to_string(),
            label: label.clone(),
        },
    );
    let mut command = Command::new("rsync");
    command
        .args(["-az", "--progress", "--chmod=ugo=rwX"])
        .arg(local)
        .arg(format!("{host}:{remote_path}"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    let (progress_tx, progress_rx) = std::sync::mpsc::channel::<String>();
    if let Some(stdout) = child.stdout.take() {
        spawn_rsync_progress_reader(stdout, progress_tx.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_rsync_progress_reader(stderr, progress_tx);
    }
    let mut rsync_progress = String::new();
    let mut last_report = Instant::now();
    loop {
        while let Ok(line) = progress_rx.try_recv() {
            rsync_progress = line;
        }
        if last_report.elapsed() >= Duration::from_secs(5) {
            send_remote_sync_event(
                tx,
                RemoteSyncEvent::StepProgress {
                    host: host.to_string(),
                    label: label.clone(),
                    detail: if rsync_progress.is_empty() {
                        "waiting for rsync progress".to_string()
                    } else {
                        rsync_progress.clone()
                    },
                    elapsed: started.elapsed(),
                },
            );
            last_report = Instant::now();
        }
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                bail!(
                    "rsync {} to {host}:{remote_path} failed with status {status}",
                    local.display()
                );
            }
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepFinished {
            host: host.to_string(),
            label,
            detail: "copied".to_string(),
            elapsed: started.elapsed(),
        },
    );
    Ok(())
}

fn rsync_dir_to_host_with_progress(
    local: &Path,
    host: &str,
    remote_path: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let dir_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input-dir");
    let label = format!("model package ({dir_name}/)");
    let started = Instant::now();
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepStarted {
            host: host.to_string(),
            label: label.clone(),
        },
    );

    run_status(
        Command::new("ssh")
            .arg("-n")
            .arg(host)
            .arg(format!("mkdir -p {}", shell_quote(remote_path))),
        &format!("create remote package dir {host}:{remote_path}"),
    )?;

    let mut command = Command::new("rsync");
    command
        .args(["-az", "--delete", "--progress", "--chmod=ugo=rwX"])
        .arg(format!("{}/", local.display()))
        .arg(format!("{host}:{}/", remote_path))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    let (progress_tx, progress_rx) = std::sync::mpsc::channel::<String>();
    if let Some(stdout) = child.stdout.take() {
        spawn_rsync_progress_reader(stdout, progress_tx.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_rsync_progress_reader(stderr, progress_tx);
    }
    let mut rsync_progress = String::new();
    let mut last_report = Instant::now();
    loop {
        while let Ok(line) = progress_rx.try_recv() {
            rsync_progress = line;
        }
        if last_report.elapsed() >= Duration::from_secs(5) {
            send_remote_sync_event(
                tx,
                RemoteSyncEvent::StepProgress {
                    host: host.to_string(),
                    label: label.clone(),
                    detail: if rsync_progress.is_empty() {
                        "waiting for rsync progress".to_string()
                    } else {
                        rsync_progress.clone()
                    },
                    elapsed: started.elapsed(),
                },
            );
            last_report = Instant::now();
        }
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                bail!(
                    "rsync directory {} to {host}:{remote_path} failed with status {status}",
                    local.display()
                );
            }
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
    send_remote_sync_event(
        tx,
        RemoteSyncEvent::StepFinished {
            host: host.to_string(),
            label,
            detail: "copied".to_string(),
            elapsed: started.elapsed(),
        },
    );
    Ok(())
}

fn rsync_to_host_cached(
    local: &Path,
    host: &str,
    remote_path: &str,
    remote_root: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let file_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input");
    let label = format!("binary {file_name}");
    let started = Instant::now();
    if remote_file_available(host, remote_path)? {
        send_remote_sync_event(
            tx,
            RemoteSyncEvent::StepFinished {
                host: host.to_string(),
                label,
                detail: "cached".to_string(),
                elapsed: started.elapsed(),
            },
        );
        return Ok(());
    }
    if promote_remote_artifact(local, host, remote_root, remote_path)? {
        send_remote_sync_event(
            tx,
            RemoteSyncEvent::StepFinished {
                host: host.to_string(),
                label,
                detail: "moved remote artifact into cache".to_string(),
                elapsed: started.elapsed(),
            },
        );
        return Ok(());
    }
    rsync_to_host_with_progress(local, host, remote_path, &label, tx)
}

fn rsync_dir_to_host_cached(
    local: &Path,
    host: &str,
    remote_path: &str,
    tx: &mpsc::Sender<RemoteSyncEvent>,
) -> Result<()> {
    let dir_name = local
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("input-dir");
    let label = format!("model package ({dir_name}/)");
    let started = Instant::now();
    if remote_package_available(local, host, remote_path)? {
        send_remote_sync_event(
            tx,
            RemoteSyncEvent::StepFinished {
                host: host.to_string(),
                label,
                detail: "cached".to_string(),
                elapsed: started.elapsed(),
            },
        );
        return Ok(());
    }
    rsync_dir_to_host_with_progress(local, host, remote_path, tx)
}

fn promote_remote_artifact(
    local: &Path,
    host: &str,
    remote_root: &str,
    remote_path: &str,
) -> Result<bool> {
    let Some(file_name) = local.file_name().and_then(|value| value.to_str()) else {
        return Ok(false);
    };
    let local_size = fs::metadata(local)
        .with_context(|| format!("stat local artifact {}", local.display()))?
        .len();
    let remote_parent = Path::new(remote_path)
        .parent()
        .and_then(Path::to_str)
        .context("remote artifact path has no parent")?;
    let command = format!(
        concat!(
            "candidate=$(find {root} -type f -name {file_name} -size {size}c ! -path {target} -print -quit 2>/dev/null); ",
            "if [ -n \"$candidate\" ]; then ",
            "mkdir -p {parent}; ",
            "mv \"$candidate\" {target}; ",
            "fi; ",
            "test -s {target}"
        ),
        root = shell_quote(remote_root),
        file_name = shell_quote(file_name),
        size = local_size,
        target = shell_quote(remote_path),
        parent = shell_quote(remote_parent),
    );
    let status = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(command)
        .status()
        .with_context(|| format!("promote remote artifact on {host}"))?;
    Ok(status.success())
}

struct PackageArtifactCheck {
    path: String,
    artifact_bytes: u64,
}

fn remote_package_available(local: &Path, host: &str, remote_path: &str) -> Result<bool> {
    let artifacts = package_artifact_checks(local)?;
    let manifest = format!("{remote_path}/model-package.json");
    let marker = format!("{remote_path}/.complete");
    let mut checks = vec![format!(
        "test -s {} -a -s {}",
        shell_quote(&manifest),
        shell_quote(&marker)
    )];
    for artifact in artifacts {
        let remote_artifact = format!("{remote_path}/{}", artifact.path);
        checks.push(format!(
            concat!(
                "(test -f {path} && ",
                "actual=$(wc -c < {path} 2>/dev/null | tr -d '[:space:]') && ",
                "test \"$actual\" = {expected})"
            ),
            path = shell_quote(&remote_artifact),
            expected = shell_quote(&artifact.artifact_bytes.to_string())
        ));
    }
    let status = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(checks.join(" && "))
        .status()
        .with_context(|| format!("check remote package {host}:{remote_path}"))?;
    Ok(status.success())
}

fn package_artifact_checks(package_dir: &Path) -> Result<Vec<PackageArtifactCheck>> {
    let manifest_path = package_dir.join("model-package.json");
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path).with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse {}", manifest_path.display()))?;
    let mut artifacts = Vec::new();

    let shared = manifest
        .get("shared")
        .and_then(Value::as_object)
        .context("package manifest missing shared artifact map")?;
    for key in ["metadata", "embeddings", "output"] {
        artifacts.push(package_artifact_check(
            shared
                .get(key)
                .with_context(|| format!("package manifest missing shared.{key}"))?,
            &format!("shared.{key}"),
        )?);
    }

    let layers = manifest
        .get("layers")
        .and_then(Value::as_array)
        .context("package manifest missing layers array")?;
    for (index, layer) in layers.iter().enumerate() {
        artifacts.push(package_artifact_check(layer, &format!("layers[{index}]"))?);
    }

    Ok(artifacts)
}

fn package_artifact_check(value: &Value, label: &str) -> Result<PackageArtifactCheck> {
    let path = value
        .get("path")
        .and_then(Value::as_str)
        .with_context(|| format!("package manifest artifact {label} missing path"))?;
    validate_package_relative_path(path, label)?;
    let artifact_bytes = value
        .get("artifact_bytes")
        .and_then(Value::as_u64)
        .with_context(|| format!("package manifest artifact {label} missing artifact_bytes"))?;
    Ok(PackageArtifactCheck {
        path: path.to_string(),
        artifact_bytes,
    })
}

fn validate_package_relative_path(path: &str, label: &str) -> Result<()> {
    let relative = Path::new(path);
    if !path.is_empty()
        && relative.components().all(|component| match component {
            Component::Normal(_) => true,
            Component::CurDir => true,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => false,
        })
    {
        Ok(())
    } else {
        bail!("package manifest artifact {label} has unsafe path {path:?}")
    }
}

fn remote_file_available(host: &str, remote_path: &str) -> Result<bool> {
    let status = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg(format!("test -s {}", shell_quote(remote_path)))
        .status()
        .with_context(|| format!("check remote artifact {host}:{remote_path}"))?;
    Ok(status.success())
}

fn spawn_rsync_progress_reader<R>(reader: R, tx: std::sync::mpsc::Sender<String>)
where
    R: std::io::Read + Send + 'static,
{
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            match reader.read_until(b'\r', &mut buffer) {
                Ok(0) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buffer)
                        .trim_matches(|ch| ch == '\r' || ch == '\n')
                        .trim()
                        .to_string();
                    if !line.is_empty() {
                        let _ = tx.send(line);
                    }
                }
                Err(_) => break,
            }
        }
    });
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_stage_mask(mask: i64) -> String {
    if mask <= 0 {
        return "-".to_string();
    }
    let stages = (0..63)
        .filter(|index| (mask & (1_i64 << index)) != 0)
        .map(|index| index.to_string())
        .collect::<Vec<_>>();
    if stages.is_empty() {
        "-".to_string()
    } else {
        stages.join(",")
    }
}

fn run_status(command: &mut Command, description: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("{description}: failed to spawn {:?}", command))?;
    if !status.success() {
        bail!("{description} failed with status {status}");
    }
    Ok(())
}

fn metrics_otlp_url(args: &PromptArgs, stages: &[LocalStage]) -> Result<String> {
    if let Some(url) = args.metrics_otlp_grpc_url.clone() {
        return Ok(url);
    }

    let Some(first_remote) = stages.iter().find_map(|stage| stage.remote.as_ref()) else {
        return Ok(format!("http://{}", args.metrics_otlp_grpc_addr));
    };

    let launcher_host = infer_launcher_host_from_ssh(&first_remote.host)?;
    Ok(format!(
        "http://{}:{}",
        launcher_host,
        args.metrics_otlp_grpc_addr.port()
    ))
}

fn metrics_otlp_bind_addr(args: &PromptArgs, remote: bool) -> String {
    if remote && args.metrics_otlp_grpc_addr.ip().is_loopback() {
        format!("0.0.0.0:{}", args.metrics_otlp_grpc_addr.port())
    } else {
        args.metrics_otlp_grpc_addr.to_string()
    }
}

fn stage_model_location(stage: &LocalStage) -> String {
    stage
        .remote
        .as_ref()
        .map(|remote| remote.model_path.clone())
        .unwrap_or_else(|| stage.model_path.display().to_string())
}

fn infer_launcher_host_from_ssh(host: &str) -> Result<String> {
    let output = Command::new("ssh")
        .arg("-n")
        .arg(host)
        .arg("sh -lc 'set -- $SSH_CONNECTION; printf %s \"$1\"'")
        .output()
        .with_context(|| format!("infer launcher host via ssh {host}"))?;
    if !output.status.success() {
        bail!(
            "infer launcher host via ssh {host} failed with status {}",
            output.status
        );
    }
    let value = String::from_utf8(output.stdout)
        .context("ssh launcher host output was not UTF-8")?
        .trim()
        .to_string();
    if value.is_empty() {
        bail!("ssh {host} did not report SSH_CONNECTION client address");
    }
    Ok(value)
}

fn model_cache_key(model_path: &Path, ranges: &[(u32, u32)]) -> Result<String> {
    let canonical = model_path
        .canonicalize()
        .with_context(|| format!("canonicalize model path {}", model_path.display()))?;
    let metadata =
        fs::metadata(&canonical).with_context(|| format!("stat model {}", canonical.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());

    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    if let Some(modified) = modified {
        modified.as_secs().hash(&mut hasher);
        modified.subsec_nanos().hash(&mut hasher);
    }
    ranges.hash(&mut hasher);

    let stem = canonical
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_cache_name)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "model".to_string());
    Ok(format!("{stem}-{:016x}", hasher.finish()))
}

fn model_package_cache_key(model_path: &Path) -> Result<String> {
    let canonical = model_path
        .canonicalize()
        .with_context(|| format!("canonicalize model path {}", model_path.display()))?;
    let metadata =
        fs::metadata(&canonical).with_context(|| format!("stat model {}", canonical.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());

    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    if let Some(modified) = modified {
        modified.as_secs().hash(&mut hasher);
        modified.subsec_nanos().hash(&mut hasher);
    }

    let stem = canonical
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_cache_name)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "model".to_string());
    Ok(format!("{stem}-package-{:016x}", hasher.finish()))
}

fn binary_cache_key(stage_server_bin: &Path, kv_server_bin: &Path) -> Result<String> {
    let mut hasher = DefaultHasher::new();
    hash_file_identity(stage_server_bin, &mut hasher)?;
    hash_file_identity(kv_server_bin, &mut hasher)?;
    Ok(format!("{:016x}", hasher.finish()))
}

fn hash_file_identity(path: &Path, hasher: &mut DefaultHasher) -> Result<()> {
    let mut file =
        fs::File::open(path).with_context(|| format!("open binary {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat binary {}", path.display()))?;
    metadata.len().hash(hasher);
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read binary {}", path.display()))?;
        if read == 0 {
            break;
        }
        buffer[..read].hash(hasher);
    }
    Ok(())
}

fn sanitize_cache_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn parse_stage_ranges(splits: &str, layer_end: u32) -> Result<Vec<(u32, u32)>> {
    if layer_end == 0 {
        bail!("layer_end must be greater than zero");
    }
    let mut boundaries = Vec::new();
    for split in splits.split(',') {
        let split = split.trim();
        if split.is_empty() {
            continue;
        }
        boundaries.push(
            split
                .parse::<u32>()
                .with_context(|| format!("parse split boundary {split:?}"))?,
        );
    }
    let mut ranges = Vec::with_capacity(boundaries.len() + 1);
    let mut start = 0;
    for boundary in boundaries {
        if boundary <= start || boundary >= layer_end {
            bail!("invalid split boundary {boundary} for layer_end {layer_end}");
        }
        ranges.push((start, boundary));
        start = boundary;
    }
    ranges.push((start, layer_end));
    Ok(ranges)
}

fn validate_prompt_topology_plan(
    args: &PromptArgs,
    layer_end: u32,
    ranges: &[(u32, u32)],
) -> Result<()> {
    let identity = format!("{} {}", args.model_id, args.model_path.display());
    let activation_width =
        u32::try_from(args.activation_width).context("activation_width must be non-negative")?;
    let family = infer_family_capability(&identity, layer_end, activation_width);
    let nodes = if args.hosts.is_empty() {
        (0..ranges.len())
            .map(|index| NodeSpec {
                node_id: format!("local-stage-{index}"),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect()
    } else {
        args.hosts
            .iter()
            .map(|host| NodeSpec {
                node_id: host.clone(),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect()
    };
    let request = TopologyPlanRequest {
        topology_id: "local-binary-kv-repl".to_string(),
        model_id: args.model_id.clone(),
        layers: dense_attention_layers(layer_end, 0),
        nodes,
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };
    let splits = split_boundaries_from_ranges(ranges);
    let plan = plan_contiguous_with_splits(&request, &splits).context("topology planner failed")?;

    if args.activation_wire_dtype.eq_ignore_ascii_case("q8") {
        match family.as_ref().map(|family| family.q8_wire_validation) {
            Some(WireValidation::Validated) => {}
            Some(WireValidation::Rejected) => {
                bail!("topology planner rejected q8 activation wire dtype for {}; use f16 or add a passing q8 correctness record", args.model_id);
            }
            Some(WireValidation::Untested) => {
                bail!("topology planner has no q8 validation for {}; use f16 until this family/split passes correctness", args.model_id);
            }
            None => {}
        }
    }

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        let reasons = rejected
            .iter()
            .map(|boundary| {
                format!(
                    "layer {}: {:?}: {}",
                    boundary.layer_boundary,
                    boundary.reason_codes,
                    boundary.messages.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("topology planner rejected split plan:\n{reasons}");
    }

    Ok(())
}

fn resolve_stage_ranges(
    single_stage: bool,
    splits: Option<&str>,
    default_stage_count: usize,
    layer_end: u32,
) -> Result<Vec<(u32, u32)>> {
    if layer_end == 0 {
        bail!("layer_end must be greater than zero");
    }
    if single_stage {
        return Ok(vec![(0, layer_end)]);
    }
    match splits {
        Some(splits) => parse_stage_ranges(splits, layer_end),
        None => even_stage_ranges(default_stage_count, layer_end),
    }
}

fn split_boundaries_from_ranges(ranges: &[(u32, u32)]) -> Vec<u32> {
    ranges
        .iter()
        .take(ranges.len().saturating_sub(1))
        .map(|(_, end)| *end)
        .collect()
}

fn even_stage_ranges(stage_count: usize, layer_end: u32) -> Result<Vec<(u32, u32)>> {
    if stage_count == 0 {
        bail!("stage count must be greater than zero");
    }
    if u32::try_from(stage_count).context("stage count exceeds u32")? > layer_end {
        bail!("stage count {stage_count} exceeds layer_end {layer_end}");
    }

    let layer_end = usize::try_from(layer_end).context("layer_end exceeds usize")?;
    let base = layer_end / stage_count;
    let remainder = layer_end % stage_count;
    let mut ranges = Vec::with_capacity(stage_count);
    let mut start = 0usize;
    for index in 0..stage_count {
        let width = base + usize::from(index < remainder);
        let end = start + width;
        ranges.push((
            u32::try_from(start).context("layer range start exceeds u32")?,
            u32::try_from(end).context("layer range end exceeds u32")?,
        ));
        start = end;
    }
    Ok(ranges)
}

fn wait_for_kv_sockets(stages: &[LocalStage], timeout_secs: u64) -> Result<()> {
    let deadline = Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        if stages.iter().all(|stage| stage.kv_uds_path.exists()) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            let missing = stages
                .iter()
                .filter(|stage| !stage.kv_uds_path.exists())
                .map(|stage| stage.kv_uds_path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            bail!("timed out waiting for kv-server sockets: {missing}");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn wait_for_socket(socket_path: &Path, timeout_secs: u64) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    loop {
        if socket_path.exists() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("timed out waiting for socket {}", socket_path.display());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod speculative_tests {
    use super::*;

    #[test]
    fn resolve_stage_ranges_supports_single_full_model_stage() {
        let ranges = resolve_stage_ranges(true, None, 4, 40).unwrap();
        assert_eq!(ranges, vec![(0, 40)]);
    }

    #[test]
    fn empty_splits_also_describe_one_full_range() {
        let ranges = resolve_stage_ranges(false, Some(""), 4, 40).unwrap();
        assert_eq!(ranges, vec![(0, 40)]);
    }

    #[test]
    fn even_stage_ranges_accepts_one_stage() {
        let ranges = even_stage_ranges(1, 40).unwrap();
        assert_eq!(ranges, vec![(0, 40)]);
    }

    #[test]
    fn resolve_stage_ranges_rejects_empty_layer_range() {
        let err = resolve_stage_ranges(true, None, 1, 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("layer_end must be greater than zero"),
            "{err:#}"
        );
    }

    #[test]
    fn thinking_filter_strips_complete_block() {
        let mut filter = ThinkingOutputFilter::new(true);
        let mut output = String::new();
        output.push_str(&filter.push("hello <think>private").text);
        output.push_str(&filter.push(" trace</think>\n\nworld").text);
        output.push_str(&filter.finish().text);
        assert_eq!(output, "hello world");
    }

    #[test]
    fn thinking_filter_handles_split_tags() {
        let mut filter = ThinkingOutputFilter::new(true);
        let mut output = String::new();
        for piece in ["a <", "thi", "nk>x</th", "ink> b"] {
            output.push_str(&filter.push(piece).text);
        }
        output.push_str(&filter.finish().text);
        assert_eq!(output, "a b");
    }

    #[test]
    fn thinking_override_respects_no_think_and_budget_zero() {
        assert_eq!(normalized_prompt_thinking(false, None), Some(false));
        assert_eq!(normalized_prompt_thinking(true, None), Some(false));
        assert_eq!(normalized_prompt_thinking(false, Some(0)), Some(false));
        assert_eq!(normalized_prompt_thinking(false, Some(128)), Some(true));
    }

    fn normalized_prompt_thinking(no_think: bool, budget: Option<usize>) -> Option<bool> {
        let reasoning = prompt_openai_reasoning_config(no_think, budget).unwrap();
        normalize_reasoning_template_options(reasoning.as_ref(), None, &BTreeMap::new())
            .unwrap()
            .enable_thinking
    }

    #[test]
    fn verify_inputs_align_with_draft_proposals() {
        assert_eq!(verify_inputs_for_proposals(10, &[]), Vec::<i32>::new());
        assert_eq!(verify_inputs_for_proposals(10, &[11]), vec![10]);
        assert_eq!(
            verify_inputs_for_proposals(10, &[11, 12, 13]),
            vec![10, 11, 12]
        );
    }

    #[test]
    fn classify_verify_span_full_accept() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 11, 12], 0, 16, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::FullAccept,
                accepted_before_reject: 3,
                repair_input_count: None,
                commit_count: 3,
            }
        );
        assert!(!decision.rejected());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_tail_reject_keeps_state() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 11, 42], 0, 16, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::TailReject,
                accepted_before_reject: 2,
                repair_input_count: Some(3),
                commit_count: 3,
            }
        );
        assert!(decision.rejected());
        assert!(decision.tail_reject());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_early_reject_requires_repair() {
        let decision =
            classify_verify_span(&[10, 11, 12, 13], &[10, 42, 77, 88], 0, 16, |_| Ok(false))
                .unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            }
        );
        assert!(decision.rejected());
        assert!(!decision.tail_reject());
        assert!(decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_accepted_eog_stops_without_growing_window() {
        let decision =
            classify_verify_span(&[10, 99, 12], &[10, 99, 12], 0, 16, |token| Ok(token == 99))
                .unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::AcceptedStop,
                accepted_before_reject: 2,
                repair_input_count: None,
                commit_count: 2,
            }
        );
        assert!(!decision.rejected());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_early_reject_at_limit_does_not_repair() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 42, 77], 2, 4, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyRejectStop,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            }
        );
        assert!(decision.rejected());
        assert!(!decision.tail_reject());
        assert!(!decision.requires_repair());
    }

    #[test]
    fn classify_verify_span_requires_complete_predictions() {
        let err = classify_verify_span(&[10, 11, 12], &[10, 11], 0, 16, |_| Ok(false)).unwrap_err();
        assert!(
            err.to_string()
                .contains("verify span returned too few tokens"),
            "{err:#}"
        );
    }

    #[test]
    fn observe_verify_decision_grows_on_full_accept_only() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 4;
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::FullAccept,
                accepted_before_reject: 4,
                repair_input_count: None,
                commit_count: 4,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 5);
        assert_eq!(stats.full_accept_windows, 1);
        assert_eq!(stats.adaptive_window_grows, 1);
        assert_eq!(stats.accepted_tokens, 4);
    }

    #[test]
    fn observe_verify_decision_stop_outcomes_do_not_move_adaptive_window() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 4;
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::AcceptedStop,
                accepted_before_reject: 2,
                repair_input_count: None,
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyRejectStop,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 4);
        assert_eq!(stats.accepted_stop_windows, 1);
        assert_eq!(stats.early_reject_stop_windows, 1);
        assert_eq!(stats.repair_required_windows, 0);
        assert_eq!(stats.adaptive_window_grows, 0);
        assert_eq!(stats.adaptive_window_shrinks, 0);
    }

    #[test]
    fn observe_verify_decision_early_reject_shrinks_and_marks_repair() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 6;
        stats.observe_verify_decision(
            VerifySpanDecision {
                kind: VerifySpanDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                repair_input_count: Some(2),
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 5);
        assert_eq!(stats.early_reject_windows, 1);
        assert_eq!(stats.repair_required_windows, 1);
        assert_eq!(stats.adaptive_window_shrinks, 1);
        assert_eq!(stats.rejected_windows, 1);
        assert_eq!(stats.first_reject_position_sum, 2);
    }

    #[test]
    fn early_reject_commits_repaired_target_tokens() {
        let draft_tokens = [10, 11, 12, 13];
        let repaired = repaired_commit_tokens(&draft_tokens, 2, 3, &[10, 11, 42]).unwrap();
        assert_eq!(repaired, vec![10, 11, 42]);
    }

    #[test]
    fn repair_commits_changed_accepted_prefix_from_restored_state() {
        let draft_tokens = [10, 11, 12, 13];
        let repaired = repaired_commit_tokens(&draft_tokens, 2, 3, &[10, 99, 42]).unwrap();
        assert_eq!(repaired, vec![10, 99, 42]);
    }

    #[test]
    fn repair_requires_the_full_repaired_prefix() {
        let draft_tokens = [10, 11, 12, 13];
        let err = repaired_commit_tokens(&draft_tokens, 2, 3, &[10, 11]).unwrap_err();
        assert!(
            err.to_string()
                .contains("recovery verify returned too few tokens"),
            "{err:#}"
        );
    }

    #[test]
    fn stable_wire_ids_are_deterministic_and_namespaced() {
        let prompt_index = 7usize.to_le_bytes();
        let session = stable_wire_id(&[b"session-a"]);
        let request = stable_wire_id(&[b"session-a", &prompt_index]);
        assert_ne!(session, 0);
        assert_eq!(session, stable_wire_id(&[b"session-a"]));
        assert_ne!(session, request);
    }
}

fn configure_process_log(command: &mut Command, log_path: &Path) -> Result<()> {
    let stdout = fs::File::create(log_path)
        .with_context(|| format!("create child log {}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .with_context(|| format!("clone child log {}", log_path.display()))?;
    command.stdout(stdout).stderr(stderr);
    Ok(())
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn connect_ready(addr: &str, timeout_secs: u64) -> Result<TcpStream> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    let mut last_error = None;
    while Instant::now() < deadline {
        match TcpStream::connect(addr) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .ok();
                match recv_ready_until_deadline(&mut stream, deadline) {
                    Ok(()) => {
                        stream.set_read_timeout(None).ok();
                        return Ok(stream);
                    }
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

fn recv_ready_until_deadline(stream: &mut TcpStream, deadline: Instant) -> io::Result<()> {
    let mut bytes = [0_u8; 4];
    let mut offset = 0usize;
    while offset < bytes.len() {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for ready handshake",
            ));
        }
        match stream.read(&mut bytes[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "ready handshake stream closed",
                ));
            }
            Ok(read) => offset += read,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted
                        | io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                ) =>
            {
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(error),
        }
    }
    let magic = i32::from_le_bytes(bytes);
    if magic != READY_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stage ready magic mismatch",
        ));
    }
    Ok(())
}

fn parse_wire_dtype(value: &str) -> Result<WireActivationDType> {
    match value {
        "fp32" | "f32" => Ok(WireActivationDType::F32),
        "fp16" | "f16" => Ok(WireActivationDType::F16),
        "q8" | "int8" | "i8" => Ok(WireActivationDType::Q8),
        _ => bail!("unsupported activation wire dtype {value}"),
    }
}

fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis()
}

#[cfg(test)]
mod package_tests {
    use super::*;

    #[test]
    fn package_artifact_checks_reads_manifest_artifacts() -> Result<()> {
        let package_dir = std::env::temp_dir().join(format!(
            "skippy-prompt-package-read-test-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        fs::create_dir_all(&package_dir)?;
        let manifest = json!({
            "shared": {
                "metadata": {"path": "shared/metadata.gguf", "artifact_bytes": 11},
                "embeddings": {"path": "shared/embeddings.gguf", "artifact_bytes": 22},
                "output": {"path": "shared/output.gguf", "artifact_bytes": 33}
            },
            "layers": [
                {"path": "layers/layer-000.gguf", "artifact_bytes": 44},
                {"path": "layers/layer-001.gguf", "artifact_bytes": 55}
            ]
        });
        fs::write(
            package_dir.join("model-package.json"),
            serde_json::to_vec(&manifest)?,
        )?;

        let checks = package_artifact_checks(&package_dir)?;
        fs::remove_dir_all(&package_dir).ok();

        assert_eq!(checks.len(), 5);
        assert_eq!(checks[0].path, "shared/metadata.gguf");
        assert_eq!(checks[0].artifact_bytes, 11);
        assert_eq!(checks[4].path, "layers/layer-001.gguf");
        assert_eq!(checks[4].artifact_bytes, 55);
        Ok(())
    }

    #[test]
    fn package_artifact_checks_rejects_unsafe_paths() -> Result<()> {
        let package_dir = std::env::temp_dir().join(format!(
            "skippy-prompt-package-unsafe-test-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        fs::create_dir_all(&package_dir)?;
        let manifest = json!({
            "shared": {
                "metadata": {"path": "../metadata.gguf", "artifact_bytes": 11},
                "embeddings": {"path": "shared/embeddings.gguf", "artifact_bytes": 22},
                "output": {"path": "shared/output.gguf", "artifact_bytes": 33}
            },
            "layers": []
        });
        fs::write(
            package_dir.join("model-package.json"),
            serde_json::to_vec(&manifest)?,
        )?;

        let result = package_artifact_checks(&package_dir);
        fs::remove_dir_all(&package_dir).ok();

        assert!(result.is_err());
        Ok(())
    }
}
