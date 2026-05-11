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
    #[arg(long, default_value = "target/debug/ngram-pool-server")]
    pub ngram_pool_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/skippy-model-package")]
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
    #[arg(long, default_value_t = DEFAULT_MESH_CTX_SIZE)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 128)]
    pub prefill_chunk_size: usize,
    #[arg(
        long,
        default_value_t = DEFAULT_MESH_PROMPT_MAX_NEW_TOKENS,
        help = "Maximum generated tokens; 0 matches mesh/OpenAI default behavior by using the remaining context budget."
    )]
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
    #[arg(long, default_value = "f16")]
    pub cache_type_k: String,
    #[arg(long, default_value = "f16")]
    pub cache_type_v: String,
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
    #[arg(long, default_value_t = DEFAULT_MESH_CTX_SIZE)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 128)]
    pub prefill_chunk_size: usize,
    #[arg(
        long,
        default_value_t = DEFAULT_MESH_PROMPT_MAX_NEW_TOKENS,
        help = "Maximum generated tokens; 0 matches mesh/OpenAI default behavior by using the remaining context budget."
    )]
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
    #[arg(skip)]
    pub diagnostics_hint: Option<String>,
    #[arg(skip)]
    log_context: Option<PromptLogContext>,
}
