use std::{net::SocketAddr, path::PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "skippy-correctness")]
#[command(about = "Validate staged llama execution against full-model execution")]
pub struct Cli {
    #[command(subcommand)]
    pub command: CommandKind,
}

#[derive(Subcommand)]
pub enum CommandKind {
    SingleStep(SingleStepArgs),
    Chain(ChainArgs),
    SplitScan(SplitScanArgs),
    DtypeMatrix(DtypeMatrixArgs),
    StateHandoff(StateHandoffArgs),
}

#[derive(Args, Clone)]
pub struct RuntimeArgs {
    #[arg(long, alias = "model-path")]
    pub model: PathBuf,
    #[arg(
        long,
        help = "Model coordinate for local model paths, for example org/repo:Q4_K_M. If omitted, Hugging Face cache paths are resolved from cache provenance."
    )]
    pub model_id: Option<String>,
    #[arg(long)]
    pub stage_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "runtime-slice")]
    pub stage_load_mode: StageLoadMode,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum StageLoadMode {
    RuntimeSlice,
    ArtifactSlice,
    LayerPackage,
}

#[derive(Args, Clone)]
pub struct ServerArgs {
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub child_logs: bool,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
}

#[derive(Args)]
pub struct OutputArgs {
    #[arg(long)]
    pub report_out: Option<PathBuf>,
}

#[derive(Args)]
pub struct SingleStepArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value_t = 15)]
    pub split_layer: u32,
    #[arg(long, default_value = "127.0.0.1:19021")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct ChainArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value = "10,20")]
    pub splits: String,
    #[arg(long, default_value = "127.0.0.1:19031")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19032")]
    pub stage2_bind_addr: SocketAddr,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct SplitScanArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value = "1..30")]
    pub splits: String,
    #[arg(long, default_value = "127.0.0.1:19041")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct DtypeMatrixArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value_t = 15)]
    pub split_layer: u32,
    #[arg(long, default_value = "127.0.0.1:19051")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "f32,f16,q8")]
    pub dtypes: String,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct StateHandoffArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value = "127.0.0.1:19061")]
    pub source_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19062")]
    pub restore_bind_addr: SocketAddr,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 0)]
    pub state_layer_start: u32,
    #[arg(long)]
    pub state_layer_end: Option<u32>,
    #[arg(long)]
    pub state_stage_index: Option<u32>,
    #[arg(long, value_enum, default_value = "full-state")]
    pub state_payload_kind: StatePayloadKind,
    #[arg(long)]
    pub prefix_token_count: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub suffix_token_count: usize,
    #[arg(long, default_value_t = 1)]
    pub cache_hit_repeats: usize,
    #[arg(long)]
    pub runtime_lane_count: Option<u32>,
    #[arg(long)]
    pub borrow_resident_hits: bool,
    #[arg(long)]
    pub cache_decoded_result_hits: bool,
    #[arg(long)]
    pub synthetic_input_activation: bool,
    #[arg(long)]
    pub binary_control: bool,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum StatePayloadKind {
    ResidentKv,
    FullState,
    RecurrentOnly,
    KvRecurrent,
}
