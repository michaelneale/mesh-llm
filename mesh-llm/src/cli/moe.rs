use clap::{Args, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub(crate) enum MoeCommand {
    /// Plan an MoE split using cached or published expert rankings.
    Plan {
        /// Model spec: local path, catalog name, HF exact ref, HF repo selector like `org/repo:BF16@main`, or HF URL.
        model: String,
        /// Override the ranking CSV path instead of resolving from cache or Hugging Face.
        #[arg(long)]
        ranking_file: Option<PathBuf>,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
        /// Cap VRAM used for planning (GB). Matches the existing global naming.
        #[arg(long)]
        max_vram: Option<f64>,
        /// Optional node count override. When omitted, mesh-llm recommends a minimum node count.
        #[arg(long)]
        nodes: Option<usize>,
        /// Published dataset repo used for MoE ranking lookup.
        #[arg(long, default_value = "meshllm/moe-rankings")]
        dataset_repo: String,
        /// Experimental shared-core floor strategy used by `moe plan`.
        #[arg(long, value_enum, default_value_t = ExperimentalMoeFloorMode::Fixed50Pct)]
        experimental_moe_floor_mode: ExperimentalMoeFloorMode,
        /// Experimental multiplier applied to active top-k experts for `topk_multiplier` and `hybrid`.
        #[arg(long)]
        experimental_moe_topk_multiplier: Option<u32>,
        /// Experimental cumulative mass threshold in [0, 1] for `mass_threshold` and `hybrid`.
        #[arg(long)]
        experimental_moe_mass_threshold: Option<f64>,
        /// Optional lower clamp for the derived shared-core floor.
        #[arg(long)]
        experimental_moe_floor_min: Option<u32>,
        /// Optional upper clamp for the derived shared-core floor.
        #[arg(long)]
        experimental_moe_floor_max: Option<u32>,
    },
    /// Run local MoE analysis and cache the result.
    Analyze {
        #[command(subcommand)]
        command: MoeAnalyzeCommand,
    },
    /// Share a local ranking artifact with other mesh-llm users via the canonical Hugging Face dataset.
    Share {
        /// Model spec: local path, catalog name, HF exact ref, HF repo selector like `org/repo:BF16@main`, or HF URL.
        model: String,
        /// Override the ranking CSV path instead of resolving a local cached artifact.
        /// This should point to a ranking CSV, such as a file produced by `mesh-llm moe analyze`.
        #[arg(long)]
        ranking_file: Option<PathBuf>,
        /// Published dataset repo used for duplicate checks and PR target reporting.
        #[arg(long, default_value = "meshllm/moe-rankings")]
        dataset_repo: String,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum MoeAnalyzeCommand {
    /// Run the canonical full MoE analysis and cache it locally.
    Full {
        /// Model spec: local path, catalog name, HF exact ref, HF repo selector like `org/repo:BF16@main`, or HF URL.
        model: String,
        /// Override context size passed to llama-moe-analyze.
        #[arg(long, default_value = "4096")]
        context_size: u32,
        /// Number of layers to offload to GPU during analysis. Use 0 for CPU-only runs.
        #[arg(long, default_value = "0")]
        n_gpu_layers: u32,
        #[command(flatten)]
        hf_job: HfJobArgs,
    },
    /// Run the canonical micro MoE analysis and cache it locally.
    Micro {
        /// Model spec: local path, catalog name, HF exact ref, HF repo selector like `org/repo:BF16@main`, or HF URL.
        model: String,
        /// Number of canonical prompts to use.
        #[arg(long, default_value = "8")]
        prompt_count: usize,
        /// Token budget per prompt.
        #[arg(long, default_value = "128")]
        token_count: u32,
        /// Override context size passed to llama-moe-analyze.
        #[arg(long, default_value = "4096")]
        context_size: u32,
        /// Number of layers to offload to GPU during analysis. Use 0 for CPU-only runs.
        #[arg(long, default_value = "0")]
        n_gpu_layers: u32,
        #[command(flatten)]
        hf_job: HfJobArgs,
    },
}

#[derive(Args, Debug, Clone)]
pub(crate) struct HfJobArgs {
    /// Submit this MoE analyze run to Hugging Face Jobs instead of running locally.
    #[arg(long)]
    pub(crate) hf_job: bool,
    /// Dataset repo to contribute to when the remote analysis succeeds.
    #[arg(long, default_value = "meshllm/moe-rankings")]
    pub(crate) dataset_repo: String,
    /// HF Jobs hardware flavor, e.g. cpu-xl, cpu-performance, l40sx1.
    #[arg(long, default_value = "cpu-xl")]
    pub(crate) hf_job_flavor: String,
    /// HF Jobs timeout, e.g. 30m, 1h, 4h.
    #[arg(long, default_value = "1h")]
    pub(crate) hf_job_timeout: String,
    /// Optional HF namespace that owns the submitted job.
    #[arg(long)]
    pub(crate) hf_job_namespace: Option<String>,
    /// GitHub repo that hosts the mesh-llm release bundle used by the remote job.
    #[arg(long, default_value = "Mesh-LLM/mesh-llm")]
    pub(crate) hf_job_release_repo: String,
    /// Release tag to download inside the remote job. Use `latest` for the latest GitHub release.
    #[arg(long, default_value = "latest")]
    pub(crate) hf_job_release_tag: String,
    /// Release bundle target to use inside the remote job.
    #[arg(long, value_enum, default_value_t = HfJobReleaseTarget::Cpu)]
    pub(crate) hf_job_release_target: HfJobReleaseTarget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum HfJobReleaseTarget {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ExperimentalMoeFloorMode {
    #[value(name = "fixed_50pct")]
    Fixed50Pct,
    #[value(name = "topk_multiplier")]
    TopkMultiplier,
    #[value(name = "mass_threshold")]
    MassThreshold,
    #[value(name = "hybrid")]
    Hybrid,
}

impl From<ExperimentalMoeFloorMode> for crate::system::moe_planner::MoeFloorMode {
    fn from(value: ExperimentalMoeFloorMode) -> Self {
        match value {
            ExperimentalMoeFloorMode::Fixed50Pct => Self::Fixed50Pct,
            ExperimentalMoeFloorMode::TopkMultiplier => Self::TopkMultiplier,
            ExperimentalMoeFloorMode::MassThreshold => Self::MassThreshold,
            ExperimentalMoeFloorMode::Hybrid => Self::Hybrid,
        }
    }
}
