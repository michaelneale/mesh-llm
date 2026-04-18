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
        /// Legacy published rankings dataset repo used only for direct ranking lookup during planning.
        #[arg(long, default_value = "meshllm/moe-rankings")]
        dataset_repo: String,
    },
    /// Run local MoE analysis and cache the result.
    Analyze {
        #[command(subcommand)]
        command: MoeAnalyzeCommand,
    },
    /// Publish a local MoE package via Hugging Face package repos and the Mesh catalog.
    Publish {
        /// Model spec: local path, catalog name, HF exact ref, HF repo selector like `org/repo:BF16@main`, or HF URL.
        model: String,
        /// Mesh catalog dataset repo used for package resolution entries.
        #[arg(long, default_value = "meshllm/catalog")]
        catalog_repo: String,
        /// Optional namespace for the package repo. Defaults to `meshllm` and
        /// falls back to your user namespace when you cannot publish there.
        #[arg(long)]
        namespace: Option<String>,
        #[command(flatten)]
        hf_job: HfJobArgs,
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
    },
}

#[derive(Args, Debug, Clone)]
pub(crate) struct HfJobArgs {
    /// Submit this MoE publish run to Hugging Face Jobs instead of running locally.
    #[arg(long)]
    pub(crate) hf_job: bool,
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
