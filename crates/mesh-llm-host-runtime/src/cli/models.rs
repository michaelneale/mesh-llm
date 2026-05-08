use clap::{Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ModelSearchSort {
    Trending,
    Downloads,
    Likes,
    Created,
    Updated,
    #[value(name = "parameters-desc", alias = "most-parameters")]
    ParametersDesc,
    #[value(name = "parameters-asc", alias = "least-parameters")]
    ParametersAsc,
}

#[derive(Subcommand, Debug)]
// CLI enums mirror clap's argument shape; boxing these fields would make the parser harder to maintain.
#[allow(clippy::large_enum_variant)]
pub enum ModelsCommand {
    /// Package a GGUF model for distributed inference by splitting it into layer files on Hugging Face Jobs.
    Package {
        /// Source Hugging Face model ref (e.g. unsloth/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL).
        source_repo: Option<String>,
        /// Quantization variant (deprecated; prefer source refs like repo:Q4_K_M).
        #[arg(long)]
        quant: Option<String>,
        /// Target repo for the layer package (auto-derived if omitted).
        #[arg(long)]
        target: Option<String>,
        /// Override model ID in the manifest.
        #[arg(long)]
        model_id: Option<String>,
        /// HF Job hardware flavor. Use auto for the default CPU splitter baseline.
        #[arg(long, default_value = "auto")]
        flavor: String,
        /// Requested job timeout; raised automatically by model-size minimums.
        #[arg(long, default_value = "1h")]
        timeout: String,
        /// Branch or tag of mesh-llm to build in the job.
        #[arg(long, default_value = "main")]
        mesh_llm_ref: String,
        /// Explicitly keep this as a dry run. This is the default unless --confirm is set.
        #[arg(long)]
        dry_run: bool,
        /// Actually submit the HF Job. Without this, the command only prints plan, spec, and max cost.
        #[arg(long)]
        confirm: bool,
        /// Stream job logs after submission until completion.
        #[arg(long)]
        follow: bool,
        /// Check status of a previously submitted job.
        #[arg(long)]
        status: Option<String>,
        /// Fetch logs for a previously submitted job.
        #[arg(long)]
        logs: Option<String>,
        /// Cancel a running job.
        #[arg(long)]
        cancel: Option<String>,
        /// List recent package jobs.
        #[arg(long)]
        list: bool,
        /// Upload the latest job script to the meshllm bucket (requires org access).
        #[arg(long)]
        update_script: bool,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// List recommended models from the remote meshllm/catalog.
    Recommended {
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// List installed local models from the HF cache.
    Installed {
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Preview or remove mesh-managed models from the Hugging Face cache.
    Cleanup {
        /// Only include models that mesh-llm has not used for the given age (for example 30d or 12h).
        #[arg(long)]
        unused_since: Option<String>,
        /// Remove the selected files instead of printing a dry run preview.
        #[arg(long)]
        yes: bool,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Remove stale derived skippy stage artifacts from the mesh cache.
    Prune {
        /// Remove files instead of printing a dry run note.
        #[arg(long)]
        yes: bool,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    // Delete variant defined with explicit clap args later in file (existing block).
    /// List remote catalog models.
    #[command(hide = true)]
    List {
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Search for catalog models and downloadable GGUF/MLX artifacts on Hugging Face.
    Search {
        /// Search terms.
        #[arg(required = true)]
        query: Vec<String>,
        /// Filter search results to GGUF artifacts (default).
        #[arg(long, conflicts_with = "mlx")]
        gguf: bool,
        /// Filter search results to MLX artifacts.
        #[arg(long, conflicts_with = "gguf")]
        mlx: bool,
        /// Search only the remote meshllm/catalog.
        #[arg(long)]
        catalog: bool,
        /// Maximum number of results to show.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Sort search results.
        #[arg(long, value_enum, default_value = "trending")]
        sort: ModelSearchSort,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Show details for one exact model reference.
    Show {
        /// Exact remote catalog id or Hugging Face ref.
        model: String,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Download one exact model reference.
    Download {
        /// Exact remote catalog id or Hugging Face ref.
        model: String,
        /// Also download the recommended draft model for speculative decoding.
        #[arg(long)]
        draft: bool,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Check or refresh cached Hugging Face repos.
    #[command(visible_alias = "update")]
    Updates {
        /// Repo id like Qwen/Qwen3-8B-GGUF.
        repo: Option<String>,
        /// Operate on every cached Hugging Face repo.
        #[arg(long)]
        all: bool,
        /// Check for newer upstream revisions without refreshing local cache.
        #[arg(long)]
        check: bool,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Delete a specific model from local storage.
    Delete {
        /// Installed model stem or Hugging Face ref (e.g. `Qwen3.5-9B-BF16`, `org/repo`, or `org/repo:BF16`).
        #[arg(required = true)]
        model: String,
        /// Skip dry-run preview and delete immediately.
        #[arg(long)]
        yes: bool,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
}
