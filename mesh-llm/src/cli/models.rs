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
pub enum ModelsCommand {
    /// List built-in recommended models.
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
    // Delete variant defined with explicit clap args later in file (existing block).
    /// List built-in catalog models.
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
        /// Search only the built-in catalog.
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
        /// Exact catalog id, Hugging Face ref, or direct URL.
        model: String,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Download one exact model reference.
    Download {
        /// Exact catalog id, Hugging Face ref, or direct URL.
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
