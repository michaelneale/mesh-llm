use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum ModelsCommand {
    /// List built-in recommended models.
    Recommended,
    /// List installed local models from the HF cache.
    Installed,
    /// List built-in catalog models.
    #[command(hide = true)]
    List,
    /// Search for catalog models and downloadable GGUF/MLX artifacts on Hugging Face.
    Search {
        /// Search terms.
        #[arg(required = true)]
        query: Vec<String>,
        /// Search only the built-in catalog.
        #[arg(long)]
        catalog: bool,
        /// Maximum number of results to show.
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Show details for one exact model reference.
    Show {
        /// Exact catalog id, Hugging Face ref, repo shorthand, or direct URL.
        model: String,
    },
    /// Download one exact model reference.
    Download {
        /// Exact catalog id, Hugging Face ref, repo shorthand, or direct URL.
        model: String,
        /// Prefer the GGUF backend when resolving an ambiguous Hugging Face repo.
        #[arg(long, conflicts_with = "mlx")]
        gguf: bool,
        /// Prefer the MLX backend when resolving an ambiguous Hugging Face repo.
        #[arg(long, conflicts_with = "gguf")]
        mlx: bool,
        /// Also download the recommended draft model for speculative decoding.
        #[arg(long)]
        draft: bool,
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
    },
}
