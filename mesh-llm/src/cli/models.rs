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
    /// Search for GGUF models in the catalog or on Hugging Face.
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
        /// Prefer text-capable variants.
        #[arg(long)]
        text: bool,
        /// Prefer vision-capable variants.
        #[arg(long)]
        vision: bool,
        /// Prefer audio-capable variants.
        #[arg(long)]
        audio: bool,
        /// Prefer multimodal variants.
        #[arg(long)]
        multimodal: bool,
    },
    /// Show details for one exact model reference.
    Show {
        /// Exact catalog id, Hugging Face ref, or direct URL.
        model: String,
        /// Prefer text-capable variants.
        #[arg(long)]
        text: bool,
        /// Prefer vision-capable variants.
        #[arg(long)]
        vision: bool,
        /// Prefer audio-capable variants.
        #[arg(long)]
        audio: bool,
        /// Prefer multimodal variants.
        #[arg(long)]
        multimodal: bool,
    },
    /// Download one exact model reference.
    Download {
        /// Exact catalog id, Hugging Face ref, or direct URL.
        model: String,
        /// Also download the recommended draft model for speculative decoding.
        #[arg(long)]
        draft: bool,
        /// Prefer text-capable variants.
        #[arg(long)]
        text: bool,
        /// Prefer vision-capable variants.
        #[arg(long)]
        vision: bool,
        /// Prefer audio-capable variants.
        #[arg(long)]
        audio: bool,
        /// Prefer multimodal variants.
        #[arg(long)]
        multimodal: bool,
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
