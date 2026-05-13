use clap::{Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub(crate) enum BenchmarkCommand {
    /// Import a prompt corpus from a supported online source into local JSONL.
    #[command(name = "import-prompts")]
    ImportPrompts {
        /// Online source to import.
        #[arg(long, value_enum)]
        source: PromptImportSource,
        /// Maximum number of prompts to import.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Optional per-prompt decode budget hint written into the corpus.
        #[arg(long)]
        max_tokens: Option<u32>,
        /// Output JSONL path.
        #[arg(long)]
        output: PathBuf,
    },
    #[command(name = "run-gpu", hide = true)]
    RunGpu {
        #[arg(long, value_enum)]
        backend: GpuBenchmarkBackend,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum GpuBenchmarkBackend {
    Metal,
    Cuda,
    Hip,
    Intel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PromptImportSource {
    MtBench,
    Gsm8k,
    Humaneval,
}
