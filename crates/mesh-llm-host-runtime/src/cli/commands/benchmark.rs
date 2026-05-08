use anyhow::Result;

use crate::cli::benchmark::{BenchmarkCommand, PromptImportSource};
use crate::system::benchmark_prompts::{self, ImportPromptsArgs};

pub(crate) async fn dispatch_benchmark_command(command: &BenchmarkCommand) -> Result<()> {
    match command {
        BenchmarkCommand::ImportPrompts {
            source,
            limit,
            max_tokens,
            output,
        } => {
            let args = ImportPromptsArgs {
                source: map_prompt_source(*source),
                limit: *limit,
                max_tokens: *max_tokens,
                output: output.clone(),
                user_agent_version: crate::VERSION,
            };
            benchmark_prompts::import_prompt_corpus(args).await
        }
    }
}

fn map_prompt_source(source: PromptImportSource) -> benchmark_prompts::PromptImportSource {
    match source {
        PromptImportSource::MtBench => benchmark_prompts::PromptImportSource::MtBench,
        PromptImportSource::Gsm8k => benchmark_prompts::PromptImportSource::Gsm8k,
        PromptImportSource::Humaneval => benchmark_prompts::PromptImportSource::Humaneval,
    }
}
