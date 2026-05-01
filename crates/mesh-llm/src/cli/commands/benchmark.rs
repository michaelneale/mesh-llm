use anyhow::Result;

use crate::cli::benchmark::{BenchmarkCommand, MoeRankingVariant, PromptImportSource};
use crate::system::benchmark_prompts::{self, ImportPromptsArgs};
use crate::system::moe_benchmark::{self, BenchmarkVariant};

pub(crate) async fn dispatch_benchmark_command(command: &BenchmarkCommand) -> Result<()> {
    match command {
        BenchmarkCommand::MoeRanking {
            model,
            nodes,
            overlap,
            min_experts,
            variants,
            analyze_ranking,
            prompts,
            output,
        } => {
            let args = moe_benchmark::MoeRankingBenchmarkArgs {
                model: model.clone(),
                nodes: *nodes,
                overlap: *overlap,
                min_experts: *min_experts,
                variants: variants.iter().copied().map(map_variant).collect(),
                analyze_ranking: analyze_ranking.clone(),
                prompts: prompts.clone(),
                output: output.clone(),
            };
            moe_benchmark::run_moe_ranking_benchmark(args).await
        }
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
            };
            benchmark_prompts::import_prompt_corpus(args).await
        }
        BenchmarkCommand::MoeMicroAnalyze {
            model,
            min_experts,
            analyze_ranking,
            prompts,
            output,
        } => {
            let args = moe_benchmark::MoeMicroAnalyzeBenchmarkArgs {
                model: model.clone(),
                min_experts: *min_experts,
                analyze_ranking: analyze_ranking.clone(),
                prompts: prompts.clone(),
                output: output.clone(),
            };
            moe_benchmark::run_moe_micro_analyze_benchmark(args).await
        }
        BenchmarkCommand::MoeGrouping {
            model,
            nodes,
            overlap,
            min_experts,
            analyze_ranking,
            output,
        } => {
            let args = moe_benchmark::MoeGroupingBenchmarkArgs {
                model: model.clone(),
                nodes: *nodes,
                overlap: *overlap,
                min_experts: *min_experts,
                analyze_ranking: analyze_ranking.clone(),
                output: output.clone(),
            };
            moe_benchmark::run_moe_grouping_benchmark(args).await
        }
        BenchmarkCommand::MoeModelMatrix {
            model,
            nodes,
            overlap,
            min_experts,
            prompts,
            analyze_ranking_dir,
            output,
        } => {
            let args = moe_benchmark::MoeModelMatrixBenchmarkArgs {
                models: model.clone(),
                nodes: *nodes,
                overlap: *overlap,
                min_experts: *min_experts,
                prompts: prompts.clone(),
                analyze_ranking_dir: analyze_ranking_dir.clone(),
                output: output.clone(),
            };
            moe_benchmark::run_moe_model_matrix_benchmark(args).await
        }
    }
}

fn map_variant(variant: MoeRankingVariant) -> BenchmarkVariant {
    match variant {
        MoeRankingVariant::Analyze => BenchmarkVariant::Analyze,
    }
}

fn map_prompt_source(source: PromptImportSource) -> benchmark_prompts::PromptImportSource {
    match source {
        PromptImportSource::MtBench => benchmark_prompts::PromptImportSource::MtBench,
        PromptImportSource::Gsm8k => benchmark_prompts::PromptImportSource::Gsm8k,
        PromptImportSource::Humaneval => benchmark_prompts::PromptImportSource::Humaneval,
    }
}
