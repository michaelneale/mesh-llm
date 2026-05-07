use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use serde_json::Value;
use skippy_runtime::{ModelInfo, RuntimeConfig, RuntimeLoadMode, StageModel, StageSession};

const DEFAULT_CORPUS: &str = "crates/skippy-bench/corpora/kv_mixed_prompts.jsonl";

#[derive(Parser)]
#[command(about = "Benchmark a target/draft model pair for speculative decoding")]
struct Args {
    #[arg(long)]
    target_model_path: PathBuf,
    #[arg(long)]
    draft_model_path: PathBuf,
    #[arg(long)]
    prompt: Vec<String>,
    #[arg(long)]
    prompt_corpus: Option<PathBuf>,
    #[arg(long)]
    prompt_id: Option<String>,
    #[arg(long)]
    prompt_limit: Option<usize>,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 4)]
    speculative_window: usize,
    #[arg(long, default_value_t = 16384)]
    ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    n_gpu_layers: i32,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    json_out: Option<PathBuf>,
    #[arg(long)]
    allow_mismatch: bool,
    #[arg(long)]
    debug_projection: bool,
}

#[derive(Debug, Clone)]
struct PromptCase {
    id: String,
    category: Option<String>,
    prompt: String,
}

#[derive(Debug, Serialize)]
struct Report {
    target_model_path: String,
    draft_model_path: String,
    ctx_size: u32,
    n_gpu_layers: i32,
    max_new_tokens: usize,
    speculative_window: usize,
    prompt_count: usize,
    summary: Summary,
    prompts: Vec<PromptReport>,
}

#[derive(Debug, Default, Serialize)]
struct Summary {
    correct_prompts: usize,
    mismatched_prompts: usize,
    prompt_tokens_total: usize,
    baseline_generated_total: usize,
    speculative_generated_total: usize,
    speculative_windows: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
    rejected_tokens: usize,
    accept_rate: f64,
    baseline_decode_ms: f64,
    speculative_target_decode_ms: f64,
    speculative_draft_decode_ms: f64,
    projected_rollback_verify_ms: f64,
    projected_rollback_total_ms: f64,
    projected_scratch_prefill_ms: f64,
    projected_scratch_verify_ms: f64,
    projected_scratch_total_ms: f64,
    baseline_tokens_per_second: f64,
    speculative_target_tokens_per_second: f64,
    draft_tokens_per_second: f64,
    projected_rollback_tokens_per_second: f64,
    projected_scratch_tokens_per_second: f64,
    projected_rollback_speedup_vs_current_spec: f64,
    projected_scratch_speedup_vs_current_spec: f64,
    mean_accepted_tokens_per_window: f64,
    projected_rollback_rewinds: usize,
}

#[derive(Debug, Serialize)]
struct PromptReport {
    id: String,
    category: Option<String>,
    prompt_tokens: usize,
    tokenizer_match: bool,
    correct: bool,
    mismatch_index: Option<usize>,
    baseline_generated: usize,
    speculative_generated: usize,
    speculative_windows: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
    rejected_tokens: usize,
    accept_rate: f64,
    baseline_prefill_ms: f64,
    baseline_decode_ms: f64,
    baseline_ttft_ms: f64,
    speculative_prefill_ms: f64,
    speculative_target_decode_ms: f64,
    speculative_draft_decode_ms: f64,
    speculative_ttft_ms: f64,
    projected_rollback_verify_ms: f64,
    projected_rollback_total_ms: f64,
    projected_scratch_prefill_ms: f64,
    projected_scratch_verify_ms: f64,
    projected_scratch_total_ms: f64,
    projected_rollback_rewinds: usize,
    baseline_text_preview: String,
    speculative_text_preview: String,
}

#[derive(Debug)]
struct Generation {
    tokens: Vec<i32>,
    prefill_ms: f64,
    decode_ms: f64,
    ttft_ms: f64,
}

#[derive(Debug)]
struct SpecGeneration {
    tokens: Vec<i32>,
    stats: SpecStats,
    target_prefill_ms: f64,
    draft_prefill_ms: f64,
    target_decode_ms: f64,
    draft_decode_ms: f64,
    ttft_ms: f64,
    projection: BatchProjectionStats,
}

#[derive(Debug, Default)]
struct SpecStats {
    windows: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
    rejected_tokens: usize,
}

#[derive(Debug, Default)]
struct BatchProjectionStats {
    rollback_verify_ms: f64,
    scratch_prefill_ms: f64,
    scratch_verify_ms: f64,
    rollback_rewinds: usize,
}

#[derive(Debug, Default)]
struct BatchProjection {
    stats: BatchProjectionStats,
    predicted_tokens: Vec<i32>,
}

#[derive(Debug, Clone)]
struct ProjectionDebug {
    prompt_id: String,
    window_index: usize,
    generated_tokens: usize,
    context_tokens: usize,
    context_tail: Vec<i32>,
    proposals: Vec<i32>,
    verify_inputs: Vec<i32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }
    if args.speculative_window == 0 {
        bail!("--speculative-window must be greater than zero");
    }
    if !args.target_model_path.is_file() {
        bail!(
            "target model does not exist: {}",
            args.target_model_path.display()
        );
    }
    if !args.draft_model_path.is_file() {
        bail!(
            "draft model does not exist: {}",
            args.draft_model_path.display()
        );
    }

    let prompts = prompt_cases(&args)?;
    if prompts.is_empty() {
        bail!("prompt set is empty");
    }

    eprintln!(
        "loading target={} draft={} prompts={} ctx={} max_new_tokens={} window={}",
        args.target_model_path.display(),
        args.draft_model_path.display(),
        prompts.len(),
        args.ctx_size,
        args.max_new_tokens,
        args.speculative_window
    );

    let target = open_full_model(&args.target_model_path, args.ctx_size, args.n_gpu_layers)
        .with_context(|| format!("open target model {}", args.target_model_path.display()))?;
    let draft = open_full_model(&args.draft_model_path, args.ctx_size, args.n_gpu_layers)
        .with_context(|| format!("open draft model {}", args.draft_model_path.display()))?;

    let mut reports = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        eprintln!("prompt {}/{} {}", index + 1, prompts.len(), prompt.id);
        reports.push(run_prompt_pair(&args, &target, &draft, prompt)?);
    }

    let summary = summarize(&reports);
    let report = Report {
        target_model_path: args.target_model_path.display().to_string(),
        draft_model_path: args.draft_model_path.display().to_string(),
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        max_new_tokens: args.max_new_tokens,
        speculative_window: args.speculative_window,
        prompt_count: reports.len(),
        summary,
        prompts: reports,
    };

    print_human_summary(&report);
    if args.json || args.json_out.is_some() {
        let json = serde_json::to_string_pretty(&report)?;
        if let Some(path) = args.json_out.as_ref() {
            fs::write(path, format!("{json}\n"))
                .with_context(|| format!("write JSON report {}", path.display()))?;
        } else {
            println!("{json}");
        }
    }

    if report.summary.mismatched_prompts > 0 && !args.allow_mismatch {
        bail!(
            "speculative verification mismatched {} prompt(s)",
            report.summary.mismatched_prompts
        );
    }
    Ok(())
}

fn open_full_model(path: &Path, ctx_size: u32, n_gpu_layers: i32) -> Result<StageModel> {
    let layer_count = model_layer_count(path)?;
    StageModel::open(
        path,
        &RuntimeConfig {
            stage_index: 0,
            layer_start: 0,
            layer_end: layer_count,
            ctx_size,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers,
            cache_type_k: skippy_runtime::GGML_TYPE_F16,
            cache_type_v: skippy_runtime::GGML_TYPE_F16,
            selected_backend_device: None,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
            load_mode: RuntimeLoadMode::RuntimeSlice,
            projector_path: None,
            include_embeddings: true,
            include_output: true,
            filter_tensors_on_load: false,
        },
    )
}

fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    info.tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .context("model has no layer-indexed tensors")
}

fn run_prompt_pair(
    args: &Args,
    target: &StageModel,
    draft: &StageModel,
    prompt: &PromptCase,
) -> Result<PromptReport> {
    let target_tokens = target
        .tokenize(&prompt.prompt, true)
        .with_context(|| format!("target tokenize prompt {}", prompt.id))?;
    let draft_tokens = draft
        .tokenize(&prompt.prompt, true)
        .with_context(|| format!("draft tokenize prompt {}", prompt.id))?;
    let tokenizer_match = target_tokens == draft_tokens;
    if target_tokens.is_empty() {
        bail!("prompt {} produced no tokens", prompt.id);
    }

    let baseline = generate_baseline(target, &target_tokens, args.max_new_tokens)
        .with_context(|| format!("baseline target generation for {}", prompt.id))?;
    let speculative = generate_speculative(
        target,
        draft,
        SpeculativeRun {
            prompt_tokens: &target_tokens,
            max_new_tokens: args.max_new_tokens,
            window: args.speculative_window,
            prompt_token_count: target_tokens.len(),
            prompt_id: &prompt.id,
            debug_projection: args.debug_projection,
        },
    )
    .with_context(|| format!("speculative target/draft generation for {}", prompt.id))?;

    let mismatch_index = first_mismatch(&baseline.tokens, &speculative.tokens);
    let correct = tokenizer_match && mismatch_index.is_none();
    let accept_rate = if speculative.stats.draft_tokens == 0 {
        0.0
    } else {
        speculative.stats.accepted_tokens as f64 / speculative.stats.draft_tokens as f64
    };

    Ok(PromptReport {
        id: prompt.id.clone(),
        category: prompt.category.clone(),
        prompt_tokens: target_tokens.len(),
        tokenizer_match,
        correct,
        mismatch_index,
        baseline_generated: baseline.tokens.len(),
        speculative_generated: speculative.tokens.len(),
        speculative_windows: speculative.stats.windows,
        draft_tokens: speculative.stats.draft_tokens,
        accepted_tokens: speculative.stats.accepted_tokens,
        rejected_tokens: speculative.stats.rejected_tokens,
        accept_rate,
        baseline_prefill_ms: baseline.prefill_ms,
        baseline_decode_ms: baseline.decode_ms,
        baseline_ttft_ms: baseline.ttft_ms,
        speculative_prefill_ms: speculative.target_prefill_ms + speculative.draft_prefill_ms,
        speculative_target_decode_ms: speculative.target_decode_ms,
        speculative_draft_decode_ms: speculative.draft_decode_ms,
        speculative_ttft_ms: speculative.ttft_ms,
        projected_rollback_verify_ms: speculative.projection.rollback_verify_ms,
        projected_rollback_total_ms: speculative.draft_decode_ms
            + speculative.projection.rollback_verify_ms,
        projected_scratch_prefill_ms: speculative.projection.scratch_prefill_ms,
        projected_scratch_verify_ms: speculative.projection.scratch_verify_ms,
        projected_scratch_total_ms: speculative.draft_decode_ms
            + speculative.projection.scratch_prefill_ms
            + speculative.projection.scratch_verify_ms,
        projected_rollback_rewinds: speculative.projection.rollback_rewinds,
        baseline_text_preview: preview_text(target, &baseline.tokens)?,
        speculative_text_preview: preview_text(target, &speculative.tokens)?,
    })
}

fn generate_baseline(
    model: &StageModel,
    prompt_tokens: &[i32],
    max_new_tokens: usize,
) -> Result<Generation> {
    let mut session = model.create_session()?;
    let prefill_started = Instant::now();
    if prompt_tokens.len() > 1 {
        session.prefill_chunk(&prompt_tokens[..prompt_tokens.len() - 1])?;
    }
    let prefill_ms = elapsed_ms(prefill_started);
    let mut current = *prompt_tokens.last().expect("checked non-empty prompt");
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut decode_ms = 0.0;
    let mut ttft_ms = 0.0;
    let started = Instant::now();
    for step in 0..max_new_tokens {
        let step_started = Instant::now();
        current = session.decode_step(current)?;
        decode_ms += elapsed_ms(step_started);
        if step == 0 {
            ttft_ms = elapsed_ms(started);
        }
        generated.push(current);
        if model.token_is_eog(current)? {
            break;
        }
    }
    Ok(Generation {
        tokens: generated,
        prefill_ms,
        decode_ms,
        ttft_ms,
    })
}

struct SpeculativeRun<'a> {
    prompt_tokens: &'a [i32],
    max_new_tokens: usize,
    window: usize,
    prompt_token_count: usize,
    prompt_id: &'a str,
    debug_projection: bool,
}

fn generate_speculative(
    target: &StageModel,
    draft: &StageModel,
    run: SpeculativeRun<'_>,
) -> Result<SpecGeneration> {
    let mut target_session = target.create_session()?;
    let mut draft_session = draft.create_session()?;
    let mut projection_session = target.create_session()?;
    let target_prefill_started = Instant::now();
    if run.prompt_tokens.len() > 1 {
        target_session.prefill_chunk(&run.prompt_tokens[..run.prompt_tokens.len() - 1])?;
    }
    let target_prefill_ms = elapsed_ms(target_prefill_started);
    let draft_prefill_started = Instant::now();
    reset_draft_to_context(&mut draft_session, run.prompt_tokens)?;
    let mut draft_prefill_ms = elapsed_ms(draft_prefill_started);

    let mut current = *run.prompt_tokens.last().expect("checked non-empty prompt");
    let mut context = run.prompt_tokens.to_vec();
    let mut generated = Vec::with_capacity(run.max_new_tokens);
    let mut stats = SpecStats::default();
    let mut target_decode_ms = 0.0;
    let mut draft_decode_ms = 0.0;
    let mut ttft_ms = 0.0;
    let mut projection = BatchProjectionStats::default();
    let started = Instant::now();

    while generated.len() < run.max_new_tokens {
        let remaining = run.max_new_tokens - generated.len();
        let propose_count = remaining.min(run.window);
        let mut proposals = Vec::with_capacity(propose_count);
        stats.windows += 1;
        for _ in 0..propose_count {
            let draft_started = Instant::now();
            current = draft_session.decode_step(current)?;
            draft_decode_ms += elapsed_ms(draft_started);
            proposals.push(current);
        }
        stats.draft_tokens += proposals.len();

        let batch_projection = measure_batch_projection(
            &mut target_session,
            &mut projection_session,
            &context,
            &proposals,
            run.prompt_token_count,
            ProjectionDebug {
                prompt_id: run.prompt_id.to_string(),
                window_index: stats.windows,
                generated_tokens: generated.len(),
                context_tokens: context.len(),
                context_tail: tail_tokens(&context, 16),
                proposals: proposals.clone(),
                verify_inputs: verify_inputs_for_proposals(&context, &proposals),
            },
            run.debug_projection,
        )?;
        projection.rollback_verify_ms += batch_projection.stats.rollback_verify_ms;
        projection.scratch_prefill_ms += batch_projection.stats.scratch_prefill_ms;
        projection.scratch_verify_ms += batch_projection.stats.scratch_verify_ms;

        let mut rejected = false;
        let base_current = *context.last().expect("context is never empty");
        let mut target_current = base_current;
        for (batch_index, proposal) in proposals.into_iter().enumerate() {
            let target_started = Instant::now();
            let verified = target_session.decode_step(target_current)?;
            target_decode_ms += elapsed_ms(target_started);
            let Some(batch_verified) = batch_projection.predicted_tokens.get(batch_index) else {
                bail!(
                    "batched target verification returned too few tokens: got {} expected at least {}",
                    batch_projection.predicted_tokens.len(),
                    batch_index + 1
                );
            };
            if *batch_verified != verified {
                bail!(
                    "batched target verification mismatch at window token {batch_index}: serial={verified} batch={batch_verified}"
                );
            }
            if generated.is_empty() {
                ttft_ms = elapsed_ms(started);
            }

            let accepted = verified == proposal;
            if accepted {
                stats.accepted_tokens += 1;
            } else {
                stats.rejected_tokens += 1;
                rejected = true;
                projection.rollback_rewinds += 1;
            }
            generated.push(verified);
            context.push(verified);
            target_current = verified;
            current = verified;
            if target.token_is_eog(verified)? || generated.len() >= run.max_new_tokens || !accepted
            {
                break;
            }
        }

        if rejected {
            let reset_started = Instant::now();
            reset_draft_to_context(&mut draft_session, &context)?;
            draft_prefill_ms += elapsed_ms(reset_started);
        }
        if generated
            .last()
            .is_some_and(|token| target.token_is_eog(*token).unwrap_or(false))
        {
            break;
        }
    }

    Ok(SpecGeneration {
        tokens: generated,
        stats,
        target_prefill_ms,
        draft_prefill_ms,
        target_decode_ms,
        draft_decode_ms,
        ttft_ms,
        projection,
    })
}

fn measure_batch_projection(
    rollback_session: &mut StageSession,
    scratch_session: &mut StageSession,
    context_tokens: &[i32],
    proposals: &[i32],
    prompt_token_count: usize,
    debug: ProjectionDebug,
    debug_projection: bool,
) -> Result<BatchProjection> {
    if proposals.is_empty() {
        return Ok(BatchProjection::default());
    }
    let mut verify_inputs = Vec::with_capacity(proposals.len());
    verify_inputs.push(*context_tokens.last().expect("context is never empty"));
    verify_inputs.extend(proposals.iter().take(proposals.len().saturating_sub(1)));

    let rollback_started = Instant::now();
    let predicted_tokens = rollback_session.verify_tokens_rewound(&verify_inputs)?;
    let rollback_verify_ms = elapsed_ms(rollback_started);

    let prefill_started = Instant::now();
    reset_scratch_to_context(scratch_session, context_tokens, prompt_token_count)?;
    let scratch_prefill_ms = elapsed_ms(prefill_started);
    let (rollback_serial, scratch_serial) = if debug_projection {
        let rollback_serial = verify_tokens_serial_rewound(rollback_session, &verify_inputs)?;
        let scratch_serial = verify_tokens_serial_rewound(scratch_session, &verify_inputs)?;
        (Some(rollback_serial), Some(scratch_serial))
    } else {
        (None, None)
    };

    let scratch_started = Instant::now();
    let scratch_predicted_tokens = scratch_session.verify_tokens(&verify_inputs)?;
    let scratch_verify_ms = elapsed_ms(scratch_started);
    if debug_projection {
        eprintln!(
            "projection debug prompt={} window={} generated={} context_tokens={} context_tail={:?} verify_inputs={:?} proposals={:?} rollback_batch={:?} scratch_batch={:?} rollback_serial={:?} scratch_serial={:?}",
            debug.prompt_id,
            debug.window_index,
            debug.generated_tokens,
            debug.context_tokens,
            debug.context_tail,
            debug.verify_inputs,
            debug.proposals,
            predicted_tokens,
            scratch_predicted_tokens,
            rollback_serial,
            scratch_serial
        );
    }
    if scratch_predicted_tokens != predicted_tokens {
        let mismatch_index = first_mismatch(&scratch_predicted_tokens, &predicted_tokens);
        bail!(
            "scratch and rollback batched verification disagreed for prompt={} window={} generated={} context_tokens={} first_mismatch={:?} context_tail={:?} verify_inputs={:?} proposals={:?} scratch={scratch_predicted_tokens:?} rollback={predicted_tokens:?}",
            debug.prompt_id,
            debug.window_index,
            debug.generated_tokens,
            debug.context_tokens,
            mismatch_index,
            debug.context_tail,
            debug.verify_inputs,
            debug.proposals
        );
    }

    Ok(BatchProjection {
        stats: BatchProjectionStats {
            rollback_verify_ms,
            scratch_prefill_ms,
            scratch_verify_ms,
            rollback_rewinds: 0,
        },
        predicted_tokens,
    })
}

fn reset_draft_to_context(session: &mut StageSession, context_tokens: &[i32]) -> Result<()> {
    session.reset()?;
    if context_tokens.len() > 1 {
        session.prefill_chunk(&context_tokens[..context_tokens.len() - 1])?;
    }
    Ok(())
}

fn reset_scratch_to_context(
    session: &mut StageSession,
    context_tokens: &[i32],
    prompt_token_count: usize,
) -> Result<()> {
    session.reset()?;
    if context_tokens.len() <= 1 {
        return Ok(());
    }
    let prompt_prefix_count = prompt_token_count
        .saturating_sub(1)
        .min(context_tokens.len() - 1);
    if prompt_prefix_count > 0 {
        session.prefill_chunk(&context_tokens[..prompt_prefix_count])?;
    }
    for token_id in &context_tokens[prompt_prefix_count..context_tokens.len() - 1] {
        session.decode_step(*token_id)?;
    }
    Ok(())
}

fn verify_tokens_serial_rewound(session: &mut StageSession, token_ids: &[i32]) -> Result<Vec<i32>> {
    let checkpoint = session.checkpoint()?;
    let mut predicted = Vec::with_capacity(token_ids.len());
    let result = (|| {
        for token_id in token_ids {
            predicted.push(session.decode_step(*token_id)?);
        }
        Ok(predicted)
    })();
    let restore_result = session.restore_checkpoint(&checkpoint);
    match (result, restore_result) {
        (Ok(predicted), Ok(())) => Ok(predicted),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

fn verify_inputs_for_proposals(context_tokens: &[i32], proposals: &[i32]) -> Vec<i32> {
    if proposals.is_empty() {
        return Vec::new();
    }
    let mut verify_inputs = Vec::with_capacity(proposals.len());
    verify_inputs.push(*context_tokens.last().expect("context is never empty"));
    verify_inputs.extend(proposals.iter().take(proposals.len().saturating_sub(1)));
    verify_inputs
}

fn tail_tokens(tokens: &[i32], limit: usize) -> Vec<i32> {
    let start = tokens.len().saturating_sub(limit);
    tokens[start..].to_vec()
}

fn first_mismatch(left: &[i32], right: &[i32]) -> Option<usize> {
    let shared = left.len().min(right.len());
    for index in 0..shared {
        if left[index] != right[index] {
            return Some(index);
        }
    }
    (left.len() != right.len()).then_some(shared)
}

fn preview_text(model: &StageModel, tokens: &[i32]) -> Result<String> {
    let text = model.detokenize(tokens)?;
    Ok(text.chars().take(240).collect())
}

fn summarize(prompts: &[PromptReport]) -> Summary {
    let mut summary = Summary::default();
    for prompt in prompts {
        summary.correct_prompts += usize::from(prompt.correct);
        summary.mismatched_prompts += usize::from(!prompt.correct);
        summary.prompt_tokens_total += prompt.prompt_tokens;
        summary.baseline_generated_total += prompt.baseline_generated;
        summary.speculative_generated_total += prompt.speculative_generated;
        summary.speculative_windows += prompt.speculative_windows;
        summary.draft_tokens += prompt.draft_tokens;
        summary.accepted_tokens += prompt.accepted_tokens;
        summary.rejected_tokens += prompt.rejected_tokens;
        summary.baseline_decode_ms += prompt.baseline_decode_ms;
        summary.speculative_target_decode_ms += prompt.speculative_target_decode_ms;
        summary.speculative_draft_decode_ms += prompt.speculative_draft_decode_ms;
        summary.projected_rollback_verify_ms += prompt.projected_rollback_verify_ms;
        summary.projected_rollback_total_ms += prompt.projected_rollback_total_ms;
        summary.projected_scratch_prefill_ms += prompt.projected_scratch_prefill_ms;
        summary.projected_scratch_verify_ms += prompt.projected_scratch_verify_ms;
        summary.projected_scratch_total_ms += prompt.projected_scratch_total_ms;
        summary.projected_rollback_rewinds += prompt.projected_rollback_rewinds;
    }
    summary.accept_rate = if summary.draft_tokens == 0 {
        0.0
    } else {
        summary.accepted_tokens as f64 / summary.draft_tokens as f64
    };
    summary.baseline_tokens_per_second =
        tokens_per_second(summary.baseline_generated_total, summary.baseline_decode_ms);
    summary.speculative_target_tokens_per_second = tokens_per_second(
        summary.speculative_generated_total,
        summary.speculative_target_decode_ms,
    );
    summary.draft_tokens_per_second =
        tokens_per_second(summary.draft_tokens, summary.speculative_draft_decode_ms);
    summary.projected_rollback_tokens_per_second = tokens_per_second(
        summary.speculative_generated_total,
        summary.projected_rollback_total_ms,
    );
    summary.projected_scratch_tokens_per_second = tokens_per_second(
        summary.speculative_generated_total,
        summary.projected_scratch_total_ms,
    );
    let current_spec_total_ms =
        summary.speculative_target_decode_ms + summary.speculative_draft_decode_ms;
    summary.projected_rollback_speedup_vs_current_spec =
        speedup(current_spec_total_ms, summary.projected_rollback_total_ms);
    summary.projected_scratch_speedup_vs_current_spec =
        speedup(current_spec_total_ms, summary.projected_scratch_total_ms);
    summary.mean_accepted_tokens_per_window = if summary.speculative_windows == 0 {
        0.0
    } else {
        summary.accepted_tokens as f64 / summary.speculative_windows as f64
    };
    summary
}

fn print_human_summary(report: &Report) {
    let summary = &report.summary;
    let current_spec_total_ms =
        summary.speculative_target_decode_ms + summary.speculative_draft_decode_ms;
    eprintln!("speculative pair benchmark:");
    eprintln!(
        "  prompts       total={} correct={} mismatched={}",
        report.prompt_count, summary.correct_prompts, summary.mismatched_prompts
    );
    eprintln!(
        "  tokens        prompt={} baseline_generated={} speculative_generated={}",
        summary.prompt_tokens_total,
        summary.baseline_generated_total,
        summary.speculative_generated_total
    );
    eprintln!(
        "  acceptance    windows={} draft={} accepted={} rejected={} rate={:.1}% mean_accepted/window={:.2}",
        summary.speculative_windows,
        summary.draft_tokens,
        summary.accepted_tokens,
        summary.rejected_tokens,
        summary.accept_rate * 100.0,
        summary.mean_accepted_tokens_per_window
    );
    eprintln!(
        "  projection    rollback_rewinds={} measured with rollback and scratch batched verification",
        summary.projected_rollback_rewinds
    );
    eprintln!();
    eprintln!(
        "{:<28} {:>12} {:>12} {:>11} {:>11} {:>10}",
        "path", "total_ms", "tok/s", "vs_target", "vs_current", "notes"
    );
    eprintln!("{}", "-".repeat(92));
    eprintln!(
        "{:<28} {:>12.2} {:>12.2} {:>10.2}x {:>10.2}x {:>10}",
        "target baseline",
        summary.baseline_decode_ms,
        summary.baseline_tokens_per_second,
        1.0,
        speedup(current_spec_total_ms, summary.baseline_decode_ms),
        "actual"
    );
    eprintln!(
        "{:<28} {:>12.2} {:>12.2} {:>10.2}x {:>10.2}x {:>10}",
        "current serial speculative",
        current_spec_total_ms,
        tokens_per_second(summary.speculative_generated_total, current_spec_total_ms),
        speedup(summary.baseline_decode_ms, current_spec_total_ms),
        1.0,
        "actual"
    );
    eprintln!(
        "{:<28} {:>12.2} {:>12.2} {:>10.2}x {:>10.2}x {:>10}",
        "batched rollback",
        summary.projected_rollback_total_ms,
        summary.projected_rollback_tokens_per_second,
        speedup(
            summary.baseline_decode_ms,
            summary.projected_rollback_total_ms
        ),
        summary.projected_rollback_speedup_vs_current_spec,
        "projected"
    );
    eprintln!(
        "{:<28} {:>12.2} {:>12.2} {:>10.2}x {:>10.2}x {:>10}",
        "batched scratch",
        summary.projected_scratch_total_ms,
        summary.projected_scratch_tokens_per_second,
        speedup(
            summary.baseline_decode_ms,
            summary.projected_scratch_total_ms
        ),
        summary.projected_scratch_speedup_vs_current_spec,
        "projected"
    );
    eprintln!();
    eprintln!(
        "  components    current_target_verify={:.2}ms current_draft={:.2}ms rollback_verify={:.2}ms scratch_prefill={:.2}ms scratch_verify={:.2}ms",
        summary.speculative_target_decode_ms,
        summary.speculative_draft_decode_ms,
        summary.projected_rollback_verify_ms,
        summary.projected_scratch_prefill_ms,
        summary.projected_scratch_verify_ms
    );
}

fn tokens_per_second(tokens: usize, elapsed_ms: f64) -> f64 {
    if tokens == 0 || elapsed_ms <= 0.0 {
        0.0
    } else {
        tokens as f64 / (elapsed_ms / 1000.0)
    }
}

fn speedup(baseline_ms: f64, candidate_ms: f64) -> f64 {
    if baseline_ms <= 0.0 || candidate_ms <= 0.0 {
        0.0
    } else {
        baseline_ms / candidate_ms
    }
}

fn prompt_cases(args: &Args) -> Result<Vec<PromptCase>> {
    let mut prompts = Vec::new();
    for (index, prompt) in args.prompt.iter().enumerate() {
        prompts.push(PromptCase {
            id: format!("cli-{index}"),
            category: Some("cli".to_string()),
            prompt: prompt.clone(),
        });
    }

    let corpus = args.prompt_corpus.clone().or_else(|| {
        Path::new(DEFAULT_CORPUS)
            .is_file()
            .then(|| PathBuf::from(DEFAULT_CORPUS))
    });
    if prompts.is_empty() {
        if let Some(path) = corpus.as_ref() {
            prompts.extend(read_prompt_corpus(path)?);
        }
    }
    if prompts.is_empty() {
        prompts.push(PromptCase {
            id: "default".to_string(),
            category: Some("smoke".to_string()),
            prompt: "What is the capital of France?".to_string(),
        });
    }
    if let Some(prompt_id) = args.prompt_id.as_ref() {
        prompts.retain(|prompt| prompt.id == *prompt_id);
        if prompts.is_empty() {
            bail!("no prompt matched --prompt-id {prompt_id}");
        }
    }
    if let Some(limit) = args.prompt_limit {
        prompts.truncate(limit);
    }
    Ok(prompts)
}

fn read_prompt_corpus(path: &Path) -> Result<Vec<PromptCase>> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut prompts = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('{') {
            let value: Value = serde_json::from_str(line).with_context(|| {
                format!("parse JSONL line {} in {}", line_index + 1, path.display())
            })?;
            prompts.push(prompt_case_from_value(&value, line_index)?);
        } else {
            prompts.push(PromptCase {
                id: format!("line-{}", line_index + 1),
                category: Some("plain".to_string()),
                prompt: line.to_string(),
            });
        }
    }
    Ok(prompts)
}

fn prompt_case_from_value(value: &Value, line_index: usize) -> Result<PromptCase> {
    let prompt = value
        .get("prompt")
        .or_else(|| value.get("text"))
        .and_then(Value::as_str)
        .context("prompt corpus row must include a string prompt or text field")?;
    let id = value
        .get("id")
        .or_else(|| value.get("prompt_id"))
        .and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| format!("line-{}", line_index + 1));
    let category = value
        .get("category")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(PromptCase {
        id,
        category,
        prompt: prompt.to_string(),
    })
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}
