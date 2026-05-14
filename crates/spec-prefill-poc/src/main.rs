//! Proof-of-concept: speculative prefill decode verification.
//!
//! Loads two GGUF models (draft + target), generates a draft response with
//! the small model using proper chat template, then verifies the draft
//! tokens using the target model's `verify_tokens`.
//!
//! Usage:
//!   spec-prefill-poc <draft.gguf> <target.gguf> [prompt]

use anyhow::{Context, Result};
use skippy_runtime::{ChatTemplateMessage, ChatTemplateOptions, RuntimeConfig, StageModel};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: spec-prefill-poc <draft.gguf> <target.gguf> [prompt]");
        std::process::exit(1);
    }

    let draft_path = &args[1];
    let target_path = &args[2];
    let prompt = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("What is the capital of Australia?");

    // ── Load draft model ─────────────────────────────────────────────
    eprintln!("Loading draft model...");
    let t0 = Instant::now();
    let draft_config = RuntimeConfig {
        n_gpu_layers: 999,
        ..RuntimeConfig::default()
    };
    let draft_model =
        StageModel::open(draft_path, &draft_config).context("failed to load draft model")?;
    eprintln!("Draft model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Apply chat template with thinking DISABLED for cleaner drafts
    let messages = vec![ChatTemplateMessage::new("user", prompt)];
    let no_think = ChatTemplateOptions {
        add_assistant: true,
        enable_thinking: Some(false),
    };
    let draft_chat_text = draft_model.apply_chat_template_with_options(&messages, no_think)?;
    let draft_prompt_tokens = draft_model.tokenize(&draft_chat_text, false)?;

    println!("Prompt: {:?}", prompt);
    println!("Chat-formatted: {} tokens", draft_prompt_tokens.len());

    // ── Phase 1: Generate draft ──────────────────────────────────────
    println!("\n=== Phase 1: Draft generation ===");
    let t_draft = Instant::now();
    let mut draft_session = draft_model.create_session()?;
    draft_session.prefill_chunked(&draft_prompt_tokens)?;

    let max_draft_tokens = 80;
    let mut draft_token_ids: Vec<i32> = Vec::new();
    let mut prev = *draft_prompt_tokens.last().unwrap();

    for _ in 0..max_draft_tokens {
        let next = draft_session.decode_step(prev)?;
        if draft_model.token_is_eog(next)? {
            break;
        }
        draft_token_ids.push(next);
        prev = next;
    }

    let draft_text = draft_model.detokenize(&draft_token_ids)?;
    let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Draft ({} tokens, {:.0}ms):\n  {}",
        draft_token_ids.len(),
        draft_ms,
        draft_text.trim()
    );

    if draft_token_ids.is_empty() {
        println!("Draft produced no tokens — nothing to verify");
        return Ok(());
    }

    // Get draft text for cross-model re-tokenization
    let draft_as_text = draft_text.clone();

    // Drop draft model to free GPU memory
    drop(draft_session);
    drop(draft_model);

    // ── Load target model ────────────────────────────────────────────
    eprintln!("\nLoading target model...");
    let t1 = Instant::now();
    let target_config = RuntimeConfig {
        n_gpu_layers: 999,
        ..RuntimeConfig::default()
    };
    let target_model =
        StageModel::open(target_path, &target_config).context("failed to load target model")?;
    eprintln!("Target model loaded in {:.1}s", t1.elapsed().as_secs_f64());

    // Re-tokenize with TARGET model's chat template + tokenizer
    // Thinking disabled for target too — we want to verify the answer, not think block
    let target_no_think = ChatTemplateOptions {
        add_assistant: true,
        enable_thinking: Some(false),
    };
    let target_chat_text =
        target_model.apply_chat_template_with_options(&messages, target_no_think)?;
    let target_prompt_tokens = target_model.tokenize(&target_chat_text, false)?;
    let target_draft_tokens = target_model.tokenize(&draft_as_text, false)?;

    println!(
        "Target: {} prompt tokens + {} draft tokens",
        target_prompt_tokens.len(),
        target_draft_tokens.len()
    );

    // ── Phase 2: Verify ──────────────────────────────────────────────
    println!("\n=== Phase 2: Verification ===");
    let t_verify = Instant::now();
    let mut target_session = target_model.create_session()?;
    target_session.prefill_chunked(&target_prompt_tokens)?;
    let prefill_ms = t_verify.elapsed().as_secs_f64() * 1000.0;

    let predicted = target_session.verify_tokens(&target_draft_tokens)?;
    let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0 - prefill_ms;

    println!("Prompt prefill: {prefill_ms:.0}ms");
    println!(
        "Verify pass:    {verify_ms:.0}ms ({} tokens)",
        target_draft_tokens.len()
    );

    // ── Phase 3: Acceptance ──────────────────────────────────────────
    println!("\n=== Phase 3: Acceptance ===");
    let mut first_divergence: Option<usize> = None;
    let mut accepted = 0usize;
    let total = target_draft_tokens.len().min(predicted.len());
    let compare_count = if total > 0 { total - 1 } else { 0 };

    for i in 0..total {
        if i + 1 >= target_draft_tokens.len() {
            break; // last position has no next token to compare
        }

        let next_draft = target_draft_tokens[i + 1];
        let matches = predicted[i] == next_draft;

        if matches {
            accepted += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(i);
        }

        // Show first 5 and divergences
        if i < 5 || !matches {
            let d = target_model.detokenize(&[next_draft]).unwrap_or_default();
            let p = target_model.detokenize(&[predicted[i]]).unwrap_or_default();
            println!(
                "  [{:3}] {} draft={:?} pred={:?}",
                i,
                if matches { "✓" } else { "✗" },
                d.replace('\n', "\\n"),
                p.replace('\n', "\\n"),
            );
        }
    }

    let acceptance_rate = if compare_count > 0 {
        accepted as f64 / compare_count as f64 * 100.0
    } else {
        0.0
    };

    println!("\n=== Results ===");
    println!("Accepted: {accepted}/{compare_count} ({acceptance_rate:.1}%)");

    if let Some(k) = first_divergence {
        let prefix = target_model.detokenize(&target_draft_tokens[..=k])?;
        println!("First divergence at position {k}");
        println!("Accepted prefix: {:?}", prefix.trim());
        println!(
            "Decode saved: {k}/{compare_count} ({:.1}%)",
            k as f64 / compare_count as f64 * 100.0
        );
    } else {
        println!("FULL ACCEPTANCE — zero decode needed!");
    }

    println!(
        "\nTiming: draft={draft_ms:.0}ms verify={verify_ms:.0}ms total={:.0}ms",
        draft_ms + prefill_ms + verify_ms
    );

    Ok(())
}
