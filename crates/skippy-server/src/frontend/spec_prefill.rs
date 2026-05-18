//! Speculative prefill: verify a draft response in one prefill pass.
//!
//! When a request includes `draft_response` (text) or `draft_tokens`
//! (pre-tokenized IDs), this module verifies the draft against the
//! target model. Accepted prefix tokens are committed without decode;
//! only the tail after the first run of mismatches is decoded normally.
//!
//! ## Acceptance strategy: K-tolerant prefix
//!
//! Speculative *prefill* (this module) is a **latency optimization**, not
//! a fidelity-preserving primitive like speculative *decode* with
//! rejection sampling. We trade some output drift for a much higher
//! acceptance rate when the draft and target are different models.
//!
//! At each position `i` the draft token `draft[i+1]` is "accepted" when:
//!   1. The target's greedy prediction matches it, OR
//!   2. The target assigns probability >= `PROB_THRESHOLD` to it.
//!
//! We then walk the positions and break when we encounter more than
//! `MAX_CONSECUTIVE_MISMATCHES` rejections in a row. A single isolated
//! rejection does NOT abort acceptance — the loop continues, and the
//! mismatched draft token is still emitted to the client.
//!
//! Why this matters: two different models will rarely have a long
//! contiguous matching prefix even when they agree on 80%+ of all
//! positions. Strict prefix acceptance ("break on first mismatch")
//! throws away most of the available agreement. K-tolerant acceptance
//! preserves it.
//!
//! The cost is **drift**: when we accept past a mismatch, the
//! subsequent tokens in the draft were conditioned on a token the
//! target wouldn't have chosen. The output therefore diverges slightly
//! from what pure sequential decode of the target would produce. For
//! chat-style outputs this is usually imperceptible; for strict
//! structured output (tool calls, JSON, code) it should be used with
//! care or disabled.
//!
//! ## Verification modes
//!
//! **Chunked** (default, `CHUNK_SIZE = 8`): verifies draft tokens in
//! small batches that build incrementally on the KV cache. Each chunk
//! sees only its own tokens in self-attention, making the computation
//! numerically closer to sequential decode. This avoids the large-batch
//! precision drift that causes spurious rejections.
//!
//! **Single-batch**: verifies all draft tokens in one batch. Fast but
//! the batch self-attention produces different numerical results than
//! sequential decode, causing rejections even for self-drafted tokens.

use super::*;

/// Result of speculative prefill verification.
pub(super) struct SpecPrefillResult {
    /// Draft token IDs that were fed to verify_tokens. Callers use
    /// this to emit the accepted prefix via `on_token`.
    pub(super) draft_token_ids: Vec<i32>,
    /// Number of draft tokens that were accepted (possibly including
    /// isolated mismatches when K-tolerant acceptance allowed them).
    /// This is the length of the prefix the caller should emit.
    pub(super) accepted_tokens: usize,
    /// Number of positions in the accepted prefix where the draft
    /// token actually matched the target (greedy_match or prob_ok).
    /// `accepted_tokens - raw_matches` is the number of tolerated
    /// mismatches that contributed to drift.
    pub(super) raw_matches: usize,
    /// Total draft tokens that were verified.
    pub(super) total_draft_tokens: usize,
    /// The target model's predicted token at the divergence point.
    /// This is the first token the target would generate that differs
    /// from the draft — it becomes `current` for the decode loop.
    /// None if the draft was fully accepted.
    pub(super) divergence_token: Option<i32>,
    /// Whether the draft was fully accepted AND the response is
    /// complete (last predicted token is EOG). Only then can we
    /// skip the decode loop entirely.
    pub(super) fully_accepted: bool,
    /// Time spent tokenizing the draft (0 when `draft_tokens` used).
    pub(super) tokenize_ms: f64,
    /// Time spent in verify_tokens.
    pub(super) verify_ms: f64,
    /// Per-position log-probability of the draft token.
    pub(super) draft_logprobs: Vec<f32>,
}

/// Chunk size for verification. Smaller = closer to sequential decode
/// (better acceptance) but more batches. 8 is a good balance.
const CHUNK_SIZE: usize = 8;

/// Probability threshold for acceptance. A draft token is accepted if
/// the target assigns probability >= this threshold.
/// exp(-0.693) ≈ 0.5.
const PROB_THRESHOLD: f32 = 0.5;

/// Maximum number of *consecutive* rejected draft tokens we tolerate
/// before breaking acceptance. A single isolated rejection is absorbed
/// (the draft token is still emitted, contributing to drift); a run
/// longer than this means the draft and target have meaningfully
/// diverged and we should fall back to sequential decode.
///
/// Setting this to 0 reverts to strict-prefix acceptance.
const MAX_CONSECUTIVE_MISMATCHES: usize = 1;

/// Verify draft token IDs (or text) against the target model.
///
/// Call this after prefilling the prompt tokens but before the decode
/// loop. The session's KV cache is warm from the prompt prefill. We
/// extend it by verifying the draft tokens in small chunks.
///
/// Returns `None` if the draft is empty or produces no tokens.
pub(super) fn verify_draft(
    runtime: &Arc<Mutex<RuntimeState>>,
    session_id: &str,
    draft_tokens: Option<&[i32]>,
    draft_text: Option<&str>,
) -> OpenAiResult<Option<SpecPrefillResult>> {
    // Resolve draft token IDs: prefer pre-tokenized, fall back to text.
    let (draft_token_ids, tokenize_ms) = if let Some(tokens) = draft_tokens {
        if tokens.is_empty() {
            return Ok(None);
        }
        (tokens.to_vec(), 0.0)
    } else if let Some(text) = draft_text {
        if text.trim().is_empty() {
            return Ok(None);
        }
        let tokenize_timer = PhaseTimer::start();
        let ids = {
            let rt = runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            rt.model
                .tokenize(text, false)
                .map_err(openai_backend_error)?
        };
        let ms = tokenize_timer.elapsed_ms();
        if ids.is_empty() {
            return Ok(None);
        }
        (ids, ms)
    } else {
        return Ok(None);
    };

    // Verify in chunks — each chunk builds on prior KV cache state,
    // making self-attention numerically closer to sequential decode.
    let verify_timer = PhaseTimer::start();
    let (predicted, draft_logprobs) = {
        let mut rt = runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        rt.verify_tokens_chunked(session_id, &draft_token_ids, CHUNK_SIZE)
            .map_err(openai_backend_error)?
    };
    let verify_ms = verify_timer.elapsed_ms();

    // Find acceptance prefix using K-tolerant matching. Accept draft
    // token at position i when:
    //   1. The target's greedy prediction matches (predicted[i] == draft[i+1]), OR
    //   2. The target assigns probability >= PROB_THRESHOLD to draft[i+1]
    //
    // Walk positions and track the longest streak of consecutive
    // rejections. When that streak exceeds MAX_CONSECUTIVE_MISMATCHES
    // we stop and rewind the accepted prefix to the last matching
    // position — we never end the prefix on a rejected token.
    let compare_len = draft_token_ids.len().saturating_sub(1);
    let mut accepted = 0usize;
    let mut raw_matches = 0usize;
    let mut consecutive_mismatches = 0usize;
    // Index *one past* the last accepted match, used to rewind if we
    // break inside a mismatch streak.
    let mut last_match_end = 0usize;
    for i in 0..compare_len {
        let greedy_match = predicted[i] == draft_token_ids[i + 1];
        let prob_ok = draft_logprobs[i].exp() >= PROB_THRESHOLD;
        if greedy_match || prob_ok {
            accepted = i + 1;
            raw_matches += 1;
            last_match_end = accepted;
            consecutive_mismatches = 0;
        } else {
            consecutive_mismatches += 1;
            if consecutive_mismatches > MAX_CONSECUTIVE_MISMATCHES {
                // Run of mismatches too long — rewind to the last match.
                accepted = last_match_end;
                break;
            }
            // Tolerated mismatch: extend the prefix past it. The draft
            // token at this position is still emitted to the client
            // (contributing to drift), and the loop continues.
            accepted = i + 1;
        }
    }

    // Check if the last predicted token is EOG (draft might be complete).
    let last_predicted_is_eog = if !predicted.is_empty() {
        let rt = runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        rt.model
            .token_is_eog(*predicted.last().unwrap())
            .map_err(openai_backend_error)?
    } else {
        false
    };

    // "Fully accepted" means all draft tokens accepted AND the response
    // is complete (last predicted token is EOG). Only then can we skip
    // the decode loop.
    let all_accepted = accepted == compare_len && compare_len > 0;
    let fully_accepted = all_accepted && last_predicted_is_eog;

    let divergence_token = if all_accepted {
        // The model agreed with all draft tokens. The predicted token
        // after the last draft token is the natural continuation (may
        // be EOG or a token to continue decoding from).
        predicted.last().copied()
    } else {
        // The model diverged at position `accepted`. Its prediction
        // there is what it would generate instead of the draft.
        Some(predicted[accepted])
    };

    let total_draft_tokens = draft_token_ids.len();
    Ok(Some(SpecPrefillResult {
        draft_token_ids,
        accepted_tokens: accepted,
        raw_matches,
        total_draft_tokens,
        divergence_token,
        fully_accepted,
        tokenize_ms,
        verify_ms,
        draft_logprobs,
    }))
}
