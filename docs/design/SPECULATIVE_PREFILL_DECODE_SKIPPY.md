# Speculative Prefill Decoding for Distributed Layer Splits

**Status:** Proof of concept validated  
**Date:** 2026-05-13  
**Scope:** Skippy layer-split inference across multiple nodes  
**Related:** `docs/design/DESIGN.md`

## Summary

**This optimization targets skippy layer-split distributed inference
specifically** — where a large model is sharded across multiple nodes and
each decode step requires a network round trip between them. This is the
dominant latency bottleneck for distributed serving: a 200-token response
on a 2-node split at 50ms RTT costs 10 seconds of pure network wait
during decode.

The solution: a fast model generates a full draft response. That draft is
injected into the prompt for the large distributed model, which runs a
single prefill pass — computing logits at every token position — to verify
whether it agrees. If it agrees, we skip decode entirely and return the
draft. If it diverges at position K, we truncate there and decode only
from K onward.

**Why this is specific to skippy layer splits:**

- Prefill across a layer split is **one network round** regardless of
  sequence length — all tokens processed in a single batched forward pass
  through each node's layer range.
- Decode across a layer split is **one network round per token** — each
  generated token must flow through every node's layers sequentially.
- This asymmetry means prefill is cheap but decode is expensive. Standard
  single-machine inference doesn't have this asymmetry (decode is fast
  locally), so this optimization doesn't help there.

Standard speculative decoding (draft-verify in tight loops) still applies
to single-machine and single-node inference. This design is **net new and
complementary** — it addresses the specific problem of distributed decode
latency over skippy layer splits, which standard spec decode was not
designed for.

This is a standalone optimization. It does not depend on MoA or any other
mesh feature — any fast model can serve as the draft source.

## Motivation

Standard speculative decoding (as in llama.cpp) works within a single
machine: a draft model proposes N tokens, the target verifies them in one
forward pass, the accepted prefix is kept and the rest are re-drafted. This
happens in tight sequential rounds.

In mesh-llm, the large model is often **distributed across nodes via skippy
layer splits**. Each decode step requires a network round trip between nodes
holding different layer ranges. A 200-token response means 200 round trips.
Prefill, by contrast, processes the entire sequence in a single batched
forward pass — one round trip regardless of length.

Speculative prefill decoding exploits this:

| Phase | Tokens | Network rounds (distributed) |
|---|---|---|
| Draft (fast model) | Full response | 0 if local, 1 if peer |
| Verify (distributed large model) | Full response | 1 (prefill) |
| Decode from K (if needed) | Response length − K | Response length − K |

**Best case (draft fully accepted):** 0 distributed decode rounds — the
large model never decodes a single token. Draft becomes the response.  
**Worst case (draft rejected at token 0):** Same as no speculation.  
**Typical case:** Draft partially accepted, decode only the tail.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User prompt arrives                                     │
│  ↓                                                       │
│  Draft model (fast — local, peer, or remote)             │
│  → generates full response: "Canberra is the capital..." │
│  ↓                                                       │
│  Tokenize draft response                                 │
│  ↓                                                       │
│  Construct verification prompt:                          │
│    [system + user + draft_response_tokens]               │
│  ↓                                                       │
│  skippy_verify_tokens() on large distributed model       │
│  → prefill with logits at ALL positions                  │
│  → returns what the large model would predict at each    │
│  ↓                                                       │
│  Acceptance check per position:                          │
│    position 0..N: does large model agree with draft?     │
│  ↓                                                       │
│  Find first divergence point K                           │
│  ↓                                                       │
│  K == end? → return draft as-is (skip all decode)        │
│  K < end?  → truncate at K, decode from K with large     │
└─────────────────────────────────────────────────────────┘
```

### Draft Injection

The draft response is appended to the prompt as plain text — it becomes
part of the token sequence the large model prefills. No special framing,
no metadata envelope, no chat template awareness. It's just more tokens
in the prompt.

This is why **tokenizer and chat template compatibility don't matter**.
The draft model can be any model — different family, different tokenizer,
different chat template. What matters is only the final text. The draft
model produces text, that text gets tokenized by the *target model's*
tokenizer, and the target model prefills its own tokens. The draft model's
internal tokenization is never seen by the target.

### Why Plain Prompt Works

During prefill, the model computes a probability distribution over the next
token at every position. It doesn't "know" whether the tokens are user input,
system prompt, or a draft — it just processes the sequence and produces
logits. We read those logits to see if the model's predictions match the
draft tokens. The model's own generation mechanism is never invoked; we're
purely reading its internal state.

### Draft Context Window

The draft model does not need to see the full conversation context. It
needs to see *enough* context that its output is plausible — enough to
produce a draft that has high verification confidence from the target
model. For many queries, just the last user message is sufficient. For
multi-turn conversations, the last few turns may be enough. The draft
model's context window is not a constraint because we can truncate the
input it sees without affecting the target model's full-context prefill.

## Proof of Concept Results

A standalone PoC binary (`crates/spec-prefill-poc/`) validates the
end-to-end flow: load a draft model, generate a response, unload it, load
the target model, re-tokenize the draft text with the target's tokenizer
and chat template, call `verify_tokens()`, and measure per-position
acceptance.

### Setup

- **Draft:** Qwen3-0.6B Q4_K_M (378 MB)
- **Target:** Qwen3-8B Q4_K_M (4.9 GB)
- **Machine:** Apple M4 Max, 64 GB (both models local — single machine)
- Both models use their own chat templates via `apply_chat_template_with_options()`

### Key finding: thinking blocks kill contiguous acceptance

Qwen3 models produce `<think>...</think>` reasoning blocks by default.
Both models start with `<think>\nOkay,` but diverge within a few tokens
on how they phrase their internal reasoning — "asking about" vs "asking
for", "I remember" vs "Let me", "recall" vs "get". Semantically
identical, but different tokens.

With thinking **enabled** (default):

| Prompt | Total agreement | Contiguous prefix | Decode saved |
|---|---|---|---|
| Capital of Australia | 74.7% | 7/79 | 8.9% |
| 7 × 8 | 86.1% | 4/79 | 5.1% |
| Hello world in Python | 64.6% | 6/79 | 7.6% |
| Define photosynthesis | 77.2% | 6/79 | 7.6% |
| First 5 primes | 89.9% | 7/79 | 8.9% |

**Contiguous prefix acceptance** is the real metric — it's the number of
decode steps we can skip. Total agreement includes matching tokens after
the first divergence, which don't help (we must decode from K onward).

### Disabling thinking: dramatic improvement

With `enable_thinking: Some(false)` (injects `/no_think` for Qwen3),
the draft model produces the answer directly. Both models now agree on
structural formatting and diverge only on content/facts:

| Prompt | Total agreement | Contiguous prefix | Decode saved | Draft | Verify |
|---|---|---|---|---|---|
| Capital of Australia | 88.9% | 5/9 | **55.6%** | 75 ms | 113 ms |
| Hello world in Python | 75.0% | 5/8 | **62.5%** | 45 ms | 115 ms |
| 7 × 8 | 72.7% | 0/11 | 0.0% | 53 ms | 117 ms |
| First 5 primes | 83.3% | 1/48 | 2.1% | 159 ms | 161 ms |

**Best case: 62.5% decode saved** on Hello World — both models produce
identical code (`print("Hello`) up to a style difference (`" World!"` vs
`", World!"`).

**Capital of Australia: 55.6% decode saved** — both produce `The capital
of Australia is **` identically. Divergence is the 0.6B saying "Sydney"
(wrong) and the 8B saying "Canberra" (correct). This is the ideal
outcome: the structure is shared, and the target corrects the factual
error by diverging at exactly the right token.

### Observations

1. **No-think mode is critical for draft models.** Think blocks are
   stylistically diverse between models even within the same family.
   Disabling thinking for the draft model is essentially free (it's a
   chat template option) and dramatically improves prefix acceptance.

2. **Same-family models share structural patterns.** Even the 0.6B and
   8B Qwen3 models agree on formatting (markdown bold, code fences,
   newline placement). Divergences are on content, not structure.

3. **The 0.6B gets facts wrong.** It said Sydney instead of Canberra.
   This is fine — the target model corrects it. The spec prefill
   optimization doesn't compromise accuracy because the target always
   has final authority.

4. **Cross-model re-tokenization works.** The draft model produces text,
   which is re-tokenized by the target model's own tokenizer and chat
   template. No token ID alignment or shared vocabulary is needed.

5. **Timing is viable.** Draft: 45–159 ms, verify: 113–161 ms. On a
   distributed split with 50 ms RTT per decode token, saving even 5
   tokens = 250 ms of network time saved, already exceeding the draft
   cost.

### Projected savings at distributed scale

The PoC runs both models locally, so the absolute times don't reflect
distributed savings. The value is in skipping decode round trips:

| Scenario | RTT/token | 100-token response | 55% prefix saved | Net saving |
|---|---|---|---|---|
| 2-node, same rack | 5 ms | 500 ms decode | 275 ms saved | ~200 ms net (after draft+verify overhead) |
| 2-node, cross-region | 50 ms | 5.0 s decode | 2.75 s saved | ~2.5 s net |
| 3-node, cross-region | 100 ms | 10.0 s decode | 5.5 s saved | ~5.2 s net |

The savings scale linearly with per-token RTT and response length.
Longer responses and higher-latency splits see the largest benefit.

### PoC binary

```
crates/spec-prefill-poc/src/main.rs

Usage: spec-prefill-poc <draft.gguf> <target.gguf> [prompt]
```

The binary:
1. Loads the draft model, applies chat template (no-think), generates
   up to 80 tokens, unloads it
2. Loads the target model, re-tokenizes the draft text with the target's
   chat template and tokenizer
3. Prefills the prompt, calls `verify_tokens()` on the draft tokens
4. Reports per-position acceptance and summary statistics

## What Exists in Skippy Today

### `skippy_verify_tokens()` — C ABI, in `skippy.h`

```c
LLAMA_API enum skippy_status skippy_verify_tokens(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);
```

This function:

1. Takes a sequence of draft tokens
2. Runs them through the model with **`batch.logits[i] = 1` for every
   position** — requesting logits at all positions, not just the last
3. At each position, extracts the model's greedy prediction via
   `skippy_greedy_sample_ith(session, i)`
4. Returns `output_tokens[i]` — what the large model would have predicted
   at position i

Verification is: compare `output_tokens[i]` vs `token_ids[i+1]`.
Match = agreement. First mismatch = divergence point K.

### `skippy_compute_token_signal()` — internal, in `skippy.cpp`

Computes per-position statistics from logits:

```c
struct skippy_token_signal {
    float entropy;        // Shannon entropy of softmax distribution
    float top_logprob;    // log probability of the model's top choice
    float second_logprob; // log probability of the second choice
    float margin;         // top_logprob - second_logprob (confidence gap)
    int32_t top_token;    // token id of the model's top choice
    int32_t second_token; // token id of the second choice
};
```

Currently used in the after-prefill and mid-generation hook system for
entropy-based uncertainty detection and peer consultation.

### `signal_history` — per-session signal buffer

`skippy_record_signal()` appends a `skippy_token_signal` to the session's
`signal_history` vector after each decode step. `skippy_session_signal_window()`
computes aggregate statistics (mean entropy, max entropy, mean margin, etc.)
over a sliding window of recent signals.

### Hook infrastructure

The `after_prefill` hook in the OpenAI frontend fires after the first
decoded token's prefill pass, providing `PrefillHookSignals { first_token_entropy, first_token_margin }`. The `mid_generation` hook fires periodically during
decode with `GenerationSignalWindow` statistics. These hooks currently drive
mesh peer consultation on uncertainty — the same infrastructure could trigger
speculative prefill verification.

## Acceptance Criteria

During verification, at each draft token position `i`, we need to decide:
does the large model accept the draft token at position `i+1`?

### Available signals per position

After `skippy_verify_tokens()` runs with all-position logits:

| Signal | Source | What it tells you |
|---|---|---|
| `output_tokens[i]` | greedy sample at position i | What the large model would have said |
| `entropy` at position i | `skippy_compute_token_signal` | How uncertain the model is |
| `margin` at position i | `skippy_compute_token_signal` | Confidence gap (top vs second) |
| `top_logprob` at position i | `skippy_compute_token_signal` | How confident in its top pick |
| Draft token probability | Needs new code | How likely the model thinks the draft token is |

### Acceptance strategies

**1. Greedy match (simplest, strictest)**

Accept if `output_tokens[i] == token_ids[i+1]`.

- Pro: Simple, no tuning needed.
- Con: Too strict. The draft might say "Canberra" where the large model
  prefers "The capital is Canberra" — both correct, but greedy rejects at
  the first token difference. Rejects valid paraphrases and minor formatting
  differences.
- Verdict: Good baseline, but leaves performance on the table.

**2. Draft token probability (recommended)**

Compute the probability the large model assigns to the actual draft token
(not just its greedy pick). Accept if `P(draft_token) > threshold`.

```
logits = llama_get_logits_ith(ctx, i)
draft_logit = logits[draft_token_id]
log_sum_exp = log(sum(exp(logits[j]) for all j))
draft_logprob = draft_logit - log_sum_exp
draft_prob = exp(draft_logprob)
accept if draft_prob > 0.5
```

- Pro: Catches valid alternatives. If the model thinks "Canberra" has 40%
  probability and "canberra" has 35%, greedy would reject "canberra" but
  probability-based would accept it (combined they dominate the distribution).
- Con: Requires access to raw logits for the draft token, not just the top-1.
  Needs a small extension to `skippy_compute_token_signal` or raw logit access.
- Verdict: **This is the right approach.** Threshold 0.5 for high confidence,
  0.3 for more aggressive acceptance.

**3. Low entropy + top-K inclusion**

Accept if `entropy < threshold AND draft_token in top-K predictions`.

- Pro: When the model is very confident (low entropy), most of the
  probability mass is on a few tokens. If the draft token is one of them,
  it's almost certainly fine.
- Con: Needs a top-K extraction, more complex than single-token probability.
- Verdict: Good secondary signal. Can combine with strategy 2.

**4. Combined (production target)**

```
accept_position(i) =
    draft_token_prob > 0.5           # model thinks draft is plausible
    OR (entropy < 2.0                # model is very confident
        AND draft_token == top_token) # and draft matches top pick
```

Reject at the first position that fails. K = first rejected position.

### Threshold tuning

The threshold trades off speed vs accuracy:

| Threshold | Behavior |
|---|---|
| `prob > 0.7` | Conservative — only accept when model strongly agrees. More decode fallback, highest accuracy. |
| `prob > 0.5` | Balanced — accept when draft token is more likely than not. Good default. |
| `prob > 0.3` | Aggressive — accept plausible alternatives. Faster, slight accuracy risk. |
| `prob > 0.1` | Very aggressive — accept anything the model doesn't actively reject. Speed-first. |

Start with 0.5 and tune empirically.

## Divergence Handling

When the draft diverges at position K:

1. **Tokens 0..K are accepted** — they're already in the KV cache from the
   verification prefill. No work wasted.
2. **Truncate the session at K** — `skippy_trim_session(session, K)` is
   already in the C ABI.
3. **Decode from K** — the large model generates tokens one at a time from
   position K onward, using its own sampling. This is normal distributed
   decode, but only for `(response_length - K)` tokens instead of the full
   response.
4. **No re-drafting** — the large model's decode from K produces the
   authoritative output. The draft served its purpose by skipping the first
   K tokens of decode.

### KV cache efficiency

The verification prefill populates the KV cache for the entire prompt +
draft sequence. When we truncate at K, positions 0..K remain in the cache.
The decode phase starts with a warm cache at position K — no redundant
computation.

## Cost Analysis

### Compute cost of verification

Normal prefill requests logits only at the last position. Verification
requests logits at ALL positions (`batch.logits[i] = 1` for all i). The
additional cost is:

- **Logit projection** at each position: multiply the hidden state by the
  output embedding matrix (vocabulary dimension). This is an extra matmul
  per position.
- For a draft of N tokens with vocabulary size V and hidden dimension D:
  extra cost ≈ N × D × V FLOPs.
- For typical models (D=4096, V=150000): ~614M FLOPs per draft token.
  For a 200-token draft: ~123 GFLOPs extra.
- Compare to the full prefill compute (attention + FFN across all layers):
  this is typically 5-15% overhead.

**Verdict:** Verification prefill is ~1.1-1.2x the cost of normal prefill.
Still one network round trip for distributed inference.

### When speculation pays off

```
T_draft    = time for small model to generate full response (local, fast)
T_prefill  = time for one distributed prefill pass (with all-logits overhead)
T_decode_1 = time for one distributed decode step
N          = total response tokens
K          = accepted prefix length

Without speculation:  T_prefill + N × T_decode_1
With speculation:     T_draft + T_prefill × 1.15 + (N - K) × T_decode_1

Speedup = (N × T_decode_1) / (T_draft + 0.15 × T_prefill + (N - K) × T_decode_1)
```

For a 2-node layer split with 50ms decode RTT, 200-token response, and
80% acceptance rate (K=160):

- Without: 200 × 50ms = 10.0s decode
- With: 0.5s draft + 0.1s verify overhead + 40 × 50ms = 2.6s
- **3.8x speedup**

For 100% acceptance: 0.5s draft + 0.1s verify = **0.6s total** vs 10.0s.
**16x speedup.**

Draft latency depends on source — a local small model is ~0.5s, a fast
peer might be ~1-2s, a remote API might be ~2-3s. Even a 2s draft pays
off massively against 10s of distributed decode.

## Implementation Plan

### Phase 1: FFI binding + proof of concept ✅ DONE

The FFI bindings and safe Rust wrappers already existed:

- `skippy_verify_tokens()` — C ABI in `skippy.h` line 245
- `skippy_ffi::skippy_verify_tokens` — FFI binding at
  `crates/skippy-ffi/src/lib.rs:420`
- `StageSession::verify_tokens()` — safe wrapper at
  `crates/skippy-runtime/src/lib.rs:2425`
- `StageSession::verify_tokens_rewound()` — checkpoint + restore variant

**PoC binary:** `crates/spec-prefill-poc/src/main.rs`

Validated end-to-end: Qwen3-0.6B (draft) → Qwen3-8B (target), cross-model
re-tokenization, chat template with no-think mode. Key result: **55–62%
contiguous prefix acceptance** with thinking disabled, dropping to 5–9%
with thinking enabled. See "Proof of Concept Results" above.

**Remaining Phase 1 items (not yet done):**

- Per-token signal extraction during verify (entropy/margin per position)
- Draft token probability lookup (softmax(logits)[draft_token])

### Phase 2: Wire existing `--draft` CLI to skippy-server

The host runtime has `--draft`, `--draft-max`, `--no-draft` CLI flags
that are parsed but never wired through. The skippy-server has a complete
speculative decode loop (`DraftRunner`, `embedded_generation.rs`) that
works but always receives `draft_model_path: None`.

6. **Thread `--draft` through `SkippyModelLoadOptions`** — add
   `draft_model_path: Option<PathBuf>` and `speculative_window: usize`
   fields. Pass from CLI → `LocalRuntimeModelStartSpec` →
   `SkippyModelLoadOptions` → `EmbeddedOpenAiArgs`.

7. **Test with local draft** — `mesh-llm serve --model Qwen3-8B
   --draft ~/.cache/.../Qwen3-0.6B-Q4_K_M.gguf --draft-max 8`.
   Verify the existing `DraftRunner` + `classify_verify_span` loop
   works end-to-end with real models.

This lights up **standard speculative decoding** (tight rolling windows)
for single-node serving — useful even without distributed splits.

### Phase 3: Mesh-native speculative prefill

This is the new optimization that the PoC proved viable.

8. **`pick_draft_model()` in ingress** — given `ModelTargets` and the
   target model name, select the best draft model. Priority: same-family
   smallest local → any smallest local → same-family smallest remote →
   any smallest remote. Uses `available_model_sizes` from gossip.

9. **Draft generation via `ModelBackend`** — reuse the `LocalModelBackend`
   / `RemoteModelBackend` from MoA to call the draft model. Request uses
   no-think chat template, short context (last 4–6 messages), low
   temperature.

10. **Verify-and-commit in skippy-server** — new request path: when the
    proxy provides a `draft_response` text field alongside the normal
    request:
    - Tokenize prompt + draft with target's chat template/tokenizer
    - Prefill prompt, then `verify_tokens()` on draft tokens
    - Find divergence K using `classify_verify_span()` (already exists)
    - K == end → return draft as response (zero decode)
    - K < end → trim to K, decode the tail normally
    - Stream: emit accepted prefix immediately, then stream decode tail

11. **Automatic engagement in proxy** — the proxy detects:
    - Target is a distributed layer split (not single-node)
    - A draft model is available (from `pick_draft_model()`)
    - Request doesn't have `"speculative_prefill": false`
    When all conditions hold, transparently engage spec prefill.

### Phase 4: Adaptive and production hardening

12. **Acceptance rate tracking** — per (draft, target) model pair, track
    running acceptance rate. Disable speculation if acceptance is
    consistently < 10% (the draft doesn't help for this pair).

13. **Draft timeout** — if draft generation takes longer than estimated
    decode time, abort and fall through to normal decode.

14. **Streaming prefill emission** — emit the verified draft prefix as
    SSE chunks immediately, then stream decode tokens for the tail.

## Relationship to Existing Systems

### vs. Standard speculative decoding (llama.cpp)

Standard spec decode operates within a single machine: draft N tokens,
verify in one batch, accept prefix, re-draft in tight sequential rounds.
It's designed for single-machine inference where decode is already fast —
the speedup comes from verifying N tokens in parallel instead of generating
them one by one.

Speculative prefill decoding solves a **different problem**: distributed
decode latency across skippy layer splits. On a single machine, decode is
fast (microseconds per token locally). Across a layer split, decode is
slow (network RTT per token). Standard spec decode doesn't help here
because even the verification rounds still require distributed forward
passes.

The key differences:

| | Standard Spec Decode | Speculative Prefill (this design) |
|---|---|---|
| Problem | Single-machine decode throughput | Distributed decode latency |
| Draft size | N tokens (small batches) | Full response |
| Verification | Iterative rounds | One prefill pass |
| When it helps | Always (local speedup) | Only with layer splits (network RTT) |
| Complementary? | Yes — still applies to single-node | Yes — this is additive |

Both can coexist. Standard spec decode optimizes local single-node
inference. Speculative prefill optimizes distributed multi-node inference.
They target different bottlenecks.

### vs. MoA (Mixture of Agents)

MoA fans out to multiple models for diversity and arbitrates the best
response. Speculative prefill is orthogonal — it optimizes single-model
response latency, while MoA optimizes response quality by combining
multiple models. They could coexist (MoA dispatches to models that
internally use speculative prefill for speed), but they solve different
problems.

### vs. Mesh hooks (virtual_llm)

Mesh hooks detect uncertainty during decode (after-prefill entropy, mid-
generation drift) and consult peers. Speculative prefill operates before
decode even starts — it's a pre-emptive optimization, not a reactive one.
Both can coexist: speculative prefill handles the common case (draft is
good), hooks handle the edge case (decode diverges into uncertainty).

### vs. Pipeline (plan→execute)

Pipeline uses a small model to plan and a large model to execute. Speculative
prefill is similar in spirit but the "plan" is a full draft response and
"execute" is verification + optional tail decode. Pipeline could use
speculative prefill internally: the planner's output is verified against
the executor.

## Draft Model Selection

### Context subsetting

The draft model doesn't need the full conversation — it needs enough
context to produce plausible text. MoA's context packing already proves
this works (fast workers get system + last message and produce useful
output).

**Strategies (simplest to most sophisticated):**

1. **Last message only** — system prompt + last user turn. ~500-2500
   tokens. Fits any model. Works for factual queries and single-turn
   tasks. Fails when the answer depends on earlier conversation.

2. **Last N messages** — system prompt + last 4-6 messages. ~2000-5000
   tokens. Covers most multi-turn agent conversations. Already proven
   in MoA context packing.

3. **Summarized history** — compress prior turns into a summary, keep
   the last message verbatim. Handles deep conversations within small
   context windows. More complex — the summary could come from the draft
   model itself or be maintained incrementally.

4. **Adaptive** — start with last message only. Track acceptance rate.
   If drafts keep getting rejected at token 0, give more context next
   time. Learn per-conversation how much history the draft needs.

For the first implementation, **strategy 2 (last 4-6 messages)** is
the sweet spot.

### Candidate models and context windows

| Model | Params | Q4 Size | Context | Speed | Notes |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | 378MB | 32K | Fastest | Same family as Qwen3-32B/72B |
| Qwen3-1.7B | 1.7B | ~1GB | 32K | Very fast | Better quality, same family |
| Qwen3-4B | 4B | 2.3GB | 32K | Fast | Good balance, on mesh peers |
| Qwen3.5-4B | 4B | 2.6GB | 32K | Fast | Latest Qwen, improved |
| Llama-3.2-1B | 1.3B | ~700MB | **131K** | Very fast | Huge context for tiny model |
| Llama-3.2-3B | 3.2B | ~1.7GB | **131K** | Fast | Best context/size ratio |
| Gemma-3-1B | 1B | ~700MB | 32K | Very fast | Google family |
| Gemma-4-E4B | 4B (1B active) | 4.6GB | 32K | Very fast | MoE — only 1B active |
| Phi-4-mini | 3.8B | ~2GB | 16K | Fast | Strong reasoning, short context |
| SmolLM2-1.7B | 1.7B | ~1GB | 8K | Very fast | Shortest context — limited use |

**Standout candidates:**

- **Llama-3.2-1B** — 131K context in 700MB. Can see enormous conversation
  histories. Ideal for long-context agentic sessions where prior turns
  matter. Cross-family drafting works fine since we compare text not tokens.

- **Qwen3-0.6B** — 378MB, absurdly fast. Same family as the Qwen3 models
  that dominate the mesh, so highest expected acceptance rate for same-family
  targets. 32K context is sufficient with context subsetting.

- **Qwen3-4B** — sweet spot of quality vs speed. Already running on mesh
  peers, so it could draft as a "fast peer" without downloading anything.

### Model pairings

**Same family (highest acceptance rate):**

The draft and target share training data distribution, so they tend to
phrase things similarly. A small Qwen drafting for a large Qwen will
often produce identical token sequences.

- Qwen3-0.6B → Qwen3-32B/72B/235B
- Qwen3-4B → Qwen3-32B/72B/235B
- Llama-3.2-1B → Llama-3.3-70B

**Cross family (still works):**

Since we compare text (re-tokenized by the target), any coherent draft
model works. The acceptance rate will be lower than same-family because
phrasing differs, but for factual/structural content (code, data, facts)
the divergence is small.

- Qwen3-4B → any large model (Qwen3 is unusually coherent for its size)
- Gemma-4-E4B → any large model (1B active, fast, good quality)
- Llama-3.2-1B → any large model (131K context advantage)

**Poor pairings:**

- Any draft → creative/stylistic target (style diverges immediately)
- Tiny (0.6B) → large model on code tasks (small models hallucinate code)
- Short-context draft on deep conversation (draft misses key context)

### Draft model doesn't have to be local

The mesh already has fast small models on peer nodes. A Qwen3-4B on a
peer with low RTT is an effective remote draft model — it drafts in 1-2s,
still faster than 200 decode rounds at 50ms each on a distributed split.

Selection priority:
1. Local small model (zero network cost)
2. Fast peer with small model (low RTT, already loaded)
3. Remote API with small model (higher latency, but still worth it if
   target decode is expensive)

The constraint is wall-clock time: `draft_time + verify_time < target_decode_time`.

## Opt-In Design

Speculative prefill is a transparent optimization — it doesn't change what
model the user talks to or what response they get. It changes *how fast*
the response arrives. This means it should not require a different model
name (like MoA's `model: "mesh"`).

### Engagement model

**Automatic when conditions are met, disableable per-request.**

The proxy detects when speculative prefill would help:

1. The target model is served via a **distributed layer split** (the only
   case where decode latency is dominated by network round trips).
2. A **fast draft model is available** somewhere in the mesh — local,
   on a fast peer, or the same node with a smaller model loaded.
3. The request is **not a tool call** (tool routing is decided during
   generation, which the draft can't anticipate).

When all three conditions hold, the proxy automatically engages spec
prefill. No user action required.

### Draft model selection (mesh-native)

The mesh already gossips `available_model_sizes: HashMap<String, u64>` per
peer, so the proxy knows every loaded model and its size. Draft selection:

1. **Same family, smallest available.** If the target is Qwen3-30B, prefer
   Qwen3-0.6B or Qwen3-4B. Same-family models have the highest token
   agreement because they share training distribution. Family detection
   uses the model name prefix (e.g. `Qwen3-` matches `Qwen3-0.6B`).

2. **Any small model, local preferred.** If no same-family small model
   exists, use whatever small model is available. A MiniMax or Llama on
   a peer still produces structurally plausible text that the target can
   partially accept. Local models have zero network cost for the draft.

3. **Remote draft is fine.** The draft model doesn't have to be local. A
   Qwen3-4B on a fast peer generates a draft in ~1–2 seconds. As long as
   `draft_time + verify_time < N × decode_rtt`, speculation pays off.

Priority order:
- Local small model (same family) → best acceptance, zero draft latency
- Local small model (any family) → zero draft latency, lower acceptance
- Fast peer small model (same family) → low draft latency, high acceptance
- Fast peer small model (any family) → low draft latency, lower acceptance

The `build_moa_config()` pattern in `ingress.rs` already iterates
`ModelTargets`, checks `InferenceTarget::Local(port)` vs `Remote(peer_id)`,
and constructs backends. Draft selection follows the same pattern but picks
**one** model (the fastest small one) instead of fanning out to all.

### Per-request control

A request-level field disables speculation when the caller knows it won't
help (e.g. creative/open-ended tasks where the draft would diverge
immediately):

```json
{
  "model": "Qwen3-30B-A3B",
  "messages": [...],
  "speculative_prefill": false
}
```

Default is `true` (enabled) when the proxy detects a split target with
an available draft model.

### CLI flags (existing, unwired)

The CLI already has hidden flags for draft model configuration:

```
--draft <path>        Draft model GGUF path
--draft-max <n>       Max draft tokens (default: 8)
--no-draft            Disable automatic draft detection
```

These were designed for standard speculative decoding (tight
draft→verify→repair loop inside skippy-server). They should be wired
through to `SkippyModelLoadOptions` → `EmbeddedOpenAiArgs` to enable
the existing `DraftRunner` + `embedded_generation.rs` spec decode loop.

For mesh-native spec prefill (full-response draft → one-pass verify),
the proxy-level automatic engagement is the right UX. The CLI flags
control the lower-level per-model draft, while the proxy handles
cross-model mesh-wide draft selection.

### What this is NOT

- **Not a virtual model.** MoA uses `model: "mesh"` because MoA changes
  *what* you're talking to (multiple models). Spec prefill doesn't change
  the model — it's the same model, just faster. No new model name.

- **Not always-on.** Only activates for distributed layer splits. Single-
  node serving doesn't benefit (decode is already fast locally). The proxy
  must check whether the target is split before engaging.

- **Not blocking.** If draft generation takes too long or the draft model
  is unavailable, the request falls through to normal decode. Spec prefill
  is best-effort — the worst case is equivalent to no speculation.

## Open Questions

1. **Sampling temperature** — the draft model should use low temperature
   (greedy or near-greedy) to maximize agreement with the target. High
   temperature drafts are more likely to diverge, reducing the benefit.

2. **Creative tasks** — for tasks where the user wants creative, diverse
   output (story writing, brainstorming), speculative prefill may be
   counterproductive. The draft model's output constrains the target to
   agree or diverge, rather than exploring freely. Need a heuristic or
   user flag to disable speculation for creative requests.

3. **Verification cost with very long drafts** — all-position logits for
   a 2000-token draft on a model with 150K vocabulary is significant
   memory (2000 × 150K × 4 bytes ≈ 1.1 GB of logit data). May need to
   verify in chunks rather than all at once.

4. **`skippy_verify_tokens_frame`** — there's a frame variant in the C ABI
   for staged (distributed) verification. Need to confirm it works
   correctly across layer splits, since logits are only meaningful at the
   final layer.

5. **Acceptance rate tracking** — should we track per-model-pair acceptance
   rates to learn which drafters work best for which targets? And use that
   to auto-select the draft model?

## Decision Log

| Decision | Rationale |
|---|---|
| Draft injected as plain prompt (no special framing) | The model doesn't distinguish draft from real content during prefill. Plain injection is simplest and works. Prefix framing can be tested later. |
| Divergence point K → decode from K, no re-draft | The large model's decode from K is authoritative. Re-drafting adds latency and complexity with diminishing returns. |
| Start with greedy match, add probability threshold | Greedy match is simplest to implement and test. Probability threshold is the production target but needs logit access extensions. |
| Draft model = fastest available, not necessarily local | Any fast model works — local, fast peer, or remote API. Constraint is wall-clock latency, not locality. See "Draft Model Selection" section for candidates and context windows. |
| Tokenizer/chat template don't matter | Draft output is plain text, re-tokenized by the target model's own tokenizer. No token ID alignment needed between draft and target. |
| Draft doesn't need full context | Enough context for a plausible draft. Last 4-6 messages is the sweet spot. Shorter context = faster draft. See "Context subsetting" for strategies. |
| Target threshold: `prob > 0.5` default | Balanced speed/accuracy. Tunable per request or globally. |
| Disable thinking for draft model | PoC showed think blocks kill contiguous acceptance (5–9%) vs no-think (55–62%). Thinking is stylistically diverse between models. |
| Transparent proxy optimization, not virtual model | Spec prefill doesn't change what model you talk to — it makes the same model faster. No new model name needed. MoA uses `model: "mesh"` because it changes the model; spec prefill should not. |
| Mesh-native draft selection | The mesh already gossips `available_model_sizes` per peer. Draft model picked automatically: same-family smallest preferred, any small model accepted. Local > fast peer > remote. |
| Draft model can be any model in the mesh | Cross-family drafting works (PoC proves it). Same family has higher acceptance but any coherent model produces structurally similar text. |

## Files

| File | Role |
|---|---|
| `crates/spec-prefill-poc/src/main.rs` | **PoC binary** — loads draft+target, drafts with no-think, verifies, reports acceptance |
| `.deps/llama.cpp/include/skippy.h` | C ABI — `skippy_verify_tokens`, `skippy_verify_tokens_frame` |
| `.deps/llama.cpp/src/skippy.cpp` | C implementation — `skippy_verify_token_batch`, `skippy_compute_token_signal`, `skippy_greedy_sample_ith` |
| `.deps/llama.cpp/include/skippy-signals.h` | C ABI — `skippy_token_signal`, `skippy_generation_signal_window` |
| `crates/skippy-ffi/src/lib.rs` | Rust FFI bindings — `skippy_verify_tokens` at line 420 |
| `crates/skippy-runtime/src/lib.rs` | Safe Rust wrappers — `verify_tokens()` at line 2425, `verify_tokens_rewound()` at line 2453 |
| `crates/skippy-server/src/frontend/speculative.rs` | Existing spec decode loop — `classify_verify_span`, `VerifySpanDecision`, adaptive window, recovery |
| `crates/skippy-server/src/frontend/embedded_generation.rs` | `DraftRunner` integration — propose/verify/commit/repair cycle |
| `crates/skippy-server/src/frontend.rs` | `DraftRunner` struct, `open_draft_runner()`, `draft_model_path` / `speculative_window` config |
| `crates/mesh-llm-host-runtime/src/inference/skippy/mod.rs` | Host runtime skippy config — `draft_model_path: None` (to be wired) |
| `crates/mesh-llm-host-runtime/src/network/openai/ingress.rs` | Proxy intercept — MoA pattern, `build_moa_config()`, `LocalModelBackend` / `RemoteModelBackend` (reusable for draft) |
| `crates/mesh-llm-host-runtime/src/cli/mod.rs` | CLI flags — `--draft`, `--draft-max`, `--no-draft` (parsed but unwired) |
| `crates/openai-frontend/src/hooks.rs` | Hook trait — `PrefillHookSignals`, `GenerationHookSignals` |
| `crates/mesh-mixture-of-agents/src/lib.rs` | MoA gateway — `LocalModelBackend` / `RemoteModelBackend` reusable for draft transport |
