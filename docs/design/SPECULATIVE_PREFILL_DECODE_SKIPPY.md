# Speculative Prefill Decoding for Distributed Layer Splits

**Status:** Design  
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

### Phase 1: FFI binding + proof of concept

1. **Add `skippy_verify_tokens` to `skippy-ffi/src/lib.rs`** — the C
   function exists but has no Rust binding.

2. **Add `verify_tokens()` to `skippy-runtime`** — safe Rust wrapper
   that calls the FFI, returns `Vec<llama_token>` (the model's greedy
   predictions at each position).

3. **Add per-token signal extraction during verify** — extend
   `skippy_verify_tokens` (or add a new `skippy_verify_tokens_with_signals`)
   to also run `skippy_compute_token_signal` at each position and return
   a `Vec<TokenSignal>` alongside the greedy predictions. This gives us
   entropy, margin, and top logprob per position without raw logit access.

4. **Draft token probability** — add a field to `skippy_token_signal` for
   the probability of an arbitrary "reference token" (the draft token).
   The logits are already computed; we just need to look up
   `softmax(logits)[draft_token]` at each position during verification.

5. **Standalone test binary** — tokenize a draft, call `verify_tokens()`,
   print per-position agreement/entropy/probability. Validate that the
   acceptance logic works before integrating.

### Phase 2: Integration with skippy-server

6. **New endpoint or request flag** — `POST /v1/chat/completions` with a
   `draft_response` field, or a separate `/v1/verify` endpoint.

7. **Verification flow in skippy-server frontend** — when a draft is
   provided:
   - Tokenize the full prompt + draft
   - Call `verify_tokens()` instead of normal prefill + decode
   - Find divergence point K using acceptance criteria
   - If K == draft length: return draft as the response
   - If K < draft length: trim session to K, decode from K normally

8. **Streaming support** — emit the accepted draft prefix immediately as
   SSE chunks, then stream the decoded tail tokens as they're generated.

### Phase 3: Integration with mesh-llm

9. **Draft model selection** — find the fastest available model that can
   take enough context to produce a plausible draft. This doesn't have to
   be local — a small model on a fast peer, or even a remote API model
   with low latency, works fine. The constraint is wall-clock time: the
   draft must arrive faster than the target model would have decoded the
   same tokens. Selection criteria:
   - Lowest expected latency (local > fast peer > remote API)
   - Sufficient context window for the query
   - Historical draft acceptance rate (learn which models draft well for
     which target)

10. **Automatic engagement** — engage speculative prefill when:
    - A fast draft model and a slower target model are both available
    - The request doesn't require tool calls (tool routing is decided by
      the draft, which may not have tool schemas)
    - The prompt is conversational / knowledge-based (not creative /
      open-ended where diversity matters more than speed)

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

## Files

| File | Role |
|---|---|
| `.deps/llama.cpp/include/skippy.h` | C ABI — `skippy_verify_tokens`, `skippy_verify_tokens_frame` |
| `.deps/llama.cpp/src/skippy.cpp` | C implementation — `skippy_verify_token_batch`, `skippy_compute_token_signal`, `skippy_greedy_sample_ith` |
| `.deps/llama.cpp/include/skippy-signals.h` | C ABI — `skippy_token_signal`, `skippy_generation_signal_window` |
| `crates/skippy-ffi/src/lib.rs` | Rust FFI bindings — needs `skippy_verify_tokens` added |
| `crates/skippy-runtime/src/lib.rs` | Safe Rust wrappers — `last_token_signal()`, `signal_window()`, needs `verify_tokens()` |
| `crates/skippy-server/src/frontend/prompting.rs` | Hook dispatch — `after_prefill`, `mid_generation` |
| `crates/openai-frontend/src/hooks.rs` | Hook trait — `PrefillHookSignals`, `GenerationHookSignals` |
| `crates/mesh-mixture-of-agents/src/lib.rs` | MoA gateway — fast worker could serve as draft source |
