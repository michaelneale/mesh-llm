# Speculative Prefill Decoding for Distributed Layer Splits

**Status:** Capped-draft pattern validated — viable for distributed agentic workloads  
**Date:** 2026-05-13 (PoC), 2026-05-12 (server + capped-draft analysis)  
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

## Current State (May 2026)

Tracking branch: **`micn/prefill-draft`** · PR: **#567** (draft) · CI: green

### What is wired up

- `--prefill-speculative <path-to-draft.gguf>` CLI flag on `mesh-llm`.
  Opt-in only. Off by default. Zero startup cost and zero per-request
  overhead when unset.
- `DraftRunner` loads the draft model at startup, generates a short draft
  prefix (`SPEC_PREFILL_DRAFT_CAP = 12` tokens) per request, fed through
  the target's `verify_tokens` in a chunked batch (`CHUNK_SIZE = 8`).
- K-tolerant prefix acceptance (`MAX_CONSECUTIVE_MISMATCHES = 1`) —
  absorbs isolated single-token rejections so two different models can
  share a useful accepted prefix even when they disagree at one spot.
  Strict-prefix behavior is recoverable by setting the constant to 0.
- Telemetry: `accepted_tokens`, `raw_matches`, `tolerated_mismatches`,
  `verify_ms`, `acceptance_rate`, per-position probability dump.
- PoC binary `crates/spec-prefill-poc/` for isolated experiments.

### What is intentionally NOT wired up

- **No auto-detection.** Earlier commits attempted to scan the HF cache
  and pick a same-family draft automatically; that path was removed
  (commit `c6623fdd`) to keep the feature inert when not requested.
- **No protocol changes.** Nothing gossiped, no new wire format. Mixed-
  version meshes are unaffected by this branch.
- **No mesh-native draft selection.** A peer can't yet act as the draft
  source over the QUIC tunnel. The draft model must be a local GGUF.
- **No streaming during verify.** The accepted prefix is emitted in one
  chunk once verify completes; only the post-divergence tail streams
  token-by-token. This is fine for the target workload (distributed
  layer-split with high per-token RTT) and out of scope to change.

### Measured single-node performance (Qwen3-0.6B → Qwen3-8B, M4 Max)

| Phase | Cost |
|---|---|
| Draft generation (0.6B, 12 tokens) | ~108ms |
| Verify pass (8B, 12 tokens, chunked) | ~130ms |
| **Total spec-prefill overhead** | **~240ms** |
| Sequential decode rate (8B local) | ~50-60 tok/s ≈ ~18ms/token |
| **Decode saved per accepted token** | **~18ms** |

Mean acceptance across a 10-prompt sweep (chat, math, code, haiku,
translation, open-ended): **95% accepted, 86% raw matches, ~1.4
tolerated mismatches per request.** Strict-prefix would have been ~9%
on the same prompts because of a Qwen3 thinking-template quirk where
the 8B target emits a second `<think>` immediately after the first.

### Cost/benefit picture

- **Single-node 8B:** roughly break-even (~240ms overhead vs ~180ms
  decode saved). Modest TTFT win because the accepted prefix lands in
  one chunk rather than streaming. Not the target use case.
- **Single-node 32B or larger:** clear net win as per-token decode cost
  grows. Estimated ~+350ms per request at 32B.
- **Distributed pipeline parallel (the real target):** clear large win.
  Decode-time per token at 2 nodes / 50ms RTT is dominated by network
  hops, not compute. Accepting 10 tokens via one verify pass saves 10
  round-trips. Design doc's earlier estimate: +290ms expected value per
  request at 50ms RTT, +610ms at 100ms.

### Concrete next steps

These are the experiments and small wins needed to move this branch
from "validated PoC" to "ready to merge":

1. **Benchmark against a larger local target** (32B / 30B-A3B). Confirm
   the cost/benefit math improves as predicted when decode-per-token
   gets more expensive. One-node-local, no networking — fastest signal.

2. **Benchmark a 2-node skippy pipeline split** with `--prefill-
   speculative` on and off. This is the deployment the feature was
   designed for. Use existing Qwen3-8B layer packages already on disk
   under `~/.cache/huggingface/hub/models--meshllm--Qwen3-8B-Q4_K_M-
   layers/`. Measure TTFT, total wall-clock, accepted-prefix length
   distribution.

3. **Tool-call / structured-output drift study.** K-tolerant trades
   fidelity for latency. Run a sweep with `MAX_CONSECUTIVE_MISMATCHES =
   0` vs `1` on tool-using prompts (goose-shaped requests) and JSON-
   schema-constrained outputs. Decide whether to:
   - gate K per-request (header/query param),
   - default to 0 when the request includes `tools` or
     `response_format`,
   - leave at 1 globally and document operator opt-out.

4. **Wire mesh-native draft selection.** When `--prefill-speculative`
   is unset but the mesh has a viable small same-family peer, propose
   using it via the existing QUIC tunnel. Reuses `LocalModelBackend` /
   `RemoteModelBackend` from the MoA crate. Big lift, real win — makes
   the feature usable without per-node configuration.

5. **Calibrate `SPEC_PREFILL_DRAFT_CAP`.** Currently hardcoded at 12.
   The "Capped-draft experiment" notes below suggest 10–15 is a good
   range; need empirical confirmation under the new K-tolerant regime
   (which may shift the optimum upward).

6. **`skippy_verify_tokens_frame` validation.** The frame variant in
   the C ABI is meant for staged (distributed) verification. Needs an
   integration test confirming logits at the final layer match the
   single-node `verify_tokens` output. Blocks step 2 if not already
   correct.

### What this branch is NOT trying to do

- Replace standard speculative decode (still applies to the
  post-divergence tail).
- Be fidelity-preserving like rejection-sampled spec decode. The drift
  trade-off is intentional; see "Revised conclusion: K-tolerant
  acceptance" below for the full reasoning.
- Auto-engage by default. Opt-in via flag, will remain so until at
  least the tool-call drift study (step 3) is complete.

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

## Server Integration Experiment

**Status:** Integrated and tested. Results reveal a fundamental blocker.

### What was built

Speculative prefill was integrated into skippy-server's local generation
path (`crates/skippy-server/src/frontend/local_generation.rs`). New
request fields `draft_response` (text) and `draft_tokens` (pre-tokenized
IDs) trigger verification before the decode loop. A `return_token_ids`
request flag makes the response include `completion_token_ids` so callers
can pass exact token IDs to avoid tokenizer round-trip issues.

The flow:
1. Normal prompt prefill (warms KV cache)
2. If `draft_tokens` or `draft_response` is set:
   - Tokenize draft text (or use provided IDs directly)
   - Call `RuntimeState::verify_tokens()` — one batched forward pass
   - Find contiguous acceptance prefix
   - Trim KV cache to `prompt_tokens + accepted + 1` on divergence
   - Emit accepted prefix via `on_token()`
   - If fully accepted → return immediately (skip decode)
   - If diverged → set `current = divergence_token`, fall through to
     normal decode loop

### Results: batch-vs-sequential numerical divergence

**Self-drafting** (same model, same tokens, temp=0):

| Prompt | Accept | Rate | Output match |
|---|---|---|---|
| Simple fact (7 tokens) | 6/6 | **100%** | ✅ |
| Simple math (95 tokens) | 1/94 | 1.1% | ❌ |
| Code fibonacci (86 tokens) | 12/85 | 14.1% | ❌ |
| Code merge sort (230 tokens) | 3/229 | 1.3% | ❌ |
| Hash table (101 tokens) | 12/100 | 12.0% | ❌ |
| Agentic tool (16 tokens) | 15/15 | **100%** | ✅ |
| Multi-turn decorator (253 tokens) | 8/252 | 3.2% | ❌ |
| Train reasoning (300 tokens) | 3/299 | 1.0% | ❌ |

**Cross-model drafting** (Qwen3.5-9B → Qwen3-8B via mesh):

| Prompt | Accept | Rate |
|---|---|---|
| Simple fact | 6/6 | **100%** |
| Code prime | 21/101 | 20.8% |
| Flask API | 20/127 | 15.7% |
| Binary search | 0/232 | 0% |
| TCP vs UDP | 0/173 | 0% |

### Critical finding: tokenizer round-trip is NOT the problem

Initial hypothesis: retokenizing generated text produces different token
IDs, causing false rejections. This was disproven. Sending exact token
IDs via `draft_tokens` produces **identical acceptance rates** as sending
text via `draft_response`:

```
simple_fact  — text: 6/6 (100%),  token_ids: 6/6 (100%)
simple_math  — text: 1/94 (1.1%), token_ids: 1/94 (1.1%)
code_fib     — text: 12/85 (14%), token_ids: 12/85 (14%)
multi_turn   — text: 8/252 (3%),  token_ids: 8/252 (3%)
```

### Root cause: batch vs sequential compute precision

`verify_tokens()` processes all draft tokens as a single batch
(`llama_batch` with all positions set). Sequential `decode()` processes
tokens one at a time. Due to floating-point accumulation order differences
in the attention computation:

- Batch: all positions computed in parallel, different reduction order
- Sequential: each position computed after prior KV entries are settled

This produces subtly different logits at the same positions. For short
responses (≤15 tokens), the difference is below the greedy threshold and
predictions match. For longer responses, accumulated precision drift
causes the greedy-sampled token to differ at some position, and all
subsequent tokens diverge because they see different prior context.

This is the same fundamental issue that affects all speculative decoding
in llama.cpp — the verify batch doesn't perfectly reproduce sequential
decode. Standard spec decode handles this via checkpoint/restore and
re-drafting. But spec PREFILL specifically bets on the batch pass
agreeing with what sequential decode would produce, and this bet fails
for anything beyond ~15 tokens.

### Implications

1. **Short responses work perfectly.** Simple facts, tool confirmations,
   and brief answers get 100% acceptance and meaningful time savings
   (55% faster). This is viable for specific use cases.

2. **Long responses are not viable with greedy batch verify.** The
   batch-vs-sequential precision gap is fundamental to the compute
   architecture, not a bug to fix.

3. **Standard spec decode is unaffected.** It works because the draft
   model generates different tokens anyway — the acceptance rate
   measures model agreement, not compute precision. The
   checkpoint/restore mechanism handles mismatches gracefully.

### Mitigation experiments (explored, all failed)

#### Probability-threshold acceptance

Added `skippy_session_token_logprob_at()` C ABI function to read the
log-probability of a specific draft token at each batch position after
`verify_tokens()`. Used `exp(logprob) >= 0.5` as the acceptance
criterion instead of exact greedy match.

**Result: identical acceptance rates.** The divergences are genuine
semantic disagreements, not precision noise. At the first rejection
point, the target assigns only 20-25% probability to the draft token:

```
code_fib:    [0-11] p>0.92, [12] p=0.203 — sharp cliff
explain:     [0-11] p>0.81, [12] p=0.225 — same pattern
multi_turn:  [0-7]  p>0.98, [8]  p=0.035 — model strongly disagrees
count_10:    [0]    p=0.000               — complete disagreement
```

No threshold can rescue these — the model truly wants a different token.

#### Chunk-wise verification

Replaced single-batch verify with chunked verification (8 tokens per
batch). Each chunk builds on the KV cache from prior chunks, making
self-attention numerically closer to sequential decode.

**Result: same acceptance rates, 3-5× slower.** The chunked approach
revealed valuable data — positions AFTER the first divergence often
show p>0.99, meaning the model agrees with the draft everywhere except
at specific structural decision points. But the divergence points
themselves are unchanged:

```
code_fib chunk=8: [12] p=0.195 (was p=0.203 single-batch)
  but [13-19] all p≥0.999 — model re-converges after the fork
```

The divergences are real semantic forks — the model has a genuine
choice between two viable continuations (e.g., "dictionary:\n\n```"
vs "dictionary to store") and batch verify consistently picks the
alternative path.

### Root cause: batch self-attention changes model decisions

Sequential decode builds KV cache entries one at a time. Each new
token's key/value vectors are computed with attention over all prior
KV entries. Batch verify processes multiple tokens simultaneously,
and each token's self-attention sees all other batch tokens' keys and
values in one pass. This isn't precision noise — it produces genuinely
different attention patterns, particularly at positions where the model
is choosing between two high-probability continuations (low margin).

The pattern is consistent:
- High-confidence positions (p>0.9) agree regardless of batch size
- Low-margin decision points (where top-2 probs are close) flip
  to the alternative, often with the draft token dropping to p~0.2
- After the flip, the model re-converges to agree with the draft

### Conclusion (original — see revision below)

Speculative prefill verification as designed — verify draft tokens in
one batch pass, accept contiguous prefix — hits a fundamental limit:
batch verify makes different choices than sequential decode at semantic
fork points. This cannot be fixed by:
- Lowering acceptance thresholds (the probabilities are genuinely low)
- Smaller batch chunks (the divergence is semantic, not numerical)
- Probability-based acceptance (accepting low-prob tokens corrupts the continuation)

**What works:** Short responses (≤15 tokens) where the model has high
confidence throughout. Simple facts, tool confirmations, brief answers.
These get 100% acceptance and genuine speedup.

**What doesn't work:** Anything with structural choices — code formatting,
explanation style, paragraph breaks. The model consistently takes
alternative paths at these decision points in batch mode.

### Revised conclusion: K-tolerant acceptance

The original conclusion was based on **strict-prefix** acceptance —
break on first mismatch. That strategy throws away most of the
agreement between draft and target when they're different models,
because two models rarely have a long contiguous matching run even
when they agree on the majority of positions.

**K-tolerant acceptance** (`MAX_CONSECUTIVE_MISMATCHES = 1`) absorbs
isolated rejections: a single mismatched position does not abort the
prefix; the draft token is still emitted, and the loop continues. We
only break when we hit a run of consecutive mismatches longer than
`K`. The cost is **drift** — accepted tokens past a mismatch were
conditioned on a token the target wouldn't have chosen, so the
output diverges slightly from what pure sequential decode would
produce.

Speculative prefill is a **latency optimization**, not a
fidelity-preserving primitive. The drift tradeoff is intentional and
documented; it lives in the spec_prefill module docstring. For
strict structured output (JSON, tool calls) operators can disable
spec prefill or set `MAX_CONSECUTIVE_MISMATCHES = 0` to recover
strict-prefix behavior.

**Live test results (Qwen3-0.6B draft → Qwen3-8B target, 10 prompts,
12-token drafts):**

| Strategy | Mean accepted | Mean raw matches | Range |
|---|---|---|---|
| Strict prefix | ~9% | ~9% | 0/11 — 11/11 |
| K-tolerant (K=1) | **~95%** | ~86% | 9/11 — 11/11 |

Across the sweep — list-of-days, capital-of-australia, math,
code-completion, haiku, translation, open-ended explanation — every
prompt accepted 9-11 of 11 draft tokens with K-tolerant. The
emitted output is fluent and on-topic in every case; the "drift" at
tolerated positions is typically a word-level synonym choice
(e.g. " is" vs " asked") that the surrounding sentence absorbs
cleanly.

The "first divergence" in the strict-prefix case was almost always
at position 1 because of a thinking-template quirk where Qwen3-8B
emits a second `<think>` token immediately after the first. That
single token tripped the entire prefix. K-tolerant absorbs that
artifact and the rest of the draft (the actual thinking content)
matches both models' natural continuations.

**Net effect**: spec prefill now produces real wall-clock benefit on
the same prompts that previously showed "1-14% acceptance". The
tradeoff is explicit, opt-in, and documented per the design
principle that prefill ≠ decode.

### Capped-draft experiment: the viable pattern

Instead of drafting the full response, draft a **short prefix** (10-15
tokens) where acceptance is reliably high. The remaining tokens are
decoded normally with all the usual speedup tricks.

Tested with self-draft at cap sizes 10, 15, 20, 30 against 11 prompts
covering agentic, code, explanation, math, and multi-turn workloads:

| Category | Prompts | Acceptance at cap=10 | Examples |
|---|---|---|---|
| Always works | 4/11 (36%) | 100% | simple_fact, agentic, yes_no |
| Good prefix | 3/11 (27%) | 80-100% | code_fib, explain, multi_turn |
| Immediate divergence | 4/11 (36%) | 0-11% | count_10, math, tool_result |

**The divergence point is fixed per prompt** — absolute accepted count
stays constant regardless of cap size. `code_fib` always accepts 12
tokens whether you draft 15, 20, or 30. This means capping at 10
keeps you inside the reliable acceptance window for most prompts.

#### Distributed economics (cap=10, 2-node 50ms RTT)

| Metric | Value |
|---|---|
| Draft cost (0.6B local, 10 tokens) | ~30ms |
| Verify overhead | ~0ms (extra tokens in existing prefill batch) |
| Savings when accepted (10 decode rounds × 50ms) | 500ms |
| Net saving when accepted | ~470ms |
| Net cost when rejected | ~30ms (wasted draft only) |
| Break-even success rate | 6% |
| Observed success rate (≥8 tokens accepted) | 64% (7/11 prompts) |
| **Expected value per request** | **+290ms** |

At 100ms RTT (cross-region): expected value +610ms per request.

#### Agentic workload fit

The prompts with 100% acceptance — tool confirmations, brief answers,
yes/no, status summaries — are the **most common responses in agentic
loops**. A Goose-like agent doing tool calls would see spec prefill help
on the majority of turns. And these are exactly the turns where latency
matters most (many short round-trips vs few long generations).

#### Recommended implementation

1. Draft model generates 10-15 tokens (local 0.6B, ~30ms)
2. Draft tokens appended to prompt before prefill (zero extra cost —
   just more tokens in the same prefill batch)
3. After prefill, verify the draft suffix
4. Accepted → skip those decode rounds, continue decoding from there
5. Rejected → trim KV cache, decode from scratch (30ms wasted)

The verify cost is embedded in the prefill pass. The only real overhead
is the draft generation, which is local and cheap. The technique
composes with standard speculative decoding for the tail decode.


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

### Phase 2: Mesh-native speculative prefill

This is the core optimization. A fast model generates a complete answer,
then the distributed target verifies it in one prefill pass — avoiding
sequential decode round trips entirely.

Speculative prefill is distinct from standard speculative decoding
(`DraftRunner` in skippy-server), which proposes small rolling windows
and verifies them iteratively. Both are valuable — standard spec decode
speeds up every decode step (including distributed ones), while
speculative prefill eliminates decode steps wholesale by verifying the
entire draft in one parallel prefill pass.

**They are perfectly complementary.** Speculative prefill tries the big
bet first: verify the whole draft in one shot. When the draft diverges
at position K, the target decodes from K onward — and that tail decode
can itself use standard spec decode to go faster.

6. **`pick_draft_model()` in ingress** — given `ModelTargets` and the
   target model name, select the best draft model. Priority: same-family
   smallest local → any smallest local → same-family smallest remote →
   any smallest remote. Uses `available_model_sizes` from gossip. Any
   model works — the mesh might have a MiniMax or Llama that can still
   produce structurally plausible text for a Qwen target.

7. **Generate complete draft via `ModelBackend`** — reuse the
   `LocalModelBackend` / `RemoteModelBackend` from MoA to call the draft
   model with:
   - Thinking disabled (`/no_think` chat template)
   - Short context (last 4–6 messages, not full history)
   - Low temperature (greedy or near-greedy)
   - The draft model produces a complete text answer, not token IDs

8. **Verify-and-commit in skippy-server** — new request path: when the
   proxy provides a `draft_response` text field alongside the normal
   chat completion request:
   - Tokenize the full prompt using the target's chat template
   - Tokenize the draft text using the target's tokenizer
   - Prefill the prompt tokens normally
   - Call `verify_tokens()` on the draft tokens — this is ONE prefill
     pass, one network round trip through all split nodes, regardless
     of how long the draft is
   - Find divergence K using `classify_verify_span()` (already exists)
   - K == end → return draft as response (zero decode round trips)
   - K < end → KV cache is warm to K, decode from K onward. The
     standard spec decode loop (`DraftRunner`) can accelerate this
     tail decode if a draft model is configured.

9. **Automatic engagement in proxy** — the proxy detects:
   - A draft model is available (from `pick_draft_model()`)
   - Request doesn't have `"speculative_prefill": false`
   When conditions hold, transparently engage spec prefill before
   forwarding the request to the target.

### Phase 3: Adaptive and production hardening

10. **Acceptance rate tracking** — per (draft, target) model pair, track
    running acceptance rate. Disable speculation if acceptance is
    consistently < 10% (the draft doesn't help for this pair).

11. **Draft timeout** — if draft generation takes longer than estimated
    decode time, abort and fall through to normal decode.

12. **Streaming prefill emission** — emit the verified draft prefix as
    SSE chunks immediately, then stream decode tokens for the tail.

## Relationship to Existing Systems

### vs. Standard speculative decoding (llama.cpp / skippy DraftRunner)

Standard spec decode drafts small batches of N tokens and verifies them
in iterative rounds. It speeds up decode everywhere — single-node and
distributed. The skippy-server already has a full implementation:
`DraftRunner`, `classify_verify_span`, adaptive window, checkpoint/restore.

Speculative prefill takes a different approach: generate the **entire
answer** up front, then verify it all in **one prefill pass**. The key
insight is that prefill is parallel (one network round trip across a
distributed split regardless of length), while decode — even with
standard spec decode — is sequential (one round trip per verify window).

| | Standard Spec Decode | Speculative Prefill (this design) |
|---|---|---|
| Draft size | N tokens (small rolling window) | Full response |
| Verification | Iterative rounds | One prefill pass |
| When it helps | Always (local + distributed) | Most on distributed splits (high per-token RTT) |
| On divergence | Re-draft from rejection point | Decode from K (can use spec decode for the tail) |
| Complementary? | Yes | Yes — prefill trick first, spec decode for the tail |

**They compose perfectly.** Speculative prefill tries the big bet: verify
the whole draft in one pass. When the draft diverges at position K, the
target decodes from K onward — and standard spec decode can accelerate
that tail decode. The prefill trick eliminates the easy prefix; spec
decode speeds up the hard tail.

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

### CLI flags

```
--prefill-speculative <path>       Local GGUF for in-process speculative prefill
--prefill-speculative-max <n>      Max draft tokens (default: 8)
```

**Opt-in only.** Speculative prefill is only active when
`--prefill-speculative` is explicitly set. There is no automatic
detection — the user must provide a path to a local GGUF.

`--prefill-speculative` loads a small model in-process via
`DraftRunner` — it shares the main model's GPU and runs
`prefill_chunk()` + `decode_step()` directly. Same-tokenizer
assumption: draft and target must share vocabulary (same model
family, e.g. Qwen3-0.6B → Qwen3-8B). The GGUF must be
downloaded locally.

Proxy-level speculative prefill (see "Automation strategy" below)
operates at the **proxy layer**, not inside skippy-server's decode
loop. The proxy selects a draft model from the mesh, gets a
complete text response, and injects it via `draft_response` into
the target's request for one-pass verification.

Both paths converge in `spec_prefill::verify_draft()`, which
accepts either `draft_tokens` (token IDs from DraftRunner) or
`draft_text` (text from proxy-level draft). They compose: when
spec prefill diverges at K, the tail decode from K benefits from
standard spec decode.

### Automation strategy

Three tiers of automatic draft model selection, from simplest to
most powerful:

**Tier 1: Local HF cache scan (in-process DraftRunner)**

On startup, scan `~/.cache/huggingface/` for a small GGUF from the
same model family as the target. If found, load it as DraftRunner
automatically. Same-tokenizer constraint applies.

Family matching: extract the family prefix from the target model name
(e.g. `Qwen3` from `Qwen3-8B`, `Llama-4` from `Llama-4-Scout`).
Look for the smallest downloaded GGUF with the same prefix under a
size threshold (e.g. 1 GB). Load it in-process.

This is zero-cost to implement — the HF cache scanner
(`scan_installed_models()`) and DraftRunner already exist. Just
connect them at startup.

**Tier 2: Mesh-native draft selection (proxy-level)**

The mesh gossips `available_model_sizes: HashMap<String, u64>` per
peer. At request time, the proxy picks the fastest available small
model as a draft source:

1. Same family, smallest, local → best (zero draft network cost)
2. Same family, smallest, fast peer → good (1 RTT for full draft)
3. Any small model, local → OK (lower acceptance, zero draft cost)
4. Any small model, fast peer → OK (1 RTT, lower acceptance)

The proxy calls the draft model via HTTP (local port or QUIC
tunnel), gets a text response, and injects it as `draft_response`
in the target request. No tokenizer constraint — the target
re-tokenizes the text with its own tokenizer.

This is the meshLLM-native approach. It reuses `build_moa_config()`
patterns from `ingress.rs` (local port vs QUIC peer lookup) and
the existing `draft_response` request field.

**Tier 3: Adaptive draft selection (learned)**

Track per-model-pair acceptance rates over time. Use observed rates
to prefer draft models that empirically agree with the target. Demote
models that consistently diverge early. This is future work but the
acceptance telemetry is already emitted.

### What conditions trigger automatic spec prefill

1. The target model is served via a **distributed layer split** (the
   only case where decode latency is dominated by network RTT).
2. A **fast draft model is available** (tier 1 or tier 2).
3. The request is **not a tool call** (tool routing depends on
   generation output, which the draft can't anticipate).
4. `speculative_prefill` is not explicitly `false` in the request.

When all four conditions hold, the proxy engages spec prefill
transparently. No user action required.

### What this is NOT

- **Not a virtual model.** MoA uses `model: "mesh"` because MoA changes
  *what* you're talking to (multiple models). Spec prefill doesn't change
  the model — it's the same model, just faster. No new model name.

- **Not blocking.** If draft generation takes too long or the draft model
  is unavailable, the request falls through to normal decode. Spec prefill
  is best-effort — the worst case is equivalent to no speculation.

- **Not exclusive.** Spec prefill and standard spec decode compose. The
  prefill trick skips the prefix; spec decode speeds up the tail.

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
