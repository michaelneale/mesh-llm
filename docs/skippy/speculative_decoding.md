# Speculative Decoding Outstanding Work

This note tracks open work for n-gram speculative decoding. The broader usage
guide and latest benchmark tables live in [`../SPECULATIVE_DECODING.md`](../SPECULATIVE_DECODING.md).

## Current State

N-gram speculative decoding is implemented and useful, especially for repeated
coding/editing sessions. It is model-free: the pool observes accepted target
tokens, proposes continuations when a context suffix repeats, and the staged
target verifies every proposed token through `VerifySpan`.

Latest local evidence:

- Qwen3.6-35B-A3B with adaptive n-gram and an 8-token maximum speculation
  window reached `1.33x` on the warm coding-loop gate while staying near-neutral
  on mixed traffic (`0.99x`). Fixed n-gram with the same wide window regressed
  mixed traffic (`0.94x`) and only reached `1.10x` on coding-warm, so wider
  windows should stay behind adaptive policy.
- Qwen3.6-35B-A3B with `--openai-ngram-auto` reached `1.18x` on the warm
  coding-loop gate in the tightened detector run, close to manual adaptive
  (`1.20x` in the same run), while mixed traffic stayed near neutral (`0.97x`).
  The auto detector woke mostly on path/code-shaped prompts and still allowed
  repeated suffix hits to probe cautiously.
- After adding the auto warmup gate, Qwen3.6-35B-A3B `--openai-ngram-auto`
  reached `1.25x` on warm coding-loop traffic and `1.00x` on mixed traffic in
  the local gate. Auto mode now observes repeated suffix hits before spending
  verifier work unless the prompt is already code/structured-shaped.
- A longer Qwen3.6-35B-A3B gate with 64 generated tokens reached `1.23x` on
  warm coding-loop traffic and `1.05x` on mixed traffic. A depth-4 concurrency
  stress run completed with zero errors and stayed neutral (`1.01x` mixed,
  `0.99x` coding-warm), so auto n-gram is default-ready for skippy serving but
  not yet a concurrency throughput claim.
- A compatible Llama 3.2 3B target plus Llama 3.2 1B draft pair loaded through
  the same benchmark loop, but draft speculation lost throughput (`0.56x` mixed,
  `0.52x` coding-warm for draft-adaptive). This validates the draft comparison
  path but does not identify a useful default assistant.

Current policy:

- Use n-gram speculation for coding-shaped sessions and repeated edit loops.
- Skippy serving should prefer `--openai-ngram-auto`
  `--openai-adaptive-speculative-window` with an 8-token speculation window.
- Do not expect large wins on cold, one-shot, open-ended chat.
- Keep the default n-gram confidence policy flat at 55% until the verifier path
  is redesigned around actual verifier cost.
- Treat n-gram pooling as independent from KV/full-state cache. It remains safe
  for recurrent families such as Qwen3.6 because it does not restore model
  state.

## Near-Term Plan

The next decode-performance lane should make adaptive n-gram speculation a
session-local policy first, then use the resulting traces to drive cheaper
batched verification. This keeps the initial change small and measurable while
building the data needed for the deeper `VerifySpan` rewrite.

### M1 - Speculative Window Telemetry

Record enough per-window data to explain whether speculation actually helped:

- proposal source: n-gram, draft, or future hybrid;
- requested window size, proposed token count, and accepted prefix length;
- rejection point and rejection reason, when known;
- `VerifySpan` wall time, verifier compute time, and downstream/stage wait time;
- repair path, repair wall time, and restored/reverified token count;
- recent non-speculative decode ms/token for the same session or model route;
- effective ms/token and throughput for the speculative window.

Acceptance:

- benchmark reports and runtime telemetry can show why a window was profitable
  or unprofitable;
- telemetry distinguishes proposal quality from verifier, transport, and repair
  costs;
- the fields are shared by n-gram and draft speculation where possible.

### M2 - Adaptive N-Gram Session Policy

Add a conservative controller that enables n-gram only when the active session
looks likely to benefit, then keeps adapting based on measured value.

Candidate enable signals:

- coding-shaped prompts: file paths, fenced code, diffs, compiler errors, stack
  traces, test output, symbols, and tool logs;
- structured-output prompts such as JSON, YAML, XML, SQL, or tool-call payloads;
- repeated prompt prefixes or suffixes inside the same session/project;
- n-gram pool hit rate above a small threshold;
- prior accepted-token volume from the same session/project pool.

Runtime policy:

- start with a small window such as `2` or `4`;
- grow the window only after full accepts or late-tail rejects beat baseline
  decode cost;
- shrink on early rejects, low accepted-token yield, or high verifier wait;
- disable for a cooldown period when effective ms/token is worse than recent
  non-speculative decode;
- retry cautiously when repeated suffix hits return after cooldown.

Acceptance:

- logs/API/telemetry can explain `enabled`, `grown`, `shrunk`, `disabled`, and
  `retried` decisions;
- disablement is treated as performance policy only; correctness still comes
  from target verification;
- pool keys remain isolated by model, tokenizer, tenant/project/session, pool
  ID, and n-gram size.

### M3 - Benchmark Gates

Use existing benchmark paths before promoting defaults:

- run smoke and long tiers for baseline, draft fixed/adaptive, n-gram
  fixed/adaptive, and adaptive-policy n-gram;
- use `just skippy-openai-ngram-bench` for the local staged OpenAI speculation
  gate; the harness compares fallback decode, fixed/adaptive n-gram, and
  optional fixed/adaptive draft-model modes when `DRAFT_MODEL_PATH` is set. It
  writes raw chat-corpus reports plus per-window telemetry summaries under
  `target/skippy-openai-spec-bench/<run-id>/`;
- include warm `coding-loop` runs because that is n-gram's expected win case;
- keep mixed-task runs so the policy proves it backs off on cold chat,
  instruction, summarization, and reasoning prompts;
- report by task type and proposal source with proposed tokens, accepted tokens,
  proposal length buckets, n-gram suffix match-order ranges, proposal stop
  reasons, skipped-proposal reasons, verifier cost, repair cost, fallback decode
  cost, completion throughput, and wall-clock speedup.

Acceptance:

- adaptive n-gram beats the matched non-speculative baseline on warm coding-loop
  runs after verifier and repair costs are included;
- adaptive policy does not regress mixed-task throughput beyond an agreed small
  tolerance;
- traces clearly identify whether remaining losses come from proposal misses,
  verifier cost, downstream wait, or repair overhead.

### M4 - Batched Verification Follow-Up

After adaptive-policy traces identify the dominant verifier costs, prototype a
cheaper linear-span verifier before adding tree-style or multi-proposer protocol
surface.

Open questions:

- how much `VerifySpan` time is target compute versus per-stage bookkeeping and
  transport;
- whether first-reject and tail-reject paths should use different repair
  strategies;
- whether block verification should stay linear-span-only or prepare for a
  later `VerifyTree`/ancestor-mask protocol;
- which telemetry fields must become stable before changing protocol messages.

Acceptance:

- verifier wall time drops for accepted multi-token spans;
- repair cost drops for common partial-accept cases;
- draft and n-gram speculation both benefit from the same verifier path;
- any protocol change remains additive or is explicitly reviewed as breaking.

## Outstanding Work

### Batched Target Verification

Verification is still the governor. Warm n-gram runs show useful acceptance, but
the live staged path still spends too much wall time in target verification,
stage forwarding, and repair bookkeeping.

Open items:

- Investigate true batched target verification for multi-token n-gram spans.
- Keep one-token verify windows on the skip-checkpoint path; they cannot need
  early-reject restore and should not pay checkpoint overhead.
- Skip one-token n-gram verification after a pool has tried verification but
  remains limited to a one-token window; record the skip, use normal decode for
  that token, and periodically re-probe.
- Let n-gram proposal use shorter suffixes when the configured n-gram order
  cannot continue, and report whether spans stop because of policy cap,
  proposal limit, min-hit filtering, or no suffix match.
- Allow bounded two-token probes when a pool with verified history is stuck at a
  one-token policy cap, and grow faster when those probes fully accept.
- Classify n-gram pools by observed quality: high-accept pools can probe caps
  more aggressively, while low-accept pools should cool down before they can
  harm mixed-prompt traffic.
- Reduce per-window protocol round trips and per-stage bookkeeping overhead.
- Compare block verification against tree-style verification before adding a
  larger public protocol surface.
- Keep measuring `verify_wall_ms`, verifier compute, downstream wait, protocol
  request count, protocol token count, max span, and average span.

### Rejection Repair

Early rejection still hurts n-gram more than proposal quality alone suggests.
The first-token early-reject fast path exists, but wider windows still pay too
much restore/reverify overhead.

Open items:

- Make repair decisions cost-aware, not only confidence/window-size aware.
- Preserve the tail-reject fast path.
- Avoid repair `VerifySpan` when a normal decode step is cheaper.
- Track repair cost by task type, not only globally.

### Pool Policy And Lifetime

N-gram pools are valuable while the user is iterating in the same context. They
are less valuable after a project/session has gone cold.

Open items:

- Add explicit pool TTL and LRU eviction policy.
- Keep pools in memory by default; avoid disk persistence until there is a clear
  reproducibility or resume requirement.
- Consider separate retention classes for session pools, project pools, and
  tenant-wide warm pools.
- Expose pool memory usage and candidate counts in telemetry.

### Concurrent Sessions

The server path needs to be boringly reliable under many prompt workers.

Open items:

- Stress-test `ngram-pool-server` with many concurrent session IDs.
- Shard or partition pool locks if contention appears.
- Verify pool keys include model, tokenizer, tenant, project, session, explicit
  pool ID, and n-gram size.
- Ensure failed or cancelled requests only observe accepted target tokens.

### Routing Policy

The OpenAI-compatible frontend should eventually route coding-shaped requests to
n-gram speculation before draft speculation when the session/project pool is
warm enough.

Open items:

- Add a conservative coding-prompt detector for file paths, fenced code, diffs,
  compiler errors, stack traces, tests, symbols, and tool logs.
- Use acceptance and verifier-cost telemetry to disable or shrink n-gram windows
  when a session is not benefiting.
- Keep routing as a performance policy only; correctness must always come from
  target verification.

### Benchmark Coverage

The current numbers are useful but not enough to lock policy.

Open items:

- Continue using HF-sourced benchmark corpora instead of checked-in large
  prompt bodies.
- Keep smoke and long tiers for all benchmark modes, not only speculation.
- Run warm coding-loop confirmation regularly because that is the expected
  n-gram win case.
- Report by task type, especially coding versus chat/instruction.
- Preserve raw logs under `target/prompt-spec-corpus/<timestamp>` for audit.

## Done Criteria For Promotion

N-gram should become an automatic first-choice coding strategy only after:

- warm coding-loop runs show consistent speedup over baseline;
- verifier wall time decreases, not just acceptance rate increasing;
- concurrent session stress runs do not show lock contention or pool bleed;
- telemetry can explain why a session enabled, shrank, or disabled n-gram;
- regression runs include at least smoke and long corpus tiers.
