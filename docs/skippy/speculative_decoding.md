# Speculative Decoding Outstanding Work

This note tracks open work for n-gram speculative decoding. The broader usage
guide and latest benchmark tables live in [`../SPECULATIVE_DECODING.md`](../SPECULATIVE_DECODING.md).

## Current State

N-gram speculative decoding is implemented and useful, especially for repeated
coding/editing sessions. It is model-free: the pool observes accepted target
tokens, proposes continuations when a context suffix repeats, and the staged
target verifies every proposed token through `VerifySpan`.

Current policy:

- Use n-gram speculation for coding-shaped sessions and repeated edit loops.
- Do not expect large wins on cold, one-shot, open-ended chat.
- Keep the default n-gram confidence policy flat at 55% until the verifier path
  is redesigned around actual verifier cost.
- Treat n-gram pooling as independent from KV/full-state cache. It remains safe
  for recurrent families such as Qwen3.6 because it does not restore model
  state.

## Outstanding Work

### Batched Target Verification

Verification is still the governor. Warm n-gram runs show useful acceptance, but
the live staged path still spends too much wall time in target verification,
stage forwarding, and repair bookkeeping.

Open items:

- Investigate true batched target verification for multi-token n-gram spans.
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
