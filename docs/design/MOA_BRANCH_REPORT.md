# `micn/moa` branch report

A summary of the critical improvements landed on the `micn/moa` branch
beyond the original MoA gateway feature. Reviewers can use this to see
what was found, what was fixed, and what is still open without having
to spelunk through 60+ commits.

## Scope at a glance

The branch started as a single feature — adding `model: "mesh"` as a
mixture-of-agents routing mode on top of the mesh. While iterating on
the PR #566 review feedback, the branch also surfaced and fixed
several **non-MoA bugs in the host runtime** that were either hard to
hit before (no other code path exercised them) or silently masked by
other bugs that have since been fixed.

The MoA-specific feature work is summarized in `docs/design/MOA_GATEWAY.md`.
This report covers:

1. The MoA review-feedback fixes that land in this PR.
2. The pre-existing host-runtime bugs the MoA work surfaced and fixed
   — those are **generally applicable** to all mesh-llm users, not
   just MoA users.
3. End-to-end validation against three different agent harnesses.
4. The one known-open item.

## MoA review-feedback fixes (items 1-6 from PR #566 review)

These are the targeted fixes for the review items themselves.

| Commit       | Review item | What it does |
|--------------|-------------|--------------|
| `f3355bfd`   | 1 + 2       | `/v1/models` advertises quant-suffix IDs that round-trip back to the correct backend. Two cooperating bugs in the model-ref resolver: `quant_selector_from_gguf_file` was uppercase-only when matching markers (lowercase quant filenames lost their tag), and `public_huggingface_model_ref` only handled artifact-as-filename, dropping the quant when artifact came in as a selector. Now lossless for the HF-quant-suffix flavors we use in practice. |
| `a396ab1e`   | 3           | `worker_summaries` reconciles aborted workers correctly via `reconcile_dispatched` in fanout. Previously aborted workers were silently dropped from the summary count, making the "0 of N succeeded" log misleading. |
| `5169d120`   | 4           | All-workers-fail returns a distinguishable error response (top-level `error` field, `finish_reason: "error"`) and the ingress layer surfaces it as HTTP 502 instead of a misleading 200 with empty content. |
| `000ae50c`   | 5           | Tool-result follow-up turns are now routed to the reducer (`pack_for_tool_result_turn`) instead of being treated as a fresh fanout turn. The classifier walks the message list backwards to find the most recent tool-call assistant message. |
| `64c4e0ec`   | 6           | Worker outputs that emit OpenAI-shape inline tool JSON (`{"name":..., "arguments":...}`) inside a text block are now normalized into structured `tool_calls`, so a worker that "almost did the right thing" doesn't get arbitrated as plain prose. |
| `d5279656`   | -           | Worker / reducer timeouts bumped from 15s to 60s so big-tier reasoning models on agent-scale prompts get a real chance to land before MoA early-exits to the survivors. Validation showed the 15s budget was being eaten entirely by reasoning preambles on big agent prompts. |

## Pre-existing host-runtime bugs surfaced by MoA — **generally applicable**

These are the bugs the MoA work uncovered that affect **every user of
mesh-llm**, not just MoA users. They were silently latent because no
other code path exercised them.

### A. Mesh QUIC keep-alive (commit `f5cf4b86`)

**Symptom**: any mesh inference call that takes more than ~30 seconds
— typical for reasoning models on large prompts — drops mid-stream
with `recv: read error: connection lost`. Host log shows
`Connection to <peer> closed: timed out` immediately followed by
`noq_proto::connection: failed closing path err=LastOpenPath`. The
remote peer is still actively producing tokens; the QUIC connection
itself is being torn down.

**Root cause**: noq-proto's default `max_idle_timeout` is ~30s and
`keep_alive_interval` is `None`. RFC 9000 §10.1.2 makes keep-alive
opt-in; quinn / noq follow that. A non-streaming inference request
(e.g. an MoA reducer call, or any `stream: false` HTTP call from a
non-streaming client) sends no application bytes while the remote
model is generating tokens, so the wire is idle for the entire
30-90s the remote model spends thinking. Under concurrent load
(parallel MoA workers + reducer + gossip + heartbeats) noq's
multipath bookkeeping closes the idle path, and when that path is
the last open one the entire connection drops.

This had been latent for as long as mesh-llm has run on noq because:

* Every previous OpenAI client (Goose, Claude Code, pi, the web UI)
  defaulted to `stream: true`. SSE chunks reset the idle timer.
* MoA `RemoteModelBackend` is the first significant non-streaming
  long-running mesh RPC in the codebase
  (`crates/mesh-mixture-of-agents/src/backend.rs` hardcodes
  `"stream": false`).
* Reasoning models with big agent prompts (MiniMax-M2.5 on a 13k
  OpenCode system prompt) routinely take 30-90s to produce a first
  useful response, which is exactly what crosses the 30s idle
  window.

**Fix**: set `keep_alive_interval = 10s`, `max_idle_timeout = 5m`,
plus matching multipath `default_path_keep_alive_interval` and
`default_path_max_idle_timeout` on the mesh QUIC transport config in
`crates/mesh-llm-host-runtime/src/mesh/mod.rs`. Cost is one PING
frame (~30-60 bytes) every 10s per connection only when no other
data has flowed for that long.

**Scope**: every long non-streaming RPC across the mesh benefits.
Single-model `auto` routing for slow reasoning models, artifact
transfers, future RPC types — all healthier with this in place.

**Verification**: on the live 2-node mesh (Mac M4 Max + Mac Studio
M3 Ultra) the same OpenCode `model: mesh` agent loop that
previously dropped 0 of 2 turns successfully now lands every turn.
The 60s idle test goes from 2× `Connection closed: timed out` events
to zero.

### B. Auto-planner lane count capped to llama-server's default (commit `1b901219`)

**Symptom**: under modest concurrency (3 simultaneous large-prompt
requests) the embedded skippy stage runtime returns HTTP 502:

    skippy ABI call failed: RuntimeError: llama_decode failed

with the underlying skippy native log showing:

    decode: failed to find a memory slot for batch of size 2048

Visible on a Mac M4 Max serving Qwen3-8B at the model's native 32k
context, where the planner had picked `slots = 16`.

**Root cause**: skippy's stage-runtime patches set
`kv_unified = true` whenever `lane_count > 1`
(`third_party/llama.cpp/patches/0034-Add-shared-execution-lanes-to-skippy-ABI.patch`).
In unified mode llama allocates **exactly `n_ctx` cells total**,
shared across all `n_seq_max` sequences. The previous planner
derived `slots` from VRAM as if each lane carved off its own
`n_ctx × bytes_per_token` allocation — which would be the
`kv_unified = false` semantics, not what skippy actually does. On
any machine with comfortable VRAM the math happily returned the
snapped maximum of 16 lanes, even though all 16 lanes race for the
same fixed pool of `n_ctx` cells. With 16 lanes admitted, 3
concurrent ~14k-token requests need ~45k cells in a 32k pool, the
third request hits `find_slot` failure, and the embedded server
surfaces it as an opaque mid-flight 502.

The recent KV-reclaim hardening fixes on main (`ab933b1d`,
`95d3fed2`, `28e142b6`) close *leakage* bugs that used to also
produce `find_slot` failures. They do not address honest
overcommit, which is what this case is.

**Fix**: drop `MAX_AUTO_PARALLEL_SLOTS` from 16 to 4 in
`crates/mesh-llm-host-runtime/src/runtime/context_planning.rs`,
matching upstream llama-server's own auto default for the same
`kv_unified = true` reason
(`.deps/llama.cpp/tools/server/server.cpp` logs `n_parallel is
set to auto, using n_parallel = 4 and kv_unified = true`). Lane
count under `kv_unified = true` is purely a concurrency-policy
knob; it does not change the KV cache allocation. Going from 16
to 4 frees zero RAM. It gates admission control to a sane number
of concurrent in-flight requests for the shared cell pool.

Operators who know their workload (short chat turns,
low-concurrency hosts) can still raise it via the existing
`parallel_override` plumbing — also exposed via
`[models.throughput] parallel = N` in the TOML config from PR #564.

**Scope**: every node running any single-GPU-class model with
unified KV is affected. The fix prevents the opaque 502 failure
mode for any client driving concurrent agent traffic — not just
MoA.

**Verification on the M4 (Qwen3-8B, 32k `n_ctx`) and Studio
(MiniMax-M2.5, 128k `n_ctx`)**:

* `n_seq_max` in the llama log goes from 16 to 4 on both boxes.
* KV cache size unchanged (one shared buffer; lane count does not
  multiply allocation): 2448 MiB on M4, 8928 MiB on studio.
* The exact 3-parallel ~15k-prompt repro that previously returned
  3× HTTP 502 now succeeds: 3/3 `finish_reason=stop`, full content,
  ~20s wall, zero `find_slot` failures, zero `llama_decode` errors,
  zero skippy ABI errors.
* Burst test (5 concurrent): 4 succeed; the 5th hits admission
  control cleanly with
  `{"type":"rate_limit_error","code":"rate_limit_exceeded"}` after
  the queue admission timeout, instead of an opaque mid-flight 502.

Adds two regression tests:

* `auto_slots_capped_at_llama_server_default` covers the high-VRAM
  small-model scenario that used to plan 16.
* `explicit_parallel_can_exceed_auto_ceiling` covers the operator
  override path.

### D. Prefix-cache budget bounded by KV cell pool, not just entries/bytes (commits `32061e8c`, `8cb6fe4b`)

**Symptom**: sustained agent traffic against a node serving via
skippy's unified-KV stage runtime exhausted the shared KV cell
pool. Observed on a Mac Studio M3 Ultra serving MiniMax-M2.5
(`n_ctx = 131072`): 20 consecutive Goose `model: "auto"` runs
against the standard `calc.py` fixture failed 14 of 20 from
request 7 onward with `RuntimeError: llama_decode failed`. The
embedded skippy native log showed
`decode: failed to find a memory slot for batch of size 1805`.

**Root cause**: the resident prefix cache
(`crates/skippy-cache/src/resident/prefix.rs`) only bounded itself
by `max_entries` and `max_bytes`. Under `kv_unified = true`
(skippy patch `0034-Add-shared-execution-lanes-to-skippy-ABI.patch`)
the cache shares the model's single `n_ctx` cell pool with the
active execution lanes — but the cache budget had no concept of
the cell pool at all. Live instrumentation showed:

    DBG record_resident_prefix ... cache_entries=12 resident_tokens=124084 estimated_bytes=15386416 evicted=0

Twelve cached prefixes averaging ~10k tokens each fit comfortably
under `max_entries = 16` and well under the `~9 GB max_bytes`
budget — yet they pinned **95% of the 131072-cell pool**. Active
lanes had nowhere to write fresh prompts.

**Three coordinated fixes**:

1. `family_policy.rs::estimate_stage_cache_max_bytes`: drop the
   `lane_count` multiplier. The total native KV memory is
   `bytes_per_token_layer * stage_layers * n_ctx` — lanes share the
   pool under `kv_unified = true`, they don't multiply it. Cache
   byte budget was 2–4× the actual KV memory.
2. `family_policy.rs::resident_kv_policy` and `kv_recurrent_policy`:
   `max_entries` 128 → 16, plus a budget-aware secondary cap
   `derive_max_entries_from_kv_cells` that clamps by
   `n_ctx / (2 * min_tokens)`.
3. **`skippy-cache::ResidentPrefixCache`**: add `max_resident_tokens`
   field to the cache config and to `evict_until_room_for`'s
   over-budget check. `ResidentCacheConfig::from_stage` derives
   the budget as `n_ctx / 2` so the cache may use at most half
   the cell pool. This is the **real fix** for the leak — LRU now
   triggers on cell pressure, not just entry/byte pressure.

**Scope**: every node serving a family that the policy puts on
`resident_kv_policy` / `kv_recurrent_policy` benefits. That is
essentially every dense LLM family the project supports today.
Not MoA-specific.

**Verification on the live 2-node mesh**:

| Setup | Goose `model: auto` 20-run | `memory slot` failures |
|---|---|---|
| Pre-fix (`max_entries=128`, no cell budget) | 6 / 20 pass | 14 |
| `max_entries=16` only (commit `32061e8c`) | 6 / 20 pass | 15 |
| Cache disabled completely (control) | 18 / 20 pass | 0 |
| **With cell-budget eviction (commit `8cb6fe4b`)** | **16 / 20 pass** | **0** |

The cell-budget eviction completely eliminates `memory slot`
failures in the studio's skippy native log. The remaining 4
flakes are unrelated agent-loop variance (a Goose `Stream decode
error` on one slow response, three model-quality outputs that
returned content other than the expected exact substring) — not
KV exhaustion.

The cache still gets its perf benefit (entries are recorded and
looked up), it just now evicts on the correct currency under
unified-KV serving.

**Tests added**:

* `token_budget_triggers_lru_before_entry_cap_under_unified_kv` in
  `crates/skippy-cache/src/resident/prefix.rs::tests`: with
  `max_resident_tokens = 4096` and 1500-token entries, the third
  insert evicts even though we're well under `max_entries = 16`.
* `zero_token_budget_disables_the_check`: legacy unbounded-by-tokens
  behavior preserved when `max_resident_tokens = 0`.

### C. Auto-router weights peers by observed throughput (commit `25248409`)

**Symptom**: on a public mesh with mixed hardware, `auto` picked
uniformly at random within the multi-digit-B tier. A fast MiniMax on
a 4090 and a slow 35B-A3B on an M2 Air were equally likely to be
chosen, even though `routing_metrics` had already measured the
throughput gap and was simply not being read in the picker.

**Fix**: each big-tier candidate is now weighted by its locally
observed `avg_tokens_per_second` (clamped to [5, 100] tok/s so
nothing fully starves and no outlier monopolizes). Models with
fewer than 3 samples get a neutral weight so they compete fairly
until data accumulates. A 15% exploration probability ignores
weights and picks uniformly, which keeps the system from locking
onto stale rankings and guarantees cold peers see traffic.

Adds `RoutingMetrics::tps_for_model` as a cheap per-model accessor
(locks only the relevant shard instead of allocating a full
HashMap snapshot in the hot path) and `Node::routing_metrics()` as
an Arc-backed public accessor.

The pre-existing `(name, score, caps)` tuple's middle slot was
literally always `0.0` at every populated call site. Replaced
with a `RoutingCandidate` struct that makes the throughput hint a
real, typed concept instead of a dangling hook.

**Scope**: every `auto`-using client on the public mesh benefits.
Not MoA-specific.

**Tests added**:

* `weighted_pick_all_cold_is_roughly_uniform` — regression safety.
* `weighted_pick_fast_wins_majority_but_slow_still_gets_some` —
  fast wins by ≥1.5× but slow still gets >30/600 picks.
* `weighted_pick_cold_model_competes_with_hot_fast` — newcomer
  gets >100/600 picks so it can accumulate samples and earn its
  score.
* `weighted_pick_low_sample_count_treated_as_cold` — 1-sample
  measurements don't dominate routing.
* `candidate_weight_clamps_extremes` — weight stays in [5, 100],
  cold = 25.

## Confirmed working end-to-end on a real 2-node mesh

All validation below was performed on the live M4 Max + Mac Studio
M3 Ultra setup with the branch HEAD binaries on both nodes.

### Direct calls

* Short prompts and reasoning prompts on `model: auto` and
  `model: <full-id>` complete cleanly to remote MiniMax through the
  QUIC tunnel even when the call takes 50-60s. Without fix A this
  would previously have hit the idle-timeout drop.
* 3 parallel ~15k-prompt requests to local Qwen3-8B all succeed
  with `finish_reason=stop`. Without fix B this returned 3× HTTP
  502 `llama_decode failed`.

### MoA primitives

* MoA fanout: 2/2 workers across local + remote model, tool-call
  proposal arbitrated correctly, structured `tool_calls` returned
  to the client.
* MoA tool-result follow-up: routed to reducer (item 5), reducer
  produces final synthesis, returned as `chat_response`.
* MoA hedged reducer ladder (`b0f5d6ad` from earlier on the
  branch): when the remote MiniMax reducer transiently 502s, MoA
  falls over to the local Qwen3-8B reducer on every tool-result
  turn without escalating the failure to the client.

### Agent harness validation

To prove `model: "mesh"` works as an agent target end-to-end — not
just per-turn — the branch was driven with three different agent
harnesses against the live 2-node mesh.

**1. `goose run` (real agent harness) with `GOOSE_MODEL=mesh`**:

```
$ GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9447 \
  OPENAI_API_KEY=mesh GOOSE_MODEL=mesh \
  goose run --no-session -t "Read /tmp/moa-opencode-test/calc.py \
  and tell me what bug you see. Be concise."

  ▸ shell
    command: cat /tmp/moa-opencode-test/calc.py
...
The bug is in the `add` function: it returns `a - b` instead of
`a + b`. The comment even notes this is a bug.
```

32 seconds, 3 MoA turns (`fanout`, `fanout`, `tool-result reducer`),
zero KV exhaustion, zero connection drops, correct answer.

**2. `goose run` with `GOOSE_MODEL=auto`** (control — confirms the
`auto` direct path still works for the same task):

3/3 runs succeed (~30s each). The router resolves to
`unsloth/MiniMax-M2.5-GGUF:Q4_K_M`, MiniMax produces native
`<minimax:tool_call>` output, skippy's chat parser interprets it
correctly, Goose acts on the structured `tool_calls`. Grammar log on
studio shows `Grammar triggered on regex: '<minimax:tool_call>'`
firing exactly as expected.

**3. Minimal Python agent loop** (small system prompt + 2 tools,
exercises `read_file` against `calc.py`):

5/5 `model: mesh` runs succeed (~7s each), every run correctly
identifies the bug. 3/3 `model: auto` runs also succeed.

**4. Multi-turn exploration agent** (the agent has to list a dir,
read three files, summarise): 7 turns, ~56 seconds total against
`model: mesh`. MoA reducer hedge ladder gracefully falls over from
remote MiniMax (which 502'd a couple of times on this run) to local
Qwen3-8B on every tool-result turn. Test completed with a coherent
summary, no KV exhaustion, no connection drops.

### What this validation does **not** claim

Single-shot Goose runs against a freshly-restarted studio (MiniMax
loading and warming up) very occasionally produce a malformed
first-token tool-call output that the chat parser rejects with
`Failed to parse input at pos N: <minimax:tool_call>...`. This was
observed once, did not reproduce across subsequent runs, and is
consistent with a transient lazy-grammar trigger race during cold
start. Worth flagging but not a regression introduced by this
branch.

## Known still-open

### 1. Large MoA reducer prompts overflowing `n_ctx`

The OpenCode `model: "mesh"` agent loop still does not run to
completion on a Qwen3-8B local reducer because the cumulative MoA
reducer prompt (OpenCode's ~14k-token system prompt plus
accumulated `assistant(tool_call)` / `tool(result)` pairs plus the
2k `max_tokens` reserve) exceeds the model's 32k `n_ctx` within a
few turns. This is **not** an MoA bug per se; it is the same shape
as any local serving setup whose `n_ctx` is smaller than the agent
harness's prompt growth. Confirmed: it reproduces equally on
`model: "auto"` routed to the same local Qwen3-8B.

Plausible avenues, in increasing order of work:

1. Serve `model: "mesh"` with a bigger-context local reducer
   (Qwen3-32B at 65-128k, MiniMax at 128k+, etc.) so OpenCode's
   prompt growth fits comfortably. This is a deployment fix, not a
   code fix.
2. Have MoA `pack_for_tool_result_turn` clamp its forwarded message
   list to the reducer model's effective context budget, not just
   walk back to the most recent `tool_calls`. The `max_tokens`
   clamp from `28e142b6` was a similar fix at the output end; the
   input side needs the same treatment.
3. Have skippy distinguish a `context_overflow` failure from
   `llama_decode failed` so the hedged reducer ladder can fail
   over to a larger-context candidate cleanly instead of treating
   it as a generic 502.

(1) is the immediate workaround for users. (2) is the right code
fix and lives in the MoA crate. (3) is defensive and orthogonal.
None of these are in scope for the current PR.

## Summary for PR review

PR #566 lands:

* The MoA mesh-native gateway feature itself
  (`crates/mesh-mixture-of-agents/`, ingress integration, design
  doc).
* The PR review items themselves (1–6).
* Four pre-existing host runtime bugs that the MoA work surfaced
  and fixed — all **generally applicable beyond MoA**:
  * **A**: mesh QUIC keep-alive (any long non-streaming mesh RPC)
  * **B**: unified-KV lane-count cap (any node, any client driving
    concurrent traffic to a single-GPU-class model)
  * **C**: throughput-weighted auto-router (any `auto`-using
    client on a public mesh)
  * **D**: prefix-cache budget bounded by KV cell pool, not just
    entries/bytes (any node hosting a dense LLM family that the
    auto policy puts on the resident-KV or KV-recurrent path —
    essentially every family the project supports; eliminates the
    "works at first then 502s under sustained agent traffic"
    failure mode)

The single open item (large MoA reducer prompts on small-`n_ctx`
local reducers) is documented above but deferred — it should be a
separate follow-up PR, ideally bundled with a deployment-side note
recommending a ≥64k-context reducer for OpenCode-class agent loops.

The branch is tagged at HEAD `b117f4d9` (this report) and the live
2-node validation above was performed on that HEAD.
