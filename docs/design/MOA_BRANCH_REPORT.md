# `micn/moa` branch report

A summary of the critical improvements landed on the `micn/moa` branch
beyond the original MoA gateway feature. These are documented here so
PR #566 reviewers and future readers can see what was found, what was
fixed, and what is still open.

## Scope at a glance

The branch started as a single feature — adding `model: "mesh"` as a
mixture-of-agents routing mode on top of the mesh. While iterating on
the PR #566 review feedback, the branch also surfaced and fixed
several **unrelated, pre-existing bugs in the host runtime** that were
either hard to hit before (no other code path exercised them) or
silently masked by other bugs that have since been fixed.

The MoA-specific feature work is summarized in `docs/design/MOA_GATEWAY.md`.
This report covers the non-MoA fixes the branch picked up along the way.

## Critical fixes beyond the MoA gateway

### 1. Mesh QUIC keep-alive (commit `f5cf4b86`)

**Symptom**: mesh inference calls that took more than ~30 seconds —
typical for reasoning models on large prompts — would suddenly drop
mid-stream with `recv: read error: connection lost` and the host log
would show `Connection to <peer> closed: timed out` immediately
followed by `noq_proto::connection: failed closing path
err=LastOpenPath`. The remote peer was still actively producing
tokens; the QUIC connection itself was being torn down.

**Root cause**: noq-proto's default `max_idle_timeout` is ~30s and
`keep_alive_interval` is `None`. RFC 9000 §10.1.2 makes keep-alive
opt-in, and quinn / noq follow that. A non-streaming inference
request (e.g. an MoA reducer call or any `stream: false` HTTP call)
sends no application bytes while the remote model is generating
tokens, so the wire looks idle for the entire 30-90s the remote
model spends thinking. Under concurrent load (parallel MoA workers
plus reducer plus gossip plus heartbeats), noq's multipath
bookkeeping closes the idle path, and when that path is the last
open one the entire connection drops.

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

**Verification**: on a 2-node mesh (Mac M4 Max + Mac Studio M3 Ultra)
the same OpenCode `model: mesh` agent loop that previously dropped 0
of 2 turns successfully now lands all turns up to the unrelated
context-overflow case described below. The 60s idle test goes from
2× `Connection closed: timed out` events to zero.

This fix is broadly applicable: any long non-streaming RPC across
the mesh benefits, not just MoA. Single-model `auto` routing for
slow reasoning models also benefits.

### 2. Auto-planner lane count capped to llama-server's default (commit `1b901219`)

**Symptom**: under modest concurrency (3 simultaneous large-prompt
requests) the embedded skippy stage runtime returned HTTP 502:

    skippy ABI call failed: RuntimeError: llama_decode failed

with the underlying skippy native log showing:

    decode: failed to find a memory slot for batch of size 2048

This was visible on a Mac M4 Max serving Qwen3-8B at the model's
native 32k context, where the planner had picked `slots = 16`.

**Root cause**: skippy's stage-runtime patches set `kv_unified = true`
whenever `lane_count > 1`
(`third_party/llama.cpp/patches/0034-Add-shared-execution-lanes-to-skippy-ABI.patch`).
In unified mode llama allocates **exactly `n_ctx` cells total**,
shared across all `n_seq_max` sequences. The previous planner derived
`slots` from VRAM as if each lane carved off its own
`n_ctx × bytes_per_token` allocation — which would be the
`kv_unified = false` semantics, not what skippy actually does. On any
machine with comfortable VRAM the math happily returned the snapped
maximum of 16 lanes, even though all 16 race for the same fixed pool
of `n_ctx` cells. With 16 lanes admitted, 3 concurrent ~14k-token
requests need ~45k cells in a 32k pool, the third request hits
`find_slot` failure, and the embedded server surfaces it as an
opaque mid-flight 502.

The recent KV-reclaim hardening fixes (`ab933b1d`, `95d3fed2`,
`28e142b6`) close *leakage* bugs that used to also produce
`find_slot` failures. They do not address honest overcommit, which
is what this case is.

**Fix**: drop `MAX_AUTO_PARALLEL_SLOTS` from 16 to 4 in
`crates/mesh-llm-host-runtime/src/runtime/context_planning.rs`,
matching upstream llama-server's own auto default for the same
`kv_unified = true` reason (`.deps/llama.cpp/tools/server/server.cpp`
logs `n_parallel is set to auto, using n_parallel = 4 and
kv_unified = true`). Lane count under `kv_unified = true` is purely
a concurrency-policy knob; it does not change the KV cache
allocation. Going from 16 to 4 frees zero RAM. It just gates
admission to a sane number of concurrent in-flight requests for the
shared cell pool.

Operators who know their workload (short chat turns,
low-concurrency hosts) can still raise it via the existing
`parallel_override` plumbing — also exposed via
`[models.throughput] parallel = N` in the TOML config from PR #564.

**Verification on the same M4**:

* `n_seq_max` in the llama log goes from 16 to 4.
* KV cache size unchanged at 2448 MiB (one shared buffer; lane
  count does not multiply allocation).
* The exact 3-parallel ~15k-prompt repro that previously returned
  3× HTTP 502 now succeeds: 3/3 `finish_reason=stop`, full content,
  ~20s wall, zero `find_slot` failures, zero `llama_decode` errors.
* Burst of 5 concurrent: 4 succeed; the 5th hits admission control
  cleanly with `{"type":"rate_limit_error","code":"rate_limit_exceeded"}`
  after the queue admission timeout, rather than an opaque
  mid-flight 502.

Adds two regression tests:

* `auto_slots_capped_at_llama_server_default` covers the high-VRAM
  small-model scenario.
* `explicit_parallel_can_exceed_auto_ceiling` covers the operator
  override path.

### 3. `/v1/models` advertises quant-suffix IDs that round-trip (commit `f3355bfd`)

**Symptom**: review items 1 + 2 of PR #566. Models in `/v1/models`
would sometimes be advertised under a name that the request router
could not route back to. Calls to those IDs returned 404 even though
the model was actively serving.

**Root cause**: two cooperating bugs in the model-ref resolver:

* `quant_selector_from_gguf_file` matched the GGUF filename's quant
  marker (`Q4_K_M`, `Q8_0`, etc.) case-sensitively, so any GGUF on
  disk whose filename used lowercase letters (e.g.
  `qwen2.5-3b-instruct-q4_k_m.gguf`) silently returned `None` and
  the resolver lost the quant tag.
* `public_huggingface_model_ref` only handled `artifact` as a GGUF
  filename. When the artifact field came in as a selector string
  (`Q4_K_M` directly) it discarded the quant component entirely and
  collapsed the public id to the bare repo.

The combined effect was that `/v1/models` would render
`unsloth/Qwen3-8B-GGUF` instead of `unsloth/Qwen3-8B-GGUF:Q4_K_M`,
which then could not be rewritten back through
`rewrite_public_model_alias` because no descriptor matched the
bare-repo form.

**Fix**: case-insensitive quant marker matching that preserves
original case in the returned selector, and a second resolver path
that accepts `artifact` as either a filename or a selector. Lossy
descriptors fall back to file-based or model-name-verbatim ids
instead of producing an id that cannot route.

**Scope**: this fix is not MoA-specific. It affects every code
path that consumes `/v1/models` or per-peer served-model
descriptors: the listing itself, request-time alias rewriting,
gossip-based served-model advertisement, auto-route classification,
the management API's `/api/model-targets`, and the web UI model
picker. Any user calling a peer-hosted model by the id printed in
`/v1/models` was previously at risk of 404; that path is now
lossless for the quant-suffixed Hugging Face flavors we use in
practice.

### 4. MoA review fixes (review items 3, 4, 5, 6)

A cluster of fixes addressing the rest of the PR #566 review:

| Commit       | Item | Description |
|--------------|------|-------------|
| `5169d120`   | 4    | All-workers-fail now returns a distinguishable error response (top-level `error`, `finish_reason: "error"`) and the ingress layer surfaces it as HTTP 502 instead of a misleading 200 with empty content. |
| `a396ab1e`   | 3    | `worker_summaries` now reconciles aborted workers correctly; previously they were silently dropped from the summary count, making the "0 of N succeeded" log misleading. |
| `000ae50c`   | 5    | Tool-result follow-up turns are now routed to the reducer (`pack_for_tool_result_turn`) instead of being treated as a fresh fanout turn. The classifier walks the message list backwards to find the most recent tool-call assistant message. |
| `64c4e0ec`   | 6    | Worker outputs that emit OpenAI-shape inline tool JSON (`{"name":..., "arguments":...}`) inside a text block are now normalized into proper structured `tool_calls`, so a worker that "almost did the right thing" doesn't get arbitrated as plain prose. |
| `d5279656`   | -    | Worker / reducer timeouts bumped from 15s to 60s so that big-tier reasoning models on agent-scale prompts get a real chance to land before MoA early-exits to the survivors. |

### 5. MoA `RemoteModelBackend` ergonomics

Smaller items not on the review list but found while debugging the
above:

* `unsloth/MiniMax-M2.5-GGUF:Q4_K_M` and similar quant-suffixed IDs
  are reliably dispatched across the mesh now that #3 above is fixed.
* The 60s timeout (item 5) is tracked separately from the QUIC
  keep-alive in #1. They are orthogonal — keep-alive prevents the
  *connection* dropping, the timeout bounds how long MoA waits for
  a single worker call's *response*.

## Confirmed working end-to-end on a real 2-node mesh

The branch was validated with these positive scenarios on the live
M4 Max + Mac Studio M3 Ultra setup:

* Direct `auto` and `mesh` routing for short prompts and reasoning
  prompts (51s `auto` to MiniMax through QUIC tunnel completes
  cleanly — would previously have hit the idle-timeout drop).
* MoA fanout: 2/2 workers across local + remote model, tool-call
  proposal arbitrated correctly, structured `tool_calls` returned
  to the client.
* MoA tool-result follow-up: routed to reducer (item 5), reducer
  produces final synthesis, returned as `chat_response`.
* 3 parallel agent-prompt requests on the local 32k Qwen3-8B (the
  exact failure case that drove the lane-cap fix in #2) now all
  succeed.

### Agent harness validation

To prove `model: "mesh"` works as an agent target end-to-end — not
just per-turn — the branch was driven with three different agent
harnesses against the live 2-node mesh:

**1. `goose run` (real agent harness)** with `GOOSE_MODEL=mesh`
pointing at the M4 OpenAI proxy:

```
$ GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9447 \
  OPENAI_API_KEY=mesh GOOSE_MODEL=mesh \
  goose run --no-session -t "Read /tmp/moa-opencode-test/calc.py \
  and tell me what bug you see. Be concise."

  ▸ shell
    command: cat /tmp/moa-opencode-test/calc.py

def add(a, b):
    return a - b  # BUG: should be `a + b`
...

The bug is in the `add` function: it returns `a - b` instead of
`a + b`. The comment even notes this is a bug.
```

32 seconds, 3 MoA turns (`fanout`, `fanout`, `tool-result reducer`),
zero KV exhaustion, zero connection drops, correct answer.

**2. Minimal Python agent loop** (system prompt small enough that
the local Qwen3-8B reducer can complete every turn without context
overflow):

```
--- TURN 1 (2 msgs) ---  status=200 elapsed=3.8s finish=tool_calls
    tool: read_file({'path': '/tmp/moa-opencode-test/calc.py'})
--- TURN 2 (4 msgs) ---  status=200 elapsed=3.9s finish=stop
  FINAL ANSWER (167 chars):
    The bug is in the `add` function: it returns `a - b` instead
    of `a + b`. When called with `add(2, 3)`, it returns `-1`
    instead of `5`, causing the assertion to fail.
  ✅ correctly identified the bug
```

`moa: 3687ms, 2/2 workers, kind=early-exit, reducer=false` for turn
1; `moa: 3806ms, 1/1 workers, kind=tool-result, reducer=true,
attempts=1` for turn 2. ~8s total.

**3. Multi-turn exploration agent** (the agent has to list a dir,
read three files, summarize): 7 turns, ~56 seconds total. MoA
reducer hedge ladder (`b0f5d6ad`) gracefully falls over from the
remote MiniMax reducer (which transiently 502'd on this run) to
the local Qwen3-8B reducer on every tool-result turn. Test
completed with a coherent summary, no KV exhaustion, no connection
drops.

### The OpenCode `model: "mesh"` case is still open

OpenCode's default system prompt is ~14k tokens. Running it against
a 32k-context local reducer (Qwen3-8B in our test) does not
complete the agent loop end-to-end — by the 3rd or 4th tool-result
turn the cumulative MoA reducer prompt (OpenCode's 14k-token system
prompt plus accumulated `assistant(tool_call)` / `tool(result)`
pairs plus the 2k `max_tokens` reserve) exceeds 32k cells and the
final reducer call fails. This is the deferred `context_overflow`
failure path documented in the next section. It reproduces on
`model: "mesh"` and on `model: "auto"` routed to Qwen3-8B — it is
not specific to MoA.

For agent harnesses with smaller system prompts (Goose, custom
harnesses) `model: "mesh"` works end-to-end on the same hardware.

## Known still-open: large MoA reducer prompts overflowing `n_ctx`

The OpenCode `model: "mesh"` agent loop still does not run to
completion on a Qwen3-8B local reducer because the cumulative MoA
reducer prompt (OpenCode's ~14k-token system prompt, plus
accumulated `assistant(tool_call)` / `tool(result)` pairs, plus the
2048-token `max_tokens` reserve) exceeds the model's 32k `n_ctx`
within a few turns. This is **not** an MoA bug per se; it is the
same shape as any local serving setup whose `n_ctx` is smaller than
the agent harness's prompt growth.

Plausible avenues, in increasing order of work:

1. Serve `model: "mesh"` with a bigger-context local reducer
   (Qwen3-32B at 65-128k, MiniMax at 128k+, etc.) so OpenCode's
   prompt growth fits comfortably. This is a deployment fix, not a
   code fix.
2. Have MoA `pack_for_tool_result_turn` clamp its forwarded
   message list to the reducer model's effective context budget,
   not just walk back to the most recent `tool_calls`. The
   `max_tokens` clamp from `28e142b6` was a similar fix at the
   other end (output budget); the input side needs the same
   treatment.
3. Have skippy distinguish a `context_overflow` failure from
   `llama_decode failed` so MoA's hedged reducer ladder can fail
   over to a larger-context candidate cleanly instead of treating
   it as a generic 502.

(1) is the immediate workaround for users. (2) is the right fix for
the MoA code path. (3) is defensive and orthogonal. None of these
are in scope for the current branch.

## What this means for the PR

PR #566 lands:

* The MoA mesh-native gateway feature itself.
* The PR review items (3, 4, 5, 6) plus the 1/2 model-id round-trip.
* Two pre-existing host runtime bugs that the MoA work surfaced and
  fixed: mesh QUIC keep-alive and unified-KV lane planning. These
  benefit every user of mesh-llm, not just MoA users.

The remaining open item (large MoA reducer prompts on small-`n_ctx`
local reducers) is documented but deferred — it should be a separate
follow-up PR, ideally bundled with a deployment-side note that
recommends a ≥64k-context reducer for OpenCode-class agent loops.
