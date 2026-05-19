# Mixture-of-Agents (MoA) Gateway

Fan out requests to multiple heterogeneous LLM endpoints in parallel,
arbitrate responses with deterministic logic, manage tool call lifecycles,
and return one coherent OpenAI-compatible response.

**Crate:** `crates/mesh-mixture-of-agents/`  
**Virtual model:** `model: "mesh"` (not advertised in `/v1/models`)  
**Status:** Integrated into mesh proxy, live-tested with mesh peers

---

## How it works

```
       Agent / Goose / Claude Code / pi
              │
              │  POST /v1/chat/completions
              │  { model: "mesh", messages, tools, stream }
              ▼
    ┌──────────────────────────────────────────────────────┐
    │  Mesh proxy (ingress.rs)                             │
    │  - intercepts model == "mesh"                        │
    │  - build_moa_config(callable models in mesh)         │
    └──────────────────────┬───────────────────────────────┘
                           │ handle_turn(config, body)
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  MoA gateway (crate: mesh-mixture-of-agents)         │
    │                                                      │
    │  session.classify_turn()                             │
    │    │                                                 │
    │    ├─ Fresh / Continuation ──► fan-out path          │
    │    └─ ToolResult ────────────► reducer-only path     │
    └────┬─────────────────────────────┬───────────────────┘
         │ fan-out path                │ tool-result path
         ▼                             │
    [LLM call #1]                      │
    assign_roles(N) → fire N workers   │
    in parallel:                       │
                                       │
       ┌─── fast       (smallest)      │
       ├─── specialist                 │
       ├─── specialist  …  (N-2 of)    │
       └─── strong     (biggest)       │
                                       │
    wall-clock = slowest worker        │
         │                             │
         ▼                             │
    [code] arbiter                     │
         │                             │
         ├─ consensus → emit one       │
         │  worker's output  ──────► done (no reducer)
         │                             │
         └─ conflict ─────┐            │
                          ▼            ▼
                       [LLM call #2] reducer
                       hedged candidate ladder
                       (top-2 strongest from same pool)
                       1 call usually, 2 if primary slow
                          │
                          ▼
                       final response
```

The client thinks it talks to one model. `mesh` is a routing directive
like `auto` — the proxy intercepts it before normal model routing.

### How many models, and how many LLM calls

The number of workers is **not fixed** — it scales with whatever's
callable in the mesh.

| Callable models | Workers fanned out | Roles assigned |
|---:|:---:|:---|
| 0 or 1 | — | MoA bails (503 to client; needs ≥2) |
| 2 | 2 | fast + strong |
| 3 | 3 | fast + specialist + strong |
| 4 | 4 | fast + specialist + specialist + strong |
| N | N | fast + (N-2) specialists + strong |

Models are tier-sorted before role assignment:
single-digit-B names ("Qwen3-8B", "llama-3-7b") form the small tier and
get `Fast`; everything else (multi-digit B, or names without an explicit
size) forms the big tier — the largest of those gets `Strong` and also
heads the reducer pool.

**The reducer is not a separate fan-out slot.** It's the same strong
model (or next-strongest if the primary is slow/broken), invoked
*after* fan-out only when the arbiter says workers disagreed.

So serially, a worst-case MoA turn is **2 LLM round-trips**:

1. **Fan-out:** N workers in parallel — wall-clock equals the *slowest*
   worker, not the sum (bounded by `worker_timeout`).
2. **Reducer:** 1 strong model (sequential, blocks the response) — fires
   only on arbiter conflict; hedged to up to 2 overlapping calls if the
   primary stalls.

Happy paths collapse to 1 round-trip:

- **Early-exit / unanimous answer:** workers agree → no reducer.
- **Tool-result turn:** skip fan-out entirely → 1 reducer call.

This shape is why reducer streaming is the right TTFT lever: it's the
one serial LLM call on the critical path that produces user-visible
text. Workers are already parallelized; their wall-clock is bounded by
the slowest one regardless of streaming (arbiter needs full outputs to
compare).

---

## Transport

The crate defines a `ModelBackend` trait and ships a default `HttpBackend`
for standalone/test use. The host runtime implements mesh-native backends:

| Backend | Transport | Where |
|---------|-----------|-------|
| `LocalModelBackend` | Direct HTTP to `127.0.0.1:{skippy_port}` | Local model on this node |
| `RemoteModelBackend` | HTTP-over-QUIC tunnel via `node.open_http_tunnel()` | Remote mesh peer |
| `HttpBackend` | Plain HTTP to any URL | Standalone testing |

All worker requests set `mesh_hooks: false` to prevent recursive virtual
LLM consultations (MoA → model → hook → consult another model → ...).

---

## Context packing

Workers get slices of the real agent context, not synthetic prompts. The
agent's actual system prompt, messages, and tool definitions flow through.
The gateway varies depth per role, not content.

| Role | System prompt | Messages | Tools | Max tokens |
|------|:---:|:---:|:---:|:---:|
| Fast | ✅ agent's + one-line preamble | last user msg | names only in system prompt | 256 |
| Specialist | ✅ agent's + one-line preamble | last 4 | name + description | 512 |
| Strong | ✅ agent's + one-line preamble | last 10 | full schemas (native) | 1024 |
| Reducer | ✅ agent's + one-line preamble | worker outputs | full schemas (native) | 2048 |

---

## Arbitration

### Early-exit consensus

Workers are raced in parallel via `gather_workers_incremental()`. After
each response, `try_early_decision()` checks whether enough evidence
exists to return immediately:

| Condition | Action |
|-----------|--------|
| 2+ answers agree, confidence ≥ 0.5 | Return immediately, cancel remaining |
| 2+ tool proposals agree on same tool | Return immediately |
| 1 survivor, all others failed/timed out | Return sole survivor |
| Tool proposal vs answer conflict | Escalate to reducer |
| Low confidence answers | Wait for more workers |

### Deterministic arbiter

The full arbiter (`arbitrate()`) runs when early-exit doesn't fire:

- **Unanimous answers** → highest confidence wins
- **Unanimous tool, same function** → emit tool_call with best arguments
- **Tool vs answer conflict** → escalate to reducer
- **Conflicting tools** → escalate to reducer
- **All uncertain** → escalate to reducer

The reducer step uses a **hedged candidate ladder** (`hedged_reducer_call`)
over the ordered list from `reducer_candidates` — big-tier models first
(multi-digit B, or names with no size), then small-tier as last-resort.

- Start candidate 0 immediately.
- If candidate 0 hasn't replied within `hedge_delay` (5s by default),
  start candidate 1 **alongside** it — don't cancel candidate 0, race them.
- If a candidate errors fast, start the next one immediately (no hedge wait).
- First success wins; cancel the rest.
- All-fail falls back to the best worker output already gathered.

Cost shape: 1 backend call on the happy path (free), up to 2 overlapping
calls when the first is slow, N calls only when everything is failing.
End-to-end wall-clock for the worst case is bounded by
`reducer_timeout + (N-1)·hedge_delay` rather than `N·reducer_timeout`.

---

## Tool calling

Tool results go to reducer only, not re-broadcast to all workers.

```
Turn 1: Client sends "Read README.md" + tools: [read_file]
  → Workers fan out, strong worker proposes read_file({"path":"README.md"})
  → Arbiter: tool consensus → emit tool_call
  → Client gets: tool_calls: [{read_file, {"path":"README.md"}}]

Turn 2: Client sends tool_result with file contents
  → Gateway detects TurnType::ToolResult → reducer only (no fan-out)
  → Reducer synthesizes tool output into final answer
  → Client gets: content: "The README contains..."
```

---

## Normalization

Models produce unreliable output. The normalizer tries in order:

1. **JSON object parse** — structured envelope with kind/confidence/payload
2. **Line-based KV extraction** — `key: value` lines
3. **Heuristic classification** — pattern matching for tool proposals,
   confidence markers, uncertainty signals

Think tags (`<think>...</think>`) and GLM-style reasoning preambles are
stripped throughout the pipeline.

---

## Crate structure

`crates/mesh-mixture-of-agents/` — zero mesh dependency.

| Module | LOC | Tests | Purpose |
|--------|----:|------:|---------|
| `lib.rs` | 454 | 0 | `handle_turn()` orchestration, `GatewayConfig`, `TurnResult`, response builders |
| `backend.rs` | 277 | 6 | `ModelBackend` trait, `HttpBackend`, `SamplingParams`, `call_backend()` |
| `reducer.rs` | 390 | 4 | Reducer candidate ordering + hedged ladder |
| `fanout.rs` | 119 | 0 | Incremental worker gathering with early-exit |
| `arbiter.rs` | 560 | 15 | Deterministic arbitration + early-exit decisions |
| `normalize.rs` | 650 | 13 | 3-tier dirty output parsing |
| `session.rs` | 487 | 4 | Canonical transcript, tool tracking, turn classification |
| `context.rs` | 560 | 8 | Role-shaped context packing |
| `worker.rs` | 298 | 5 | Role assignment, tier sort, think-tag stripping |
| `tool_guard.rs` | 116 | 4 | Allowed-tool enforcement on reducer outputs |

---

## Integration

The MoA intercept lives in `ingress.rs` (~line 234). When `model == "mesh"`:

1. `build_moa_config()` collects all models from `ModelTargets` + `callable`
   list, deduplicates aliases, wraps each in `LocalModelBackend` or
   `RemoteModelBackend`
2. `handle_turn()` runs the stateless MoA pipeline
3. Response is sent as SSE (streaming clients) or plain JSON

Activation: requires ≥2 distinct models available in the mesh. Returns 503
with explanation if fewer.

---

## Relationship to existing systems

| System | What it does | Relationship to MoA |
|--------|-------------|---------------------|
| `auto` | Routes to best single model | MoA fans out to ALL models |
| Hooks (`virtual_llm.rs`) | Reactive during inference (entropy/drift/image) | MoA is proactive before inference |
| Consult (`consult.rs`) | Single peer consultation over QUIC | MoA does parallel multi-peer |
| Pipeline (`pipeline.rs`) | 2-model plan→execute for code tasks | Complementary, used at ingress line 279 |

Hooks and MoA are independent. Hooks fire reactively during inference.
MoA fires proactively before inference. The two are intentionally kept
separate — worker requests set `mesh_hooks: false` so the hook pipeline
never re-enters a worker call.

---

## Test & evaluation plan

### Unit tests (29 tests, run in CI)

```bash
cargo test -p mesh-mixture-of-agents --lib
```

Covers: arbiter voting (10 scenarios), normalizer parsing (12 scenarios),
session turn classification (3), worker role assignment (3).

### Local smoke test

Start mesh with at least one model + discover mesh peers:

```bash
mesh-llm serve --model "unsloth/GLM-4.7-Flash-GGUF:Q4_K_M" --auto
```

Wait for ≥2 models in `curl -s http://localhost:9337/v1/models`.

**Factual accuracy:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"What is the capital of Japan? One word only."}],"max_tokens":32,"stream":false}'
```
Expected: "Tokyo" (possibly with reasoning preamble from GLM).

**Tool calling:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"Read the file README.md"}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"max_tokens":256,"stream":false}'
```
Expected: `tool_calls` with `read_file({"path":"README.md"})`.

**Streaming:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"What is 7*8?"}],"max_tokens":16,"stream":true}'
```
Expected: SSE chunks with `data: {...}` lines, final `data: [DONE]`.

**Reasoning:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? One sentence."}],"max_tokens":128,"stream":false}'
```
Expected: Correct "No" with valid syllogistic reasoning.

### Agent integration test

```bash
# Goose
GOOSE_PROVIDER=mesh GOOSE_MODEL=mesh goose run

# pi
# Set model to mesh in provider config
```

Verify: tool calls work end-to-end, multi-turn context retained by agent
client, no recursive hook loops.

### Performance characteristics

| Scenario | Expected behavior |
|----------|-------------------|
| 2+ workers agree quickly | Early-exit, faster than single-model |
| 1 worker much faster than others | Returns fast worker if confident |
| Remote peer timeout (15s worker / 15s reducer) | Degrades to local-only, adds latency |
| First reducer candidate slow / cold KV | Hedges to second candidate after 5s, races for first OK |
| First reducer candidate broken (502s) | Fast-fails to next candidate immediately, no hedge wait |
| Only 1 model available | Returns 503, does not activate MoA |
| All workers fail | Returns error response |

### What to watch for

- **GLM chain-of-thought leaking** — GLM uses numbered markdown lists for
  reasoning, not `<think>` tags. When GLM is sole survivor, its internal
  deliberation can leak into the response.
- **Sole-survivor wait** — with exactly two workers and one slow/dead,
  the survivor waits up to the worker timeout (15s) before being released by
  the majority-failed early-exit. With 3+ workers this rarely bites.
- **Remote peer 503s** — mesh peers can be flaky. Gateway degrades to
  fewer workers but this limits model diversity.

---
## Diverse-mesh vs split-large: the equal-VRAM trade

> **Status: research plan, not yet run.** No comparison numbers exist.
> This section is the experimental design — including what would falsify
> the hypothesis, so we cannot claim a win after the fact by moving the
> goalposts.

### The real question

Given fixed aggregate (V)RAM across a mesh — say 128 GB on one machine
and 256 GB on another, 384 GB total — there are two ways to spend it:

| Option | Shape | Network on hot path? |
|---|---|---|
| **Split-large** | One big model (e.g. 70B Q4) sharded across both machines via Skippy | **Every token.** Each layer-boundary crossing pays an RTT. |
| **Mix-diverse** | 2–3 mid-size models (e.g. 32B + 32B different families), each fully local to one node, fanned out via `model: "mesh"` | **Once per turn.** Fan-out + collect + reducer. Workers run independently. |

The question this eval answers is **not** "is mixture smarter than one big
model on equal compute?" It is:

> At equal aggregate (V)RAM, on a real mesh with realistic network
> conditions, **where does mix-diverse beat split-large on the
> quality-vs-latency Pareto, and how does that frontier shift as the link
> degrades?**

This is a sharper claim because it predicts when each approach should
win, and it makes the two existing systems (Skippy split, MoA fan-out)
**complementary** rather than competing. The network condition decides
which one is correct.

### Why this is the load-bearing framing

Split-large already wins on quality in the datacenter case. Nobody
should claim 32B + 32B mixture beats a 70B with a fast interconnect.
Pretending otherwise is the kind of overreach that costs the project
credibility.

The mesh case is exactly where split-large hurts most. Every cross-node
hop is unpredictable on real-world links — laptops, home machines,
different ISPs, QUIC over the open internet. A 70B split across two
homes-with-fiber will be slower and more fragile than the same model in
a single rack, and there is no fix for that short of "stop splitting."

Conversely, MoA's apparent "weaknesses" become features in mesh
conditions: per-worker latency variance doesn't matter much when the
alternative is variance from network-on-critical-path. A slow peer
becomes a degraded turn, not a hung turn. A dropped peer becomes fewer
workers, not a failed turn.

### Hypothesis (sharpened)

The mix-diverse path **dominates** split-large on the quality-vs-latency
Pareto frontier when **any of these holds**:

1. **Cross-node link is high-latency** (≥ 20 ms RTT between mesh peers)
2. **Cross-node link is lossy or jittery** (≥ 1% packet loss, or RTT
   stddev comparable to the mean)
3. **Mesh has ≥ 3 nodes** (split's per-token latency grows with
   boundary count; MoA's wall-clock stays bounded by slowest worker)

Conversely, split-large dominates when the link is LAN-grade and the
mesh is exactly two nodes. Both are defensible regions of operation.

### What would falsify the hypothesis

Pre-committed, so we don't move goalposts after seeing data:

- On a degraded link (e.g. +30 ms RTT, 1% loss) at equal aggregate VRAM,
  split-large still dominates mix-diverse on the Pareto front for **every**
  task type → **falsified**, the network-tolerance claim was wrong.
- Mix-diverse with N copies of the **same** model performs the same as
  mix-diverse with N **different** models within grader noise →
  **falsified as "diverse"**; the value is sampling-variance reduction,
  not mixture of perspectives, and we should rename and re-pitch the
  feature.
- Mix-diverse's lead does not grow as mesh size increases from 2 → 3 → 4
  nodes → **the scalability claim is wrong** and split-large's
  multi-node story isn't actually worse.

### Step 1 — measure the variance floor first

Before any A/B claim is meaningful, establish the noise band. Same
single model, same prompt, 10 runs with different sampling seeds.
The standard deviation of grades sets the minimum detectable effect.
If a single 32B model fluctuates by ±0.8 on a 1–5 grade across seeds,
then a 0.3-point lead is in the noise and is not a result.

Most A/B writeups skip this and claim effects smaller than their own
measurement error. We will not.

### Step 2 — the three-way comparison at equal VRAM

All three configurations on the same physical hardware, fixed
aggregate VRAM budget, fixed traffic. This is the headline experiment.

| Config | Example on 128 GB + 256 GB mesh | What it represents |
|---|---|---|
| **Split-large** | One 70B Q4 sharded across both machines (Skippy) | "Spend the VRAM on one bigger mind" |
| **Mix-diverse** | 32B model A on box 1, 32B model B on box 2, optional 8B on either, via `model: "mesh"` | "Spend the VRAM on multiple independent minds" |
| **Single-mid** | `auto` picks the strongest of the mid-size models | Sanity floor — what one of the diverse models gives you alone |

Single-mid is the baseline that catches "MoA = single-best + overhead"
silently. If mix-diverse is not measurably above single-mid, the
fan-out is paying latency for nothing.

### Step 3 — the network sweep is the actual experiment

Quality alone is not the headline. The deliverable is a 2-D scatter:
quality on Y, latency p50 on X, one curve per config per network
condition.

| Network condition | How to simulate | What we expect |
|---|---|---|
| LAN baseline | Two boxes on same switch | Split-large wins outright |
| +20 ms RTT | `tc qdisc add ... netem delay 20ms` on inter-node link | Crossover begins |
| +50 ms RTT | `netem delay 50ms` | Mix-diverse leads on latency, ties on quality |
| 1% packet loss | `netem loss 1%` | Split-large becomes flaky; mix-diverse mostly unaffected |
| 5% packet loss | `netem loss 5%` | Split-large unusable; mix-diverse degrades gracefully |

The interesting point in each plot is the **crossover** — the network
condition at which mix-diverse starts dominating. That's the headline
number, not raw win counts.

### Step 4 — mesh-size scaling

Run the same equal-VRAM comparison at 2, 3, and 4 nodes (with VRAM
budget scaled accordingly).

- **Split-large:** per-token latency should grow roughly linearly with
  boundary count, because every additional split adds one more RTT to
  the hot path.
- **Mix-diverse:** wall-clock stays bounded by slowest worker. Adding
  a fourth node gives one more independent worker for arbitration,
  which can either improve quality or do nothing — but should not
  materially worsen latency.

If split-large's latency does *not* grow with node count, the
"mix-diverse scales better" claim is wrong and should be retracted.

### Step 5 — task-type breakdown (quality measurement)

Within the equal-VRAM, network-sweep frame, the quality axis still
needs to be measured against real task types. Adversarial design from
the earlier draft applies here as the *implementation* of quality, not
the headline:

**Tasks mix-diverse should win on (decomposable correctness)**
- Tool routing with strict format constraints
- Multi-symptom bugs (two independent defects)
- Constraint adherence ("don't touch the imports")

**Tasks both should tie on (single coherent thread)**
- Long-form prose
- Step-by-step procedural answers

**Tasks mix-diverse could lose on (failure-mode amplification)**
- Plausible-but-wrong defaults — if multiple workers make the same
  natural-looking mistake, consensus *amplifies* it
- Single-dominant-fact recall — split-large's bigger weights are more
  likely to know the fact
- Recency-sensitive questions

Without the third group the result is a confirmation exercise, not a
test.

### Step 6 — ablations to find the mechanism

If mix-diverse wins, *why*? Component-by-component:

| Ablation | What it isolates |
|---|---|
| Mix with early-exit disabled (always reducer) | Is the win from consensus or from reducer synthesis? |
| Mix with reducer disabled (return best worker) | Is the reducer paying its latency cost? |
| N=2 vs N=3 vs N=5 | Marginal value of each additional worker |
| **All workers same model** (sampling diversity only) | Mixture vs variance reduction |
| Random reducer choice vs strongest-first | Does reducer selection matter? |

The same-model-N-workers ablation is the most important: if it
performs the same as different-model-N-workers, the feature is
"consensus voting for variance reduction" and should be described that
way, not as "mixture of agents."

### Step 7 — grader robustness

Agent-as-judge has well-documented failure modes. Required defenses:

- **Position swap.** Re-grade with config order reversed; agreement
  must hold above ~95%.
- **Dual grader.** Two different grader models on the same outputs;
  correlate.
- **Manual spot-check.** Hand-grade 10% of pairs; agreement with
  automated graders must be ≥ ~80%.

### Step 8 — real-task replay

Curated scenarios are prone to experimenter intuition. Replay a
sample of real first-turn prompts (from contributors' own goose /
Claude Code / pi sessions, with consent) through all three configs at
each network condition. If mix-diverse wins on curated scenarios but
loses on real-traffic replay, the curated set was the wrong selection.
Real traffic is the acceptance test.

### Reporting discipline

Every result table must include:

- N per cell
- Variance-floor reference for the relevant model size
- Grader version (model + rubric version)
- Mesh composition (exact model list and node placement)
- **Network condition (RTT, loss, jitter)**
- Aggregate VRAM budget (so equal-VRAM claim is auditable)
- Latency percentiles (p50, p95), not just mean
- Early-exit rate from `x-moa-*` headers (mix-diverse only)
- For split-large: cross-node bytes per token (so we can correlate
  with network condition)

A win is a result that exceeds the variance floor by ≥ 1.5×, holds
under position swap, replicates with a second grader, and is visible
as Pareto dominance — not just a higher mean.

### Worker-set knob (deferred, harness-side only)

Several steps above (equal-VRAM mix-diverse, same-model ablation,
composition sweep) require restricting which models MoA fans out to.
Currently `model: "mesh"` uses every callable model. The knob belongs
in the eval harness, not in the crate: either pre-flight filter
`/v1/models` to the allowed set, or send a header
(e.g. `x-moa-workers-include: A,B`) the gateway honors. The crate
already has the machinery to filter the worker pool; no semantic
change required, just a knob.

### Status / next step

No matrix has been run. The first concrete step is **Step 1, the
variance floor**, on whichever mesh we have available. Without it, no
later result can be claimed significant. After variance floor, the
priority is the **Step 2 three-way at LAN baseline** — if split-large
doesn't dominate there, our network-condition framing is wrong and we
need to rethink before running the network sweep.
