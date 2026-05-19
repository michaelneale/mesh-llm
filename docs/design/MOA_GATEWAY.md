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

## Pressure-testing the mixture hypothesis

> **Status: research plan, not yet run.** No comparison numbers exist.
> This section is the experimental design — including what would falsify
> the hypothesis, so we cannot claim a win after the fact by moving the
> goalposts.

### Sharpened hypothesis

MoA's value comes from **arbitration across genuinely different workers**,
and that value is largest for **modest model pools** (single-digit-B
models, or mid-tier 13–32B mixes) on tasks with **decomposable correctness
signals** — places where any single worker is fallible but workers fail in
*different* ways.

Three corollaries that should all hold if the hypothesis is right:

1. MoA's lead over `auto` should **shrink toward zero** as the mesh gains
   a clearly-dominant model.
2. MoA with N copies of the **same** model should be measurably worse
   than MoA with N **different** models of comparable size — otherwise
   we're measuring sampling-variance reduction, not mixture.
3. MoA should be **at or below parity** on tasks where correctness is
   one indivisible thread (long-form prose, recency-sensitive facts), and
   **above parity** on tasks where workers' errors are uncorrelated
   (tool routing, constraint adherence, multi-symptom bugs).

If any of those three fail, the hypothesis as stated is wrong and needs
to be reformulated before we keep recommending the feature.

### What would falsify the hypothesis

Pre-committed, so we don't move goalposts after seeing data:

- On a mixed two-model mesh of comparable-size workers (e.g. 8B + 8B
  different families), MoA does not dominate `auto` on the Pareto front
  for **any** latency budget across the adversarial scenario set →
  **falsified for modest-model meshes**, the core claim.
- Same-model-N-workers ≈ different-model-N-workers within grader noise
  → **falsified as "mixture"**; the value, if any, is variance reduction
  and should be described that way.
- MoA's lead does *not* shrink when a 30B+ model is added to the pool
  → **the "modest models" framing is wrong** and we need to retract it.

### Step 1 — measure the variance floor first

Before any A/B claim is meaningful, establish the noise band. Same
single model, same prompt, 10 runs with different sampling seeds, on
each grader rubric. The standard deviation of grades sets the minimum
detectable effect. If a single 8B model fluctuates by ±0.8 on a 1-5
grade across seeds, then a 0.3-point MoA "win" is in the noise and
should not be reported as a win.

Most A/B writeups skip this step and end up claiming effects smaller
than their own measurement error. We will not.

### Step 2 — adversarial scenario design

The scenario set must include cases hostile to the hypothesis, not just
favorable ones. At minimum:

**Tasks MoA should win on (decomposable correctness)**
- Tool routing with strict format constraints (exact-N bullets, exact
  schema, exact file location)
- Multi-symptom bugs in small code (two independent defects; workers
  often each catch one)
- Constraint adherence ("don't touch the imports") where each worker is
  ~70% reliable per constraint and consensus filters violators

**Tasks MoA should tie on (single coherent thread)**
- Long-form prose ("write a 400-word argument for X") — reducer has to
  pick one whole answer, so this collapses to "best single worker
  survives" with extra latency
- Step-by-step procedural answers with no decomposable check

**Tasks MoA could lose on (failure-mode amplification)**
- Plausible-but-wrong defaults — code that's *almost* right with one
  wrong-but-natural-looking fix (e.g. wrong regex, off-by-one). If 3 of
  4 workers all make the same plausible-but-wrong fix, consensus
  *amplifies* the error. This is the reducer-only setup's failure mode
  that proponents never advertise.
- Single dominant fact — "capital of Bhutan" where one worker knows and
  three don't. Arbiter has no signal to prefer the knower.
- Recency-sensitive questions — workers with different knowledge cutoffs
  give different answers, none of which the arbiter can verify.

Without the third group, we're not stress-testing — we're confirming.

### Step 3 — Pareto curve, not win/tie/loss

The headline deliverable is not "MoA wins 7-2-1." It is a quality vs
median-latency scatter, with every config a point:

- single fast model (smallest, baseline floor)
- single medium model (`auto` on a small-only mesh)
- single strong model (`auto` on a mesh that includes one)
- MoA N=2 (smallest budget)
- MoA N=3
- MoA N=4
- MoA with hedge_delay=3s vs 5s vs 10s

A point dominates another if it is both higher quality *and* lower
latency. The interesting questions are:

- Does any MoA point dominate any `auto` point?
- If MoA wins only at the top-right (more latency for more quality),
  what is the marginal cost per quality point, and is it worth it?
- Where on the curve does early-exit fire most often, and how much
  latency does that save?

### Step 4 — ablations to find the mechanism

If MoA wins, *why*? Component-by-component:

| Ablation | What it isolates |
|---|---|
| MoA with early-exit disabled (always reducer) | Is the win from consensus check, or from reducer synthesis? |
| MoA with reducer disabled (return best worker on conflict) | Is the reducer doing real work, or just adding latency? |
| N=2 vs N=3 vs N=5 | Marginal value of each additional worker |
| **All workers same model** (sampling diversity only) | Mixture-of-perspectives vs variance reduction |
| Random reducer choice vs strongest-first | Does reducer selection matter, or any decent model? |
| `mesh_hooks: true` (currently false for workers) | Does worker-side hook escalation add or subtract? |

Same-model-N-workers is the most important one: if it's close to
different-model-N-workers, we should re-label the feature as "consensus
voting for variance reduction," not "mixture of agents."

### Step 5 — composition sweep

The "modest models" framing is the load-bearing claim. Test it directly:

| Mesh shape | Prediction if hypothesis holds |
|---|---|
| 2× 4B (same family) | MoA win small; same-model ablation should tie |
| 2× 8B (different families) | MoA win largest here |
| 8B + 13B | MoA win persists, shrinks slightly |
| 8B + 32B | MoA win shrinks substantially; `auto` picks 32B |
| 8B + 32B + 70B | MoA win near zero or negative; `auto` dominates |

If the curve doesn't bend that way — if MoA's lead is flat across
compositions, or grows as you add bigger models — the hypothesis is
wrong as stated.

### Step 6 — grader robustness

Agent-as-judge has well-documented failure modes: verbosity bias,
position bias, same-family preference. Defenses, all required:

- **Position swap.** Re-grade each pair (`mesh` first vs `auto` first)
  and compute agreement. Disagreement rate above ~5% means grader is
  measuring position more than quality.
- **Dual grader.** Run the full grade pass with two different grader
  models and check correlation. Low correlation means the result is
  grader idiosyncrasy, not agent quality.
- **Manual spot-check.** Hand-grade 10% of pairs and compute agreement
  with each automated grader. Below ~80% agreement, the grader is
  unreliable and the eval result is suspect.

### Step 7 — real-task replay

Curated scenarios are prone to experimenter intuition. The strongest
defense against cherry-picking is replaying real traffic:

- Collect a sample of first-turn prompts from real goose / Claude Code /
  pi sessions (with consent, from contributors' own logs)
- Run each through both `mesh` and `auto` with everything else fixed
- Grade with the same rubric set

If MoA wins on curated scenarios but loses on real-traffic replay, the
curated scenarios were the wrong selection. Real traffic is the
acceptance test.

### Reporting discipline

Every result table must include:

- N per cell
- Variance-floor reference for the relevant model size
- Grader version (model + rubric version)
- Mesh composition (exact model list)
- Latency percentiles (p50, p95), not just mean
- Early-exit rate from `x-moa-*` headers

A win is a result that exceeds the variance floor by at least 1.5×,
holds under position swap, replicates with a second grader, and is
visible on the Pareto curve — not just in a win/tie/loss count.

### Worker-set knob (deferred, harness-side only)

The composition sweep and same-model ablation both need a way to
restrict which models MoA fans out to. The current `model: "mesh"`
uses every callable model.

When we need this, it belongs in the eval harness, not the crate:
either pre-flight filter `/v1/models` to the allowed set before
launching, or have the harness send a header (e.g.
`x-moa-workers-include: A,B`) that the gateway honors. The crate
already has the machinery to filter the worker pool; no semantic
change required, just a knob.

### Status / next step

No matrix has been run. The first concrete step is **Step 1, the
variance floor**, against whichever mesh we have available. That
single measurement gates everything else — without it, no later
result can be claimed as significant.
