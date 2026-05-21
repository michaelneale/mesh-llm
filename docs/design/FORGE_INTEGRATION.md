# Forge Integration

Plan for porting [forge](https://github.com/antoinezambelli/forge)'s
small-model tool-calling reliability rules — and its small-context
compaction — into mesh-llm.

**Status:** proposal — not implemented
**Related:** [MOA_GATEWAY.md](MOA_GATEWAY.md), [VIRTUAL_LLM.md](VIRTUAL_LLM.md)

---

## Why

Mesh-llm routes OpenAI requests to a heterogeneous set of peers. Two
failure modes that forge is built to solve show up at the small end
of that pool:

1. **Tool-calling reliability on small models.** 8B-class models —
   the `Fast` worker tier in MoA, and anything an agent might
   directly request — fail in characteristic ways even when the
   backend exposes native function calling (the structured
   `tool_calls` channel in the OpenAI API): malformed JSON, tool
   name leaking into `content`, args returned as a stringified
   blob, unknown-tool hallucinations, or the call emitted as free
   text instead of through the structured channel at all. Native FC
   is the *happy path*; forge exists for when small models fall off
   it. The rescue parser handles four known fall-off shapes (see
   the port table) including Qwen3-Coder's XML form, which mesh
   already hosts.

2. **Context overflow on small-context hosts.** A multi-turn session
   on a host with a small `n_ctx` (4K–8K) accumulates tool results
   and reasoning until it 400s. Mesh's Smart Router TODO covers
   *routing around* small-context hosts; forge solves the dual
   problem by *compacting in place* so the host can keep serving.

Forge's published claim covers both — README opening: "guardrails
(rescue parsing, retry nudges, step enforcement) **and** context
management (VRAM-aware budgets, tiered compaction)." On 26 multi-step
scenarios an 8B Ministral with both halves on hits 86.5% overall,
76% on the hard tier.

We port both, at the right layer of mesh-llm, each gated on the
condition that actually warrants it.

## Where it belongs: the model-hosting layer

The instinct to put any of this at the OpenAI frontend, or at the
MoA worker, is wrong. Both are too high. A small model needs the
same guardrails whether it's serving a direct `/v1/chat/completions`
from goose, a worker call inside MoA fanout, a Virtual LLM consult
from a bigger model on another peer, or a `/v1/responses`
translation. A small-context host needs the same compaction
regardless of which of those is calling it.

Putting either at *each* of those sites means N copies of the same
logic, gated by N different predicates, that will drift.

Mesh-llm already has the right seam: the host wraps a single model
and exposes it as an `OpenAiBackend`. Every consumer above calls
that trait. We wrap the **host's** backend impl in **two
independent decorators**, each gated on a different condition that
actually warrants engaging it:

```
   /v1 ingress     MoA worker call     virtual-LLM consult
        │                │                    │
        └────────────────┴────────────────────┘
                         │
                         ▼
              Arc<dyn OpenAiBackend>           ← decorate at host construction
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  GuardrailBackend                  (large model + large ctx:
  (engaged for small-tier             passthrough — neither
   models, when tools                  decorator constructed)
   are present)
        │
        ▼
  CompactingBackend
  (engaged for small-context
   hosts, when an inbound request
   is near the host's n_ctx)
        │
        ▼
  inner backend → llama-server / local inference
```

The two decorators are **orthogonal** and gated on **different
predicates**:

| Decorator | What | Predicate |
|---|---|---|
| `GuardrailBackend` | Rescue / validate / retry / `respond` injection | small-tier *model* (single-digit-B name) AND request has tools |
| `CompactingBackend` | Tiered in-place compaction + context warnings | host's *n_ctx is small* (≤ configured threshold) AND request is approaching it |

A small model on a 128K-ctx host gets the guardrail decorator only.
A large model on an 8K-ctx host gets the compacting decorator only.
The common 8B-on-8K case gets both. Large model on large host: no
wraps, no overhead.

Both are constructed conditionally at host setup, so the
passthrough case really is free — there's no per-request branch in
production, just an unwrapped `Arc<dyn OpenAiBackend>`.

## What gets ported

One new crate, `crates/mesh-llm-guardrails/`. Pure logic in the
lower modules; the two `OpenAiBackend` impls layered on top.

| forge (Python) | mesh-llm-guardrails (Rust) | Job |
|---|---|---|
| `guardrails/response_validator.py` | `validator.rs` | Validate `LlmResponse`, classify as `Execute` / `Retry` / `UnknownTool`. |
| `prompts/templates.py::rescue_tool_call` | `rescue.rs` | When native function-calling fails (the model emits a tool call as free text instead of through the structured `tool_calls` channel), parse it out anyway. Four strategies, tried in order: (1) JSON `{"tool", "args"}` with code-fence / embedded-JSON tolerance, (2) `tool_name[ARGS]{...}` rehearsal syntax from reasoning models, (3) Qwen3-Coder `<function=…><parameter=…>…</parameter></function>` XML, (4) Granite-4.0 `<tool_call>{"name", "arguments"}</tool_call>`. Port all four — JSON-only would silently miss Qwen models, which mesh already hosts. Think tags (`<think>…</think>`, `[THINK]…[/THINK]`) get stripped first. |
| `guardrails/step_enforcer.py` | `step_enforcer.rs` | Required-steps tracker + tier-escalating nudges. |
| `guardrails/error_tracker.py` | `error_tracker.rs` | Retry / tool-error budgets. |
| `prompts/nudges.py` | `nudges.rs` | String templates for each nudge kind. |
| `tools/respond.py` | `respond.rs` | Synthetic `respond(message)` tool spec + strip helper. |
| `guardrails/guardrails.py` | `lib.rs` (`Guardrails` facade) | Mirrors forge's `examples/foreign_loop.py` simple API. |
| `context/strategies.py::TieredCompact` | `compact.rs` | Three-phase tiered compaction (drop nudges + truncate results → drop results → drop reasoning). |
| `context/manager.py` (`check_thresholds`, `default_context_warning`) | `compact.rs` (warning helpers) | Threshold-driven "you're running out of room" message injection. |

Tests come along the same way — port `forge/tests/unit/test_*` to
`crates/mesh-llm-guardrails/tests/`. The biggest two are
`test_argument_transformation.py` (rescue parsing of stringified
args, fence stripping, embedded JSON in prose) and the
strategy/threshold tests from `test_context_manager.py` /
`test_strategies.py` / `test_context_thresholds.py`.

The facade shape, matching forge:

```rust
pub enum GuardrailAction {
    Execute   { tool_calls: Vec<ToolCall> },
    Retry     { nudge: Nudge },
    StepBlocked { nudge: Nudge },
    Fatal     { reason: String },
}

impl Guardrails {
    pub fn new(cfg: GuardrailConfig) -> Self;
    pub fn check(&mut self, resp: LlmResponse) -> GuardrailAction;
    pub fn record(&mut self, executed: &[String]) -> bool;
}
```

## The two decorators

Both live in `crates/mesh-llm-guardrails/src/backend.rs`. They share
the `OpenAiBackend` decorator shape; they differ in trigger and in
what they mutate.

### `GuardrailBackend` — tool-calling reliability

```rust
pub struct GuardrailBackend {
    inner: Arc<dyn OpenAiBackend>,
    config: GuardrailConfig,
}

#[async_trait]
impl OpenAiBackend for GuardrailBackend {
    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        if !self.should_guard(&request) {
            return self.inner.chat_completion(request).await;
        }

        let mut state = Guardrails::new(&self.config);
        state.inject_respond_tool(&mut request);

        for _ in 0..self.config.max_retries {
            let response = self.inner.chat_completion(request.clone()).await?;
            match state.check(&response) {
                GuardrailAction::Execute => {
                    return Ok(state.finalize(response)); // strips respond()
                }
                GuardrailAction::Retry { nudge }
                | GuardrailAction::StepBlocked { nudge } => {
                    request.messages.push(nudge.into_message());
                    continue;
                }
                GuardrailAction::Fatal { .. } => {
                    return Ok(state.passthrough_text(response));
                }
            }
        }
        Ok(state.passthrough_last())
    }

    async fn chat_completion_stream(&self, req, ctx) -> ... {
        // Buffered-tool-only: collect inner stream into a response,
        // run the loop above, emit final SSE chunks at the end.
        // Bare-text turns short-circuit via should_guard and stream
        // unchanged.
    }

    // Pure delegation:
    async fn models(&self) -> ...                  { self.inner.models().await }
    async fn completion(&self, r) -> ...           { self.inner.completion(r).await }
    async fn completion_stream(&self, r, c) -> ... { self.inner.completion_stream(r, c).await }
}
```

Two things to call out:

- **Retries re-enter the inner backend.** Each retry passes through
  any sibling decorators below it (`CompactingBackend`,
  `HookedOpenAiBackend`, Virtual LLM hooks). Correct semantics: a
  retry *is* a fresh inference.
- **`Fatal` returns text, not an error.** Mirrors forge's
  `proxy/handler.py` behavior — when retries are exhausted, surface
  the last text so the upstream loop can react. Don't 5xx the agent.

`should_guard` returns true when both hold:

1. `request.tools` is non-empty (nothing to validate otherwise).
2. The host's model is in the small tier — reuse the single-digit-B
   name heuristic that already exists in
   `mesh-mixture-of-agents/src/worker.rs`. Centralize it into a
   shared helper (`mesh-llm-routing` or `mesh-llm-types`) so the
   two call sites can't drift.

Per-request override knob: `request.extra["mesh_guardrails"]`,
matching the existing `MESH_HOOKS_FIELD` pattern in
`openai-frontend/src/hooks.rs`. Tests and explicit clients can
force on/off.

### `CompactingBackend` — small-context survival

Engages only when the *host's* `n_ctx` is small (configurable, default
≤ 8192 tokens) and an inbound request is approaching that budget. A
small model on a large-context host gets nothing; a large model on a
tiny-context host gets compaction without guardrails.

```rust
pub struct CompactingBackend {
    inner: Arc<dyn OpenAiBackend>,
    n_ctx: u32,                  // the host's actual ctx limit
    config: CompactConfig,       // thresholds, keep_recent, strategy
}

#[async_trait]
impl OpenAiBackend for CompactingBackend {
    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let tokens = estimate_request_tokens(&request);
        if !self.should_compact(tokens) {
            return self.inner.chat_completion(request).await;
        }

        // Tiered compaction over the message list — phase 1 drops
        // nudges and truncates tool results, phase 2 drops tool
        // results entirely, phase 3 drops reasoning/text. Always
        // preserves the system prompt and the original user turn.
        let phase = TieredCompact::new(&self.config).compact(
            &mut request.messages,
            self.n_ctx,
        );

        // Inject a single-shot context warning so the model knows it's
        // running out of room and should summarize critical findings.
        if let Some(warn) = self.config.threshold_warning(tokens, self.n_ctx) {
            inject_text_into_chat_messages(&mut request.messages, warn);
        }

        self.inner.chat_completion(request).await
    }

    async fn chat_completion_stream(&self, req, ctx) -> ... {
        // Same logic — compaction is on the inbound request, not the
        // outbound stream, so streaming is untouched.
    }

    // Pure delegation for the rest.
}
```

Key differences from forge's original:

- **Stateless per request.** Forge's `ContextManager` carries
  `_last_known_tokens`, `_fired_thresholds`, and per-session state
  across turns. Mesh-llm hosts are stateless turn-to-turn — sessions
  are reconstructed from the full message list each request — so
  the decorator estimates from the request itself each time. This is
  simpler and matches mesh's existing model.
- **Token estimate is char/4.** Same heuristic forge uses when it
  hasn't yet seen a usage report. Good enough for threshold
  decisions; not used for routing.
- **No `step_index` metadata on messages.** Forge tracks iteration
  boundaries via `MessageMeta.step_index` to keep parallel-tool
  batches together during compaction. Mesh-llm's `ChatMessage`
  doesn't have that. Approximation: treat each
  `assistant{tool_calls}` + the following `tool` messages as one
  iteration boundary. Port `_find_eligible_end` against that
  approximation, with a unit test confirming behavior matches
  forge's on equivalent inputs.

`should_compact(tokens)` returns true when:

1. `self.n_ctx <= small_ctx_threshold` (decorator was constructed at
   all — see Wiring), AND
2. `tokens >= n_ctx * compact_threshold` (default `compact_threshold
   = 0.75`).

Per-request override: `request.extra["mesh_compact"]`.

## Wiring

At the point in the host runtime that constructs the backend for a
loaded model, conditionally wrap. Each decorator is opt-in based on
the *host's* characteristics — model size for guardrails, `n_ctx`
for compaction — so the wraps are decided once at construction, not
per request.

```rust
let backend: Arc<dyn OpenAiBackend> = build_llama_backend(model_handle);

// Inner-most: compaction (so guardrail retries see compacted state)
let backend = if model_handle.n_ctx <= compact_cfg.small_ctx_threshold {
    Arc::new(CompactingBackend::new(backend, model_handle.n_ctx, compact_cfg))
} else {
    backend
};

// Outer: guardrails
let backend = if is_small_tier(&model_handle.descriptor) {
    Arc::new(GuardrailBackend::new(backend, guardrail_cfg))
} else {
    backend
};

register_hosted_backend(model_handle.id, backend);
```

Full decorator stack, when everything applies:

```
GuardrailBackend             ← outermost
  → CompactingBackend        ← compaction sees guardrail retries
    → HookedOpenAiBackend    ← virtual-LLM / prompt-injection hooks
      → real backend
```

Why this order:

- **Guardrails outside compaction.** Each guardrail retry triggers
  the full inner pipeline. If compaction is below guardrails, every
  retry re-checks the (now slightly longer, because of nudges)
  message list against the budget. A retry should look like a fresh
  request to the compactor.
- **Compaction outside hooks.** The virtual-LLM hook may inject
  text (e.g. an image caption) into the request before inference.
  Compaction should see the post-hook request shape so it doesn't
  trim something the hook just added. Hooks fire on the
  already-compacted message list.

Config block:

```toml
[guardrails]
enabled = true
mode = "small_models_only"   # "small_models_only" | "all" | "off"
max_retries = 3
rescue_enabled = true
inject_respond_tool = true
small_param_threshold_b = 12

[guardrails.compact]
enabled = true
small_ctx_threshold = 8192   # only wrap hosts with n_ctx at or below this
compact_threshold = 0.75     # fraction of n_ctx at which compaction fires
keep_recent = 2              # most-recent iterations preserved fully
inject_threshold_warning = true
```

## What this gives you, for free

Because the wraps are at the host layer:

- **Direct `/v1/chat/completions`** — guarded when routed to a
  small-tier host, compacted when routed to a small-ctx host,
  passthrough on a large model + large ctx host. No frontend change.
- **`/v1/responses`** — `router.rs::responses` calls
  `state.backend.chat_completion(...)` and translates the result.
  Both decorators fire before the translation. Free.
- **MoA worker calls** — each worker addresses a hosted backend via
  the same trait. The `Fast` worker is by definition small-tier and
  is guarded; if that worker also runs on a small-ctx host, it's
  compacted too. The `Strong` worker on a large host gets neither.
  No change in `worker.rs`.
- **Virtual LLM consults** — when a large model consults a small
  peer for image captioning or similar, the small peer's host
  applies whichever decorators it has on, so the consulting model
  sees clean tool calls and a request that fits.

## Validation

One validation gate per decorator, both required before defaulting
either one on.

### Guardrail validation

**Pick three tool-calling scenarios from `evals/scenarios/` that
fail or are flaky today on an 8B-class host** (e.g. `find-and-fix/`,
`debug-session/`, `refactor/`). Run each 20 times against the same
mesh, both:

- `guardrails.mode = "off"` — baseline
- `guardrails.mode = "small_models_only"` — wrap engaged

Measure per scenario: pass rate (n/20), mean tokens per turn, mean
wall-clock, rescue rate (how often `rescue_tool_call` salvaged a turn
that would have been a retry).

**Pass criterion:** at least one scenario shows ≥20 percentage-point
pass-rate lift with token cost <2× baseline.

### Compaction validation

**Pick one multi-turn coding scenario long enough to push past 8K
tokens** (an extended `debug-session/` or `refactor/` run). Host on a
deliberately small-ctx model (`-c 8192` on llama-server). Run 10
times in each configuration:

- `guardrails.compact.enabled = false` — baseline (expect 400s or
  truncations once the budget overflows)
- `guardrails.compact.enabled = true` — wrap engaged

Measure: turn-completion rate (how many turns succeed before the
session dies), final-state quality (does the model still know what
it was doing after compaction?), and a sanity check that compaction
phases trigger in the expected order (phase 1 before 2 before 3).

**Pass criterion:** baseline dies before turn N, with-compaction
survives to ≥2× turn N with the model still tracking the task. If
the model "forgets what it's doing" after phase 3 — which is the
known weakness of dropping reasoning — we say so, ship phase 1+2
only by default, and gate phase 3 behind explicit opt-in.

### Why hard gates instead of phased rollout

The whole point of forge's claim is reproducible lift on small
models *and* graceful behavior on small contexts. We reproduce both
on *our* models before defaulting either on. Either it works on the
chosen scenarios or it doesn't. Phased rollout buys nothing here —
the port is small enough that landing it and validating it are the
same week of work.

### Optional: pre-port sidecar smoke

Before writing any Rust, you can run forge's published Python
package as a sidecar in front of an unmodified mesh-llm and get a
same-afternoon read on whether the port is worth committing to:

```bash
pip install forge-guardrails
forge-proxy --backend-url http://localhost:9337 --port 8081 &
mesh-llm goose --host 127.0.0.1:8081 --model "<some 8B model>"
```

Run the same three scenarios as the guardrail gate above through
this stack. If they don't lift here, they won't lift after a port
either — stop before writing code. If they do, you have evidence
the port will pay off, *and* a behavioral reference implementation
to compare the Rust port against during development.

This is not a phase, not a gate, and not a deliverable — it's a
cheap signal available because forge ships a working product.

## Open issues

1. **Streaming for guardrails.** First cut is buffered-tool-only:
   `chat_completion_stream` collects the inner stream into a
   response, runs the loop, emits the final SSE chunks. Bare-text
   streaming is untouched via the `should_guard` short-circuit.
   Agents (goose, opencode, claude code) don't act on partial
   tool-call deltas, so this is observationally equivalent for them.
   A fully-streaming version with late SSE rewrite is doable later
   and not needed first.

2. **Streaming for compaction is fine.** Compaction mutates the
   *inbound* request, not the outbound stream. Streaming flows
   through untouched.

3. **`StepEnforcer` has nothing to enforce against on raw agent
   traffic.** Forge's step enforcer requires a declared list of
   `required_steps` and a `terminal_tool`. Agents calling mesh-llm
   directly don't declare those. The decorator should expose
   `StepEnforcer` in the config but default it off for ingress
   traffic. It becomes useful if/when MoA grows a notion of required
   tools per role, at which point the MoA-side construction can pass
   a non-empty step config for that worker.

4. **Compaction without `step_index` metadata.** Forge tracks
   iteration boundaries via `MessageMeta.step_index` to keep
   parallel-tool batches together during compaction. Mesh-llm's
   `ChatMessage` doesn't carry that. The port approximates: each
   `assistant{tool_calls}` plus the following `tool` messages is one
   iteration. Validate the approximation with a unit test that
   compares behavior on equivalent inputs to forge's Python
   implementation.

5. **Compaction is stateless per request.** Mesh-llm hosts are
   stateless turn-to-turn — sessions are reconstructed from the full
   message list each call. The decorator estimates tokens from the
   request itself rather than carrying a `_last_known_tokens` across
   turns. Simpler, matches mesh's model, and the threshold logic
   still works because the request *is* the history.

6. **Forge nudge templates are tuned on Ministral/Qwen3/Mistral.**
   Plain English, should transfer, but keep the strings configurable
   so we can A/B them on whatever small models we actually host.

## Appendix: forge file inventory

```
forge/src/forge/
  guardrails/
    response_validator.py   → validator.rs
    step_enforcer.py        → step_enforcer.rs
    error_tracker.py        → error_tracker.rs
    guardrails.py           → lib.rs (Guardrails facade)
    nudge.py                → types.rs (Nudge struct)
  prompts/
    templates.py            → rescue.rs (rescue_tool_call only)
    nudges.py               → nudges.rs
  tools/
    respond.py              → respond.rs
  context/
    strategies.py           → compact.rs (TieredCompact + phases)
    manager.py              → compact.rs (threshold + warning helpers)
  core/
    steps.py                → step_enforcer.rs (StepTracker is internal)
    workflow.py             → types.rs (ToolCall, TextResponse)
    messages.py             → types.rs (MessageType enum, minus step_index)

forge/tests/unit/
  test_response_validator.py
  test_step_enforcer.py
  test_error_tracker.py
  test_respond_tool.py
  test_argument_transformation.py
  test_nudges.py
  test_strategies.py            ← compaction phases
  test_context_manager.py       ← thresholds + estimation
  test_context_thresholds.py    ← warning injection
```

Deliberately *not* ported (mesh-llm handles these at a different
layer, or they don't apply):

- `clients/*` — mesh has its own backends
- `clients/sampling_defaults.py` — *worth a separate look later.*
  Forge 0.6.0 ships a curated per-model sampling-parameter map
  (`temperature`, `top_p`, `top_k`, `min_p`, etc.) sourced from each
  model's HuggingFace card and verified one entry at a time. It
  covers Qwen3/3.5/3.6, Qwen3-Coder, Gemma 4, Mistral Small 3.2,
  Devstral Small 2, Ministral 3, Mistral Nemo, Granite 4.0. This is
  exactly the kind of curated metadata mesh-llm's `Smart Router`
  TODO will want — but it's a routing-layer concern, not a
  guardrails concern, so it's out of scope here. Flag for whoever
  picks up that work.
- `proxy/*` — replaced by the `OpenAiBackend` decorators above
- `server.py` — mesh has its own backend lifecycle / VRAM mgmt
- `context/hardware.py` — mesh's model-resolver already detects this
- `core/runner.py`, `core/inference.py` — full agent loop; mesh
  doesn't own the loop, the agents (goose, opencode) do
- `core/slot_worker.py` — mesh routes between hosts, not slots
  within one process

### Source

- Forge repo: <https://github.com/antoinezambelli/forge>
- Pin the port to forge **v0.6.0** (released 2026-04-29) for
  reproducibility of the validation gates and clean provenance.
  Later versions can be cherry-picked deliberately, not silently.
- **Cherry-pick PR #72 on top of v0.6.0** before porting. It fixes a
  bug in `Guardrails.record()` where the facade drops tool `args`
  when calling through to `StepEnforcer.record()`, breaking
  argument-based prerequisite checks (they always see missing data
  and produce false negatives). Unmerged upstream at time of
  writing; port the *fixed* shape, not the v0.6.0-tagged shape.
  It's the only known correctness bug on the surfaces we port.
- Paper: Zambelli, A. *Forge: A Reliability Layer for Self-Hosted
  LLM Tool-Calling.* <https://doi.org/10.1145/3786335.3813193>
  — published method, peer reviewed; cite when the port lands.
