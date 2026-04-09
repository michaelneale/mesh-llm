# Virtual LLM Engine

Design for in-engine mesh hooks — making llama-server itself aware of the mesh so it can consult other models **during** inference, not just before or after.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## Why in-engine, not proxy-level

The proxy layer (`network/proxy.rs`) can rewrite requests before inference and inspect responses after. But it can't participate **during** token generation. Any proxy-level pipeline that needs to react to what the model is generating must buffer the response, cancel inference, reconstruct a new request, re-send, and stitch the output back together. This is expensive (re-prefill, multiple HTTP round-trips) and fragile (stream reconstruction, response format normalization).

By adding hooks inside llama-server's C++ token generation loop (`server-context.cpp`), the virtual LLM can inject context into the live KV cache, pause a slot while consulting the mesh, and resume generation — all without restarting inference or breaking the streaming connection.

| Capability | Proxy Level | In-Engine |
|---|---|---|
| Image captioning for text models | Rewrite request before inference (pre-flight) or cancel + re-send | Async fetch during prefill, inject caption into live KV cache. Zero restart. |
| Response verification | Buffer full response, send to verifier, potentially regenerate | Check at EOS, inject correction, continue generating in same slot. |
| Confidence-triggered help | Buffer tokens, score, cancel stream, re-route | Per-token logprob check, pause slot, consult mesh, inject, resume. Stream uninterrupted. |
| Context enrichment mid-generation | Cannot do during streaming | Inject tokens into live KV cache. Model's attention updates in-place. |
| Dynamic knowledge retrieval | Cannot do during streaming | Model generates uncertainty signal → hook detects → fetches from mesh → injects → model continues with the knowledge. |
| Seamless streaming to client | Must buffer, cancel, stitch SSE frames | Native — same slot, same stream, brief pause at most. |

---

## Architecture: Hooks and Signals

The C++ side is deliberately simple: it **measures signals** and **calls back to mesh-llm**. All decision-making (which model to consult, what context to fetch, whether to escalate) lives in the Rust mesh-llm code. The hooks are generic callbacks with signal data attached.

### The callback

Every hook invocation is the same shape — an HTTP POST to mesh-llm:

```
POST http://localhost:{mesh_port}/mesh/hook
{
  "hook": "per_token" | "pre_inference" | "post_prefill" | "pre_response" | "complete",
  "slot_id": 0,
  "request_id": "chatcmpl-abc123",

  // signal data (varies by hook, see below)
  "signals": { ... },

  // context for mesh-llm to make decisions
  "generated_text": "The function in auth.rs...",
  "n_decoded": 42,
  "model": "qwen3-32b",
  "original_request": { ... }  // or a reference/hash to avoid re-sending
}
```

Response from mesh-llm:

```json
{
  "action": "none" | "inject" | "continue" | "stop",
  "inject_text": "The functions in auth.rs are: verify_token(), ...",
  "inject_role": "system",
  "stop_reason": "redirect",
  "metadata": { ... }
}
```

The hook tokenizes `inject_text`, adds it to the KV cache, and resumes. That's it. The C++ side doesn't know or care *why* mesh-llm decided to inject — it just does the mechanical token injection.

### Callback types

| Type | When | Blocking? | Notes |
|---|---|---|---|
| **Async fire** | Hook 1 (pre-inference) | No — fire and continue | Start consultation, check result later at Hook 3 |
| **Sync call** | Hook 1, 2, 4 | Yes — slot waits | Slot pauses (`SLOT_STATE_WAITING_MESH`), other slots continue |
| **Poll check** | Hook 3 (per-token) | No — check `std::future`, return immediately if not ready | Zero cost when nothing pending |
| **Notify** | Hook 5 (complete) | No — fire and forget | Telemetry, quality data, routing feedback |

---

## Uncertainty signals

The per-token hook (Hook 3) computes signals from the logit distribution and passes them to mesh-llm. The C++ side computes the raw numbers. mesh-llm decides what they mean.

### Signal A: Entropy

Computed from the full post-softmax distribution at each token:

```
H = -Σ p_i * log(p_i)
```

- **Low entropy** → peaked distribution → model is confident in its choice
- **High entropy** → spread out → model is uncertain, many plausible continuations

Available via `get_token_probabilities(ctx, idx)` which already computes the sorted softmax distribution. Entropy is a cheap sum over that array.

Reported per-token, plus a rolling window average (e.g., last 8 tokens). A sudden entropy spike mid-generation is a strong signal.

### Signal B: Top-token margin

The gap between the top two candidates:

```
margin = p_top1 - p_top2
```

- **Large margin** → strong decision, model clearly prefers one token
- **Small margin** → ambiguous, could have gone either way

Already available — `get_token_probabilities` returns sorted by probability. Just `cur[0].p - cur[1].p`.

Cheaper than entropy (two values, not a sum over vocab). Good for fast per-token checks. When margin is small AND entropy is high, the model is genuinely lost.

### Signal C: Sequence variance (confidence trajectory)

Track entropy and margin over the generation window. The pattern matters:

- **Confident → uncertain → confident** = possible hallucination zone in the middle
- **Confident → increasingly uncertain** = model running out of knowledge
- **Uniformly uncertain** = task is too hard for this model

The C++ side maintains a rolling buffer of per-token entropy/margin values. On each Hook 3 call, it sends the recent window to mesh-llm. mesh-llm can detect these patterns and decide when to intervene.

```cpp
struct signal_window {
    std::deque<float> entropy;    // last N tokens
    std::deque<float> margin;     // last N tokens
    float entropy_mean;           // rolling mean
    float entropy_max;            // max in window (spike detection)
    float margin_min;             // min in window (worst ambiguity)
    int   uncertain_streak;       // consecutive tokens above threshold
};
```

### Signal D: Self-consistency (multi-completion)

The strongest signal, but the most expensive. Sample multiple completions from the same point, measure agreement.

**How it works in llama-server**: Use `n_cmpl > 1` (the existing multi-completion feature). The engine does one prefill, copies KV cache to N child slots via `copy_state_to()`, generates N independent completions. Compare them.

- **Outputs converge** → high confidence, probably correct
- **Outputs diverge** → low confidence, model is guessing

This isn't a per-token signal — it's a per-request strategy. mesh-llm decides at request time whether a request warrants self-consistency checking (e.g., complex reasoning, factual claims). If so, it sets `n_cmpl: 3` and at Hook 4 (pre-response) compares the N completions, picks the best, or detects disagreement and consults the mesh.

The C++ side doesn't need special support — `n_cmpl` already exists. mesh-llm just needs to:
1. Set `n_cmpl: N` and `stream: false` in the request params
2. At the pre-response hook, receive all N completions
3. Measure agreement (exact match, semantic similarity, or ask another model to judge)
4. Pick the best or trigger a consultation

### Signal summary

| Signal | Cost | Granularity | What it catches |
|---|---|---|---|
| Entropy | Cheap (sum over softmaxed logits) | Per-token | General uncertainty, confusion |
| Margin (top1 - top2) | Very cheap (2 values) | Per-token | Ambiguous choices, near-ties |
| Sequence variance | Cheap (rolling stats over window) | Per-window | Hallucination zones, knowledge fadeout |
| Self-consistency | Expensive (N generations) | Per-request | Correctness, factual reliability |

The C++ hook computes A, B, C on every token (cheap — tens of microseconds) and includes them in the callback data. mesh-llm decides if/when to act. Signal D is a request-level decision made by mesh-llm before inference starts.

---

## Hook points in llama-server

The token generation loop in `server-context.cpp` follows this path per slot per decode step:

```
llama_decode(batch)
  → common_sampler_sample() — produces a token
  → process_token()         — accumulates text, checks stop words, emits SSE chunk
  → stop condition check    — EOS, limit, time, stop word
  → send_partial_response() — pushes chunk to HTTP response queue
  ... or on stop:
  → send_final_response()   — pushes final result
  → slot.release()          — slot goes idle, KV cache retained
```

### Hook 1: Pre-inference (`launch_slot_with_task`)

Fires once when a request is assigned to a slot, before any tokens are evaluated.

**Available state**: full request body, messages, model capabilities.

**What the C++ does**: Calls `POST /mesh/hook` with `hook: "pre_inference"` and the full request context. If mesh-llm responds with `action: "inject"`, the hook adds the injected text to the prompt messages before tokenization. If mesh-llm responds with `action: "none"`, the hook also optionally fires an **async** consultation (mesh-llm can return `async: true` to say "I'm working on it, check back later").

**Use cases**: Image captioning, context pre-fetch, request classification.

### Hook 2: Post-prefill (`SLOT_STATE_DONE_PROMPT → SLOT_STATE_GENERATING`)

Fires once after the prompt is fully evaluated but before the first token is sampled. The model has "read" the prompt — logits for the first generation position are available.

**Available state**: First-token logit distribution (entropy, margin, top candidates), prompt token count, cache hit ratio.

**What the C++ does**: Computes first-token entropy and margin. Calls `POST /mesh/hook` with `hook: "post_prefill"` and these signals. If mesh-llm responds with `action: "inject"`, tokens are injected and re-evaluated before generation starts. The client hasn't seen anything yet — this delays TTFT but ensures the first token is well-informed.

**Use cases**: First-token confidence check, knowledge retrieval when model shows high initial uncertainty.

### Hook 3: Per-token (`process_token`)

Fires after every sampled token. The fast path (no action needed) is a branch check on a flag.

**Available state**: Token, probability, entropy, margin, rolling signal window, generated text so far, async consultation status.

**What the C++ does**:
1. Update rolling signal window (entropy, margin, streak counters)
2. Check if an async consultation (from Hook 1) has a result ready (`std::future::wait_for(0ms)`)
3. If result ready → inject tokens into KV cache, reset signal window
4. If no async result and no signal threshold exceeded → return immediately (fast path)
5. If signal threshold exceeded (high entropy streak, pattern match, etc.) → call `POST /mesh/hook` with `hook: "per_token"` and signal data → mesh-llm decides action

**Callback data sent to mesh-llm**:
```json
{
  "hook": "per_token",
  "signals": {
    "token_entropy": 4.2,
    "token_margin": 0.03,
    "window_entropy_mean": 3.8,
    "window_entropy_max": 5.1,
    "window_margin_min": 0.01,
    "uncertain_streak": 5
  },
  "generated_text": "The function in auth.rs that handles...",
  "n_decoded": 42
}
```

**Use cases**: Confidence-triggered retrieval, async result injection, hallucination detection.

### Hook 4: Pre-response (`send_final_response`)

Fires when generation is complete but before the response is sent.

**Available state**: Complete generated text, all signal history, timings, stop reason, full token probability log.

**What the C++ does**: Calls `POST /mesh/hook` with `hook: "pre_response"` and the full response. If mesh-llm responds with `action: "inject"` + `continue: true`, the hook injects correction tokens, sets `has_next_token = true`, and resumes generation in the same slot. If `action: "none"`, the response is sent as-is.

**Use cases**: Response verification, quality gating, correction injection.

### Hook 5: Complete (`slot.release()`)

Fires after the response is sent, as the slot goes idle.

**What the C++ does**: Fire-and-forget notification to `POST /mesh/hook` with `hook: "complete"` and telemetry (signal history, timing, whether consultations were triggered, response quality metrics).

**Use cases**: Routing feedback, confidence calibration, learning which requests need help.

---

## Token injection

The core mechanism: adding tokens to a running slot's KV cache without restarting inference.

The existing codebase already does this in two places:
- **Speculative decoding**: adds draft tokens to the batch, evaluates them, rolls back rejected ones
- **Context shifting**: removes old tokens from KV cache, shifts positions, continues

Token injection for mesh hooks follows the same pattern:
1. Tokenize the consultation result (e.g., image caption text)
2. Add tokens to the slot's batch via `common_batch_add()`
3. Push tokens to `slot.prompt.tokens`
4. Evaluate the batch (`llama_decode`)
5. The KV cache now includes the injected content
6. Continue sampling — next token is conditioned on everything including the injection

The injected tokens don't appear in the SSE stream to the client. They're internal context.

---

## Dynamic knowledge retrieval

Three approaches using the hooks above, depending on what you know when:

### Pre-flight (Hook 1): you know upfront

Request itself reveals the need. "Tell me about auth.rs" or images in a text-only model request.

```
Hook 1 → detect need → sync call to mesh → inject into prompt → prefill with context
```

TTFT cost: consultation time. But response quality is much higher.

### Async pre-fetch (Hook 1 + Hook 3): hedge your bets

Fire consultation async, start generating immediately, inject when ready.

```
Hook 1 → fire async → mesh starts working
Prefill + generation start immediately
Hook 3 (token 12) → async result ready → inject 200 tokens into KV
Token 13+ conditioned on injected context
```

TTFT cost: zero. Brief stall at injection point. First few tokens may be suboptimal.

### Triggered (Hook 3): the model tells you

Signals detect the model is uncertain. Hook fires, mesh consults, result injected.

```
Hook 3 → entropy spike for 5 tokens → threshold exceeded
→ sync call to mesh → mesh fetches context → injects into KV
→ model continues with the knowledge
```

TTFT cost: zero. Mid-stream pause during consultation. Stream uninterrupted.

---

## Communication: llama-server → mesh-llm

The hook makes HTTP requests back to mesh-llm on localhost. mesh-llm routes to the appropriate model anywhere in the mesh using existing routing, QUIC tunneling, and model discovery.

```
llama-server hook
  → HTTP POST localhost:{mesh_port}/mesh/hook
  → mesh-llm receives, routes to best model (local or remote peer)
  → inference happens on consulting model
  → result returned to llama-server hook
  → hook injects tokens into slot KV cache
  → generation continues
```

cpp-httplib (already linked in llama-server) handles the HTTP client side. mesh-llm exposes the hook endpoint on its management API port.

---

## Changes required

### llama.cpp fork (C++)

- `server_slot`: add `mesh_hook_context` struct with:
  - mesh port, request ID, async state (`std::future`)
  - signal window: rolling entropy, margin, streak counters
  - hook configuration: thresholds, enabled flags
- `server_slot`: add `SLOT_STATE_WAITING_MESH` state
- `task_params`: parse `mesh_*` fields from request JSON
- `launch_slot_with_task` (Hook 1): call mesh, fire async consultations
- `DONE_PROMPT → GENERATING` transition (Hook 2): first-token signal check
- `process_token` (Hook 3): update signal window, check async results, conditional callback
- `send_final_response` (Hook 4): optional verification callback
- `slot.release` (Hook 5): telemetry notification
- Token injection function using existing `common_batch_add` + `llama_decode` pattern

### mesh-llm (Rust)

- `inference/launch.rs`: pass mesh port to llama-server
- `api/`: new `POST /mesh/hook` endpoint — receives hook callbacks, decides actions
- `inference/virtual.rs`: decision logic — when to inject, what to fetch, which model to consult
- Request preparation: set `mesh_hooks: true` and signal thresholds per request
- Consultation routing: use existing `route_model_request` to send sub-requests to mesh models
