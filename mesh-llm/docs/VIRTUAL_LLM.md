# Virtual LLM Engine

Hooks inside llama-server's token generation loop that call back to mesh-llm over localhost JSON, so the inference engine itself can consult other models during generation.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## Overview

llama-server gets a small set of hooks in its C++ generation loop. At each hook point, it computes cheap signals (entropy, margin) and — when a threshold is exceeded — makes an HTTP POST to mesh-llm on localhost. mesh-llm does all the thinking: decides whether to consult another model, which one, what to ask. It replies with an action (`none`, `inject`, `stop`). llama-server executes the action mechanically — inject tokens into the KV cache, or stop generation — then continues.

The C++ side is dumb plumbing. The Rust side is the brain.

```
┌─────────────────────────────────────────────────────┐
│ llama-server (C++)                                  │
│                                                     │
│  sample token                                       │
│    → compute entropy, margin                        │
│    → threshold exceeded?                            │
│       no  → continue (fast path, ~1μs)              │
│       yes → POST localhost:{port}/mesh/hook         │
│             ← { action: "inject", text: "..." }     │
│             → tokenize, inject into KV, continue    │
└──────────────────────┬──────────────────────────────┘
                       │ localhost HTTP
┌──────────────────────▼──────────────────────────────┐
│ mesh-llm (Rust)                                     │
│                                                     │
│  receive hook callback                              │
│    → inspect signals + generated text + request     │
│    → decide: consult another model? which one?      │
│    → route sub-request through mesh (local or QUIC) │
│    → return action to llama-server                  │
└─────────────────────────────────────────────────────┘
```

---

## Why JSON over localhost, not a native bridge

Options considered:
- **FFI / shared library**: Tight coupling, complicates build, llama-server would need to link Rust code. Fragile across llama.cpp updates.
- **Unix socket / shared memory**: Faster IPC, but marginal — the bottleneck is the consulting model's inference time (seconds), not the localhost round-trip (~0.1ms).
- **JSON over localhost HTTP**: cpp-httplib is already linked in llama-server. mesh-llm already runs an HTTP server (port 3131). The callback is one `httplib::Client::Post()` call. Easy to debug (`curl`), easy to test, easy to version the protocol independently.

JSON over localhost wins because:
1. **Already wired up** — cpp-httplib on C++ side, axum on Rust side
2. **Debuggable** — you can `curl localhost:3131/mesh/hook` to test
3. **Decoupled** — llama.cpp fork stays a clean fork, hooks are a thin addition
4. **Latency is irrelevant** — the slow part is consulting another model, not the callback itself

---

## Hooks

Five hook points. One callback shape. The C++ changes are minimal — a struct on the slot, signal computation in `process_token`, and `httplib::Client::Post` calls.

### The callback protocol

Every hook fires the same request shape:

```
POST http://localhost:{mesh_port}/mesh/hook
Content-Type: application/json

{
  "hook": "pre_inference" | "post_prefill" | "generating" | "pre_response" | "complete",
  "slot_id": 0,
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "n_decoded": 42,
  "generated_text": "The function in auth.rs...",
  "signals": { ... },
  "request_hash": "sha256-of-original-request"
}
```

Response:

```json
{
  "action": "none" | "inject" | "stop",
  "text": "Context: auth.rs contains verify_token(), check_session()...",
  "continue": false,
  "async_id": "consultation-123"
}
```

Actions:
- `none` — do nothing, continue generating
- `inject` — tokenize `text`, add to KV cache, continue generating (or start generating if pre-inference). If `continue: true` at pre-response, resume generation after injection.
- `stop` — halt generation, release slot

The `async_id` field: mesh-llm can return `action: "none"` with an `async_id` to say "I've started working on something, the llama-server hook should poll for it." The per-token hook checks for completed async results via a separate lightweight poll (see below).

### Hook 1: `pre_inference`

**When**: `launch_slot_with_task` — request assigned to slot, before tokenization/prefill.

**What C++ sends**: Full request body (messages, images, params). No signals yet.

**What mesh-llm can do**:
- Detect images + text-only model → caption synchronously → inject as system message
- Detect complex query → pre-fetch context → inject into prompt
- Return `async_id` to start a background consultation that will be checked at Hook 3
- Set signal thresholds for this request (override defaults)

**Cost**: Delays TTFT by the callback round-trip. If mesh-llm returns `none` immediately (~0.1ms), negligible. If it returns `inject` with context fetched synchronously, TTFT increases by the consultation time.

**C++ code**: One `httplib::Client::Post`, parse response, if `inject` then prepend text to messages before tokenization. Store any `async_id`.

### Hook 2: `post_prefill`

**When**: `SLOT_STATE_DONE_PROMPT → SLOT_STATE_GENERATING` — prompt evaluated, first-token logits available, nothing streamed to client yet.

**What C++ sends**: First-token signals (entropy, margin, top-5 token IDs and probs).

**What mesh-llm can do**:
- High first-token entropy → fetch context → inject → re-evaluate (model "re-reads" with new context)
- First-token looks like refusal → inject steering prompt
- Return `none` for the fast path

**Cost**: Delays TTFT. But if first-token entropy is high, the model was going to produce a bad response anyway — better to delay and improve.

**C++ code**: Compute entropy/margin from `get_token_probabilities(ctx, tok_idx)`, POST, if `inject` then add tokens via `common_batch_add`, re-evaluate.

### Hook 3: `generating`

**When**: Inside `process_token`, after each sampled token.

This is the hot path. It must be fast when nothing is happening. The structure:

```cpp
// 1. Always: update signal window (cheap, no allocation)
mesh_ctx.update_signals(entropy, margin);

// 2. Check async: is a previously-started consultation ready?
if (mesh_ctx.has_pending_async()) {
    auto result = mesh_ctx.poll_async();  // non-blocking check
    if (result.has_value()) {
        inject_tokens(slot, result->text);
        return;
    }
}

// 3. Check threshold: should we call mesh-llm?
if (!mesh_ctx.threshold_exceeded()) {
    return;  // fast path — nothing to do
}

// 4. Threshold exceeded — call mesh-llm
auto response = mesh_ctx.client.Post("/mesh/hook", payload, "application/json");
// ... handle action ...
```

Steps 1-3 are branch checks + arithmetic — microseconds. Step 4 only fires when the model is actually uncertain.

**What C++ sends**: Current token, signals (see below), rolling window stats, generated text so far.

**What mesh-llm can do**:
- Decide the model needs help → consult another model → return `inject` with context
- Decide this is a hallucination zone → return `stop` to cut generation
- Return `none` (most common case)
- Return `async_id` for a new background consultation

**Cost on fast path**: ~1-5μs per token (signal update + branch check). Negligible vs the ~10-50ms per token for generation.

**Cost when triggered**: The sync POST + mesh-llm decision time + optional consultation. Stream pauses briefly. Other slots continue.

### Hook 4: `pre_response`

**When**: `send_final_response` — generation complete, full text available.

**What C++ sends**: Complete generated text, signal history summary, stop reason, timing.

**What mesh-llm can do**:
- Verify response with another model → if bad, return `inject` + `continue: true` → correction injected, generation resumes
- Score quality for routing feedback
- Return `none` to send response as-is

**Cost**: Delays final response delivery. For streaming, partial chunks already sent — this only affects the final chunk/metadata.

**C++ code**: POST, if `inject` + `continue`, add tokens to KV, set `has_next_token = true`, return to generation loop.

### Hook 5: `complete`

**When**: `slot.release()` — after response fully sent.

**What C++ sends**: Telemetry — full signal history, consultation count, timing breakdown.

**What mesh-llm can do**: Learn patterns. "Requests about auth.rs from this model triggered consultation 80% of the time → pre-fetch next time." No action returned.

**Cost**: Fire-and-forget. Detached thread, no blocking.

**C++ code**: `std::thread([...]{ client.Post(...); }).detach();`

---

## Signals

Computed per-token inside `process_token`. The C++ side computes raw numbers and sends them to mesh-llm. mesh-llm decides what they mean.

### Entropy

```
H = -Σ p_i × log₂(p_i)
```

Computed from the softmax distribution that `get_token_probabilities()` already calculates.

- Low (~1-2 bits) → confident, peaked distribution
- High (~8+ bits) → uncertain, spread across many tokens
- Sudden spike mid-generation → model hit something it doesn't know

### Margin

```
margin = p_top1 - p_top2
```

Two array lookups from the sorted probability array.

- Large (>0.3) → clear winner
- Small (<0.05) → coin flip between two tokens

Cheap proxy for uncertainty. When margin is small AND entropy is high, the model is genuinely lost.

### Window stats

Rolling buffer of last N tokens (default N=8):

```cpp
struct mesh_signal_window {
    static constexpr int SIZE = 8;
    float entropy[SIZE];
    float margin[SIZE];
    int   pos = 0;
    int   count = 0;

    // derived (updated on push)
    float entropy_mean;
    float entropy_max;
    float margin_min;
    int   uncertain_streak;  // consecutive tokens with entropy > threshold
};
```

The window captures **patterns**:
- Confident → uncertain → confident = **hallucination zone** (middle was fabricated)
- Confident → steadily uncertain = **knowledge fadeout** (model ran out of things it knows)
- Uniformly uncertain = **task too hard** for this model

The C++ code pushes to the ring buffer and updates the derived stats. That's it. mesh-llm reads the pattern.

### Self-consistency (request-level, not per-token)

Use existing `n_cmpl` parameter. mesh-llm sets `n_cmpl: 3` for requests that warrant it. llama-server does one prefill, copies KV to 3 child slots (`copy_state_to`), generates 3 completions. At Hook 4, mesh-llm receives all 3 and measures agreement.

No C++ changes needed — `n_cmpl` already exists. mesh-llm just needs to:
1. Set `n_cmpl: N, stream: false` on the request
2. Collect all N results at Hook 4
3. Compare (exact match, or ask another model to judge)

### Threshold configuration

mesh-llm sets thresholds per-request via extra JSON fields:

```json
{
  "mesh_hooks": true,
  "mesh_port": 3131,
  "mesh_entropy_threshold": 5.0,
  "mesh_margin_threshold": 0.05,
  "mesh_streak_threshold": 5,
  "mesh_hook_budget_ms": 3000
}
```

These are parsed in `task_params_from_json` alongside existing fields like `cache_prompt`, `n_predict`, etc. When `mesh_hooks` is false or absent, the hook code is completely skipped — zero overhead.

`mesh_hook_budget_ms` caps total time spent on consultations per request. If exceeded, hooks stop firing for the remainder of generation.

### Signal data in callbacks

```json
{
  "signals": {
    "entropy": 4.2,
    "margin": 0.03,
    "window_entropy_mean": 3.8,
    "window_entropy_max": 5.1,
    "window_margin_min": 0.01,
    "uncertain_streak": 5,
    "top_tokens": [
      {"id": 1234, "text": "verify", "prob": 0.15},
      {"id": 5678, "text": "check",  "prob": 0.12},
      {"id": 9012, "text": "auth",   "prob": 0.11}
    ]
  }
}
```

---

## Async consultation flow

The most common pattern: fire a consultation at Hook 1, check for results at Hook 3.

```
Hook 1 (pre_inference):
  C++ → POST /mesh/hook { hook: "pre_inference", ... }
  mesh-llm: "I'll caption these images in the background"
  mesh-llm → { action: "none", async_id: "cap-001" }
  C++ stores async_id, proceeds to prefill

  Meanwhile, mesh-llm sends images to vision model in mesh...

Hook 3 (generating), token 1-11:
  C++ checks: poll GET /mesh/hook/async/cap-001 → 202 (not ready)
  Continue generating (fast path)

Hook 3 (generating), token 12:
  C++ checks: poll GET /mesh/hook/async/cap-001
  mesh-llm → 200 { text: "The image shows a code snippet with..." }
  C++ injects caption tokens into KV cache
  Generation continues, now conditioned on the caption
```

The poll is a lightweight GET — no request body, mesh-llm just checks if the async result is ready. Returns 202 (not ready) or 200 (ready + result). This keeps the per-token cost near zero while the consultation runs in parallel.

```
GET http://localhost:{mesh_port}/mesh/hook/async/{async_id}

→ 202 (not ready)
→ 200 { "text": "...", "action": "inject" }
```

---

## Token injection

When a hook returns `action: "inject"`, the C++ code:

1. Tokenizes the text: `common_tokenize(vocab, text, false, true)`
2. Wraps it in a context marker: `\n[mesh-context]\n{text}\n[/mesh-context]\n`
3. Adds tokens to the slot's batch: `common_batch_add(batch, tok, pos, {slot.id}, true)` for each token
4. Pushes to `slot.prompt.tokens` to keep the token list in sync
5. Evaluates: `llama_decode(ctx, batch)` — KV cache updated in-place
6. Resets signal window (fresh start after injection)
7. Continues sampling from the new state

The injected tokens are invisible to the client. They don't appear in SSE chunks. They do affect all subsequent generation because the model's attention now includes them.

The wrapping tags (`[mesh-context]`/`[/mesh-context]`) let the model distinguish injected context from its own generation. These can be tuned per model — some models respond better to `<|system|>` markers, etc. mesh-llm can specify the wrapper format in the inject response.

---

## C++ changes summary

### New struct on `server_slot`

```cpp
struct mesh_hook_ctx {
    bool enabled = false;
    int  port = 0;
    std::string request_id;

    // signal window
    mesh_signal_window signals;

    // thresholds (set per-request from task_params)
    float entropy_threshold = 5.0f;
    float margin_threshold  = 0.05f;
    int   streak_threshold  = 5;
    int   hook_budget_ms    = 3000;
    int   hook_time_spent_ms = 0;

    // async state
    std::string pending_async_id;

    // HTTP client (reused across calls for this slot)
    std::unique_ptr<httplib::Client> client;

    void init(int mesh_port) {
        client = std::make_unique<httplib::Client>("localhost", mesh_port);
        client->set_connection_timeout(0, 100000); // 100ms connect
        client->set_read_timeout(30);              // 30s read (consultations can be slow)
    }

    bool budget_remaining() const {
        return hook_time_spent_ms < hook_budget_ms;
    }
};
```

### New fields on `task_params`

```cpp
bool  mesh_hooks            = false;
int   mesh_port             = 3131;
float mesh_entropy_threshold = 5.0f;
float mesh_margin_threshold  = 0.05f;
int   mesh_streak_threshold  = 5;
int   mesh_hook_budget_ms    = 3000;
```

Parsed from request JSON alongside existing params.

### New slot state

```cpp
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER,
    SLOT_STATE_STARTED,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
    SLOT_STATE_WAITING_MESH,  // NEW: paused, waiting for mesh consultation
};
```

Main loop skips `SLOT_STATE_WAITING_MESH` during batching. When the sync POST returns, slot transitions back to `SLOT_STATE_GENERATING`.

### Hook insertion points

1. **`launch_slot_with_task`** (~line 1070): After task assignment, before tokenization. POST pre_inference.
2. **`DONE_PROMPT → GENERATING`** (~line 2850): After prompt eval, before first sample. POST post_prefill.
3. **`process_token`** (~line 2890, after `common_sampler_accept`): Update signals, check async, check threshold. POST generating if threshold exceeded.
4. **`send_final_response`** (~line 2900): Before pushing final result. POST pre_response.
5. **`slot.release`** (~line 2905): Fire-and-forget POST complete.

---

## Rust changes summary

### New endpoint: `POST /mesh/hook`

Added to the existing axum router in `api/mod.rs`. Receives hook callbacks, runs decision logic, returns action.

### New module: `inference/virtual.rs`

Decision engine. Given a hook callback with signals + context:
- Should we consult another model?
- Which model? (use existing `route_model_request` / model discovery)
- What to ask? (construct prompt for the consulting model)
- What to inject? (format the consultation result)

### Changes to `inference/launch.rs`

Pass `--mesh-port {api_port}` to llama-server so it knows where to call back. This is a new CLI arg we add to the llama.cpp fork.

### Changes to `network/proxy.rs`

When routing a request to a text-only model and the request has images, set `mesh_hooks: true` and `mesh_caption_images: true` in the forwarded request body so the hooks know to fire.

---

## Why in-engine, not proxy-level

| Capability | Proxy Level | In-Engine |
|---|---|---|
| Image captioning for text models | Rewrite request before inference, or cancel + re-send | Async fetch during prefill, inject into live KV cache. Zero restart. |
| Response verification | Buffer full response, send to verifier, regenerate | Check at EOS, inject correction, resume generating in same slot. |
| Confidence-triggered help | Buffer tokens, score externally, cancel stream, re-route | Per-token signal check, pause slot, consult mesh, inject, resume. Stream uninterrupted. |
| Context enrichment mid-generation | Cannot do during streaming | Inject tokens into live KV cache. Model's attention updates in-place. |
| Dynamic knowledge retrieval | Cannot do during streaming | Model shows uncertainty → hook fires → mesh fetches context → injects → model continues with knowledge. |
| Seamless streaming to client | Must buffer, cancel, stitch SSE frames | Same slot, same stream, brief pause at most. |
