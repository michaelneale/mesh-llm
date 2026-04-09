# Virtual LLM Engine

Callback hooks from llama-server into mesh-llm during inference. The model being served can ask the mesh for help — caption images, fetch context, verify its output — without the caller knowing.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## Callback protocol

llama-server POSTs JSON to mesh-llm on localhost at defined hook points during inference. mesh-llm decides what to do and replies with an action.

**Request** (llama-server → mesh-llm):
```
POST http://localhost:{mesh_port}/mesh/hook
Content-Type: application/json

{
  "hook": "pre_inference" | "post_prefill" | "pre_response" | "complete",
  ... hook-specific fields ...
}
```

**Response** (mesh-llm → llama-server):
```json
{
  "action": "none" | "inject" | "stop",
  "text": "...",
  "continue": false
}
```

**Actions**:
- `none` — do nothing, continue normally
- `inject` — tokenize `text`, add to prompt or KV cache, continue
- `stop` — halt generation, release slot

---

## Hooks

### 1. Pre-inference

Fires when a request is assigned to a slot, before tokenization or prefill. Nothing has started. The full original request is available.

**Callback data**:
```json
{
  "hook": "pre_inference",
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "model_capabilities": ["text", "code", "tool_use"],
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
    ]}
  ],
  "has_images": true,
  "has_audio": false
}
```

Sends the original messages array including image/audio data URLs, model capabilities, and media flags.

**Example — image captioning**:

Request has images but the model is text-only. mesh-llm detects the mismatch, forwards the image blocks to a vision model in the mesh, gets a caption back:

```json
{
  "action": "inject",
  "text": "[Image description: A Python code snippet showing a recursive fibonacci function with a bug on line 3 where the base case returns n instead of 1]"
}
```

The caption is prepended to the prompt as a system message before tokenization. The text model generates with the caption in context from the start.

**Example — context pre-fetch**:

Request asks about a specific topic. mesh-llm uses a small fast model to summarize relevant context:

```json
{
  "action": "inject",
  "text": "Context: auth.rs contains three functions: verify_token() for JWT validation, check_session() for session lookup, and refresh_auth() for token renewal."
}
```

**Example — no action needed**:

Model has vision capability and can handle the images directly:

```json
{ "action": "none" }
```

**Where in C++**: `launch_slot_with_task()`, after task assignment, before tokenization. The parsed request JSON (`data`) is still available.

**Latency**: Delays time-to-first-token by the callback time. If `none`, ~0.1ms. If captioning, the vision model inference time.

---

### 2. Post-prefill

Fires after the prompt is fully evaluated but before the first token is sampled. The model has "read" the prompt — first-token logits are available. Nothing has been streamed to the client yet.

**Callback data**:
```json
{
  "hook": "post_prefill",
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "n_prompt_tokens": 847,
  "signals": {
    "first_token_entropy": 6.8,
    "first_token_margin": 0.02,
    "top_tokens": [
      {"text": "I",      "prob": 0.08},
      {"text": "The",    "prob": 0.06},
      {"text": "Based",  "prob": 0.06},
      {"text": "Sorry",  "prob": 0.05},
      {"text": "\n",     "prob": 0.04}
    ]
  }
}
```

Sends first-token entropy (how spread out the distribution is), margin (gap between top two candidates), and the top-5 candidates. No message content — mesh-llm already has the request from Hook 1, keyed by `request_id`.

**Signals explained**:
- **Entropy**: `H = -Σ p_i × log₂(p_i)`. Low (~1-2 bits) = confident. High (~8+ bits) = confused.
- **Margin**: `p_top1 - p_top2`. Large (>0.3) = clear winner. Small (<0.05) = coin flip.

**Example — model is confused**:

Entropy is 6.8, margin is 0.02 — the model doesn't know how to start. mesh-llm fetches context from another model in the mesh:

```json
{
  "action": "inject",
  "text": "The user is asking about the authentication flow. The relevant code is in auth.rs which uses JWT tokens validated by the verify_token function."
}
```

Tokens are injected into the KV cache and re-evaluated. The model now has context to work with. First token sampled from the updated state.

**Example — model is about to refuse**:

Top token is "Sorry" with 0.05 probability. mesh-llm injects a steering nudge:

```json
{
  "action": "inject",
  "text": "Answer the question directly based on your knowledge."
}
```

**Example — model is confident**:

Entropy is 1.4, margin is 0.45 — model knows exactly what to say:

```json
{ "action": "none" }
```

**Where in C++**: The `SLOT_STATE_DONE_PROMPT → SLOT_STATE_GENERATING` transition. Entropy and margin computed from `get_token_probabilities(ctx, tok_idx)`.

**Latency**: Delays first token. But if entropy is high, the model was going to produce a bad response anyway — better to delay and improve.

---

### 3. Pre-response

Fires when generation is complete (EOS, stop word, token limit). Full generated text available. For non-streaming requests, nothing has been sent to the client. For streaming, partial chunks already went out but the final chunk hasn't.

**Callback data**:
```json
{
  "hook": "pre_response",
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "generated_text": "The verify_token() function in auth.rs handles session validation by checking the JWT signature...",
  "n_decoded": 156,
  "stop_reason": "eos",
  "signals": {
    "mean_entropy": 2.4,
    "max_entropy": 7.1,
    "min_margin": 0.01,
    "uncertain_token_count": 12,
    "generation_time_ms": 3200
  }
}
```

Sends the complete generated text and summary stats over the whole generation: mean and max entropy, minimum margin, how many tokens were uncertain, total time.

**Example — verification**:

mesh-llm sends the generated text to a verifier model. The verifier flags an issue. mesh-llm returns a correction:

```json
{
  "action": "inject",
  "text": "\n\nCorrection: verify_token() checks the JWT expiry, not the signature. Signature validation is handled by check_session().",
  "continue": true
}
```

With `continue: true`, llama-server injects the correction tokens, sets `has_next_token = true`, and resumes generation. The model continues from the correction and produces an updated conclusion. For streaming, this appears as more content arriving after a brief pause.

**Example — quality scoring (no intervention)**:

Signal stats look fine (low entropy, good margin). mesh-llm logs the stats for routing feedback but takes no action:

```json
{ "action": "none" }
```

**Where in C++**: `send_final_response()`, before pushing the result to the HTTP response queue.

**Latency**: Delays final response. For non-streaming, the client waits. For streaming, only the final chunk is delayed.

---

### 4. Complete (telemetry)

Fires after the response is fully sent and the slot is released. Fire-and-forget — no response expected.

**Callback data**:
```json
{
  "hook": "complete",
  "request_id": "chatcmpl-abc123",
  "model": "qwen3-32b",
  "n_prompt_tokens": 847,
  "n_decoded": 156,
  "generation_time_ms": 3200,
  "hooks_fired": ["pre_inference", "post_prefill"],
  "hooks_injected": ["pre_inference"],
  "injection_time_ms": 450,
  "signal_summary": {
    "mean_entropy": 2.4,
    "max_entropy": 7.1,
    "min_margin": 0.01,
    "uncertain_token_count": 12
  }
}
```

**What mesh-llm does**: Learns patterns. "Requests about auth.rs from qwen3-32b needed captioning 80% of the time → pre-fetch faster next time." Feeds routing decisions for future requests.

**Where in C++**: `slot.release()`. Runs in a detached thread so it doesn't block anything.

---

## Signal computation

Entropy and margin are computed per-token inside the C++ generation loop — pure arithmetic on the softmax distribution that `get_token_probabilities()` already produces. No callback, no allocation, ~1μs per token.

The slot maintains a rolling window:

```cpp
struct mesh_signal_window {
    static constexpr int SIZE = 16;
    float entropy[SIZE];
    float margin[SIZE];
    int   pos = 0;
    int   count = 0;

    // derived stats, updated on each push
    float entropy_mean;
    float entropy_max;
    float margin_min;
    int   uncertain_streak;  // consecutive tokens above entropy threshold
    int   uncertain_count;   // total uncertain tokens
};
```

These stats are sent as summaries at Hook 2 (first-token only) and Hook 3 (full generation summary). They're never sent per-token.

### Self-consistency

A request-level strategy, not a per-token signal. mesh-llm sets `n_cmpl: 3` on the request. llama-server prefills once, copies KV cache to 3 child slots via `copy_state_to()`, generates 3 independent completions. At Hook 3, mesh-llm receives all 3 and measures agreement. Convergence = correct. Divergence = uncertain.

Uses existing `n_cmpl` machinery — no new C++ code for the generation part.

---

## Token injection

How `inject` works at each hook:

**Hook 1 (pre-inference)**: Text is prepended to the prompt as a system message. Happens before tokenization. No KV cache manipulation — it's just part of the prompt.

**Hook 2 (post-prefill)**: Tokens are added to the live KV cache via `common_batch_add()` + `llama_decode()`. The model's attention now includes the injected content. Same pattern as speculative decoding (adding draft tokens) and context shifting (removing/shifting tokens). Next token is sampled from the updated state.

**Hook 3 (pre-response)**: If `continue: true`, tokens are appended after the generated text in the KV cache. `has_next_token` set to true, generation resumes — model continues from the injection.

Injected tokens are invisible to the client's SSE stream.

---

## Mid-generation intervention (future)

The four hooks above cover pre, post-prefill, post-generation, and telemetry. If mid-generation intervention is needed later, two approaches:

**Threshold breakout**: C++ monitors the rolling signal window. When `uncertain_streak` exceeds a configured limit, it fires a one-time callback. This is a rare event — maybe 0-2 times per generation, not per-token. Slot pauses, mesh-llm consults, injects, slot resumes. Stream pauses briefly but doesn't break.

**Async poll**: Hook 1 starts a background consultation (mesh-llm returns `async_id`). C++ polls `GET /mesh/hook/async/{id}` every N tokens (8-16, not every token). When ready, inject. Zero TTFT cost, background work runs parallel to generation.

---

## llama.cpp fork changes

### New fields on `server_slot`

```cpp
struct mesh_hook_ctx {
    bool enabled = false;
    int  port = 0;
    std::string request_id;
    json original_request;  // stored at Hook 1 for reference

    mesh_signal_window signals;

    std::unique_ptr<httplib::Client> client;

    void init(int mesh_port) {
        client = std::make_unique<httplib::Client>("localhost", mesh_port);
        client->set_connection_timeout(0, 100000); // 100ms connect
        client->set_read_timeout(30);              // 30s read
    }
};
```

### New fields on `task_params`

```cpp
bool mesh_hooks = false;
int  mesh_port  = 3131;
```

Parsed from request JSON alongside existing fields (`cache_prompt`, `n_predict`, etc.).

### New slot state

```cpp
SLOT_STATE_WAITING_MESH  // paused during sync callback, other slots continue
```

### New CLI flag

```
--mesh-port PORT    Port for mesh hook callbacks (enables hook system)
```

When set, hooks are enabled for requests that include `mesh_hooks: true`.

### Hook insertion points

| Hook | Location | Code point |
|---|---|---|
| pre_inference | `launch_slot_with_task()` | After task assignment, before tokenization |
| post_prefill | `update_slots()` | At `DONE_PROMPT → GENERATING` transition |
| pre_response | `send_final_response()` | Before pushing result to queue |
| complete | `slot.release()` | After response sent, detached thread |

### Signal computation

In the generation loop, after `common_sampler_sample()` and `common_sampler_accept()`:

```cpp
if (slot.mesh_hook.enabled) {
    auto probs = get_token_probabilities(ctx, tok_idx);
    float entropy = compute_entropy(probs);
    float margin = probs[0].p - probs[1].p;
    slot.mesh_hook.signals.push(entropy, margin);
}
```

No callback. Just arithmetic and a ring buffer write.

---

## mesh-llm changes

### New API endpoint

`POST /mesh/hook` on the management API port (3131). Receives callbacks, runs decision logic, returns action.

### New module: `inference/virtual.rs`

Decision engine for each hook type:
- Pre-inference: detect media mismatches, decide what to pre-fetch
- Post-prefill: evaluate first-token signals, decide if context needed
- Pre-response: optionally verify, score, or correct
- Complete: record telemetry, update routing heuristics

Consultation requests routed through existing mesh infrastructure — same model discovery, QUIC tunneling, and routing that normal requests use.

### `inference/launch.rs`

Pass `--mesh-port {api_port}` when spawning llama-server.

### `network/proxy.rs`

When forwarding a request to llama-server, set `mesh_hooks: true` in the JSON body when hooks should be active (e.g., images present + text-only model, or complex request that might benefit from verification).
