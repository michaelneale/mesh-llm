# Virtual LLM Engine

Callback hooks from llama-server into mesh-llm during inference.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## What it does

llama-server calls mesh-llm at key points during inference. mesh-llm can inject context, caption images, or verify output — using any model in the mesh. The caller sees one seamless response.

## Callback

All hooks POST JSON to `http://localhost:{mesh_port}/mesh/hook`.

Response:
```json
{ "action": "none" }
{ "action": "inject", "text": "..." }
{ "action": "pending", "async_id": "abc" }
{ "action": "stop" }
```

- **none** — continue normally
- **inject** — add text to prompt or KV cache
- **pending** — mesh-llm is working on it in the background, poll for result later
- **stop** — halt generation

Pending results polled via `GET /mesh/hook/poll/{async_id}` → 202 (not ready) or 200 (ready, with inject action).

---

## Hooks

### 1. Pre-inference

Before tokenization. Full request available.

```json
{
  "hook": "pre_inference",
  "request_id": "chatcmpl-abc",
  "model": "qwen3-32b",
  "model_capabilities": ["text", "code"],
  "messages": [ ... ],
  "has_images": true,
  "has_audio": false
}
```

Messages include image/audio data URLs as-is from the original request.

**Sync example** — image caption (fast vision model available):
```json
{ "action": "inject", "text": "[Image: Python fibonacci function with bug on line 3]" }
```
Caption goes into the prompt before prefill. Blocks TTFT by the captioning time.

**Async example** — image caption (slow/remote vision model):
```json
{ "action": "pending", "async_id": "cap-001" }
```
Generation starts immediately. Caption injected into KV cache when it arrives (see [Polling](#polling)).

**No action** — model handles it natively:
```json
{ "action": "none" }
```

---

### 2. Post-prefill

Prompt evaluated, first-token logits ready, nothing streamed yet.

```json
{
  "hook": "post_prefill",
  "request_id": "chatcmpl-abc",
  "n_prompt_tokens": 847,
  "signals": {
    "first_token_entropy": 6.8,
    "first_token_margin": 0.02,
    "top_tokens": [
      { "text": "I", "prob": 0.08 },
      { "text": "The", "prob": 0.06 },
      { "text": "Sorry", "prob": 0.05 }
    ]
  }
}
```

Always sync. mesh-llm looks at the numbers and returns immediately — no model consultation, <1ms.

**Inject** — model is confused (high entropy), mesh-llm adds context:
```json
{ "action": "inject", "text": "The relevant code is in auth.rs: verify_token() checks JWT signatures." }
```

**None** — model is confident (low entropy):
```json
{ "action": "none" }
```

---

### 3. Pre-response

Generation complete. Full text available.

```json
{
  "hook": "pre_response",
  "request_id": "chatcmpl-abc",
  "generated_text": "The verify_token() function handles...",
  "n_decoded": 156,
  "stop_reason": "eos",
  "signals": {
    "mean_entropy": 2.4,
    "max_entropy": 7.1,
    "min_margin": 0.01,
    "uncertain_token_count": 12
  }
}
```

Always sync — need the verdict before sending to client.

**Correction** — verification failed, resume generation with fix:
```json
{ "action": "inject", "text": "\n\nCorrection: verify_token() checks expiry, not signatures.", "continue": true }
```

**None** — response is fine:
```json
{ "action": "none" }
```

---

## Polling

When Hook 1 returns `pending`, llama-server polls every 16 tokens during generation:

```
GET /mesh/hook/poll/cap-001 → 202         (not ready)
GET /mesh/hook/poll/cap-001 → 200 + json  (ready)
```

Ready response:
```json
{ "action": "inject", "text": "[Image: fibonacci function with bug on line 3]" }
```

On 200, tokens are injected into the live KV cache via `common_batch_add()` + `llama_decode()`. Generation continues conditioned on the injected content.

Poll cost: ~0.1ms per poll. At 50 tok/s with interval 16, that's ~3 polls/second.

---

## Signals

Computed per-token in C++ (no callback). Cheap arithmetic on the softmax distribution.

**Entropy**: `H = -Σ p_i × log₂(p_i)`. Low = confident. High = uncertain.

**Margin**: `p_top1 - p_top2`. Large = clear choice. Small = coin flip.

Accumulated in a rolling window on the slot. Sent as summary at Hook 2 (first token only) and Hook 3 (full generation).

**Self-consistency**: mesh-llm sets `n_cmpl: 3` on the request. llama-server prefills once, generates 3 completions via existing `copy_state_to()`. mesh-llm compares them at Hook 3. No new C++ code.

---

## Token injection

**Hook 1** (pre-inference): Text prepended to prompt as system message. Before tokenization — just part of the prompt.

**Hook 2** (post-prefill): Tokens added to live KV cache. `common_batch_add()` + `llama_decode()`. Model attention now includes the injection.

**Hook 3** (pre-response): Tokens appended after generated text. With `continue: true`, generation resumes from the injection.

**Polling**: Same as Hook 2 — injected into live KV cache mid-generation.

Injected tokens are invisible to the client.

---

## What we build

### llama.cpp fork

**New struct on `server_slot`**:
```cpp
struct mesh_hook_ctx {
    bool enabled = false;
    int  port = 0;
    std::string request_id;
    json original_request;

    // signal window
    mesh_signal_window signals;

    // async poll state
    std::vector<std::string> pending_async_ids;
    int tokens_since_last_poll = 0;
    int poll_interval = 16;

    // reusable HTTP client
    std::unique_ptr<httplib::Client> client;
};
```

**New on `task_params`**: `bool mesh_hooks`, `int mesh_port`. Parsed from request JSON.

**New slot state**: `SLOT_STATE_WAITING_MESH` — paused during sync callback, other slots continue.

**New CLI flag**: `--mesh-port PORT` — enables hook system.

**Hook points**:

| Hook | Where | Blocking |
|---|---|---|
| pre_inference | `launch_slot_with_task()` | sync or async |
| post_prefill | `DONE_PROMPT → GENERATING` | sync (<1ms) |
| pre_response | `send_final_response()` | sync |

**Generation loop** (after `common_sampler_accept`):
```cpp
if (slot.mesh_hook.enabled) {
    // update signal stats
    auto probs = get_token_probabilities(ctx, tok_idx);
    slot.mesh_hook.signals.push(compute_entropy(probs), probs[0].p - probs[1].p);

    // poll async results every N tokens
    if (slot.mesh_hook.should_poll()) {
        for (auto & id : slot.mesh_hook.pending_async_ids) {
            auto res = slot.mesh_hook.client->Get("/mesh/hook/poll/" + id);
            if (res && res->status == 200) {
                inject_tokens(slot, json::parse(res->body)["text"]);
                // remove from pending
            }
        }
    }
}
```

### mesh-llm

**New endpoints** on management API (port 3131):
- `POST /mesh/hook` — receives callbacks, returns action
- `GET /mesh/hook/poll/{async_id}` — returns 202 or 200 with result

**New module** `inference/virtual.rs`:
- Decision logic per hook type
- Spawns tokio tasks for async consultations (image captioning, context fetch)
- Stores results keyed by async_id
- Routes consultation requests through existing mesh infrastructure

**`inference/launch.rs`**: Pass `--mesh-port {api_port}` to llama-server.

**Request preparation**: Set `mesh_hooks: true` in forwarded request body when hooks should be active.
