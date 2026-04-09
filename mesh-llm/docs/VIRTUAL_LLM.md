# Virtual LLM Engine

Callback hooks from llama-server into mesh-llm during inference.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## What it does

llama-server calls mesh-llm when it detects something it might need help with. mesh-llm can inject context, caption images, summarize long conversations, or verify output — using any model in the mesh. The caller sees one seamless response.

---

## Callback protocol

Hooks POST JSON to `http://localhost:{mesh_port}/mesh/hook`.

Response:
```json
{ "action": "none" }
{ "action": "inject", "text": "..." }
{ "action": "pending", "async_id": "abc" }
{ "action": "stop" }
```

- **none** — continue normally
- **inject** — add text to prompt or KV cache
- **pending** — mesh-llm is working on it, poll for result during generation
- **stop** — halt generation

Hook 1's response also configures the rest of the request:
```json
{
  "action": "pending",
  "async_id": "cap-001",
  "entropy_threshold": 5.0,
  "verify": true
}
```

Pending results polled via `GET /mesh/hook/poll/{async_id}` → 202 (not ready) or 200 (ready).

---

## Triggers

Hooks don't fire on every request. C++ checks cheap structural criteria and only calls mesh-llm when something looks like it needs attention. When `mesh_hooks` is false or absent, all hook code is skipped entirely.

### Hook 1 — pre-inference triggers

Checked before tokenization. All are cheap — JSON key scans, integer comparisons, array lengths.

| Trigger | Check | What it catches |
|---|---|---|
| Images + no multimodal | `has_images && mctx == nullptr` | Model can't see images, needs captioning |
| Audio + no audio support | `has_audio && !audio_capable` | Model can't hear audio |
| Tool calls + no tool support | `tools` array present + model lacks tool capability | Model may botch the tool format |
| Prompt > 75% of context | `n_tokens > n_ctx * 0.75` | Context pressure — may need conversation summary |
| Many turns | `message_count > 10` | Long session drift |
| Large user message | Last user message > 2000 tokens AND previous were short | User pasted code/docs, may need help understanding it |

**Example — images + text-only model:**
```json
{
  "hook": "pre_inference",
  "trigger": "images_no_multimodal",
  "request_id": "chatcmpl-abc",
  "model": "qwen3-32b",
  "model_capabilities": ["text", "code"],
  "messages": [ ... ],
  "has_images": true
}
```
mesh-llm captions via vision model in mesh → `{ "action": "inject", "text": "[Image: fibonacci function...]" }`

**Example — prompt filling context window:**
```json
{
  "hook": "pre_inference",
  "trigger": "context_pressure",
  "request_id": "chatcmpl-abc",
  "model": "qwen3-32b",
  "n_prompt_tokens": 14200,
  "n_ctx": 16384,
  "message_count": 24,
  "messages": [ ... ]
}
```
mesh-llm summarizes early conversation turns via fast model → `{ "action": "inject", "text": "[Conversation summary: user asked about auth flow, then debugging JWT tokens, then...]" }`

The injected summary replaces the early messages in the prompt, freeing context for the model to generate a full response.

**Example — large paste in conversation:**
```json
{
  "hook": "pre_inference",
  "trigger": "large_user_message",
  "request_id": "chatcmpl-abc",
  "model": "qwen3-32b",
  "n_prompt_tokens": 6400,
  "last_user_message_tokens": 4800,
  "messages": [ ... ]
}
```
mesh-llm uses a fast model to extract the key parts → `{ "action": "inject", "text": "Key context from pasted code: the main function calls auth.validate() which..." }`

Or async: `{ "action": "pending", "async_id": "ctx-001" }` — start extracting, generation begins immediately, inject when ready.

**No trigger met** — hook doesn't fire. Zero cost.

### Hook 2 — post-prefill triggers

Checked after prompt evaluation, before first token. C++ already has the logit distribution from prefill.

| Trigger | Check | What it catches |
|---|---|---|
| High entropy | `first_token_entropy > threshold` | Model doesn't know how to start |
| Low margin | `first_token_margin < 0.05` | Model torn between multiple starts |
| Top tokens are hedging | Top-3 tokens are "I", "Sorry", "Well", "Unfortunately" | Model about to refuse or waffle |

The entropy threshold is set by Hook 1's response (`entropy_threshold` field). If Hook 1 didn't fire or didn't set a threshold, Hook 2 is skipped.

```json
{
  "hook": "post_prefill",
  "trigger": "high_entropy",
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

mesh-llm returns context or steering → `{ "action": "inject", "text": "Answer directly based on..." }`

### Hook 3 — pre-response triggers

Checked after generation completes. C++ has the signal summary from the rolling window.

| Trigger | Check | What it catches |
|---|---|---|
| Cut off | `stop_reason == "max_tokens"` | Response incomplete |
| Very short | `n_decoded < 10` and prompt was substantial | Refusal or punt |
| High uncertainty | `uncertain_token_count > 30%` of total | Model was guessing throughout |
| Bad ending | Entropy spike in last 16 tokens (tail entropy >> mean) | Ending may be hallucinated |
| Mid-sentence | Last token isn't EOS or sentence-ending punctuation | Abrupt cutoff |

Hook 3 only fires if Hook 1 set `verify: true` OR if a trigger from this list is met. So even without explicit verification, C++ catches obvious problems.

```json
{
  "hook": "pre_response",
  "trigger": "max_tokens",
  "request_id": "chatcmpl-abc",
  "generated_text": "The verify_token() function handles session validation by...",
  "n_decoded": 4096,
  "n_predict": 4096,
  "stop_reason": "max_tokens",
  "signals": {
    "mean_entropy": 2.4,
    "max_entropy": 7.1,
    "min_margin": 0.01,
    "uncertain_token_count": 12,
    "tail_entropy_mean": 5.8
  }
}
```

**Example — response cut off at max_tokens:**

mesh-llm can:
- Summarize the partial response + re-request with "continue from..." → inject + continue
- Route to a model with larger context if available in the mesh
- Return `none` and let the truncated response through (honest but incomplete)

**Example — very short response (likely refusal):**
```json
{
  "hook": "pre_response",
  "trigger": "very_short",
  "generated_text": "I don't have access to that information.",
  "n_decoded": 8
}
```
mesh-llm rephrases the question via another model or injects encouragement → `{ "action": "inject", "text": "\n\nLet me try again with what I know:\n", "continue": true }`

**Example — high tail entropy (hallucinated ending):**
```json
{
  "hook": "pre_response",
  "trigger": "tail_entropy_spike",
  "generated_text": "...the function returns true. Additionally, it also validates the quantum signature of the hyperspace manifold...",
  "signals": { "mean_entropy": 2.1, "tail_entropy_mean": 6.4 }
}
```
mesh-llm detects the drift, asks a verifier model to check → if bad, injects correction → `{ "action": "inject", "text": "\n\n(The previous statement about quantum signatures is incorrect. The function simply returns true after validation.)", "continue": true }`

---

## Polling

When any hook returns `pending`, llama-server polls every 16 tokens during generation:

```
GET /mesh/hook/poll/cap-001 → 202         (not ready)
GET /mesh/hook/poll/cap-001 → 200 + json  (ready)
```

On 200, tokens injected into live KV cache. Generation continues conditioned on the new content.

Only runs when `pending_async_ids` is non-empty. No async work = no polling = zero cost.

---

## Signals

Computed per-token in C++ (no callback). Cheap arithmetic on the softmax distribution.

**Entropy**: `H = -Σ p_i × log₂(p_i)`. Low = confident. High = uncertain.

**Margin**: `p_top1 - p_top2`. Large = clear choice. Small = coin flip.

Accumulated in a rolling window (last 16 tokens). Sent at Hook 2 (first token) and Hook 3 (generation summary).

**Self-consistency**: mesh-llm sets `n_cmpl: 3` on the request. llama-server prefills once, generates 3 completions via existing `copy_state_to()`. mesh-llm compares at Hook 3. No new C++ code.

---

## Token injection

**Hook 1**: Text prepended to prompt as system message. Before tokenization — just part of the prompt.

**Hook 2**: Tokens added to live KV cache via `common_batch_add()` + `llama_decode()`.

**Hook 3**: Tokens appended after generated text. With `continue: true`, generation resumes.

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

    // set by Hook 1 response
    float entropy_threshold = -1.0f;  // <0 = skip Hook 2
    bool  verify = false;             // false = skip Hook 3

    // trigger state (computed at pre-inference)
    bool has_images_no_multimodal = false;
    bool context_pressure = false;
    bool long_session = false;
    bool large_user_message = false;

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

| Hook | Where | Fires when |
|---|---|---|
| pre_inference | `launch_slot_with_task()` | Any pre-inference trigger met |
| post_prefill | `DONE_PROMPT → GENERATING` | entropy_threshold set AND exceeded |
| pre_response | `send_final_response()` | verify set OR any pre-response trigger met |
| polling | generation loop | pending_async_ids non-empty |

**Generation loop** (after `common_sampler_accept`):
```cpp
if (slot.mesh_hook.enabled) {
    // update signal stats (always, ~1μs)
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
- Spawns tokio tasks for async consultations
- Stores results keyed by async_id
- Routes consultations through existing mesh infrastructure

**`inference/launch.rs`**: Pass `--mesh-port {api_port}` to llama-server.

**Request preparation**: Set `mesh_hooks: true` when hooks should be active.
