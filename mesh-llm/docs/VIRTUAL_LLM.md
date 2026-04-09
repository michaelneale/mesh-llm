# Virtual LLM Engine

Callback hooks from llama-server into mesh-llm during inference.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## What it does

llama-server detects when it might need help and calls mesh-llm on localhost. mesh-llm consults other models in the mesh and replies with context to inject. The caller sees one seamless response.

Three hooks, all conditional — they only fire when cheap structural checks detect something worth acting on.

### Hook 1: Pre-inference

Before tokenization. Fires when the request has something the model can't handle alone.

| Trigger | Check |
|---|---|
| Images + text-only model | `has_images && mctx == nullptr` |
| Audio + no audio support | `has_audio && !audio_capable` |
| Tool calls + no tool support | `tools` array present, model lacks capability |
| Prompt > 75% of context window | `n_tokens > n_ctx * 0.75` |
| Long session (>10 turns) | `message_count > 10` |
| Large user paste | Last user message > 2000 tokens, prior messages short |

**What mesh-llm gets**:
```json
{
  "hook": "pre_inference",
  "trigger": "context_pressure",
  "request_id": "chatcmpl-abc",
  "model": "qwen3-32b",
  "model_capabilities": ["text", "code"],
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "explain the auth flow"},
    {"role": "assistant", "content": "The auth flow starts with..."},
    {"role": "user", "content": [
      {"type": "text", "text": "What about this code?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
    ]}
  ],
  "has_images": true,
  "has_audio": false,
  "n_prompt_tokens": 14200,
  "n_ctx": 16384,
  "message_count": 24,
  "last_user_message_tokens": 4800
}
```

mesh-llm gets the **full messages array** — all user/assistant/system messages including image and audio data URLs as-is from the original request. Plus token counts, context size, and which trigger fired. This is everything mesh-llm needs to caption images, summarize conversations, or extract context from large pastes.

### Hook 2: Post-prefill

After prompt evaluation, before first token. Fires when the model looks uncertain.

| Trigger | Check |
|---|---|
| High first-token entropy | `entropy > threshold` (threshold set by Hook 1 response) |
| Low first-token margin | `margin < 0.05` |
| Top tokens are hedging | Top-3 are "I", "Sorry", "Well", "Unfortunately" |

**What mesh-llm gets**:
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
      {"text": "I", "prob": 0.08},
      {"text": "The", "prob": 0.06},
      {"text": "Sorry", "prob": 0.05},
      {"text": "Based", "prob": 0.04},
      {"text": "\n", "prob": 0.03}
    ]
  }
}
```

No messages — mesh-llm already has the request from Hook 1, keyed by `request_id`. Just signals: entropy, margin, and top-5 token candidates with probabilities.

### Hook 3: Pre-response

After generation completes, before sending to client. Fires when the output looks problematic.

| Trigger | Check |
|---|---|
| Cut off | `stop_reason == "max_tokens"` |
| Very short response | `n_decoded < 10` with substantial prompt |
| High uncertainty throughout | `uncertain_token_count > 30%` of total |
| Hallucinated ending | Tail entropy (last 16 tokens) >> mean entropy |
| Mid-sentence cutoff | Last token isn't EOS or sentence-ending punctuation |

**What mesh-llm gets**:
```json
{
  "hook": "pre_response",
  "trigger": "max_tokens",
  "request_id": "chatcmpl-abc",
  "generated_text": "The verify_token() function handles session validation by checking the JWT signature against the stored secret. It first decodes the token header to determine the algorithm, then...",
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

mesh-llm gets the **full generated text**, the stop reason, token counts, and signal summary over the whole generation. With the original request (from Hook 1 via `request_id`), mesh-llm has the full picture: what was asked, what was generated, and how confident the model was.

### Polling

Not a hook — runs in the generation loop every 16 tokens when Hook 1 started async work. Checks if the result is ready, injects into KV cache if so.

---

## Callback protocol

All hooks POST to `http://localhost:{mesh_port}/mesh/hook`.

Response actions:
- **none** — continue normally
- **inject** — tokenize `text`, add to prompt or KV cache
- **pending** — started background work, poll `GET /mesh/hook/poll/{async_id}` during generation
- **stop** — halt generation

Hook 1's response also configures hooks 2 and 3:
- `entropy_threshold` — enables Hook 2 (default: Hook 2 skipped)
- `verify` — enables Hook 3 even when no trigger is met (default: Hook 3 only fires on triggers)

---

## Examples: what mesh-llm does

### Image captioning

Hook 1 fires: images present, model is text-only.

mesh-llm finds a vision model in the mesh, sends the image, gets a caption.

**Sync** (fast vision model, local):
```json
→ { "hook": "pre_inference", "trigger": "images_no_multimodal", "messages": [...], "has_images": true }
← { "action": "inject", "text": "[Image: Python fibonacci function with bug on line 3]" }
```
Caption prepended to prompt. TTFT delayed by captioning time.

**Async** (slow/remote vision model):
```json
→ { "hook": "pre_inference", "trigger": "images_no_multimodal", ... }
← { "action": "pending", "async_id": "cap-001", "entropy_threshold": 5.0 }
```
Generation starts immediately. Caption injected mid-generation when poll returns 200.

### Conversation summarization

Hook 1 fires: prompt is 14200 tokens in a 16384 context.

mesh-llm sends the early messages to a fast model: "summarize this conversation so far."

```json
→ { "hook": "pre_inference", "trigger": "context_pressure", "n_prompt_tokens": 14200, "n_ctx": 16384, "message_count": 24 }
← { "action": "inject", "text": "[Conversation summary: user asked about auth flow, then debugging JWT tokens, then refactoring verify_token()...]" }
```

Injected summary replaces early messages in the prompt. Model has room to generate a full response.

### Context pre-fetch for large paste

Hook 1 fires: user pasted 4800 tokens of code, previous messages were ~50 tokens each.

mesh-llm uses a fast model to extract the key parts:

```json
→ { "hook": "pre_inference", "trigger": "large_user_message", "last_user_message_tokens": 4800 }
← { "action": "pending", "async_id": "ctx-001" }
```

Extraction runs async. When ready, injected into KV cache: "Key context from pasted code: the main function calls auth.validate() which..."

### Uncertain model getting help

Hook 2 fires: first-token entropy is 6.8, margin is 0.02.

mesh-llm returns context to help the model start:

```json
→ { "hook": "post_prefill", "trigger": "high_entropy", "signals": { "first_token_entropy": 6.8, "first_token_margin": 0.02, "top_tokens": [{"text": "I", "prob": 0.08}, {"text": "Sorry", "prob": 0.05}] } }
← { "action": "inject", "text": "The relevant code is in auth.rs: verify_token() checks JWT signatures." }
```

Tokens injected into KV cache, re-evaluated. Model starts from a confident state.

### Refusing model nudged

Hook 2 fires: top tokens are all hedging words.

```json
→ { "hook": "post_prefill", "trigger": "top_tokens_hedging", "signals": { "top_tokens": [{"text": "I", "prob": 0.12}, {"text": "Sorry", "prob": 0.08}, {"text": "Unfortunately", "prob": 0.06}] } }
← { "action": "inject", "text": "Answer the question directly based on your knowledge." }
```

### Truncated response continued

Hook 3 fires: generation hit max_tokens.

mesh-llm summarizes the partial response and continues:

```json
→ { "hook": "pre_response", "trigger": "max_tokens", "generated_text": "The verify_token() function handles session validation by...", "n_decoded": 4096, "stop_reason": "max_tokens" }
← { "action": "inject", "text": "\n\n[Continuing...]\n", "continue": true }
```

Or mesh-llm routes to a model with more context. Or returns `none` and lets the truncation through.

### Short refusal retried

Hook 3 fires: model generated 8 tokens for a 847-token prompt.

```json
→ { "hook": "pre_response", "trigger": "very_short", "generated_text": "I don't have access to that information.", "n_decoded": 8 }
← { "action": "inject", "text": "\n\nLet me try again with what I know:\n", "continue": true }
```

Generation resumes from the injection. Model gets a second chance.

### Hallucinated ending caught

Hook 3 fires: entropy spiked in the last 16 tokens.

```json
→ { "hook": "pre_response", "trigger": "tail_entropy_spike", "generated_text": "...returns true. Additionally, it validates the quantum signature of the hyperspace manifold...", "signals": { "mean_entropy": 2.1, "tail_entropy_mean": 6.4 } }
← { "action": "inject", "text": "\n\n(Correction: the function simply returns true after validation.)", "continue": true }
```

Or mesh-llm sends the ending to a verifier model and decides based on the verdict.

---

## Implementation: llama.cpp fork

All changes are in `tools/server/`.

### New struct on `server_slot` (server-context.cpp)

```cpp
struct mesh_hook_ctx {
    bool enabled = false;
    int  port = 0;
    std::string request_id;
    json original_request;

    // configured by Hook 1 response
    float entropy_threshold = -1.0f;  // <0 = skip Hook 2
    bool  verify = false;             // false = Hook 3 only on triggers

    // pre-inference trigger state
    bool has_images_no_multimodal = false;
    bool context_pressure = false;
    bool long_session = false;
    bool large_user_message = false;

    // signal window (updated per-token, no callback)
    mesh_signal_window signals;

    // async poll state
    std::vector<std::string> pending_async_ids;
    int tokens_since_last_poll = 0;
    int poll_interval = 16;

    // reusable HTTP client
    std::unique_ptr<httplib::Client> client;

    bool any_pre_inference_trigger() const {
        return has_images_no_multimodal || context_pressure
            || long_session || large_user_message;
    }

    bool should_poll() {
        if (pending_async_ids.empty()) return false;
        return ++tokens_since_last_poll >= poll_interval;
    }
};
```

### New fields on `task_params` (server-task.h)

```cpp
bool mesh_hooks = false;
int  mesh_port  = 3131;
```

Parsed in `task_params_from_json` alongside existing fields like `cache_prompt`, `n_predict`.

### New slot state (server-context.cpp)

```cpp
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER,
    SLOT_STATE_STARTED,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
    SLOT_STATE_WAITING_MESH,  // paused during sync hook, other slots continue
};
```

### New CLI flag (common/arg.cpp)

```
--mesh-port PORT    Port for mesh hook callbacks (default: disabled)
```

When set, `mesh_hooks` defaults to true for all requests (can be overridden per-request).

### Hook 1: launch_slot_with_task() ~line 1105

After task assignment, before tokenization. Evaluate triggers from the parsed request JSON.

```cpp
if (slot.mesh_hook.enabled && slot.mesh_hook.any_pre_inference_trigger()) {
    slot.state = SLOT_STATE_WAITING_MESH;

    json payload = {
        {"hook", "pre_inference"},
        {"trigger", /* first matched trigger name */},
        {"request_id", slot.mesh_hook.request_id},
        {"model", params_base.model.alias.empty() ? params_base.model.path : *params_base.model.alias.begin()},
        {"messages", slot.mesh_hook.original_request["messages"]},
        {"has_images", slot.mesh_hook.has_images_no_multimodal},
        {"n_prompt_tokens", task.n_tokens()},
        {"n_ctx", slot.n_ctx},
        {"message_count", slot.mesh_hook.original_request["messages"].size()}
    };

    auto res = slot.mesh_hook.client->Post("/mesh/hook", payload.dump(), "application/json");

    if (res && res->status == 200) {
        auto body = json::parse(res->body);
        auto action = body.value("action", "none");

        if (action == "inject") {
            // prepend text to prompt before tokenization
            // ... modify task messages or prompt ...
        } else if (action == "pending") {
            slot.mesh_hook.pending_async_ids.push_back(body["async_id"]);
        }

        // configure hooks 2 and 3
        if (body.contains("entropy_threshold")) {
            slot.mesh_hook.entropy_threshold = body["entropy_threshold"];
        }
        if (body.value("verify", false)) {
            slot.mesh_hook.verify = true;
        }
    }

    slot.state = SLOT_STATE_STARTED;
}
```

### Hook 2: DONE_PROMPT → GENERATING transition ~line 2850

After prompt evaluation, before first token is sampled.

```cpp
if (slot.state == SLOT_STATE_DONE_PROMPT) {
    slot.state = SLOT_STATE_GENERATING;

    // mesh hook 2: post-prefill check
    if (slot.mesh_hook.enabled && slot.mesh_hook.entropy_threshold >= 0) {
        auto probs = get_token_probabilities(ctx, slot.i_batch - i);
        float entropy = compute_entropy(probs);
        float margin = probs.size() >= 2 ? probs[0].p - probs[1].p : 1.0f;

        bool trigger = entropy > slot.mesh_hook.entropy_threshold
                    || margin < 0.05f;
        // also check for hedging tokens in top-3
        // ...

        if (trigger) {
            slot.state = SLOT_STATE_WAITING_MESH;

            json payload = {
                {"hook", "post_prefill"},
                {"request_id", slot.mesh_hook.request_id},
                {"n_prompt_tokens", slot.prompt.n_tokens()},
                {"signals", {
                    {"first_token_entropy", entropy},
                    {"first_token_margin", margin},
                    {"top_tokens", /* top 5 from probs */}
                }}
            };

            auto res = slot.mesh_hook.client->Post("/mesh/hook", payload.dump(), "application/json");
            // handle inject / none
            slot.state = SLOT_STATE_GENERATING;
        }
    }
}
```

### Generation loop: signal update + async poll ~line 2890

After `common_sampler_accept`, every token:

```cpp
if (slot.mesh_hook.enabled) {
    auto probs = get_token_probabilities(ctx, tok_idx);
    slot.mesh_hook.signals.push(compute_entropy(probs), probs[0].p - probs[1].p);

    if (slot.mesh_hook.should_poll()) {
        slot.mesh_hook.tokens_since_last_poll = 0;
        for (auto it = slot.mesh_hook.pending_async_ids.begin();
             it != slot.mesh_hook.pending_async_ids.end(); ) {
            auto res = slot.mesh_hook.client->Get("/mesh/hook/poll/" + *it);
            if (res && res->status == 200) {
                auto body = json::parse(res->body);
                if (body.value("action", "") == "inject") {
                    inject_tokens(slot, body["text"].get<std::string>());
                }
                it = slot.mesh_hook.pending_async_ids.erase(it);
            } else {
                ++it;
            }
        }
    }
}
```

### Hook 3: send_final_response() ~line 2900

Before pushing the result to the HTTP response queue.

```cpp
if (slot.mesh_hook.enabled) {
    bool trigger = slot.mesh_hook.verify
        || slot.stop == STOP_TYPE_LIMIT                          // max_tokens
        || (slot.n_decoded < 10 && slot.prompt.n_tokens() > 100) // very short
        || slot.mesh_hook.signals.uncertain_ratio() > 0.3f       // high uncertainty
        || slot.mesh_hook.signals.tail_entropy_spike()            // bad ending
        || (!slot.has_new_line && slot.stop != STOP_TYPE_EOS);   // mid-sentence

    if (trigger) {
        slot.state = SLOT_STATE_WAITING_MESH;

        json payload = {
            {"hook", "pre_response"},
            {"trigger", /* first matched trigger name */},
            {"request_id", slot.mesh_hook.request_id},
            {"generated_text", slot.generated_text},
            {"n_decoded", slot.n_decoded},
            {"stop_reason", /* "eos" | "max_tokens" | "stop_word" */},
            {"signals", {
                {"mean_entropy", slot.mesh_hook.signals.entropy_mean},
                {"max_entropy", slot.mesh_hook.signals.entropy_max},
                {"min_margin", slot.mesh_hook.signals.margin_min},
                {"uncertain_token_count", slot.mesh_hook.signals.uncertain_count},
                {"tail_entropy_mean", slot.mesh_hook.signals.tail_entropy_mean()}
            }}
        };

        auto res = slot.mesh_hook.client->Post("/mesh/hook", payload.dump(), "application/json");

        if (res && res->status == 200) {
            auto body = json::parse(res->body);
            if (body.value("action", "") == "inject") {
                inject_tokens(slot, body["text"].get<std::string>());
                if (body.value("continue", false)) {
                    slot.has_next_token = true;
                    slot.state = SLOT_STATE_GENERATING;
                    return;  // back to generation loop
                }
            }
        }

        slot.state = SLOT_STATE_GENERATING;
    }
}
// ... send response as normal ...
```

### Token injection helper

```cpp
void inject_tokens(server_slot & slot, const std::string & text) {
    auto tokens = common_tokenize(vocab, text, false, true);
    for (auto tok : tokens) {
        common_batch_add(batch, tok, slot.prompt.tokens.pos_next(), { slot.id }, true);
        slot.prompt.tokens.push_back(tok);
    }
    llama_decode(ctx, batch);
    common_batch_clear(batch);
    slot.mesh_hook.signals.reset();  // fresh start after injection
}
```

### Signal window

```cpp
struct mesh_signal_window {
    static constexpr int SIZE = 16;
    float entropy[SIZE] = {};
    float margin[SIZE] = {};
    int   pos = 0;
    int   count = 0;

    float entropy_mean = 0;
    float entropy_max  = 0;
    float margin_min   = 1;
    int   uncertain_count = 0;  // total tokens with entropy > 4.0

    void push(float e, float m) {
        entropy[pos] = e;
        margin[pos] = m;
        pos = (pos + 1) % SIZE;
        count++;
        // update running stats
        entropy_max = std::max(entropy_max, e);
        margin_min = std::min(margin_min, m);
        entropy_mean = ((entropy_mean * (count - 1)) + e) / count;
        if (e > 4.0f) uncertain_count++;
    }

    float uncertain_ratio() const { return count > 0 ? (float)uncertain_count / count : 0; }

    float tail_entropy_mean() const {
        float sum = 0;
        int n = std::min(count, SIZE);
        for (int i = 0; i < n; i++) sum += entropy[i];
        return n > 0 ? sum / n : 0;
    }

    bool tail_entropy_spike() const {
        return count > SIZE && tail_entropy_mean() > entropy_mean * 2.0f;
    }

    void reset() {
        pos = 0; count = 0;
        entropy_mean = 0; entropy_max = 0; margin_min = 1; uncertain_count = 0;
    }
};
```

---

## Implementation: mesh-llm

### New endpoints on management API (port 3131)

`POST /mesh/hook` — receives hook callbacks. Routes to `inference/virtual.rs` decision logic.

`GET /mesh/hook/poll/{async_id}` — returns 202 (not ready) or 200 with inject action.

### New module: `inference/virtual.rs`

Decision logic per hook type:
- **pre_inference**: detect what's needed, choose sync vs async, pick which model in the mesh to consult, construct the consultation prompt
- **post_prefill**: read signals, decide whether to inject context (fast — no model consultation, just logic)
- **pre_response**: check triggers, optionally send to verifier model, decide inject+continue vs none

Async consultations are tokio tasks. Results stored in a `DashMap<String, Option<String>>` keyed by async_id. Poll endpoint checks the map.

### `inference/launch.rs`

Pass `--mesh-port {api_port}` to llama-server when spawning.

### Request preparation

mesh-llm sets `mesh_hooks: true` and `mesh_port: 3131` in the request body when forwarding to llama-server. Can be set always, or selectively based on what mesh-llm knows about the request and available models.
