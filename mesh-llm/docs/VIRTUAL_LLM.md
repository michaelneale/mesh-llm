# Virtual LLM Engine

Callback hooks from llama-server into mesh-llm during inference.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165)

---

## What it does

llama-server detects when it might need help and calls mesh-llm on localhost. mesh-llm consults other models in the mesh and replies with context to inject. The caller sees one seamless response.

Three hooks, all synchronous — each is a blocking POST to `http://localhost:{mesh_port}/mesh/hook`. mesh-llm may start background work on Hook 1 and collect results on Hook 3.

---

## Hooks

### Hook 1: Pre-inference

Before generation starts. Fires when the request has something the model can't handle alone.

| Trigger | Check |
|---|---|
| Images + text-only model | Request has media but model has no mmproj |
| Context pressure | Prompt tokens > 75% of context window |
| Long session | Message count > 10 turns |
| Large user paste | Last user message much larger than prior messages |

**What llama-server sends:**
```json
{
  "hook": "pre_inference",
  "trigger": "images_no_multimodal",
  "request_id": "chatcmpl-abc",
  "mesh_request_id": "req-7f3a",
  "model": "Qwen3-8B-Q4_K_M",
  "n_prompt_tokens": 847,
  "n_ctx": 4096
}
```

**What mesh-llm does:** Looks up the original request body by `mesh_request_id` (stored when the proxy forwarded the request). Has full access to messages, images, audio — everything the caller sent.

**Response options:**
- `{"action": "inject", "text": "[Image: a cat on a laptop]"}` — text tokenized and appended to prompt
- `{"action": "none", "entropy_threshold": 5.0, "verify": true}` — skip injection, enable Hook 2 and Hook 3

### Hook 2: Post-prefill

After prompt evaluation, before first token. Fires when the model looks uncertain about how to start.

| Trigger | Check |
|---|---|
| High first-token entropy | `entropy > threshold` (threshold set by Hook 1 response) |
| Low first-token margin | Top-1 minus top-2 probability < 0.05 |

**What llama-server sends:**
```json
{
  "hook": "post_prefill",
  "trigger": "high_entropy",
  "request_id": "chatcmpl-abc",
  "mesh_request_id": "req-7f3a",
  "model": "Qwen3-8B-Q4_K_M",
  "n_prompt_tokens": 847,
  "signals": {
    "first_token_entropy": 6.8,
    "first_token_margin": 0.02,
    "top_tokens": [
      {"text": "I", "prob": 0.08},
      {"text": "The", "prob": 0.06},
      {"text": "Sorry", "prob": 0.05}
    ]
  }
}
```

Always armed with a default `entropy_threshold` of 5.0. Hook 1 can override the threshold if it fires (e.g. lower it for specific request types).

**Response options:**
- `{"action": "inject", "text": "The relevant code is in auth.rs: ..."}` — text is tokenized and decoded into the KV cache. The model processes the injected context as if it were part of the original prompt, then generates from that informed state. This is the key mechanism: a small uncertain model receives context from a stronger model and generates a correct response.
- `{"action": "none"}` — let the model generate from its uncertain state.

**How injection works:** The inject text is tokenized, added to a temporary batch, and decoded in chunks (same as normal prefill). The KV cache is extended for this slot's sequence. After injection, `slot.i_batch` is updated so sampling reads logits from the last injected token. The signal window is reset since the model state has changed.

### Hook 3: Pre-response

After generation, before response is sent. Fires when the output looks problematic.

| Trigger | Check |
|---|---|
| `very_short` | < 10 tokens generated for a substantial prompt |
| `high_uncertainty` | > 30% of tokens had entropy above 4.0 |
| `tail_entropy_spike` | Last 16 tokens had 2x the mean entropy |
| `verify` | Hook 1 explicitly requested verification |

Hook 3 does **not** fire on `max_tokens` — hitting the token limit is normal operation, and `finish_reason: "length"` already signals it to the caller.

**What llama-server sends:**
```json
{
  "hook": "pre_response",
  "trigger": "max_tokens",
  "request_id": "chatcmpl-abc",
  "mesh_request_id": "req-7f3a",
  "model": "Qwen3-8B-Q4_K_M",
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

**Key:** `generated_text` is the only data that only C++ has — the actual model output. mesh-llm combines this with the original request (looked up by `mesh_request_id`) for verification or augmentation.

**Response options:**
- `{"action": "inject", "text": "\n\n[Correction: ...]"}` — appended to response (e.g. after verification)
- `{"action": "none"}` — let it through as-is

---

## Request correlation

mesh-llm generates a `mesh_request_id` and includes it in the request body when forwarding to llama-server. llama-server passes it back in every hook payload.

mesh-llm stores the original request body (messages, images, etc.) keyed by this ID. When a hook fires, mesh-llm looks up the original request — it has the full conversation, images, audio, everything the caller sent.

Entries are cleaned up after Hook 3 fires, or after a timeout.

---

## Background work pattern

All hooks are synchronous from C++'s perspective — each is a blocking POST.

For work that takes longer than a hook should block (e.g. verifying with a strong model), mesh-llm uses this pattern:

1. **Hook 1** starts a tokio background task, stores a handle in a DashMap keyed by `mesh_request_id`
2. Returns `{"action": "none"}` immediately (doesn't block)
3. Generation proceeds normally (seconds)
4. **Hook 3** fires — checks the DashMap. If the background task finished, uses the result. If not, lets the response go as-is.

No polling. Hook 3 is the natural collection point because it always fires before the response leaves.

---

## Scenarios

### Image captioning

User sends an image to a text-only model.

1. Proxy stores the request (with base64 image) keyed by `mesh_request_id`
2. Hook 1 fires: `images_no_multimodal`
3. mesh-llm looks up the request, finds the image
4. Sends the image to a vision model in the mesh: "Describe this image concisely"
5. Returns `{"action": "inject", "text": "[Image: Python fibonacci function with bug on line 3]"}`
6. Caption injected into prompt. Text-only model answers about the image.

Caller has no idea the model can't see images — they just get a useful answer.

### Context summarization

Prompt is 3500 tokens in a 4096 context.

1. Hook 1 fires: `context_pressure`
2. mesh-llm looks up the request, sees a long conversation history
3. Sends early turns to a fast model: "Summarize this conversation"
4. Returns `{"action": "inject", "text": "[Summary: user asked about auth, then JWT debugging...]"}`
5. Summary injected. Model has room for a full response.

### Uncertain model gets live help (Hook 2)

User asks a factual question. Small model finishes prefill but is uncertain how to start.

1. Prefill completes. First-token entropy is 6.8, margin is 0.02 — exceeds default threshold of 5.0, Hook 2 fires.
3. mesh-llm sees the model is uncertain, looks up the original request, sends it to a stronger model in the mesh.
4. Stronger model returns a concise answer: "The auth flow uses JWT tokens validated in auth.rs."
5. mesh-llm returns `{"action": "inject", "text": "The auth flow uses JWT tokens validated in auth.rs."}`.
6. llama-server tokenizes the inject text, decodes it into the KV cache — the small model "reads" this context.
7. Small model now generates confidently from the injected context, producing a correct detailed response.

The caller sees a fast, correct response from the small model. They don't know a stronger model helped.

### Uncertain model — background verification

Model is uncertain about a factual question. Instead of blocking at Hook 2, use async verification.

1. Hook 1 fires (some trigger). mesh-llm starts a background task: send the same prompt to a stronger model.
2. Returns `{"action": "none", "verify": true}` — fast, doesn't block.
3. Small model generates its response (3-5 seconds).
4. Hook 3 fires: `verify`. mesh-llm checks the DashMap — background task finished.
5. Compares the two responses. If they agree, lets it through. If they disagree, appends a note.

### Hallucination detection

Entropy spiked at the end of generation.

1. Hook 3 fires: `tail_entropy_spike`
2. mesh-llm sees mean_entropy=2.1 but tail=5.8
3. Sends the response to a verifier model: "Is this last part accurate?"
4. Based on verdict, either lets it through or truncates at the spike point.

---

## Implementation

### C++ side (llama.cpp fork, `mesh-hooks` branch)

Files modified (all additive, zero deletions from upstream):

| File | Change |
|---|---|
| `server-mesh-hook.h` | **New.** `mesh_hook_ctx`, `mesh_signal_window`, `mesh_compute_entropy()` |
| `server-context.cpp` | Hook calls at 3 insertion points, KV cache injection on Hook 2, signal computation in generation loop |
| `server-task.h` | `mesh_hooks`, `mesh_port`, `mesh_n_turns`, `mesh_request_id` on `task_params` |
| `server-task.cpp` | Parse those fields from request JSON |
| `server-common.h` | `has_media()` accessor on `server_tokens` |
| `common.h` | `mesh_port` on `common_params` |
| `arg.cpp` | `--mesh-port` CLI flag |

### Rust side (mesh-llm, `micn/virtual-llm` branch)

| File | Purpose |
|---|---|
| `inference/virtual_llm.rs` | Decision engine — handler per hook type, model-aware context |
| `api/routes/mesh_hook.rs` | Route handler — parses payload, delegates to virtual_llm |
| `api/routes/mod.rs` | Route dispatch for `POST /mesh/hook` |
| `inference/launch.rs` | Passes `--mesh-port` to llama-server |
| `runtime/mod.rs` | Exports API port as env var |

### Temporary co-iteration setup

During development, C++ source files are copied into `mesh-llm/llama-patches/` so both C++ and Rust changes live in one repo / one PR. `scripts/build-mac.sh` syncs them into `llama.cpp/` before building.

When stable: push C++ to fork's `mesh-hooks` branch, delete `llama-patches/`, remove sync from build script.
