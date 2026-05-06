# Virtual LLM Engine

Callback hooks from llama-server into mesh-llm during inference.

Related: [#183](https://github.com/michaelneale/mesh-llm/issues/183), [#165](https://github.com/michaelneale/mesh-llm/issues/165), [PR #225](https://github.com/michaelneale/mesh-llm/pull/225)

---

## What it does

llama-server detects when it might need help and calls mesh-llm on localhost. mesh-llm consults other models in the mesh and replies with context to inject. The caller sees one seamless response.

Three hook points (Hook 1, Hook 2, Hook 2b), all synchronous — each is a blocking POST to `http://localhost:{mesh_port}/mesh/hook`.

---

## Hooks

### Hook 1: Pre-inference

Before generation starts. Fires when the request has media the model can't handle.

| Trigger | Check |
|---|---|
| `images_no_multimodal` | Request has images but model has no mmproj |
| `audio_no_support` | Request has audio but model can't process it |

When mesh hooks are enabled and the model can't handle images, the content parser strips images to `[image attached]` text (instead of rejecting with a 500 error) and preserves the original image URL as `mesh_image_url` in the messages JSON. The hook detects `mesh_image_url` and fires.

**What llama-server sends:**
```json
{
  "hook": "pre_inference",
  "trigger": "images_no_multimodal",
  "request_id": "chatcmpl-abc",
  "model": "Qwen3-8B-Q4_K_M",
  "n_prompt_tokens": 847,
  "n_ctx": 4096,
  "messages": [...]
}
```

**What mesh-llm does:** Extracts the image URL from `mesh_image_url` in messages, finds a vision-capable peer, sends the image for captioning, returns the caption for injection into the prompt.

**Response:** `{"action": "inject", "text": "[Image description: ...]\n\n"}` or `{"action": "none"}`

### Hook 2: Post-prefill

After prompt evaluation, before first token. Fires when first-token entropy exceeds threshold (default 3.0).

**Trigger:** `entropy > entropy_threshold` (default 3.0, ~8 equally-likely tokens). Also requires `margin < 0.05` (top two tokens close in probability).

In practice, fires on creative/uncertain prompts where the model genuinely doesn't know how to start. Does not fire on thinking models (first token is always `<think>`, very confident) or on confidently wrong models.

**What llama-server sends:**
```json
{
  "hook": "post_prefill",
  "trigger": "high_entropy",
  "model": "gemma-4-E4B-it-Q4_K_M",
  "messages": [...],
  "signals": {
    "first_token_entropy": 3.07,
    "first_token_margin": 0.027
  }
}
```

**What mesh-llm does:** Finds up to 2 peers serving different models, races them for a second opinion, injects the winner's answer with "Reference answer" framing.

**Response:** `{"action": "inject", "text": "\n\nReference answer: ...\n\nUse the reference above to provide an accurate response.\n"}` or `{"action": "none"}`

**How KV injection works:** The inject text is tokenized, added to a temporary batch, and decoded in chunks (same as normal prefill). The KV cache is extended for this slot's sequence. The signal window is reset since the model state has changed.

### Hook 2b: Mid-generation

During token generation. Three independent triggers, any can fire:

| Trigger | Check | What it catches |
|---|---|---|
| `sustained_entropy_spike` | 75% of 16-token window has entropy > 4.0 | Model is lost/generating gibberish |
| `repetition_loop` | 3-gram repeat ratio > 0.18 in last 32 tokens | Degenerate looping (even with low entropy) |
| `surprise_break` | 2+ tokens with z-score > 2.5σ after a calm run | Confident flow that suddenly breaks |

All triggers share a cooldown (32 tokens between fires, 8 in debug mode) and require 12+ tokens generated (skips `<think>` warmup).

The repetition trigger is the most reliable — catches real failures that entropy misses (model confidently repeating itself with entropy 0.06-0.56).

**What llama-server sends:**
```json
{
  "hook": "mid_generation",
  "trigger": "repetition_loop",
  "model": "gemma-4-E4B-it-Q4_K_M",
  "generated_text": "...",
  "n_decoded": 43,
  "messages": [...],
  "signals": {
    "mean_entropy": 0.35,
    "repetition_ratio": 0.20,
    "surprise_break": false,
    "total_tokens": 43
  }
}
```

**Response:** Same as Hook 2 — inject or none. Same KV injection mechanism.

---

## Consultation

When a hook fires, mesh-llm consults other models on the mesh:

- **Peer selection:** Exclude smaller models (no point asking weaker), prefer larger tier, deduplicate by model name. Score: `tier_distance * 1000 + rtt_ms`.
- **Fan-out:** Race up to 2 peers via tokio `JoinSet`. First `Ok` wins, loser aborted.
- **Timeout:** 20s for all hook types.
- **Recursion guard:** Outgoing consultations include `"mesh_hooks": false` in the request body. Prevents peer A consulting peer B which consults peer C.
- **Transport:** `node.open_http_tunnel(peer_id)` — same QUIC path as normal mesh inference.

---

## Injection framing

Tested 5 framings on Qwen3-0.6B (baseline 1/5 correct):

| Framing | Score | Notes |
|---|---|---|
| `reference` | 5/5 | **Production choice.** Clean output, no leaks. |
| `assistant_draft` | 5/5 | Same quality as reference |
| `current` | 5/5 | Leaks `</think>` into content field |
| `rag` | 5/5 | Sometimes echoes question |
| `compact` ("Key fact: X") | 4/5 | Model echoes "Key fact:" — needs instruction to act on hint |

Production framing: `"\n\nReference answer: {hint}\n\nUse the reference above to provide an accurate response.\n"`

---

## Signal detection research

Investigated 5 signals for detecting when a model needs help during generation:

| Signal | Status | Result |
|---|---|---|
| **Entropy** (sustained spike) | Active | Works on weak models, doesn't fire on confident-but-wrong |
| **Repetition** (3-gram ratio) | Active | Best trigger — catches confident looping, zero false positives |
| **Surprise break** (EWMA z-spike) | Active | Never fired yet — needs more data |
| **Rank instability** (top-8 Jaccard) | Removed | 100% false positive on Gemma-4B — top-8 inherently unstable |
| **Margin curvature** (2nd derivative) | Not implemented | Too noisy per GPT-5.4 assessment |

Key insight: entropy doesn't measure "model is about to be wrong." It measures "model doesn't know what token to produce next." A confidently wrong model has low entropy. Repetition detection catches a different failure mode — the model is *confidently* stuck in a loop.

---

## Implementation

### C++ side (llama.cpp fork, `mesh-hooks` branch)

Files modified (all additive to upstream):

| File | Change |
|---|---|
| `server-mesh-hook.h` | **New.** `mesh_hook_ctx`, `mesh_signal_window` (entropy, repetition, surprise), `mesh_compute_entropy()` |
| `server-context.cpp` | Hook calls at 3 points (Hook 1, 2, 2b), KV cache injection, signal computation in generation loop |
| `server-common.cpp` | Image stripping when mesh hooks enabled (instead of 500 error), `mesh_image_url` preservation |
| `server-common.h` | `mesh_hooks` field on `server_chat_params`, `has_media()` accessor |
| `server-task.h` | `mesh_hooks`, `mesh_port`, `mesh_n_turns`, `mesh_messages` on `task_params` |
| `server-task.cpp` | Parse those fields from request JSON |
| `common.h` | `mesh_port`, `mesh_hook_debug` on `common_params` |
| `arg.cpp` | `--mesh-port`, `--mesh-hook-debug` CLI flags |

### Rust side (mesh-llm, `micn/virtual-llm` branch)

| File | Purpose |
|---|---|
| `inference/virtual_llm.rs` | Typed handlers: `handle_image`, `handle_uncertain`, `handle_drift`, `get_peer_hint` |
| `inference/consult.rs` | Peer discovery, fan-out racing, QUIC consultation, recursion guard |
| `api/routes/mesh_hook.rs` | Route handler — parses JSON, dispatches to typed handlers |
| `inference/launch.rs` | Passes `--mesh-port` and `--mesh-hook-debug` to llama-server |

### Port layout

- **9337** (`cli.port`): Proxy — inference requests from users
- **3131** (`cli.console`): Management API — hook callbacks from llama-server, `/api/status`, console
- **random**: Internal llama-server port

Hooks: llama-server → localhost:3131 → Rust handler → QUIC → peer.

### Temporary co-iteration setup

C++ source files are copied into `mesh-llm/llama-patches/` so both C++ and Rust changes live in one repo / one PR. `sync.sh` copies them into `llama.cpp/` before building.

When stable: push C++ to fork's `mesh-hooks` branch, delete `llama-patches/`.

### Debug mode

`MESH_HOOK_DEBUG=1` or `--mesh-hook-debug`: lowers all thresholds, always fires Hook 2, shorter cooldowns. Essential for testing since entropy gating rarely triggers on production models.
