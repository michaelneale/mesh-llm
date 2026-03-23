# mesh-llm TODO

## Mixture of Models (MoM)

Route different requests to specialized models based on task type. Instead of one "best" model, the mesh becomes smarter about which model handles what.

**Paper:** [Mixture of Models: An Intra-Model Ensemble Approach](https://arxiv.org/pdf/2601.16863)

The paper shows ensemble routing across heterogeneous models outperforms any single model. Our mesh already has the ingredients — multiple models, a router that classifies requests. The gap is making the router model-aware (which models are good at what) and potentially splitting complex requests across models.

**Relates to:** Smart Router (below), Vision routing (vision requests → vision model), Multi-Model Per Host.

## Multi-Model Per Host

Currently each host runs one llama-server serving one model. Hosts with spare VRAM could serve multiple simultaneously.

**Options:**
1. **Multiple llama-server processes** — each on a different port, proxy routes by model. Simple but duplicates KV cache overhead.
2. **llama-server native multi-model** — newer versions support `--model` multiple times. Single process, shared infrastructure.

**Why it matters:**
- Studio (206GB) could serve MiniMax (130GB) + a vision model (20GB)
- Mini (16GB) could serve Qwen3.5-9B (5.5GB) + draft model
- Enables MoM routing across models on the same host

## Peer-to-Peer Model Transfer

Fetch model files directly from mesh peers instead of HuggingFace. Peers already have QUIC connections — add a new stream type where the requester sends a filename and offset, the responder streams the file back.

**Why:** LAN transfers are massively faster than HuggingFace downloads. Two machines on the same network could transfer a 47GB model in minutes instead of an hour. Also works when HF is slow, rate-limited, or down.

**Design:**
- New bi-stream type (`STREAM_FILE_TRANSFER`): requester sends filename + resume offset, responder reads from `~/.models/` and streams back
- Only serve files from `~/.models/` — no path traversal
- Resume support via byte offset
- Prefer low-RTT peers (LAN) over high-RTT (relay)
- Download logic tries peers first, falls back to HuggingFace
- Extend gossip to include filenames on disk so peers know what's fetchable

## SSD Expert Streaming

Run giant MoE models on a single node by streaming active experts from NVMe instead of fitting everything in RAM.

[flash-moe](https://github.com/danveloper/flash-moe) already does this — runs Qwen3.5-397B-A17B at 5.5 tok/s on a 48GB M3 Max with 6GB resident memory. See [ROADMAP.md](../ROADMAP.md).

**Plan:** Use flash-moe as an alternative backend. Mesh-llm spawns it like llama-server. Needs HTTP/SSE endpoint (currently CLI only) and OpenAI-compatible `/v1/chat/completions`.

## MoE Expert Sharding

Design: [MoE_PLAN.md](../MoE_PLAN.md) · Auto-deploy: [MoE_DEPLOY_DESIGN.md](../MoE_DEPLOY_DESIGN.md) · Validation: [MoE_SPLIT_REPORT.md](../MoE_SPLIT_REPORT.md)

- [ ] **Lazy `moe-analyze`** — auto-run ranking for unknown MoE models.
- [ ] **Scale testing** — Mixtral 8×22B, Qwen3-235B-A22B across multi-node.

## Thinking / Reasoning Budget

Currently `--reasoning-format deepseek` is set on all llama-servers, and `--reasoning-budget` defaults to `-1` (unlimited). Thinking models (Qwen3.x, MiniMax-M2.5) will think as long as they want before answering.

**Current state (v0.39.0):**
- Web UI sends `chat_template_kwargs: {"enable_thinking": false}` per-request → no thinking, fast responses
- API passthrough does NOT send this → external users still get thinking by default

**Decision needed: should we turn reasoning off server-wide (`--reasoning-budget 0`)?**

### Benchmark results (thinking ON vs OFF)

| Model | Task | Think ON | Think OFF | Quality diff |
|---|---|---|---|---|
| **Qwen3.5-9B** | Math ($1.10 bat/ball) | 79.4s (all reasoning, no content — burned max_tokens) | 0.9s, correct | OFF much better — ON can't even answer |
| **Qwen3.5-9B** | Code (palindrome) | 79.0s (no content) | 2.3s, correct code | OFF wins |
| **Qwen3.5-9B** | Chat (recipe) | 79.0s (no content) | 71.2s, good recipe | OFF wins — ON burns tokens on reasoning |
| **Qwen3.5-9B** | Reasoning (roses/flowers) | 79.1s (no content) | 8.4s, correct answer | OFF wins |
| **MiniMax-M2.5** | Math | 5.5s, correct | 0.2s, correct | Same quality, 27x faster |
| **MiniMax-M2.5** | Code | 7.4s, correct | 1.0s, correct+docstring | OFF slightly more verbose |
| **MiniMax-M2.5** | Chat (recipe) | 23.5s (TTFC 7.5s) | 10.7s (TTFC 0.1s) | Similar quality, 2x faster |
| **MiniMax-M2.5** | Reasoning | 12.0s, correct "No" | 2.0s, correct "No" | Same quality, 6x faster |
| **MiniMax-M2.5** | Trivia (antechinus) | 22.7s (TTFC 4.4s) | 14.8s (TTFC 0.1s) | Similar quality |
| **Qwen2.5-72B** | All tasks | No thinking (not a thinking model) | Same | No difference — not affected |

**Key findings:**
- **Qwen3.5-9B is broken with thinking ON** — reasoning burns the entire max_tokens budget, content is always empty. This model is unusable with thinking enabled at default max_tokens.
- **MiniMax-M2.5 works with thinking but is 2-27x slower** for no meaningful quality gain on these tasks.
- **Qwen2.5-72B is unaffected** — not a thinking model.
- Per-request control (`chat_template_kwargs: {"enable_thinking": false}`) works perfectly and is what the UI now uses.

**Options:**
1. **Keep current** — UI has no-think, API has thinking. External API users get slow responses by default but can send `chat_template_kwargs: {"enable_thinking": false}` themselves.
2. **Turn off server-wide** (`--reasoning-budget 0`) — everyone gets fast responses. Users who want thinking can't get it.
3. **Default off, opt-in** — Set `--reasoning-budget 0` but document that API users can send `chat_template_kwargs: {"enable_thinking": true}` to get it back. (Need to verify this works with budget=0.)

**Recommendation:** Option 2 or 3. Thinking burns tokens wastefully on the current mesh models and actively breaks Qwen3.5-9B. The quality gain is negligible for chat/code/trivia tasks. For hard math/reasoning where thinking helps, users should use a dedicated reasoning model (DeepSeek-R1) not a general chat model.

## Smart Router
- [ ] **Context-aware routing**: Hosts advertise `n_ctx` in gossip. Router estimates request token count and skips hosts that can't fit it. Today a long chat routed to a small-context host returns 400 with no fallback.
- [ ] **Retry on 400**: If a host returns 400 (context overflow, bad request), try the next host instead of forwarding the error. Requires reading the response status before committing to the byte-pipe tunnel. Non-trivial — the current `relay_tcp_via_quic` is a blind bidirectional copy.
- [ ] **Static speed estimates**: `tok_s: f64` on ModelProfile. Quick tasks prefer fast models.
- [ ] **Response quality checks**: Detect empty/repetitive/truncated responses, retry with different model.
- [ ] **MoM-aware routing**: Route by task type to best-suited model (see Mixture of Models above).
- [ ] **Vision-aware routing**: Auto-route image requests to vision-capable models.

## Resilience
- [ ] **Multi-node tensor split recovery**: If one split peer dies, re-split across remaining.

## Vision — Future
- [ ] **More catalog entries**: Gemma-3-12B, Pixtral-12B, larger Qwen3.5 (35B-A3B MoE, 122B-A10B MoE)
- [ ] **Image generation**: Not supported by llama.cpp (transformers only), but could add diffusion backend later.
