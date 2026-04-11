# mesh-llm TODO

## Mixture of Models (MoM)

Route different requests to specialized models based on task type. Instead of one "best" model, the mesh becomes smarter about which model handles what.

**Paper:** [Mixture of Models: An Intra-Model Ensemble Approach](https://arxiv.org/pdf/2601.16863)

The paper shows ensemble routing across heterogeneous models outperforms any single model. Our mesh already has the ingredients — multiple models, a router that classifies requests. The gap is making the router model-aware (which models are good at what) and potentially splitting complex requests across models.

**Relates to:** Smart Router (below), Multi-Model Per Host.

## Node Owner Identity

Design: [NODE_OWNER_IDENTITY.md](docs/NODE_OWNER_IDENTITY.md)

- [x] Add non-breaking owner attestation for node identities.
- [x] Surface verified owner state in gossip, `/api/status`, and the console.
- [x] Add optional trust policy and owner allowlists for private meshes.

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
- New bi-stream type (`STREAM_FILE_TRANSFER`): requester sends filename + resume offset, responder reads from the Hugging Face cache and streams back
- Only serve files from the managed Hugging Face cache — no path traversal
- Resume support via byte offset
- Prefer low-RTT peers (LAN) over high-RTT (relay)
- Download logic tries peers first, falls back to HuggingFace
- Extend gossip to include filenames on disk so peers know what's fetchable

## SSD Expert Streaming

Run giant MoE models on a single node by streaming active experts from NVMe instead of fitting everything in RAM.

[flash-moe](https://github.com/danveloper/flash-moe) already does this — runs Qwen3.5-397B-A17B at 5.5 tok/s on a 48GB M3 Max with 6GB resident memory. See [ROADMAP.md](../ROADMAP.md).

**Plan:** Use flash-moe as an alternative backend. Mesh-llm spawns it like llama-server. Needs HTTP/SSE endpoint (currently CLI only) and OpenAI-compatible `/v1/chat/completions`.

## MoE Expert Sharding

Design: [MoE_PLAN.md](docs/MoE_PLAN.md) · Auto-deploy: [MoE_DEPLOY_DESIGN.md](docs/MoE_DEPLOY_DESIGN.md) · Validation: [MoE_SPLIT_REPORT.md](docs/MoE_SPLIT_REPORT.md)

- [ ] **Lazy `moe-analyze`** — auto-run ranking for unknown MoE models.
- [ ] **Scale testing** — Mixtral 8×22B, Qwen3-235B-A22B across multi-node.

## Smart Router
- [ ] **Context-aware routing**: Hosts advertise `n_ctx` in gossip. Router estimates request token count and skips hosts that can't fit it. Today a long chat routed to a small-context host returns 400 with no fallback.
- [ ] **Retry on 400**: If a host returns 400 (context overflow, bad request), try the next host instead of forwarding the error. Requires reading the response status before committing to the byte-pipe tunnel. Non-trivial — the current `relay_tcp_via_quic` is a blind bidirectional copy.
- [ ] **Static speed estimates**: `tok_s: f64` on ModelProfile. Quick tasks prefer fast models.
- [ ] **Response quality checks**: Detect empty/repetitive/truncated responses, retry with different model.
- [ ] **MoM-aware routing**: Route by task type to best-suited model (see Mixture of Models above).

## Multi-Modal — Remaining

Core multimodal is shipped: capability model, gossip advertisement, vision/audio-aware routing, blob plugin, console uploads with attachment state, multimodal `/v1/chat/completions` and `/v1/responses`. See [MULTI_MODAL.md](docs/MULTI_MODAL.md).

- [ ] **Runtime vision validation**: When mmproj is missing at launch time, downgrade vision capability and re-gossip corrected descriptor. Today the node advertises `vision: supported` even when mmproj wasn't loaded.
- [ ] **Audio transcription shim**: Optional `/v1/audio/transcriptions` compatibility layer.
- [ ] **Realtime shim**: Optional `v1/realtime` compatibility layer for text and media session orchestration.

## Resilience
- [ ] **Multi-node tensor split recovery**: If one split peer dies, re-split across remaining.

## Vision — Future
- [ ] **More catalog entries**: Gemma-3-12B, Pixtral-12B, larger Qwen3.5 (35B-A3B MoE, 122B-A10B MoE)
- [ ] **Image generation**: Not supported by llama.cpp (transformers only), but could add diffusion backend later.
