# Roadmap

High-level directions for mesh-llm. Not promises — just things we're thinking about.

## Smart model router ✅ (Phase 1)

Implemented. Heuristic classifier detects Code/Reasoning/Chat/Creative/ToolCall with Quick/Moderate/Deep complexity. Task-dominant scoring ensures the right model handles each request. Tool capability is a hard filter. Multi-model per node with auto packs by VRAM tier.

Next: static speed estimates in model profiles, response quality checks (retry on garbage), complexity-aware token budgets. See [mesh-llm/docs/ROUTER_V2.md](mesh-llm/docs/ROUTER_V2.md) for the full phased plan.

## Mobile chat app (exemplar)

A native mobile app that joins a mesh by scanning a QR code. Client-only — no GPU, no model serving. Just a beautiful chat interface backed by the mesh's GPU pool.

- Scan QR code → join mesh → chat with any model the mesh serves
- Uses iroh relay for connectivity (works through NAT, cellular, WiFi)
- OpenAI-compatible API underneath (same as any mesh client)
- iOS first (Swift + iroh-ffi), Android follow-up
- "AirDrop for AI" — one scan and you're talking to a 235B parameter model

This is the best way to show what mesh-llm does: zero setup, zero config, just scan and chat.

## Connection stability

Relay connections degrade over hours on some nodes (Studio pattern: fresh=250ms, 10h=isolated). Need relay health monitoring, periodic reconnect, and better understanding of iroh's relay lifecycle. See [mesh-llm/TODO.md](mesh-llm/TODO.md) for investigation notes.

## Production relay infrastructure

mesh-llm now uses managed iroh relays via [services.iroh.computer](https://services.iroh.computer) as the default:
- `usw1-2.relay.michaelneale.mesh-llm.iroh.link` (US West)
- `aps1-1.relay.michaelneale.mesh-llm.iroh.link` (Asia-Pacific South)

The old self-hosted Fly.io relay (`tools/relay-fly-legacy/`) is no longer in use. Adding more relay regions may help with the relay decay issue above.

## Agent launcher

`mesh-llm run` as a one-command way to launch AI agents talking to the mesh:

```bash
mesh-llm run goose          # launch goose session with mesh backend
mesh-llm run pi             # launch pi with --provider mesh
mesh-llm run opencode       # opencode pointed at mesh API
```

We already print launch commands when the mesh is ready and show them in the web console. There's also a native Goose provider (`micn/mesh-provider-v2` branch on `block/goose`) with emulated tool calling.

## Single binary distribution

Currently ships as a 3-binary bundle (`mesh-llm` + `llama-server` + `rpc-server`). Could compile llama.cpp directly into the Rust binary via [llama-cpp-2](https://crates.io/crates/llama-cpp-2) — one binary, no bundle.

## MoE expert sharding ✅

Implemented. Auto-detects MoE, computes overlapping expert assignments, splits locally, session-sticky routing. Zero cross-node traffic. See [MoE_PLAN.md](mesh-llm/docs/MoE_PLAN.md).

Remaining: optimized rankings for unknown models, scale testing on Mixtral 8×22B / Qwen3-235B.

## SSD expert streaming

Run MoE models that are far too large for memory on a single node by streaming only the active experts from NVMe SSD per token. The trunk (attention, norms, embeddings) stays resident in memory; expert weights live on disk and are `pread()`'d on demand.

This is a single-node strategy. The goal is running e.g. Qwen3.5-397B-A17B (~209GB at Q4) on a 48GB Mac — no mesh needed.

**Proven by [flash-moe](https://github.com/danveloper/flash-moe):** a from-scratch C/Metal inference engine that runs the full 397B model at 5.5 tok/s on a MacBook Pro M3 Max (48GB) by streaming experts from SSD. Key results:

- 120GB of expert weights at 2-bit quant, streamed via parallel `pread()` (4 threads, one per active expert)
- Only K=4 experts activated per layer per token → ~600MB read from SSD per token
- Apple NVMe delivers 5.5 GB/s sustained random reads (17.5 GB/s sequential)
- Custom Metal compute shaders for 2-bit and 4-bit dequantized matvec
- Pipeline: GPU attention projections → CPU linear attention → GPU routing → SSD expert read → GPU expert forward, all overlapped

**Key lessons from flash-moe that apply here:**

- **Trust the OS page cache.** Every custom expert cache they built (Metal LRU, malloc, tiered I/O) made things worse — wired memory squeezes the OS page cache, triggers compressor thrashing. Deleting the custom cache was a 38% speedup. Same lesson as PostgreSQL's `shared_buffers`: don't take more than 25% of RAM.
- **pread() >> mmap() for expert loading.** mmap triggers 240 individual page faults for a 3.9MB expert (240 × 16KB pages). One `pread()` call issues one NVMe command. 5× faster.
- **2-bit expert quantization preserves quality.** 44% size reduction over 4-bit, RMSE ~0.001. Quality holds across math, code, reasoning. Biggest single throughput win (cuts I/O time per layer from 2.6ms to 1.5ms).
- **Kernel I/O hints are useless or harmful on Apple Silicon.** F_RDADVISE, MADV_RANDOM, MADV_SEQUENTIAL, MADV_WILLNEED — all neutral or negative. The macOS kernel already optimizes for Apple's NVMe controller.
- **2MB-aligned DMA buffers give 3.6× better throughput** for page-cache-resident reads (free optimization via `posix_memalign`).
- **Speculative routing and prefetching don't work.** 65-80% of predictions are wrong, waste bandwidth.

**How this fits mesh-llm:**

Today mesh-llm has two MoE modes: **solo** (model fits in memory, run it whole) and **split** (model doesn't fit, shard experts across nodes). SSD streaming would be a third mode: model doesn't fit in memory but *does* fit on one node's SSD. No mesh coordination, no cross-node traffic, no splitting — just one machine streaming experts from disk.

**Plan:** Use flash-moe directly as an alternative backend, not hack SSD streaming into llama.cpp. llama.cpp's `ggml_mul_mat_id` assumes all expert weights resident in one contiguous tensor — changing that is deep surgery across ggml, the Metal backend, and the model loader. Flash-moe is a working engine. Mesh-llm spawns it like it spawns llama-server — process management + HTTP wrapper.

Only supports Qwen3.5-397B for now (hardcoded architecture). That's fine — it's the model we want to run.

## Blackboard ✅

Implemented. Shared ephemeral text messages across the mesh — agents post status, findings, questions, and answers. Multi-term OR search, convention prefixes (STATUS/QUESTION/FINDING/TIP/DONE), PII auto-scrub, flood-fill propagation with digest sync. Works on any node with or without models. MCP server (`mesh-llm blackboard --mcp`) exposes tools for agent integration. Agent skill installable via `mesh-llm blackboard install-skill`.

## Demand-based rebalancing

Partially done. Unified demand map via gossip, standby nodes promote to serve. Next: large-VRAM hosts auto-upgrade models when demand warrants it.

## Resilience

Done: Nostr re-discovery (v0.26.1), llama-server watchdog (v0.27.0), multi-host load balancing (v0.27.0), API deadlock fix (v0.35.1), VRAM-scaled context (v0.35.1). Next: tensor split recovery when a peer dies, relay health monitoring.
