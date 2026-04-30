# Staged Inference Roadmap

Replace `llama-server` + `rpc-server` + MoE expert sharding with a unified
`skippy-server` stage pipeline as the sole inference engine in mesh-llm.

## Decision: In-Process vs Sidecar

**Option A — Pull skippy crates into mesh workspace (in-process):**
- mesh-llm calls `serve_binary()` / `serve_openai()` directly as tokio tasks
- No child processes, no config JSONs on disk, no binary distribution
- Requires: add patched llama.cpp C++ build to mesh build system
- The C++ build already exists in skippy (`prepare-llama.sh` + CMake)
- Eliminates: binary bundling, version management, process lifecycle

**Option B — Keep skippy separate, spawn as sidecar binary:**
- mesh-llm spawns `skippy-server` as child processes (like llama-server today)
- Simpler build (no C++ in mesh), but adds binary distribution + version sync
- Config JSONs written to disk, logs in separate files

**Recommendation:** Option A. The C++ build is already solved in skippy. Moving
the crates into the mesh workspace means one `cargo build` produces everything.
No version drift, no binary distribution, no IPC overhead. The stage loop
becomes just another async task inside mesh-llm.

**Investigation needed:**
- [ ] Can `skippy-ffi/build.rs` work as-is inside the mesh workspace?
- [ ] Does the patched llama.cpp build integrate with mesh CI?
- [ ] Size impact on the mesh-llm binary (llama.cpp adds ~20-30 MB)
- [ ] Can we feature-gate it (`--features staged`) during transition?

---

## Current State

`inference/staged.rs` exists on branch `micn/staged-inference` with:
- ✅ Manifest parsing (`model-package.json`)
- ✅ Topology planning (assign layers proportional to node RAM)
- ✅ Stage server process launch
- ✅ OpenAI driver process launch
- ✅ `start_staged()` — returns an HTTP port, same interface as `start_llama()`
- ✅ Handles both plain GGUF (solo mode) and layer packages (multi-stage)
- ✅ Tested end-to-end locally: 2-stage Qwen3-0.6B, OpenAI HTTP + SSE streaming

## Architecture

```
mesh-llm (orchestrator)
│
├── Detects model type (GGUF or layer-package)
├── Plans topology (which node gets which layers)
├── Launches skippy-server processes
├── Proxies HTTP to the OpenAI driver port
│
├── Solo (plain GGUF, 1 node):
│     └── skippy-server serve-openai (loads full model, serves HTTP)
│
├── Multi-stage (layer package, 1 node):
│     ├── skippy-server serve-binary (stage-0, layers 0..14)
│     ├── skippy-server serve-binary (stage-1, layers 14..28)
│     └── skippy-server serve-openai --first-stage-addr (driver, HTTP)
│
└── Multi-stage (layer package, N nodes):
      ├── Local: skippy-server serve-binary (stage-0)
      ├── Remote (via tunnel): skippy-server serve-binary (stage-1..N)
      └── Local: skippy-server serve-openai --first-stage-addr (driver)
```

The OpenAI driver exposes `/v1/chat/completions` (streaming + non-streaming)
identical to `llama-server`. mesh-llm's proxy routes to it unchanged.

## Roadmap

### 1. Binary Distribution

**Problem:** mesh-llm currently bundles `llama-server` and `rpc-server`.
Need to bundle `skippy-server` instead (or alongside during transition).

**Tasks:**
- [ ] Add `skippy-server` to the binary resolution logic in `inference/launch.rs`
- [ ] Build `skippy-server` for all release targets (macOS arm64, Linux x86_64)
- [ ] Include in the mesh-llm release archive (same as llama-server today)
- [ ] Add version check / compatibility validation

**Files:** `inference/launch.rs`, `system/release_target.rs`, CI/build scripts

### 2. Model Detection

**Problem:** mesh-llm needs to know if a model is a plain GGUF or a layer
package, and route to the appropriate launch path.

**Tasks:**
- [ ] Check for `model-package.json` in the model directory
- [ ] Support `hf://` layer package references in model config
- [ ] Add layer package as a model format in the inventory/catalog system
- [ ] Show layer package info in `mesh-llm models list`

**Files:** `models/local.rs`, `models/inventory.rs`, `models/catalog.rs`,
`inference/staged.rs`

### 3. Wire into Election Loop

**Problem:** The election loop currently calls `start_llama()`. Need to
call `start_staged()` when the model is a layer package (or always, since
it handles plain GGUFs too).

**Tasks:**
- [ ] Add model type detection before the MoE/dense branch
- [ ] Call `start_staged()` for layer packages
- [ ] Optionally: call `start_staged()` for ALL models (replaces llama-server entirely)
- [ ] Handle the returned port + process handles (same lifecycle as llama-server)
- [ ] Publish `InferenceTarget::Local(port)` — proxy works unchanged

**Files:** `inference/election.rs`

### 4. Layer Package Download

**Problem:** When a model is a layer package on HF, nodes need to download
their assigned layers before starting stages.

**Tasks:**
- [ ] Resolve `hf://` package refs to local cache paths
- [ ] Download manifest first (tiny, gives layer count + sizes)
- [ ] Download only the layers assigned to this node
- [ ] Use HF cache (`~/.cache/huggingface/hub/`) for dedup
- [ ] Show download progress in CLI output
- [ ] Cache materialized stage GGUFs persistently (not /tmp)

**Files:** `inference/staged.rs`, new `models/layer_package.rs`

### 5. Multi-Node Coordination

**Problem:** In multi-node mode, the orchestrator needs to tell peers
which layers to download and when to start their stage servers.

**Tasks:**
- [ ] Add a gossip message type: `StageAssignment { layer_start, layer_end, package_ref }`
- [ ] Peers receive assignment → download layers → start stage server → report ready
- [ ] Orchestrator waits for all peers ready before starting the driver
- [ ] Handle peer departure (kill stage, re-plan, restart)
- [ ] Handle peer arrival (re-plan if beneficial, or ignore)

**Files:** `mesh/mod.rs` (gossip), `inference/staged.rs`, `inference/election.rs`

### 6. Tunnel Wiring for Inter-Stage Connections

**Problem:** Stage-0 connects downstream to stage-1 via TCP. When stages
are on different nodes, this TCP connection must go through the iroh QUIC
tunnel.

**Tasks:**
- [ ] Stage config gets `downstream: tcp://127.0.0.1:<tunnel_port>`
- [ ] Tunnel manager relays to peer's stage server port
- [ ] Add `STREAM_STAGE` tunnel type (or reuse existing RPC tunnel)
- [ ] LAN optimization: when peers are directly reachable, bypass tunnel

**Files:** `network/tunnel.rs`, `inference/staged.rs`

### 7. Activation Width Auto-Detection

**Problem:** The stage server needs `--activation-width` which varies by
model (1024 for 0.6B, 4096 for 235B, etc.). Currently hardcoded.

**Tasks:**
- [ ] Read `n_embd` from the manifest metadata or GGUF header
- [ ] Pass it through to stage server launch args
- [ ] Fallback: infer from model name / catalog

**Files:** `inference/staged.rs`

### 8. Remove Old Code Paths

Once staged inference is validated end-to-end in production:

**Delete:**
- [ ] `rpc-server` launch/management in `inference/launch.rs`
- [ ] MoE expert sharding in `inference/moe.rs`
- [ ] Expert analysis/ranking system
- [ ] Dense launch planning (`build_dense_launch_plan`)
- [ ] RPC port rewriting in `network/rewrite.rs`
- [ ] `llama-server` binary bundling (if fully replaced)

**Keep:**
- Mesh networking (iroh, gossip, tunnels)
- OpenAI proxy/routing
- Model catalog/inventory
- CLI
- Crypto/identity

### 9. Materialization Cache

**Problem:** Stage GGUFs are assembled from layer files on first load.
Currently cached in `/tmp` (lost on reboot).

**Tasks:**
- [ ] Move cache to `~/.cache/mesh-llm/stage-materialized/`
- [ ] Key by: model_id + layer_range + manifest_sha256
- [ ] Auto-invalidate on manifest change
- [ ] Add cache size management (LRU eviction)
- [ ] Show cache status in CLI (`mesh-llm models cache`)

**Files:** `inference/staged.rs`, `models/local.rs`

### 10. Speculative Decoding Integration

**Problem:** The stage runtime supports draft model speculation (3× speedup).
mesh-llm should be able to configure this.

**Tasks:**
- [ ] Allow specifying a draft model in mesh-llm config
- [ ] Pass draft config through to the OpenAI driver
- [ ] Auto-detect compatible draft models from inventory

**Files:** `inference/staged.rs`, model config

## File Management Summary

| What | Where | Who manages |
|---|---|---|
| Layer package manifest | `~/.cache/huggingface/hub/models--<repo>/` | HF cache (standard) |
| Layer GGUF files | `~/.cache/huggingface/hub/models--<repo>/` | HF cache (standard) |
| Materialized stage GGUFs | `~/.cache/mesh-llm/stage-materialized/` | mesh-llm cache |
| Stage config JSONs | `/tmp/mesh-llm-staged/<run-id>/` | Ephemeral per-run |
| Stage logs | `/tmp/mesh-llm-staged/<run-id>/` | Ephemeral per-run |
| skippy-server binary | `<mesh-llm-bin-dir>/skippy-server` | Bundled with release |
| Plain GGUF models | `~/.cache/huggingface/hub/` or user path | Existing model system |

## Download Flow

```
User runs: mesh-llm --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers

1. mesh-llm downloads model-package.json (tiny)
2. Reads manifest: 94 layers, ~1.4 GB each
3. Discovers peers, plans topology:
   - This node (256 GB): layers 0..70
   - Peer B (64 GB): layers 70..94
4. Downloads layers 0..70 from HF (~101 GB)
5. Tells peer B via gossip: "download layers 70..94"
6. Peer B downloads layers 70..94 (~35 GB)
7. Both nodes materialize their stage GGUFs (cached)
8. Start stage servers, connect via tunnel
9. Start OpenAI driver
10. Ready — proxy routes requests to driver port
```

## Priority Order

1. **Binary distribution** — can't do anything without the binary
2. **Model detection** — know when to use staged path
3. **Election loop wiring** — actually call start_staged()
4. **Single-node layer package** — prove it works without multi-node complexity
5. **Multi-node coordination** — gossip + download + tunnel
6. **Remove old paths** — once stable
