# CommitLLM Plugin for Mesh LLM

Cryptographic verification of LLM inference receipts within the mesh.

## What it does

This plugin integrates [CommitLLM](https://github.com/lambdaclass/commitllm)'s
verification protocol into mesh-llm as an external plugin. It enables mesh
clients to verify that serving nodes actually ran the claimed model with the
claimed configuration.

The plugin runs the CommitLLM verifier (pure Rust, CPU-only, ~1.3ms per
challenged token) and tracks per-peer trust scores based on verification
history.

## Architecture

```
┌────────────────────────────────────────────────┐
│ Mesh LLM node                                  │
│                                                 │
│  ┌─────────────────┐   ┌────────────────────┐  │
│  │ mesh-llm host   │   │ commitllm plugin   │  │
│  │                 │◄──┤                    │  │
│  │ • proxy         │   │ • verify receipts  │  │
│  │ • gossip        │   │ • hash GGUFs       │  │
│  │ • plugin IPC    │   │ • track trust      │  │
│  │                 │──►│ • MCP tools        │  │
│  └─────────────────┘   └────────────────────┘  │
│          │                       │              │
│          │ mesh gossip           │ MCP          │
│          ▼                       ▼              │
│   other nodes              agent tools          │
└────────────────────────────────────────────────┘
```

## Tools

| Tool | Description |
|------|-------------|
| `commitllm_verify_receipt` | Verify a binary receipt against a verifier key |
| `commitllm_hash_gguf` | Compute SHA-256 model identity hash from a GGUF file |
| `commitllm_peer_trust` | Show trust scores for mesh peers |
| `commitllm_status` | Show plugin status and verification statistics |

## Configuration

Add to `~/.mesh-llm/config.toml`:

```toml
[[plugin]]
name = "commitllm"
command = "/path/to/commitllm-plugin"
enabled = true
```

Or if built from this workspace, the binary will be at:
```
target/release/commitllm-plugin
```

## Channel Protocol

The plugin uses the `commitllm.v1` channel for mesh-wide traffic:

- **`receipt`** — a node shares a receipt for verification
- **`verification_result`** — a node shares its verification result

This allows verification results to propagate across the mesh, building
a shared trust picture without requiring every node to verify every response.

## Building

```bash
cargo build -p commitllm-plugin --release
```

## Dependencies

The plugin pulls `verilm-core` and `verilm-verify` from the CommitLLM
repository as git dependencies. These are lightweight pure Rust crates
with no GPU, Python, or vLLM dependencies.

## Current Status

**Phase 1**: Verifier-side plugin — verify receipts from any CommitLLM-enabled
server, track peer trust, expose MCP tools. No changes to llama.cpp.

**Future phases**:
- Witness-mode capture from llama.cpp (GGUF hash + logit commitment)
- Q8_0 Freivalds verification (the math is already implemented in verilm-core)
- Trust-score-based routing preferences
- Pipeline parallel and MoE receipt stitching
