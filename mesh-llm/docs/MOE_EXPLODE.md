# MoE Expert Sharding: Explode & Assemble

> **Status: Experimental** — the explode/assemble tooling works and is tested, but the end-to-end flow (auto-download from HF → assemble → serve) has not been tested in production yet. The existing `moe-split` path (download full model → split locally) remains the default.

## Overview

MoE models (Qwen3-30B, GLM-5, etc.) have most of their size in expert tensors. When distributing across nodes, each node only needs a **subset** of experts. Instead of every node downloading the full model:

| Approach | Download per node | Requires |
|----------|------------------|----------|
| Full model + moe-split | 17 GB (Qwen3-30B) | Full model on disk |
| Exploded experts | ~7 GB (trunk + 46 experts) | HF repo with exploded files |

## How It Works

### 1. Explode (one-time, offline)

Split a full MoE GGUF into trunk + per-expert files:

```bash
mesh-llm explode ~/.models/Qwen3-30B-A3B-Q4_K_M.gguf -o /tmp/exploded/
```

This produces:
- `trunk.gguf` — attention, norms, embeddings, router gates (expert_count=0)
- `expert-000.gguf` through `expert-127.gguf` — one file per expert

For Qwen3-30B-A3B Q4_K_M:
- Trunk: ~1.0 GB
- Each expert: ~131 MB
- Total: ~17.8 GB (same as original, just split into files)

### 2. Upload to HuggingFace (one-time)

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create Qwen3-30B-A3B-Q4_K_M-exploded --type model
huggingface-cli upload michaelneale/Qwen3-30B-A3B-Q4_K_M-exploded /tmp/exploded/ .
```

### 3. Wire into the catalog

In `download.rs`, set `exploded_repo` on the catalog entry:

```rust
CatalogModel {
    name: "Qwen3-30B-A3B-Q4_K_M",
    exploded_repo: Some("https://huggingface.co/michaelneale/Qwen3-30B-A3B-Q4_K_M-exploded/resolve/main"),
    ...
}
```

### 4. Nodes auto-download (automatic)

When the election loop assigns a MoE model to multiple nodes and the catalog entry has `exploded_repo`:

1. Compute expert assignments with 2× overlap (every expert on ≥2 nodes)
2. Download `trunk.gguf` + assigned `expert-NNN.gguf` files from HF
3. Assemble into a shard GGUF using the Rust assembler
4. Launch llama-server with the shard

If the HF repo doesn't exist or download fails, falls back to the existing path (download full model → moe-split).

## Technical Details

### Expert Assignment

- **Shared core**: top N experts by gate mass, replicated to every node
- **Remaining**: distributed round-robin with configurable overlap (2× for exploded, 1× for local split)
- **N** = `min_experts_per_node` from MoE config (e.g. 46 for Qwen3-30B)

### GGUF Format

Expert tensors in GGUF are 3D: `[hidden, intermediate, n_experts]`. The explode step slices along the expert dimension. Expert files store 2D tensors (the trailing dim=1 is dropped by ggml convention). The assemble step restores the dimension.

Router gate tensors (`ffn_gate_inp`) stay in the trunk at full size. During assembly, only the columns for selected experts are gathered.

### Overlap

With 2× overlap on 3 nodes (Qwen3-30B, 128 experts, min 46):
- Each node gets ~87 experts (~12 GB)
- Every expert on ≥2 nodes (redundancy for node loss)

With 2× overlap on 2 nodes: every expert on both nodes (same as full model).

### VRAM Budget (Qwen3-30B-A3B Q4_K_M)

| Nodes | Experts/node (2× overlap) | Download/node |
|-------|---------------------------|---------------|
| 2 | 128 (all) | 17.8 GB |
| 3 | ~87 | ~12.4 GB |
| 4 | ~64 | ~9.4 GB |
| 8 | ~32 | ~5.2 GB |

## What's Tested

- ✅ Rust explode: byte-identical expert files to C++ implementation
- ✅ Rust assemble: byte-identical shards to C++ implementation  
- ✅ Round-trip: explode → assemble all 128 → matches original (18.6 GB)
- ✅ Partial shards: 4-expert and 64-expert shards produce valid GGUF
- ✅ 64-expert shard loads and runs in llama-server (verified with C++ shard)
- ✅ 46 unit tests covering parse, explode, assemble, overlap, assignments

## What's NOT Tested Yet

- ⬜ End-to-end: node joins → election → HF download → assemble → serve
- ⬜ Multi-node scenario with complementary expert shards
- ⬜ HF repo for Qwen3-30B (not yet created/uploaded)
- ⬜ GLM-5 (256 experts) — needs someone with 256GB machine to test
- ⬜ Inference quality with partial expert shards vs full model

## Fallback

If anything goes wrong with the exploded path, nodes fall back to:
1. Download the full model from HF
2. Run `moe-split` locally to create the shard
3. Serve the shard

This is the same path that exists today and has been tested in production.
