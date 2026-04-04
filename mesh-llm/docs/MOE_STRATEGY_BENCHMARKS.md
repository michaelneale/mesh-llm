# MoE Strategy Benchmarks

This document summarizes the offline MoE strategy benchmark suite.

Models tested on `studio54.local`:

- `GLM-4.7-Flash-Q4_K_M`
- `Qwen3-Coder-Next-Q4_K_M`

The suite compares three questions:

1. How much does ranking quality matter for expert placement?
2. Can a short `micro-analyze` replace a full analyze pass?
3. Which grouping shape works best with a good ranking?

## Bottom Line

- Gold standard: full `llama-moe-analyze`
- Best practical default: `micro-analyze` with `--all-layers` (the `auto` path)
- Sequential fallback is acceptable when analysis isn't possible

Weight-only heuristics (gate norm scoring) were evaluated and removed — they produce rankings no better than random. Expert popularity is an emergent property of the full forward pass; static weight analysis cannot recover the signal.

## Ranking Results

### GLM-4.7-Flash-Q4_K_M

| Strategy | Spearman | Recall@24 | Weighted recall@24 | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `analyze` | `1.00` | `1.00` | `1.00` | `44.27s` |
| `micro-1p-8t-all-layers` | `1.00` | `1.00` | `1.00` | `17.29s` |

### Qwen3-Coder-Next-Q4_K_M

| Strategy | Spearman | Recall@256 | Weighted recall@256 | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `analyze` | `1.00` | `1.00` | `1.00` | `106.74s` |
| `micro-1p-8t-all-layers` | `0.951` | `0.930` | `0.966` | `32.09s` |
| `micro-4p-32t-all-layers` | `1.00` | `1.00` | `1.00` | `314.95s` |

## Grouping Results

### GLM-4.7-Flash-Q4_K_M

| Grouping strategy | Ranking source | Shared mass | Mean node mass | Imbalance |
| --- | --- | ---: | ---: | ---: |
| `current-analyze` | `analyze` | `52.90%` | `76.45%` | `0.0808%` |
| `snake-analyze-replicated` | `analyze` | `52.90%` | `76.45%` | `0.0338%` |
| `current-sequential` | `sequential` | `51.03%` | `75.52%` | lower risk fallback |

### Qwen3-Coder-Next-Q4_K_M

| Grouping strategy | Ranking source | Shared mass | Mean node mass | Imbalance |
| --- | --- | ---: | ---: | ---: |
| `current-analyze` | `analyze` | `71.01%` | `85.50%` | `0.0271%` |
| `snake-analyze-replicated` | `analyze` | `71.01%` | `85.50%` | `0.00819%` |
| `current-sequential` | `sequential` | `66.61%` | `83.30%` | lower risk fallback |

Practical interpretation:

- Ranking quality matters more than grouping shape.
- `snake-draft` is worth testing live, but only when paired with a good ranking source.

## Analysis Cost

Startup cost by strategy:

| Strategy | Work done at startup | Measured cost |
| --- | --- | ---: |
| `bundled / cached analyze` | Local config or CSV read only | file read only |
| `micro-analyze` | Short `llama-moe-analyze` run | model-dependent |
| `analyze` | Full `llama-moe-analyze` run | model-dependent |

### Measured analyze timings

Timed on `studio54.local` with:

```bash
/usr/bin/time -lp ./llama-moe-analyze -m MODEL --all-layers --export-ranking /tmp/ranking.csv -n 32 -c 4096 -ngl 99
```

| Model | Full analyze | Micro analyze (`1p/8t/all-layers`) | Notes |
| --- | ---: | ---: | --- |
| `GLM-4.7-Flash-Q4_K_M` | `44.27s` | `17.29s` | micro matched full analyze exactly |
| `Qwen3-Coder-Next-Q4_K_M` | `106.74s` | `32.09s` | micro was already close; larger micro run reached exact match |

## Ranking Strategy

The `auto` default path:

1. Use cached or peer-shared ranking if available
2. Run `micro-analyze` if the model fits locally
3. Fall back to sequential `[0, 1, 2, ..., N]`

`analyze` and `micro-analyze` can be requested explicitly via `--moe-ranking`.

## Benchmark Commands

Import a small fixed corpus:

```bash
mesh-llm benchmark import-prompts \
  --source mt-bench \
  --limit 8 \
  --max-tokens 256 \
  --output evals/moe/prompts/mt-bench-8.jsonl
```

Run the full offline suite:

```bash
mesh-llm benchmark moe-model-matrix \
  --model /Volumes/External/models/GLM-4.7-Flash-Q4_K_M.gguf \
  --model /Volumes/External/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf \
  --nodes 2 \
  --prompts evals/moe/prompts/mt-bench-8.jsonl \
  --output /tmp/moe-model-matrix.json
```

Run individual slices:

```bash
mesh-llm benchmark moe-grouping --model /path/to/model.gguf --nodes 2
mesh-llm benchmark moe-micro-analyze --model /path/to/model.gguf --prompts evals/moe/prompts/mt-bench-8.jsonl
```

## Live Runtime Examples

Full analyze before split:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking analyze \
  --moe-grouping shared-core
```

Micro analyze before split:

```bash
mesh-llm --model /path/to/model.gguf --split \
  --moe-ranking micro-analyze \
  --moe-micro-prompt-count 1 \
  --moe-micro-tokens 8 \
  --moe-micro-layers all \
  --moe-grouping shared-core
```

## Why Heuristics Were Removed

Weight-only heuristic ranking (scoring experts by gate weight norms) was extensively benchmarked and removed. Key findings:

- Best heuristic achieved Spearman correlation of 0.042 and Recall@24 of 0.417 on GLM-4.7-Flash — essentially random
- Expert 0 carries 35.6% of all gate mass at runtime but has completely average weight norms — the signal is invisible to static analysis
- Over 10 alternative approaches were tested (centrality, PCA, bias-weighting, entropy, kurtosis, SVD, random-input simulation) — best achievable was Spearman 0.386
- Expert popularity is determined by the hidden state distribution flowing through the model, not by gate weight properties
- Since expert outputs at layer N feed into the hidden state at layer N+1's router, there is no shortcut that avoids running the full forward pass
