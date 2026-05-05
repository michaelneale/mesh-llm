---
name: skippy-cache-family-bench
description: Use this skill when benchmarking Skippy exact-prefix cache across model families, comparing Skippy against llama-server, producing README benchmark tables, updating crates/skippy-cache/README.md evidence, or diagnosing cache benchmark gaps by family or Hugging Face use case.
metadata:
  short-description: Benchmark Skippy cache by family
---

# skippy-cache-family-bench

Use this skill for reproducible Skippy cache benchmark evidence. The goal is to
compare production cache payloads only: `ResidentKv` for dense families and
`KvRecurrent` for recurrent/hybrid families. Do not report `FullState` as a
production cache mode.

## Workflow

1. Run the reproducible wrapper. It builds by default, then runs full-GGUF
   baselines, Hugging Face use-case prompts, and the README report renderer:

   ```bash
   evals/skippy-cache-family-bench.sh /tmp/skippy-cache-family-bench
   ```

2. For fast iteration after a build, skip the build step:

   ```bash
   SKIPPY_CACHE_SKIP_BUILD=1 evals/skippy-cache-family-bench.sh /tmp/skippy-cache-family-bench
   ```

3. If running the pieces manually, keep these matched settings for full-GGUF
   family baselines:

   ```bash
   LLAMA_STAGE_BUILD_DIR=.deps/llama-build/build-stage-abi-cpu \
     python3 evals/skippy-cache-production-bench.py \
       --output-dir /tmp/skippy-cache-family-bench/full-gguf \
       --runtime-lane-count 1 \
       --llama-parallel 1 \
       --prefix-tokens 128
   ```

4. Run the use-case benchmark matrix against the same family set:

   ```bash
   LLAMA_STAGE_BUILD_DIR=.deps/llama-build/build-stage-abi-cpu \
     python3 evals/skippy-cache-production-bench.py \
       --output-dir /tmp/skippy-cache-family-bench/use-cases \
       --runtime-lane-count 1 \
       --llama-parallel 1 \
       --prefix-tokens 128 \
       --use-case all
   ```

5. Render README-ready tables from the combined JSON outputs:

   ```bash
   python3 evals/skippy-cache-family-report.py \
     --input /tmp/skippy-cache-family-bench/full-gguf/production-cache-bench.json \
     --input /tmp/skippy-cache-family-bench/use-cases/production-cache-bench.json \
     --output /tmp/skippy-cache-family-bench/readme-tables.md
   ```

## Reporting Rules

- Keep rows and columns ordered by related family:
  Qwen3Next, Falcon-H1, Llama, Qwen3 dense, DeepSeek2, GLM-4.7 Flash, GLM4,
  Gemma4 A4B, Gemma4 E4B, Gemma3, Gemma2, OLMo, MiniMax M2.7.
- Always report Skippy versus llama-server for full-GGUF rows.
- Keep DeepSeek3 in package-only evidence unless a machine can run a monolithic
  full-GGUF llama-server baseline.
- Use one generated token and matched prefix tokens for apples-to-apples rows.
- If a family fails correctness, leave the benchmark row out of promoted README
  evidence and explain the failure in `crates/skippy-cache/TODO.md`.
- Preserve raw outputs under `/tmp/...` or another explicit run directory; do
  not paste ad hoc numbers without the backing `production-cache-bench.json`.
