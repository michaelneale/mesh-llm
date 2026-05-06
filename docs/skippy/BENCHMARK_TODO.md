# Focused Runtime Benchmark TODO

This benchmark keeps generated artifacts in ignored local output directories and
separates internal staged-runtime numbers from OpenAI-compatible `llama-benchy`
numbers.

## Goals

- Establish repeatable baseline output before optimizing activation dtype, KV
  reuse, startup materialization, or prefill scheduling.
- Use existing benchmark entry points first: `kv-server bench-page-sizes`,
  `skippy-bench`, `llama-spec-bench`, and the OpenAI smoke/benchy scripts.
- Emit machine-readable JSON plus a short Markdown summary for each run.
- Record the model identity, corpus, context size, generation limit, topology,
  commit SHA, hardware, and command line beside every performance result.

## Output Policy

- Model-free local scratch output: `benchmark-output/<run-id>/`
- Generated corpora: keep under `target/bench-corpora/` via
  `scripts/generate-bench-corpus.py`.
- OpenAI smoke and benchy scratch output: keep under `/tmp/skippy-openai-smoke/`
  or another explicit `/tmp` path.
- Existing `skippy-bench` distributed output: keep under the configured bench root
  such as `/Volumes/External/skippy-runtime-bench`, unless a run explicitly writes
  to an ignored local directory.
- Commit only the plan, docs, and code needed to reproduce benchmarks. Do not
  commit generated `benchmark-output/` or `bench-output/` files.
- If a result becomes canonical evidence, copy a curated summary into the
  appropriate doc with the exact source command, commit SHA, model/topology, and
  hardware.

## TODO

### M0 - Baseline smoke output

- [x] Run a model-free local smoke using `kv-server bench-page-sizes`.
  - Acceptance: save raw JSON and a Markdown table under
    `benchmark-output/<run-id>/`.
  - Acceptance: summary includes page tokens, payload bytes, reserve/write/commit
    means, attach/evict means, and LZ4 compression/decompression means.
- [x] Record local environment metadata with the smoke output.
  - Acceptance: include commit SHA, OS/arch, Rust version, and exact command.

### M1 - Staged runtime benchmark harness gaps

Existing baseline: `skippy-bench run` already launches staged runtime benchmark
runs and writes `report.json` plus driver timing output, including first-token
and latency summary fields. The remaining work is to make the focused scenarios
easy to run, compare, and validate consistently rather than building a second
stage-chain runner.

- [x] Add a focused `skippy-bench` preset or wrapper for cold startup,
  first-token latency, steady decode, and optional KV warm reuse.
  - Acceptance: each scenario writes JSON with p50/p95 latency, token throughput,
    prompt/decode token counts, topology, and model identity.
  - Acceptance: compact output exposes `topology`, `model`, `latency_ms`,
    `throughput_tokens_per_second`, and `token_counts` fields for consumers.
  - Acceptance: cold-start reporting separates startup readiness from full run
    wall time.
  - Acceptance: output reuses existing `skippy-bench` report fields where they
    already exist and adds only the missing fields needed for comparisons.
  - Acceptance: the preset/wrapper reuses existing launcher/orchestration code
    instead of creating a second stage-chain runner.
- [x] Keep a tiny smoke and leave full performance runs opt-in.

### M2 - KV warm reuse lane

- [ ] Extend or wrap `skippy-bench kv-hit-regression` to report warm-prefix
  timing deltas, not only pass/fail cache behavior.
  - Acceptance: run 1 vs run 2 captures lookup hit/miss, runtime import/export
    wait, and decode timing.
  - Acceptance: correctness still compares against recompute before reporting
    performance numbers.

### M3 - OpenAI frontend / llama-benchy lane

- [ ] Keep `llama-benchy` as the OpenAI-surface benchmark path using
  `scripts/openai-smoke.sh` and `scripts/run-llama-benchy-openai.sh`.
  - Acceptance: docs show `RUN_BENCHY=1 scripts/openai-smoke.sh` for smoke runs.
  - Acceptance: docs show direct `SAVE_RESULT=... scripts/run-llama-benchy-openai.sh`
    usage for already-running `serve-openai` endpoints.
  - Acceptance: docs mention that the wrapper can auto-discover `MODEL` from
    `/v1/models` and defaults to Markdown output unless `FORMAT` is set.
- [ ] Do not treat `llama-benchy` as a replacement for `skippy-bench`.
  - Acceptance: runtime-path metrics and OpenAI-path metrics are labeled
    separately in output summaries.

### M4 - Result formatting

- [ ] Add or document a formatter that converts raw JSON results into Markdown.
  - Acceptance: generated Markdown has a command block, environment block, and a
    compact table of the key latency/throughput fields.
  - Acceptance: formatter works for the model-free `kv-server bench-page-sizes`
    smoke before being generalized to `skippy-bench` reports.

## Initial Commands

Model-free smoke:

```bash
cargo build -p kv-server
mkdir -p benchmark-output/kv-page-sizes
target/debug/kv-server bench-page-sizes \
  --page-root benchmark-output/kv-page-sizes/pages \
  --iterations 5 \
  --run-id kv-page-sizes-local \
  > benchmark-output/kv-page-sizes/report.json
```

OpenAI-surface smoke with benchy, after `just build` or an equivalent build:

```bash
RUN_BENCHY=1 \
SAVE_RESULT=/tmp/skippy-openai-smoke/benchy-smoke.md \
scripts/openai-smoke.sh
```

OpenAI-surface benchy against an already-running endpoint. `MODEL` can be
omitted when `/v1/models` returns the served model id, and Markdown is the
default output format unless `FORMAT` is set:

```bash
BASE_URL=http://127.0.0.1:9337/v1 \
TOKENIZER=meta-llama/Llama-3.2-1B-Instruct \
PP="128 512" \
TG="16 32" \
RUNS=3 \
SAVE_RESULT=/tmp/skippy-benchy.md \
scripts/run-llama-benchy-openai.sh
```
