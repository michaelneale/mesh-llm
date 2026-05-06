# Benchmark Corpus

This document records the first production-style corpus plan for staged prefill
and TTFT benchmarks. The goal is to cover realistic prompt shapes without
turning every run into an all-day evaluation.

## First Corpus Shape

Target size: 60 prompts.

- 10 short chat and instruction prompts
- 10 medium instruction, summarization, and rewriting prompts
- 10 math reasoning prompts
- 10 code generation and code reasoning prompts
- 10 long document QA prompts
- 10 repeated-prefix or retrieval-style prompts with shared context

The benchmark driver should preserve category labels in `driver-result.json` so
reports can show both overall latency and per-bucket latency.

## Candidate Hugging Face Sources

- `HuggingFaceH4/mt_bench_prompts` for compact multi-turn chat prompts
- `databricks/databricks-dolly-15k` for instruction and summarization prompts
- `openai/gsm8k` for math reasoning prompts
- `openai/openai_humaneval` for code generation prompts
- `codeparrot/apps` for broader programming problem prompts
- `allenai/qasper` for long document QA prompts
- `zai-org/LongBench` for long-context and multi-document tasks
- `nvidia/ChatRAG-Bench` for retrieval-style prompt shapes
- `SWE-bench/SWE-smith-trajectories` for native multi-turn coding-agent edit
  sessions and warm n-gram speculative decoding benchmarks

## Avoid In The First Pass

- gated datasets that block unattended lab setup
- non-commercial datasets for benchmark corpora that may be redistributed
- huge raw corpora where prompt extraction would dominate the benchmark work
- datasets without clear licensing until reviewed

## Extraction Rules

- store extracted prompts as JSONL with stable `prompt_id`, `category`, and
  `prompt` fields
- keep the source dataset and source row identifier in metadata when available
- cap prompt text by token count per bucket instead of by byte length
- keep repeated-prefix prompts explicit so the benchmark can later test KV reuse
  and prefix caching separately from ordinary one-shot prompts
- do not mix decode optimization experiments into this corpus pass

## KV Mixed Corpus

The initial KV-focused checked-in corpus lives at:

```text
crates/skippy-bench/corpora/kv_mixed_prompts.jsonl
```

It is intentionally smaller than the full 60-prompt production corpus while the
runtime KV export/import hooks are still pending. It covers:

- very short prompts for fixed overhead
- short and medium prompts for common interactive traffic
- long document QA and analysis prompts for page-size pressure
- adjacent repeated-prefix groups for future local prefix-hit tests

## Coding Loop Corpus

The generated `coding-loop` tier is for speculative decoding policy work,
especially n-gram pooling. It is generated from
`SWE-bench/SWE-smith-trajectories` on Hugging Face rather than checked into the
repository:

```bash
just bench-corpus coding-loop
```

The current tier samples 20 SWE-smith trajectories and expands each trajectory
into 8 adjacent prompt rows, producing 160 prompts under:

```text
target/bench-corpora/coding-loop/corpus.jsonl
target/bench-corpora/coding-loop/manifest.json
```

Rows use family `coding_edit_loop`, preserve the original trajectory
`session_group`, and carry `metadata.benchmark_shape = "repeated_edit_loop"`.
This gives n-gram pooling repeated software-engineering context instead of the
cold one-pass prompts in the broader `long` corpus.

## Cache Use-Case Matrix Corpus

The cache benchmark use-case matrix uses a small checked-in HF-derived corpus at:

```text
evals/skippy-usecase-corpus.json
```

It keeps the source dataset/config/split/row metadata beside each prompt so the
README benchmark matrix can be reproduced and audited without guessing which
prompt shape was used.

| Use case | Source dataset | Config | Split | Row |
| --- | --- | --- | --- | ---: |
| Tool calling | `glaiveai/glaive-function-calling-v2` | `default` | `train` | 1 |
| Text-to-SQL | `gretelai/synthetic_text_to_sql` | `default` | `test` | 0 |
| Coding agent loop | `SWE-bench/SWE-smith-trajectories` | `default` | `tool` | 0 |
| Issue fixing | `SWE-bench/SWE-bench` | `default` | `dev` | 0 |
| Code refinement | `google/code_x_glue_cc_code_refinement` | `small` | `test` | 0 |
| Few-shot reasoning | `openai/gsm8k` | `main` | `test` | 0 |
| Open chat | `HuggingFaceH4/mt_bench_prompts` | `default` | `train` | 0 |
| Summarization/RAG | `nvidia/ChatRAG-Bench` | `doc2dial` | `test` | 0 |
