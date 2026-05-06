---
name: skippy-spec-bench
description: Use this skill when testing or benchmarking target/draft GGUF pairs for speculative decoding compatibility, tokenizer agreement, draft acceptance rate, or staged verification behavior.
metadata:
  short-description: Benchmark speculative target/draft pairs
---

# skippy-spec-bench

Use this skill for target/draft speculative compatibility work.

## What It Checks

- Target and draft tokenization agreement.
- Baseline target decode versus draft-verified decode.
- Draft acceptance/rejection behavior.
- Batched verification and checkpoint/restore behavior.
- Recurrent-state implications for rollback.

## Repo Notes

The old source repo used a standalone `llama-spec-bench` crate. It may not be
present in this mesh checkout yet, so verify available packages before running
commands:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

If the spec bench is imported, keep it as a diagnostics/benchmark tool. Do not
make normal mesh serving depend on it.
