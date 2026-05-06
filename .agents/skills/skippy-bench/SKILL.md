---
name: skippy-bench
description: Use this skill when running benchmark orchestration, local single-stage or split benchmarks, benchmark report flow, or performance-oriented skippy runtime checks.
metadata:
  short-description: Benchmark skippy stage runtime
---

# skippy-bench

Use this skill for performance, orchestration, and report-oriented checks.
Use `skippy-correctness` when the question is pass/fail exactness.

## Current Repo Shape

Standalone `skippy-bench` may not be present in this mesh checkout yet. Confirm
available packages before using old source-repo commands:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

Useful current checks:

```bash
cargo test -p skippy-server --lib
cargo test -p mesh-llm inference::skippy --lib
```

When benchmark harnesses are imported, keep reporting separate from request-path
serving. Stage runtimes emit telemetry; benchmark/report tooling owns reports.
