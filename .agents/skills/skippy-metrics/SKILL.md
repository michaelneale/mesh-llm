---
name: skippy-metrics
description: Use this skill when working on skippy telemetry attributes, OTLP emission, benchmark metric names, runtime lifecycle telemetry, or separating telemetry/reporting ownership from stage runtime serving.
metadata:
  short-description: Work on skippy telemetry
---

# skippy-metrics

Use this skill for telemetry attributes, lifecycle instrumentation, and
benchmark/report integration.

## Ownership

`crates/skippy-metrics` owns shared attribute names. Stage servers may emit
OTLP/telemetry, but request-path serving must not block on telemetry export.
`crates/metrics-server` owns benchmark/debug telemetry ingest, DuckDB storage,
run lifecycle, and canonical report export.

Mesh API runtime status is not a telemetry dump. Keep public runtime status
backend-neutral and stable; expose backend details only when intentionally part
of the status shape.

## Validation

```bash
cargo test -p skippy-server --lib
cargo test -p mesh-llm --lib
```

Keep canonical benchmark report export in `metrics-server` rather than inside
stage serving.
