---
name: metrics-server
description: Use this skill when working on benchmark telemetry ingest, metrics-server run lifecycle, OTLP collection, DuckDB storage, benchmark report export, or separating telemetry/reporting ownership from staged runtime servers.
metadata:
  short-description: Work on benchmark telemetry and reports
---

# metrics-server

Use this skill when working on benchmark telemetry ingest, run lifecycle, or
report export.

## Commands

```bash
cargo build -p metrics-server

target/debug/metrics-server serve \
  --db /tmp/metrics.duckdb \
  --http-addr 127.0.0.1:18080 \
  --otlp-grpc-addr 127.0.0.1:14317
```

Benchmark reports should come from metrics-server data. Stage servers emit OTLP;
they do not own canonical report export.

## Workflow

- Start `metrics-server` before a benchmark or experimental skippy run.
- Pass the OTLP endpoint to skippy stages with `--metrics-otlp-grpc`.
- Use `--debug-retain-raw-otlp` only for surgical debugging; default reports
  should avoid retaining raw payloads.
- Finalize the run through the HTTP API and export `report.json` from
  metrics-server data.
