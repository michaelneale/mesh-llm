# skippy-metrics

Shared telemetry naming conventions for staged runtime components.

Use this crate for stable attribute keys, metric names, and report vocabulary
that must line up across `skippy-server`, `metrics-server`, benchmarks, and
correctness tooling.

## Architecture Role

`skippy-metrics` is the shared vocabulary for the staged request path. It
does not collect or store telemetry itself; it keeps names stable while other
crates emit or consume OTLP data.

```mermaid
flowchart LR
    Mesh["mesh-llm<br/>topology + lifecycle metadata"] -.-> M["metrics-server"]
    S0["stage servers<br/>request summaries"] -.-> M
    B["bench / diagnostic drivers<br/>run metadata"] -.-> M
    M --> D["metrics.duckdb"]
    M --> R["report.json<br/>DuckDB-backed views"]

    V["skippy-metrics<br/>names + attributes"] -.-> S0
    V -.-> B
    V -.-> M
```

The hot inference path must not block on telemetry. Stage servers and runtime
cache operations emit best-effort summaries, while `metrics-server` owns
ingestion, DuckDB storage, and report export.
