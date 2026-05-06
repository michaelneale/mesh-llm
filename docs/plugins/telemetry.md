# Telemetry Plugin

The built-in `telemetry` plugin enables metrics-only OTLP/HTTP export for local
model lifecycle and routing telemetry. The plugin is enabled by default, while
export still requires a configured OTLP metrics endpoint.

## Configuration

Configure an OTLP metrics endpoint:

```toml
[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "https://otel.example.com"
headers = { "authorization" = "Bearer TOKEN" }
export_interval_secs = 15
queue_size = 2048

[telemetry.metrics]
endpoint = "https://otel.example.com/v1/metrics"

[[plugin]]
name = "telemetry"
enabled = true
```

The `[[plugin]]` entry is optional when telemetry should stay enabled. To opt
out of the built-in plugin entirely, set:

```toml
[[plugin]]
name = "telemetry"
enabled = false
```

Endpoint precedence is:

1. `telemetry.metrics.endpoint`
2. `telemetry.endpoint` normalized to `/v1/metrics`
3. `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`
4. `OTEL_EXPORTER_OTLP_ENDPOINT` normalized to `/v1/metrics`

If no endpoint is configured, telemetry export stays disabled.

## Exported Metrics

Request and route metrics are emitted per fronting node. A collector or
dashboard can aggregate `mesh_llm_requests_inflight` across nodes for a
mesh-wide in-flight request view.

Counters:

- `mesh_llm_model_launch_total`
- `mesh_llm_model_launch_success_total`
- `mesh_llm_model_launch_failure_total`
- `mesh_llm_model_unload_total`
- `mesh_llm_model_exit_unexpected_total`
- `mesh_llm_model_request_total`
- `mesh_llm_route_attempt_total`

Gauges:

- `mesh_llm_loaded_models`
- `mesh_llm_model_loaded`
- `mesh_llm_model_context_length`
- `mesh_llm_requests_inflight`

Histograms:

- `mesh_llm_model_launch_duration_ms`
- `mesh_llm_model_uptime_s`

## Privacy Boundary

The telemetry plugin exports metrics only. It does not export prompts,
completions, logs, traces, hostnames, mesh gossip, relay messages, raw node IDs,
raw GPU stable IDs, endpoint URLs, or prompt hashes.

Local absolute and path-like model labels are reduced to filenames before export.
Hugging Face refs are preserved. GPU stable IDs and node IDs are hashed before
export. Route-attempt metrics label local, remote, and endpoint target kinds;
remote target IDs are exported only as stable hashes so collectors can aggregate
node-to-node traffic without exposing raw peer IDs.

## Runtime Safety

Telemetry exporter setup failures disable telemetry without failing inference
startup. Runtime events are buffered through a bounded queue; when the queue is
full, the oldest event is dropped instead of blocking inference.
