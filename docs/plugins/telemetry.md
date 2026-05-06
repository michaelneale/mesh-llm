# Telemetry Plugin

The built-in `telemetry` plugin enables metrics-only OTLP/HTTP export for local
model lifecycle telemetry. It is opt-in and records from the runtime load,
unload, and exit paths instead of polling local APIs.

## Configuration

Enable the plugin with `[[plugin]] name = "telemetry"` and configure an OTLP
metrics endpoint:

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

Endpoint precedence is:

1. `telemetry.metrics.endpoint`
2. `telemetry.endpoint` normalized to `/v1/metrics`
3. `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`
4. `OTEL_EXPORTER_OTLP_ENDPOINT` normalized to `/v1/metrics`

If no endpoint is configured, telemetry export stays disabled.

## Exported Metrics

Counters:

- `mesh_llm_model_launch_total`
- `mesh_llm_model_launch_success_total`
- `mesh_llm_model_launch_failure_total`
- `mesh_llm_model_unload_total`
- `mesh_llm_model_exit_unexpected_total`

Gauges:

- `mesh_llm_loaded_models`
- `mesh_llm_model_loaded`
- `mesh_llm_model_context_length`

Histograms:

- `mesh_llm_model_launch_duration_ms`
- `mesh_llm_model_uptime_s`

## Privacy Boundary

The telemetry plugin exports lifecycle metrics only. It does not export prompts,
completions, logs, traces, hostnames, mesh gossip, relay messages, raw GPU stable
IDs, or prompt hashes.

Local absolute and path-like model labels are reduced to filenames before export.
Hugging Face refs are preserved. GPU stable IDs are hashed before export.

## Runtime Safety

Telemetry exporter setup failures disable telemetry without failing inference
startup. Runtime events are buffered through a bounded queue; when the queue is
full, the oldest event is dropped instead of blocking inference.
