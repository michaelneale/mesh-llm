---
name: telemetry-privacy-review
description: Use this skill when adding, renaming, removing, or reviewing mesh-llm OTLP metrics, telemetry attributes, metrics exporter settings, or telemetry documentation.
metadata:
  short-description: Review mesh-llm telemetry privacy
---

# telemetry-privacy-review

Use this skill before changing mesh-llm OTLP metrics, exporter activation, or
telemetry attribute names.

## Review Contract

- Keep telemetry metrics-only. Do not export prompts, completions, logs, traces,
  hostnames, mesh gossip, relay messages, raw node IDs, raw GPU stable IDs,
  endpoint URLs, local absolute paths, or prompt hashes.
- Keep egress explicit. There must be no hard-coded collector. Generic OTel env
  endpoints may only be consumed after `telemetry.enabled = true`; mesh config
  endpoints are explicit operator configuration.
- Treat hashed IDs as stable pseudonymous identifiers, not anonymous data.
- Keep request-path telemetry non-blocking and bounded.
- Keep model labels sanitized with the runtime telemetry model-label helper.
- Prefer bounded enums, buckets, counts, and hashes over high-cardinality raw
  values.

## Required Updates

- Update `TELEMETRY_ATTRIBUTE_ALLOWLIST` in
  `crates/mesh-llm/src/runtime/survey.rs` for every new exported attribute.
- Update `docs/plugins/telemetry.md` with the metric or attribute inventory and
  privacy handling.
- Add focused tests for private-path, raw-ID, endpoint-URL, prompt, and
  completion exclusion when the change touches those surfaces.

## Validation

Run the narrowest relevant checks for the touched area. For telemetry runtime
changes, start with:

```bash
cargo test -p mesh-llm runtime::survey::tests --lib
cargo test -p mesh-llm telemetry_config --lib
```

If routing telemetry changed, also run the focused mesh routing telemetry test:

```bash
cargo test -p mesh-llm routing_telemetry_sink_receives_request_pressure_and_attempt_events --lib
```
