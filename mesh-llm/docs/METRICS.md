# Metrics

This document describes the bounded routing metrics currently exposed through
the local management API.

These metrics follow the vocabulary proposed in issue `#275`:

- `runtime`: directly useful for routing, failover, or placement decisions
- `information`: primarily exposed for operator and API visibility
- `strategy`: primarily useful for roadmap or longer-horizon analysis
- `local-only`: measured and consumed on one node only
- `peer-advertised`: published by a node to peers
- `mesh-derived`: computed locally from existing peer/API/state snapshots

The groups below are the only metric groups added by this PR. All of them are
`local-only`. None of them are new gossip or protocol fields.

The code-level registry in [`src/network/metrics.rs`](../src/network/metrics.rs)
is the authoritative classification source for these exported groups.

## `/api/status` `routing_metrics`

Measured on the current node only. These values are not mesh-wide aggregates.

| Field(s) | Meaning | Layer | Scope |
| --- | --- | --- | --- |
| `request_count`, `successful_requests`, `success_rate` | Request-level routing outcomes for traffic fronted by this node | information | local-only |
| `retry_count`, `failover_count` | Local reroute and failover pressure | information | local-only |
| `attempt_timeout_count`, `attempt_unavailable_count`, `attempt_context_overflow_count`, `attempt_reject_count` | Attempt outcome breakdown observed by this node | information | local-only |
| `avg_queue_wait_ms`, `avg_attempt_ms`, `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Bounded timing and throughput summary from locally observed attempts | information | local-only |

## `/api/status` `routing_metrics.local_node`

Measured on the current node only. These are local routing pressure and
lightweight utilization proxies, not a full utilization model.

| Field(s) | Meaning | Layer | Scope |
| --- | --- | --- | --- |
| `current_inflight_requests`, `peak_inflight_requests` | Live and recent peak request pressure on this node | runtime | local-only |
| `local_attempt_count`, `remote_attempt_count`, `endpoint_attempt_count` | Attempt mix by local, remote, and endpoint targets | runtime | local-only |
| `avg_queue_wait_ms`, `avg_attempt_ms`, `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Current-node latency and throughput proxies for locally observed attempts | runtime | local-only |

## `/api/status` `routing_metrics.pressure`

Measured on the current node only. These shares are derived from requests this
node fronted; they are not mesh-wide demand totals.

| Field(s) | Meaning | Layer | Scope |
| --- | --- | --- | --- |
| `fronted_request_count`, `locally_served_request_count`, `remotely_served_request_count`, `endpoint_request_count` | Service mix for requests fronted by this node | information | local-only |
| `local_service_share`, `remote_service_share`, `endpoint_service_share` | Normalized local, remote, and endpoint shares for locally fronted traffic | information | local-only |

## `/api/models[]` `routing_metrics`

Measured on the current node only. These values describe what this node has
observed while routing requests for that model.

| Field(s) | Meaning | Layer | Scope |
| --- | --- | --- | --- |
| `request_count`, `successful_requests`, `success_rate` | Per-model request outcomes observed locally | information | local-only |
| `retry_count`, `failover_count` | Per-model instability and recovery pressure observed locally | information | local-only |
| `attempt_timeout_count`, `attempt_unavailable_count`, `attempt_context_overflow_count`, `attempt_reject_count` | Per-model attempt outcome breakdown observed locally | information | local-only |
| `avg_queue_wait_ms`, `avg_attempt_ms`, `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Per-model bounded timing and throughput summary observed locally | information | local-only |

## `/api/models[]` `routing_metrics.targets[]`

Measured on the current node only. These entries are the most route-adjacent
memory in this PR.

| Field(s) | Meaning | Layer | Scope |
| --- | --- | --- | --- |
| `target`, `kind`, `last_updated_secs_ago` | Which target the current node observed and how recent that memory is | runtime | local-only |
| `attempt_count`, `success_count`, `success_rate` | Per-target success history observed locally | runtime | local-only |
| `timeout_count`, `timeout_rate`, `unavailable_count`, `context_overflow_count`, `reject_count` | Per-target failure and degradation breakdown observed locally | runtime | local-only |
| `avg_queue_wait_ms`, `avg_attempt_ms`, `avg_tokens_per_second`, `completion_tokens_observed`, `throughput_samples` | Per-target latency and throughput summary observed locally | runtime | local-only |
