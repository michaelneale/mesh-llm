# OutputEvent taxonomy

`OutputEvent` is the typed terminal-output contract. Pretty/TUI output is rendered on `stderr`; JSON mode writes newline-delimited records to `stdout`; tracing remains a separate `stderr` stream.

## Stream contract

- JSON records always include `timestamp`, `level`, `event`, and `message`, plus the event fields below. `error` records also include `error_type`.
- Pretty mode consumes the same events to update the dashboard, event history, endpoint cards, model progress, process rows, and the join-token panel.
- `ready` is the aggregate runtime-ready event. Multi-model startup emits it only after every declared startup model reaches readiness, then queues the first `>` prompt.
- Interactive pretty mode is only used when stdin and stderr are TTYs. The fallback line renderer still honors `h` for help, `i` for an info snapshot, and `q` for clean shutdown.

## TUI rewrite maintenance notes

Keep these details current when changing `OutputEvent` or dashboard state:

- The dashboard state is event-driven via `PrettyDashboardState::apply_output_event()` plus periodic `PrettyDashboardSnapshot` refreshes for process/model/request telemetry.
- `invite_token` includes `mesh_name?`; the dashboard uses it for the join-token panel.
- `model_download_progress` is emitted during catalog preparation when the interactive TUI is active and drives the model-progress panel.
- `ready` may include `pi_command` and `goose_command`; these are operational hints shown after startup.
- Some variants are schema/dashboard-supported before all of them have production emitters. Mark that explicitly rather than leaving stale source-search notes.

## Events

`?` means optional. Struct-like field names (`moe`, `distribution`, `progress`, `status`) are summarized below the table.

| Event | Fields | Emit/use notes |
| --- | --- | --- |
| `info` | `message`, `context?` | Shared informational helper for runtime, discovery, routing, mesh, and tracing-to-output bridge notes. |
| `startup` | `version`, `message?` | Formatter/dashboard-supported process bootstrap record; no production emitter was found in this pass. |
| `node_identity` | `node_id`, `mesh_id?` | Formatter/dashboard-supported node header seed; no production emitter was found in this pass. |
| `invite_token` | `token`, `mesh_id`, `mesh_name?` | Emitted when an invite token is ready; also fills the dashboard join-token panel. |
| `discovery_starting` | `source` | Discovery or re-discovery path is starting. |
| `mesh_found` | `mesh`, `peers`, `region?` | A discovery candidate was found before join. |
| `discovery_joined` | `mesh` | Discovery candidate joined successfully. |
| `discovery_failed` | `message`, `detail?` | Discovery or join attempt failed. |
| `waiting_for_peers` | `detail?` | Startup is waiting for peer capacity, local model selection, or a better placement. |
| `passive_mode` | `role`, `status`, `capacity_gb?`, `models_on_disk?`, `detail?` | Client/standby startup and passive capacity visibility. |
| `peer_joined` | `peer_id`, `label?` | Dashboard-supported peer membership event; no production emitter was found in this pass. |
| `peer_left` | `peer_id`, `reason?` | Dashboard-supported peer membership event; no production emitter was found in this pass. |
| `model_queued` | `model` | Dashboard-supported model lifecycle state; no production emitter was found in this pass. |
| `model_loading` | `model`, `source?` | Dashboard-supported model lifecycle state; no production emitter was found in this pass. |
| `model_loaded` | `model`, `bytes?`, `moe?` | Dashboard-supported model lifecycle state; no production emitter was found in this pass. |
| `moe_detected` | `model`, `moe`, `fits_locally?`, `capacity_gb?`, `model_gb?` | MoE detection during model planning and placement. |
| `moe_distribution` | `model`, `moe`, `distribution` | MoE split plan after ranking/sharding. |
| `moe_status` | `model`, `status` | MoE planning, ranking, placement, and standby status. |
| `moe_analysis_progress` | `model`, `progress` | MoE expert-ranking or artifact-scan progress. |
| `host_elected` | `model`, `host`, `role?`, `capacity_gb?` | Model host election, including demand-based rebalancing. |
| `rpc_server_starting` | `port`, `device`, `log_path?` | `rpc-server` launch started in `inference/launch.rs`. |
| `rpc_ready` | `port`, `device`, `log_path?` | Formatter/dashboard-supported ready transition; no production emitter was found in this pass. |
| `llama_starting` | `model?`, `http_port`, `ctx_size?`, `log_path?` | `llama-server` launch started in `inference/launch.rs`. |
| `llama_ready` | `model?`, `port`, `ctx_size?`, `log_path?` | `llama-server` is ready, before aggregate `ready`. |
| `model_ready` | `model`, `internal_port?`, `role?` | Model-serving readiness. JSON includes both `port` and `internal_port` for compatibility when a port exists. |
| `multi_model_mode` | `count`, `models` | Startup declared more than one model. |
| `webserver_starting` | `url` | Formatter/dashboard-supported console startup state; no production emitter was found in this pass. |
| `webserver_ready` | `url` | Web console ready. |
| `api_starting` | `url` | Formatter/dashboard-supported API startup state; no production emitter was found in this pass. |
| `api_ready` | `url` | OpenAI-compatible API ready for normal runtime/passive paths. Bootstrap proxy readiness currently emits generic `info` events. |
| `ready` | `api_url`, `console_url?`, `api_port`, `console_port?`, `models_count?`, `pi_command?`, `goose_command?` | Aggregate runtime readiness. Keep this after startup model readiness and before the first prompt. |
| `model_download_progress` | `label`, `file?`, `downloaded_bytes?`, `total_bytes?`, `status` | Catalog/model preparation progress for the interactive TUI. `status` is `ensuring`, `downloading`, or `ready`. |
| `request_routed` | `model`, `target` | Formatter/dashboard-supported routing decision; no production emitter was found in this pass. |
| `warning` | `message`, `context?` | Shared warning helper for non-fatal runtime, mesh, launch, and tracing bridge conditions. |
| `error` | `message`, `context?` | Shared fatal/error helper. JSON adds `error_type` from the classifier. |
| `shutdown` | `reason?` | Clean shutdown from Ctrl+C, `q`, or another stop path. |

## Nested field shapes

- `moe`: `{ experts, top_k }`
- `distribution`: `{ leader, active_nodes, fallback_nodes, shard_index, shard_count, ranking_source?, ranking_origin?, overlap?, shared_experts?, unique_experts? }`
- `status`: `{ phase, detail? }`
- `progress`: `{ mode, spinner, current, total?, elapsed_secs }`
- `RuntimeStatus`: `starting`, `ready`, `shutting down`, `stopped`, `exited`, `warning`, `error`

## Extension guide

When adding or changing an event:

1. Update the `OutputEvent` variant and `event_name()`, `message()`, `summary_line()`, and `json_fields()`.
2. If it affects the TUI, update `PrettyDashboardState::apply_output_event()` and any snapshot/provider fields it depends on.
3. Add or update pretty and JSON tests in `crates/mesh-llm/src/cli/output/mod.rs`.
4. Emit through the shared output manager/helper path; do not write directly to `stdout` or `stderr` for user-facing output.
5. For startup readiness, preserve `stdout` JSON / `stderr` pretty separation and keep aggregate `ready` last.
