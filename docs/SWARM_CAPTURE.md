# Passive swarm capture

`mesh-llm` includes an opt-in local debug capture for diagnosing mesh
membership churn, gossip propagation, connection lifecycle, and peer
behaviour. The capture sink is disabled by default and writes only to a
directory chosen by the operator:

```bash
mesh-llm client --join '<invite-token>' --headless --swarm-capture /tmp/mesh-swarm-capture
# or
MESH_LLM_SWARM_CAPTURE=/tmp/mesh-swarm-capture mesh-llm client --auto --headless
```

The node writes best-effort append-only JSONL to `swarm-capture.jsonl`. Events
are passive: gossip observations, peer add/update/seen/rejection state, mesh
stream types, route-table requests, selected direct/relay path hints, connection
lifecycle outcomes, local prune/removal decisions, and local HTTP request
metadata on the management and OpenAI-compatible ports.

Capture-enabled clients register the normal runtime `owner.json` metadata even
when running in `client` mode. That makes detached observers visible to
`mesh-llm stop` and avoids orphaned long-running capture processes:

```bash
# One terminal/session manager can detach this command; no invite token is put on
# the command line when using --auto.
mesh-llm client --auto --headless \
  --console 3132 --port 9338 \
  --swarm-capture /tmp/mesh-swarm-capture

# Later, from the same runtime root:
mesh-llm stop
```

Privacy and safety boundaries:

- Use a dedicated private capture directory. On Unix, `mesh-llm` creates missing
  final capture directories with mode `0700`; if the directory already exists,
  it must already be private (`0700` or stricter) and must not be a symlink.
  The JSONL file is opened without following symlinks.
- The capture file is local-only. It is not exported through OTLP telemetry or
  any plugin.
- Request bodies and body hashes are never stored. The capture includes body
  byte length only.
- Invite tokens and owner-control payloads are not recorded by the capture
  layer.
- The node does not issue commands or probes to other peers. It records
  metadata observed through normal mesh gossip, route-table, stream, heartbeat,
  and HTTP handling. Some events label outcomes of existing mesh reachability
  checks; the capture layer does not add new checks.
- Captured endpoint IDs, IP addresses, hostnames, and owner labels are
  diagnostic artifacts and may be personal or pseudonymous data. Keep retention
  narrow and correlate with relay/edge/firewall logs before making
  source-attribution claims.
- `peer_path_observed.observed_direct_remote_addr` is recorded only when iroh's
  selected path is a direct `IP:port` transport address. Relay observations set
  `observed_via_relay=true` and must not be interpreted as the peer source IP.
- Peer add/update/seen/rejection events intentionally omit advertised endpoint
  addresses. Those addresses can be transitive or self-advertised and are not
  treated as observed source IP data.

Useful event families:

- `peer_direct_add`, `peer_direct_update`, `peer_direct_seen`
- `peer_transitive_add`, `peer_transitive_update`, `peer_transitive_seen`
- `peer_rejected`
- `gossip_inbound`, `peer_path_observed`
- `peer_connection_accepted`, `peer_connection_opened`
- `peer_connection_closed`, `peer_connection_failed`
- `peer_direct_proof_of_life`
- `peer_down_received`, `peer_down_rejected`, `peer_down_confirmed`
- `peer_dead_marked`, `peer_dead_ttl_expired`
- `peer_leaving_received`, `peer_pruned`, `peer_removed`
- `mesh_stream_observed`, `mesh_stream_rejected`, `route_request`
- `management_http_request`, `openai_ingress_http_request`

Interpretation notes:

- `peer_direct_proof_of_life` means this node completed a non-transitive gossip
  exchange or heartbeat gossip with that endpoint. It is stronger than a
  transitive mention, but it may still be relay-mediated and does not identify a
  human operator, prove a direct network path, or prove model serving.
- `peer_transitive_*` means a bridge peer mentioned another endpoint in gossip;
  use the `bridge` field to study concentration and repeated re-advertisement.
- `peer_pruned`, `peer_removed`, and `peer_dead_ttl_expired` are local state
  transitions. They show this observer aged out or removed a peer; they are not
  authoritative proof that the remote process exited.
- `observed_direct_remote_addr` is an observed transport socket address, not a
  person, geography, or host-ownership claim. NAT, VPN, shared infrastructure,
  relay fallback, and port reuse can all weaken attribution.
- Capture is intentionally non-blocking. During overload or abrupt shutdown the
  bounded writer queue may drop tail events, so the JSONL file should be treated
  as a best-effort diagnostic log, not a lossless packet/event ledger.
