# Mesh Workflows

Mesh LLM nodes expose an OpenAI-compatible inference API on `9337` and a
management API plus optional web console on `3131`. A node can serve models,
join as an API-only client, or do both.

## Try the public mesh

```bash
mesh-llm serve --auto
```

`--auto` discovers published meshes, chooses the best candidate, joins it, and
starts serving if this machine has usable hardware. Use this when you just want
the system to work end to end.

## Immutable mesh requirements

Mesh requirements are fixed when the mesh is created. For requirement-aware
meshes, changing mesh requirements creates a new mesh: changing the required
node version, protocol generation, or release-attestation policy derives a new
mesh id from the new policy hash. Trust policy is a local owner-trust setting,
not part of the immutable mesh requirements hash.

The release-attestation side of mesh requirements is build provenance only:
certified-build admission is not remote runtime attestation. A signed release
attestation proves a peer's binary was published by a trusted release signer.
It does not prove the remote process is actually running unmodified code, nor
that the host OS or hardware has not been tampered with.

Requirement-aware meshes use signed bootstrap tokens. Unrestricted legacy and
private meshes still keep the older unsigned invite-token path.

Create an unrestricted mesh:

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M
```

Create a release-attestation-required public mesh:

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M --publish \
  --require-release-attestation \
  --release-signer-key ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --owner-key ~/.mesh-llm/owner-keystore.json \
  --owner-required \
  --trust-policy require-owned \
  --node-label lab-a
```

The release-attestation flags above are creation-time mesh requirements. The
owner-key and trust-policy flags are local owner-identity policy; they do not
change the mesh requirements hash.

Equivalent creation-time config:

```toml
[mesh_requirements]
require_release_attestation = true
release_signer_keys = ["ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"]
```

Join via signed bootstrap token:

```bash
mesh-llm serve --join <signed-bootstrap-token>
```

If a node does not satisfy the certified-build gate, the human-facing outcome is
"certified build required". The machine reason codes surfaced in logs, status,
and evidence are underscored: `certified_binary_required`,
`build_proof_invalid`, and `release_signer_untrusted`.

When migrating from an unrestricted legacy mesh to a requirement-aware mesh,
recreate the mesh, republish it, and issue new signed bootstrap tokens.

For an API-only node that does not serve models:

```bash
mesh-llm client --auto
```

## Start a private mesh

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M
```

This starts a private mesh, loads the requested model, opens the local API and
console, and prints an invite token. Only nodes with the token can join.

Join from another GPU node:

```bash
mesh-llm serve --join <token>
```

Join from an API-only client:

```bash
mesh-llm client --join <token>
```

### Multi-interface Linux and Docker hosts

On Linux hosts with several kernel-visible interfaces, especially
`docker run --network host` systems, iroh can discover and advertise Docker or
CNI bridge addresses such as `172.17.0.1`. If every host has the same bridge
address, peers may race the wrong local bridge instead of the real management
network.

Choose the host-to-host interface explicitly:

```bash
# seed
mesh-llm serve --split --bind-ip 10.1.2.3 --bind-port 47916 --model Qwen3-8B-Q4_K_M

# worker
mesh-llm serve --split --join <token> --model Qwen3-8B-Q4_K_M
```

`--bind-ip` binds mesh QUIC to that local address and filters the invite/gossip
direct-address set to the selected IP while keeping relay/public candidates.
Use `--listen-all` only for the local HTTP API/console listener; it does not
select the mesh QUIC interface.

## Publish your own mesh

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M --publish
```

`--publish` advertises the mesh for Nostr discovery so other users can find it
with `--auto`, `--discover`, or `mesh-llm discover`. Published meshes are
republished periodically and include a TTL; if the node exits, the publication
ages out.

Public discovery is separate from admission. A public mesh can still reject
nodes that do not satisfy its creation-time requirements.

Add a friendly discovery name:

```bash
mesh-llm serve --model Qwen3-8B-Q4_K_M --publish --mesh-name "lab-a"
```

Join a named mesh:

```bash
mesh-llm serve --discover "lab-a"
mesh-llm client --discover "lab-a"
```

Without `--publish`, `--mesh-name` is only a local/friendly label. The mesh is
still private unless you share the invite token.

## Blackboard privacy

The built-in blackboard shares status, questions, and notes with peers in the
current mesh. Private meshes enable blackboard by default because membership is
invite-token scoped. Published or auto-joined public meshes require an explicit
opt-in:

```bash
mesh-llm serve --auto --blackboard --name alice
```

On a public mesh, blackboard posts are visible to all peers in that mesh. Do not
post secrets, credentials, private model paths, customer data, or anything that
should not leave your trust boundary. Use a private mesh plus owner/trust flags
when blackboard messages need to stay inside a controlled group.

## Browse discovery

```bash
mesh-llm discover
mesh-llm discover --name "lab-a"
mesh-llm discover --model qwen --min-vram 24
mesh-llm discover --auto
```

`discover --auto` prints the best invite token, which is useful for scripts.

## Console and management API

The console is available at:

```text
http://localhost:3131
```

The management API stays available even when the UI is hidden with
`--headless`:

```bash
mesh-llm serve --auto --headless
curl -s http://localhost:3131/api/status | jq .
curl -s http://localhost:3131/api/discover | jq .
```

`/api/status` reports whether the local mesh publication is `private`,
`public`, or `publish_failed`.

## Private ownership and trust

For owner-attested private deployments, initialize an owner key and start nodes
with the current runtime flags:

```bash
mesh-llm auth init

mesh-llm serve --model Qwen3-14B \
  --owner-key ~/.mesh-llm/owner-keystore.json \
  --node-label studio \
  --trust-policy allowlist \
  --trust-owner <owner-id>
```

Related commands:

```bash
mesh-llm auth status
mesh-llm auth sign-node --node-label studio
mesh-llm auth trust add <owner-id>
mesh-llm auth trust list
mesh-llm auth trust remove <owner-id>
```

## Networking notes

- Discovery uses Nostr relays by default.
- Mesh connectivity uses managed iroh relay infrastructure by default when
  direct paths are unavailable.
- Hidden relay override flags exist for lab/debug deployments, but normal users
  should not need to run their own relay.
- `/v1` request routing and Skippy stage traffic are separate paths. HTTP
  routing is latency-tolerant; stage splits require selected peers with suitable
  topology and latency.
