# Node Owner Identity Proposal

Status: proposal

Operational incident-response guidance:

- [IDENTITY_INCIDENT_RESPONSE.md](IDENTITY_INCIDENT_RESPONSE.md)

## Summary

Use the existing stable **owner identity** from the keystore to attest to one
or more mesh nodes.

Security model:

- keep owner identity and node identity separate
- use short-lived signed node certificates
- verify full attestations locally anywhere trust matters
- support revocation at both the owner and node-certificate level

Today mesh-llm has two separate identity concepts:

- **Node identity**: the QUIC/iroh endpoint key used on the wire
- **Owner identity**: the `auth` keystore identity derived from the owner signing key

The missing piece is a cryptographic binding between them. This proposal does
not introduce a second owner identity. It adds a non-breaking ownership
attestation so the mesh can say:

> this node is operated by owner `X`

without replacing the current node transport identity.

## Goals

- Keep node transport identity separate from owner identity.
- Let one owner identity attest to many nodes.
- Let nodes rotate their transport key without rotating the owner identity.
- Let peers and the console distinguish:
  - verified owner
  - unverified/legacy node
  - invalid ownership claim
- Preserve mixed-version mesh interoperability.
- Avoid breaking the existing plugin or node protocol.

## Non-goals

- Replacing the QUIC node key with the owner key.
- Requiring public meshes to reject legacy nodes immediately.
- Building a full PKI, CA hierarchy, or remote revocation service in phase 1.
- Changing invite tokens or transport admission semantics in phase 1.

## Current State

Current `main` already has the owner identity we want to use:

- `mesh-llm auth init` creates an owner keypair in the keystore
- `owner_id` is derived from the owner signing public key
- the keystore can be copied to multiple trusted nodes to share that same owner identity

But runtime node startup still uses a separate QUIC secret key stored in
`~/.mesh-llm/key` or an ephemeral key for clients. That node identity is what
the mesh actually sees and trusts on the wire.

As a result, the mesh can authenticate:

- that a peer is the holder of the current node key

but not:

- which long-lived user or operator that node belongs to

## Proposed Model

### Identities

- **Owner identity**
  - Stable across machines
  - This is the existing keystore identity created by `mesh-llm auth init`
  - Identified by `owner_id = sha256(owner_signing_public_key)`

- **Node identity**
  - The existing QUIC endpoint public key
  - Stable per machine unless rotated
  - May be ephemeral for clients when requested

- **Owner attestation**
  - A signed statement from the existing keystore owner identity authorizing a
    specific node identity
  - Carried by the node in gossip and status surfaces

This proposal adds the third item only. The first two already exist.

### Core rule

The owner key does **not** replace the node key.

Instead, the owner signs:

- node public key / endpoint ID
- certificate ID / serial number
- issued-at time
- required expiry
- optional node label
- optional hostname hint
- optional capabilities or scope

Peers verify that signature against the advertised node identity.

## Why this shape

This is the right split for mesh-llm:

- QUIC transport already depends on the node key.
- Existing anti-spoofing checks are built around the QUIC peer identity.
- A single user may run several nodes.
- A single machine may need to rotate its node key without changing who owns it.
- Client mode may still need ephemeral transport identities for same-machine usage.

If we derived the node key from the owner key, we would lose:

- independent node-key rotation
- clean separation between transport and operator identity
- support for multiple nodes per owner without keystore duplication pressure

## Proposed Architecture Changes

### `mesh-llm/src/crypto/`

Add a new ownership module that owns the attestation format and verification
logic for the existing keystore owner identity.

Suggested files:

- `mesh-llm/src/crypto/ownership.rs`
  - `NodeOwnershipClaim`
  - `SignedNodeOwnership`
  - canonical serialization for signing
  - sign/verify helpers
- `mesh-llm/src/crypto/mod.rs`
  - re-export ownership helpers

Suggested data shape:

```rust
pub struct NodeOwnershipClaim {
    pub version: u32,
    pub cert_id: String,
    pub owner_id: String,
    pub owner_sign_public_key: String,
    pub node_endpoint_id: [u8; 32],
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub node_label: Option<String>,
    pub hostname_hint: Option<String>,
}

pub struct SignedNodeOwnership {
    pub claim: NodeOwnershipClaim,
    pub signature: String,
}
```

Verification rules:

- `owner_id` must match the owner signing public key
- signature must verify over canonical claim bytes with a fixed domain separation tag
- `node_endpoint_id` must match the actual QUIC peer identity
- `expires_at_unix_ms` is required
- expired claims are not considered verified
- revoked owners are not considered verified
- revoked certificate IDs are not considered verified
- optionally, revoked node endpoint IDs are not considered verified

Suggested signing domain tag:

```text
mesh-llm-node-ownership-v1
```

### `mesh-llm/src/runtime/`

Runtime startup gains optional binding from the existing keystore owner identity
to the current node key.

Suggested flow:

1. Load the existing node key as today.
2. Optionally load the owner keystore.
3. Build or load a signed ownership attestation for this node key.
4. Attach it to self-announcements and API status.

Suggested behavior:

- If no owner keystore is configured, startup behavior is unchanged.
- If owner loading fails and owner mode is optional, start as an unattributed node and emit a strong warning.
- If owner loading fails and owner mode is required, fail startup.
- If the node certificate is expired or revoked, do not surface it as verified.

### Certificate Lifecycle And Rotation UX

Recommended split:

- certificate renewal is automatic
- node identity rotation is manual
- owner identity rotation is manual

Reason:

- certificate renewal should be low-friction routine maintenance
- rotating the node key changes the node's transport identity and should be explicit
- rotating the owner identity is a larger trust event and should always be explicit

Suggested commands:

```bash
mesh-llm auth renew-node
mesh-llm auth rotate-node
mesh-llm auth rotate-node --revoke-current
mesh-llm auth rotate-owner
```

Suggested behavior:

- `auth renew-node`
  - keep the same node key
  - mint a fresh short-lived certificate
- `auth rotate-node`
  - generate a new node key
  - mint a fresh certificate under the current owner
- `auth rotate-node --revoke-current`
  - revoke the current node certificate
  - optionally block the current node endpoint ID
  - generate a new node key
  - mint a fresh certificate
- `auth rotate-owner`
  - create a new owner identity and begin re-enrollment under that owner

### `mesh-llm/src/mesh/`

Peer state stores ownership verification results next to the existing node ID.

Suggested additions:

- `PeerAnnouncement` carries optional owner attestation
- `PeerInfo` stores:
  - advertised owner ID
  - certificate ID
  - verification status
  - expiry
  - optional label
- mesh verification runs after gossip decode and before ownership is surfaced to
  the rest of the node

Important: ownership verification should **not** replace transport admission.

Admission stays:

- QUIC connection identity
- valid gossip frame
- existing protobuf / JSON compatibility checks

Ownership verification is an additional metadata layer.

Suggested verification order:

1. verify the owner signing key matches `owner_id`
2. verify the signature over canonical claim bytes
3. verify `node_endpoint_id` matches the actual QUIC peer identity
4. reject expired certificates
5. reject revoked owners
6. reject revoked certificate IDs
7. optionally reject revoked node endpoint IDs
8. apply local trust policy / allowlist

### `mesh-llm/src/api/`

Expose ownership data through `/api/status` so the console and CLI can display it.

Suggested additions:

- top-level local owner summary
- per-peer owner summary
- counters for verified/unverified/invalid peers
- owner certificate ID for the local node and peers

### `mesh-llm/ui/`

The console consumes the new `/api/status` fields and surfaces operator
attribution, trust state, and warnings.

## Proposed Protocol Changes

This proposal is deliberately additive and non-breaking.

### `/1` protobuf changes

Extend `PeerAnnouncement` with optional owner attestation fields derived from
the existing keystore owner identity.

Suggested shape:

```proto
message SignedNodeOwnership {
  uint32 version = 1;
  string cert_id = 2;
  string owner_id = 3;
  bytes owner_sign_public_key = 4;   // 32 bytes
  bytes node_endpoint_id = 5;        // 32 bytes
  uint64 issued_at_unix_ms = 6;
  uint64 expires_at_unix_ms = 7;     // required
  optional string node_label = 8;
  optional string hostname_hint = 9;
  bytes signature = 10;              // Ed25519 signature over canonical claim bytes
}
```

Then in `PeerAnnouncement`:

```proto
optional SignedNodeOwnership owner_attestation = <next field number>;
```

This allows new nodes to carry verified owner metadata without changing the
existing transport identity.

### Route table changes

Passive nodes only consume the route table, not full gossip. If we want owner
awareness for passive clients, they must receive enough data to verify ownership
locally.

Clean rule:

- do not send `owner_verified` as a forwarded boolean
- either send no owner data in route tables
- or send the full `SignedNodeOwnership` attestation

Recommendation:

- phase 1 keeps owner trust out of route tables
- a later phase may attach full attestations to route-table responses if passive
  clients need owner-aware policy

### API payload changes

Suggested `/api/status` additions:

```json
{
  "owner": {
    "owner_id": "abc...",
    "cert_id": "cert-1234",
    "verified": true,
    "expires_at": 1770000000000,
    "node_label": "studio-mac"
  },
  "peers": [
    {
      "id": "node123",
      "owner": {
        "owner_id": "abc...",
        "cert_id": "cert-5678",
        "verified": true,
        "expires_at": 1770000000000,
        "node_label": "worker-1",
        "status": "verified"
      }
    }
  ]
}
```

Suggested owner status enum:

- `verified`
- `unsigned`
- `expired`
- `invalid_signature`
- `mismatched_node_id`
- `revoked_owner`
- `revoked_cert`
- `unsupported_protocol`

### Nostr listing changes

Not required in phase 1.

Nostr mesh discovery already has a mesh publisher identity. That is not the
same thing as per-node owner identity. We should keep them separate unless we
have a concrete need to expose owner attribution in public discovery.

## Backward Compatibility

This proposal is intended to remain backward compatible.

### No protocol-generation bump

Do **not** bump `NODE_PROTOCOL_GENERATION` for this change.

Reason:

- the change is additive
- protobuf unknown fields are ignored by older `/1` nodes
- existing validation rules can stay intact

### `/0` compatibility

Legacy `/0` links do not carry owner attestation.

Behavior:

- `/0` peers remain valid participants
- they surface as `owner.status = unsupported_protocol` or `unsigned`
- they are never treated as verified owners

### Mixed meshes

In a mixed mesh:

- new nodes verify ownership for other new `/1` nodes
- new nodes treat old `/0` or old `/1` nodes as unattributed
- old nodes ignore the new fields and continue operating normally

### Join and admission behavior

Default behavior stays permissive:

- missing owner attestation does not block admission
- invalid attestation does not break transport compatibility by default
- ownership state is advisory until an explicit strict policy is enabled

However, nodes should emit a visible warning when a previously verified local
owner configuration falls back to unattributed or invalid state.

### Strict policy

If we later add owner-based admission policy, it must be opt-in.

Suggested modes:

- `off`
  - current behavior
- `prefer-owned`
  - warn on unattributed peers
- `require-owned`
  - reject peers without a valid owner attestation
- `allowlist`
  - only accept verified peers whose `owner_id` is explicitly trusted

For compatibility, new policy modes should default to `off`.

## CLI UX Proposal

### Runtime flags

Add runtime controls for using the existing keystore owner identity at the
top-level CLI, not just under `auth`.

Suggested flags:

```bash
mesh-llm --owner-key ~/.mesh-llm/owner-keystore.json
mesh-llm --owner-required
mesh-llm --owner-label studio-mac
mesh-llm --trust-owner <owner_id>
mesh-llm --trust-policy off|prefer-owned|require-owned|allowlist
```

Behavior:

- `--owner-key`
  - load the existing owner keystore and attach ownership to this node
- `--owner-required`
  - fail startup if the owner keystore cannot be loaded or cannot sign
- `--owner-label`
  - human-friendly label embedded in the attestation and shown in the UI
- `--trust-owner`
  - add one or more trusted owner IDs for allowlist policy
- `--trust-policy`
  - choose enforcement mode

### `auth` subcommands

Keep `auth init` and `auth status`, but expand them for workflows built around
the existing keystore owner identity.

Suggested additions:

```bash
mesh-llm auth init
mesh-llm auth status
mesh-llm auth renew-node
mesh-llm auth sign-node --node-key ~/.mesh-llm/key --out ~/.mesh-llm/node-ownership.json
mesh-llm auth verify-node --file ~/.mesh-llm/node-ownership.json
mesh-llm auth rotate-node
mesh-llm auth rotate-node --revoke-current
mesh-llm auth revoke-owner <owner_id>
mesh-llm auth revoke-node --cert-id <cert_id>
mesh-llm auth revoke-node --node-id <node_endpoint_id>
mesh-llm auth rotate-owner
mesh-llm auth trust add <owner_id>
mesh-llm auth trust list
mesh-llm auth trust remove <owner_id>
```

Notes:

- `sign-node` is useful for air-gapped or pre-provisioned machines.
- Runtime can auto-sign on startup, but explicit signing is useful for debugging
  and provisioning.
- Revocation management should distinguish owner-level and node-certificate-level
  revocation.
- Trust management should live under `auth` because it is identity policy, not
  transport policy.

### CLI examples

Single owner, two trusted nodes:

```bash
mesh-llm auth init

mesh-llm --model Qwen3-14B --owner-key ~/.mesh-llm/owner-keystore.json --owner-label studio

mesh-llm --join <TOKEN> --owner-key ~/.mesh-llm/owner-keystore.json --owner-label mini
```

Strict private mesh:

```bash
mesh-llm --model Qwen3-30B \
  --owner-key ~/.mesh-llm/owner-keystore.json \
  --trust-policy allowlist \
  --trust-owner <owner-id-1> \
  --trust-owner <owner-id-2>
```

### CLI output changes

Suggested `auth status` additions:

- owner keystore path
- owner ID
- current certificate ID
- signing key
- unlock source
- trust policy summary
- trusted owner count

Suggested runtime startup output:

```text
Owner identity:  verified
Owner ID:        8bd4...
Cert ID:         cert-1234
Node label:      studio
Ownership cert:  valid until 2026-04-13T12:00:00Z
Trust policy:    off
```

Or when absent:

```text
Owner identity:  not configured
Node will appear as unattributed to peers
```

## Console UX Proposal

### Top-level status

Add an owner card near mesh identity and node identity.

Show:

- owner ID
- certificate ID
- verification badge
- node label
- certificate expiry
- trust policy

### Topology graph

Each node card should display owner state.

Suggested treatments:

- verified owner: green or neutral verified badge
- unsigned legacy node: muted badge
- invalid/expired claim: warning badge

Optional grouping:

- group nodes with the same owner
- or add an owner chip below the node ID

### Peer list and node details

Add columns / fields:

- owner ID
- certificate ID
- owner label
- ownership status
- expiry

This lets operators answer:

- which nodes belong to me
- which nodes belong to the same collaborator
- which nodes are legacy
- which nodes have broken or expired ownership

### Warnings and actions

If trust policy is stricter than `off`, surface visible warnings:

- "2 peers are unattributed"
- "1 peer has an invalid owner signature"
- "allowlist mode rejected 1 join attempt"

Suggested detail views:

- copy owner ID
- explain why verification failed
- show expiry countdown

### Discovery and invite UX

For local mesh management surfaces:

- show whether the local node is signed before copying an invite token
- if running a strict private mesh, explain that only trusted owners can join

For now, public discovery does not need owner badges unless the data is exposed
in Nostr or another API.

## Trust Store Proposal

Store trust and revocation state separately from the owner keystore.

Suggested path:

- `~/.mesh-llm/trusted-owners.json`

Reason:

- keeps identity material separate from policy
- makes sharing trust config optional
- allows read-only distribution of trust policy without copying the private key

Suggested format:

```json
{
  "version": 1,
  "policy": "off",
  "trusted_owners": [
    {
      "owner_id": "abc...",
      "label": "James"
    }
  ],
  "revoked_owners": [
    {
      "owner_id": "deadbeef...",
      "reason": "owner key compromised"
    }
  ],
  "revoked_node_certs": [
    {
      "cert_id": "cert-1234",
      "reason": "node stolen"
    }
  ],
  "revoked_node_ids": [
    {
      "node_endpoint_id": "abcd...",
      "reason": "emergency transport-key block"
    }
  ]
}
```

## Failure Modes

### Owner keystore missing

- Node starts unattributed unless `--owner-required` is set.

### Signature invalid

- Peer remains connected under default policy.
- Ownership status becomes `invalid_signature`.
- Strict policy may reject or quarantine the peer.

### Owner revoked

- All certificates issued by that owner become untrusted locally.
- Status becomes `revoked_owner`.

### Node certificate revoked

- Only that specific node certificate becomes untrusted locally.
- Status becomes `revoked_cert`.

### Node key rotated

- Old ownership claim no longer verifies.
- Node should auto-sign a new claim at startup if the owner key is available.

### Claim expired

- Transport still works under permissive policy.
- Status becomes `expired`.
- Strict policy may reject it.

Recommendation:

- phase 1 should require expiry on all node certificates
- certificate lifetime should be short-lived by default, measured in hours or days

## Compromise Response

Detailed operator guidance lives in:

- [IDENTITY_INCIDENT_RESPONSE.md](IDENTITY_INCIDENT_RESPONSE.md)

Clean response model:

- **node compromise**
  - revoke that node certificate
  - optionally block that node endpoint ID
  - rotate the node identity if the machine returns to service
- **owner compromise**
  - revoke that owner
  - treat all certificates from that owner as untrusted
  - create a new owner identity
  - re-enroll trusted nodes under the new owner

Recommendation:

- a single compromised machine should not force owner rotation by default
- a compromised owner keystore should force owner rotation and broad re-enrollment

### Ephemeral client nodes

Two valid options:

- allow unsigned ephemeral clients by default
- or sign ephemeral clients for the lifetime of the process when an owner key is available

Recommendation:

- phase 1 keeps ephemeral clients unsigned by default
- later add optional signed ephemeral client mode if there is a clear use case

## Rollout Plan

### Phase 1: attribution only

- add signing and verification
- expose status in gossip and `/api/status`
- add console badges
- no enforcement by default

### Phase 2: trust policy

- add trust store
- add `--trust-policy` and allowlist
- add rejection/warning paths

### Phase 3: provisioning

- add explicit offline signing workflows
- add expiry rotation helpers
- optionally expose owner summaries in discovery

## Open Questions

- Should route-table responses include owner metadata in phase 1, or only full
  gossip and `/api/status`?
- Should `--client` with ephemeral node keys be signable, or remain intentionally
  unattributed?
- Should ownership claims be persisted on disk separately, or always re-signed
  at startup?
- Should hostnames be included in claims, or only shown as an unsigned hint?
- Do we want allowlist enforcement on all peers, or only on nodes that can
  serve or host models?

## Recommended Direction

Implement phase 1 as:

- additive protobuf fields on `/1`
- no generation bump
- no admission change
- optional runtime `--owner-key`
- required certificate expiry
- automatic certificate renewal
- manual node-key rotation
- manual owner rotation
- local verification of full attestations only
- per-owner and per-node-certificate revocation support
- `/api/status` owner surfaces
- console badges and grouping

That gives mesh-llm a durable answer to:

> does this node belong to this user?

while preserving current behavior for old nodes, client mode, and mixed meshes.
