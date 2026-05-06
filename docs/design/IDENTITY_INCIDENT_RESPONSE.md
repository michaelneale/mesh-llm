# Identity Incident Response

Status: draft runbook for the node owner identity design

This document describes the expected operator workflow when a **node identity**
or **owner identity** is suspected to be compromised.

It assumes the design in [NODE_OWNER_IDENTITY.md](NODE_OWNER_IDENTITY.md):

- owner identity comes from the existing keystore
- node identity is the QUIC endpoint key
- the owner signs short-lived node certificates
- trust and revocation are local policy decisions

## Summary

There are two different compromise classes:

- **node compromise**
  - one machine or one node key is no longer trusted
  - revoke that node certificate
  - optionally block that node endpoint ID
  - rotate the node key
  - issue a new node certificate

- **owner compromise**
  - the owner keystore is no longer trusted
  - revoke the owner everywhere
  - treat all node certificates from that owner as untrusted
  - create a new owner identity
  - re-enroll trusted nodes under the new owner

## Operator Model

The clean UX should split certificate renewal from identity rotation:

- **certificate renewal**
  - automatic
  - happens on startup or before expiry
  - does not change the node ID

- **node identity rotation**
  - manual
  - changes the node ID
  - used for compromise response, reprovisioning, or privacy reset

- **owner identity rotation**
  - manual and high-impact
  - used only when the keystore itself is compromised or retired

## Recommended CLI UX

Suggested commands:

```bash
mesh-llm auth renew-node
mesh-llm auth rotate-node
mesh-llm auth rotate-node --revoke-current
mesh-llm auth revoke-node --cert-id <cert_id>
mesh-llm auth revoke-node --node-id <node_endpoint_id>
mesh-llm auth revoke-owner <owner_id>
mesh-llm auth rotate-owner
```

Suggested behavior:

- `auth renew-node`
  - keep the current node key
  - mint a fresh short-lived node certificate

- `auth rotate-node`
  - generate a new node key
  - mint a new node certificate from the current owner
  - print old node ID and new node ID

- `auth rotate-node --revoke-current`
  - revoke the current node certificate
  - optionally block the current node endpoint ID
  - generate a new node key
  - mint a new node certificate

- `auth revoke-node --cert-id ...`
  - revoke one specific node certificate
  - preferred response for a single compromised node

- `auth revoke-node --node-id ...`
  - emergency transport-key block
  - useful if certificate tracking is incomplete

- `auth revoke-owner ...`
  - revoke the owner itself
  - all certificates from that owner become untrusted

- `auth rotate-owner`
  - create a new owner identity
  - intended for owner compromise or planned migration

## Automatic Behavior

### Certificate renewal

Node certificate renewal should be automatic.

Recommended behavior:

- if no node certificate exists, sign one at startup
- if the certificate is close to expiry, renew it at startup
- if the certificate is expired, renew it before advertising verified ownership

Recommended renewal window:

- renew when less than 20-25% of lifetime remains

Example:

- 7-day certificate
- auto-renew in the final 24-36 hours

### Node key rotation

Node key rotation should **not** be automatic.

Reason:

- rotating node identity changes how the mesh sees that node
- it breaks any policy or operational references tied to the old node ID
- operators should opt into that change explicitly

## Node Compromise Runbook

Use this when:

- one machine is stolen
- one machine is suspected of key theft
- one node key or node certificate should no longer be trusted

### Response goals

- stop trusting the compromised node
- keep the owner identity intact
- avoid disrupting other trusted nodes from the same owner

### Recommended response

1. Revoke the compromised node certificate ID.
2. If needed, revoke the compromised node endpoint ID too.
3. Remove the machine from service.
4. If the machine will return to service, rotate its node key.
5. Mint a new node certificate.
6. Restart and verify the new node appears with a new node ID and certificate ID.

### Preferred command flow

```bash
mesh-llm auth revoke-node --cert-id <old-cert-id>
mesh-llm auth revoke-node --node-id <old-node-id>   # optional emergency block

# on the rebuilt or recovered machine
mesh-llm auth rotate-node
```

### Notes

- Prefer certificate revocation first because it is more precise.
- Use node-ID revocation as a coarse emergency block.
- Do not rotate the owner identity for a single-node incident unless the owner
  keystore was also exposed.

## Owner Compromise Runbook

Use this when:

- the owner keystore file is stolen
- the owner passphrase or keychain unlock is compromised
- you no longer trust the owner signing key

### Response goals

- stop trusting all current certificates from that owner
- establish a new trusted owner identity
- re-enroll known-good machines

### Impact

Owner compromise is much larger than node compromise:

- an attacker can mint fresh certificates for arbitrary node IDs
- revoking only individual node certificates is not sufficient
- all nodes under that owner should be treated as suspect until re-enrolled

### Recommended response

1. Revoke the compromised owner ID everywhere.
2. Remove that owner from trust allowlists.
3. Generate a new owner identity.
4. Re-enroll each trusted machine under the new owner.
5. Prefer rotating node keys on re-enrollment if the old owner keystore was
   present on those machines.
6. Roll out updated trust stores to peers and operators.

### Preferred command flow

```bash
mesh-llm auth revoke-owner <old-owner-id>
mesh-llm auth rotate-owner

# for each trusted machine
mesh-llm auth rotate-node
```

### Notes

- If the old owner keystore was copied to multiple machines, assume those
  machines may all need re-enrollment.
- If you cannot confidently determine which machines were exposed, rotate both:
  - owner identity
  - node identities on all machines under that owner

## Recovery Verification

After any compromise response, verify:

### For node compromise

- old certificate ID is revoked
- old node ID is blocked if emergency block was used
- new node appears with:
  - new node ID if rotated
  - new certificate ID
  - verified owner status

### For owner compromise

- old owner ID appears as revoked
- old owner no longer passes local allowlist checks
- all recovered nodes appear under the new owner ID
- no stale certificates from the old owner are shown as verified

## Console UX Expectations

The console should make incident response legible.

Suggested surfaces:

- node detail panel
  - owner ID
  - certificate ID
  - expiry
  - verification status

- warnings
  - revoked owner
  - revoked certificate
  - expired certificate
  - unattributed node

- actions
  - copy owner ID
  - copy node ID
  - copy certificate ID

## `/api/status` Expectations

For incident response, `/api/status` should expose enough data to identify and
revoke the right entity.

Suggested fields:

```json
{
  "owner": {
    "owner_id": "owner-123",
    "cert_id": "cert-456",
    "status": "verified",
    "expires_at": 1770000000000
  },
  "peers": [
    {
      "id": "node-abc",
      "owner": {
        "owner_id": "owner-123",
        "cert_id": "cert-789",
        "status": "revoked_cert"
      }
    }
  ]
}
```

## Certificate Lifetime Guidance

Recommended default:

- certificates should live for hours or days, not indefinitely

Recommended tradeoff:

- short enough to limit compromise blast radius
- long enough to avoid fragile day-to-day operations

Good starting point:

- 7 days lifetime
- auto-renew in the final 24-36 hours

More sensitive deployments can shorten this.

## What Not To Do

- Do not auto-rotate node IDs silently.
- Do not treat `owner_verified = true` from an intermediary as sufficient proof.
- Do not rely on owner revocation alone for a single-node incident.
- Do not rely on node revocation alone for an owner-keystore compromise.

## Recommended Default Policy

- automatic certificate renewal
- manual node-key rotation
- manual owner rotation
- support both:
  - per-node-certificate revocation
  - per-owner revocation

That gives operators a clean model:

- routine maintenance is automatic
- identity-changing events are explicit
- compromise response can be targeted or global depending on what was exposed
