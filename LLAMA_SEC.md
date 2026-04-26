# LLAMA_SEC: Hardware-Verified Private Inference for Mesh LLM

Privacy and attestation design for Apple Silicon nodes in mesh-llm.
Goal: a malicious node operator cannot read prompts, and peers can
verify this cryptographically.

## Background

Centralized inference services solve this with a coordinator that
verifies every provider. We don't have a coordinator — we need the
same guarantees in a P2P mesh where every peer verifies every other
peer.

What we already have that helps:
- QUIC/TLS 1.3 on all peer links (transport encryption)
- Owner identity: Ed25519 signing + X25519 encryption (crypto/)
- Node ownership claims: SignedNodeOwnership flows in gossip already
- Trust store: Off / PreferOwned / RequireOwned / Allowlist policies
- Signed+encrypted envelopes: crypto/envelope.rs (SalsaBox + Ed25519)
- PeerAnnouncement carries owner_attestation today

What we don't have:
- Inference payload encryption (prompts are plaintext JSON in proxy/tunnel)
- Hardware attestation (any node can claim anything)
- Runtime hardening (no anti-debug, no integrity checks)
- Any trust gate on host election (pure VRAM, no verification)

---

## Plan

### Phase 1: E2E Inference Encryption

Encrypt prompts from the API entry point to the host node. Tunnel
relays see only ciphertext. Uses the same NaCl box primitives we
already have in crypto/envelope.rs.

#### New field in gossip

```rust
// mesh/mod.rs — PeerAnnouncement
pub(crate) inference_public_key: Option<String>,  // base64 X25519
```

Each node generates an X25519 keypair at startup. The public key flows
in gossip. On Mac nodes with SE (phase 3), this key gets sealed to
hardware.

#### Encryption flow

```
API proxy (localhost:9337)
  ├─ Parse JSON: extract "model" field only
  ├─ Resolve host node for this model (election result)
  ├─ Generate ephemeral X25519 keypair (per-request → forward secrecy)
  ├─ NaCl box: encrypt raw JSON body to host's inference_public_key
  └─ Send: { model, ephemeral_public_key, ciphertext }

Tunnel / QUIC
  └─ Carries opaque bytes. Cannot read prompt. Model field is cleartext
     for routing.

Host node
  ├─ NaCl box.open with inference_private_key + ephemeral_public_key
  ├─ Forward plaintext to local llama-server (localhost:port)
  ├─ Encrypt each response chunk back: NaCl box(inference_priv, ephemeral_pub)
  └─ Stream encrypted chunks back

API proxy
  └─ Decrypt with ephemeral_priv + host's inference_public_key
     Serve to consumer as normal HTTP/SSE
```

#### Wire format

```rust
// protocol/ — new types
struct EncryptedInferencePayload {
    model: String,                  // cleartext for routing
    ephemeral_public_key: String,   // base64, one-time per request
    ciphertext: String,             // base64, NaCl box of full JSON body
}

struct EncryptedResponseChunk {
    ciphertext: String,             // base64, NaCl box of SSE chunk data
}
```

#### Where to change

- `network/openai/transport.rs` — before forwarding request, encrypt
  body if host has inference_public_key. On response, decrypt chunks.
- `network/openai/ingress.rs` — thread ephemeral keypair through
  request context for response decryption.
- `network/tunnel.rs` — no changes. Already carries opaque bytes.
- `mesh/mod.rs` — add inference_public_key to PeerAnnouncement and
  PeerInfo. Add to gossip encoding/decoding.
- `crypto/` — add lightweight NaCl box helper (the envelope format
  has owner identity overhead we don't need per-request).

#### Policy

Three levels, most important first:

**Client-local** (primary mechanism — no coordination needed):
```
mesh-llm --require-attested-hosts    # only send MY prompts to attested nodes
mesh-llm --encrypt-inference         # only send MY prompts encrypted
```
This is purely local. Your node already has every peer's attestation
and inference key from gossip. It just filters before routing. No
mesh-wide agreement needed. Your node, your rules.

**Mesh-wide** (convenience — originator sets a minimum bar):
```
mesh-llm --mesh-policy require-hardware-attest   # all nodes must attest
```
Propagates in gossip. Old unattested nodes excluded from election.
Useful when the mesh originator wants uniform guarantees.

**Gateway** (for external clients):
```
mesh-llm --gateway --require-attested-hosts
```
The gateway enforces on behalf of API clients who aren't mesh peers.

The client-local flag is the most important. A client shouldn't have
to trust the mesh originator to set the right policy — they enforce
their own standard. Default: off (backward compatible).

#### MoE / pipeline notes

- MoE with expert sharding: each shard has its own llama-server. API
  proxy picks one node via session hash. E2E encryption targets that
  node. Same as non-MoE.
- Pipeline parallel: host runs llama-server, workers run rpc-server.
  Only the host decrypts prompts. Tensor traffic between workers is
  numerical (not text) and already QUIC-encrypted.
- Virtual LLM / consult: the planner model sees prompts. If planner
  runs on a different node, its payload also needs encryption to that
  node. The consult flow in inference/consult.rs would need to encrypt
  the forwarded prompt to the consulted peer's inference key.

#### What this buys

Relay nodes, gossip peers, and network observers cannot read prompts.
Only the elected host decrypts. This is the single biggest privacy
improvement — it makes public meshes usable for sensitive workloads.

#### What this doesn't solve

The host node sees plaintext. A malicious host can exfiltrate. That's
what phases 2-4 address.

---

### Phase 2: Runtime Hardening

Make it hard for the operator to inspect the running process. These are
all syscalls and checks — no dependencies, small code, big effect.

#### New module: system/hardening.rs

```rust
pub struct SecurityPosture {
    pub sip_enabled: bool,
    pub rdma_disabled: bool,
    pub debugger_blocked: bool,
    pub core_dumps_disabled: bool,
    pub env_scrubbed: bool,
    pub binary_hash: String,
}

/// Call at startup before joining mesh. Returns posture report.
pub fn harden_runtime() -> Result<SecurityPosture, String> {
    // 1. PT_DENY_ATTACH (macOS): block all debugger attachment
    //    ptrace(PT_DENY_ATTACH, 0, null, 0)
    //    Even root can't override while SIP is on.
    deny_debugger_attachment()?;

    // 2. RLIMIT_CORE = 0: no core dumps (could contain plaintext prompts)
    disable_core_dumps()?;

    // 3. Scrub: DYLD_INSERT_LIBRARIES, LD_PRELOAD, PYTHONPATH, etc.
    scrub_dangerous_env();

    // 4. Check SIP via /usr/bin/csrutil status
    let sip = check_sip_enabled();

    // 5. Check RDMA via /usr/bin/rdma_ctl status
    let rdma_off = check_rdma_disabled();

    // 6. SHA-256 of own binary (for attestation blob)
    let hash = self_binary_hash()?;

    Ok(SecurityPosture {
        sip_enabled: sip,
        rdma_disabled: rdma_off,
        debugger_blocked: true,
        core_dumps_disabled: true,
        env_scrubbed: true,
        binary_hash: hash,
    })
}
```

These functions are standard macOS security hardening calls. All
macOS-specific with `#[cfg(target_os = "macos")]` fallbacks.

#### Gossip integration

```rust
// mesh/mod.rs — PeerAnnouncement additions
pub(crate) security_posture: Option<SecurityPosture>,
```

Peers see each other's security posture. Used by election (phase 4)
and displayed in the console UI.

#### Non-Mac nodes

Linux nodes report what they can: binary hash, no core dumps, env
scrubbed. SIP/RDMA/PT_DENY_ATTACH are macOS-only. The posture struct
has booleans for each — peers see exactly what's hardened and what isn't.

---

### Phase 3: Secure Enclave Attestation

Hardware-backed proof that a node is running on genuine Apple Silicon
with the security posture it claims.

#### What the Secure Enclave gives us

- A P-256 private key that lives in tamper-resistant hardware
- The key cannot be exported, copied, or accessed by software
- Signing operations happen inside the SE — only the signature comes out
- SIP protects the SE interface from userland tampering

#### Attestation blob

```rust
struct HardwareAttestation {
    // What hardware
    chip_name: String,              // "Apple M4 Max"
    hardware_model: String,         // "Mac16,1"
    unified_memory_bytes: u64,      // real hardware RAM — prevents VRAM spoofing

    // Security state (from system queries, not self-reported)
    sip_enabled: bool,
    secure_boot_enabled: bool,
    rdma_disabled: bool,

    // Key binding — ties this attestation to a specific mesh identity
    node_endpoint_id: String,       // iroh endpoint ID
    inference_public_key: String,   // X25519 from phase 1
    se_public_key: String,          // P-256 SE public key

    // Integrity
    binary_hash: String,            // SHA-256 of mesh-llm binary

    // Freshness
    timestamp_unix_ms: u64,
}

struct SignedHardwareAttestation {
    attestation: HardwareAttestation,
    se_signature: String,  // P-256 ECDSA, signed inside SE
}
```

#### SE helper

Two options:

**Option A: Swift helper binary** — a standalone Swift package wrapping Security.framework.
Build with `swift build`, call via C FFI or shell out to CLI:

```
mesh-llm-enclave attest --node-id <id> --inference-key <key>
mesh-llm-enclave sign --data <base64>
mesh-llm-enclave info
```

Pro: proven code, clean separation. Con: extra binary to build/ship.

**Option B: Rust native via security-framework crate** — call Apple's
Security.framework directly from Rust:

```rust
use security_framework::key::*;

// Create P-256 key in Secure Enclave
let key = SecKey::generate(KeyType::ec_p256(), 256,
    &GenerateKeyOptions::new()
        .set_token(Token::SecureEnclave))?;

// Sign data
let signature = key.sign(Algorithm::ECDSASignatureMessageX962SHA256, &data)?;
```

Pro: single binary, no Swift dependency. Con: security-framework crate
may not expose all SE features (need to verify key permanence and
access control).

**Recommendation**: Start with Option A (Swift helper). It's proven and
the Swift enclave helper is clean and self-contained. Ship as part of the bundle.
Migrate to native Rust later if the Swift dependency is a pain point.

#### X25519 key sealing

The inference private key (from phase 1) should be sealed to the SE:

```
1. SE creates P-256 key (permanent, in hardware)
2. Derive shared secret: ECDH(SE_priv, ephemeral_P256_pub)
3. AES-GCM encrypt the X25519 secret key using derived key
4. Write sealed blob to ~/.mesh-llm/inference_key.sealed
5. On startup: unseal via SE (SE key never leaves hardware)
```

This means: copy the disk to another Mac → can't decrypt the inference
key → can't impersonate this node.

This uses ECIES (X963SHA256 + AES-GCM) — a standard pattern for
sealing secrets to hardware keys.

#### Gossip integration

```rust
// mesh/mod.rs — PeerAnnouncement
pub(crate) hardware_attestation: Option<SignedHardwareAttestation>,
```

Peers verify on receipt of announcement:

```rust
fn verify_hardware_attestation(
    att: &SignedHardwareAttestation,
    peer_endpoint_id: &str,
    peer_inference_key: &str,
) -> AttestationResult {
    // 1. Verify P-256 ECDSA signature against se_public_key
    // 2. Check node_endpoint_id matches the QUIC peer identity
    // 3. Check inference_public_key matches what peer advertises
    // 4. Check sip_enabled == true, secure_boot_enabled == true
    // 5. Check timestamp within 10 minutes
    // 6. Optionally: check binary_hash against blessed set
}
```

Verification is ~0.1ms per peer. Runs once on join and on re-attestation.

#### Challenge-response (continuous verification)

Centralized services challenge providers every 5 minutes. We do the
same peer-to-peer during heartbeats:

```
Peer A → Peer B: heartbeat { nonce: random_32_bytes }
Peer B → Peer A: heartbeat_ack { nonce_signature: SE_sign(nonce) }
```

Peer A verifies the signature against B's known SE public key. This
proves the SE key is still live on that hardware right now.

If a peer fails 3 consecutive challenges → downgrade to untrusted.
Stop routing inference to it.

Integrate with existing mesh/heartbeat.rs — the heartbeat probe
already exists, just add nonce + signature fields.

#### Trust policy extension

```rust
pub enum TrustPolicy {
    Off,                        // existing
    PreferOwned,                // existing
    RequireOwned,               // existing
    Allowlist,                  // existing
    RequireHardwareAttest,      // NEW: Mac SE attestation required
    RequireVerifiedBinary,      // NEW: attestation + binary hash in blessed set
}
```

Mesh originator sets this. Propagates in gossip. Nodes that can't meet
the policy are visible in the mesh but excluded from election and
inference routing.

#### Blessed binary hashes

For `RequireVerifiedBinary`, someone needs to publish the set of
acceptable binary hashes. Options:

1. **Mesh originator publishes in gossip** — the originator (who creates
   the invite token) includes a `blessed_hashes: Vec<String>` in mesh
   config. Peers only route to nodes matching these hashes.

2. **GitHub releases** — the release artifact includes SHA-256 hashes.
   Nodes can optionally verify against a URL or embedded list.

3. **Owner-signed hash list** — an owner signs a list of blessed hashes
   with their Ed25519 key. Peers verify the owner's signature. This
   is the most decentralized option.

Start with option 1 (originator publishes). It's the simplest and fits
the existing mesh config pattern.

---

### Phase 4: Verified Election

Currently `inference/election.rs` picks the host by highest VRAM. A
malicious node can spoof VRAM to win election and see all prompts.

#### Changes to election

```rust
fn should_be_host(
    candidates: &[PeerInfo],
    model: &str,
    mesh_policy: &TrustPolicy,
) -> Option<EndpointId> {
    let eligible: Vec<_> = candidates.iter().filter(|p| {
        match mesh_policy {
            TrustPolicy::RequireHardwareAttest
            | TrustPolicy::RequireVerifiedBinary => {
                // Must have valid hardware attestation
                p.hardware_attestation_valid()
                // Use attested unified_memory_bytes, not self-reported vram
                && p.attested_memory_bytes().is_some()
                // Must be runtime-hardened
                && p.security_posture.as_ref()
                    .map(|s| s.sip_enabled && s.debugger_blocked)
                    .unwrap_or(false)
            }
            _ => true  // Off/PreferOwned/RequireOwned: no hardware gate
        }
    }).collect();

    // Among eligible: sort by verified memory (attested) not claimed VRAM
    // Tie-break by endpoint ID for determinism
    eligible.iter()
        .max_by_key(|p| {
            let mem = p.attested_memory_bytes()
                .unwrap_or(p.vram_bytes);
            (mem, &p.id)
        })
        .map(|p| p.id.clone())
}
```

Key change: in verified meshes, election uses `unified_memory_bytes`
from the hardware attestation (signed by SE, can't be spoofed) instead
of self-reported `vram_bytes` from gossip.

#### Routing gate

The API proxy should also verify before sending encrypted payloads:

```rust
// network/openai/transport.rs — before encrypting and forwarding
if mesh_policy.requires_attestation() {
    let host = peers.get(&host_id)?;
    if !host.hardware_attestation_valid() {
        // Don't send prompts to unattested host
        return Err(RoutingError::HostNotAttested);
    }
}
```

---

### Phase 5 (Future): RDMA / Memory Isolation

Not for initial implementation. Document for later.

Hypervisor.framework can create Stage 2 page tables making inference
memory invisible to Thunderbolt 5 RDMA (80 Gb/s DMA attack). This
works best when inference runs in-process (e.g. MLX) and the process
controls the memory allocator.

mesh-llm runs llama-server as a child process. Can't VM-map another
process's memory without OS-level changes.

**For now**: check RDMA status and refuse to serve if enabled. RDMA is
disabled by default on macOS. This covers the threat practically.

```rust
// In harden_runtime():
if !check_rdma_disabled() {
    return Err("RDMA enabled — refusing to serve. \
        Disable in System Settings → Sharing → Remote Direct Memory Access");
}
```

**Later options** (if needed):
- Run llama-server inside a Virtualization.framework VM (full isolation)
- Modify llama.cpp to accept a pre-mapped shared memory region
- Switch to in-process inference (mlx-rs or similar) and use the
  Hypervisor pool allocator directly

---

## Implementation Order

Phase 1: E2E inference encryption        ~1-2 weeks
  - inference_public_key in gossip
  - NaCl box encrypt/decrypt in transport.rs
  - Encrypted wire format
  - --encrypt-inference flag

Phase 2: Runtime hardening               ~2-3 days
  - system/hardening.rs module
  - PT_DENY_ATTACH, SIP check, RDMA check, env scrub, binary hash
  - SecurityPosture in gossip
  - RDMA refuse-to-serve gate

Phase 3: SE attestation                  ~1-2 weeks
  - SE helper (Swift binary or Rust native)
  - HardwareAttestation struct + signing
  - Gossip integration
  - Peer-side verification
  - Challenge-response in heartbeats
  - X25519 key sealing
  - TrustPolicy extensions

Phase 4: Verified election               ~2-3 days
  - Attestation-aware election
  - Routing gate
  - Attested memory instead of self-reported VRAM

Phase 5: RDMA / memory isolation         future
  - Hypervisor.framework pool (requires in-process inference)
  - Or: Virtualization.framework for llama-server isolation

---

## File Map (where changes go)

```
mesh-llm/src/
  crypto/
    inference_key.rs       NEW — X25519 inference keypair, NaCl box helpers
    se_attestation.rs      NEW — HardwareAttestation struct, SE signing,
                                  verification, challenge-response
    (mod.rs)               ADD — pub use new modules
  system/
    hardening.rs           NEW — runtime hardening checks, SecurityPosture
  mesh/
    mod.rs                 MOD — add inference_public_key, security_posture,
                                  hardware_attestation to PeerAnnouncement/PeerInfo
    gossip.rs              MOD — encode/decode new fields
    heartbeat.rs           MOD — add attestation challenge nonce + signature
  network/
    openai/transport.rs    MOD — encrypt/decrypt inference payloads
    openai/ingress.rs      MOD — thread ephemeral keys through request context
  inference/
    election.rs            MOD — attestation-aware election, verified VRAM
    consult.rs             MOD — encrypt prompts to consulted peer

enclave/                   NEW directory (Swift SE helper)
  mesh-llm-enclave         Swift CLI: attest, sign, info
  (or native Rust via security-framework)
```

## Client-Side Privacy Guarantees (Without Centralization)

### Why centralization isn't required

In centralized inference services, the consumer trusts the coordinator
to verify providers. The consumer has no direct proof — it's transitive
trust: "I trust the coordinator, the coordinator verified the provider."

In mesh-llm, the consumer's own node IS a mesh participant. It verifies
everything directly:

```
Consumer app → localhost:9337 (own mesh-llm node)
                    │
                    ├─ Receives attestation via gossip
                    ├─ Verifies P-256 SE signature (Apple hardware root of trust)
                    ├─ Checks security posture (SIP, RDMA, binary hash)
                    ├─ Sends heartbeat challenges, verifies responses
                    ├─ Encrypts prompt to host's hardware-sealed key
                    └─ Routes ONLY to peers that pass all checks
```

The trust chain: Consumer → own machine → Apple Secure Enclave on
remote peer. No third party. The local node has first-hand cryptographic
proof of the remote peer's hardware identity and security state.

This is stronger than the centralized model: you trust your own machine
(which you physically control) rather than trusting a company's cloud
coordinator. Centralized consumers must trust that:
- The coordinator operator is honest
- The Confidential VM is real and correctly configured
- The coordinator actually enforces attestation checks
- "We never log prompt content" is true

mesh-llm's consumer verifies everything locally. No faith required.

### What the local node can expose to clients

The API proxy on :9337 can include verification metadata in responses:

```
X-Mesh-Provider-Attested: true
X-Mesh-Provider-SE-Key: <fingerprint>
X-Mesh-Provider-Binary-Hash: <sha256>
X-Mesh-Provider-SIP: enabled
X-Mesh-Provider-Hardened: true
X-Mesh-Encrypted: true
```

A consumer app can check these headers. The local node is the one
making the claims, and the local node verified them cryptographically.
This is analogous to how a browser shows a TLS lock icon — the browser
(local software) verified the certificate chain, not a third party.

For programmatic clients, a verification endpoint:

```
GET /v1/mesh/attestation/<model>
→ Returns the current host's full attestation blob + verification status
   Client can independently verify the SE signature if they want.
```

### Where centralization helps: external API access

If someone hits the mesh API from outside — a mobile app, a SaaS
backend, a third-party integration that doesn't run mesh-llm — they
can't verify attestations themselves. They're trusting the node
running the API endpoint.

For this use case, an optional central proxy makes sense:

#### Optional: Verified Gateway Mode

A mesh-llm node can run in "gateway" mode — it exposes the API
publicly and serves as a verification proxy for external clients:

```
External client → HTTPS → Gateway node → QUIC → Mesh peers
                              │
                              ├─ Verifies all peer attestations
                              ├─ Encrypts prompts to attested hosts only
                              ├─ Provides attestation proof to clients
                              └─ Optionally: adds billing/auth/rate-limiting
```

This is essentially what a centralized coordinator does, but:
1. It's optional — the mesh works without it
2. Anyone can run one — not a single company's infrastructure
3. Multiple gateways can exist for the same mesh
4. The gateway is a standard mesh-llm node with a flag, not
   special software

```
mesh-llm --gateway --bind 0.0.0.0:443 --tls-cert cert.pem
```

The gateway adds:
- TLS termination for external clients
- API key auth (optional)
- Rate limiting (optional)
- Attestation verification on behalf of external clients
- Response header: which peer served, attestation status

This gives external clients equivalent privacy guarantees without
requiring a single-vendor coordinator. Multiple orgs
can run gateways for the same mesh. Clients choose which gateway to
trust — or run their own node for zero-trust.

#### Membership gating via gateway

A gateway operator can set mesh policy:

```
mesh-llm --gateway --trust-policy require-hardware-attest \
         --blessed-hashes hashes.txt
```

Only peers meeting the policy participate in inference routing.
The gateway operator is choosing to centralize verification for
their clients — but the mesh itself remains P2P.

This is the "someone could make a central proxy and gate membership"
scenario. It works as a layer on top, not a replacement for the P2P
verification.

### Summary: what requires centralization and what doesn't

```
Capability                          Centralized?   Why
──────────────────────────────────────────────────────────
Attestation verification            No — each peer verifies directly
E2E prompt encryption               No — consumer node encrypts to host
Trust policy enforcement            No — each node enforces locally
Continuous challenge-response       No — peer-to-peer heartbeats
VRAM spoofing prevention            No — attested memory in gossip
Binary hash verification            No — blessed list in mesh config

API access for external clients     Helps — gateway mode (optional)
Billing / usage metering            Yes — needs a settlement layer
Blessed hash list management        Helps — originator or gateway publishes
Revocation (ban a bad node)         Both — gossip revocation works P2P,
                                    gateway can enforce centrally too
```

The core privacy guarantees are P2P. Centralization is additive — it
helps with external access, billing, and operational convenience, but
the cryptographic verification doesn't depend on it.

---

## Comparison: What This Gets Us vs Centralized Services

```
Feature                    Centralized        Mesh LLM (after)
─────────────────────────────────────────────────────────────
E2E prompt encryption      ✓ (NaCl box)       ✓ (same primitives)
Forward secrecy            ✓ (per-request)     ✓ (per-request)
SE attestation             ✓ (P-256 in SE)     ✓ (same)
SE key sealing             ✓ (ECIES)           ✓ (same)
Anti-debug                 ✓ (PT_DENY_ATTACH)  ✓ (same)
SIP verification           ✓                   ✓
RDMA check                 ✓                   ✓
Binary integrity           ✓ (self-hash)       ✓ (self-hash)
Memory isolation           ✓ (Hypervisor.fw)   ✗ (RDMA check only, for now)
Challenge-response         ✓ (coordinator)     ✓ (peer-to-peer heartbeat)
Verified routing           ✓ (coordinator)     ✓ (verified election)
VRAM spoofing prevention   ✓ (attested)        ✓ (attested memory in election)
Non-Mac support            ✗                   ✓ (tiered trust)
Central trust dependency   ✓ (coordinator)     ✗ (P2P, SE root of trust)
Model sharding             ✗                   ✓ (pipeline + MoE)
```

The only gap is full Hypervisor.framework memory isolation, which needs
in-process inference to work properly. The RDMA-disabled check covers
the practical threat (RDMA is off by default; enabling requires Recovery
OS boot).
