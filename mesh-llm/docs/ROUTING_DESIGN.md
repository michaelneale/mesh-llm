# Routing & Session Design

## The Problem

A request arrives at a mesh proxy. We need to decide:
1. **Which model** — auto or user-specified
2. **Which host** — one of potentially many serving that model
3. **Sticky or not** — should follow-up requests go to the same host?

These three decisions interact. Getting them wrong wastes GPU memory (cold KV),
sends requests to overloaded hosts, or routes agentic work to weak models.

## Current State (v0.49)

```
Request → proxy → pick model → pick host (round-robin or first-reachable) → tunnel TCP
```

- No session stickiness in the normal path (only MoE has hash-based routing)
- Load signal is wrong: each proxy counts its own inflight, not actual server load
- Auto mode picks one model; if it fails, gives up or falls back

## Proposed Design

### Layer 1: Model Selection

Two modes, both produce a **ranked model list**:

**Auto mode** (`model=auto` or unset):
- Classify the request (agentic/chat/code/reasoning)
- Rank all served models by quality for that category + load penalty
- Agentic: only tool-capable models. Chat: all models (more flexibility)
- Result: ordered list like `[Qwen3-Coder-Next, Qwen2.5-Coder-32B, Qwen3-8B]`

**Named model** (`model=Qwen2.5-Coder-32B`):
- That model is the primary choice
- Alternatives are only tried if ALL hosts for the named model are **unreachable**
  (not busy — unreachable, i.e. connection fails)
- This is important: user asked for a specific model, respect that

### Layer 2: Host Selection (within a model)

Once a model is chosen, pick a host. This is where stickiness and load matter.

**Session affinity:**
- Extract session hint from request (`user` field, `session_id`, or client IP)
- Hash the session hint → sticky host assignment
- Same session always goes to the same host (warm KV cache, faster TTFT)
- Stickiness holds as long as the host is **reachable and not saturated**

**When stickiness breaks:**
- Host is unreachable (connection fails) → try next host, re-bind session
- Host is saturated (all slots full, confirmed by the host) → try next host
- Host left the mesh → re-bind

**Without a session hint:**
- Round-robin or least-loaded (no KV to preserve anyway)

### Layer 3: Load Signal (source of truth)

The load signal must come from the **worker node running llama-server**, not from proxies guessing.

**What the worker knows:**
- llama-server `/health` returns: `{"status":"ok"}` or `{"status":"no slot available","slots_idle":0,"slots_processing":1}`
- llama-server `/slots` returns per-slot state: idle, processing, prompt length, tokens generated
- The worker node can poll this cheaply (localhost HTTP, every 5-10s)

**What gets gossipped:**
```rust
pub struct PeerAnnouncement {
    // ... existing fields ...
    #[serde(default)]
    pub slots_idle: u8,      // how many KV slots are free
    #[serde(default)]
    pub slots_total: u8,     // total slots (usually 1 for big models)
}
```

**Why slots, not inflight:**
- `slots_idle=0` means the GPU literally cannot accept another request right now
- `slots_idle>0` means there's capacity, regardless of how many proxies are sending work
- This is the real signal. A proxy's local inflight count is meaningless in a multi-proxy mesh.

**Staleness (60s gossip):**
- Acceptable for sustained load (agentic sessions last minutes)
- For bursty chat: stale=idle is fine (worst case you send to a now-busy host, it queues)
- For bursty chat: stale=busy is fine (worst case you avoid a host that freed up, hit another)
- The only bad case: stale=available but actually crashed. Connection failure handles that.

### How It All Fits Together

```
Request arrives at proxy
  │
  ├─ auto mode?
  │   ├─ classify request (agentic/chat/code)
  │   └─ rank models by (quality for category - load penalty)
  │       chat: more models eligible, bigger pool
  │       agentic: only tool-capable models
  │
  ├─ named model?
  │   └─ that model is the list (fallback only on unreachable)
  │
  ▼
  For each model in ranked list:
    │
    ├─ session hint present?
    │   ├─ sticky host assigned and healthy? → use it
    │   └─ otherwise → pick least-loaded host, bind session
    │
    ├─ no session hint?
    │   └─ pick least-loaded host
    │
    ├─ try tunnel to host
    │   ├─ success → done
    │   └─ fail → try next host for this model
    │
    └─ all hosts failed → try next model in ranked list
```

### Load Penalty Details

The penalty applies at **model selection** (Layer 1), using the *minimum* slots_idle
across hosts serving that model. This is conservative — if any host has capacity, the
model isn't considered overloaded.

| Scenario | Agentic | Chat |
|----------|---------|------|
| All hosts have idle slots | No penalty | No penalty |
| Some hosts saturated | No penalty (others have capacity) | No penalty |
| All hosts saturated | Penalty (but never hard-block) | Larger penalty |

Chat gets penalized harder → spills to smaller models earlier → frees up big models for agentic.
Agentic holds the line → keeps the best tool model even under load.

### Session Stickiness Details

**Binding:**
- Key: `(model_name, session_hint)` → `host_id`
- Created on first request for a session
- Stored in-memory on the proxy (not gossipped — each proxy has its own bindings)

**Expiry:**
- Binding expires after 10 minutes of no requests (KV cache likely evicted anyway)
- Binding breaks immediately on connection failure

**No session hint:**
- Client IP as fallback? Or just no stickiness.
- Most agentic clients send `user` field. Chat UIs vary.

### What Changes

1. **Worker nodes poll llama-server `/health`** every 5-10s, store slots_idle/slots_total
2. **Gossip includes slots_idle/slots_total** instead of proxy inflight count
3. **Proxy maintains session→host bindings** (simple HashMap with TTL)
4. **Host selection prefers sticky host**, falls back to least-loaded
5. **Model selection uses slots_idle** for load penalty instead of inflight

### What Doesn't Change

- Gossip frequency stays at 60s (no extra traffic)
- Wire protocol is backward compatible (`#[serde(default)]`)
- Named model still gets priority (fallback only on unreachable)
- Never returns 503 when any host is reachable (soft penalties only)

## Host-Side Request Interception (Cooperative Inference)

### The Idea

The host node sits between the mesh and llama-server. Right now it's a dumb TCP
relay (`handle_inbound_http_stream` in `tunnel.rs`). It could be smarter — inspect
the request, detect things llama-server can't handle, and call back into the mesh
for help before forwarding.

### Why the host, not the proxy?

The proxy picks the model and host up front, then tunnels raw bytes. The host node
is where llama-server actually runs — it knows:
- What model is loaded (vision? tools? context limit?)
- What the request contains (images? tool schemas? huge context?)
- What the local llama-server's capacity is (slots, health)

The host node also has a `Node` handle with full mesh access — it can route
sub-requests to other models in the mesh.

### Architecture

```
Current:
  QUIC stream → TCP relay → llama-server → response streams back

Proposed:
  QUIC stream → parse request → [maybe rewrite] → TCP → llama-server
                                      │                     │
                                      │                     └─ response streams
                                      │                        back untouched
                                      │
                                      └─ if needed: sub-request
                                         to mesh (e.g. vision model)
```

No extra process, no extra TCP hop. The interception happens inline in
`handle_inbound_http_stream`. Response streaming is never touched — SSE tokens
flow straight through.

### Latency Impact

**Normal requests (no interception needed):** ~1ms added — parse the JSON request
body, decide nothing special is needed, forward to llama-server. The request body
is fully sent before streaming starts, so there's no streaming latency hit.

**Intercepted requests (e.g. image rewrite):** Added latency of the sub-request
(e.g. 2-5s for a vision model to describe an image). But without this, the request
would have failed entirely, so it's latency vs failure.

The response path is never in the critical loop — it's still a raw byte relay from
llama-server back through QUIC.

### Use Cases

**Vision fallback (most concrete):**
```
User sends image + question → lands on Qwen3-Coder-Next (no vision)
  Host intercepts: sees base64 image, model has no mmproj
  Host calls mesh: POST /v1/chat/completions with image → Qwen3.5-27B (vision)
  Vision model returns: "This is a screenshot of a React component with..."
  Host rewrites: replaces image_url with text description in messages
  Host forwards: rewritten request → local llama-server
  Response streams back normally
```

**Tool schema adaptation:**
- Request has tool schemas but model has `tools: false` in profile
- Host could strip tools and reformulate as a system prompt instruction
- Or call a tool-capable model in the mesh for the tool-calling part

**Context overflow:**
- Request exceeds the model's context window
- Host could summarise earlier messages via a fast small model
- Or truncate intelligently (keep system prompt + recent messages)

**Draft/verification pattern:**
- Small fast model drafts a response
- Host routes to a bigger model to verify/refine
- Returns the refined response

**Mid-inference model consultation:**

The most powerful pattern: llama-server is generating a response and realizes it
needs help. Rather than failing or hallucinating, it consults another model in
the mesh for specific content, then incorporates the result.

This could work at two levels:

*Pre-inference (host intercepts before forwarding):*
- Analyse the request, identify parts that would benefit from specialist help
- E.g. a coding question with a math sub-problem: ask a reasoning model to
  solve the math, inject the solution as context, then let the coder generate
- E.g. a multi-language request: ask a translation model for the non-English
  parts, provide translations as context

*Mid-inference (requires llama-server changes or tool-use loop):*
- llama-server generates a tool call like `{"function": "consult_model", "arguments": {"model": "reasoning", "query": "prove that..."}}`
- Host intercepts the tool call, routes it through the mesh to a reasoning model
- Returns the result as a tool response, llama-server continues generating
- This is essentially **models using other models as tools**

The tool-use loop version is the most natural fit — it uses the existing
tool-calling protocol. The host just needs to recognize mesh-internal tool calls
and handle them before they reach the client:

```
Client sends request → host → llama-server
  llama-server generates: tool_call("consult_vision", {image: ...})
  Host intercepts tool_call (not a real client tool)
  Host routes to vision model in mesh → gets description
  Host feeds tool_result back to llama-server
  llama-server continues generating final response
  Client sees a normal response (no tool calls leaked)
```

This means any model in the mesh can leverage any other model's strengths —
a coder can use a reasoning model for complex logic, a chat model can use a
vision model for images, a small model can escalate hard questions to a bigger
one — all transparently to the client.

### Implementation Path

1. Replace the TCP relay in `handle_inbound_http_stream` with request parsing
2. Buffer request headers + body (already small — JSON payload, not streamed)
3. Check: does request need interception? (image on non-vision model, etc.)
4. If no: forward to llama-server immediately, relay response — minimal overhead
5. If yes: make sub-request via mesh, rewrite, then forward
6. Response path unchanged — raw byte relay from llama-server through QUIC

The host already has `Node` access. Sub-requests use the same mesh routing
(model selection, host selection) as any other request.

### Open Questions

- Should session bindings be shared across proxies (gossip them)? Probably not — adds complexity, and if a client switches proxy entry point, cold KV is the least of the problems.
- Should we use `/slots` for richer signal (how many tokens generated per slot = how close to eviction)? Probably overkill for now.
- llama-server with `--parallel 1` (default for big models): slots_total=1, slots_idle is 0 or 1. Binary signal. Is that enough? Probably yes — it means "busy or not."
- What about split/tensor-parallel setups where one llama-server spans multiple nodes? The host election already handles this — the elected host is the one running llama-server, and it's the one that reports load.
- For cooperative inference: how to avoid loops? (Host A intercepts, calls mesh, lands on Host B which intercepts again.) Tag the request with `X-Mesh-Rewritten: true` or similar.
- Should the host advertise its capabilities (vision, tools, context length) in gossip so the proxy can make smarter choices up front? This would reduce the need for host-side interception but adds gossip complexity.
