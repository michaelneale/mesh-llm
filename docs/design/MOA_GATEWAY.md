# Mixture-of-Agents (MoA) Gateway

Fan out requests to multiple heterogeneous LLM endpoints in parallel,
arbitrate responses with deterministic logic, manage tool call lifecycles,
and return one coherent OpenAI-compatible response.

**Crate:** `crates/mesh-mixture-of-agents/`  
**Virtual model:** `model: "mesh"` (not advertised in `/v1/models`)  
**Status:** Integrated into mesh proxy, live-tested with mesh peers

---

## How it works

```
Agent / Goose / pi
    │
    │  POST /v1/chat/completions { "model": "mesh" }
    ▼
 Mesh proxy (ingress.rs)
   │
   │  intercepts model == "mesh", builds GatewayConfig
   ▼
 MoA handle_turn()
   ├─ role-shaped context slicing (real agent context, not synthetic)
   ├─ parallel fan-out via ModelBackend trait
   ├─ early-exit on consensus (cancel slow workers)
   ├─ deterministic arbiter (code, not models)
   ├─ reducer escalation only on genuine conflict
   └─ native tool_call support through full pipeline
         │
         ├──► fast worker    — system prompt + last msg + tool names
         ├──► specialist     — system prompt + 4 msgs + tool summaries
         └──► strong/reducer — system prompt + 10 msgs + full tool schemas
```

The client thinks it talks to one model. `mesh` is a routing directive
like `auto` — the proxy intercepts it before normal model routing.

---

## Transport

The crate defines a `ModelBackend` trait and ships a default `HttpBackend`
for standalone/test use. The host runtime implements mesh-native backends:

| Backend | Transport | Where |
|---------|-----------|-------|
| `LocalModelBackend` | Direct HTTP to `127.0.0.1:{skippy_port}` | Local model on this node |
| `RemoteModelBackend` | HTTP-over-QUIC tunnel via `node.open_http_tunnel()` | Remote mesh peer |
| `HttpBackend` | Plain HTTP to any URL | Standalone testing |

All worker requests set `mesh_hooks: false` to prevent recursive virtual
LLM consultations (MoA → model → hook → consult another model → ...).

---

## Context packing

Workers get slices of the real agent context, not synthetic prompts. The
agent's actual system prompt, messages, and tool definitions flow through.
The gateway varies depth per role, not content.

| Role | System prompt | Messages | Tools | Max tokens |
|------|:---:|:---:|:---:|:---:|
| Fast | ✅ agent's + one-line preamble | last user msg | names only in system prompt | 256 |
| Specialist | ✅ agent's + one-line preamble | last 4 | name + description | 512 |
| Strong | ✅ agent's + one-line preamble | last 10 | full schemas (native) | 1024 |
| Reducer | ✅ agent's + one-line preamble | worker outputs | full schemas (native) | 2048 |

---

## Arbitration

### Early-exit consensus

Workers are raced in parallel via `gather_workers_incremental()`. After
each response, `try_early_decision()` checks whether enough evidence
exists to return immediately:

| Condition | Action |
|-----------|--------|
| 2+ answers agree, confidence ≥ 0.5 | Return immediately, cancel remaining |
| 2+ tool proposals agree on same tool | Return immediately |
| 1 survivor, all others failed/timed out | Return sole survivor |
| Tool proposal vs answer conflict | Escalate to reducer |
| Low confidence answers | Wait for more workers |

### Deterministic arbiter

The full arbiter (`arbitrate()`) runs when early-exit doesn't fire:

- **Unanimous answers** → highest confidence wins
- **Unanimous tool, same function** → emit tool_call with best arguments
- **Tool vs answer conflict** → escalate to reducer
- **Conflicting tools** → escalate to reducer
- **All uncertain** → escalate to reducer

The reducer is the first model in the config (typically local, zero RTT).
It sees all worker outputs and produces the final decision.

---

## Tool calling

Tool results go to reducer only, not re-broadcast to all workers.

```
Turn 1: Client sends "Read README.md" + tools: [read_file]
  → Workers fan out, strong worker proposes read_file({"path":"README.md"})
  → Arbiter: tool consensus → emit tool_call
  → Client gets: tool_calls: [{read_file, {"path":"README.md"}}]

Turn 2: Client sends tool_result with file contents
  → Gateway detects TurnType::ToolResult → reducer only (no fan-out)
  → Reducer synthesizes tool output into final answer
  → Client gets: content: "The README contains..."
```

---

## Normalization

Models produce unreliable output. The normalizer tries in order:

1. **JSON object parse** — structured envelope with kind/confidence/payload
2. **Line-based KV extraction** — `key: value` lines
3. **Heuristic classification** — pattern matching for tool proposals,
   confidence markers, uncertainty signals

Think tags (`<think>...</think>`) and GLM-style reasoning preambles are
stripped throughout the pipeline.

---

## Crate structure

`crates/mesh-mixture-of-agents/` — zero mesh dependency, 2714 LOC, 29 tests.

| Module | LOC | Tests | Purpose |
|--------|----:|------:|---------|
| `lib.rs` | 714 | 0 | `handle_turn()`, `ModelBackend` trait, `GatewayConfig`, fan-out, SSE |
| `normalize.rs` | 636 | 12 | 3-tier dirty output parsing |
| `arbiter.rs` | 462 | 11 | Deterministic arbitration + early-exit |
| `session.rs` | 456 | 3 | Canonical transcript, tool tracking, turn classification |
| `context.rs` | 276 | 0 | Role-shaped context packing |
| `worker.rs` | 170 | 3 | Role assignment, think-tag stripping |

---

## Integration

The MoA intercept lives in `ingress.rs` (~line 234). When `model == "mesh"`:

1. `build_moa_config()` collects all models from `ModelTargets` + `callable`
   list, deduplicates aliases, wraps each in `LocalModelBackend` or
   `RemoteModelBackend`
2. `handle_turn()` runs the stateless MoA pipeline
3. Response is sent as SSE (streaming clients) or plain JSON

Activation: requires ≥2 distinct models available in the mesh. Returns 503
with explanation if fewer.

---

## Relationship to existing systems

| System | What it does | Relationship to MoA |
|--------|-------------|---------------------|
| `auto` | Routes to best single model | MoA fans out to ALL models |
| Hooks (`virtual_llm.rs`) | Reactive during inference (entropy/drift/image) | MoA is proactive before inference |
| Consult (`consult.rs`) | Single peer consultation over QUIC | MoA does parallel multi-peer |
| Pipeline (`pipeline.rs`) | 2-model plan→execute for code tasks | Complementary, used at ingress line 279 |

Hooks and MoA are independent. Hooks fire reactively during inference.
MoA fires proactively before inference. They could integrate later (hook
signals as quality metadata for arbiter weighting).

---

## Test & evaluation plan

### Unit tests (29 tests, run in CI)

```bash
cargo test -p mesh-mixture-of-agents --lib
```

Covers: arbiter voting (10 scenarios), normalizer parsing (12 scenarios),
session turn classification (3), worker role assignment (3).

### Local smoke test

Start mesh with at least one model + discover mesh peers:

```bash
mesh-llm serve --model "unsloth/GLM-4.7-Flash-GGUF:Q4_K_M" --auto
```

Wait for ≥2 models in `curl -s http://localhost:9337/v1/models`.

**Factual accuracy:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"What is the capital of Japan? One word only."}],"max_tokens":32,"stream":false}'
```
Expected: "Tokyo" (possibly with reasoning preamble from GLM).

**Tool calling:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"Read the file README.md"}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"max_tokens":256,"stream":false}'
```
Expected: `tool_calls` with `read_file({"path":"README.md"})`.

**Streaming:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"What is 7*8?"}],"max_tokens":16,"stream":true}'
```
Expected: SSE chunks with `data: {...}` lines, final `data: [DONE]`.

**Reasoning:**
```bash
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? One sentence."}],"max_tokens":128,"stream":false}'
```
Expected: Correct "No" with valid syllogistic reasoning.

### Agent integration test

```bash
# Goose
GOOSE_PROVIDER=mesh GOOSE_MODEL=mesh goose run

# pi
# Set model to mesh in provider config
```

Verify: tool calls work end-to-end, multi-turn context retained by agent
client, no recursive hook loops.

### Performance characteristics

| Scenario | Expected behavior |
|----------|-------------------|
| 2+ workers agree quickly | Early-exit, faster than single-model |
| 1 worker much faster than others | Returns fast worker if confident |
| Remote peer timeout (30s) | Degrades to local-only, adds latency |
| Only 1 model available | Returns 503, does not activate MoA |
| All workers fail | Returns error response |

### What to watch for

- **GLM chain-of-thought leaking** — GLM uses numbered markdown lists for
  reasoning, not `<think>` tags. When GLM is sole survivor, its internal
  deliberation can leak into the response.
- **30s timeout penalty** — when one of two workers times out, early-exit
  needs ≥2 agreements, so the sole survivor waits for the timeout.
- **Remote peer 503s** — mesh peers can be flaky. Gateway degrades to
  fewer workers but this limits model diversity.
