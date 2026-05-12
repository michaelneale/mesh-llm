# Mixture-of-Agents (MoA) Gateway

Stateful, tool-aware MoA gateway for distributed heterogeneous models.

**Crate:** `crates/moa-gateway/`
**Branch:** `micn/mixture-of-agents`
**Status:** Working prototype, tested live against 3 models (ollama)

---

## What it is

A stateful gateway that sits between an agent client and N model endpoints.
The client thinks it's talking to one model.  The gateway owns the session,
fans out to workers in parallel, arbitrates with deterministic logic, manages
the full tool call lifecycle, and returns one coherent OpenAI-compatible
response.

```
Agent / Goose / pi
    │
    │  POST /v1/chat/completions
    │  model: "moa"
    ▼
MoA Gateway (stateful, owns session)
  ├─ canonical transcript
  ├─ tool call / result tracking
  ├─ context packer (role-shaped)
  ├─ deterministic arbiter
  └─ worker dispatcher
        │
        ├──► endpoint A  (fast worker)
        ├──► endpoint B  (specialist)
        └──► endpoint C  (strong / reducer)
```

This is a **layer above** the mesh.  The gateway doesn't care whether
endpoints are mesh peers, local skippy/llama-server instances, ollama, or
remote APIs.  It just needs URLs and model names.  The mesh provides model
discovery, QUIC tunnels, and capability metadata — but the core MoA logic
is pure application code with zero mesh dependency.

---

## Key design choices

1. **Deterministic logic first.**  Parsing, normalizing, voting, thresholding,
   tool schema validation are all code.  Models are called (as reducer) only
   when there's genuine semantic ambiguity.

2. **Gateway owns tool lifecycle.**  Workers propose tools; only the gateway
   emits `tool_calls`.  Tool results go to the reducer, not re-broadcast to
   all workers.

3. **Context is role-shaped.**  Full context enters the gateway but each worker
   gets a tailored packet.  Fast workers get the current task + tool names.
   Specialists get recent history + compact tool descriptions.  The reducer
   gets worker outputs + full tool schemas.

4. **Session state across turns.**  The gateway tracks the canonical transcript,
   pending tool calls, tool results, and turn classification.

5. **Transport-agnostic.**  Backends are just `(base_url, model_name)` pairs.

---

## Architecture

### Modules

| Module | Purpose |
|---|---|
| `lib.rs` | `Gateway` struct (stateful), `turn()` entry point, fan-out, reducer dispatch |
| `session.rs` | Canonical session: message history, tool tracking, turn classification |
| `context.rs` | Context packing: role-shaped packets for fast/specialist/strong/reducer |
| `normalize.rs` | Dirty output normalization: JSON → KV → heuristic classification |
| `arbiter.rs` | Deterministic arbitration: unanimous answer, tool consensus, conflict escalation |
| `worker.rs` | Worker dispatch: role assignment, HTTP calls, think-tag stripping |

### Turn flow

```
turn(body) {
  1. session.ingest(messages, tools)
  2. classify_turn() → Fresh | Continuation | ToolResult

  if ToolResult:
    → reducer only (not full fan-out)
    → pack: original task + tool result summary
    → return answer or next tool call

  if Fresh | Continuation:
    → assign roles to endpoints
    → context-pack per role
    → fan out in parallel (JoinSet)
    → normalize dirty outputs
    → arbitrate:
        unanimous answer     → emit highest-confidence
        unanimous tool       → emit tool call
        tool vs answer       → escalate to reducer
        conflicting tools    → escalate to reducer
        all uncertain        → escalate to reducer
    → record response in session
    → return
}
```

### Worker roles

| Role | What it gets | max_tokens |
|---|---|---|
| Fast | Current task, tool names, brief history summary | 256 |
| Specialist | Recent 6 messages, compact tool descriptions | 512 |
| Strong | Recent 10 messages, full tool schemas, system prompt | 1024 |
| Reducer | Worker outputs, full tool schemas, original task | 2048 |

### Worker output contract

Workers are prompted to produce structured envelopes:

```
kind: answer | tool_proposal | critique | uncertainty
confidence: 0.0-1.0
tool: (optional) tool name
arguments: (optional) JSON
payload: response text
```

But models are unreliable.  The normalizer tries in order:
1. JSON object parse
2. Line-based `key: value` extraction
3. Heuristic classification (patterns for tool proposals, critiques, uncertainty)

Anything the model returns is treated as dirty input.

### Arbiter decision tree

```
tool proposals + answers → NeedsReducer (conflict)
tool proposals only, same tool → ToolCall (highest confidence args)
tool proposals only, different tools → NeedsReducer (conflict)
answers only → Answer (highest confidence)
high-confidence tool + no dissent → ToolCall
critique opposing tool → NeedsReducer
all uncertain → NeedsReducer
```

The reducer is the strongest available model.  It sees all worker outputs
and produces the final decision.  Its output goes through the same normalizer
but is accepted without further escalation.

---

## Tool calling (end-to-end)

This is the critical path.  Proven working:

```
Turn 1:
  Client → Gateway: "Look up Bitcoin price" + tools: [web_search]
  Gateway → workers (parallel):
    Fast (llama3.2:3b):     "I can't look up prices" (Answer, conf=0.9)
    Specialist (qwen3:4b):  "Use web_search" (ToolProposal, conf=0.6)
    Strong (qwen3.6:27b):   "Use web_search" (ToolProposal, conf=1.0)
  Arbiter: tool vs answer conflict → NeedsReducer
  Reducer (qwen3.6:27b): web_search({"query": "current price of Bitcoin"})
  Gateway → Client: tool_calls: [{web_search, ...}]

Turn 2:
  Client → Gateway: tool result: "Bitcoin at $104,250, up 2.3%"
  Gateway: TurnType::ToolResult → reducer only (no fan-out!)
  Reducer: "Bitcoin is currently trading at $104,250 USD, up 2.3%"
  Gateway → Client: content: "Bitcoin is currently..."
```

Key: tool results don't trigger full fan-out.  The gateway sends
`original task + tool result` to the reducer only.  This keeps coherency:
one transcript, one tool stream, one authoritative session.

---

## Test results

Tested against 3 ollama models: `llama3.2:3b`, `qwen3:4b`, `qwen3.6:27b`.

| Test | Result |
|---|---|
| Knowledge (factual) | ✅ All workers correct, MoA picks highest-confidence answer |
| Reasoning (bat & ball) | ✅ MoA produces correct $0.05 answer with step-by-step reasoning |
| Tool calling | ✅ Correctly produces `get_weather({location: "Tokyo"})` via arbiter escalation |
| Tool lifecycle | ✅ Full cycle: query → tool_call → tool_result → final answer |
| Code | ✅ Produces palindrome function (though solo llama3.2 was also correct) |

Observed patterns:
- The arbiter correctly escalates when workers disagree (answer vs tool)
- Tool result turns correctly bypass full fan-out
- The reducer correctly synthesizes tool results into final answers
- Think-tag stripping handles Qwen3 reasoning models
- Ollama's `reasoning` field is used as fallback for empty content

Known limitations:
- Large models (27b) time out under ollama's resource sharing when 3 models
  run concurrently on the same GPU — not an issue with mesh-served models on
  separate nodes
- Some reasoning models produce all thinking and no visible content with
  the structured envelope prompt

---

## Integration with mesh-llm

The gateway is a standalone crate with zero mesh dependency.  Integration
into mesh-llm would add:

1. **`model: "moa"` alias** in the proxy ingress — intercept and dispatch
   to `Gateway::turn()` instead of normal routing

2. **Endpoint discovery from mesh** — map `ModelTargets` + remote peers
   into `Vec<Endpoint>` using local ports and QUIC tunnel endpoints

3. **Activation criteria** — only offer `moa` when ≥2 distinct models
   are available in the mesh

4. **Session management** — one `Gateway` instance per client connection
   or per conversation ID

5. **May also serve a model** — the node running the MoA gateway can also
   serve a model itself, which becomes one of the endpoints

---

## Running the test

```bash
# Start ollama with models
ollama pull llama3.2:3b && ollama pull qwen3:4b && ollama pull qwen3.6:27b

# Run the comparison test
cargo run -p moa-gateway --bin moa-test
```
