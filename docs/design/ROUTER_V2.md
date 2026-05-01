# Router V2: Adaptive Multi-Stage Routing

## Status: Design proposal

## Problem

Router v1 (current) is a static keyword classifier. It detects Code/Reasoning/Creative/ToolCall/Chat and picks the best model by strength match. This works for obvious cases ("write a Python function") but fails on:

- Ambiguous requests that need judgment
- Quality verification — did the model actually answer well?
- Adapting when the first attempt fails
- Distinguishing "quick fact" from "deep analysis" within Chat
- Using cheap signals before expensive ones

## Design

Single API endpoint. Router internally infers intent, tracks state, and adapts.

### 1. Intent Classification (cheap → expensive)

Three tiers, evaluated in order. Stop as soon as confident.

**Tier 0: Heuristics (µs, no model)**
- Tool hints: `tools:[]` present → needs_tools=true
- File extensions in content: `.py`, `.rs`, `.tsx` → code
- Prompt patterns: "fix", "debug", "refactor" → code-fix
- Length: <20 tokens → quick-fact
- System prompt: "developer", "coding assistant" → code
- Previous turns: continuation of code conversation → code

**Tier 1: Embedding similarity (ms, cached)**
- Embed the user message (last turn only)
- Compare against pre-computed cluster centroids:
  - code-fix, code-generate, repo-search, factual-lookup, reasoning-chain, creative-write, casual-chat, tool-use, desktop-action
- Cosine similarity → top cluster + confidence score
- If confidence > 0.8, route immediately

**Tier 2: Small classifier model (100ms, 3-8B)**
- Only invoked when Tier 0+1 are ambiguous
- Structured output: `{ category, needs_repo, needs_tools, needs_vision, complexity: "quick"|"moderate"|"deep", latency_budget_ms }`
- Uses the smallest loaded model (Qwen3-4B, Qwen2.5-3B)
- Cached by message hash for repeated patterns

### 2. Route Profile

Classification produces:

```
RouteProfile {
    category: Code | Reasoning | Creative | ToolCall | QuickFact | DeepAnalysis | Chat,
    attributes: {
        needs_tools: bool,
        needs_repo: bool,      // future: code context
        needs_vision: bool,    // future: multimodal
        complexity: Quick | Moderate | Deep,
        latency_budget_ms: u32,
    },
    confidence: f32,           // 0.0-1.0 from classification
}
```

Model selection uses this:
- `Quick` + `Chat` → smallest fast model
- `Deep` + `Reasoning` → biggest available model
- `Code` + `needs_tools` → tool-capable code model
- `Quick` + factual → small model, fast response
- Low confidence → hedge (send to two models)

### 3. Request State Machine

Each request moves through states:

```
New → Candidate → [Accepted | ChecksFailed → Repairing | Hedged]
                                    ↓
                              [Accepted | Rejected → Escalated]
```

- **New**: classify, pick route
- **Candidate**: first model attempt in flight
- **Accepted**: confidence checks pass, return to caller
- **ChecksFailed**: deterministic checks failed (syntax, schema, truncation)
- **Repairing**: retry with same or different model, injecting error context
- **Hedged**: low confidence → parallel attempt with a second model
- **Escalated**: failed multiple times → use the biggest model available

Most requests: New → Candidate → Accepted (single pass, no overhead).

### 4. Confidence Scoring (cheap first)

When a completion arrives, score confidence:

**Deterministic checks (µs, always run)**
- Response not empty
- Not truncated (finish_reason != "length")
- Valid JSON if JSON was requested
- Code blocks parse (basic syntax check)
- Response length proportional to request complexity

**Heuristic checks (ms, cheap)**
- Repetition detection (model stuck in a loop)
- Hallucination signals (made-up URLs, impossible dates)
- Coherence with the question (embedding similarity of response to query)
- Historical win-rate for this model × category combination

**Tool checks (10-100ms, when applicable)**
- If code: does it compile/lint?
- If structured output: schema validation
- If factual: retrieval grounding against known sources

**Judge model (100ms+, only when needed)**
- Only invoked when:
  - Multiple candidates survive hedging
  - Confidence is in "medium" bucket (0.4-0.7)
- Small 7-14B model compares outputs using rubric
- NOT used for clear accept/reject

Confidence buckets:
- **Accept** (>0.8): return immediately
- **Medium** (0.4-0.8): judge if hedged, else accept
- **Weak** (0.2-0.4): retry with different model class
- **Reject** (<0.2): escalate to biggest model

### 5. What This Means For mesh-llm

**Phase 1 (current, v1):** Static keyword classifier + strength-match scoring + tool filter. No state machine. Works.

**Phase 2 (next):** Add speed tracking. Fast models score higher. Quick-fact requests naturally go to small fast models.

**Phase 3:** Add complexity detection to classifier. Split Chat into Quick/Moderate/Deep. Route quick to small, deep to big. Still heuristic, no LLM classifier.

**Phase 4:** Deterministic confidence checks on responses. Detect truncation, repetition, empty responses. Auto-retry on failure with a different model.

**Phase 5:** Embedding-based classification. Pre-compute centroids for task clusters. Better than keywords for ambiguous requests.

**Phase 6:** LLM classifier + hedging + judge. Full state machine. Only makes sense at scale with 5+ models.

### Key Principle

Most quality decisions should come from **tool checks and heuristics**, not model-vs-model comparison. The LLM judge is the last resort, not the default path. 90% of requests should be: classify → route → return. No hedging, no judge, no retry.
