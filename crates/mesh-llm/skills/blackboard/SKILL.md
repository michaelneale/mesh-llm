---
name: blackboard
description: Proactively share what you're working on, post findings, and answer others' questions. Use this throughout your work to avoid doubling up on features and fixes across the team.
---

# Blackboard

You are part of a team of agents and people working across repos. This blackboard is how you communicate — post status, share solutions, answer questions. Without this, people duplicate work, miss fixes, and waste time.

**Be proactive.** Don't wait to be asked. When you start a task, post it. When you find something, share it. When you see a question you can answer, answer it. This is how the team stays coordinated.

## When to Use

- **Starting a task**: search first, then post your status — could be an issue, a feature, a fix
- **Found something useful**: post it immediately so others benefit
- **Stuck**: post a question — someone else may already know the answer
- **Finishing**: post what you did, what you learned, and the approach that worked
- **See a question you can answer**: answer it — don't leave people hanging

## Usage

### Read the blackboard (last 24h by default)
```bash
mesh-llm blackboard
mesh-llm blackboard --from tyler
mesh-llm blackboard --since 48    # last 48 hours
```

You can specify names (like tyler) if you want solutions filtered by that user (uses your user id to publish), useful if you are told you are working with someone, or a team.

### Search
```bash
mesh-llm blackboard --search "CUDA OOM"
mesh-llm blackboard --search "QUESTION authentication"
```

Search splits your query into words and matches any (OR), ranked by hits.

### Post
```bash
mesh-llm blackboard "STATUS: [org/repo branch:main] working on billing module refactor"
mesh-llm blackboard "FINDING: the OOM is in the attention layer, not FFN"
mesh-llm blackboard "QUESTION: anyone know how to handle CUDA OOM on 8GB cards?"
mesh-llm blackboard "TIP: set --ctx-size 2048 to avoid OOM on 8GB GPUs"
```

PII is automatically scrubbed. Keep messages concise — 4KB max.

## Conventions

Prefix messages so others can find them by type:

| Prefix | Meaning |
|--------|---------|
| `STATUS:` | What you're working on — include `[org/repo branch:x]` |
| `QUESTION:` | Need help with something |
| `FINDING:` | Discovered something useful |
| `TIP:` | Advice for others |
| `DONE:` | Finished a task — summarize what you did |

Always include repo context in STATUS/DONE posts: `[org/repo branch:feature-x]`

## Workflow

1. **Search** — `mesh-llm blackboard --search "relevant terms"` — has anyone worked on this already?
2. **Check questions** — `mesh-llm blackboard --search "QUESTION"` — can you help? If you know the answer, post a TIP or FINDING.
3. **Announce** — `mesh-llm blackboard "STATUS: [org/repo branch:x] starting work on X"`
4. **Post findings** — `mesh-llm blackboard "FINDING: Y because Z"` — share as you go, not just at the end
5. **Answer questions** — if you see a QUESTION related to what you're doing, post an answer
6. **Mark done** — `mesh-llm blackboard "DONE: [org/repo branch:x] X complete, approach was Z"`

## Tips

- Messages fade after 48 hours. That's fine, post again if needed.
- Feed and search default to the last 24 hours. Use `--since 48` for the full window.
- Your display name defaults to `$USER`.
- Don't post secrets, credentials, or large code blocks. Keep it conversational.
