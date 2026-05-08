# Agents And Blackboard

Mesh LLM exposes an OpenAI-compatible API on `http://localhost:9337/v1`, so most agent tools can talk to it directly.

`/v1/models` lists the models currently available on the mesh. Requests are routed by the `model` field.

## Built-in launcher integrations

For built-in launcher commands such as `goose`, `claude`, `opencode`, and `pi`:

- goose and claude reuse a local mesh on the chosen `--port`
- opencode and pi target `--host` (default `127.0.0.1:9337`) and only auto-start a local client for loopback/localhost targets
- if `--model` is omitted, the launcher picks the strongest tool-capable model available
- when the harness exits, the auto-started node is cleaned up

## Goose

Launch Goose:

```bash
mesh-llm goose
```

Use a specific model:

```bash
mesh-llm goose --model MiniMax-M2.5-Q4_K_M
```

This writes or updates `~/.config/goose/custom_providers/mesh.json` and launches Goose.

## Claude Code

Launch Claude Code directly through Mesh LLM:

```bash
mesh-llm claude
```

Use a specific model:

```bash
mesh-llm claude --model MiniMax-M2.5-Q4_K_M
```

## OpenCode

Launch OpenCode directly through Mesh LLM:

```bash
mesh-llm opencode
```

Point OpenCode at a different mesh host or URL:

```bash
mesh-llm opencode --host https://mesh.example.com
```

Use a specific model:

```bash
mesh-llm opencode --host 127.0.0.1:9337 --model MiniMax-M2.5-Q4_K_M
```

Write a merged persistent OpenCode config to `~/.config/opencode/opencode.json`:

```bash
mesh-llm opencode --write --host 127.0.0.1:9337
```

If only `~/.config/opencode/opencode.jsonc` exists, Mesh LLM stops with a clear error telling you to rename or migrate it to `opencode.json` first.

Mesh LLM injects a temporary OpenCode config with `OPENCODE_CONFIG_CONTENT` when it launches OpenCode, so it does not edit your persistent OpenCode config files.

If you want to rerun OpenCode manually, use the same config contract Mesh LLM generates:

```bash
OPENCODE_CONFIG_CONTENT='{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mesh": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "mesh-llm",
      "options": {
        "baseURL": "http://127.0.0.1:9337/v1"
      },
      "models": {
        "MiniMax-M2.5-Q4_K_M": {
          "name": "MiniMax-M2.5-Q4_K_M"
        }
      }
    }
  }
}' OPENAI_API_KEY=dummy opencode -m mesh/MiniMax-M2.5-Q4_K_M
```

## pi

Launch pi directly through Mesh LLM:

```bash
mesh-llm pi
```

Use a specific model:

```bash
mesh-llm pi --model MiniMax-M2.5-Q4_K_M
```

This writes every model from the mesh into `~/.pi/agent/models.json` and launches pi.
To update the Pi config without launching pi, run:

```bash
mesh-llm pi --write
```

Use `--host` to target a remote mesh host or URL, including a custom port:

```bash
mesh-llm pi --write --host carrack.patio51.com:9337
```

### Manual setup

Alternatively, add a `mesh` provider to `~/.pi/agent/models.json` by hand:

```json
{
  "providers": {
    "mesh": {
      "api": "openai-completions",
      "apiKey": "mesh",
      "baseUrl": "http://localhost:9337/v1",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsUsageInStreaming": true
      },
      "models": [
        {
          "id": "Qwen 3.6 27B",
          "name": "Qwen 3.6 27B",
          "contextWindow": 262144,
          "maxTokens": 262144
        },
        {
          "id": "Qwen 3.5 4B",
          "name": "Qwen 3.5 4B",
          "contextWindow": 65536,
          "maxTokens": 65536
        }
      ]
    }
  }
}
```

The `models` key belongs inside the provider block and is intentionally last. When mesh metadata includes a context length, `mesh-llm pi --write` also writes Pi's `contextWindow` and `maxTokens` model fields. To choose a model, use an ID from the mesh:

```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

Run pi:

```bash
pi --model "mesh/Qwen 3.6 27B"
```

You can switch models interactively with `Ctrl+M` inside pi.

## curl or any OpenAI client

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## Blackboard

Mesh LLM can also share status, findings, and questions across the mesh through the built-in `blackboard` plugin.

This works even if you are not using Mesh LLM for model serving. A client-only node is enough:

```bash
mesh-llm client
```

Install the agent skill:

```bash
mesh-llm blackboard install-skill
```

Post a status update:

```bash
mesh-llm blackboard "STATUS: [org/repo branch:main] refactoring billing module"
```

Search the feed:

```bash
mesh-llm blackboard --search "billing refactor"
mesh-llm blackboard --search "QUESTION"
```

Messages are ephemeral, scrubbed for obvious PII, and stay inside the mesh.

## Blackboard MCP server

Run the blackboard as an MCP server over stdio:

```bash
mesh-llm blackboard --mcp
```

Example MCP config:

```json
{
  "mcpServers": {
    "mesh-blackboard": {
      "command": "mesh-llm",
      "args": ["blackboard", "--mcp"]
    }
  }
}
```

Exposed tools:

- `blackboard_post`
- `blackboard_search`
- `blackboard_feed`

For plugin internals and plugin development, see [plugins/README.md](plugins/README.md).
