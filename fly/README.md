# mesh-llm Fly.io Console

Fly app running mesh-llm in `--client` mode — no GPU, just QUIC tunnels to mesh nodes.

| App | URL | Fly config |
|---|---|---|
| **console** | [mesh-llm-console.fly.dev](https://mesh-llm-console.fly.dev) | `fly/console/fly.toml` |

Also available at [www.mesh-llm.com](https://www.mesh-llm.com) and [meshllm.cloud](https://meshllm.cloud).

## Architecture

```
                              ┌─────────────────────────┐
Browser/curl ──HTTPS──→ Fly   │  mesh-llm --client      │
                              │  discovers mesh via      │──QUIC──→ GPU nodes
                              │  Nostr, tunnels requests │
                              └─────────────────────────┘
```

Exposes `:3131` (dashboard, chat, topology) and proxies inference to mesh GPU nodes.

## Deploy

From the **repo root**:

```bash
fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile
```

## Run locally

```bash
# Same as what the Fly app runs — no Docker needed
mesh-llm --client --auto
```

## Docker (local)

```bash
docker build -f fly/Dockerfile -t mesh-llm-console .
docker run -p 3131:3131 -p 9337:9337 mesh-llm-console
```
