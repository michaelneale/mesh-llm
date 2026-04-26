# External Backends

mesh-llm can act as a mesh gateway in front of any OpenAI-compatible inference
server. The external server does all the heavy lifting — model loading, GPU
management, batching, quantization. mesh-llm handles mesh membership, peer
discovery, request routing, and the local API surface.

This is useful when:

- You already run vLLM, TGI, Ollama, or another inference server and want to
  expose it to a mesh without replacing your stack.
- You have a cloud-hosted inference endpoint and want mesh nodes to route to it.
- You want to mix llama.cpp nodes and vLLM nodes in the same mesh.

## Quick start

Start your inference server (this is your responsibility):

```bash
# vLLM example
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct --port 8000

# Ollama example
ollama serve  # default port 11434

# TGI example
text-generation-launcher --model-id meta-llama/Llama-3.1-8B-Instruct --port 8000
```

Then point mesh-llm at it:

```bash
mesh-llm serve --external-backend http://localhost:8000
```

That's it. mesh-llm:

1. Probes `GET /v1/models` on the backend to discover the model name.
2. Joins the mesh and gossips the model to all peers.
3. Becomes a Host node immediately — no election, no GPU detection.
4. Proxies inference requests from the mesh to the backend.
5. Runs a health check every 15 seconds. If the backend goes down, the model
   is withdrawn from the mesh. When it comes back, it's re-advertised.

## What mesh-llm does NOT do

- Does not start, stop, restart, or manage the external server.
- Does not download models for the external server.
- Does not configure batching, quantization, or tensor parallelism.
- Does not translate between API formats — the backend must speak the OpenAI
  chat completions API (`/v1/chat/completions`, `/v1/models`).

The external server is a black box. mesh-llm sends HTTP requests and relays
responses. If the backend supports streaming, streaming works. If it supports
tool calls, tool calls work. mesh-llm doesn't interpret or modify the payload.

## CLI flags

```
--external-backend <URL>   URL of the external OpenAI-compatible server.
                           Required. Example: http://gpu-box:8000

--external-model <NAME>    Override the model name advertised to the mesh.
                           Optional. If omitted, mesh-llm uses the first
                           model returned by GET /v1/models on the backend.
```

## Examples

### Standalone node with vLLM

```bash
mesh-llm serve --external-backend http://localhost:8000
```

### Join a mesh

```bash
mesh-llm serve --external-backend http://localhost:8000 --join <token>
```

### Auto-discover and join

```bash
mesh-llm serve --external-backend http://localhost:8000 --auto
```

### Override model name

Some backends return model IDs like `models/meta-llama/Llama-3.1-8B-Instruct`
or generic names. Use `--external-model` to control what the mesh sees:

```bash
mesh-llm serve --external-backend http://localhost:8000 --external-model llama-70b
```

### Publish a named mesh with an external backend

```bash
mesh-llm serve --external-backend http://gpu-server:8000 \
  --mesh-name "team-inference" --publish
```

### Ollama

Ollama's OpenAI compatibility layer lives at `/v1`:

```bash
mesh-llm serve --external-backend http://localhost:11434
```

## Architecture

```
  mesh peers ──QUIC──▶ this node ──TCP──▶ external backend
                        │                  (vLLM/TGI/Ollama)
                        │
                        ├─ gossip: model name, host role
                        ├─ API proxy on :9337
                        ├─ management console on :3131
                        ├─ QUIC tunnel listener (remote peers)
                        └─ health check loop (15s)
```

When a remote mesh peer sends an inference request, it arrives via QUIC HTTP
tunnel, hits the local backend proxy (an ephemeral TCP listener), which
forwards it to the external server. The response streams back the same path.

This uses the same `InferenceTarget::Local(port)` mechanism as llama.cpp
backends — the rest of the mesh can't tell the difference.

## What gets skipped

Compared to a normal `mesh-llm serve --model ...`, the external backend path
skips:

- Model download / GGUF parsing
- rpc-server launch
- llama-server launch
- Election (host election / tensor split calculation)
- Draft model detection
- GPU memory bandwidth benchmark (the external server handles its own GPU)

The node still participates in:

- Mesh gossip (peer discovery, model advertisement)
- Request routing (including prefix-affinity routing)
- QUIC tunneling (remote peers can route to this node)
- Health monitoring
- Management API and web console

## Health checks

mesh-llm hits `GET /v1/models` on the external backend every 15 seconds.

- If the backend becomes unreachable, the model is **withdrawn** from the mesh
  (peers stop routing to this node).
- When the backend recovers, the model is **re-advertised** and peers resume
  routing.

This means you can restart your vLLM server and mesh-llm will automatically
recover without manual intervention.

## Testing locally

A mock server is included for development:

```bash
python3 tools/mock-vllm.py 8000
mesh-llm serve --external-backend http://localhost:8000

# In another terminal:
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hello"}]}'
```
