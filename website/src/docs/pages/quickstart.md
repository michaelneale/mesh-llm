# Quickstart

Install Mesh, join a mesh, and point clients at the local OpenAI-compatible endpoint.

## Install

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | sh
```

## Join the public mesh

Serve from this machine if it has useful capacity:

```sh
mesh-llm serve --auto
```

Join as an API-only client:

```sh
mesh-llm client --auto
```

## Serve a model

Use a catalog model that fits on common 8GB VRAM machines:

```sh
mesh-llm serve --model gemma-4-26B-A4B-it-UD-Q4_K_M
```

## Use the API

Mesh exposes a local OpenAI-compatible endpoint:

```sh
export OPENAI_BASE_URL=http://localhost:3131/v1
```

Use the same endpoint from agent tools, SDKs, or curl.

