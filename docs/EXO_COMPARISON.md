# Mesh LLM and Exo

Use this comparison to understand how Mesh LLM and
[Exo](https://github.com/exo-explore/exo) differ across architecture, runtime,
networking, API compatibility, and packaging models.

## Summary

- **Exo** connects local devices into an AI cluster with automatic discovery,
  MLX/MLX distributed backends, a local dashboard, and multiple API
  compatibility layers.
- **Mesh LLM** connects machines into private or published inference meshes with
  OpenAI-compatible routing, embedded Skippy/llama.cpp staged execution, and
  package-backed GGUF layer splits.

Exo is a strong fit for local multi-device clusters, especially Apple Silicon
and MLX-oriented environments. Mesh LLM is a strong fit for operator-controlled
distributed serving, private/public mesh workflows, and very large GGUF models
that use package-backed layer splits.

## Feature comparison

| Topic | Exo | Mesh LLM |
|---|---|---|
| Core positioning | Runs frontier AI locally by connecting devices into an AI cluster. | Pools GPUs across a mesh and exposes one OpenAI-compatible API. |
| Cluster shape | Local multi-device cluster with automatic discovery. | Private invite meshes, published discoverable meshes, and API-only clients. |
| Inference backend | MLX and MLX distributed. | Embedded Skippy/llama.cpp stage runtime with package-backed GGUF splits. |
| Model splitting | Topology-aware auto parallelism, tensor parallelism, placement preview, and placement compute APIs. | Topology planning assigns contiguous layer ranges and loads package-backed stage artifacts. |
| Hardware emphasis | Apple Silicon/macOS support with MLX; Linux CPU support; Linux GPU support is under development. | Release flavors for macOS, Linux, Windows, CUDA, ROCm, Vulkan, Metal, and CPU. |
| Networking | libp2p, mDNS, bootstrap peers, and namespace isolation with `EXO_LIBP2P_NAMESPACE`. | QUIC/iroh mesh paths, Nostr discovery for published meshes, and managed relays by default. |
| API compatibility | OpenAI Chat Completions, Claude Messages, OpenAI Responses, and Ollama API compatibility. | OpenAI-compatible `/v1/models`, chat completions, completions, and Responses through `openai-frontend`. |
| Dashboard and ports | Local API and dashboard at `http://localhost:52415`; macOS app available. | Web console and management API at `http://localhost:3131`; inference API at `http://localhost:9337/v1`. |
| Artifact model | Model placement and runner-oriented runtime model. | Layer package repositories with manifest, artifact layout, validation, certification, and HF Jobs publishing workflow. |
| Operational model | Local cluster coordination across nearby devices. | Operator-controlled distributed serving across private or published meshes. |

## From Exo's documentation

Sources checked during this docs update:

- Repository: <https://github.com/exo-explore/exo>
- README: <https://github.com/exo-explore/exo/blob/main/README.md>
- Architecture: <https://github.com/exo-explore/exo/blob/main/docs/architecture.md>
- API: <https://github.com/exo-explore/exo/blob/main/docs/api.md>
- Latest checked release: `v1.0.71`, published 2026-04-23.

Key documented Exo architecture details:

- event sourcing and Erlang-style message passing,
- Master, Worker, Runner, API, and Election systems,
- automatic device discovery,
- libp2p, mDNS, bootstrap-peer networking, and namespace isolation,
- topology-aware auto parallelism and tensor parallelism,
- placement preview and placement compute APIs,
- MLX and MLX distributed,
- RDMA over Thunderbolt 5 on supported macOS systems,
- local API and dashboard at `http://localhost:52415`.

## Mesh LLM docs for comparison

- [MESHES.md](MESHES.md) for private/public mesh behavior.
- [SKIPPY_SPLITS.md](SKIPPY_SPLITS.md) for package-backed stage splits.
- [LAYER_PACKAGE_REPOS.md](LAYER_PACKAGE_REPOS.md) for package publishing.
- [AGENTS.md](AGENTS.md) for agent/client integrations.
