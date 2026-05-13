# Flash-MoE Plugin

The built-in `flash-moe` plugin connects mesh-llm to a Flash-MoE OpenAI-compatible HTTP server.

Use it for the SSD expert streaming roadmap path: a giant MoE model fits on one node's local NVMe, but not in RAM. Mesh-llm owns process lifecycle, plugin health, endpoint discovery, and request routing. Flash-MoE owns model execution.

This is intentionally a single-node backend adapter. It does not change the mesh protocol, Skippy stage protocol, model-package format, or llama.cpp patch queue.

## Prerequisites

Flash-MoE is an external backend. Mesh-llm does not vendor, install, or build the Flash-MoE binary, model conversion tooling, or SSD-streaming artifacts.

Install or build Flash-MoE separately from [danveloper/flash-moe](https://github.com/danveloper/flash-moe), prepare the model files with its tooling, then either:

- set `command` to the local Flash-MoE `infer` binary for managed process mode; or
- set `url` to an already-running Flash-MoE `/v1` endpoint.

If upstream release artifacts are not available for your platform, use a source build or deployment-managed binary and point mesh-llm at that binary. This adapter intentionally keeps Flash-MoE ownership outside mesh-llm until the backend and packaging surface are mature enough to standardize.

## Managed Process Mode

Point `command` at the Flash-MoE `infer` binary and pass the normal model arguments in `args`.

```toml
[[plugin]]
name = "flash-moe"
command = "/opt/flash-moe/metal_infer/infer"
args = [
  "--model", "/models/qwen3.5-397b/model.gguf",
  "--weights", "/models/qwen3.5-397b/experts.bin",
  "--manifest", "/models/qwen3.5-397b/manifest.json",
  "--vocab", "/models/qwen3.5-397b/vocab.json"
]
```

Mesh-llm allocates a local port and appends:

```text
--serve <port>
```

Do not pass `--serve` in config. Keeping the port host-owned prevents collisions between plugins, external backends, and local llama.cpp serving.

When the plugin starts, it registers an OpenAI-compatible inference endpoint like:

```text
http://127.0.0.1:<port>/v1
```

The host probes `GET /v1/models` through the normal plugin endpoint health path, so Flash-MoE models appear and disappear the same way other plugin-backed models do.

## Existing Endpoint Mode

If Flash-MoE is already running, attach the endpoint instead of letting mesh-llm spawn it:

```toml
[[plugin]]
name = "flash-moe"
url = "http://127.0.0.1:8000/v1"
```

This mode leaves process lifecycle outside mesh-llm and only registers the endpoint.

## Config Rules

- Configure either `command` or `url`, not both.
- `args` require `command`.
- `--serve` is host-owned and must not appear in `args`.
- Model paths and weights stay local to the node running Flash-MoE.
- No HuggingFace token or private credential is required by the plugin itself.

## Packaging Boundary

The first adapter stays in-tree because it depends on mesh-llm's host-runtime plugin lifecycle, local endpoint registration, environment handoff, process ownership, and health model. A separate adapter crate or repository is a better follow-up once the plugin SDK and crates.io surface are stable enough for out-of-tree inference providers.

## Scope

Included:

- bundled `flash-moe` plugin entrypoint
- managed Flash-MoE process launch
- existing HTTP endpoint attachment
- OpenAI-compatible endpoint registration
- plugin health and lifecycle checks
- model discovery via the existing `/v1/models` probe path

Not included:

- mesh-distributed SSD expert streaming
- Skippy package slicing changes
- Flash-MoE binary vendoring
- Flash-MoE installation or source-build automation
- Flash-MoE model conversion or download automation
- out-of-tree adapter crate/repository packaging
- changes to public mesh protocol fields
