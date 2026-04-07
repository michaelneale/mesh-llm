# Agent Notes

## Repo Overview

This repo (`mesh-llm`) contains mesh-llm — a Rust binary that pools GPUs over QUIC for distributed LLM inference using llama.cpp.

## Principles

These guide every design decision in this project:

- **One command to run.** `mesh-llm serve --auto` should be all you need.
- **Batteries included.** Models download automatically. Backends are bundled. The web console ships inside the binary.
- **Sensible defaults.** Solo when the model fits. Split only when it has to. Draft models auto-paired. Context sized to VRAM.
- **Always compatible in the mesh.** Older and newer nodes must coexist. Protocol negotiation keeps mixed meshes working. Rolling upgrades, not flag days.
- **Public and private.** `--auto` for public meshes, `--join <token>` for private. Same binary, same API.
- **Support as many platforms as possible.** macOS Metal, Linux CUDA/ROCm/Vulkan/CPU, Jetson/Tegra, Windows.

## Key Docs

| Doc | What it covers |
|---|---|
| `README.md` | User-facing: install, usage, CLI, examples, agents |
| `HUMANS.md` | Developer-facing: architecture, testing, release, roadmap, plugins |
| `mesh-llm/docs/PLUGINS.md` | Full plugin architecture spec |
| `mesh-llm/docs/PLUGINS_PLAN.md` | Plugin implementation sequencing |
| `mesh-llm/docs/message_protocol.md` | Wire protocol spec |
| `mesh-llm/docs/MULTI_MODAL.md` | Multimodal capability details |
| `mesh-llm/docs/MoE_PLAN.md` | MoE expert sharding design |
| `mesh-llm/docs/MoE_DEPLOY_DESIGN.md` | MoE auto-deploy implementation |
| `fly/README.md` | Fly.io deployment |
| `relay/README.md` | Self-hosted iroh relay |

## Building

Always use `just`. Never build manually.

```bash
just build    # llama.cpp fork + mesh-llm + UI
just bundle   # portable tarball
just stop     # kill mesh/rpc/llama processes
just test     # quick inference test against :9337
just auto     # build + stop + start with --auto
just ui-dev   # vite dev server with HMR
```

See `HUMANS.md` for full dev workflow.

## Project Structure

- `mesh-llm/src/` — Rust source
- `mesh-llm/ui/` — React web console (shadcn/ui patterns, see https://ui.shadcn.com/llms.txt)
- `mesh-llm/docs/` — Design and protocol docs
- `fly/` — Fly.io deployment (console + API client apps)
- `relay/` — Self-hosted iroh relay
- `evals/` — Benchmarking and evaluation scripts

## Module Structure Rules

The crate root should stay minimal.

- Keep `mesh-llm/src/lib.rs` and `mesh-llm/src/main.rs` as the only root `.rs` files unless there is a strong reason otherwise.
- New code should go into an existing domain directory when possible.

Use semantic ownership for module placement.

- `mesh-llm/src/cli/` — Clap types, command parsing, command dispatch, and user-facing command handlers.
- `mesh-llm/src/runtime/` — top-level process orchestration and startup/runtime coordination.
- `mesh-llm/src/network/` — request routing, proxying, tunneling, relay/discovery networking, request-affinity logic, and endpoint rewrite support.
- `mesh-llm/src/inference/` — model-serving logic, election, launch, pipeline, and MoE behavior.
- `mesh-llm/src/system/` — machine-local environment and platform concerns such as hardware detection, benchmarking, self-update, and local system integration.
- `mesh-llm/src/models/` — model catalog, resolution, downloads, local model storage, and model metadata.
- `mesh-llm/src/mesh/` — peer membership, gossip, identity, peer state, and mesh node behavior.
- `mesh-llm/src/plugin/` — plugin host, plugin runtime, transport, config, and MCP bridge support.
- `mesh-llm/src/api/` — management API surface and route handling.
- `mesh-llm/src/protocol/` — wire protocol types, encoding/decoding, and conversions.

CLI ownership rule.

- All command handlers belong under `mesh-llm/src/cli/`, usually `mesh-llm/src/cli/commands/`.
- Domain modules should not own Clap parsing or top-level command dispatch.
- Domain modules may expose reusable functions that CLI handlers call.

Do not introduce generic buckets.

- Avoid directories or modules named `app`, `utils`, `misc`, `common`, or similar catch-alls.
- Name modules after the responsibility they own.

Keep shared code honest.

- If code is only used by one subsystem, keep it inside that subsystem.
- Only move code to a shared module when it is truly cross-domain.
- Do not create shared helpers prematurely.

Prefer semantic grouping over symmetry.

- Do not create one directory per file just for visual symmetry.
- A single `foo.rs` file is already a Rust module; use a directory only when `foo` has meaningful substructure.

Minimize crate-root re-exports.

- Root re-exports are acceptable as temporary compatibility shims during refactors.
- New code should prefer importing from the owning module directly.
- Remove transitional re-exports once call sites have been updated.

When to split a file.

- Split a file when it contains multiple separable responsibilities, when navigation becomes difficult, or when tests naturally cluster by concern.
- Do not split purely to reduce line count if the code still represents one coherent object or subsystem.

Naming rule.

- File and module names should describe responsibility, not implementation detail.
- Prefer names like `affinity`, `discovery`, `transport`, `maintenance`, `warnings`.
- Avoid vague names like `helpers`, `stuff`, `logic`, or `manager` unless the abstraction is genuinely that broad.

Current structure notes.

- Request-affinity code belongs with networking/routing behavior, not `system/`.
- Plugin MCP support belongs inside `mesh-llm/src/plugin/`, not as a separate root module.
- Model command handlers belong in `mesh-llm/src/cli/commands/`; `mesh-llm/src/models/` should stay domain-focused.

## Plugin Protocol Compatibility

When iterating on the plugin protocol, always consider protocol compatibility.

- If a protocol change may be breaking, explicitly ask the developer whether the change is intended to be breaking.
- If the change is not intended to be breaking, the previous version of the plugin protocol must continue to be supported.
- Do not silently ship plugin protocol changes that strand older plugins or hosts without confirming that outcome is acceptable.

## UI Notes

For changes in `mesh-llm/ui/`, use components and compose interfaces consistently with shadcn/ui patterns. Prefer extending existing primitives in `ui/src/components/ui/` over ad-hoc markup.

## Testing

Read `HUMANS.md` (Testing section) before running tests. It has all test scenarios, remote deploy instructions, and cleanup commands.

## Formatting

Before committing Rust changes, format only the changed Rust files from the repo root, for example with `cargo fmt --all -- path/to/file.rs`, and include those formatting changes in the commit.

## Warnings

Do not leave Rust compiler warnings behind in code you touched.

- Fix or remove unused code, dead code, and other warnings introduced or surfaced by your change before committing.
- Do not silence warnings with `#[allow(...)]` unless there is a clear reason and the developer has asked for that tradeoff.

## Pull Requests

Pull request titles and descriptions should be user-focused by default.

- Title PRs around the user-visible change or capability, not the implementation detail.
- Start the description with what the user can now do, see, or understand after the change.
- Keep architectural refactors, internal state reshaping, and code-organization notes out of the opening summary unless they directly change user behavior.
- If there are important architectural changes, add a separate `## Architecture` section.
- If there are protocol or compatibility implications, add a separate `## Protocol` section that clearly calls out compatibility, migration, or breaking-change impact.
- If the PR changes CLI behavior or touches user-facing CLI flows, include example commands and representative output in the PR description.
- If the PR changes the UI, include at least one screenshot in the PR description.
- Validation and screenshots should stay separate from the user-facing summary.

## Deploy Checklist — MANDATORY

**Every deploy to test machines MUST follow this checklist.**

### Before starting nodes
1. **Bump VERSION** in `main.rs` so you can verify the running binary is new code.
2. `just build && just bundle`
3. Kill ALL processes on ALL nodes — `pkill -9 -f mesh-llm; pkill -9 -f llama-server; pkill -9 -f rpc-server`
4. Verify clean — `ps -eo pid,args | grep -E 'mesh-llm|llama-server|rpc-server' | grep -v grep` must be empty.
5. Deploy bundle — scp + tar + codesign on remote nodes.
6. Verify version — `mesh-llm --version` on every node.

### After starting nodes
7. Verify exactly 1 mesh-llm process per node.
8. Verify child processes (at most 1 rpc-server + 1 llama-server per mesh-llm).
9. `curl -s http://localhost:3131/api/status` returns valid JSON on every node.
10. Check `/api/status` peers for new version string.
11. Verify expected peer count.
12. Test inference through every model in `/v1/models`.
13. Test `/v1/` passthrough on port 3131.

### Debugging llama-server startup

If llama-server fails to start (stuck at "⏳ Starting llama-server..."), check its log file:

```bash
cat "$(python3 -c 'import tempfile; print(tempfile.gettempdir())')/mesh-llm-llama-server.log"
```

Typical path: `/var/folders/XX/.../T/mesh-llm-llama-server.log`. rpc-server logs are in the same directory as `mesh-llm-rpc-{port}.log`.

### Common failures
- **nohup over SSH doesn't stick** — use `bash -c "nohup ... & disown"`, verify process survives disconnect.
- **Duplicate processes** — always kill-verify-start.
- **codesign changes the hash** — don't compare local vs codesigned remote.

## Releasing

See `HUMANS.md` (Release process section) for the full process.

Quick version:

```bash
just build && just bundle     # build and verify
just release v0.X.Y           # bump, commit, tag, push
```

Pushing a `v*` tag triggers CI which builds all platform artifacts and creates the GitHub release.

## Credentials

Test machine IPs, SSH details, and passwords are in `~/Documents/private-note.txt` (outside the repo). **Never commit credentials to any tracked file.**

## What NOT to add

- **No `api_key_token` feature** — explicitly rejected, removed in v0.26.0
- **No credentials in tracked files** — IPs, passwords, SSH commands belong in `~/Documents/private-note.txt` only
