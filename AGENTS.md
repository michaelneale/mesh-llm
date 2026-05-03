# Agent Notes

## Repo Overview

This repo (`mesh-llm`) contains mesh-llm — a Rust binary that pools GPUs over QUIC for distributed LLM inference using llama.cpp.

## Key Docs

| Doc | What it covers |
|---|---|
| `README.md` | Usage, install, CLI flags, examples |
| `CONTRIBUTING.md` | Build from source, dev workflow, UI dev |
| `RELEASE.md` | Release process (build, bundle, tag, GitHub release) |
| `ROADMAP.md` | Future directions |
| `crates/mesh-llm/TODO.md` | Current work items and backlog |
| `crates/mesh-llm/README.md` | Rust crate overview and file map |
| `docs/README.md` | Documentation map and topic directory guide |
| `docs/design/DESIGN.md` | Architecture, protocols, features |
| `docs/design/TESTING.md` | Test playbook, scenarios, remote deploy |
| `docs/design/MULTI_MODAL.md` | Multimodal design: capability model, blob plugin, console, routing |
| `docs/design/MoE_PLAN.md` | MoE expert sharding design |
| `docs/design/MoE_DEPLOY_DESIGN.md` | MoE auto-deploy UX |
| `docs/design/VIRTUAL_LLM.md` | Virtual LLM engine (inter-model collaboration) |
| `docs/design/LLAMA_CPP_FORK.md` | llama.cpp fork: what's patched, how to update, how to sync |
| `docs/moe/README.md` | MoE analyzer, placement, and CLI planning notes |
| `docs/plugins/README.md` | Plugin architecture and plugin development |
| `fly/README.md` | Fly.io deployment (console + API apps) |
| `tools/relay-fly-legacy/README.md` | Legacy self-hosted iroh relay (not in use — now using services.iroh.computer) |

## Building

Always use `just`. Never build manually.

```bash
just build    # llama.cpp fork + mesh-llm + UI
just bundle   # portable tarball
just stop     # kill mesh/rpc/llama processes
just test     # quick inference test against :9337
just auto     # build + stop + start with --auto
just ui-dev   # vite dev server with HMR
just clean-ui # nuke node_modules + dist (fixes stale npm state)
```

### npm "Exit handler never called" error

If `just build` fails on the UI step with `npm error Exit handler never called!`, run:

```bash
just clean-ui
just build
```

This is an npm bug that surfaces when `node_modules` gets into a bad state (e.g. after branch switches that change `package-lock.json`). Nuking `node_modules` and letting `npm ci` reinstall from scratch fixes it.

See `CONTRIBUTING.md` for full dev workflow.

## llama.cpp ABI Patch Queue

mesh-llm embeds the stage runtime and links patched llama.cpp static ABI
libraries. The only durable llama.cpp patch queue is
`third_party/llama.cpp/patches`, pinned by `third_party/llama.cpp/upstream.txt`.

- `just build` prepares `.deps/llama.cpp`, applies the ABI patch queue, builds
  the static libraries, builds the UI, and builds `mesh-llm`.
- Do not reintroduce an external `llama-server` / `rpc-server` runtime lane.
- If you need to update upstream llama.cpp, use `scripts/prepare-llama.sh`,
  `scripts/build-llama.sh`, `scripts/update-llama-pin.sh`, and
  `scripts/summarize-llama-upstream.sh`.

## Project Structure

- `crates/mesh-llm/src/` — Rust source
- `crates/mesh-llm/ui/` — React web console (shadcn/ui patterns, see https://ui.shadcn.com/llms.txt)
- `docs/` — Project docs, grouped by topic
- `docs/design/` — Architecture, protocol, and testing docs
- `docs/moe/` — MoE ranking, placement, and CLI plans
- `docs/plugins/` — Plugin architecture docs and plans
- `fly/` — Fly.io deployment (console + API client apps)
- `tools/relay-fly-legacy/` — Legacy self-hosted iroh relay (not in use — now using services.iroh.computer)
- `evals/` — Benchmarking and evaluation scripts

## Module Structure Rules

The crate root should stay minimal.

- Keep `crates/mesh-llm/src/lib.rs` and `crates/mesh-llm/src/main.rs` as the only root `.rs` files unless there is a strong reason otherwise.
- New code should go into an existing domain directory when possible.

Use semantic ownership for module placement.

- `crates/mesh-llm/src/cli/` — Clap types, command parsing, command dispatch, and user-facing command handlers.
- `crates/mesh-llm/src/runtime/` — top-level process orchestration and startup/runtime coordination.
- `crates/mesh-llm/src/network/` — request routing, proxying, tunneling, relay/discovery networking, request-affinity logic, and endpoint rewrite support.
- `crates/mesh-llm/src/inference/` — model-serving logic, election, launch, pipeline, and MoE behavior.
- `crates/mesh-llm/src/system/` — machine-local environment and platform concerns such as hardware detection, benchmarking, self-update, and local system integration.
- `crates/mesh-llm/src/models/` — model catalog, resolution, downloads, local model storage, and model metadata.
- `crates/mesh-llm/src/mesh/` — peer membership, gossip, identity, peer state, and mesh node behavior.
- `crates/mesh-llm/src/plugin/` — plugin host, plugin runtime, transport, config, and MCP bridge support.
- `crates/mesh-llm/src/api/` — management API surface and route handling.
- `crates/mesh-llm/src/protocol/` — wire protocol types, encoding/decoding, and conversions.

CLI ownership rule.

- All command handlers belong under `crates/mesh-llm/src/cli/`, usually `crates/mesh-llm/src/cli/commands/`.
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
- Plugin MCP support belongs inside `crates/mesh-llm/src/plugin/`, not as a separate root module.
- Model command handlers belong in `crates/mesh-llm/src/cli/commands/`; `crates/mesh-llm/src/models/` should stay domain-focused.

## Key Source Files

- `crates/mesh-llm/src/main.rs` — Binary entrypoint; calls `mesh_llm::run()`
- `crates/mesh-llm/src/runtime/mod.rs` — Top-level startup flows, runtime orchestration, and command dispatch
- `crates/mesh-llm/src/mesh/mod.rs` — `Node` struct, gossip, mesh_id, peer management
- `crates/mesh-llm/src/inference/election.rs` — Host election, tensor split calculation
- `crates/mesh-llm/src/inference/launch.rs` — llama-server/rpc-server process management
- `crates/mesh-llm/src/inference/moe.rs` — MoE detection, expert rankings, split orchestration
- `crates/mesh-llm/src/network/proxy.rs` — HTTP proxy: request parsing, model routing, response helpers
- `crates/mesh-llm/src/network/router.rs` — Request classification, model scoring, multimodal routing
- `crates/mesh-llm/src/network/nostr.rs` — Nostr discovery, `score_mesh()`, `smart_auto()`
- `crates/mesh-llm/src/network/tunnel.rs` — TCP ↔ QUIC relay (RPC + HTTP)
- `crates/mesh-llm/src/api/mod.rs` — Management API (:3131): `/api/status`, `/api/events`, `/api/discover`, `/api/join`
- `crates/mesh-llm/src/models/catalog.rs` — Model catalog, HuggingFace downloads
- `crates/mesh-llm/src/models/capabilities.rs` — Multimodal/vision/audio/reasoning capability inference
- `crates/mesh-llm/src/plugins/blobstore/mod.rs` — Request-scoped media object storage for multimodal
- `crates/mesh-llm/src/runtime/instance.rs` — Per-instance runtime directory management: `InstanceRuntime`, pidfiles, flock liveness, scoped orphan reaping, local instance scanning

## Mesh Protocol Compatibility

Mesh compatibility across versions is critical. Nodes in the wild run different versions and must interoperate.

- The mesh supports mixed-version operation: QUIC ALPN `mesh-llm/1` (protobuf) and `mesh-llm/0` (legacy JSON) nodes coexist. Do not break this.
- Gossip fields, stream types, and protobuf schemas must be additive. New fields should be optional and ignored by older nodes. Do not repurpose or remove existing fields.
- When adding new gossip fields, stream types, or changing wire format, explicitly consider what happens when an older node receives the new data and when a newer node talks to an older peer.
- Capability advertisement (vision, audio, multimodal, reasoning, tool_use, moe) is gossiped to all peers and consumed by routing, the API, and the UI. Changes to capability semantics affect the whole mesh, not just the local node.
- If a change would break mixed-version meshes, explicitly flag it as a breaking protocol change and ask the developer before proceeding.
- Test compatibility by running the current branch against a released binary on a second node. Verify gossip, routing, and inference work across the version boundary.

## Plugin Protocol Compatibility

When iterating on the plugin protocol, always consider protocol compatibility.

- If a protocol change may be breaking, explicitly ask the developer whether the change is intended to be breaking.
- If the change is not intended to be breaking, the previous version of the plugin protocol must continue to be supported.
- Do not silently ship plugin protocol changes that strand older plugins or hosts without confirming that outcome is acceptable.

## UI Notes

For changes in `crates/mesh-llm/ui/`, use components and compose interfaces consistently with shadcn/ui patterns. Prefer extending existing primitives in `ui/src/components/ui/` over ad-hoc markup.

## Testing

Read `docs/design/TESTING.md` before running tests. It has all test scenarios, remote deploy instructions, and cleanup commands.

Testing matters more than usual in this project because:

- Nodes run on different machines with different hardware and OS versions. Bugs that don't reproduce locally can appear in real deployments.
- The mesh protocol is a distributed system — gossip, election, and routing interact across nodes. Single-node unit tests don't catch protocol-level regressions.
- The public mesh at anarchai.org runs continuously. Breaking changes that pass local tests can take down live inference for real users.
- Multimodal, MoE splitting, and multi-model routing all have complex interaction paths that are hard to reason about statically.

When making changes that touch gossip, routing, proxy, election, or capability advertisement, test against at least two nodes before merging. The deploy checklist above is not optional.

### Cargo Concurrency

Run `cargo` commands serially. Do not run multiple `cargo` commands in parallel (including parallel test runs), because this repo frequently hits Cargo lock conflicts (`package cache` / `artifact directory`) under concurrent invocation.

## Pre-Commit Checklist

Before committing, run the local checks most likely to fail in CI for the files you touched. Do not rely on CI to catch basic formatting, compile, or stale UI build issues.

### Minimum bar before every commit

- Rust-only change — format the changed Rust files and run `cargo check -p mesh-llm`.
- UI-only change — run `just build`.
- Mixed Rust and UI change — run `just build`.

### Rust changes

- Format only the changed Rust files from the repo root, for example with `cargo fmt --all -- path/to/file.rs`, and include those formatting changes in the commit.
- Before committing Rust changes, ensure the formatting check passes with `cargo fmt --all -- --check`.
- After Rust changes, run `cargo check -p mesh-llm`.
- If you touched tests, public APIs, routing, inference, gossip, plugin protocol, or CLI behavior, run the relevant tests before committing.
- If you touched `proto/`, `crates/mesh-llm/src/protocol/`, `crates/mesh-llm/src/mesh/gossip.rs`, `crates/mesh-llm/src/mesh/mod.rs`, routing, election, or API serialization, do not stop at build-only validation: run at least `cargo test -p mesh-llm --lib` and wait for it to exit successfully before committing.
- Do not report a build or test step as complete until the command has actually exited with code `0`.
- Run Rust validation serially. Do not run multiple `cargo` commands at the same time.

### UI changes

- Use the repo's supported workflow and run `just build`.
- If `just build` fails on the UI step with `npm error Exit handler never called!`, run `just clean-ui` and then rerun `just build`.

### Commit standard

- Do not commit if formatting has not been applied.
- Do not commit if basic local validation for your change type has not been run.
- Do not commit known warnings in code you touched.

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

### Deploy to Remote

```bash
just bundle
# scp bundle to remote, tar xzf, codesign -s - the three binaries
```

### Cleanup

Clean shutdown removes the instance's runtime directory automatically. Prefer the scoped runtime-aware commands first:

```bash
mesh-llm stop
just stop
```

Those paths use the runtime metadata under `~/.mesh-llm/runtime/` to stop the tracked mesh-llm instance and its child servers cleanly.

If an instance is wedged badly enough that the scoped stop path cannot reach it, fall back to an emergency kill:

```bash
pkill -f mesh-llm
```

## Deploy Checklist — MANDATORY

**Every deploy to test machines MUST follow this checklist.**

### Before starting nodes
1. **Bump VERSION** in `main.rs` so you can verify the running binary is new code.
2. `just build && just bundle`
3. Kill ALL processes on ALL nodes — `pkill -9 -f mesh-llm`
4. Verify clean — `ps -eo pid,args | grep -E 'mesh-llm' | grep -v grep` must be empty.
5. Deploy bundle — scp + tar + codesign on remote nodes.
6. Verify version — `mesh-llm --version` on every node.

### After starting nodes
7. Verify exactly 1 mesh-llm process per node.
8. Verify no external llama serving child processes are required.
9. `curl -s http://localhost:3131/api/status` returns valid JSON on every node.
10. Check `/api/status` peers for new version string.
11. Verify expected peer count.
12. Test inference through every model in `/v1/models`.
13. Test `/v1/` passthrough on port 3131.

### Debugging Embedded Runtime Startup

If the embedded runtime fails to load, check mesh-llm stderr/log output and
`~/.mesh-llm/runtime/` for the active instance metadata. The old external
`llama-server` and `rpc-server` log files are no longer produced.

To override the runtime root (e.g., for tests or systemd):
- `MESH_LLM_RUNTIME_ROOT=/path/to/custom/root` — highest priority
- `XDG_RUNTIME_DIR` — if set (typical on systemd: `/run/user/{uid}/mesh-llm/runtime`)
- `$HOME/.mesh-llm/runtime` — default fallback

For stale instances (crashed mesh-llm leaving behind a runtime dir):
- Other running mesh-llm instances GC dead-owner dirs older than 1 hour on startup
- Manual cleanup: `rm -rf ~/.mesh-llm/runtime/<stale_pid>/`

### Common failures
- **nohup over SSH doesn't stick** — use `bash -c "nohup ... & disown"`, verify process survives disconnect.
- **Duplicate processes** — always kill-verify-start.
- **codesign changes the hash** — don't compare local vs codesigned remote.

## Releasing

See `RELEASE.md` for the full process.

Current release flow:

1. Build and verify locally:
   ```bash
   just build
   just bundle
   ```
2. Release from a clean local `main` branch:
   ```bash
   just release v0.X.Y
   ```
   This bumps the version, refreshes `Cargo.lock` without upgrading dependencies, commits as `v0.X.Y: release`, pushes `main`, and then pushes only the new release tag.
3. Pushing a `v*` tag triggers `.github/workflows/release.yml`, which builds the release artifacts on Linux CPU, Linux CUDA, and macOS and creates the GitHub release automatically.

## Credentials

Test machine IPs, SSH details, and passwords are in `~/Documents/private-note.txt` (outside the repo). **Never commit credentials to any tracked file.**

## What NOT to add

- **No `api_key_token` feature** — explicitly rejected, removed in v0.26.0
- **No credentials in tracked files** — IPs, passwords, SSH commands belong in `~/Documents/private-note.txt` only
