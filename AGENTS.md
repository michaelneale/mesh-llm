# Agent Notes

## Repo Overview

This repo (`mesh-llm`) contains mesh-llm — a Rust binary that pools GPUs over QUIC for distributed LLM inference using llama.cpp.

The workspace is split across many crates under `crates/`. The shipped binary `mesh-llm` is a thin shim (`crates/mesh-llm/`) that re-exports `mesh-llm-host-runtime`, where the bulk of host-side logic lives. A lighter parallel crate `mesh-client` (`mesh-llm-client`) carries the same domain shape for client-only usage. Embedded llama.cpp staged-runtime support lives in the `skippy-*` crates.

## Key Docs

| Doc | What it covers |
|---|---|
| `README.md` | Quickstart and documentation hub |
| `docs/MESHES.md` | Public/private meshes, publishing, discovery, join flows |
| `docs/SKIPPY_SPLITS.md` | Running big models with Skippy split serving |
| `docs/LAYER_PACKAGE_REPOS.md` | Contributing and publishing layer package repos |
| `docs/EXO_COMPARISON.md` | mesh-llm vs Exo comparison |
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
| `tools/relay-fly-legacy/README.md` | Archived self-hosted iroh relay reference; production uses services.iroh.computer |

## Building

Always use `just`. Never build manually.

```bash
just build         # DEBUG build → ./target/debug/mesh-llm (fast, for iteration)
just release-build # RELEASE build → ./target/release/mesh-llm (slow, for serious testing / deploy)
just bundle        # portable tarball (uses the release binary)
just stop          # stop tracked mesh-llm runtime processes
just test          # quick inference test against :9337
just auto          # build + stop + start with --auto
just ui-dev        # vite dev server with HMR
just clean-ui      # nuke node_modules + dist (fixes stale npm state)
```

**Which build to use:**

- `just build` → produces `./target/debug/mesh-llm`. Use for fast local iteration
  and sanity-checking that the code compiles end-to-end (llama.cpp ABI + UI +
  mesh-llm). Do **not** use this binary for serious behavior testing, perf
  testing, or deploying to test machines — debug builds are slow and can hide
  or surface bugs that release builds don't.
- `just release-build` → produces `./target/release/mesh-llm`. Use this for any
  serious testing, deploying to test machines, bundling, or releases. This is
  what `just bundle` consumes and what CI builds.
- `./target/release/mesh-llm` may exist from a *previous* `just release-build`
  or `just build-dev` invocation even after you run only `just build` — its
  presence is **not** evidence that your latest code is in it. When in doubt,
  check `stat ./target/release/mesh-llm` against the time you last ran
  `just release-build`, or just re-run `just release-build`.
- `cargo check` / `cargo build` do **not** count as a build for this repo —
  they skip llama.cpp ABI prep and the UI, and `cargo check` produces no
  binary at all.

When in doubt for testing or shipping changes: use `just release-build` and
then copy `./target/release/mesh-llm`.

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

## Workspace Crates

The workspace lives under `crates/`. The most important crates:

- `mesh-llm/` — shipped binary; thin shim with `main.rs` building the Tokio runtime and `lib.rs` re-exporting `mesh-llm-host-runtime`. Almost no domain code here.
- `mesh-llm-host-runtime/` — the host-side monolith. Owns runtime orchestration, mesh, inference, networking, API, CLI, plugins, models, system integration. This is where most changes land.
- `mesh-client/` (`mesh-llm-client`) — lighter parallel client surface with its own `inference/`, `network/`, `models/`, `mesh/` modules. Used as a dev/test surface and for client-only deployments.
- `mesh-llm-ui/` — React web console and embedded asset crate (shadcn/ui patterns, see https://ui.shadcn.com/llms.txt).
- `mesh-llm-types/` — shared model/capability types used across crates.
- `mesh-llm-protocol/` — wire protocol types and protobuf bindings.
- `mesh-llm-routing/` — routing primitives shared across host and client.
- `mesh-llm-system/` — machine-local hardware, benchmark, autoupdate, process helpers.
- `mesh-llm-plugin/` — plugin runtime/DSL primitives.
- `mesh-llm-identity/` — identity primitives.
- `mesh-api/`, `mesh-api-ffi/` — management API surface and FFI bindings.
- `mesh-host-core/` — minimal shared host core.
- `openai-frontend/` — OpenAI-compatible HTTP frontend (chat, completions, responses, models).
- `model-artifact/`, `model-hf/`, `model-package/`, `model-ref/`, `model-resolver/` — model catalog, HuggingFace download, packaging, reference resolution.
- `skippy-ffi/` — Rust ABI bindings to the patched llama.cpp staged runtime.
- `skippy-runtime/` — Rust-side staged runtime, package materialization, model info.
- `skippy-server/` — embedded staged-runtime serving (frontend, binary transport, runtime state, embedded HTTP).
- `skippy-protocol/`, `skippy-topology/`, `skippy-coordinator/`, `skippy-cache/`, `skippy-prompt/`, `skippy-metrics/`, `skippy-bench/`, `skippy-correctness/`, `skippy-model-package/` — supporting skippy infrastructure.
- `metrics-server/` — standalone metrics collector binary.
- `mesh-llm-gpu-bench/`, `llama-spec-bench/`, `mesh-llm-test-harness/` — benchmarking and test harness binaries.

Other top-level directories:

- `docs/` — Project docs, grouped by topic.
- `docs/design/` — Architecture, protocol, and testing docs.
- `docs/moe/` — MoE ranking, placement, and CLI plans.
- `docs/plugins/` — Plugin architecture docs and plans.
- `fly/` — Fly.io deployment (console + API client apps).
- `tools/relay-fly-legacy/` — Archived self-hosted iroh relay reference; production uses services.iroh.computer.
- `evals/` — Benchmarking and evaluation scripts.
- `third_party/llama.cpp/patches/` — durable llama.cpp patch queue, pinned by `upstream.txt`.

## Module Structure Rules

These rules apply primarily inside `crates/mesh-llm-host-runtime/src/` (the main host monolith), and by analogy inside `crates/mesh-client/src/`. New peer crates should still follow the semantic-ownership principles below.

The host-runtime crate root should stay minimal.

- Keep `crates/mesh-llm-host-runtime/src/lib.rs` slim — it is a small entry point, not a junk drawer.
- New code should go into an existing domain directory when possible.

Use semantic ownership for module placement. Inside `crates/mesh-llm-host-runtime/src/`:

- `cli/` — Clap types, command parsing, command dispatch, and user-facing command handlers.
- `runtime/` — top-level process orchestration, startup/runtime coordination, runtime instance, capacity, split planning, proxy lifecycle.
- `network/` — request routing, proxying, tunneling, relay/discovery networking, request-affinity logic, endpoint rewrite, target health, OpenAI transport glue.
- `inference/` — model-serving logic, election, launch, pipeline, MoE behavior, embedded skippy integration.
- `system/` — machine-local environment and platform concerns (hardware detection, benchmarking, self-update, local system integration).
- `models/` — model catalog, resolution, downloads, local model storage, model metadata.
- `mesh/` — peer membership, gossip, heartbeats, identity, peer state, mesh node behavior.
- `plugin/` — plugin host, plugin runtime, transport, config, MCP bridge support.
- `plugins/` — concrete plugins (blobstore, flash_moe, openai_endpoint, telemetry, blackboard).
- `api/` — management API surface and route handling.
- `protocol/` — wire protocol types, encoding/decoding, conversions.
- `runtime_data/` — runtime data collection, API views, status snapshots.
- `crypto/` — host-side crypto helpers.

CLI ownership rule.

- All command handlers belong under `crates/mesh-llm-host-runtime/src/cli/`, usually `cli/commands/`.
- Domain modules should not own Clap parsing or top-level command dispatch.
- Domain modules may expose reusable functions that CLI handlers call.

Do not introduce generic buckets.

- Avoid directories or modules named `app`, `utils`, `misc`, `common`, or similar catch-alls.
- Name modules after the responsibility they own.

Keep shared code honest.

- If code is only used by one subsystem, keep it inside that subsystem.
- Only move code to a shared module (or a shared workspace crate like `mesh-llm-types` / `mesh-llm-routing`) when it is truly cross-domain.
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

1k LoC refactoring rule.

- When touching a source file that is already over 1,000 lines, first check whether the change adds or exposes a separable responsibility.
- If it does, split that responsibility into a semantically named module as part of the change, and keep the new file under 1,000 lines.
- If a full split is too risky for the current task, make the smallest useful extraction and call out the remaining oversized file in the final summary.
- Add or move tests so the extracted module owns tests for the behavior it now owns.
- Do not create generic buckets just to reduce line count; split by domain responsibility and keep ownership obvious.

Naming rule.

- File and module names should describe responsibility, not implementation detail.
- Prefer names like `affinity`, `discovery`, `transport`, `maintenance`, `warnings`.
- Avoid vague names like `helpers`, `stuff`, `logic`, or `manager` unless the abstraction is genuinely that broad.

When to add a new workspace crate.

- Prefer adding modules inside an existing crate first.
- Add a new `crates/<name>/` only when the responsibility is genuinely cross-cutting (used by host and client, or host and a separate binary) or when isolating compile time / dependencies for a specific binary or FFI surface.
- New crates should be named after the responsibility they own, not the consumer (e.g., `model-resolver` not `mesh-llm-model-helpers`).

Current structure notes.

- Request-affinity code belongs with networking/routing behavior (`network/affinity.rs`), not `system/`.
- Plugin MCP support belongs inside `mesh-llm-host-runtime/src/plugin/`, not as a separate root module.
- Model command handlers belong in `mesh-llm-host-runtime/src/cli/commands/`; `models/` should stay domain-focused.
- The shipped binary crate (`crates/mesh-llm/`) should remain a thin shim; do not move domain logic into it.

## Code Quality Rules for New Code

- Do not add Rust methods or functions over the configured Clippy line-count
  limit. Split long logic into semantically named helpers before it reaches the
  configured `too_many_lines` threshold.
- Do not add Rust source files over 2,000 lines. If a file is approaching that
  size, split it by responsibility into an owning module instead of adding more
  code to the oversized file.
- Do not add Rust code over the configured cognitive-complexity limit. Prefer
  small, named decision helpers and clear control-flow phases instead of nested
  branching.
- Treat these as design constraints for new code, not cleanup suggestions after
  the fact. CI runs Clippy with warnings denied, so configured Clippy warnings
  must be resolved before a PR can pass.

## Key Source Files

Host runtime (main monolith — `crates/mesh-llm-host-runtime/src/`):

- `lib.rs` — crate entry; exposes `run_main` (called from `crates/mesh-llm/src/main.rs`).
- `runtime/mod.rs` — top-level startup flows, runtime orchestration, command dispatch.
- `runtime/instance.rs` — per-instance runtime directory management: `InstanceRuntime`, pidfiles, flock liveness, scoped orphan reaping, local instance scanning.
- `runtime/local.rs` — local model startup loop.
- `runtime/discovery.rs` — discovery loops and auto-mode coordination.
- `runtime/proxy.rs`, `runtime/proxy/` — HTTP proxy lifecycle from the runtime side.
- `runtime/capacity.rs`, `runtime/split_planning.rs`, `runtime/context_planning.rs` — placement/sizing decisions.
- `mesh/mod.rs` — `Node` struct, mesh_id, peer management.
- `mesh/gossip.rs` — gossip wire format and peer state updates.
- `mesh/heartbeat.rs` — heartbeat publishing and freshness.
- `inference/election.rs` — host election, tensor split calculation.
- `inference/skippy/` — embedded staged runtime integration.
- `inference/pipeline.rs` — inference pipeline coordination.
- `inference/virtual_llm.rs` — virtual LLM (inter-model collaboration).
- `network/proxy.rs` — HTTP proxy: request parsing, model routing, response helpers.
- `network/router.rs` — request classification, model scoring, multimodal routing.
- `network/nostr.rs` — Nostr discovery, `score_mesh()`, `smart_auto()`.
- `network/tunnel.rs` — TCP ↔ QUIC relay (RPC + HTTP).
- `network/affinity.rs` — request-affinity tracking.
- `network/target_health.rs` — target health tracking.
- `network/openai/` — OpenAI transport glue.
- `api/mod.rs`, `api/routes/` — management API (:3131): `/api/status`, `/api/events`, `/api/discover`.
- `models/catalog.rs` — model catalog, HuggingFace downloads.
- `models/capabilities.rs` — multimodal/vision/audio/reasoning capability inference.
- `models/resolve/` — model reference resolution.
- `plugins/blobstore/mod.rs` — request-scoped media object storage for multimodal.
- `plugins/flash_moe/`, `plugins/openai_endpoint/`, `plugins/telemetry/`, `plugins/blackboard/` — other in-tree plugins.
- `cli/mod.rs`, `cli/commands/` — Clap command surface and dispatch.

Shipped binary (`crates/mesh-llm/src/`):

- `main.rs` — builds the Tokio runtime (custom stack size via `MESH_TOKIO_STACK_SIZE`) and calls `mesh_llm::run_main()`.
- `lib.rs` — `pub use mesh_llm_host_runtime::*;` (transitional re-export).

Embedded staged runtime (`crates/skippy-*`):

- `skippy-ffi/src/lib.rs` — Rust ABI mirror of the patched llama.cpp staged runtime; `ABI_VERSION_*` constants must stay in sync with `skippy/common.h` in the patch queue.
- `skippy-runtime/src/package.rs` — layer-package materialization, identity-bound cache.
- `skippy-runtime/src/devices.rs` — backend device enumeration.
- `skippy-server/src/frontend.rs`, `skippy-server/src/frontend/` — embedded chat/generation frontend.
- `skippy-server/src/runtime_state.rs` — KV-slot, lane, session state machine.
- `skippy-server/src/binary_transport.rs`, `binary_transport/` — binary transport to embedded server.

OpenAI-compatible HTTP frontend (`crates/openai-frontend/src/`):

- `router.rs`, `chat.rs`, `completions.rs`, `responses.rs`, `models.rs`, `sse.rs`, `backend.rs` — OpenAI surface.

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

## Skippy ABI Compatibility

The patched llama.cpp staged runtime has its own ABI version, tracked in `skippy/common.h` (inside the patch queue) and mirrored by `SKIPPY_ABI_VERSION_*` constants in `crates/skippy-ffi/src/lib.rs`.

- When changing the staged-runtime ABI in the patch queue, bump `SKIPPY_ABI_VERSION_PATCH` (or MINOR/MAJOR) in `skippy/common.h` AND keep the Rust constants in `skippy-ffi/src/lib.rs` in sync in the same change.
- `skippy-runtime` consumes the ABI version for package loading and feature probing; an out-of-sync mirror will silently advertise the wrong version.
- Treat the staged-runtime ABI the same as the mesh wire protocol: additive changes preferred, breaking changes need explicit acknowledgement.

## UI Notes

For changes in `crates/mesh-llm-ui/`, use components and compose interfaces consistently with shadcn/ui patterns. Prefer extending existing primitives in `src/components/ui/` over ad-hoc markup.

## Testing

Read `docs/design/TESTING.md` before running tests. It has all test scenarios, remote deploy instructions, and cleanup commands.

Testing matters more than usual in this project because:

- Nodes run on different machines with different hardware and OS versions. Bugs that don't reproduce locally can appear in real deployments.
- The mesh protocol is a distributed system — gossip, election, and routing interact across nodes. Single-node unit tests don't catch protocol-level regressions.
- The public mesh at meshllm.cloud runs continuously. Breaking changes that pass local tests can take down live inference for real users.
- Multimodal, MoE splitting, and multi-model routing all have complex interaction paths that are hard to reason about statically.

When making changes that touch gossip, routing, proxy, election, or capability advertisement, test against at least two nodes before merging. The deploy checklist below is not optional.

### Confidence Testing (multi-node, when warranted)

For changes that affect routing, MoA, gossip, the OpenAI surface, agent harnesses, or anything multi-node, validate with these three shapes before declaring a branch ready:

1. **2-node private mesh** — start one node with `mesh-llm serve --model <big> --port 9337 --console 3131`, grab its invite token from the JSON log, and start the second node with `mesh-llm serve --gguf <small.gguf> --port 9447 --console 3145 --join <token>`. Confirm peers=1 on both consoles and `/v1/models` returns the union. Exercises QUIC tunnelling and cross-node routing.
2. **Public mesh as a client** — `mesh-llm client --auto` from a workstation. Confirm `discovery_joined` + `Client ready` in the log and an inference call against a mesh-advertised model returns. Exercises the read-only routing path agent users hit.
3. **Agent harness** — run ≥ 1 of the harnesses (“mini-agent” Python loops at `/tmp/mini-agent*.py`, Goose, OpenCode) against the local proxy with both `model=auto` and `model=mesh` to catch tool-call and reducer regressions that simple curl checks miss.

### Cargo Concurrency

Run `cargo` commands serially. Do not run multiple `cargo` commands in parallel (including parallel test runs), because this repo frequently hits Cargo lock conflicts (`package cache` / `artifact directory`) under concurrent invocation.

### Which crate to `-p`

- Touched `mesh-llm-host-runtime` or the shipped `mesh-llm` binary — use `-p mesh-llm` for build/check (it pulls the host runtime through its single dep) and `-p mesh-llm-host-runtime` for focused tests.
- Touched a specific workspace crate (e.g., `skippy-runtime`, `openai-frontend`, `mesh-client`) — run `cargo check -p <crate>` and `cargo test -p <crate> --lib` for fast iteration.
- For broad refactors, fall back to `cargo check --workspace` (serially!).

## Running mesh-llm locally

Default the launch to a normal foreground run (TUI visible) unless you have a
specific reason to suppress UI surfaces. Most observation/debug tasks do not
need the TUI suppressed.

- `mesh-llm client --auto` — normal foreground run with the TUI. Use this by
  default.
- `--log-format json` — emits machine-parseable JSON log lines. Use this when
  you want to programmatically read events.
- `--headless` — disables the **embedded web UI**, not the TUI. The TUI still
  draws. Only use `--headless` when you are intentionally avoiding the
  management web console — it is **not** the way to get a quiet background run.
- `--no-console` — fully disables the management console (HTTP API on the
  console port).
- `nohup … &` with a foreground binary that draws a TUI will appear to run but
  often exits or behaves oddly when the TUI cannot attach to a terminal. Prefer
  letting the developer launch the binary in their own terminal and observing
  via `/api/status`, `--log-format json`, or by reading stderr.

Do not reach for `--headless` to "go quiet" — that is a recurring mistake. If
you want quiet output, use `--log-format json` and parse what you need.

## Pre-Commit Checklist

Before committing, run the local checks most likely to fail in CI for the files you touched. Do not rely on CI to catch basic formatting, compile, or stale UI build issues.

### Minimum bar before every commit

- Rust-only change — format the changed Rust files and run `cargo check -p <touched-crate>` (and `cargo check -p mesh-llm` if you touched anything reachable from the shipped binary).
- UI-only change — run `just build`.
- Mixed Rust and UI change — run `just build`.

### Rust changes

- Format only the changed Rust files from the repo root, for example with `cargo fmt --all -- path/to/file.rs`, and include those formatting changes in the commit.
- Before committing Rust changes, ensure the formatting check passes with `cargo fmt --all -- --check`.
- After Rust changes, run `cargo check` for each touched crate (`-p <crate>`), and at least `cargo check -p mesh-llm` if the change is reachable from the shipped binary.
- If you touched tests, public APIs, routing, inference, gossip, plugin protocol, skippy ABI, or CLI behavior, run the relevant tests before committing.
- If you touched `proto/`, any `protocol/` module, `mesh-llm-host-runtime/src/mesh/gossip.rs`, `mesh-llm-host-runtime/src/mesh/mod.rs`, routing, election, API serialization, or `skippy-ffi` ABI constants, do not stop at build-only validation: run at least `cargo test -p mesh-llm-host-runtime --lib` (plus `cargo test -p skippy-ffi --lib` / `-p skippy-runtime --lib` when ABI is touched) and wait for it to exit successfully before committing.
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

## Running mesh-llm in the Background (for Testing)

When running `mesh-llm serve` from an agent for testing, the process is non-interactive — it just runs. There is no interactive prompt or TUI to worry about. Use standard backgrounding:

```bash
bash -c './target/debug/mesh-llm serve --model "..." --auto > /tmp/mesh.log 2>&1 & disown; echo "PID=$!"'
```

- **Do not use `--headless`** — it disables the web UI but does not change process behavior. The name is misleading and does not help with backgrounding.
- The mesh process writes TUI-formatted output to stderr which looks like errors but is normal.
- Wait for models to appear via polling `curl -s http://localhost:9337/v1/models` before sending requests.
- Kill with `pkill -f "target/debug/mesh-llm"` or `pkill -f mesh-llm`.

## Deploy Checklist — MANDATORY

**Every deploy to test machines MUST follow this checklist.**

### Before starting nodes
1. **Bump VERSION** in `crates/mesh-llm/Cargo.toml` (the shipped binary crate) so you can verify the running binary is new code.
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
`~/.mesh-llm/runtime/` for the active instance metadata. Embedded
skippy/llama.cpp native logs are redirected away from the TUI into the active
instance runtime directory:

```text
<runtime-root>/<pid>/logs/skippy-native.log
```

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

- **No `api_key_token` feature** — explicitly rejected, removed in v0.26.0.
- **No credentials in tracked files** — IPs, passwords, SSH commands belong in `~/Documents/private-note.txt` only.
- **No domain logic in `crates/mesh-llm/src/`** — that crate is a thin shim over `mesh-llm-host-runtime`; put new code in the host-runtime crate (or a more specific peer crate).
- **No external `llama-server` / `rpc-server` runtime lane** — the embedded staged runtime via patched llama.cpp is the only supported path.
