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
| `PLAN.md` | Historical design notes and benchmarks |
| `mesh-llm/TODO.md` | Current work items and backlog |
| `mesh-llm/README.md` | Rust crate overview and file map |
| `mesh-llm/docs/DESIGN.md` | Architecture, protocols, features |
| `mesh-llm/docs/TESTING.md` | Test playbook, scenarios, remote deploy |
| `mesh-llm/docs/SAME_ORIGIN_PARITY_WORKFLOW.md` | Workflow for downloading original checkpoints, converting same-origin GGUF/MLX pairs, validating them, publishing them to `meshllm`, and switching matrix rows |
| `mesh-llm/docs/MoE_PLAN.md` | MoE expert sharding design |
| `mesh-llm/docs/MoE_DEPLOY_DESIGN.md` | MoE auto-deploy UX |
| `mesh-llm/docs/MoE_SPLIT_REPORT.md` | MoE splitting validation results |
| `fly/README.md` | Fly.io deployment (console + API apps) |
| `relay/README.md` | Self-hosted iroh relay on Fly |

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

See `CONTRIBUTING.md` for full dev workflow.

## Project Structure

- `mesh-llm/src/` — Rust source
- `mesh-llm/ui/` — React web console (shadcn/ui patterns, see https://ui.shadcn.com/llms.txt)
- `mesh-llm/docs/` — Design and testing docs
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

## Key Source Files

- `mesh-llm/src/main.rs` — CLI args, orchestration: `run_auto()`, `run_idle()`, `run_passive()`
- `mesh-llm/src/mesh.rs` — `Node` struct, gossip, mesh_id, peer management
- `mesh-llm/src/election.rs` — Host election, tensor split calculation
- `mesh-llm/src/proxy.rs` — HTTP proxy: request parsing, model routing, response helpers
- `mesh-llm/src/api.rs` — Management API (:3131): `/api/status`, `/api/events`, `/api/discover`, `/api/join`
- `mesh-llm/src/nostr.rs` — Nostr discovery, `score_mesh()`, `smart_auto()`
- `mesh-llm/src/download.rs` — Model catalog (`MODEL_CATALOG`), HuggingFace downloads
- `mesh-llm/src/moe.rs` — MoE detection, expert rankings, split orchestration
- `mesh-llm/src/launch.rs` — llama-server/rpc-server process management

## Plugin Protocol Compatibility

When iterating on the plugin protocol, always consider protocol compatibility.

- If a protocol change may be breaking, explicitly ask the developer whether the change is intended to be breaking.
- If the change is not intended to be breaking, the previous version of the plugin protocol must continue to be supported.
- Do not silently ship plugin protocol changes that strand older plugins or hosts without confirming that outcome is acceptable.

## UI Notes

For changes in `mesh-llm/ui/`, use components and compose interfaces consistently with shadcn/ui patterns. Prefer extending existing primitives in `ui/src/components/ui/` over ad-hoc markup.

## Testing

Read `mesh-llm/docs/TESTING.md` before running tests. It has all test scenarios, remote deploy instructions, and cleanup commands.

## Validation Baselines

If the bundled `llama.cpp` source, branch pin, or effective build commit changes,
rerun the checked-in validation matrix and compare the new results to the
checked-in baseline data under `testdata/validation/` before treating the new
build as equivalent.

At minimum, rerun the canonical GGUF side:

```bash
python3 scripts/run-validation-matrix.py --backend gguf --skip-build --stamp <stamp>
```

Review:

- `MLX_VALIDATION_RESULTS/<stamp>/exact-baseline-comparison.tsv`
- `MLX_VALIDATION_RESULTS/<stamp>/behavior-baseline-comparison.tsv`
- `MLX_VALIDATION_RESULTS/<stamp>/parity-vs-canonical-baseline.tsv`

When replacing a noisy public pair with a same-origin pair derived from the
original upstream checkpoint, follow:

- `mesh-llm/docs/SAME_ORIGIN_PARITY_WORKFLOW.md`

Do not silently assume `llama.cpp` changes preserve serving behavior, even when
the Rust code is unchanged.

## Command Concurrency

Do not run Rust build, test, or format commands in parallel in the same worktree.

- Never run multiple `cargo build`, `cargo test`, `cargo check`, or `cargo fmt` commands at the same time.
- Never run `just build` in parallel with any Cargo command.
- Prefer sequential Rust verification steps to avoid Cargo package-cache and target-dir lock contention.
- If a Rust build/test command is already running, wait for it to finish before starting another.
- Parallel tool use is fine for reads like `rg`, `sed`, `git status`, and `git diff`, but not for Rust build/test commands.
- When using `multi_tool_use.parallel`, do not include more than one Rust build/test/format command in the same batch.

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

### Deploy to Remote

```bash
just bundle
# scp bundle to remote, tar xzf, codesign -s - the three binaries
```

### Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

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

If llama-server fails to start (stuck at "⏳ Starting llama-server..."), check its log file. Rust's `std::env::temp_dir()` on macOS points to the per-user temp dir, **not** `/tmp`:

```bash
cat "$(python3 -c 'import tempfile; print(tempfile.gettempdir())')/mesh-llm-llama-server.log"
```

Typical path: `/var/folders/XX/.../T/mesh-llm-llama-server.log`. rpc-server logs are in the same directory as `mesh-llm-rpc-{port}.log`.

### Common failures
- **Use `tmux` for any long-running remote process** — downloads, deploy steps, validation runs, server startup, and similar remote work should be launched inside `tmux`, not as a plain foreground SSH command and not via `nohup`. This keeps the login environment intact, survives disconnects more reliably, and makes progress/log inspection easier.
- **Use the native login shell style for remote commands** — when SSHing to a Mac, prefer `zsh -lc "<command>"`; when SSHing to Linux, prefer `bash -lc "<command>"`. This loads the normal user environment, including `PATH` updates, exported env vars such as `HF_TOKEN`, Homebrew paths on macOS, and user-installed CLI tools.
- **Prefer `scp` + remote scripts over nested SSH one-liners** — for nontrivial remote macOS work, copy a small script to the remote host and run it via `zsh -lc`, optionally inside `tmux` for long-lived tasks. Avoid deeply nested `ssh 'zsh -lc ... tmux ...'` command chains when a script would be clearer and less error-prone.
- **Match generated remote scripts to the host shell** — when generating a script for a remote Mac, write it as a `zsh` script (for example `#!/bin/zsh`); when generating a script for remote Linux, write it as a `bash` script (for example `#!/usr/bin/env bash` or `#!/bin/bash`). Keep the script shell consistent with the remote command shell you use to launch it.
- **Verify `tmux` sessions actually stayed up** — when launching a long-running remote process in `tmux`, do not assume `tmux new-session -d ...` means success. Launch a small remote script, redirect its stdout/stderr to real log files, then verify the session with `tmux has-session -t <name>` or `tmux ls` both immediately and again after a short delay before treating it as live.
- **Use absolute paths for remote tools when needed** — on remote Macs, even with `zsh -lc`, important tools may still be more reliable when referenced explicitly, especially Homebrew binaries like `/opt/homebrew/bin/tmux` and user-installed CLIs under `$HOME/Library/Python/.../bin`.
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
