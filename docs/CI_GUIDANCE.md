# CI guidance

Use these CI design rules to keep workflow responsibilities clear across the
repo.

The core rule is to keep ordinary pull request CI fast and targeted, while keeping release-grade artifact production and publish gating in the release workflow.

## PR workflow split

The PR CI design splits early quality checks from build and smoke validation.

- `pr_quality.yml` is quality-only and is named **PR Quality Checks**. It runs formatting, UI quality, and binpacked clippy before build workflows finish.
- `pr_ci.yml` owns PR build producers, integration tests, smoke tests, SDK smokes, and PR platform matrices.
- `pr_docker.yml` owns PR Docker build validation without publishing images.
- `pr_cleanup.yml` deletes PR-scoped GitHub Actions caches when a pull request is closed.
- `ci.yml`, `docker.yml`, and `release.yml` are non-PR workflows for `main`, tags, dispatches, and release-grade publishing.

## invariants

Smallest possible build unit on PR — only affected crates compile and test, and native backend lanes rebuild only when backend inputs change.
Code quality always runs and is separate from builds — `pr_quality.yml` is fully independent of `pr_ci.yml`.
Documentation-only updates never trigger code builds — `docs_only` short-circuit gates every heavy job.
SDK smokes start from the first relevant producer binary instead of waiting for unrelated inference smokes.

## Change-matrix

| Change class | Behavior |
|---|---|
| `**.md` only | Runs changes summary, sets `docs_only`, and gates every heavy job. |
| Single leaf crate | Runs affected and reverse-dep quality/test routing; backend GPU and Windows full-build lanes stay skipped unless backend inputs changed. |
| UI only | Runs UI quality and the Linux and macOS UI build and test cache path, with no Rust steps. |
| `Cargo.lock` | Takes the full-workspace path. |
| `third_party/llama.cpp/**` | Takes the full-workspace ABI path. |
| SDK/API crate | Runs SDK/API tests and starts native/Kotlin/Swift SDK smokes as soon as Linux/macOS producer binaries are available. |

## UI dist cache

Contract for the UI dist cache:

- Writer: Linux on `main` when `ui == true`.
- Reader: macOS.
- Miss fallback: `pnpm i` and rebuild the UI dist.
- Guard: fail if `crates/mesh-llm-ui/dist` is still missing after the restore or rebuild path.
- Cache key: `${CACHE_NAMESPACE}-ui-dist-<git-hash-object of crates/mesh-llm-ui + .github/cache-version.txt>`.
- GPU and SDK jobs do not use this cache. They rely on `MESH_LLM_SKIP_UI=1` placeholder semantics instead.

## affected-crates

The affected-crates and routing contract is:

- Input comes from stdin file list or CLI args.
- Output JSON shape includes `{affected,test_crates,batches,all_rust,ui_changed}`.
- The script fails open by emitting `all_rust` with the full workspace and exits 0.
- The composite `compute-changes` action adds `clippy_batches_json`, `backend_changed`, `sdk_smoke_required`, `docs_only`, and `rust_changed` for workflow routing.
- `scripts/plan-clippy-batches.sh` is the repeatable clippy binpack planner. It uses deterministic weights and first-fit decreasing placement so full-workspace clippy is not concentrated in one runner.

## Required status

Branch protection should migrate to these names:

- `PR Quality Checks / summary`
- `PR CI / linux`
- `PR CI / inference_smoke_tests`
- Other currently required checks can stay as-is during migration, but they should point at the new workflow names once the split lands.

## CI design goals

The workflow layout should preserve these invariants:

1. Pull request feedback should optimize for fast validation, not release-like fidelity everywhere.
2. Slim CI GPU shapes should stay distinct from fat release artifact shapes.
3. Release-only behavior should stay in `.github/workflows/release.yml`.
4. Script-level tuning for determinism and cache correctness should remain intact.

## Workflow responsibilities

### `.github/workflows/pr_quality.yml`

PR Quality Checks is the earliest feedback lane.

- Keep formatting, clippy, and UI quality independent from build producer jobs.
- Use `clippy_batches_json` from `compute-changes`, not a hardcoded batch list.
- Keep clippy deterministic: the same affected crate set must produce the same bins and job names.

### `.github/workflows/pr_ci.yml`

PR CI is the fast validation path.

- Keep the `changes` path filter gate so docs-only and other low-impact edits do not trigger unnecessary backend work.
- Keep Linux and macOS producer lanes focused on the smallest work that still proves correctness.
- Keep GPU PR lanes slim and representative (single arch), not release-style rebuilds.
- Allow producer jobs to upload the exact binary shape they already build for their lane.
- Keep cheap CLI and boot smokes in producer jobs when they provide fast early failure.
- Gate backend platform rebuilds on `backend_changed`, not on every Rust change.
- Gate SDK smokes on `sdk_smoke_required` and producer readiness, not on the slower inference smoke path.

### `.github/workflows/pr_docker.yml`

PR Docker validates the client image build without publishing images.

- Keep all PR Docker behavior in `pr_docker.yml`; `docker.yml` remains for dispatch/tag publishing.
- Use short-lived PR build validation rather than shared release tags.

### `.github/workflows/pr_cleanup.yml`

PR cleanup removes non-shared PR cache data.

- Use `pull_request_target` only for cache API cleanup.
- Do not check out or run pull request code.
- Delete caches by `refs/pull/<number>/merge` so main/shared caches remain intact.

### `.github/workflows/ci.yml`

Main CI owns the `main` branch validation shape and no longer handles PR events.

### `.github/workflows/smoke.yml`

Smoke testing should consume previously built Linux inference binaries instead of rebuilding them.

- Download the uploaded artifact from the producer job.
- Stage the built `mesh-llm` binary and any current runtime assets into the expected paths.
- Own the heavier inference checks, including real inference, OpenAI compatibility, and staged serving smokes.

### `.github/workflows/reset-caches.yml`

Cache reset deletes all repository caches. Use it sparingly when cache corruption or stale state needs a full purge.

### `.github/workflows/release.yml`

Release workflows own shipping artifacts and release gating.

- Build the full release artifact set here, not in ordinary PR CI.
- Produce Linux inference binaries for downstream release smoke testing.
- Keep `publish` gated on successful release smoke tests.

## Artifact handoff rules

Artifact reuse is good when it avoids duplicate rebuilds. The producer lane should emit the binary shape it is already responsible for validating, and downstream smoke jobs should reuse those binaries.

Do not widen a producer lane from debug or slim CI shape to release or fat shape just because artifact upload is convenient.

For inference and SDK smoke reuse in PR CI:

- the Linux producer should upload the already-validated `mesh-llm` binary for Linux inference, native SDK, Kotlin SDK, and two-node smokes
- the macOS producer should upload the already-validated `mesh-llm` binary for Swift SDK smoke
- downstream smoke jobs should download and stage those files
- smoke jobs should not perform a meaningful rebuild of `mesh-llm` or llama.cpp

## Cache boundaries

CI caching is straightforward: sccache handles Rust compilation artifacts, Swatinem/rust-cache provides artifact reuse, and actions/cache stores integration test models. No separate GPU cache warming mechanism exists.

Keep these rules in place:

- PR merge refs should not save the large shared Rust caches.
- Main remains the place where shared caches are written and refreshed.
- PR artifacts should use short retention windows.
- `pr_cleanup.yml` deletes cache entries for closed PR merge refs.

## Build shape rules

Keep CI validation shape separate from release shape.

- PR CPU and macOS lanes should stay optimized for fast feedback.
- PR CUDA and ROCm lanes should stay slim and representative.
- Release-only settings such as broader GPU matrices or safer full release defaults must remain release-only.
- Do not silently disable release safety settings for shipping artifacts.

## Docker publish contract

`.github/workflows/docker.yml` publishes to `ghcr.io/<owner>/mesh-llm` with two classes of tags:

- **Public tags** are the stable tags users pull: `latest`, `client`, `<version>`, `sha-<short>`, `cpu`, `<version>-cpu`, `sha-<short>-cpu`, `vulkan`, `<version>-vulkan`, `sha-<short>-vulkan`, `cuda`, `<version>-cuda`, `sha-<short>-cuda`, `rocm`, `<version>-rocm`, and `sha-<short>-rocm`.
- **Merge-source tags** are internal per-architecture tags used only so the merge jobs can assemble multi-arch manifests.

Keep these merge-source edges intact:

- `docker-client-merge` consumes `sha-<short>-amd64` and `sha-<short>-arm64`.
- `docker-cpu-merge` consumes `sha-<short>-cpu-amd64` and `sha-<short>-cpu-arm64`.
- `docker-vulkan-merge` consumes `sha-<short>-vulkan-amd64` and `sha-<short>-vulkan-arm64`.
- `docker-cuda` and `docker-rocm` are amd64-only publishers, so they do not have merge jobs.

`latest` must continue to resolve to the merged `client` image. Any workflow change is incorrect if a merge job references a source tag that its producer jobs do not push exactly.

## Script expectations

The workflow design depends on the build scripts preserving the distinction between CI-friendly and release-friendly builds.

- `scripts/build-linux.sh` should keep support for pinned llama.cpp SHAs used for deterministic cache correctness.
- CI-only opt-outs should stay clearly scoped as CI-only behavior.
- Release-oriented defaults should remain the safer defaults for shipping builds.

## AMD / NVIDIA / Intel naming

Vendor aliases are acceptable when they improve readability, but they should remain thin wrappers over the ROCm / CUDA / oneAPI behavior rather than introducing new artifact semantics.

## Validation checklist

Changes to CI are only correct when all of the following remain true:

- docs-only changes still skip expensive backend work
- UI-only changes still avoid the full backend and GPU matrix
- leaf crate changes still keep backend platform lanes skipped unless backend inputs changed
- clippy batches are generated by the binpack script rather than hand-maintained
- SDK smokes start from producer binaries instead of waiting for unrelated inference smokes
- GPU PR lanes stay slim (single arch) and representative
- Linux inference smokes reuse uploaded binaries instead of rebuilding the same payload
- macOS Swift SDK smokes reuse the macOS producer binary instead of rebuilding it
- release workflows still build shipping artifacts separately from PR CI
- release publish remains gated on release smoke success
- no step reintroduces duplicate builds that tuned CI intentionally removed
- no step widens slim CI GPU inputs into release defaults without a measured reason

## Short version

Keep fast PR feedback split across `pr_*.yml` workflows, reuse producer artifacts for smokes, route work by affected crates and backend inputs, and keep release-grade builds and publish gating outside PR CI.
