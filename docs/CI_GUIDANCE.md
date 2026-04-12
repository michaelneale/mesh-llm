# CI guidance

This document captures the repo's CI design rules and workflow responsibilities.

The core rule is to keep ordinary pull request CI fast and targeted, while keeping release-grade artifact production and publish gating in the release workflow.

## CI design goals

The workflow layout should preserve these invariants:

1. Pull request feedback should optimize for fast validation, not release-like fidelity everywhere.
2. GPU caches restored in PR CI should be warmed on `main`, with PR jobs acting as restore-only consumers.
3. Slim CI GPU shapes should stay distinct from fat release artifact shapes.
4. Release-only behavior should stay in `.github/workflows/release.yml`.
5. Script-level tuning for determinism and cache correctness should remain intact.

## Workflow responsibilities

### `.github/workflows/ci.yml`

PR CI is the fast validation path.

- Keep the `changes` path filter gate so docs-only and other low-impact edits do not trigger unnecessary backend work.
- Keep Linux and macOS producer lanes focused on the smallest work that still proves correctness.
- Keep GPU PR lanes slim and tuned for cache restore plus representative validation, not release-style rebuilds.
- Allow producer jobs to upload the exact binary shape they already build for their lane.
- Keep cheap CLI and boot smokes in producer jobs when they provide fast early failure.

### `.github/workflows/smoke.yml`

Smoke testing should consume previously built Linux inference binaries instead of rebuilding them.

- Download the uploaded artifact from the producer job.
- Stage `mesh-llm`, `rpc-server`, `llama-server`, `llama-moe-analyze`, and `llama-moe-split` into the expected paths.
- Own the heavier inference checks, including real inference, OpenAI compatibility, split-mode, and MoE smokes.

### `.github/workflows/warm-caches.yml`

This workflow is the single writer for warmed GPU caches.

- Keep explicit cache input hashing and pruning.
- Preserve both slim and fat warming where needed.
- Do not move warmed GPU cache writes back into PR CI.

### `.github/workflows/gpu-warm-cache-job.yml`

This reusable job is cache warm plumbing, not a PR CI artifact producer.

- Preserve the restore, short-circuit, build, save flow.
- Keep verification of restored and saved binaries.

### `.github/workflows/release.yml`

Release workflows own shipping artifacts and release gating.

- Build the full release artifact set here, not in ordinary PR CI.
- Produce Linux inference binaries for downstream release smoke testing.
- Keep `publish` gated on successful release smoke tests.

## Artifact handoff rules

Artifact reuse is good when it avoids duplicate rebuilds. The producer lane should emit the binary shape it is already responsible for validating, and downstream smoke jobs should reuse those binaries.

Do not widen a producer lane from debug or slim CI shape to release or fat shape just because artifact upload is convenient.

For Linux inference smoke reuse in PR CI:

- the producer job should upload the already-validated `mesh-llm` binary and required llama.cpp executables
- the downstream smoke job should download and stage those files
- the smoke job should not perform a meaningful rebuild of `mesh-llm` or llama.cpp

## Cache boundaries

GPU cache behavior is intentionally asymmetric.

- PR CI restores warmed GPU caches.
- `warm-caches.yml` writes warmed GPU caches for `main`.
- Cache keys and pruning rules should stay explicit and deterministic.
- CI cache tuning in scripts and workflows should not be weakened just to make packaging easier.

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
- PR CI does not write warmed GPU caches
- GPU PR lanes still consume slim warmed caches
- Linux inference smokes reuse uploaded binaries instead of rebuilding the same payload
- release workflows still build shipping artifacts separately from PR CI
- release publish remains gated on release smoke success
- no step reintroduces duplicate builds that tuned CI intentionally removed
- no step widens slim CI GPU inputs into release defaults without a measured reason

## Short version

Keep the fast PR CI mechanics, reuse artifacts to separate producer work from heavier smokes, and keep release-grade builds and publish gating in the release workflow.
