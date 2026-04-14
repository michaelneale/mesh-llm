# Issue #115 Spec Compliance Tracker

**Issue**: https://github.com/Mesh-LLM/mesh-llm/issues/115

This file is the human-readable counterpart to `.github/workflows/docker-precheck.yml`. Both must agree. If they disagree, the reusable workflow is authoritative.

> **Spec Correction #1**: The spec says `COPY proto/ proto/` but the actual path is `mesh-llm/proto/` (verified via filesystem inspection: `mesh-llm/proto/node.proto`, `mesh-llm/proto/plugin.proto` exist; no root-level `proto/`). All Dockerfiles use the corrected path `COPY mesh-llm/proto/ mesh-llm/proto/`.

## Open Question Resolutions

| Question | Answer | Evidence |
|----------|--------|----------|
| Q1: CLI form | `mesh-llm --client --auto --port 9337 --console 3131 --listen-all` (bare flags) | mesh-llm/src/cli/mod.rs — top-level Cli struct accepts bare flags |
| Q2: :latest tag target | `client` (most portable, smallest, no GPU dependency) | Design decision |
| Q3: CUDA compute_120 (Blackwell) | Works with CUDA 12.8.0 (compute_120 included in Justfile release-build-cuda) | Justfile line 53, release.yml line 78 |
| Q4: Non-root user policy | Stay root (matches existing fly/Dockerfile; K3S users override via securityContext) | Design decision |

## Spec Compliance Table

| # | Spec Requirement | Enforcing File:Line | Grep Verifier | Status |
|---|------------------|---------------------|---------------|--------|
| 1 | ui-builder runs before rust-builder in every Dockerfile | docker/Dockerfile.client:46,53; docker/Dockerfile.cpu:50,57; docker/Dockerfile.vulkan:45,52; docker/Dockerfile.rocm:48,55; fly/Dockerfile:49,56 | `grep -n "COPY --from=ui-builder"` | ✅ verified (client, cpu, vulkan, rocm, fly) |
| 2 | cargo-chef full workspace copy (incl. mesh-llm/proto/ corrected path) — NO stub lib.rs | docker/Dockerfile.client:30-36; docker/Dockerfile.cpu:29-37; docker/Dockerfile.vulkan:29-37; docker/Dockerfile.rocm:30-38; fly/Dockerfile:34-40 | `grep -n "COPY mesh-llm/proto"` | ✅ verified (client, cpu, vulkan, rocm, fly) |
| 3 | libdbus-1-dev in rust-builder apt | docker/Dockerfile.client:24; docker/Dockerfile.cpu:28; docker/Dockerfile.vulkan:23; docker/Dockerfile.rocm:28; fly/Dockerfile:27 | `grep -n "libdbus-1-dev"` | ✅ verified (client, cpu, vulkan, rocm, fly) |
| 4 | master branch (NOT rebase-upstream-master) | docker/Dockerfile.cpu:66; docker/Dockerfile.vulkan:69; docker/Dockerfile.rocm:65 | `grep -n "master"` | ✅ verified (cpu, vulkan, rocm) |
| 5 | GGML_CUDA_FA_ALL_QUANTS=ON in Dockerfile.cuda | docker/Dockerfile.cuda (cmake flags section) | `grep -n "GGML_CUDA_FA_ALL_QUANTS=ON"` | ✅ verified (cuda) |
| 6 | Flavored binary naming in /usr/local/lib/mesh-llm/bin/ | docker/Dockerfile.cpu:80-82,89-91; docker/Dockerfile.vulkan:84-86,93-95; docker/Dockerfile.rocm:79-81,90-92 | `grep -n "rpc-server-cpu\|llama-server-cpu\|rpc-server-vulkan\|llama-server-vulkan\|rpc-server-rocm\|llama-server-rocm"` | ✅ verified (cpu, vulkan, rocm) |
| 7 | No `just bundle` in any Dockerfile | docker/Dockerfile.client (absent); docker/Dockerfile.vulkan (absent); docker/Dockerfile.rocm (absent); fly/Dockerfile (absent) | `` ! grep -rn "just bundle" docker/ fly/Dockerfile `` | ✅ verified (client, cpu, vulkan, rocm, fly) |
| 8 | No `protobuf-compiler` apt install | docker/Dockerfile.client (absent); docker/Dockerfile.vulkan (absent); docker/Dockerfile.rocm (absent); fly/Dockerfile (absent) | `` ! grep -rn "protobuf-compiler" docker/ fly/Dockerfile `` | ✅ verified (client, cpu, vulkan, rocm, fly) |
| 9 | No QEMU in docker.yml workflow | .github/workflows/docker.yml:31-77,148,313,477 (6 reusable precheck jobs; 3 native ubuntu-24.04-arm runners; no setup-qemu-action) | `` ! grep -in "qemu" .github/workflows/docker.yml `` | ✅ verified (17 jobs, 3 native arm64 runners, 0 QEMU references) |
| 10 | fly/Dockerfile has ui-builder stage | fly/Dockerfile:18 (`FROM node:24-alpine AS ui-builder`); fly/Dockerfile:22 (`npm ci`); fly/Dockerfile:24 (`npm run build`); fly/Dockerfile:49 (`COPY --from=ui-builder`) | `grep -n "AS ui-builder" fly/Dockerfile` | ✅ verified (fly — latent bug fixed) |
| 11 | Shared cmake flags in every llama-builder | docker/Dockerfile.cpu:70-73; docker/Dockerfile.vulkan:73-76; docker/Dockerfile.rocm:69-74 | `grep -n "DGGML_RPC=ON\|DBUILD_SHARED_LIBS=OFF\|DLLAMA_OPENSSL=OFF\|DCMAKE_BUILD_TYPE=Release"` | ✅ verified (cpu, vulkan, rocm) |
| 12 | Runtime apt: ca-certificates, libgomp1, libdbus-1-3 | docker/Dockerfile.client:59; docker/Dockerfile.cpu:87; docker/Dockerfile.vulkan:90; docker/Dockerfile.rocm:91; fly/Dockerfile:62 | `grep -n "libdbus-1-3"` | ✅ verified (client, cpu, vulkan, rocm, fly) |
| 13 | entrypoint.sh 3-mode case (console/worker/default) — no api-only mode (UI always with runtime) | docker/entrypoint.sh:10-20 | `grep -n "worker)" docker/entrypoint.sh && ! grep -q "api)" docker/entrypoint.sh` | ✅ verified |

## Image Size Budgets

| Variant | Budget |
|---------|--------|
| client | < 250 MB |
| cpu | < 600 MB |
| vulkan | < 900 MB |
| cuda | < 4.5 GB |
| rocm | < 12 GB |

## Files Allowed to be Touched by This Plan

- `docker/Dockerfile.client` (create)
- `docker/Dockerfile.cpu` (create)
- `docker/Dockerfile.cuda` (create)
- `docker/Dockerfile.rocm` (create)
- `docker/Dockerfile.vulkan` (create)
- `docker/entrypoint.sh` (create)
- `docker/SPEC_COMPLIANCE.md` (create)
- `.github/workflows/docker-precheck.yml` (create)
- `.github/workflows/docker.yml` (create)
- `Justfile` (append-only; do not delete existing recipes)
- `fly/Dockerfile` (rewrite as multi-stage)
- `.dockerignore` (append-only; do not remove existing entries)

**Any change to a file outside this list is a plan violation.**

## Verification Log

<!-- Each task appends a 1-line entry here after running the reusable docker precheck workflow -->
Task 0.5 — docker/entrypoint.sh — verified spec item 13 (3-mode case, no api-only mode) at 2026-04-08
Task 1.1 — docker/Dockerfile.client — verified spec items 1, 2, 3, 7, 8, 12 (client image, 52MB, 0 missing libs) at 2026-04-08
Task 1.2 — docker/Dockerfile.cpu — verified spec items 4, 6, 11 (cpu image, 58MB, binaries: rpc-server-cpu llama-server-cpu llama-moe-split, llama-sha: b2f6a68) at 2026-04-08
Task 1.3 — docker/Dockerfile.vulkan — verified spec items 1, 2, 3, 4, 6, 7, 8, 11, 12 (vulkan image, 75MB, binaries: rpc-server-vulkan llama-server-vulkan llama-moe-split, mesh-llm 0.58.0) at 2026-04-08
Task 1.4 — docker/Dockerfile.cuda — verified spec item 5 (GGML_CUDA_FA_ALL_QUANTS=ON, cuda image 2717MB, binaries: rpc-server-cuda llama-server-cuda llama-moe-split, mesh-llm 0.58.0) at 2026-04-08
Task 1.5 — docker/Dockerfile.rocm — verified spec items 1, 2, 3, 4, 6, 7, 8, 11, 12 (rocm image 1305MB, binaries: rpc-server-rocm llama-server-rocm llama-moe-split, mesh-llm 0.58.0, HIPCXX/HIP_PATH exports, DGGML_HIP=ON, DAMDGPU_TARGETS ARG) at 2026-04-08
Task 2.2 — fly/Dockerfile — verified spec item 10 (ui-builder stage present, latent bug fixed); rows 1, 2, 3, 7, 8, 12 updated with fly/Dockerfile file:line refs; cleanroom build (no local ui/dist) → exit 0; smoke test mesh-llm 0.58.0; APP_MODE=console confirmed at 2026-04-08
Task 3.1 — .github/workflows/docker.yml — verified spec item 9 (no QEMU, 17 jobs total, 6 reusable precheck jobs, 3 native ubuntu-24.04-arm runners for client/cpu/vulkan arm64, cuda+rocm amd64-only, manifest merge jobs); reusable docker precheck encoded the same assertions in YAML at 2026-04-08
Task 3.2 — Final Verification — all 13 spec items verified via `.github/workflows/docker-precheck.yml`; SPEC_COMPLIANCE.md updated with reusable workflow authority at 2026-04-08

## Final Verification

All 13 spec items verified on 2026-04-08 via `.github/workflows/docker-precheck.yml`:

- Git commit: e15288b4d5d7312bf61ddfc0a952d5d9ad4f64d9
- Exit code: 0
- SKIP count: 0
- ✅ count: 52
- ❌ count: 0

Run in CI: `.github/workflows/docker.yml` calls `.github/workflows/docker-precheck.yml` as reusable pre-check jobs before Docker builds.
