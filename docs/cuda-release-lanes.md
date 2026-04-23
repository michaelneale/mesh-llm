# CUDA release lanes

mesh-llm publishes **two CUDA release bundles per tagged release**. They
share source, features, and upstream llama.cpp pin, and differ only in
the CUDA toolkit version and GPU architecture coverage. Pick the one
that matches your NVIDIA driver.

## Why two lanes

nvcc from the **CUDA 12.8** toolkit emits cubins whose minor-version
metadata the **R535-series driver** (native CUDA 12.2) rejects at kernel
load time. This manifests as `CUDA error: device kernel image is
invalid` on the first matmul when running against sm_80 (A30/A100), even
though sm_80 cubins are physically present in the binary
(`cuobjdump --list-elf` confirms). Rebuilding the identical source tree
on the **CUDA 12.6.3** toolkit produces a working bundle on the same
hardware/driver.

At the same time, Blackwell compute capabilities (sm_100 B100/B200,
sm_120 RTX 50-series) were **first introduced in CUDA 12.8**; nvcc 12.6
cannot emit them at all. (sm_103 is a related Blackwell variant, but
nvcc 12.8.0 does not know it — that arch landed in a later CUDA
release; it's therefore omitted from our 12.8-toolkit Blackwell lane.)

There is no single toolkit that satisfies both audiences, so the release
workflow builds both.

## Lane summary

| Asset suffix | Toolkit | Arch coverage | Driver requirement |
|---|---|---|---|
| `-cuda` (primary) | CUDA 12.6.3 | sm_75, sm_80, sm_86, sm_87, sm_89, sm_90 (Turing → Hopper) | R535+ (CUDA 12.2 native) |
| `-cuda-blackwell` | CUDA 12.8 | sm_75..sm_90 plus sm_100, sm_120 (adds Blackwell) | R550+ (CUDA 12.4 native) |

- **Primary `-cuda`** covers the currently-deployed A30/A100/Ada/Hopper
  fleet on the stable R535 driver series. This is the recommended
  default.
- **`-cuda-blackwell`** is required only if you have Blackwell hardware
  (B100, B200, Thor, RTX 50-series). It will NOT load on R535 drivers
  even on older sm_80 hardware because the R535 driver rejects any
  12.8-minor-tagged cubin.

## How to pick one

- On any R535-series driver (default for Ampere-era HGX and most
  non-freshly-imaged datacenter hosts): use `-cuda`.
- On R550+ drivers running Blackwell hardware: use `-cuda-blackwell`.
- On R550+ drivers running only pre-Blackwell hardware (A30/A100/L40/
  H100 etc.): either bundle will work; prefer `-cuda` to avoid pulling
  in arch cubins you will not execute.

Check your driver:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

## Asset naming

The outer archive filename distinguishes the lanes:

- `mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz`
- `mesh-llm-x86_64-unknown-linux-gnu-cuda-blackwell.tar.gz`

**Inner binary filenames are identical across both bundles**
(`llama-server-cuda`, `rpc-server-cuda`, etc.) because the mesh-llm
runtime's `BinaryFlavor` enum treats both as the same CUDA flavor — only
the cubin contents differ. You can swap one bundle for the other without
changing anything in the mesh-llm invocation; `--llama-flavor cuda` is
correct for both.

## Installer behavior

`install.sh` exposes both as flavor strings:

```bash
# primary (default NVIDIA recommendation)
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash

# explicit Blackwell
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh \
  | MESH_LLM_INSTALL_FLAVOR=cuda-blackwell bash
```

The auto-detection path (`recommended_flavor`) does NOT pick
`cuda-blackwell` on its own; the primary lane is always the safe
recommendation. Users on Blackwell must opt in explicitly until a
future change teaches the installer to probe
`nvidia-smi --query-gpu=compute_cap`.

## Building locally

Both lanes are exposed in the `Justfile`:

```bash
# primary (CUDA 12.6.3 toolkit required on the host / container)
just release-build-cuda
just release-bundle-cuda "$VERSION"

# Blackwell (CUDA 12.8 toolkit required on the host / container)
just release-build-cuda-blackwell
just release-bundle-cuda-blackwell "$VERSION"
```

You can override arches with a positional argument, e.g.:
`just release-build-cuda "80;89"`.

## CI wiring

- `.github/workflows/release.yml` defines two sibling jobs:
  `build_linux_cuda` (12.6.3 container) and `build_linux_cuda_blackwell`
  (12.8 container). Both upload to the same GitHub release via the
  `publish` job's `needs:` list.
- The default toolkit versions are configurable at the repo level via
  Actions variables `vars.CUDA_VERSION` (primary, default `12.6.3`) and
  `vars.CUDA_BLACKWELL_VERSION` (Blackwell, default `12.8.0`).
- `.github/workflows/llama-cache-keys.yml` emits
  `cuda_fat_cache_key` for the primary lane and
  `cuda_blackwell_fat_cache_key` for the Blackwell lane so the warm
  caches do not collide.

## Docker image

The `mesh-llm:cuda` Docker image (built by
`.github/workflows/docker.yml`, `docker/Dockerfile.cuda`) currently
tracks the **primary lane only** (CUDA 12.6.3, sm_75..sm_90). A
`mesh-llm:cuda-blackwell` tag is a planned follow-up; for now, Blackwell
Docker users should build locally with
`just docker-build-cuda mesh-llm:cuda-blackwell "75;80;86;87;89;90;100;120" 12.8.0`.

## History

The split was introduced in [PR #309](https://github.com/Mesh-LLM/mesh-llm/pull/309)
after a reproducible A30 crash on R535 drivers was traced to the nvcc
12.8 / R535 cubin incompatibility. See issue
[#304](https://github.com/Mesh-LLM/mesh-llm/issues/304) for the
investigation, A/B build matrix, and 29-GPU test results.
