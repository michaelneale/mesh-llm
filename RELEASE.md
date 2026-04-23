# Releasing mesh-llm

## Prerequisites

- `just` installed (`brew install just`)
- `cmake` installed (`brew install cmake`)
- `cargo` installed (packaged with rust)
- `gh` CLI authenticated (`gh auth status`)
- llama.cpp fork cloned (`just build` does this automatically)
- `CARGO_REGISTRY_TOKEN` GitHub Actions secret configured if you want tagged stable releases to publish `mesh-llm-client` and `mesh-api` to crates.io

## Steps

### 1. Build everything fresh

```bash
just build
```

On macOS, this clones/updates the llama.cpp fork if needed, builds with `-DGGML_METAL=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF`, and builds the Rust mesh-llm binary. Linux release workflows build CPU, ARM64 CPU, CUDA, ROCm, and Vulkan variants separately. The Linux ARM64 CPU bundle is `mesh-llm-aarch64-unknown-linux-gnu.tar.gz`, and `arm64` and `aarch64` mean the same 64-bit ARM target in release and install docs.

On Windows, use the release-specific recipes directly:

```powershell
just release-build-windows
just release-build-cuda-windows
just release-build-rocm-windows
just release-build-vulkan-windows
```

### 2. Verify no homebrew dependencies

```bash
otool -L llama.cpp/build/bin/llama-server | grep -v /System | grep -v /usr/lib
otool -L llama.cpp/build/bin/rpc-server | grep -v /System | grep -v /usr/lib
otool -L target/release/mesh-llm | grep -v /System | grep -v /usr/lib
```

Each should only show the binary name — no `/opt/homebrew/` paths.

### 3. Create the bundle

```bash
just bundle
```

Creates `/tmp/mesh-bundle.tar.gz` containing `mesh-llm`, flavor-specific llama.cpp runtime binaries, `llama-moe-analyze` for MoE ranking generation, and `llama-moe-split` for MoE shard generation.

Bundle naming now follows the same convention everywhere:

- macOS bundles package `rpc-server-metal` and `llama-server-metal`
- generic Linux bundles package `rpc-server-cpu` and `llama-server-cpu`
- CUDA Linux bundles package `rpc-server-cuda` and `llama-server-cuda`
- ROCm Linux bundles package `rpc-server-rocm` and `llama-server-rocm`
- Vulkan Linux bundles package `rpc-server-vulkan` and `llama-server-vulkan`

On Windows, create release archives directly:

```powershell
just release-bundle-windows v0.X.0
just release-bundle-cuda-windows v0.X.0
just release-bundle-rocm-windows v0.X.0
just release-bundle-vulkan-windows v0.X.0
```

Those commands emit `.zip` assets in `dist/` with `mesh-llm.exe`, plus flavor-specific `rpc-server-<flavor>.exe` and `llama-server-<flavor>.exe`.
If optional Windows benchmark binaries such as `membench-fingerprint-cuda.exe` or `membench-fingerprint-hip.exe` are present in `mesh-llm/target/release/`, the PowerShell packager also includes them in the `.zip`.

### 4. Smoke test the bundle

```bash
mkdir /tmp/test-bundle && tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
# Should download model, start solo, API on :9337, console on :3131
# Hit http://localhost:9337/v1/chat/completions to verify inference works
# Ctrl+C to stop
rm -rf /tmp/test-bundle
```

### 5. Release

```bash
gh workflow run release.yml -f version=v0.X.0 -f prerelease=false -f target_branch=main
```

The Release workflow is now the source of truth for stable releases. It checks out `main`, runs the release consistency checks, bumps the version in source + Cargo manifests, refreshes `Cargo.lock` without upgrading dependencies, creates the release commit directly on `main`, creates and pushes the release tag, builds the release artifacts, and publishes the GitHub release.

On native Windows, `just check-release` still runs the Rust/docs/workflow invariant checks, but it skips the Bash-only `install.sh` and `scripts/package-release.sh` parity checks. Run the release-target parity check on macOS or Linux before cutting a tag if you need full shell-script coverage.

### 5a. Prerelease

```bash
gh workflow run release.yml -f version=v0.X.0-rc.1 -f prerelease=true -f target_branch=feature/your-branch
```

The same Release workflow handles prereleases. Set `prerelease=true` and provide the branch you want to cut the prerelease from. The workflow creates the prerelease commit directly on that branch, pushes the branch update, creates and pushes the prerelease tag, builds the artifacts, and publishes a GitHub prerelease.

If you want a faster prerelease cut without the Linux CUDA, ROCm, and Vulkan bundles, add:

```bash
gh workflow run release.yml -f version=v0.X.0-rc.1 -f prerelease=true -f skip_gpu_bundles=true -f target_branch=feature/your-branch
```

That flag is prerelease-only. Stable releases must continue to publish the full Linux CPU, ARM64 CPU, CUDA, ROCm, and Vulkan set.

### 6. Let GitHub Actions build and publish the release

Running `.github/workflows/release.yml` via `workflow_dispatch` triggers the release flow, which:

- builds the Swift XCFramework zip on macOS before the tag exists so SwiftPM gets the exact release URL and checksum baked into the tagged `Package.swift`
- creates and pushes the release commit and tag before any build jobs start
- serializes releases so two manual runs cannot race each other
- builds release bundles on macOS, Linux CPU, and Linux ARM64 CPU
- also builds Linux CUDA (primary lane, CUDA 12.6.3 toolkit, R535-compatible), Linux CUDA Blackwell (CUDA 12.8 toolkit, R550+ required), Linux ROCm, and Linux Vulkan unless `skip_gpu_bundles=true` is set on a prerelease run
- keeps the Windows publish block commented out for now, so GitHub release publishing does not currently upload Windows bundles
- still leaves the local Windows bundle recipes available in `Justfile` for manual builds
- uploads `MeshLLMFFI.xcframework.zip` for Swift Package Manager consumers
- publishes the Android AAR to GitHub Packages as `ai.meshllm:meshllm-android:<version>`
- uploads versioned assets such as `mesh-llm-v0.X.0-aarch64-apple-darwin.tar.gz`
- uploads the Linux ARM64 CPU asset as `mesh-llm-aarch64-unknown-linux-gnu.tar.gz`
- uploads stable `latest` assets such as `mesh-llm-x86_64-unknown-linux-gnu.tar.gz`
- uploads CUDA-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz` (primary lane) and `mesh-llm-x86_64-unknown-linux-gnu-cuda-blackwell.tar.gz` (Blackwell lane — see `docs/cuda-release-lanes.md`)
- uploads ROCm-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz`
- uploads Vulkan-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz`
- keeps the legacy macOS `mesh-bundle.tar.gz` asset available for direct archive installs
- creates the GitHub release automatically with generated notes
- marks hyphenated tags such as `v0.X.0-rc.1` as GitHub prereleases
- publishes `mesh-llm-client` and `mesh-api` to crates.io after the release succeeds, including prerelease tags such as `v0.X.0-rc.1`
- resets the target branch back to the placeholder Swift `Package.swift` after the release finishes, so day-to-day branch builds do not keep pointing at the most recent published XCFramework

### 6a. Autoupdater behavior and compatibility

- Stable releases still use GitHub's `releases/latest` endpoint, so ordinary installs only see stable releases.
- GitHub prereleases are excluded from `releases/latest`, so publishing `v0.X.0-rc.1` does not advertise that prerelease to older stable clients.
- This change updates mesh-llm's version comparison to proper semver ordering, so a prerelease binary such as `0.X.0-rc.1` will correctly upgrade to the eventual stable `0.X.0` release, or to a specific tagged release when you run `mesh-llm update --version vX.Y.Z`.
- Older binaries that predate this change use a dot-splitting numeric comparison instead of semver. If one of those binaries somehow carries a prerelease version string such as `0.X.0-rc.1`, it can mis-order versions and may fail to recognize `0.X.0` or `0.X.1` as newer. In practice that only affects manually produced prerelease builds, because the old release tooling did not support `-rc.N` tags.
- Result: the change is backward compatible for existing stable users, and it fixes updater behavior for official prerelease builds going forward.

### 7. Verify the release assets

After the workflow finishes, verify:

- `MeshLLMFFI.xcframework.zip` exists for Swift Package Manager installs
- `ai.meshllm:meshllm-android:<version>` is visible in the GitHub Packages Maven registry for the repo
- `mesh-bundle.tar.gz` still exists for direct macOS archive installs
- `mesh-llm-aarch64-apple-darwin.tar.gz` exists
- `mesh-llm-aarch64-unknown-linux-gnu.tar.gz` exists
- `mesh-llm-x86_64-unknown-linux-gnu.tar.gz` exists
- `mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz` exists (primary R535-compatible lane) unless this was a prerelease with `skip_gpu_bundles=true`
- `mesh-llm-x86_64-unknown-linux-gnu-cuda-blackwell.tar.gz` exists (Blackwell lane, R550+ only) unless this was a prerelease with `skip_gpu_bundles=true`
- `mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz` exists unless this was a prerelease with `skip_gpu_bundles=true`
- `mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz` exists unless this was a prerelease with `skip_gpu_bundles=true`
- Windows release bundles are not expected from the current GitHub Actions workflow while the publish block stays commented out

## Notes

- The unversioned asset name `mesh-bundle.tar.gz` is still kept for compatibility with direct archive installs.
- The default Linux release bundle is a generic CPU build.
- Windows source builds exist, and the `*-windows` release recipes in `Justfile` still generate local `.zip` artifacts.
- The workflow is now responsible for creating and pushing release tags; pushing a tag manually does not trigger a release build anymore.
- The workflow mutates the target branch by pushing the release commit before it starts the build matrix, then pushes a follow-up commit that restores the placeholder Swift package manifest after a successful release.
- Tagged GitHub releases do not currently publish Windows bundles because the Windows release job remains commented out in `.github/workflows/release.yml`.
- Android Maven publication currently targets GitHub Packages, not Maven Central.
- Release bundles use flavor-specific `rpc-server-<flavor>` and `llama-server-<flavor>` names so multiple flavors can coexist in one install directory. Use `mesh-llm --llama-flavor <flavor>` to force a specific pair.
- Prereleases can optionally skip the Linux CUDA, ROCm, and Vulkan bundles via the `skip_gpu_bundles=true` workflow input. Those tags will not be installable or updatable on CUDA/ROCm/Vulkan bundle installs until a later prerelease or stable release publishes matching assets.
- The CUDA Linux release bundle is built in CI with an explicit multi-arch `CMAKE_CUDA_ARCHITECTURES` list and is not runtime-tested during the workflow.
- The ROCm and Vulkan Linux release bundles are compile-tested in CI, but not runtime-tested against real GPUs during the workflow.
- `codesign` and `xattr` may be needed on the receiving machine if macOS Gatekeeper blocks unsigned binaries:
  ```bash
  codesign -s - /usr/local/bin/mesh-llm /usr/local/bin/rpc-server-metal /usr/local/bin/llama-server-metal /usr/local/bin/llama-moe-analyze /usr/local/bin/llama-moe-split
  xattr -cr /usr/local/bin/mesh-llm /usr/local/bin/rpc-server-metal /usr/local/bin/llama-server-metal /usr/local/bin/llama-moe-analyze /usr/local/bin/llama-moe-split
  ```
