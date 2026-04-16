# Distributed LLM Inference — build & run tasks

llama_dir := "llama.cpp"
build_dir := llama_dir / "build"
mesh_dir := "mesh-llm"
ui_dir := mesh_dir / "ui"
benchmark_src_dir := mesh_dir / "benchmarks"
home_dir := if os_family() == "windows" { env("USERPROFILE") } else { env("HOME") }
xdg_cache_dir := env("XDG_CACHE_HOME", home_dir / ".cache")
hf_home := env("HF_HOME", xdg_cache_dir / "huggingface")
models_dir := env("HF_HUB_CACHE", hf_home / "hub")
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Build for the current platform (macOS→Metal, Linux/Windows→auto backend)
[macos]
build: build-mac

# Linux overrides:
#   just build backend=cpu
#   just build backend=cuda cuda_arch='120;86'
#   just build backend=rocm rocm_arch='gfx942;gfx90a'
#   just build backend=vulkan
[linux]
build backend="" cuda_arch="" rocm_arch="":
    @scripts/build-linux.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Windows overrides:
#   just build backend=cpu
#   just build backend=cuda cuda_arch='120;86'
#   just build backend=rocm rocm_arch='gfx942;gfx90a'
#   just build backend=vulkan
[windows]
build backend="" cuda_arch="" rocm_arch="":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend "{{backend}}" -CudaArch "{{cuda_arch}}" -RocmArch "{{rocm_arch}}"

# Build on macOS Apple Silicon (Metal + RPC)
build-mac:
    @scripts/build-mac.sh

# Build on Linux with CUDA, ROCm, or Vulkan — delegates to scripts/build-linux.sh
build-linux backend="" cuda_arch="" rocm_arch="":
    @scripts/build-linux.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Build release artifacts for the current platform.

# GitHub release builds use CPU backends on Linux and Windows, and Metal on macOS.
release-build:
    @scripts/build-release.sh

# Build a Linux ARM64 CPU release artifact on a native ARM64 runner.
release-build-arm64:
    @scripts/build-release.sh

release-build-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cpu

# Build a Linux CUDA release artifact with an explicit architecture list.
release-build-cuda cuda_arch="75;80;86;87;89;90;100;103;120":
    @scripts/build-linux.sh --backend cuda --cuda-arch "{{ cuda_arch }}"

release-build-cuda-windows cuda_arch="75;80;86;87;89;90;100;103;120":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cuda -CudaArch "{{cuda_arch}}"

# Build a Linux ROCm release artifact with an explicit architecture list.
release-build-rocm rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @scripts/build-linux-rocm.sh "{{ rocm_arch }}"

release-build-rocm-windows rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend rocm -RocmArch "{{rocm_arch}}"

# Build a Linux Vulkan release artifact.
release-build-vulkan:
    @scripts/build-linux.sh --backend vulkan

release-build-vulkan-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend vulkan

# Bump release version consistently across source and Cargo manifests.
release-version version:
    @scripts/release-version.sh "{{ version }}"

# Tag and push a release. Bumps version, updates Cargo.lock, commits, tags, pushes.
# CI builds and publishes the GitHub release automatically.
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    current_branch="$(git branch --show-current)"
    if [[ "$current_branch" != "main" ]]; then
        echo "Error: release must be run from the 'main' branch (current: ${current_branch:-detached HEAD})" >&2
        exit 1
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Error: working tree is not clean. Commit or stash changes before releasing." >&2
        exit 1
    fi
    just check-release
    tag="{{ version }}"
    if [[ "$tag" != v* ]]; then
        tag="v$tag"
    fi
    scripts/release-version.sh "$tag"
    git add -A
    git commit -m "$tag: release"
    git tag "$tag"
    git push origin main
    git push origin "$tag"

# Tag and push a prerelease from the current branch. Bumps version, updates
# Cargo.lock, commits, tags, and pushes the branch plus prerelease tag.
prerelease version:
    #!/usr/bin/env bash
    set -euo pipefail
    current_branch="$(git branch --show-current)"
    if [[ -z "$current_branch" ]]; then
        echo "Error: prerelease must be run from a branch, not detached HEAD." >&2
        exit 1
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Error: working tree is not clean. Commit or stash changes before prereleasing." >&2
        exit 1
    fi
    tag="{{ version }}"
    if [[ "$tag" != v* ]]; then
        tag="v$tag"
    fi
    if [[ "$tag" != *-* ]]; then
        echo "Error: prerelease tag must include a prerelease suffix such as -rc.1 (got: $tag)" >&2
        exit 1
    fi
    scripts/release-version.sh "$tag"
    git add -A
    git commit -m "$tag: prerelease"
    git tag "$tag"
    git push origin "$current_branch"
    git push origin "$tag"

# Download the default model (GLM-4.7-Flash Q4_K_M, 17GB)
download-model:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{ models_dir }}"
    if [ -f "{{ model }}" ]; then
        echo "Model already exists: {{ model }}"
    else
        echo "Downloading GLM-4.7-Flash Q4_K_M (~17GB)..."
        curl -L -o "{{ model }}" \
            "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
    fi

# ── Raw TCP (no mesh) ──────────────────────────────────────────

# Start rpc-server (worker) with local GGUF loading
worker host="0.0.0.0" port="50052" device="" gguf=model:
    #!/usr/bin/env bash
    set -euo pipefail
    DEVICE="{{ device }}"
    if [ -z "$DEVICE" ]; then
        DEVICE="$(scripts/detect-llama-device.sh "{{ build_dir }}/bin/rpc-server")"
    fi
    exec {{ build_dir }}/bin/rpc-server --host {{ host }} --port {{ port }} -d "$DEVICE" --gguf {{ gguf }}

# Start llama-server (orchestrator) pointing at an RPC worker
serve rpc="127.0.0.1:50052" port="8080" gguf=model:
    {{ build_dir }}/bin/llama-server \
        --model {{ gguf }} \
        --rpc {{ rpc }} \
        -ngl 99 -fit off \
        --port {{ port }}

# Start both worker + server on localhost for testing
local: build download-model
    #!/usr/bin/env bash
    set -euo pipefail
    DEVICE="$(scripts/detect-llama-device.sh "{{ build_dir }}/bin/rpc-server")"
    echo "Starting rpc-server (worker)..."
    {{ build_dir }}/bin/rpc-server --host 127.0.0.1 --port 50052 -d "$DEVICE" --gguf {{ model }} &
    WORKER_PID=$!
    sleep 3
    echo "Starting llama-server (orchestrator)..."
    {{ build_dir }}/bin/llama-server \
        --model {{ model }} \
        --rpc 127.0.0.1:50052 \
        -ngl 99 -fit off \
        --port 8080 &
    SERVER_PID=$!
    echo "Waiting for server..."
    for i in $(seq 1 120); do
        curl -s http://localhost:8080/health 2>/dev/null | grep -q '"ok"' && break
        sleep 1
    done
    echo "Ready: http://localhost:8080"
    echo "Worker PID: $WORKER_PID  Server PID: $SERVER_PID"
    echo "Press Ctrl+C to stop"
    wait

# ── QUIC Mesh ──────────────────────────────────────────────────

mesh_bin := "target/release/mesh-llm"

# Start a mesh worker (no llama-server, just rpc-server + mesh)

# Prints an invite token for other nodes to join.
mesh-worker gguf=model:
    {{ mesh_bin }} --model {{ gguf }} --bin-dir {{ build_dir }}/bin

# Join an existing mesh. Auto-elects host, starts llama-server or contributes as worker.
mesh-join join="" port="9337" gguf=model split="":
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS="--model {{ gguf }} --bin-dir {{ build_dir }}/bin --port {{ port }}"
    if [ -n "{{ join }}" ]; then
        ARGS="$ARGS --join {{ join }}"
    fi
    if [ -n "{{ split }}" ]; then
        ARGS="$ARGS --tensor-split {{ split }}"
    fi
    exec {{ mesh_bin }} $ARGS

# Create a portable tarball with all binaries for deployment to another machine
bundle output="/tmp/mesh-bundle.tar.gz":
    #!/usr/bin/env bash
    set -euo pipefail
    DIR=$(mktemp -d)
    BUNDLE="$DIR/mesh-bundle"
    mkdir -p "$BUNDLE"
    case "$(uname -s)" in
        Darwin) LLAMA_FLAVOR="metal" ;;
        Linux) LLAMA_FLAVOR="cpu" ;;
        *) LLAMA_FLAVOR="" ;;
    esac
    rpc_name="rpc-server"
    llama_name="llama-server"
    if [ -n "$LLAMA_FLAVOR" ]; then
        rpc_name="rpc-server-$LLAMA_FLAVOR"
        llama_name="llama-server-$LLAMA_FLAVOR"
    fi
    cp {{ mesh_bin }} "$BUNDLE/"
    cp {{ build_dir }}/bin/rpc-server "$BUNDLE/$rpc_name"
    cp {{ build_dir }}/bin/llama-server "$BUNDLE/$llama_name"
    cp {{ build_dir }}/bin/llama-moe-analyze "$BUNDLE/"
    cp {{ build_dir }}/bin/llama-moe-split "$BUNDLE/"
    for lib in {{ build_dir }}/bin/*.dylib; do
        cp "$lib" "$BUNDLE/" 2>/dev/null || true
    done
    # Fix rpaths for portability
    for bin in "$BUNDLE/mesh-llm" "$BUNDLE/$rpc_name" "$BUNDLE/$llama_name" "$BUNDLE/llama-moe-analyze" "$BUNDLE/llama-moe-split"; do
        [ -f "$bin" ] || continue
        install_name_tool -add_rpath @executable_path/ "$bin" 2>/dev/null || true
    done
    # Include Apple Silicon benchmark binary if built
    BENCH="target/release/membench-fingerprint"
    if [ -f "$BENCH" ]; then
        cp "$BENCH" "$BUNDLE/"
        echo "Included: membench-fingerprint"
    else
        echo "Note: membench-fingerprint not found — run 'just benchmark-build-apple' to include it"
    fi
    tar czf {{ output }} -C "$DIR" mesh-bundle/
    rm -rf "$DIR"
    echo "Bundle: {{ output }} ($(du -sh {{ output }} | cut -f1))"

# Create release archive(s) for the current platform.

# `version` should be a tag like v0.30.0.
release-bundle version output="dist":
    @scripts/package-release.sh "{{ version }}" "{{ output }}"

# Create a Linux ARM64 CPU release archive on a native ARM64 runner.
release-bundle-arm64 version output="dist":
    @scripts/package-release.sh "{{ version }}" "{{ output }}"

# Run repo-level release-target consistency checks.
[unix]
check-release:
    cargo run -p xtask -- repo-consistency release-targets

[windows]
check-release:
    cargo run -p xtask -- repo-consistency release-targets

release-bundle-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{version}}" -OutputDir "{{output}}"

# Create Linux CUDA release archive(s).
release-bundle-cuda version output="dist":
    MESH_RELEASE_FLAVOR=cuda scripts/package-release.sh "{{ version }}" "{{ output }}"

release-bundle-cuda-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{version}}" -OutputDir "{{output}}" -Flavor cuda

# Create Linux ROCm release archive(s).
release-bundle-rocm version output="dist":
    MESH_RELEASE_FLAVOR=rocm scripts/package-release.sh "{{ version }}" "{{ output }}"

release-bundle-rocm-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{version}}" -OutputDir "{{output}}" -Flavor rocm

# Create Linux Vulkan release archive(s).
release-bundle-vulkan version output="dist":
    MESH_RELEASE_FLAVOR=vulkan scripts/package-release.sh "{{ version }}" "{{ output }}"

release-bundle-vulkan-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{version}}" -OutputDir "{{output}}" -Flavor vulkan

# ── Benchmark Binaries ────────────────────────────────────────────────────────

# Build Apple Silicon memory bandwidth benchmark (macOS only)
[macos]
benchmark-build-apple:
    swiftc -O {{ benchmark_src_dir }}/membench-fingerprint.swift -o target/release/membench-fingerprint
    echo "Built: target/release/membench-fingerprint"

# Build NVIDIA CUDA memory bandwidth benchmark (requires CUDA toolkit)
benchmark-build-cuda:
    nvcc -O3 -o target/release/membench-fingerprint-cuda {{ benchmark_src_dir }}/membench-fingerprint.cu
    echo "Built: target/release/membench-fingerprint-cuda"

[windows]
benchmark-build-cuda-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "nvcc -O3 -o 'target/release/membench-fingerprint-cuda.exe' '{{ benchmark_src_dir }}/membench-fingerprint.cu'; if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }; Write-Host 'Built: target/release/membench-fingerprint-cuda.exe'"

# Build AMD ROCm/HIP memory bandwidth benchmark (requires ROCm)
benchmark-build-hip:
    hipcc -O3 -std=c++17 -o target/release/membench-fingerprint-hip {{ benchmark_src_dir }}/membench-fingerprint.hip
    echo "Built: target/release/membench-fingerprint-hip"

[windows]
benchmark-build-hip-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "hipcc -O3 -std=c++17 -o 'target/release/membench-fingerprint-hip.exe' '{{ benchmark_src_dir }}/membench-fingerprint.hip'; if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }; Write-Host 'Built: target/release/membench-fingerprint-hip.exe'"

# Build Intel Arc SYCL memory bandwidth benchmark (requires Intel oneAPI) — UNVALIDATED
benchmark-build-intel:
    @echo "WARNING: Intel Arc benchmark is unvalidated — no Intel Arc hardware has been tested"
    icpx -O3 -fsycl -o target/release/membench-fingerprint-intel {{ benchmark_src_dir }}/membench-fingerprint-intel.cpp
    echo "Built: target/release/membench-fingerprint-intel"

[windows]
benchmark-build-intel-windows:
    @echo "WARNING: Intel Arc benchmark is unvalidated — no Intel Arc hardware has been tested"
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "icpx -O3 -fsycl -o 'target/release/membench-fingerprint-intel.exe' '{{ benchmark_src_dir }}/membench-fingerprint-intel.cpp'; if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }; Write-Host 'Built: target/release/membench-fingerprint-intel.exe'"

# Run the UI with Vite HMR and proxy /api to mesh-llm (default: http://127.0.0.1:3131)
ui-dev api="http://127.0.0.1:3131" port="5173":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ ui_dir }}"
    MESH_UI_API_ORIGIN="{{ api }}" npm run dev -- --host 127.0.0.1 --port {{ port }}

# Run the UI with Vite HMR proxying to the public anarchai.org API
ui-dev-public: (ui-dev "https://www.anarchai.org")

# Start a lite client — no GPU, no model, just a local HTTP proxy to the mesh host.

# Only needs the mesh-llm binary (no llama.cpp binaries or model).
mesh-client join="" port="9337":
    {{ mesh_bin }} --client --port {{ port }} --join {{ join }}

# Build and auto-join a mesh (discover via Nostr)
auto: build
    {{ mesh_bin }} --auto --bin-dir {{ build_dir }}/bin

# ── Utilities ──────────────────────────────────────────────────

# Clean UI build artifacts (node_modules, dist). Fixes stale npm state.
[unix]
clean-ui:
    cd "{{ ui_dir }}" && rm -rf node_modules dist
    echo "Cleaned UI: node_modules + dist removed"

[windows]
clean-ui:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location '{{ ui_dir }}'; Remove-Item -Recurse -Force node_modules,dist -ErrorAction SilentlyContinue"
    echo "Cleaned UI: node_modules + dist removed"
# Stop all running servers
stop:
    pkill -f "mesh-llm" 2>/dev/null || true
    pkill -f "rpc-server" 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
    echo "Stopped"

# Quick test inference (works with any running server on 8080 or 8090)
test port="9337":
    curl -s http://localhost:{{ port }}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Write a haiku about distributed computing."}],"max_tokens":50}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(d['choices'][0]['message'].get('content','')[:200]); print(f\"  prompt: {t['prompt_per_second']:.1f} tok/s  gen: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tok)\")"

# Optional SDK compatibility smoke: 2 mesh nodes + 1 lite client.
compat-smoke model mmproj="":
    scripts/ci-compat-smoke.sh "target/release/mesh-llm" "llama.cpp/build/bin" "{{ model }}" "{{ mmproj }}"

# Direct splitter smoke for the MoE families we actively use.
moe-split-smoke families="all":
    scripts/moe-split-smoke.sh "llama.cpp/build/bin" {{ families }}

# Validate an already-running MoE deployment end-to-end through one API/console pair.
moe-live-smoke model api_url console_url expected_nodes="2" timeout="120":
    scripts/moe-live-smoke.sh --expected-nodes {{ expected_nodes }} --timeout {{ timeout }} "{{ model }}" "{{ api_url }}" "{{ console_url }}"

# Benchmark sticky-only vs prefix-only affinity on a 3-node local mesh.
bench-prefix-affinity:
    @scripts/benchmark-prefix-affinity.sh

# Show our custom commits on top of upstream llama.cpp
diff:
    cd {{ llama_dir }} && git log --oneline --ancestry-path $(git merge-base HEAD upstream/master 2>/dev/null || echo HEAD~8)..HEAD

# Build the client-only Docker image (no GPU, no llama.cpp)
[unix]
docker-build-client tag="mesh-llm:client":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.client -t {{ tag }} .

[windows]
docker-build-client tag="mesh-llm:client":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.client -t '{{ tag }}' ."

# Build the CPU full-node Docker image
[unix]
docker-build-cpu tag="mesh-llm:cpu":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cpu -t {{ tag }} .

[windows]
docker-build-cpu tag="mesh-llm:cpu":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.cpu -t '{{ tag }}' ."

# Build the CUDA full-node Docker image
[unix]
docker-build-cuda tag="mesh-llm:cuda" cuda_arch="75;80;86;87;89;90;100;103;120":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cuda \
        --build-arg CUDA_ARCH="{{ cuda_arch }}" \
        -t {{ tag }} .

[windows]
docker-build-cuda tag="mesh-llm:cuda" cuda_arch="75;80;86;87;89;90;100;103;120":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.cuda --build-arg CUDA_ARCH='{{ cuda_arch }}' -t '{{ tag }}' ."

# Build the ROCm full-node Docker image
[unix]
docker-build-rocm tag="mesh-llm:rocm" rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm \
        --build-arg ROCM_ARCH="{{ rocm_arch }}" \
        -t {{ tag }} .

[windows]
docker-build-rocm tag="mesh-llm:rocm" rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.rocm --build-arg ROCM_ARCH='{{ rocm_arch }}' -t '{{ tag }}' ."

# Build the Vulkan full-node Docker image
[unix]
docker-build-vulkan tag="mesh-llm:vulkan":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.vulkan -t {{ tag }} .

[windows]
docker-build-vulkan tag="mesh-llm:vulkan":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.vulkan -t '{{ tag }}' ."

# Run the client console image locally
docker-run-client tag="mesh-llm:client":
    docker run --rm -p 3131:3131 -p 9337:9337 -e APP_MODE=console {{ tag }}

# Run a CPU worker node locally (requires model volume mount)
docker-run-cpu models=(home_dir / ".models") tag="mesh-llm:cpu":
    docker run --rm -p 9337:9337 \
        -v {{ models }}:/root/.models \
        -e APP_MODE=worker {{ tag }}
