# Distributed LLM Inference — build & run tasks

llama_dir := env("MESH_LLM_LLAMA_DIR", ".deps/llama.cpp")
llama_build_root := env("MESH_LLM_LLAMA_BUILD_ROOT", ".deps/llama-build")
build_dir := env("LLAMA_STAGE_BUILD_DIR", llama_build_root / "build-stage-abi-cpu")
mesh_dir := "mesh-llm"
ui_dir := mesh_dir / "ui"
benchmark_src_dir := mesh_dir / "benchmarks"
home_dir := if os_family() == "windows" { env("USERPROFILE") } else { env("HOME") }
xdg_cache_dir := env("XDG_CACHE_HOME", home_dir / ".cache")
hf_home := env("HF_HOME", xdg_cache_dir / "huggingface")
models_dir := env("HF_HUB_CACHE", hf_home / "hub")
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Build for the current platform (macOS Metal ABI, Linux/Windows auto ABI backend)
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

# Build on macOS Apple Silicon (Metal ABI)
build-mac:
    @scripts/build-mac.sh

# Build patched llama.cpp ABI and mesh-llm on Linux
build-linux backend="" cuda_arch="" rocm_arch="":
    @scripts/build-linux.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Build patched llama.cpp ABI and mesh-llm on Linux without rebuilding the UI.
[linux]
build-runtime backend="" cuda_arch="" rocm_arch="":
    @scripts/build-linux.sh --skip-ui --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Build release artifacts for the current platform.

# GitHub release builds use embedded ABI libraries.
release-build:
    @scripts/build-release.sh

# Build a Linux ARM64 CPU release artifact on a native ARM64 runner.
release-build-arm64:
    @scripts/build-release.sh

# Prepare the pinned llama.cpp checkout and apply the Mesh-LLM ABI patch queue.
llama-prepare:
    @scripts/prepare-llama.sh pinned

# Prepare llama.cpp at upstream master and apply the Mesh-LLM ABI patch queue.
llama-prepare-latest:
    @scripts/prepare-llama.sh latest

# Build the patched llama.cpp ABI static libraries.
llama-build: llama-prepare
    @scripts/build-llama.sh

release-build-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cpu

# Build a Linux CUDA release artifact (primary / R535-compatible lane).
release-build-cuda cuda_arch="75;80;86;87;89;90":
    @scripts/build-linux.sh --backend cuda --cuda-arch "{{ cuda_arch }}"

release-build-cuda-blackwell cuda_arch="75;80;86;87;89;90;100;120":
    @scripts/build-linux.sh --backend cuda --cuda-arch "{{ cuda_arch }}"

release-build-cuda-windows cuda_arch="75;80;86;87;89;90":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cuda -CudaArch "{{cuda_arch}}"

release-build-cuda-blackwell-windows cuda_arch="75;80;86;87;89;90;100;120":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cuda -CudaArch "{{cuda_arch}}"

# Build a Linux ROCm ABI release artifact with an explicit architecture list.
release-build-rocm rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @scripts/build-linux-rocm.sh "{{ rocm_arch }}"

release-build-rocm-windows rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend rocm -RocmArch "{{rocm_arch}}"

# Build a Linux Vulkan ABI release artifact.
release-build-vulkan:
    @scripts/build-linux.sh --backend vulkan

release-build-vulkan-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend vulkan

# Build the skippy benchmark/debug telemetry collector.
metrics-server-build:
    cargo build -p metrics-server

# Generate a reproducible benchmark corpus for skippy bench tooling.
bench-corpus tier="smoke" *ARGS:
    scripts/generate-bench-corpus.py "{{ tier }}" {{ ARGS }}

# Run skippy family certification checks.
family-certify *ARGS:
    scripts/family-certify.sh {{ ARGS }}

# Run target/draft speculative compatibility checks.
spec-bench target draft *ARGS:
    LLAMA_STAGE_BUILD_DIR=".deps/llama.cpp/build-stage-abi-static" cargo build -p llama-spec-bench
    LLAMA_STAGE_BUILD_DIR=".deps/llama.cpp/build-stage-abi-static" target/debug/llama-spec-bench --target-model-path "{{ target }}" --draft-model-path "{{ draft }}" {{ ARGS }}

# Smoke a standalone skippy OpenAI frontend stage.
skippy-openai-smoke *ARGS:
    scripts/skippy-openai-smoke.sh {{ ARGS }}

# Run the skippy benchmark/debug telemetry collector.
metrics-server db="/tmp/mesh-metrics.duckdb" http_addr="127.0.0.1:18080" otlp_addr="127.0.0.1:14317" *ARGS: metrics-server-build
    target/debug/metrics-server serve --db "{{ db }}" --http-addr "{{ http_addr }}" --otlp-grpc-addr "{{ otlp_addr }}" {{ ARGS }}

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

# ── QUIC Mesh ──────────────────────────────────────────────────

mesh_bin := "target/release/mesh-llm"

# Prints an invite token for other nodes to join.
mesh-worker gguf=model:
    {{ mesh_bin }} --model {{ gguf }}

# Join an existing mesh and serve through the embedded runtime.
mesh-join join="" port="9337" gguf=model split="":
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS="--model {{ gguf }} --port {{ port }}"
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
    cp {{ mesh_bin }} "$BUNDLE/"
    # Fix rpaths for portability
    for bin in "$BUNDLE/mesh-llm"; do
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

release-bundle-cuda-blackwell version output="dist":
    MESH_RELEASE_FLAVOR=cuda-blackwell scripts/package-release.sh "{{ version }}" "{{ output }}"

release-bundle-cuda-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{version}}" -OutputDir "{{output}}" -Flavor cuda

release-bundle-cuda-blackwell-windows version output="dist":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package-release.ps1 -Version "{{version}}" -OutputDir "{{output}}" -Flavor cuda-blackwell

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
    MESH_UI_API_ORIGIN="{{ api }}" npm run dev -- --host 0.0.0.0 --port {{ port }}

# Run the UI with Vite HMR proxying to the public anarchai.org API
ui-dev-public: (ui-dev "https://www.anarchai.org")

# Run UI unit tests (vitest)
ui-test:
    cd "{{ ui_dir }}" && npm test

# Start a lite client — no GPU, no model, just a local HTTP proxy to the mesh host.

# Only needs the mesh-llm binary (no llama.cpp binaries or model).
mesh-client join="" port="9337":
    {{ mesh_bin }} --client --port {{ port }} --join {{ join }}

# Build and auto-join a mesh (discover via Nostr)
auto: build
    {{ mesh_bin }} --auto

# ── Utilities ──────────────────────────────────────────────────

# Update both tracked llama.cpp pin files from the prepared checkout.
llama-update-pin:
    scripts/update-llama-pin.sh

# Render a Markdown summary for a llama.cpp upstream pin change.
llama-summary old new:
    scripts/summarize-llama-upstream.sh "{{ old }}" "{{ new }}"

# Clean UI build artifacts (node_modules, dist). Fixes stale npm state.
[unix]
clean-ui:
    cd "{{ ui_dir }}" && rm -rf node_modules dist
    echo "Cleaned UI: node_modules + dist removed"

[windows]
clean-ui:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location '{{ ui_dir }}'; Remove-Item -Recurse -Force node_modules,dist -ErrorAction SilentlyContinue"
    echo "Cleaned UI: node_modules + dist removed"
# Stop mesh-llm processes
stop:
    pkill -f "mesh-llm" 2>/dev/null || true
    echo "Stopped"

# Quick test inference (works with any running server on 8080 or 8090)
test port="9337":
    curl -s http://localhost:{{ port }}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Write a haiku about distributed computing."}],"max_tokens":50}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(d['choices'][0]['message'].get('content','')[:200]); print(f\"  prompt: {t['prompt_per_second']:.1f} tok/s  gen: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tok)\")"

# Show the local llama.cpp ABI patch queue
diff:
    ls -1 third_party/llama.cpp/patches

# Build the client-only Docker image
[unix]
docker-build-client tag="mesh-llm:client":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.client -t {{ tag }} .

[windows]
docker-build-client tag="mesh-llm:client":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.client -t '{{ tag }}' ."

# Run the client console image locally
docker-run-client tag="mesh-llm:client":
    docker run --rm -p 3131:3131 -p 9337:9337 -e APP_MODE=console {{ tag }}
