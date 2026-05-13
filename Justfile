# Distributed LLM Inference — build & run tasks

llama_dir := env("MESH_LLM_LLAMA_DIR", ".deps/llama.cpp")
llama_build_root := env("MESH_LLM_LLAMA_BUILD_ROOT", ".deps/llama-build")
mesh_dir := "crates/mesh-llm"
ui_dir := "crates/mesh-llm-ui"
ui_legacy_dir := "crates/mesh-llm/ui-legacy"
home_dir := if os_family() == "windows" { env("USERPROFILE") } else { env("HOME") }
xdg_cache_dir := env("XDG_CACHE_HOME", home_dir / ".cache")
hf_home := env("HF_HOME", xdg_cache_dir / "huggingface")
models_dir := env("HF_HUB_CACHE", hf_home / "hub")
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Build for the current platform.
default: build

[private]
[unix]
_lld-cargo-config:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .cargo
    config=".cargo/config.toml"
    tmp="$(mktemp)"
    if [[ -f "$config" ]]; then
        awk '
            /^# BEGIN Mesh-LLM lld config$/ { skip = 1; next }
            /^# END Mesh-LLM lld config$/ { skip = 0; next }
            !skip { print }
        ' "$config" > "$tmp"
    else
        : > "$tmp"
    fi
    if [[ -s "$tmp" && "$(tail -c 1 "$tmp")" != "" ]]; then
        printf '\n' >> "$tmp"
    fi
    case "$(uname -s)" in
        Linux)
            if ! command -v ld.lld >/dev/null 2>&1; then
                cat >&2 <<'EOF'
    Error: LLVM ld.lld was not found.

    lld is required for faster Rust builds (measured up to 26% faster locally).

    Install lld, then rerun the just command. Common Linux packages:
      Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y lld
      Fedora:        sudo dnf install lld
      Arch Linux:    sudo pacman -S lld
      openSUSE:      sudo zypper install lld

    The build requires ld.lld to be available on PATH.
    EOF
                exit 1
            fi
            cat >> "$tmp" <<'EOF'
    # BEGIN Mesh-LLM lld config
    [target.x86_64-unknown-linux-gnu]
    rustflags = ["-C", "link-arg=-fuse-ld=lld"]

    [target.aarch64-unknown-linux-gnu]
    rustflags = ["-C", "link-arg=-fuse-ld=lld"]
    # END Mesh-LLM lld config
    EOF
            ;;
        Darwin)
            lld=""
            if command -v ld64.lld >/dev/null 2>&1; then
                lld="$(command -v ld64.lld)"
            elif command -v brew >/dev/null 2>&1; then
                lld_prefix="$(brew --prefix lld 2>/dev/null || true)"
                if [[ -n "$lld_prefix" && -x "$lld_prefix/bin/ld64.lld" ]]; then
                    lld="$lld_prefix/bin/ld64.lld"
                fi
            fi
            if [[ -z "$lld" ]]; then
                for candidate in /opt/homebrew/opt/lld/bin/ld64.lld /usr/local/opt/lld/bin/ld64.lld; do
                    if [[ -x "$candidate" ]]; then
                        lld="$candidate"
                        break
                    fi
                done
            fi
            if [[ -z "$lld" ]]; then
                cat >&2 <<'EOF'
    Error: LLVM ld64.lld was not found.

    lld is required for faster Rust builds (measured up to 26% faster locally).

    Install lld, then rerun the just command:
      brew install lld

    If Homebrew installed lld but it is not on PATH, Mesh-LLM also checks:
      $(brew --prefix lld)/bin/ld64.lld
      /opt/homebrew/opt/lld/bin/ld64.lld
      /usr/local/opt/lld/bin/ld64.lld
    EOF
                exit 1
            fi
            cat >> "$tmp" <<EOF
    # BEGIN Mesh-LLM lld config
    [target.aarch64-apple-darwin]
    rustflags = ["-C", "link-arg=-fuse-ld=$lld"]

    [target.x86_64-apple-darwin]
    rustflags = ["-C", "link-arg=-fuse-ld=$lld"]
    # END Mesh-LLM lld config
    EOF
            ;;
        *)
            echo "Unsupported OS for lld cargo config: $(uname -s)" >&2
            exit 1
            ;;
    esac
    mv "$tmp" "$config"

[private]
[windows]
_lld-cargo-config:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$$ErrorActionPreference = 'Stop'; $$linker = $$null; try { $$sysroot = (& rustc --print sysroot).Trim(); foreach ($$target in @('x86_64-pc-windows-msvc', 'aarch64-pc-windows-msvc')) { $$candidate = Join-Path $$sysroot \"lib\rustlib\$$target\bin\rust-lld.exe\"; if (Test-Path $$candidate) { $$linker = $$candidate; break } } } catch {}; if (-not $$linker) { foreach ($$name in @('rust-lld.exe', 'lld-link.exe')) { $$command = Get-Command $$name -ErrorAction SilentlyContinue; if ($$command) { $$linker = $$command.Source; break } } }; if (-not $$linker) { Write-Error \"LLVM lld was not found for the Windows MSVC target.`n`nlld is required for faster Rust builds (measured up to 26% faster locally).`n`nInstall one of these, then rerun the just command:`n  rustup component add llvm-tools-preview`n`nOr install LLVM lld-link:`n  winget install LLVM.LLVM`n  choco install llvm`n`nThe build requires lld. It looks for rust-lld.exe in the active Rust sysroot first, then falls back to rust-lld.exe or lld-link.exe on PATH.\"; exit 1 }; New-Item -ItemType Directory -Force -Path .cargo | Out-Null; $$config = '.cargo/config.toml'; $$body = if (Test-Path $$config) { Get-Content -Raw $$config } else { '' }; $$body = [regex]::Replace($$body, '(?ms)^# BEGIN Mesh-LLM lld config\\r?\\n.*?^# END Mesh-LLM lld config\\r?\\n?', ''); if ($$body -and -not $$body.EndsWith(\"`n\")) { $$body += \"`n\" }; $$body += \"# BEGIN Mesh-LLM lld config`n[target.x86_64-pc-windows-msvc]`nlinker = `\"$$linker`\"`n`n[target.aarch64-pc-windows-msvc]`nlinker = `\"$$linker`\"`n# END Mesh-LLM lld config`n\"; Set-Content -Path $$config -Value $$body"

# Build for the current platform (macOS Metal ABI, Linux/Windows auto ABI backend)
[macos]
build: build-mac

# Fast local iteration build: patched llama.cpp + UI + debug mesh-llm.
[macos]
build-dev:
    @MESH_LLM_BUILD_PROFILE=dev scripts/build-mac.sh

# Linux overrides:
#   just build backend=cpu
#   just build backend=cuda cuda_arch='120;86'
#   just build backend=rocm rocm_arch='gfx942;gfx90a'
#   just build backend=vulkan
[linux]
build backend="" cuda_arch="" rocm_arch="":
    @scripts/build-linux.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Fast local iteration build: patched llama.cpp + UI + debug mesh-llm.
[linux]
build-dev backend="" cuda_arch="" rocm_arch="":
    @MESH_LLM_BUILD_PROFILE=dev scripts/build-linux.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Windows overrides:
#   just build backend=cpu
#   just build backend=cuda cuda_arch='120;86'
#   just build backend=rocm rocm_arch='gfx942;gfx90a'
#   just build backend=vulkan
[windows]
build backend="" cuda_arch="" rocm_arch="":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend "{{backend}}" -CudaArch "{{cuda_arch}}" -RocmArch "{{rocm_arch}}"

# Fast local iteration build: patched llama.cpp + UI + debug mesh-llm.
[windows]
build-dev backend="" cuda_arch="" rocm_arch="":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:MESH_LLM_BUILD_PROFILE='dev'; & './scripts/build-windows.ps1' -Backend '{{backend}}' -CudaArch '{{cuda_arch}}' -RocmArch '{{rocm_arch}}'"

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
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cpu -BuildProfile release

# Build a Linux CUDA release artifact (primary / R535-compatible lane).
release-build-cuda cuda_arch="75;80;86;87;89;90":
    @MESH_LLM_BUILD_PROFILE=release scripts/build-linux.sh --backend cuda --cuda-arch "{{ cuda_arch }}"

release-build-cuda-blackwell cuda_arch="75;80;86;87;89;90;100;120":
    @MESH_LLM_BUILD_PROFILE=release scripts/build-linux.sh --backend cuda --cuda-arch "{{ cuda_arch }}"

release-build-cuda-windows cuda_arch="75;80;86;87;89;90":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cuda -CudaArch "{{cuda_arch}}" -BuildProfile release

release-build-cuda-blackwell-windows cuda_arch="75;80;86;87;89;90;100;120":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend cuda -CudaArch "{{cuda_arch}}" -BuildProfile release

# Build a Linux ROCm ABI release artifact with an explicit architecture list.
release-build-rocm rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @MESH_LLM_BUILD_PROFILE=release scripts/build-linux-rocm.sh "{{ rocm_arch }}"

release-build-rocm-windows rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend rocm -RocmArch "{{rocm_arch}}" -BuildProfile release

# Build a Linux Vulkan ABI release artifact.
release-build-vulkan:
    @MESH_LLM_BUILD_PROFILE=release scripts/build-linux.sh --backend vulkan

release-build-vulkan-windows:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backend vulkan -BuildProfile release

# Build the skippy benchmark/debug telemetry collector.
[unix]
metrics-server-build: _lld-cargo-config
    cargo build -p metrics-server

[windows]
metrics-server-build: _lld-cargo-config
    @cargo build -p metrics-server

# Build the binaries copied into the Skippy WAN Docker lab image.
[linux]
skippy-wan-lab-build-bins:
    cargo build --release --locked -p skippy-server -p skippy-prompt -p metrics-server -p skippy-model-package

# Generate a reproducible benchmark corpus for skippy bench tooling.
bench-corpus tier="smoke" *ARGS:
    scripts/generate-bench-corpus.py "{{ tier }}" {{ ARGS }}

# Run skippy family certification checks.
family-certify *ARGS: _lld-cargo-config
    scripts/family-certify.sh {{ ARGS }}

# Run target/draft speculative compatibility checks.
spec-bench target draft *ARGS: _lld-cargo-config
    LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" cargo build -p llama-spec-bench
    LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" target/debug/llama-spec-bench --target-model-path "{{ target }}" --draft-model-path "{{ draft }}" {{ ARGS }}

# Smoke a standalone skippy OpenAI frontend stage.
skippy-openai-smoke *ARGS: _lld-cargo-config
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
check-release: _lld-cargo-config
    cargo run -p xtask -- repo-consistency release-targets

[windows]
check-release: _lld-cargo-config
    @cargo run -p xtask -- repo-consistency release-targets

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

# Run the UI dev server with Vite HMR, proxying /api to mesh-llm (default: http://127.0.0.1:3131)
ui-dev api="http://127.0.0.1:3131" port="5173":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ ui_dir }}"
    MESH_UI_API_ORIGIN="{{ api }}" VITE_API_URL="{{ api }}" pnpm run dev -- --host 0.0.0.0 --port {{ port }}

# Run the UI dev server proxying to the public meshllm.cloud API
ui-dev-public: (ui-dev "https://meshllm.cloud")

# Run the legacy UI dev server with Vite HMR (default: http://127.0.0.1:3131)
ui-legacy api="http://127.0.0.1:3131" port="5174":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ ui_legacy_dir }}"
    MESH_UI_API_ORIGIN="{{ api }}" npm run dev -- --host 0.0.0.0 --port {{ port }}

# Run the legacy UI dev server proxying to the public meshllm.cloud API
ui-legacy-public: (ui-legacy "https://meshllm.cloud")

# Run UI unit tests (vitest)
ui-test:
    cd "{{ ui_dir }}" && pnpm test

# ── Full Validation Gate ───────────────────────────────────────

# Run all checks: Rust tests, fmt, clippy, ESLint, Prettier, E2E smoke.
test-all: _lld-cargo-config
    #!/usr/bin/env bash
    set -euo pipefail

    native_backend="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-${LLAMA_BACKEND:-}}}"
    if [[ -z "$native_backend" ]]; then
        case "$(uname -s)" in
            Darwin) native_backend="metal" ;;
            *) native_backend="cpu" ;;
        esac
    fi
    export LLAMA_STAGE_BACKEND="$native_backend"

    if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
        LLAMA_STAGE_BUILD_DIR="$(scripts/build-llama.sh --print-build-dir)"
        export LLAMA_STAGE_BUILD_DIR
    fi

    echo "=== Native llama.cpp ABI ($LLAMA_STAGE_BACKEND) ==="
    echo "Build dir: $LLAMA_STAGE_BUILD_DIR"
    scripts/prepare-llama.sh
    scripts/build-llama.sh
    echo ""

    # Each UI step runs in a subshell so cd doesn't leak between steps.
    echo "=== 1/7 Rust format check ==="
    cargo fmt --all -- --check
    echo ""
    echo "=== 2/7 Clippy ==="
    cargo clippy -p mesh-llm -- -D warnings
    echo ""
    echo "=== 3/7 Rust tests ==="
    echo "--- mesh-llm ---"
    cargo test -p mesh-llm
    echo "--- skippy-runtime lib ---"
    cargo test -p skippy-runtime --lib
    echo ""
    echo "=== 4/7 ESLint + Prettier ==="
    (cd "{{ ui_dir }}" && pnpm run lint)
    echo ""
    echo "=== 5/7 UI type check (tsc) ==="
    (cd "{{ ui_dir }}" && pnpm run typecheck)
    echo ""
    echo "=== 6/7 UI unit tests (vitest) ==="
    (cd "{{ ui_dir }}" && pnpm test)
    echo ""
    echo "=== 7/7 E2E smoke tests (Playwright) ==="
    if curl -sf http://127.0.0.1:3131/health >/dev/null 2>&1; then
        (cd "{{ ui_dir }}" && pnpm run test:e2e)
    else
        echo "No server on port 3131 — starting UI dev server with public mesh..."

        # Start dev server in background, capture PID tree for cleanup
        MESH_UI_API_ORIGIN="https://meshllm.cloud" VITE_API_URL="https://meshllm.cloud" bash -c 'cd "{{ ui_dir }}" && pnpm exec vite --host 0.0.0.0 --port 5173' &
        DEV_PID=$!

        # Wait for dev server to be ready (up to 30s)
        READY=false
        for i in $(seq 1 30); do
            if curl -sf http://127.0.0.1:5173/ >/dev/null 2>&1; then
                READY=true
                break
            fi
            sleep 1
        done

        # Cleanup function - always stop the dev server
        cleanup_dev() {
            kill $DEV_PID 2>/dev/null || true
            wait $DEV_PID >/dev/null 2>&1 || true
        }

        if [ "$READY" = true ]; then
            (cd "{{ ui_dir }}" && PLAYWRIGHT_PORT=5173 pnpm run test:e2e e2e/smoke/home.spec.ts e2e/smoke/topnav-responsive.spec.ts)
            E2E_EXIT=$?
            cleanup_dev
            echo "Stopped UI dev server."

            exit $E2E_EXIT
        else
            cleanup_dev
            echo "WARNING: UI dev server didn't start in time — skipping E2E tests."
            echo "Run E2E manually:"
            echo "  cd {{ ui_dir }} && pnpm run test:e2e"
        fi
    fi
    echo ""
    echo "All checks passed."

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

# Clean Rust, llama.cpp, and UI build artifacts.
[unix]
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf \
        target \
        .deps/llama.cpp/build-stage-abi-* \
        .deps/llama-build/build-stage-abi-* \
        "{{ ui_dir }}/node_modules" \
        "{{ ui_dir }}/dist"
    echo "Cleaned Rust target, llama.cpp build dirs, and UI artifacts"

[windows]
clean:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -Recurse -Force target,'.deps/llama.cpp/build-stage-abi-*','.deps/llama-build/build-stage-abi-*','{{ ui_dir }}/node_modules','{{ ui_dir }}/dist' -ErrorAction SilentlyContinue"
    echo "Cleaned Rust target, llama.cpp build dirs, and UI artifacts"

# Clean UI build artifacts (node_modules, dist). Fixes stale pnpm state.
[unix]
ui-clean:
    cd "{{ ui_dir }}" && rm -rf node_modules dist
    echo "Cleaned UI: node_modules + dist removed"

[windows]
ui-clean:
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
