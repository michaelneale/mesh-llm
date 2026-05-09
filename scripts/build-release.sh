#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$REPO_ROOT/.deps/llama-build}"
UI_DIR="$REPO_ROOT/crates/mesh-llm-ui"

append_rustflag() {
    local flag="$1"
    case " ${RUSTFLAGS:-} " in
        *" $flag "*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$flag" ;;
    esac
}

configure_lld_linker() {
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
            append_rustflag "-C link-arg=-fuse-ld=lld"
            echo "Using Rust linker: $(command -v ld.lld)"
            ;;
        Darwin)
            local lld=""
            local lld_prefix=""
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
            append_rustflag "-C link-arg=-fuse-ld=$lld"
            echo "Using Rust linker: $lld"
            ;;
        *)
            echo "Unsupported OS for release build: $(uname -s)" >&2
            exit 1
            ;;
    esac
}

os_name="$(uname -s)"
case "$os_name" in
    Darwin)
        BACKEND="${LLAMA_STAGE_BACKEND:-metal}"
        ;;
    Linux)
        BACKEND="${LLAMA_STAGE_BACKEND:-cpu}"
        ;;
    *)
        echo "Unsupported OS for release build: $os_name" >&2
        exit 1
        ;;
esac

if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" && -n "${SKIPPY_LLAMA_BUILD_DIR:-}" ]]; then
    export LLAMA_STAGE_BUILD_DIR="$SKIPPY_LLAMA_BUILD_DIR"
fi
if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    if [[ "$BACKEND" == "cpu" ]]; then
        export LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_ROOT/build-stage-abi-static"
    else
        export LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_ROOT/build-stage-abi-$BACKEND"
    fi
fi

configure_lld_linker

echo "Preparing patched llama.cpp ABI checkout..."
LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Building patched llama.cpp ABI ($BACKEND)..."
LLAMA_WORKDIR="$LLAMA_DIR" \
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    LLAMA_STAGE_BACKEND="$BACKEND" \
    "$SCRIPT_DIR/build-llama.sh"

echo "Building UI..."
"$SCRIPT_DIR/build-ui.sh" "$UI_DIR"

echo "Building mesh-llm..."
(cd "$REPO_ROOT" && cargo build --release --locked -p mesh-llm)
