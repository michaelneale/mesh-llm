#!/usr/bin/env zsh
# build-mac.sh — build patched llama.cpp ABI libraries + mesh-llm on macOS.

setopt errexit nounset pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$REPO_ROOT/.deps/llama-build}"
MESH_DIR="$REPO_ROOT/crates/mesh-llm"
UI_DIR="$REPO_ROOT/crates/mesh-llm-ui"
build_profile="${MESH_LLM_BUILD_PROFILE:-debug}"
rustc_wrapper=""
build_profile="${build_profile:l}"

append_rustflag() {
    local flag="$1"
    case " ${RUSTFLAGS:-} " in
        *" $flag "*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$flag" ;;
    esac
}

configure_lld_linker() {
    local lld=""
    local lld_prefix=""

    if (( $+commands[ld64.lld] )); then
        lld="$(command -v ld64.lld)"
    elif (( $+commands[brew] )); then
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
}

configure_rust_cache() {
    if (( $+commands[sccache] )); then
        rustc_wrapper="$(command -v sccache)"
        echo "Using Rust compiler wrapper: $rustc_wrapper"
    fi
}

export LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-${SKIPPY_LLAMA_BUILD_DIR:-$LLAMA_BUILD_ROOT/build-stage-abi-metal}}"

configure_lld_linker

echo "Preparing patched llama.cpp ABI checkout..."
LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Building patched llama.cpp ABI (metal)..."
LLAMA_WORKDIR="$LLAMA_DIR" \
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    LLAMA_STAGE_BACKEND="${LLAMA_STAGE_BACKEND:-metal}" \
    "$SCRIPT_DIR/build-llama.sh"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
    fi

    configure_rust_cache
    case "$build_profile" in
        dev|debug)
            echo "Building mesh-llm (profile: dev, bin only)..."
            if [[ -n "$rustc_wrapper" ]]; then
                (cd "$REPO_ROOT" && RUSTC_WRAPPER="$rustc_wrapper" cargo build -p mesh-llm --bin mesh-llm)
            else
                (cd "$REPO_ROOT" && cargo build -p mesh-llm --bin mesh-llm)
            fi
            echo "Mesh binary: target/debug/mesh-llm"
            ;;
        release)
            echo "Building mesh-llm (profile: release)..."
            if [[ -n "$rustc_wrapper" ]]; then
                (cd "$REPO_ROOT" && RUSTC_WRAPPER="$rustc_wrapper" cargo build --release -p mesh-llm)
            else
                (cd "$REPO_ROOT" && cargo build --release -p mesh-llm)
            fi
            echo "Mesh binary: target/release/mesh-llm"
            ;;
        *)
            echo "Unsupported MESH_LLM_BUILD_PROFILE '$build_profile'. Expected debug, dev, or release." >&2
            exit 1
            ;;
    esac
fi
