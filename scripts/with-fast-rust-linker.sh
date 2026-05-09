#!/usr/bin/env bash
# Configure the platform fast linker for Rust builds, then optionally exec a command.

set -euo pipefail

append_rustflag() {
    local flag="$1"
    case " ${RUSTFLAGS:-} " in
        *" $flag "*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$flag" ;;
    esac
}

linux_install_message() {
    cat >&2 <<'EOF'
Error: Rust fast linker 'mold' was not found.

Install mold, then rerun the just command. Common Linux packages:
  Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y mold
  Fedora:        sudo dnf install mold
  Arch Linux:    sudo pacman -S mold
  openSUSE:      sudo zypper install mold

If your distribution does not package mold, install it from your distro's
development tools repository or from https://github.com/rui314/mold.
EOF
}

macos_install_message() {
    cat >&2 <<'EOF'
Error: LLVM ld64.lld was not found.

Install LLVM, then rerun the just command:
  brew install llvm

If Homebrew installed LLVM but the tools are not on PATH, that is okay for
Mesh-LLM builds. The build scripts look under:
  $(brew --prefix llvm)/bin
  /opt/homebrew/opt/llvm/bin
  /usr/local/opt/llvm/bin
EOF
}

configure_linux_linker() {
    if ! command -v mold >/dev/null 2>&1; then
        linux_install_message
        exit 1
    fi
    append_rustflag "-C link-arg=-fuse-ld=mold"
    echo "Using Rust linker: mold"
}

configure_macos_linker() {
    local llvm_prefix=""
    local candidate

    if command -v brew >/dev/null 2>&1; then
        llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
    fi

    for candidate in \
        "$llvm_prefix" \
        /opt/homebrew/opt/llvm \
        /usr/local/opt/llvm; do
        if [[ -n "$candidate" && -x "$candidate/bin/clang" && -x "$candidate/bin/ld64.lld" ]]; then
            export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$candidate/bin/clang"
            export CARGO_TARGET_X86_64_APPLE_DARWIN_LINKER="$candidate/bin/clang"
            append_rustflag "-C link-arg=-fuse-ld=lld"
            echo "Using Rust linker: $candidate/bin/ld64.lld"
            return
        fi
    done

    macos_install_message
    exit 1
}

case "$(uname -s)" in
    Linux)
        configure_linux_linker
        ;;
    Darwin)
        configure_macos_linker
        ;;
    *)
        echo "Error: fast Rust linker preflight is only supported on Linux and macOS by this script." >&2
        exit 1
        ;;
esac

if [[ $# -gt 0 ]]; then
    exec "$@"
fi
