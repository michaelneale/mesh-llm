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
  brew install llvm lld

If Homebrew installed LLVM but the tools are not on PATH, that is okay for
Mesh-LLM builds. The build scripts look under:
  $(brew --prefix llvm)/bin
  $(brew --prefix lld)/bin
  /opt/homebrew/opt/llvm/bin
  /opt/homebrew/opt/lld/bin
  /usr/local/opt/llvm/bin
  /usr/local/opt/lld/bin
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
    local lld_prefix=""
    local candidate

    if command -v brew >/dev/null 2>&1; then
        llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
        lld_prefix="$(brew --prefix lld 2>/dev/null || true)"
    fi

    local clang_prefix
    local ld_prefix
    for clang_prefix in \
        "$llvm_prefix" \
        /opt/homebrew/opt/llvm \
        /usr/local/opt/llvm; do
        [[ -n "$clang_prefix" && -x "$clang_prefix/bin/clang" ]] || continue

        for ld_prefix in \
            "$clang_prefix" \
            "$lld_prefix" \
            /opt/homebrew/opt/lld \
            /usr/local/opt/lld; do
            [[ -n "$ld_prefix" && -x "$ld_prefix/bin/ld64.lld" ]] || continue
            local shim_dir="$PWD/.deps/fast-rust-linker/ld64-lld-bin"
            mkdir -p "$shim_dir"
            rm -f "$shim_dir/ld"
            {
                printf '#!/usr/bin/env bash\n'
                printf 'exec %q "$@"\n' "$ld_prefix/bin/ld64.lld"
            } >"$shim_dir/ld"
            chmod +x "$shim_dir/ld"
            export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$clang_prefix/bin/clang"
            export CARGO_TARGET_X86_64_APPLE_DARWIN_LINKER="$clang_prefix/bin/clang"
            append_rustflag "-C link-arg=-B$shim_dir"
            echo "Using Rust linker: $ld_prefix/bin/ld64.lld"
            return
        done
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
