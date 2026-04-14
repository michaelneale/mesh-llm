#!/usr/bin/env bash

set -euo pipefail

RELEASE_FLAVOR="${MESH_RELEASE_FLAVOR:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_BIN_DIR="$REPO_ROOT/llama.cpp/build/bin"
RELEASE_BIN_DIR="$REPO_ROOT/target/release"

python_bin() {
    if command -v python3 >/dev/null 2>&1; then
        echo python3
    elif command -v python >/dev/null 2>&1; then
        echo python
    else
        echo "python3 or python is required for packaging" >&2
        exit 1
    fi
}

release_os_name() {
    if [[ -n "${MESH_RELEASE_OS:-}" ]]; then
        printf '%s\n' "$MESH_RELEASE_OS"
        return 0
    fi

    uname -s
}

release_arch_name() {
    if [[ -n "${MESH_RELEASE_ARCH:-}" ]]; then
        printf '%s\n' "$MESH_RELEASE_ARCH"
        return 0
    fi

    uname -m
}

canonical_release_arch() {
    case "$(release_arch_name)" in
        x86_64|amd64)
            printf 'x86_64\n'
            ;;
        arm64|aarch64)
            printf 'aarch64\n'
            ;;
        arm|armv6l|armv6hf|armv7l|armv7hf)
            printf 'arm\n'
            ;;
        *)
            printf '%s\n' "$(release_arch_name)"
            ;;
    esac
}

flavor_suffix() {
    case "$1" in
        ""|cpu|metal)
            printf '\n'
            ;;
        *)
            printf -- '-%s\n' "$1"
            ;;
    esac
}

copy_runtime_libs() {
    local bundle_dir="$1"
    shopt -s nullglob
    case "$(release_os_name)" in
        Darwin)
            for lib in "$BUILD_BIN_DIR"/*.dylib; do
                cp "$lib" "$bundle_dir/"
            done
            ;;
        Linux)
            for lib in "$BUILD_BIN_DIR"/*.so "$BUILD_BIN_DIR"/*.so.*; do
                cp "$lib" "$bundle_dir/"
            done
            ;;
    esac
    shopt -u nullglob
}

bundle_bin_name() {
    local name="$1"
    if [[ "$name" == "mesh-llm" ]]; then
        echo "$name"
        return
    fi

    local binary_flavor="$RELEASE_FLAVOR"
    if [[ -z "$binary_flavor" ]]; then
        case "$(release_os_name)" in
            Darwin) binary_flavor="metal" ;;
            Linux) binary_flavor="cpu" ;;
        esac
    fi

    if [[ -n "$binary_flavor" ]]; then
        echo "${name}-${binary_flavor}"
    else
        echo "$name"
    fi
}

create_archive() {
    local source_dir="$1"
    local archive_path="$2"
    local archive_kind="$3"
    local py
    py="$(python_bin)"

    rm -f "$archive_path"
    mkdir -p "$(dirname "$archive_path")"

    "$py" - "$source_dir" "$archive_path" "$archive_kind" <<'PY'
import os
import sys
import tarfile
import zipfile

source_dir, archive_path, archive_kind = sys.argv[1:4]
base = os.path.basename(os.path.normpath(source_dir))
root = os.path.dirname(os.path.normpath(source_dir))

if archive_kind == "tar.gz":
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(source_dir, arcname=base)
elif archive_kind == "zip":
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for current_root, dirs, files in os.walk(source_dir):
            dirs.sort()
            files.sort()
            rel_root = os.path.relpath(current_root, root)
            if rel_root != ".":
                zf.write(current_root, rel_root)
            for filename in files:
                path = os.path.join(current_root, filename)
                rel = os.path.relpath(path, root)
                zf.write(path, rel)
else:
    raise SystemExit(f"unsupported archive kind: {archive_kind}")
PY
}

normalized_release_platform() {
    local os_name
    local arch_name

    os_name="$(release_os_name)"
    arch_name="$(canonical_release_arch)"

    case "$os_name/$arch_name" in
        Darwin/aarch64)
            printf 'macos/aarch64\n'
            ;;
        Linux/x86_64)
            printf 'linux/x86_64\n'
            ;;
        Linux/aarch64)
            printf 'linux/aarch64\n'
            ;;
        Linux/arm)
            printf 'linux/arm\n'
            ;;
        *)
            printf 'unsupported\n'
            ;;
    esac
}

effective_release_flavor() {
    case "$(normalized_release_platform)" in
        macos/aarch64)
            printf '%s\n' "${RELEASE_FLAVOR:-metal}"
            ;;
        linux/x86_64|linux/aarch64|linux/arm)
            printf '%s\n' "${RELEASE_FLAVOR:-cpu}"
            ;;
        *)
            printf '%s\n' "$RELEASE_FLAVOR"
            ;;
    esac
}

supported_release_flavors() {
    case "$(normalized_release_platform)" in
        macos/aarch64)
            printf 'metal\n'
            ;;
        linux/x86_64)
            printf 'cpu cuda rocm vulkan\n'
            ;;
        linux/aarch64)
            printf 'cpu\n'
            ;;
        *)
            printf '\n'
            ;;
    esac
}

release_target_flavor_supported() {
    local effective_flavor
    local supported_flavor

    effective_flavor="$(effective_release_flavor)"
    for supported_flavor in $(supported_release_flavors); do
        if [[ "$supported_flavor" == "$effective_flavor" ]]; then
            return 0
        fi
    done

    return 1
}

release_target_support() {
    case "$(normalized_release_platform)" in
        linux/arm)
            printf 'recognized-unsupported\n'
            ;;
        unsupported)
            printf 'unsupported\n'
            ;;
        *)
            if release_target_flavor_supported; then
                printf 'supported\n'
            else
                printf 'unsupported\n'
            fi
            ;;
    esac
}

release_target_error_message() {
    local os_name
    local arch_name
    local normalized
    local effective_flavor
    local support

    os_name="$(release_os_name)"
    arch_name="$(release_arch_name)"
    normalized="$(normalized_release_platform)"
    effective_flavor="$(effective_release_flavor)"
    support="$(release_target_support)"

    case "$support" in
        supported)
            printf 'release target is supported: %s\n' "$normalized"
            ;;
        recognized-unsupported)
            printf 'Recognized but unsupported release target: %s/%s (normalized: %s)\n' "$os_name" "$arch_name" "$normalized"
            ;;
        *)
            if [[ "$normalized" == "unsupported" ]]; then
                printf 'Unsupported OS/arch for packaging: %s/%s\n' "$os_name" "$arch_name"
            else
                printf 'Unsupported release target/flavor for packaging: %s/%s with flavor %s (normalized: %s)\n' "$os_name" "$arch_name" "$effective_flavor" "$normalized"
            fi
            ;;
    esac
}

resolve_release_target() {
    local normalized
    local support
    local effective_flavor

    normalized="$(normalized_release_platform)"
    support="$(release_target_support)"
    effective_flavor="$(effective_release_flavor)"
    BIN_EXT=""
    ARCHIVE_EXT="tar.gz"
    LEGACY_ASSET=""

    case "$support" in
        recognized-unsupported)
            return 2
            ;;
        unsupported)
            return 1
            ;;
    esac

    case "$normalized" in
        macos/aarch64)
            TARGET_TRIPLE="aarch64-apple-darwin"
            LEGACY_ASSET="mesh-bundle.tar.gz"
            ;;
        linux/x86_64)
            TARGET_TRIPLE="x86_64-unknown-linux-gnu"
            ;;
        linux/aarch64)
            TARGET_TRIPLE="aarch64-unknown-linux-gnu"
            ;;
        *)
            return 1
            ;;
    esac

    STABLE_ASSET="$(printf 'mesh-llm-%s%s.%s\n' "$TARGET_TRIPLE" "$(flavor_suffix "$effective_flavor")" "$ARCHIVE_EXT")"
    TARGET_TRIPLE="${TARGET_TRIPLE}$(flavor_suffix "$effective_flavor")"

    return 0
}

versioned_asset_name() {
    local version="$1"

    resolve_release_target
    printf 'mesh-llm-%s-%s.%s\n' "$version" "$TARGET_TRIPLE" "$ARCHIVE_EXT"
}

usage() {
    echo "usage: scripts/package-release.sh <version> [output_dir]" >&2
}

main() {
    if [[ $# -lt 1 || -z "${1:-}" ]]; then
        usage
        exit 1
    fi

    local version="$1"
    local output_dir="${2:-dist}"
    local os_name
    local staging_dir
    local bundle_dir
    local versioned_asset

    if ! resolve_release_target; then
        release_target_error_message >&2
        exit 1
    fi

    versioned_asset="$(versioned_asset_name "$version")"
    os_name="$(release_os_name)"

    mkdir -p "$output_dir"
    staging_dir="$(mktemp -d)"
    trap 'rm -rf "$staging_dir"' EXIT

    bundle_dir="$staging_dir/mesh-bundle"
    mkdir -p "$bundle_dir"

    cp "$RELEASE_BIN_DIR/mesh-llm${BIN_EXT}" "$bundle_dir/$(bundle_bin_name mesh-llm)"
    cp "$BUILD_BIN_DIR/rpc-server${BIN_EXT}" "$bundle_dir/$(bundle_bin_name rpc-server)"
    cp "$BUILD_BIN_DIR/llama-server${BIN_EXT}" "$bundle_dir/$(bundle_bin_name llama-server)"
    cp "$BUILD_BIN_DIR/llama-moe-analyze${BIN_EXT}" "$bundle_dir/llama-moe-analyze"
    cp "$BUILD_BIN_DIR/llama-moe-split${BIN_EXT}" "$bundle_dir/llama-moe-split"
    copy_runtime_libs "$bundle_dir"

    if [[ "$os_name" == "Darwin" ]]; then
        for bin in "$bundle_dir/$(bundle_bin_name mesh-llm)" "$bundle_dir/$(bundle_bin_name rpc-server)" "$bundle_dir/$(bundle_bin_name llama-server)" "$bundle_dir/llama-moe-analyze" "$bundle_dir/llama-moe-split"; do
            [[ -f "$bin" ]] || continue
            install_name_tool -add_rpath @executable_path/ "$bin" 2>/dev/null || true
        done
    fi

    create_archive "$bundle_dir" "$output_dir/$versioned_asset" "$ARCHIVE_EXT"
    create_archive "$bundle_dir" "$output_dir/$STABLE_ASSET" "$ARCHIVE_EXT"

    if [[ -n "$LEGACY_ASSET" ]]; then
        cp "$output_dir/$STABLE_ASSET" "$output_dir/$LEGACY_ASSET"
    fi

    echo "Created release archives:"
    find "$output_dir" -maxdepth 1 -type f -print | sort
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
