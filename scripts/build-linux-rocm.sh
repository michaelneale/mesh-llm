#!/usr/bin/env bash
# Compatibility wrapper for ROCm builds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_ARCH="${1:-}"

if [[ -n "$ROCM_ARCH" ]]; then
    exec "$SCRIPT_DIR/build-linux.sh" --backend rocm --rocm-arch "$ROCM_ARCH"
fi

exec "$SCRIPT_DIR/build-linux.sh" --backend rocm
