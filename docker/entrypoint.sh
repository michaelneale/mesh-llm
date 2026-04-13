#!/bin/sh
# mesh-llm docker entrypoint — selects runtime mode via APP_MODE env var
# The UI is always embedded in the binary via include_dir!; there is no separate
# UI-less build.
#
# Modes (set via APP_MODE env var or ARG CMD in Dockerfile):
#   console  — client node: API on port 9337 + web console on port 3131 (default)
#   worker   — full mesh-llm node with bundled llama binaries (full-node images only)
#   (default) — pass through all args directly to mesh-llm
set -e
case "$APP_MODE" in
  console)
    exec mesh-llm --client --auto --port 9337 --console 3131 --listen-all
    ;;
  worker)
    BIN_DIR=/usr/local/lib/mesh-llm/bin
    set -- "$BIN_DIR"/rpc-server-*
    RPC_SERVER="$1"
    set -- "$BIN_DIR"/llama-server-*
    LLAMA_SERVER="$1"
    if [ ! -e "$RPC_SERVER" ] || [ ! -e "$LLAMA_SERVER" ] || [ ! -x "$BIN_DIR/llama-moe-split" ]; then
      echo "APP_MODE=worker requires bundled llama binaries in $BIN_DIR; use a full-node image (:cpu/:cuda/:rocm/:vulkan) or APP_MODE=console." >&2
      exit 1
    fi
    exec mesh-llm --auto --port 9337 --console 3131 --bin-dir "$BIN_DIR" --listen-all
    ;;
  *)
    exec mesh-llm "$@"
    ;;
esac
