#!/bin/sh
# mesh-llm docker entrypoint - selects runtime mode via APP_MODE.
set -e

HEADLESS_FLAG=""
if [ "$MESH_HEADLESS" = "1" ] || [ "$MESH_HEADLESS" = "true" ]; then
  HEADLESS_FLAG="--headless"
fi

case "$APP_MODE" in
  console|"")
    # shellcheck disable=SC2086
    exec mesh-llm --client --auto --port 9337 --console 3131 --listen-all $HEADLESS_FLAG
    ;;
  worker)
    echo "APP_MODE=worker is not supported by the client-only image; use a build that includes the staged runtime." >&2
    exit 1
    ;;
  *)
    exec mesh-llm "$@"
    ;;
esac
