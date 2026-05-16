#!/usr/bin/env bash

set -euo pipefail

UI_DIR="${1:?usage: build-ui.sh /path/to/ui}"
UI_DIR="$(cd "$UI_DIR" && pwd)"
DIST_DIR="$UI_DIR/dist"
NODE_MODULES_DIR="$UI_DIR/node_modules"
UI_BUILD_ENV_STAMP="$DIST_DIR/.mesh-llm-ui-build-env"
MESH_LLM_UI_BUILD_PROFILE="${MESH_LLM_BUILD_PROFILE:-debug}"
MESH_LLM_UI_BUILD_PROFILE="${MESH_LLM_UI_BUILD_PROFILE,,}"

case "$MESH_LLM_UI_BUILD_PROFILE" in
    dev|debug)
        : "${VITE_MESH_LLM_DEBUG_UI:=true}"
        ;;
    release)
        VITE_MESH_LLM_DEBUG_UI=false
        ;;
    *)
        echo "Unsupported MESH_LLM_BUILD_PROFILE '$MESH_LLM_UI_BUILD_PROFILE'. Expected debug, dev, or release." >&2
        exit 1
        ;;
esac

export VITE_MESH_LLM_DEBUG_UI

ui_build_inputs=(
    "$UI_DIR/package.json"
    "$UI_DIR/pnpm-lock.yaml"
    "$UI_DIR/vite.config.ts"
    "$UI_DIR/tsconfig.json"
    "$UI_DIR/tsconfig.app.json"
    "$UI_DIR/tsconfig.node.json"
    "$UI_DIR/biome.json"
    "$UI_DIR/index.html"
    "$UI_DIR/src"
    "$UI_DIR/public"
)

dist_has_output() {
    [[ -d "$DIST_DIR" ]] && find "$DIST_DIR" -type f -print -quit | grep -q .
}

expected_build_env_stamp() {
    cat <<EOF
MESH_LLM_BUILD_PROFILE=$MESH_LLM_UI_BUILD_PROFILE
VITE_MESH_LLM_DEBUG_UI=$VITE_MESH_LLM_DEBUG_UI
VITE_BASE_PATH=${VITE_BASE_PATH:-}
VITE_ROUTER_BASE_PATH=${VITE_ROUTER_BASE_PATH:-}
VITE_STORAGE_NAMESPACE=${VITE_STORAGE_NAMESPACE:-}
EOF
}

ui_build_env_is_stale() {
    if [[ ! -f "$UI_BUILD_ENV_STAMP" ]]; then
        return 0
    fi

    if ! diff -q <(expected_build_env_stamp) "$UI_BUILD_ENV_STAMP" >/dev/null; then
        return 0
    fi

    return 1
}

ui_build_is_stale() {
    if ! dist_has_output; then
        return 0
    fi

    if ui_build_env_is_stale; then
        return 0
    fi

    for path in "${ui_build_inputs[@]}"; do
        [[ -e "$path" ]] || continue
        if find "$path" -type f -newer "$DIST_DIR" -print -quit | grep -q .; then
            return 0
        fi
    done

    return 1
}

pnpm_install_is_stale() {
    if [[ ! -d "$NODE_MODULES_DIR" ]]; then
        return 0
    fi

    local manifest
    for manifest in "$UI_DIR/package.json" "$UI_DIR/pnpm-lock.yaml"; do
        [[ -e "$manifest" ]] || continue
        if [[ "$manifest" -nt "$NODE_MODULES_DIR" ]]; then
            return 0
        fi
    done

    return 1
}

if ui_build_is_stale; then
    echo "Building mesh-llm UI (profile: $MESH_LLM_UI_BUILD_PROFILE, debug UI: $VITE_MESH_LLM_DEBUG_UI)..."
    cd "$UI_DIR"

    if pnpm_install_is_stale; then
        pnpm install --frozen-lockfile
    fi

    pnpm run build
    expected_build_env_stamp > "$UI_BUILD_ENV_STAMP"
else
    echo "Skipping mesh-llm UI build; dist is up to date (profile: $MESH_LLM_UI_BUILD_PROFILE, debug UI: $VITE_MESH_LLM_DEBUG_UI)."
fi
