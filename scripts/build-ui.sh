#!/usr/bin/env bash

set -euo pipefail

UI_DIR="${1:?usage: build-ui.sh /path/to/ui}"
DIST_DIR="$UI_DIR/dist"
NODE_MODULES_DIR="$UI_DIR/node_modules"

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

ui_build_is_stale() {
    if ! dist_has_output; then
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
    echo "Building mesh-llm UI..."
    cd "$UI_DIR"

    if pnpm_install_is_stale; then
        pnpm install --frozen-lockfile
    fi

    pnpm run build
else
    echo "Skipping mesh-llm UI build; dist is up to date."
fi
