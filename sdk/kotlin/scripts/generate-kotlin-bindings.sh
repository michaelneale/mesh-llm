#!/usr/bin/env bash
set -euo pipefail

# Regenerate Kotlin UniFFI bindings from mesh-api-ffi/src/mesh_ffi.udl.
#
# Uses the uniffi-bindgen CLI installed on demand into a cached cargo root.
# The generated bindings are copied into the library source set AND the
# JVM example source set. Both currently keep their own copy; this script
# keeps them in lockstep.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UDL="$REPO_ROOT/mesh-api-ffi/src/mesh_ffi.udl"
KOTLIN_LIB_DIR="$REPO_ROOT/sdk/kotlin/src/main/kotlin/uniffi/mesh_ffi"
KOTLIN_EXAMPLE_DIR="$REPO_ROOT/sdk/kotlin/example/example-jvm/src/main/kotlin/uniffi/mesh_ffi"

BINDGEN_ROOT="${MESH_UNIFFI_BINDGEN_ROOT:-$HOME/.cache/mesh-llm/uniffi-bindgen-0.31.0}"
BINDGEN="$BINDGEN_ROOT/bin/uniffi-bindgen"

if [ ! -x "$BINDGEN" ]; then
    echo "Installing uniffi-bindgen 0.31.0 into $BINDGEN_ROOT ..."
    cargo install uniffi --version =0.31.0 --features cli --bin uniffi-bindgen \
        --root "$BINDGEN_ROOT"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

"$BINDGEN" generate "$UDL" --language kotlin --out-dir "$TMP_DIR" --no-format

SRC="$TMP_DIR/uniffi/mesh_ffi/mesh_ffi.kt"
if [ ! -f "$SRC" ]; then
    echo "Expected generator to produce $SRC" >&2
    exit 1
fi

mkdir -p "$KOTLIN_LIB_DIR" "$KOTLIN_EXAMPLE_DIR"
cp "$SRC" "$KOTLIN_LIB_DIR/mesh_ffi.kt"
cp "$SRC" "$KOTLIN_EXAMPLE_DIR/mesh_ffi.kt"

# The library dir historically carried a hand-written stub (MeshFfi.kt) that
# declared the same interface; the real bindings now replace it. Remove the
# stale stub if present so consumers don't resolve against it.
if [ -f "$KOTLIN_LIB_DIR/MeshFfi.kt" ]; then
    rm "$KOTLIN_LIB_DIR/MeshFfi.kt"
fi

echo "Regenerated:"
echo "  $KOTLIN_LIB_DIR/mesh_ffi.kt"
echo "  $KOTLIN_EXAMPLE_DIR/mesh_ffi.kt"
