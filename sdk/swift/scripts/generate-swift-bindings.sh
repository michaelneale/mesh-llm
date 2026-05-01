#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMP_DIR="$(mktemp -d)"
OUT_DIR="$TMP_DIR/out"
RUNNER_DIR="$TMP_DIR/runner"
SWIFT_SOURCE_DIR="$REPO_ROOT/sdk/swift/Sources/MeshLLM/Generated"
FFI_DIR="$REPO_ROOT/sdk/swift/Generated/FFI"
FFI_MODULE_NAME="MeshLLMFFI"
FFI_HEADER_NAME="${FFI_MODULE_NAME}.h"
FFI_MODULEMAP_NAME="${FFI_MODULE_NAME}.modulemap"
UNIFFI_HEADER_SOURCE="mesh_ffiFFI.h"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$RUNNER_DIR/src" "$OUT_DIR" "$SWIFT_SOURCE_DIR" "$FFI_DIR"
rm -f \
  "$FFI_DIR/mesh_ffiFFI.h" \
  "$FFI_DIR/mesh_ffiFFI.modulemap" \
  "$FFI_DIR/$FFI_HEADER_NAME" \
  "$FFI_DIR/$FFI_MODULEMAP_NAME"

cat > "$RUNNER_DIR/Cargo.toml" <<'EOF'
[package]
name = "swift_bindgen_runner"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
camino = "1"
uniffi_bindgen = "=0.31.0"
EOF

cat > "$RUNNER_DIR/src/main.rs" <<EOF
use anyhow::Result;
use camino::Utf8PathBuf;
use uniffi_bindgen::bindings::{generate_swift_bindings, SwiftBindingsOptions};

fn main() -> Result<()> {
    let out_dir = Utf8PathBuf::from("$OUT_DIR");
    std::fs::create_dir_all(out_dir.as_std_path())?;
    generate_swift_bindings(SwiftBindingsOptions {
        generate_swift_sources: true,
        generate_headers: true,
        generate_modulemap: true,
        source: Utf8PathBuf::from("$REPO_ROOT/crates/mesh-api-ffi/src/mesh_ffi.udl"),
        out_dir,
        xcframework: false,
        module_name: Some("$FFI_MODULE_NAME".into()),
        modulemap_filename: Some("$FFI_MODULEMAP_NAME".into()),
        metadata_no_deps: true,
        link_frameworks: Vec::new(),
    })?;
    Ok(())
}
EOF

cargo run --manifest-path "$RUNNER_DIR/Cargo.toml"

cp "$OUT_DIR/mesh_ffi.swift" "$SWIFT_SOURCE_DIR/mesh_ffi.swift"
perl -0pi -e 's/#if canImport\(mesh_ffiFFI\)/#if canImport('"$FFI_MODULE_NAME"')/g; s/import mesh_ffiFFI/import '"$FFI_MODULE_NAME"'/g' "$SWIFT_SOURCE_DIR/mesh_ffi.swift"
cp "$OUT_DIR/$UNIFFI_HEADER_SOURCE" "$FFI_DIR/$FFI_HEADER_NAME"
cat > "$FFI_DIR/$FFI_MODULEMAP_NAME" <<EOF
framework module $FFI_MODULE_NAME {
  header "$FFI_HEADER_NAME"
  export *
  use "Darwin"
  use "_Builtin_stdbool"
  use "_Builtin_stdint"
}
EOF
