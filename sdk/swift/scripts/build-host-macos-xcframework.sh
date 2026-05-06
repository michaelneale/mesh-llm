#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWIFT_DIR="$REPO_ROOT/sdk/swift"
FFI_DIR="$SWIFT_DIR/Generated/FFI"
TARGET_DIR="$REPO_ROOT/target"
XCFRAMEWORK_DIR="$SWIFT_DIR/Generated"
FRAMEWORK_NAME="MeshLLMFFI"
GENERATED_SWIFT="$SWIFT_DIR/Sources/MeshLLM/Generated/mesh_ffi.swift"

echo "Building host macOS $FRAMEWORK_NAME XCFramework..."
echo "Repo root: $REPO_ROOT"

if ! cargo metadata --no-deps --format-version 1 2>/dev/null | grep -q '"name":"mesh-api-ffi"'; then
  echo "ERROR: mesh-api-ffi crate not found. Ensure the workspace is configured."
  exit 1
fi

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
  arm64|aarch64)
    RUST_TARGET="aarch64-apple-darwin"
    XCFRAMEWORK_ID="macos-arm64"
    SUPPORTED_ARCH="arm64"
    ;;
  x86_64)
    RUST_TARGET="x86_64-apple-darwin"
    XCFRAMEWORK_ID="macos-x86_64"
    SUPPORTED_ARCH="x86_64"
    ;;
  *)
    echo "Unsupported macOS host architecture: $HOST_ARCH" >&2
    exit 1
    ;;
esac

rustup target add "$RUST_TARGET" 2>/dev/null || true

"$SWIFT_DIR/scripts/generate-swift-bindings.sh"

RUSTUP_RUSTC="$(rustup run stable which rustc)"
echo "Using rustc: $RUSTUP_RUSTC"
echo "Building for $RUST_TARGET..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target "$RUST_TARGET" --no-default-features

LIB_PATH="$TARGET_DIR/$RUST_TARGET/release/libmesh_ffi.a"

echo "Syncing UniFFI API checksums into generated Swift bindings..."
python3 - "$LIB_PATH" "$GENERATED_SWIFT" <<'PY'
import pathlib
import re
import subprocess
import sys

lib_path = pathlib.Path(sys.argv[1])
swift_path = pathlib.Path(sys.argv[2])

disassembly = subprocess.run(
    ["otool", "-tvV", str(lib_path)],
    check=True,
    capture_output=True,
    text=True,
).stdout

pattern = re.compile(
    r"_uniffi_mesh_ffi_(checksum_[A-Za-z0-9_]+):\n[0-9a-f]+\s+mov\s+w0, #0x([0-9a-f]+)\n[0-9a-f]+\s+ret",
    re.MULTILINE,
)
checksums = {name: int(value, 16) for name, value in pattern.findall(disassembly)}

swift = swift_path.read_text()

for name, value in checksums.items():
    call = f"{name}()"
    swift = re.sub(
        rf"({re.escape(call)} != )\d+",
        rf"\g<1>{value}",
        swift,
    )

swift_path.write_text(swift)
PY

FRAMEWORK_DIR="$TARGET_DIR/frameworks/macos-host/$FRAMEWORK_NAME.framework"
rm -rf "$FRAMEWORK_DIR"
mkdir -p "$FRAMEWORK_DIR/Headers"
mkdir -p "$FRAMEWORK_DIR/Modules"

cp "$LIB_PATH" "$FRAMEWORK_DIR/$FRAMEWORK_NAME"
cp "$FFI_DIR/MeshLLMFFI.h" "$FRAMEWORK_DIR/Headers/MeshLLMFFI.h"
cp "$FFI_DIR/MeshLLMFFI.modulemap" "$FRAMEWORK_DIR/Modules/module.modulemap"

if [ -f "$SWIFT_DIR/PrivacyInfo.xcprivacy" ]; then
  cp "$SWIFT_DIR/PrivacyInfo.xcprivacy" "$FRAMEWORK_DIR/PrivacyInfo.xcprivacy"
  echo "  Embedded PrivacyInfo.xcprivacy in host macOS framework"
else
  echo "WARNING: PrivacyInfo.xcprivacy not found at $SWIFT_DIR/PrivacyInfo.xcprivacy"
fi

cat > "$FRAMEWORK_DIR/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>ai.meshllm.MeshLLMFFI</string>
    <key>CFBundleName</key>
    <string>MeshLLMFFI</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>13.0</string>
</dict>
</plist>
PLIST

echo "Creating host macOS XCFramework..."
XCFW_OUT="$XCFRAMEWORK_DIR/$FRAMEWORK_NAME.xcframework"
rm -rf "$XCFW_OUT"
mkdir -p "$XCFW_OUT/$XCFRAMEWORK_ID/$FRAMEWORK_NAME.framework"
cp -R "$FRAMEWORK_DIR/" "$XCFW_OUT/$XCFRAMEWORK_ID/$FRAMEWORK_NAME.framework/"

cat > "$XCFW_OUT/Info.plist" << XCINFO
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>AvailableLibraries</key>
    <array>
        <dict>
            <key>BinaryPath</key>
            <string>MeshLLMFFI.framework/MeshLLMFFI</string>
            <key>LibraryIdentifier</key>
            <string>$XCFRAMEWORK_ID</string>
            <key>LibraryPath</key>
            <string>MeshLLMFFI.framework</string>
            <key>SupportedArchitectures</key>
            <array><string>$SUPPORTED_ARCH</string></array>
            <key>SupportedPlatform</key>
            <string>macos</string>
        </dict>
    </array>
    <key>CFBundlePackageType</key>
    <string>XFWK</string>
    <key>XCFrameworkFormatVersion</key>
    <string>1.0</string>
</dict>
</plist>
XCINFO

echo "XCFramework created at: $XCFW_OUT"

PRIVACY_COUNT=$(find "$XCFW_OUT" -name "PrivacyInfo.xcprivacy" | wc -l | tr -d ' ')
echo "Found $PRIVACY_COUNT PrivacyInfo.xcprivacy file(s) inside XCFramework"
if [ "$PRIVACY_COUNT" -lt 1 ]; then
  echo "ERROR: PrivacyInfo.xcprivacy not embedded in XCFramework!"
  exit 1
fi

echo "Host macOS XCFramework build complete!"
