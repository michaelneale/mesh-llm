#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWIFT_DIR="$REPO_ROOT/sdk/swift"
FFI_DIR="$SWIFT_DIR/Generated/FFI"
TARGET_DIR="$REPO_ROOT/target"
XCFRAMEWORK_DIR="$SWIFT_DIR/Generated"
FRAMEWORK_NAME="MeshLLMFFI"
GENERATED_SWIFT="$SWIFT_DIR/Sources/MeshLLM/Generated/mesh_ffi.swift"

echo "Building $FRAMEWORK_NAME XCFramework..."
echo "Repo root: $REPO_ROOT"

if ! cargo metadata --no-deps --format-version 1 2>/dev/null | grep -q '"name":"mesh-api-ffi"'; then
  echo "ERROR: mesh-api-ffi crate not found. Ensure the workspace is configured."
  exit 1
fi

rustup target add \
  aarch64-apple-ios \
  aarch64-apple-ios-sim \
  x86_64-apple-ios \
  aarch64-apple-ios-macabi \
  x86_64-apple-ios-macabi \
  aarch64-apple-darwin \
  x86_64-apple-darwin \
  2>/dev/null || true

"$SWIFT_DIR/scripts/generate-swift-bindings.sh"

# Resolve stable rustc from rustup (avoids Homebrew rustc shadowing)
RUSTUP_RUSTC="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc"
if [ ! -x "$RUSTUP_RUSTC" ]; then
  # Fallback: find any stable toolchain
  STABLE_TOOLCHAIN=$(rustup toolchain list | grep stable | head -1 | awk '{print $1}')
  RUSTUP_RUSTC="$(rustup run "$STABLE_TOOLCHAIN" which rustc)"
fi
echo "Using rustc: $RUSTUP_RUSTC"

echo "Building for aarch64-apple-ios..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target aarch64-apple-ios --no-default-features

echo "Building for aarch64-apple-ios-sim..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target aarch64-apple-ios-sim --no-default-features

echo "Building for x86_64-apple-ios..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target x86_64-apple-ios --no-default-features

echo "Building for aarch64-apple-ios-macabi (Mac Catalyst)..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target aarch64-apple-ios-macabi --no-default-features

echo "Building for x86_64-apple-ios-macabi (Mac Catalyst)..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target x86_64-apple-ios-macabi --no-default-features

echo "Building for aarch64-apple-darwin (macOS)..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target aarch64-apple-darwin --no-default-features

echo "Building for x86_64-apple-darwin (macOS)..."
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-api-ffi --target x86_64-apple-darwin --no-default-features

echo "Syncing UniFFI API checksums into generated Swift bindings..."
python3 - "$TARGET_DIR/aarch64-apple-darwin/release/libmesh_ffi.a" "$GENERATED_SWIFT" <<'PY'
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

echo "Creating fat library for iOS simulator..."
mkdir -p "$TARGET_DIR/ios-sim-fat"
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-sim/release/libmesh_ffi.a" \
  "$TARGET_DIR/x86_64-apple-ios/release/libmesh_ffi.a" \
  -output "$TARGET_DIR/ios-sim-fat/libmesh_ffi.a"

echo "Creating fat library for macOS..."
mkdir -p "$TARGET_DIR/macos-fat"
lipo -create \
  "$TARGET_DIR/aarch64-apple-darwin/release/libmesh_ffi.a" \
  "$TARGET_DIR/x86_64-apple-darwin/release/libmesh_ffi.a" \
  -output "$TARGET_DIR/macos-fat/libmesh_ffi.a"

echo "Creating fat library for Mac Catalyst..."
mkdir -p "$TARGET_DIR/ios-macabi-fat"
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-macabi/release/libmesh_ffi.a" \
  "$TARGET_DIR/x86_64-apple-ios-macabi/release/libmesh_ffi.a" \
  -output "$TARGET_DIR/ios-macabi-fat/libmesh_ffi.a"

create_framework() {
  local ARCH="$1"
  local LIB_PATH="$2"
  local FRAMEWORK_DIR="$TARGET_DIR/frameworks/$ARCH/$FRAMEWORK_NAME.framework"

  mkdir -p "$FRAMEWORK_DIR/Headers"
  mkdir -p "$FRAMEWORK_DIR/Modules"

  # Copy static library as the framework binary (no extension)
  cp "$LIB_PATH" "$FRAMEWORK_DIR/$FRAMEWORK_NAME"
  cp "$FFI_DIR/MeshLLMFFI.h" "$FRAMEWORK_DIR/Headers/MeshLLMFFI.h"
  cp "$FFI_DIR/MeshLLMFFI.modulemap" "$FRAMEWORK_DIR/Modules/module.modulemap"

  # Embed PrivacyInfo.xcprivacy (required for App Store submission)
  if [ -f "$SWIFT_DIR/PrivacyInfo.xcprivacy" ]; then
    cp "$SWIFT_DIR/PrivacyInfo.xcprivacy" "$FRAMEWORK_DIR/PrivacyInfo.xcprivacy"
    echo "  Embedded PrivacyInfo.xcprivacy in $ARCH framework"
  else
    echo "WARNING: PrivacyInfo.xcprivacy not found at $SWIFT_DIR/PrivacyInfo.xcprivacy"
  fi

  # Minimal Info.plist (required by xcodebuild -create-xcframework)
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
    <string>16.0</string>
</dict>
</plist>
PLIST

  echo "  Created framework bundle for $ARCH"
}

echo "Assembling framework bundles..."
create_framework "ios"     "$TARGET_DIR/aarch64-apple-ios/release/libmesh_ffi.a"
create_framework "ios-sim" "$TARGET_DIR/ios-sim-fat/libmesh_ffi.a"
create_framework "ios-macabi" "$TARGET_DIR/ios-macabi-fat/libmesh_ffi.a"
create_framework "macos"   "$TARGET_DIR/macos-fat/libmesh_ffi.a"

echo "Creating XCFramework..."
rm -rf "$XCFRAMEWORK_DIR/$FRAMEWORK_NAME.xcframework"
mkdir -p "$XCFRAMEWORK_DIR"

XCFW_OUT="$XCFRAMEWORK_DIR/$FRAMEWORK_NAME.xcframework"

xcodebuild -create-xcframework \
  -framework "$TARGET_DIR/frameworks/ios/$FRAMEWORK_NAME.framework" \
  -framework "$TARGET_DIR/frameworks/ios-sim/$FRAMEWORK_NAME.framework" \
  -framework "$TARGET_DIR/frameworks/ios-macabi/$FRAMEWORK_NAME.framework" \
  -framework "$TARGET_DIR/frameworks/macos/$FRAMEWORK_NAME.framework" \
  -output "$XCFW_OUT" 2>/dev/null || true

if [ ! -d "$XCFW_OUT" ]; then
  echo "xcodebuild unavailable or failed; assembling XCFramework manually..."
  mkdir -p "$XCFW_OUT/ios-arm64/$FRAMEWORK_NAME.framework"
  mkdir -p "$XCFW_OUT/ios-arm64_x86_64-simulator/$FRAMEWORK_NAME.framework"
  mkdir -p "$XCFW_OUT/ios-arm64_x86_64-maccatalyst/$FRAMEWORK_NAME.framework"
  mkdir -p "$XCFW_OUT/macos-arm64_x86_64/$FRAMEWORK_NAME.framework"

  cp -R "$TARGET_DIR/frameworks/ios/$FRAMEWORK_NAME.framework/"     "$XCFW_OUT/ios-arm64/$FRAMEWORK_NAME.framework/"
  cp -R "$TARGET_DIR/frameworks/ios-sim/$FRAMEWORK_NAME.framework/" "$XCFW_OUT/ios-arm64_x86_64-simulator/$FRAMEWORK_NAME.framework/"
  cp -R "$TARGET_DIR/frameworks/ios-macabi/$FRAMEWORK_NAME.framework/" "$XCFW_OUT/ios-arm64_x86_64-maccatalyst/$FRAMEWORK_NAME.framework/"
  cp -R "$TARGET_DIR/frameworks/macos/$FRAMEWORK_NAME.framework/"   "$XCFW_OUT/macos-arm64_x86_64/$FRAMEWORK_NAME.framework/"

  cat > "$XCFW_OUT/Info.plist" << 'XCINFO'
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
            <string>ios-arm64</string>
            <key>LibraryPath</key>
            <string>MeshLLMFFI.framework</string>
            <key>SupportedArchitectures</key>
            <array><string>arm64</string></array>
            <key>SupportedPlatform</key>
            <string>ios</string>
        </dict>
        <dict>
            <key>BinaryPath</key>
            <string>MeshLLMFFI.framework/MeshLLMFFI</string>
            <key>LibraryIdentifier</key>
            <string>ios-arm64_x86_64-simulator</string>
            <key>LibraryPath</key>
            <string>MeshLLMFFI.framework</string>
            <key>SupportedArchitectures</key>
            <array><string>arm64</string><string>x86_64</string></array>
            <key>SupportedPlatform</key>
            <string>ios</string>
            <key>SupportedPlatformVariant</key>
            <string>simulator</string>
        </dict>
        <dict>
            <key>BinaryPath</key>
            <string>MeshLLMFFI.framework/MeshLLMFFI</string>
            <key>LibraryIdentifier</key>
            <string>ios-arm64_x86_64-maccatalyst</string>
            <key>LibraryPath</key>
            <string>MeshLLMFFI.framework</string>
            <key>SupportedArchitectures</key>
            <array><string>arm64</string><string>x86_64</string></array>
            <key>SupportedPlatform</key>
            <string>ios</string>
            <key>SupportedPlatformVariant</key>
            <string>maccatalyst</string>
        </dict>
        <dict>
            <key>BinaryPath</key>
            <string>MeshLLMFFI.framework/MeshLLMFFI</string>
            <key>LibraryIdentifier</key>
            <string>macos-arm64_x86_64</string>
            <key>LibraryPath</key>
            <string>MeshLLMFFI.framework</string>
            <key>SupportedArchitectures</key>
            <array><string>arm64</string><string>x86_64</string></array>
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
fi

echo ""
echo "XCFramework created at: $XCFW_OUT"

echo "Verifying PrivacyInfo.xcprivacy embedding..."
PRIVACY_COUNT=$(find "$XCFW_OUT" -name "PrivacyInfo.xcprivacy" | wc -l | tr -d ' ')
echo "Found $PRIVACY_COUNT PrivacyInfo.xcprivacy file(s) inside XCFramework"
if [ "$PRIVACY_COUNT" -lt 1 ]; then
  echo "ERROR: PrivacyInfo.xcprivacy not embedded in XCFramework!"
  exit 1
fi

echo ""
echo "Build complete!"
