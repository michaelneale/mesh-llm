# Releasing mesh-llm

## Prerequisites

- `just` installed (`brew install just`)
- `cmake` installed (`brew install cmake`)
- `cargo` installed (packaged with rust)
- `gh` CLI authenticated (`gh auth status`)
- llama.cpp fork cloned (`just build` does this automatically)

## Steps

### 1. Build everything fresh

```bash
just build
```

This clones/updates the llama.cpp fork if needed, builds with `-DGGML_METAL=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF`, and builds the Rust mesh-llm binary.

### 2. Verify no homebrew dependencies

```bash
otool -L llama.cpp/build/bin/llama-server | grep -v /System | grep -v /usr/lib
otool -L llama.cpp/build/bin/rpc-server | grep -v /System | grep -v /usr/lib
otool -L target/release/mesh-llm | grep -v /System | grep -v /usr/lib
```

Each should only show the binary name — no `/opt/homebrew/` paths.

### 3. Create the bundle

```bash
just bundle
```

Creates `/tmp/mesh-bundle.tar.gz` containing `mesh-llm`, `rpc-server`, `llama-server`.

### 4. Smoke test the bundle

```bash
mkdir /tmp/test-bundle && tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
# Should download model, start solo, API on :9337, console on :3131
# Hit http://localhost:9337/v1/chat/completions to verify inference works
# Ctrl+C to stop
rm -rf /tmp/test-bundle
```

### 5. Commit, tag, push

```bash
just release-version v0.X.0
git add -A && git commit -m "v0.X.0: <summary>"
git tag v0.X.0
git push origin main --tags
```

### 6. Let GitHub Actions build and publish the release

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which:

- builds release bundles on macOS, Linux, and Windows
- uploads versioned assets such as `mesh-llm-v0.X.0-aarch64-apple-darwin.tar.gz`
- uploads stable `latest` assets such as `mesh-llm-x86_64-unknown-linux-gnu.tar.gz`
- keeps the legacy macOS `mesh-bundle.tar.gz` asset for the README install one-liner
- creates the GitHub release automatically with generated notes

### 7. Verify the release assets

After the workflow finishes, verify:

- `mesh-bundle.tar.gz` still exists for macOS latest installs
- `mesh-llm-aarch64-apple-darwin.tar.gz` exists
- `mesh-llm-x86_64-unknown-linux-gnu.tar.gz` exists
- `mesh-llm-x86_64-pc-windows-msvc.zip` exists

## Notes

- The unversioned asset name `mesh-bundle.tar.gz` is still required for the README's macOS install one-liner.
- Linux and Windows release bundles are built on GitHub-hosted runners without CUDA, so they are generic CPU bundles. CUDA-specific Linux builds still need a source build.
- `codesign` and `xattr` may be needed on the receiving machine if macOS Gatekeeper blocks unsigned binaries:
  ```bash
  codesign -s - /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server
  xattr -cr /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server
  ```
