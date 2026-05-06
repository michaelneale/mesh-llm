---
name: skippy-model-package
description: Use this skill when inspecting GGUF models, planning layer ranges, generating or validating skippy package artifacts, fake packages for direct GGUFs, materialized stage cache behavior, or GGUF writer integration.
metadata:
  short-description: Inspect and package GGUF stages
---

# skippy-model-package

Use this skill for model inspection, package planning, stage materialization,
and cache behavior.

## Ownership

Rust owns package manifests, topology planning inputs, cache policy, and mesh
model-storage integration. The patched llama/skippy ABI owns GGUF tensor
inspection and GGUF artifact writing.

Direct GGUF loading in mesh should materialize as a fake package identity in
the skippy runtime so the split-serving path can use the same package-backed
stage machinery as Hugging Face packages.

## Commands

Check current package names before running commands:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

Useful current checks in this repo:

```bash
cargo test -p skippy-runtime --lib
cargo test -p skippy-topology --lib
cargo test -p mesh-llm inference::skippy --lib
```

If standalone slicer/package CLI crates are imported later, prefer those
repo-native commands over ad hoc scripts.

## Cache Policy

Materialized stages are derived cache. Model storage commands may evict
materialized stage artifacts without deleting the source model/package. Preserve
pinned materialized artifacts unless the command explicitly asks for a stronger
cleanup.
