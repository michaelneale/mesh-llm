#!/usr/bin/env bash
# ci-hf-download-smoke.sh — exercise the Rust HuggingFace download pipeline.
#
# Runs the model-hf integration tests that hit real HuggingFace API and
# download endpoints. These tests verify the code path a user exercises
# when running `mesh-llm serve --model org/repo:Q4_K_M`.
#
# Usage:
#   scripts/ci-hf-download-smoke.sh
#
# Environment:
#   HF_TOKEN               — optional, speeds up API calls / avoids rate limits

set -euo pipefail

echo "=== CI HuggingFace Download Smoke ==="
echo "  rust toolchain: $(rustc --version 2>/dev/null || echo 'not found')"
echo "  os:             $(uname -s)"
echo ""

echo "Running model-hf integration tests (API-only: resolve, list, artifact resolution)..."
cargo test -p model-hf --test hf_download -- \
    --ignored \
    resolve_revision_returns_commit_sha \
    list_files_single_gguf_repo \
    list_files_split_gguf_repo \
    resolve_artifact_ref_single_gguf \
    resolve_artifact_ref_split_gguf \
    resolve_nonexistent_repo_returns_error

echo ""
echo "Running model-hf download tests (downloads ~100 MB GGUF via Rust HF client)..."
cargo test -p model-hf --test hf_download -- \
    --ignored \
    download_single_gguf_file \
    download_nonexistent_file_returns_error \
    full_resolve_download_identity_pipeline

echo ""
echo "HuggingFace download smoke passed"
