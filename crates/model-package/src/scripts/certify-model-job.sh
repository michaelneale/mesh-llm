#!/bin/bash
set -euo pipefail

# This script runs inside an HF Job container.
# It clones mesh-llm, builds pinned llama.cpp/skippy correctness tools, runs
# family certification against a mounted GGUF repo, and uploads artifacts.
#
# Environment variables:
#   SOURCE_REPO, SOURCE_FILE, MODEL_ID, FAMILY
#   MESH_LLM_REF — git ref to build from (default: main)
#   SOURCE_REVISION — source model revision (default: main)
#   ARTIFACT_REPO — optional dataset repo for certification artifacts
#   HF_TOKEN — injected as a secret when artifact upload is requested
#
# Optional family-certify controls:
#   RUN_ID, LAYER_END, SPLIT_LAYER, SPLITS, ACTIVATION_WIDTH, PROMPT, CTX_SIZE
#   N_GPU_LAYERS, WIRE_DTYPE, WIRE_DTYPES, STARTUP_TIMEOUT_SECS
#   ALLOW_MISMATCH, STRICT_DTYPE, SKIP_CORRECTNESS, SKIP_DTYPE, SKIP_STATE
#   PREFIX_TOKEN_COUNT, CACHE_HIT_REPEATS, BORROW_RESIDENT_HITS
#
# Volumes:
#   /source  — source GGUF repo (read-only mount)
#   /bucket  — writable storage bucket for script + certification workspace

MESH_LLM_REF="${MESH_LLM_REF:-main}"
SOURCE_REVISION="${SOURCE_REVISION:-main}"
PREBUILT_MESH_LLM_ROOT="${PREBUILT_MESH_LLM_ROOT:-/opt/mesh-llm}"
RUN_ID="${RUN_ID:-hf-cert-$(date +%Y%m%d-%H%M%S)}"
JOB_WORK_ROOT="${JOB_WORK_ROOT:-/bucket/job-work}"
MESH_ROOT=""
LLAMA_BUILD_DIR=""
PREBUILT_TOOLS=0
PYTHON_BIN=""
if [ -z "${JOB_WORK_DIR:-}" ]; then
    SAFE_FAMILY="$(printf '%s' "$FAMILY" | tr -c '[:alnum:]._-' '_')"
    JOB_WORK_DIR="${JOB_WORK_ROOT}/cert-${SAFE_FAMILY}-$(date +%Y%m%d%H%M%S)-$$"
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-true}"
else
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-false}"
fi
CERT_ROOT="${CERT_ROOT:-${JOB_WORK_DIR}/family-certify}"
export JOB_WORK_DIR
export RUN_ID CERT_ROOT

use_prebuilt_mesh_llm() {
    if [ ! -d "$PREBUILT_MESH_LLM_ROOT" ] || [ ! -f "$PREBUILT_MESH_LLM_ROOT/.mesh-llm-ref" ]; then
        return 1
    fi

    local image_ref
    image_ref="$(cat "$PREBUILT_MESH_LLM_ROOT/.mesh-llm-ref")"
    if [ "$MESH_LLM_REF" != "main" ] && [ "$MESH_LLM_REF" != "$image_ref" ]; then
        if [ "${ALLOW_IMAGE_REF_MISMATCH:-false}" != "true" ] && [ "${ALLOW_IMAGE_REF_MISMATCH:-0}" != "1" ]; then
            echo "Prebuilt image ref $image_ref does not match requested mesh-llm ref $MESH_LLM_REF; falling back to in-job build."
            return 1
        fi
        echo "WARNING: using prebuilt image ref $image_ref for requested mesh-llm ref $MESH_LLM_REF because ALLOW_IMAGE_REF_MISMATCH is set."
    fi

    MESH_ROOT="$PREBUILT_MESH_LLM_ROOT"
    LLAMA_BUILD_DIR="$MESH_ROOT/.deps/llama-build/build-stage-abi-cpu"
    if [ ! -x "$MESH_ROOT/target/debug/skippy-correctness" ] || \
       [ ! -x "$MESH_ROOT/target/debug/skippy-server" ] || \
       [ ! -x "$MESH_ROOT/target/debug/llama-spec-bench" ]; then
        echo "Prebuilt image is missing correctness binaries; falling back to in-job build."
        return 1
    fi
    if [ ! -d "$LLAMA_BUILD_DIR" ]; then
        echo "Prebuilt image is missing llama build dir $LLAMA_BUILD_DIR; falling back to in-job build."
        return 1
    fi

    if [ -f /root/.cargo/env ]; then
        source /root/.cargo/env
    fi
    cd "$MESH_ROOT"
    PREBUILT_TOOLS=1
    echo "  ✓ Using prebuilt HF jobs image tools from $MESH_ROOT (image ref $image_ref)"
    return 0
}

ensure_hf_python() {
    if [ -n "$PYTHON_BIN" ]; then
        return
    fi
    if python3 - <<'PYTHON' >/dev/null 2>&1
import huggingface_hub
PYTHON
    then
        PYTHON_BIN=python3
    else
        python3 -m venv /tmp/venv > /dev/null
        /tmp/venv/bin/pip install -q huggingface_hub
        PYTHON_BIN=/tmp/venv/bin/python3
    fi
}

cleanup_job_work_dir() {
    local exit_code=$?
    if [ "${CLEANUP_JOB_WORK_DIR}" = "true" ] && [ -n "${JOB_WORK_DIR:-}" ]; then
        if [ "$exit_code" -eq 0 ]; then
            echo "Cleaning job work dir: ${JOB_WORK_DIR}"
            rm -rf "$JOB_WORK_DIR"
        else
            echo "Preserving job work dir after exit ${exit_code}: ${JOB_WORK_DIR}"
        fi
    fi
}
trap cleanup_job_work_dir EXIT

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Family Certification Job                               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Source: ${SOURCE_REPO}/${SOURCE_FILE}"
echo "║  Model:  ${MODEL_ID}"
echo "║  Family: ${FAMILY}"
echo "║  Build:  mesh-llm @ ${MESH_LLM_REF}"
echo "║  Run:    ${RUN_ID}"
echo "║  Work:   ${JOB_WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if use_prebuilt_mesh_llm; then
    echo "=== [1/8] Using prebuilt HF Jobs image ==="
    echo "  Build dependencies, Rust, llama.cpp ABI, and correctness binaries are already present."
else
    echo "=== [1/8] Installing build dependencies ==="
    apt-get update -qq && apt-get install -y -qq \
        cmake git curl build-essential pkg-config libssl-dev \
        python3-pip python3-venv jq > /dev/null 2>&1

    echo "=== [2/8] Installing Rust ==="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
    source /root/.cargo/env

    echo "=== [3/8] Cloning mesh-llm ==="
    git clone --filter=blob:none https://github.com/Mesh-LLM/mesh-llm.git /tmp/build
    cd /tmp/build
    if git ls-remote --exit-code --heads origin "$MESH_LLM_REF" >/dev/null 2>&1 || \
       git ls-remote --exit-code --tags origin "$MESH_LLM_REF" >/dev/null 2>&1; then
        git fetch --depth 1 origin "$MESH_LLM_REF"
        git checkout --detach FETCH_HEAD
    else
        git fetch --depth 1 origin "$MESH_LLM_REF"
        git checkout --detach FETCH_HEAD
    fi

    # Full clone needed for git-am patches in prepare-llama.
    sed -i 's/--filter=blob:none //' scripts/prepare-llama.sh

    echo "=== [4/8] Building pinned llama.cpp CPU ABI ==="
    scripts/prepare-llama.sh pinned 2>&1 | tail -20
    MESH_ROOT=/tmp/build
    LLAMA_BUILD_DIR="/tmp/build/.deps/llama-build/build-stage-abi-cpu"
    LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" scripts/build-llama.sh 2>&1 | tail -20
fi

echo "=== [5/8] Verifying source model ==="
SOURCE_PATH="/source/${SOURCE_FILE}"
if [ ! -f "$SOURCE_PATH" ]; then
    echo "ERROR: Source file not found at $SOURCE_PATH"
    echo ""
    echo "Available GGUF files in /source:"
    find /source -name "*.gguf" -type f | sort | head -30
    exit 1
fi
echo "  Source: $SOURCE_PATH ($(du -h "$SOURCE_PATH" | cut -f1))"

echo "=== [6/8] Running family certification ==="
CERT_ARGS=(
    --family "$FAMILY"
    --target-model "$SOURCE_PATH"
    --model-id "$MODEL_ID"
    --cert-root "$CERT_ROOT"
    --run-id "$RUN_ID"
)

append_opt() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"
    if [ -n "$value" ]; then
        CERT_ARGS+=("$flag" "$value")
    fi
}

append_bool() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"
    if [ "$value" = "true" ] || [ "$value" = "1" ]; then
        CERT_ARGS+=("$flag")
    fi
}

append_opt LAYER_END --layer-end
append_opt SPLIT_LAYER --split-layer
append_opt SPLITS --splits
append_opt ACTIVATION_WIDTH --activation-width
append_opt PROMPT --prompt
append_opt CTX_SIZE --ctx-size
append_opt N_GPU_LAYERS --n-gpu-layers
append_opt STARTUP_TIMEOUT_SECS --startup-timeout-secs
append_opt WIRE_DTYPE --wire-dtype
append_opt WIRE_DTYPES --wire-dtypes
append_opt PREFIX_TOKEN_COUNT --prefix-token-count
append_opt CACHE_HIT_REPEATS --cache-hit-repeats
append_bool ALLOW_MISMATCH --allow-mismatch
append_bool STRICT_DTYPE --strict-dtype
append_bool SKIP_CORRECTNESS --skip-correctness
append_bool SKIP_DTYPE --skip-dtype
append_bool SKIP_STATE --skip-state
append_bool BORROW_RESIDENT_HITS --borrow-resident-hits
if [ "$PREBUILT_TOOLS" = "1" ]; then
    CERT_ARGS+=(--skip-build)
fi

CERT_EXIT_CODE=0
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" scripts/family-certify.sh "${CERT_ARGS[@]}" || CERT_EXIT_CODE=$?
if [ "$CERT_EXIT_CODE" -ne 0 ]; then
    echo "  ! Certification command exited ${CERT_EXIT_CODE}; publishing available artifacts before failing the job."
fi

ARTIFACT_DIR="$(find "$CERT_ROOT/$RUN_ID" -mindepth 2 -maxdepth 2 -type d | head -1)"
if [ -z "$ARTIFACT_DIR" ] || [ ! -d "$ARTIFACT_DIR" ]; then
    echo "ERROR: certification artifact directory not found under $CERT_ROOT/$RUN_ID"
    exit 1
fi
export ARTIFACT_DIR
echo "  ✓ Artifacts: $ARTIFACT_DIR"

echo "=== [7/8] Uploading artifacts ==="
if [ -z "${ARTIFACT_REPO:-}" ]; then
    echo "  ARTIFACT_REPO not set; leaving artifacts in the job workspace."
else
    ensure_hf_python
    "$PYTHON_BIN" << PYTHON
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
artifact_repo = os.environ["ARTIFACT_REPO"]
run_id = os.environ["RUN_ID"]
family = os.environ["FAMILY"]
model_id = os.environ["MODEL_ID"]
artifact_dir = os.environ["ARTIFACT_DIR"]

api.create_repo(artifact_repo, repo_type="dataset", exist_ok=True)
path_in_repo = f"runs/{run_id}"
api.upload_folder(
    repo_id=artifact_repo,
    repo_type="dataset",
    folder_path=artifact_dir,
    path_in_repo=path_in_repo,
    commit_message=f"Certification artifacts for {family} ({model_id})",
)
print(f"  ✓ Published: https://huggingface.co/datasets/{artifact_repo}/tree/main/{path_in_repo}")
PYTHON
fi

echo "=== [8/8] Done ==="
if [ "$CERT_EXIT_CODE" -ne 0 ]; then
    echo "Certification run failed for ${MODEL_ID} (${FAMILY}); artifacts were preserved for diagnosis."
    exit "$CERT_EXIT_CODE"
fi

echo "Certification run completed for ${MODEL_ID} (${FAMILY})"
