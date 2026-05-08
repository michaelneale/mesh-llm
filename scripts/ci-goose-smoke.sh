#!/usr/bin/env bash
# ci-goose-smoke.sh - run Goose against a live mesh OpenAI-compatible endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci-agent-live-fixture-lib.sh
source "${SCRIPT_DIR}/ci-agent-live-fixture-lib.sh"

if ! command -v goose >/dev/null 2>&1; then
    echo "goose is not installed or is not on PATH" >&2
    exit 1
fi

RAW_BASE_URL="${MESH_GOOSE_BASE_URL:-${MESH_AGENT_BASE_URL:-${MESH_OPENCODE_BASE_URL:-}}}"
if [[ -z "$RAW_BASE_URL" && -n "${MESH_CLIENT_API_BASE:-}" ]]; then
    RAW_BASE_URL="${MESH_CLIENT_API_BASE%/}/v1"
fi
if [[ -z "$RAW_BASE_URL" ]]; then
    echo "Set MESH_AGENT_BASE_URL or MESH_GOOSE_BASE_URL to a mesh /v1 endpoint" >&2
    exit 1
fi

BASE_URL="$(agent_smoke_normalize_v1_base "$RAW_BASE_URL")"
MODEL="$(agent_smoke_pick_model "$BASE_URL" "${MESH_GOOSE_MODEL:-${MESH_AGENT_MODEL:-${MESH_OPENCODE_MODEL:-}}}")"
if [[ -z "$MODEL" ]]; then
    echo "Mesh endpoint returned no models from ${BASE_URL%/}/models" >&2
    exit 1
fi

TIMEOUT_SECONDS="${GOOSE_SMOKE_TIMEOUT:-360}"
WORK_DIR="${GOOSE_SMOKE_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-goose-smoke.XXXXXX")}"
OUTPUT_JSONL="${GOOSE_SMOKE_OUTPUT:-${WORK_DIR}/goose-output.jsonl}"
ERROR_LOG="${GOOSE_SMOKE_ERROR_LOG:-${WORK_DIR}/goose-stderr.log}"
PROMPT="$(agent_smoke_prompt)"
INITIAL_IMPL_SHA="$(agent_smoke_write_fixture "$WORK_DIR")"

export GOOSE_PATH_ROOT="${WORK_DIR}/goose-root"
export GOOSE_PROVIDER="mesh"
export GOOSE_MODEL="$MODEL"
export GOOSE_MODE="auto"
export GOOSE_DISABLE_KEYRING="true"
export GOOSE_PROVIDER_SKIP_BACKOFF="true"
export GOOSE_CLI_THEME="ansi"

mkdir -p "${GOOSE_PATH_ROOT}/config/custom_providers"
python3 - "$BASE_URL" "$MODEL" "${GOOSE_PATH_ROOT}/config/custom_providers/mesh.json" "${GOOSE_PATH_ROOT}/config/config.yaml" <<'PY'
import json
import sys
from pathlib import Path

base_url, model, provider_path, config_path = sys.argv[1:5]
provider = {
    "name": "mesh",
    "engine": "openai",
    "display_name": "mesh-llm",
    "description": "Distributed LLM inference via mesh-llm",
    "api_key_env": "",
    "base_url": base_url.rstrip("/"),
    "models": [{"name": model, "context_limit": 32768}],
    "timeout_seconds": 600,
    "supports_streaming": True,
    "requires_auth": False,
}
Path(provider_path).write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
Path(config_path).write_text(
    "GOOSE_PROVIDER: mesh\n"
    f"GOOSE_MODEL: {json.dumps(model)}\n"
    "GOOSE_MODE: auto\n"
    "GOOSE_DISABLE_KEYRING: true\n",
    encoding="utf-8",
)
PY

echo "=== CI Goose Live Smoke ==="
echo "  mesh:     ${BASE_URL%/}"
echo "  model:    ${MODEL}"
echo "  goose:    $(goose --version 2>/dev/null || echo unknown)"
echo "  work dir: ${WORK_DIR}"
echo "  output:   ${OUTPUT_JSONL}"

agent_smoke_long_prompt_soak "$BASE_URL" "$MODEL" "$WORK_DIR" "Goose"

if command -v timeout >/dev/null 2>&1; then
    GOOSE_COMMAND=(timeout "${TIMEOUT_SECONDS}" goose)
else
    GOOSE_COMMAND=(goose)
fi

if ! (
    cd "$WORK_DIR"
    "${GOOSE_COMMAND[@]}" run \
        --provider mesh \
        --model "$MODEL" \
        --with-builtin developer \
        --no-profile \
        --no-session \
        --max-turns 24 \
        --output-format stream-json \
        --text "$PROMPT" >"$OUTPUT_JSONL" 2>"$ERROR_LOG"
); then
    echo "Goose smoke failed" >&2
    echo "--- goose stderr ---" >&2
    tail -160 "$ERROR_LOG" >&2 || true
    echo "--- goose output ---" >&2
    tail -160 "$OUTPUT_JSONL" >&2 || true
    exit 1
fi

agent_smoke_validate_fixture "$WORK_DIR" "$INITIAL_IMPL_SHA" "$OUTPUT_JSONL" "Goose" true
