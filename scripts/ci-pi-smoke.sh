#!/usr/bin/env bash
# ci-pi-smoke.sh - run Pi against a live mesh OpenAI-compatible endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci-agent-live-fixture-lib.sh
source "${SCRIPT_DIR}/ci-agent-live-fixture-lib.sh"

if ! command -v pi >/dev/null 2>&1; then
    echo "pi is not installed or is not on PATH" >&2
    exit 1
fi

RAW_BASE_URL="${MESH_PI_BASE_URL:-${MESH_AGENT_BASE_URL:-${MESH_OPENCODE_BASE_URL:-}}}"
if [[ -z "$RAW_BASE_URL" && -n "${MESH_CLIENT_API_BASE:-}" ]]; then
    RAW_BASE_URL="${MESH_CLIENT_API_BASE%/}/v1"
fi
if [[ -z "$RAW_BASE_URL" ]]; then
    echo "Set MESH_AGENT_BASE_URL or MESH_PI_BASE_URL to a mesh /v1 endpoint" >&2
    exit 1
fi

BASE_URL="$(agent_smoke_normalize_v1_base "$RAW_BASE_URL")"
MODEL="$(agent_smoke_pick_model "$BASE_URL" "${MESH_PI_MODEL:-${MESH_AGENT_MODEL:-${MESH_OPENCODE_MODEL:-}}}")"
if [[ -z "$MODEL" ]]; then
    echo "Mesh endpoint returned no models from ${BASE_URL%/}/models" >&2
    exit 1
fi

TIMEOUT_SECONDS="${PI_SMOKE_TIMEOUT:-360}"
WORK_DIR="${PI_SMOKE_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-pi-smoke.XXXXXX")}"
OUTPUT_JSONL="${PI_SMOKE_OUTPUT:-${WORK_DIR}/pi-output.jsonl}"
ERROR_LOG="${PI_SMOKE_ERROR_LOG:-${WORK_DIR}/pi-stderr.log}"
PROMPT="$(agent_smoke_prompt)"
INITIAL_IMPL_SHA="$(agent_smoke_write_fixture "$WORK_DIR")"

export HOME="${WORK_DIR}/home"
mkdir -p "${HOME}/.pi/agent"
python3 - "$BASE_URL" "$MODEL" "${HOME}/.pi/agent/models.json" <<'PY'
import json
import sys
from pathlib import Path

base_url, model, path = sys.argv[1:4]
config = {
    "providers": {
        "mesh": {
            "api": "openai-completions",
            "apiKey": "mesh",
            "baseUrl": base_url.rstrip("/"),
            "compat": {
                "supportsStore": False,
                "supportsDeveloperRole": False,
                "supportsUsageInStreaming": True,
            },
            "models": [
                {
                    "id": model,
                    "name": model,
                    "contextWindow": 32768,
                    "maxTokens": 4096,
                }
            ],
        }
    }
}
Path(path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

echo "=== CI Pi Live Smoke ==="
echo "  mesh:     ${BASE_URL%/}"
echo "  model:    mesh/${MODEL}"
echo "  pi:       $(pi --version 2>/dev/null || echo unknown)"
echo "  work dir: ${WORK_DIR}"
echo "  output:   ${OUTPUT_JSONL}"

agent_smoke_long_prompt_soak "$BASE_URL" "$MODEL" "$WORK_DIR" "Pi"

if command -v timeout >/dev/null 2>&1; then
    PI_COMMAND=(timeout "${TIMEOUT_SECONDS}" pi)
else
    PI_COMMAND=(pi)
fi

if ! (
    cd "$WORK_DIR"
    "${PI_COMMAND[@]}" --mode json --print --model "mesh/${MODEL}" "$PROMPT" >"$OUTPUT_JSONL" 2>"$ERROR_LOG"
); then
    echo "Pi smoke failed" >&2
    echo "--- pi stderr ---" >&2
    tail -160 "$ERROR_LOG" >&2 || true
    echo "--- pi output ---" >&2
    tail -160 "$OUTPUT_JSONL" >&2 || true
    exit 1
fi

agent_smoke_validate_fixture "$WORK_DIR" "$INITIAL_IMPL_SHA" "$OUTPUT_JSONL" "Pi" true
