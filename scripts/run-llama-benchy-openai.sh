#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:9337/v1}"
API_KEY="${API_KEY:-EMPTY}"
LLAMA_BENCHY_FROM="${LLAMA_BENCHY_FROM:-git+https://github.com/eugr/llama-benchy}"

MODEL="${MODEL:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
TOKENIZER="${TOKENIZER:-}"

PP="${PP:-128}"
TG="${TG:-16}"
DEPTH="${DEPTH:-0}"
RUNS="${RUNS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
LATENCY_MODE="${LATENCY_MODE:-generation}"
FORMAT="${FORMAT:-md}"
SAVE_RESULT="${SAVE_RESULT:-}"
SKIP_COHERENCE="${SKIP_COHERENCE:-1}"
NO_ADAPT_PROMPT="${NO_ADAPT_PROMPT:-0}"
NO_CACHE="${NO_CACHE:-0}"

discover_model() {
  python3 - "$BASE_URL" "$API_KEY" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
api_key = sys.argv[2]
request = urllib.request.Request(
    f"{base_url}/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    payload = json.load(response)
data = payload.get("data") or []
if not data:
    raise SystemExit("no models returned from /v1/models")
print(data[0]["id"])
PY
}

if [[ -z "$MODEL" ]]; then
  MODEL="$(discover_model)"
fi

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$MODEL"
fi

read -r -a pp_values <<<"$PP"
read -r -a tg_values <<<"$TG"
read -r -a depth_values <<<"$DEPTH"
read -r -a concurrency_values <<<"$CONCURRENCY"

cmd=(
  uvx
  --from "$LLAMA_BENCHY_FROM"
  llama-benchy
  --base-url "$BASE_URL"
  --api-key "$API_KEY"
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --pp "${pp_values[@]}"
  --tg "${tg_values[@]}"
  --depth "${depth_values[@]}"
  --runs "$RUNS"
  --concurrency "${concurrency_values[@]}"
  --latency-mode "$LATENCY_MODE"
  --format "$FORMAT"
)

if [[ -n "$TOKENIZER" ]]; then
  cmd+=(--tokenizer "$TOKENIZER")
fi
if [[ -n "$SAVE_RESULT" ]]; then
  cmd+=(--save-result "$SAVE_RESULT")
fi
if [[ "$SKIP_COHERENCE" == "1" ]]; then
  cmd+=(--skip-coherence)
fi
if [[ "$NO_ADAPT_PROMPT" == "1" ]]; then
  cmd+=(--no-adapt-prompt)
fi
if [[ "$NO_CACHE" == "1" ]]; then
  cmd+=(--no-cache)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
