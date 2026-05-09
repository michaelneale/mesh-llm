#!/usr/bin/env bash
set -euo pipefail

LLAMA_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-.deps/llama-build/build-stage-abi-static}"
MODEL_REPO="${MODEL_REPO:-jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF}"
MODEL_FILE="${MODEL_FILE:-SmolLM2-135M-Instruct.Q4_K_M.gguf}"
MODEL_SELECTOR="${MODEL_SELECTOR:-Q4_K_M}"
MODEL_ID="${MODEL_ID:-${MODEL_REPO}:${MODEL_SELECTOR}}"
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER="${TOKENIZER:-HuggingFaceTB/SmolLM2-135M-Instruct}"
WORK_DIR="${WORK_DIR:-/tmp/skippy-openai-smoke}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9337}"
BASE_URL="http://${HOST}:${PORT}/v1"
CTX_SIZE="${CTX_SIZE:-256}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-2}"
RUN_BENCHY="${RUN_BENCHY:-0}"
KEEP_SERVER="${KEEP_SERVER:-0}"

SERVER_PID=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    if [[ "$KEEP_SERVER" == "1" ]]; then
      echo "leaving serve-openai running on ${BASE_URL} with pid ${SERVER_PID}"
    else
      kill "$SERVER_PID" >/dev/null 2>&1 || true
      wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

require_cmd curl
require_cmd jq
require_cmd lsof
require_cmd python3

if [[ ! -d "$LLAMA_BUILD_DIR" ]]; then
  echo "llama build dir not found: $LLAMA_BUILD_DIR" >&2
  echo "run: just build" >&2
  exit 1
fi

mkdir -p "$WORK_DIR/model"

if [[ -z "$MODEL_PATH" ]]; then
  require_cmd hf
  echo "downloading ${MODEL_REPO}/${MODEL_FILE} into ${WORK_DIR}/model"
  MODEL_PATH="$(hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$WORK_DIR/model" | sed -n 's/^path=//p' | tail -n 1)"
  if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="${WORK_DIR}/model/${MODEL_FILE}"
  fi
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model path not found: $MODEL_PATH" >&2
  exit 1
fi

echo "building skippy-server and skippy-model-package"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" cargo build -p skippy-server -p skippy-model-package

echo "inferring layer_end from $MODEL_PATH"
LAYER_END="$(
  LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" target/debug/skippy-model-package inspect "$MODEL_PATH" \
    | jq '[.tensors[] | select(.role == "layer") | .layer_index] | max + 1'
)"
if [[ -z "$LAYER_END" || "$LAYER_END" == "null" ]]; then
  echo "failed to infer layer_end from model" >&2
  exit 1
fi

CONFIG_PATH="${WORK_DIR}/stage-openai-smoke.json"
python3 - "$CONFIG_PATH" "$MODEL_ID" "$MODEL_PATH" "$LAYER_END" "$CTX_SIZE" <<'PY'
import json
import sys

config_path, model_id, model_path, layer_end, ctx_size = sys.argv[1:]
config = {
    "run_id": "openai-smoke",
    "topology_id": "openai-smoke-single-stage",
    "model_id": model_id,
    "model_path": model_path,
    "stage_id": "stage-0",
    "stage_index": 0,
    "layer_start": 0,
    "layer_end": int(layer_end),
    "ctx_size": int(ctx_size),
    "n_gpu_layers": 0,
    "filter_tensors_on_load": False,
    "load_mode": "runtime-slice",
    "bind_addr": "127.0.0.1:19000",
    "upstream": None,
    "downstream": None,
    "kv_server": None,
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port ${PORT} is already listening" >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
  exit 1
fi

SERVER_LOG="${WORK_DIR}/serve-openai.log"
echo "starting serve-openai on ${BASE_URL}"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  target/debug/skippy-server serve-openai \
    --config "$CONFIG_PATH" \
    --bind-addr "${HOST}:${PORT}" \
    --default-max-tokens "$DEFAULT_MAX_TOKENS" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

for _ in {1..120}; do
  if curl -fsS --max-time 1 "${BASE_URL}/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "serve-openai exited early; log follows" >&2
    sed -n '1,220p' "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl -fsS --max-time 2 "${BASE_URL}/models" >/dev/null; then
  echo "serve-openai did not become ready; log follows" >&2
  sed -n '1,220p' "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "probing /v1/models"
models_json="$(curl -fsS --max-time 10 "${BASE_URL}/models")"
echo "$models_json" | jq -e --arg model "$MODEL_ID" '.object == "list" and (.data[]?.id == $model)' >/dev/null

chat_request="$(jq -cn --arg model "$MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Say hi"}],
  max_tokens: 2
}')"

echo "probing non-stream chat completion"
chat_json="$(curl -fsS --max-time 30 "${BASE_URL}/chat/completions" -H 'content-type: application/json' -d "$chat_request")"
echo "$chat_json" | jq -e '
  .object == "chat.completion"
  and .choices[0].message.role == "assistant"
  and .choices[0].finish_reason == "length"
  and .usage.prompt_tokens > 0
  and .usage.completion_tokens == 2
  and .usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)
' >/dev/null

echo "probing stream chat completion"
stream_request="$(jq -cn --arg model "$MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Say hi"}],
  stream: true,
  stream_options: {include_usage: true},
  max_tokens: 2
}')"
stream_out="${WORK_DIR}/stream.out"
curl -fsS --max-time 30 -N "${BASE_URL}/chat/completions" \
  -H 'content-type: application/json' \
  -d "$stream_request" >"$stream_out"
grep -q '"role":"assistant"' "$stream_out"
grep -q '"usage":{"prompt_tokens":' "$stream_out"
grep -q '"finish_reason":"length"' "$stream_out"
grep -q 'data: \[DONE\]' "$stream_out"

echo "probing supported sampling controls"
sampling_request="$(jq -cn --arg model "$MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Say hi"}],
  temperature: 0.7,
  top_p: 0.9,
  seed: 42,
  logit_bias: {"0": -1.0},
  max_tokens: 2
}')"
sampling_response="${WORK_DIR}/sampling-response.json"
sampling_status="$(
  curl -sS --max-time 10 \
    -o "$sampling_response" \
    -w '%{http_code}' \
    "${BASE_URL}/chat/completions" \
    -H 'content-type: application/json' \
    -d "$sampling_request"
)"
if [[ "$sampling_status" != "200" ]]; then
  echo "expected supported sampling to return HTTP 200, got ${sampling_status}" >&2
  cat "$sampling_response" >&2 || true
  exit 1
fi
sampling_json="$(cat "$sampling_response")"
echo "$sampling_json" | jq -e '.choices[0].finish_reason == "length"' >/dev/null

echo "probing unsupported sampling rejection"
unsupported_sampling_request="$(jq -cn --arg model "$MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Say hi"}],
  min_p: 0.1,
  max_tokens: 2
}')"
unsupported_sampling_response="${WORK_DIR}/unsupported-sampling-response.json"
unsupported_sampling_status="$(
  curl -sS --max-time 10 \
    -o "$unsupported_sampling_response" \
    -w '%{http_code}' \
    "${BASE_URL}/chat/completions" \
    -H 'content-type: application/json' \
    -d "$unsupported_sampling_request"
)"
if [[ "$unsupported_sampling_status" != "400" ]]; then
  echo "expected unsupported sampling to return HTTP 400, got ${unsupported_sampling_status}" >&2
  cat "$unsupported_sampling_response" >&2 || true
  exit 1
fi
unsupported_sampling_json="$(cat "$unsupported_sampling_response")"
echo "$unsupported_sampling_json" | jq -e '.error.code == "unsupported_model_feature"' >/dev/null

if [[ "$RUN_BENCHY" == "1" ]]; then
  require_cmd uvx
  echo "running llama-benchy smoke"
  BASE_URL="$BASE_URL" \
    MODEL="$MODEL_ID" \
    TOKENIZER="$TOKENIZER" \
    PP="${PP:-8}" \
    TG="${TG:-2}" \
    RUNS="${RUNS:-1}" \
    DEPTH="${DEPTH:-0}" \
    CONCURRENCY="${CONCURRENCY:-1}" \
    SAVE_RESULT="${SAVE_RESULT:-$WORK_DIR/benchy-smoke.md}" \
    scripts/run-llama-benchy-openai.sh
fi

echo "OpenAI smoke passed"
echo "  config: $CONFIG_PATH"
echo "  server log: $SERVER_LOG"
if [[ "$RUN_BENCHY" == "1" ]]; then
  echo "  benchy result: ${SAVE_RESULT:-$WORK_DIR/benchy-smoke.md}"
fi
