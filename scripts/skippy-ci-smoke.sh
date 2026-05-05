#!/usr/bin/env bash
set -euo pipefail

LLAMA_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-.deps/llama.cpp/build-stage-abi-static}"
WORK_DIR="${WORK_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/skippy-ci-smoke}"
REPORT_DIR="${WORK_DIR}/reports"
MODEL_DIR="${WORK_DIR}/models"

DENSE_MODEL_REPO="${DENSE_MODEL_REPO:-jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF}"
DENSE_MODEL_FILE="${DENSE_MODEL_FILE:-SmolLM2-135M-Instruct.Q4_K_M.gguf}"
DENSE_MODEL_SELECTOR="${DENSE_MODEL_SELECTOR:-Q4_K_M}"
DENSE_MODEL_ID="${DENSE_MODEL_ID:-${DENSE_MODEL_REPO}:${DENSE_MODEL_SELECTOR}}"
DENSE_MODEL_PATH="${DENSE_MODEL_PATH:-}"

RECURRENT_MODEL_REPO="${RECURRENT_MODEL_REPO:-tiiuae/Falcon-H1-0.5B-Instruct-GGUF}"
RECURRENT_MODEL_FILE="${RECURRENT_MODEL_FILE:-Falcon-H1-0.5B-Instruct-Q4_K_M.gguf}"
RECURRENT_MODEL_SELECTOR="${RECURRENT_MODEL_SELECTOR:-Q4_K_M}"
RECURRENT_MODEL_ID="${RECURRENT_MODEL_ID:-${RECURRENT_MODEL_REPO}:${RECURRENT_MODEL_SELECTOR}}"
RECURRENT_MODEL_PATH="${RECURRENT_MODEL_PATH:-}"

CTX_SIZE="${CTX_SIZE:-256}"
PROMPT_CTX_SIZE="${PROMPT_CTX_SIZE:-1536}"
STATE_PREFIX_TOKENS="${STATE_PREFIX_TOKENS:-128}"
PROMPT_PREFILL_CHUNK_SIZE="${PROMPT_PREFILL_CHUNK_SIZE:-128}"
PROMPT_MAX_NEW_TOKENS="${PROMPT_MAX_NEW_TOKENS:-32}"
STAGE_SERVER_BIN="${STAGE_SERVER_BIN:-target/debug/skippy-server}"

SERVER_PID=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

download_model() {
  local repo="$1"
  local file="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  echo "downloading ${repo}/${file}" >&2
  local output path
  output="$(hf download "$repo" "$file" --local-dir "$out_dir")"
  path="$(printf '%s\n' "$output" | sed -n 's/^path=//p' | tail -n 1)"
  if [[ -z "$path" ]]; then
    path="${out_dir}/${file}"
  fi
  if [[ ! -f "$path" ]]; then
    echo "downloaded model path not found: $path" >&2
    printf '%s\n' "$output" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

model_layer_end() {
  local model_path="$1"
  LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
    target/debug/skippy-model-package inspect "$model_path" \
      | jq -r '[.tensors[] | select(.role == "layer") | .layer_index] | max + 1'
}

pick_port() {
  python3 - <<'PY'
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local pid="$3"
  for _ in $(seq 1 120); do
    if python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=1):
    pass
PY
    then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      return 1
    fi
    sleep 1
  done
  return 1
}

assert_json() {
  local file="$1"
  local filter="$2"
  if ! jq -e "$filter" "$file" >/dev/null; then
    echo "JSON assertion failed for ${file}: ${filter}" >&2
    jq . "$file" >&2 || true
    exit 1
  fi
}

write_stage_config() {
  local config_path="$1"
  local model_id="$2"
  local model_path="$3"
  local layer_end="$4"
  local ctx_size="$5"
  local bind_addr="$6"
  local payload="$7"
  python3 - "$config_path" "$model_id" "$model_path" "$layer_end" "$ctx_size" "$bind_addr" "$payload" <<'PY'
import json
import sys

config_path, model_id, model_path, layer_end, ctx_size, bind_addr, payload = sys.argv[1:]
config = {
    "run_id": "skippy-ci-smoke",
    "topology_id": "skippy-ci-smoke-single-stage",
    "model_id": model_id,
    "model_path": model_path,
    "stage_id": "stage-0",
    "stage_index": 0,
    "layer_start": 0,
    "layer_end": int(layer_end),
    "ctx_size": int(ctx_size),
    "lane_count": 4,
    "n_gpu_layers": 0,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "filter_tensors_on_load": False,
    "load_mode": "runtime-slice",
    "bind_addr": bind_addr,
    "upstream": None,
    "downstream": None,
    "kv_cache": {
        "mode": "lookup-record",
        "payload": payload,
        "max_entries": 32,
        "max_bytes": 0,
        "min_tokens": 64,
        "shared_prefix_stride_tokens": 128,
        "shared_prefix_record_limit": 2,
    },
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY
}

make_long_prompt_file() {
  local path="$1"
  python3 - "$path" <<'PY'
import sys

path = sys.argv[1]
sentence = (
    "We are validating exact prefix cache reuse in the binary serving path with "
    "a deterministic long prompt, stable wording, and enough repeated context "
    "to cross the restore threshold without depending on model creativity."
)
prompt = "Summarize this cache smoke paragraph in one short sentence. " + " ".join(
    f"{i:03d}. {sentence}" for i in range(30)
)
with open(path, "w", encoding="utf-8") as handle:
    handle.write(":noappend\n")
    handle.write(prompt)
    handle.write("\n")
    handle.write(prompt)
    handle.write("\n:append\n")
    handle.write("Reply with exactly the single word OK and then stop.\n")
    handle.write("Reply with exactly the single word OK again and then stop.\n")
    handle.write(":quit\n")
PY
}

require_cmd jq
require_cmd python3
require_cmd hf
require_cmd curl

if [[ ! -d "$LLAMA_BUILD_DIR" ]]; then
  echo "llama build dir not found: $LLAMA_BUILD_DIR" >&2
  echo "run scripts/prepare-llama.sh pinned && scripts/build-llama.sh first" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR" "$MODEL_DIR"

if [[ -z "$DENSE_MODEL_PATH" ]]; then
  DENSE_MODEL_PATH="$(download_model "$DENSE_MODEL_REPO" "$DENSE_MODEL_FILE" "${MODEL_DIR}/dense")"
fi
if [[ -z "$RECURRENT_MODEL_PATH" ]]; then
  RECURRENT_MODEL_PATH="$(download_model "$RECURRENT_MODEL_REPO" "$RECURRENT_MODEL_FILE" "${MODEL_DIR}/recurrent")"
fi

echo "building skippy smoke binaries"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  cargo build -p skippy-server -p skippy-correctness -p skippy-model-package -p skippy-prompt

DENSE_LAYER_END="$(model_layer_end "$DENSE_MODEL_PATH")"
RECURRENT_LAYER_END="$(model_layer_end "$RECURRENT_MODEL_PATH")"
if [[ -z "$DENSE_LAYER_END" || "$DENSE_LAYER_END" == "null" || "$DENSE_LAYER_END" -lt 3 ]]; then
  echo "failed to infer dense layer count from $DENSE_MODEL_PATH" >&2
  exit 1
fi
if [[ -z "$RECURRENT_LAYER_END" || "$RECURRENT_LAYER_END" == "null" || "$RECURRENT_LAYER_END" -lt 1 ]]; then
  echo "failed to infer recurrent layer count from $RECURRENT_MODEL_PATH" >&2
  exit 1
fi

DENSE_SPLIT_1="$((DENSE_LAYER_END / 3))"
DENSE_SPLIT_2="$(((DENSE_LAYER_END * 2) / 3))"
if [[ "$DENSE_SPLIT_1" -lt 1 ]]; then
  DENSE_SPLIT_1=1
fi
if [[ "$DENSE_SPLIT_2" -le "$DENSE_SPLIT_1" ]]; then
  DENSE_SPLIT_2=$((DENSE_SPLIT_1 + 1))
fi
if [[ "$DENSE_SPLIT_2" -ge "$DENSE_LAYER_END" ]]; then
  DENSE_SPLIT_2=$((DENSE_LAYER_END - 1))
fi

CHAIN_PORT_1="$(pick_port)"
CHAIN_PORT_2="$(pick_port)"
echo "smoke: dense 3-stage split ${DENSE_SPLIT_1},${DENSE_SPLIT_2} over ${DENSE_LAYER_END} layers"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  target/debug/skippy-correctness chain \
    --model "$DENSE_MODEL_PATH" \
    --model-id "$DENSE_MODEL_ID" \
    --layer-end "$DENSE_LAYER_END" \
    --ctx-size "$CTX_SIZE" \
    --prompt "Say hi in three words." \
    --splits "${DENSE_SPLIT_1},${DENSE_SPLIT_2}" \
    --stage1-bind-addr "127.0.0.1:${CHAIN_PORT_1}" \
    --stage2-bind-addr "127.0.0.1:${CHAIN_PORT_2}" \
    --stage-server-bin "$STAGE_SERVER_BIN" \
    --report-out "$REPORT_DIR/dense-chain.json"
assert_json "$REPORT_DIR/dense-chain.json" \
  '.matches == true and (.stages | length) == 3 and any(.stages[]; .forwarded_over_binary == true) and any(.stages[]; .returned_predicted_token == true)'

echo "smoke: dense ResidentKv cache hit correctness"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  target/debug/skippy-correctness state-handoff \
    --model "$DENSE_MODEL_PATH" \
    --model-id "$DENSE_MODEL_ID" \
    --layer-end "$DENSE_LAYER_END" \
    --ctx-size "$CTX_SIZE" \
    --prompt "Dense resident KV cache smoke." \
    --state-layer-start 0 \
    --state-layer-end "$DENSE_LAYER_END" \
    --state-stage-index 0 \
    --state-payload-kind resident-kv \
    --prefix-token-count "$STATE_PREFIX_TOKENS" \
    --cache-hit-repeats 2 \
    --runtime-lane-count 4 \
    --borrow-resident-hits \
    --report-out "$REPORT_DIR/dense-resident-kv.json"
assert_json "$REPORT_DIR/dense-resident-kv.json" \
  '.matches == true and .cache_hit_matches == true and .state_payload_kind == "resident-kv" and .borrowed_resident_hits == true and .cache_hit_repeats == 2 and ((.cache_storage_bytes // 0) > 0 or (.resident_state_bytes // 0) > 0)'

echo "smoke: recurrent KvRecurrent cache hit correctness"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  target/debug/skippy-correctness state-handoff \
    --model "$RECURRENT_MODEL_PATH" \
    --model-id "$RECURRENT_MODEL_ID" \
    --layer-end "$RECURRENT_LAYER_END" \
    --ctx-size "$CTX_SIZE" \
    --prompt "Recurrent KV cache smoke." \
    --state-layer-start 0 \
    --state-layer-end "$RECURRENT_LAYER_END" \
    --state-stage-index 0 \
    --state-payload-kind kv-recurrent \
    --prefix-token-count "$STATE_PREFIX_TOKENS" \
    --cache-hit-repeats 2 \
    --report-out "$REPORT_DIR/recurrent-kv-recurrent.json"
assert_json "$REPORT_DIR/recurrent-kv-recurrent.json" \
  '.matches == true and .cache_hit_matches == true and .state_payload_kind == "kv-recurrent" and .cache_hit_repeats == 2 and .state_bytes > 0 and .payload_digest.recurrent_bytes > 0 and .payload_digest.kv_bytes > 0'

PROMPT_PORT="$(pick_port)"
PROMPT_CONFIG="$WORK_DIR/prompt-stage.json"
PROMPT_LOG="$WORK_DIR/prompt-stage.log"
PROMPT_IN="$WORK_DIR/prompt-input.txt"
PROMPT_OUT="$WORK_DIR/prompt-output.log"
PROMPT_BIND="127.0.0.1:${PROMPT_PORT}"
write_stage_config "$PROMPT_CONFIG" "$DENSE_MODEL_ID" "$DENSE_MODEL_PATH" "$DENSE_LAYER_END" "$PROMPT_CTX_SIZE" "$PROMPT_BIND" "resident-kv"
make_long_prompt_file "$PROMPT_IN"

OPENAI_PORT="$(pick_port)"
OPENAI_STAGE_PORT="$(pick_port)"
OPENAI_CONFIG="$WORK_DIR/openai-stage.json"
OPENAI_LOG="$WORK_DIR/openai-server.log"
OPENAI_BASE_URL="http://127.0.0.1:${OPENAI_PORT}/v1"
write_stage_config "$OPENAI_CONFIG" "$DENSE_MODEL_ID" "$DENSE_MODEL_PATH" "$DENSE_LAYER_END" "$CTX_SIZE" "127.0.0.1:${OPENAI_STAGE_PORT}" "resident-kv"

echo "smoke: OpenAI /v1/chat/completions streaming/tools/logprobs/structured-output"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  "$STAGE_SERVER_BIN" serve-openai \
    --config "$OPENAI_CONFIG" \
    --bind-addr "127.0.0.1:${OPENAI_PORT}" \
    --default-max-tokens 2 \
    >"$OPENAI_LOG" 2>&1 &
SERVER_PID="$!"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 1 "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "OpenAI server exited early; log follows" >&2
    sed -n '1,260p' "$OPENAI_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if ! curl -fsS --max-time 2 "${OPENAI_BASE_URL}/models" >/dev/null; then
  echo "OpenAI server did not become ready; log follows" >&2
  sed -n '1,260p' "$OPENAI_LOG" >&2 || true
  exit 1
fi

openai_stream_request="$(jq -cn --arg model "$DENSE_MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Say hi"}],
  stream: true,
  stream_options: {include_usage: true},
  max_tokens: 2
}')"
openai_stream_out="$WORK_DIR/openai-stream.out"
curl -fsS --max-time 30 -N "${OPENAI_BASE_URL}/chat/completions" \
  -H 'content-type: application/json' \
  -d "$openai_stream_request" >"$openai_stream_out"
grep -q '"object":"chat.completion.chunk"' "$openai_stream_out"
grep -q '"usage":{"prompt_tokens":' "$openai_stream_out"
grep -q 'data: \[DONE\]' "$openai_stream_out"

openai_tools_request="$(jq -cn --arg model "$DENSE_MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Say hi"}],
  tools: [{
    type: "function",
    function: {
      name: "lookup",
      description: "Look up a value",
      parameters: {
        type: "object",
        properties: {city: {type: "string"}},
        required: ["city"]
      }
    }
  }],
  tool_choice: "auto",
  parallel_tool_calls: true,
  max_tokens: 2
}')"
openai_tools_response="$WORK_DIR/openai-tools.json"
openai_tools_status="$(
  curl -sS --max-time 30 \
    -o "$openai_tools_response" \
    -w '%{http_code}' \
    "${OPENAI_BASE_URL}/chat/completions" \
    -H 'content-type: application/json' \
    -d "$openai_tools_request"
)"
if [[ "$openai_tools_status" != "200" ]]; then
  echo "expected tools request to return HTTP 200, got ${openai_tools_status}" >&2
  cat "$openai_tools_response" >&2 || true
  exit 1
fi
assert_json "$openai_tools_response" \
  '.object == "chat.completion" and .choices[0].message.role == "assistant" and .usage.prompt_tokens > 0'

openai_logprobs_request="$(jq -cn --arg model "$DENSE_MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Say hi"}],
  logprobs: true,
  top_logprobs: 2,
  max_tokens: 2
}')"
openai_logprobs_response="$WORK_DIR/openai-logprobs.json"
openai_logprobs_status="$(
  curl -sS --max-time 10 \
    -o "$openai_logprobs_response" \
    -w '%{http_code}' \
    "${OPENAI_BASE_URL}/chat/completions" \
    -H 'content-type: application/json' \
    -d "$openai_logprobs_request"
)"
if [[ "$openai_logprobs_status" != "400" ]]; then
  echo "expected logprobs request to return HTTP 400 until backend support lands, got ${openai_logprobs_status}" >&2
  cat "$openai_logprobs_response" >&2 || true
  exit 1
fi
assert_json "$openai_logprobs_response" '.error.code == "unsupported_model_feature"'

openai_structured_response="$WORK_DIR/openai-structured-output.json"
openai_structured_request="$(jq -cn --arg model "$DENSE_MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Return {\"ok\":true}."}],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "answer",
      schema: {
        type: "object",
        properties: {ok: {type: "boolean"}},
        required: ["ok"],
        additionalProperties: false
      }
    }
  },
  max_tokens: 2
}')"
openai_structured_status="$(
  curl -sS --max-time 10 \
    -o "$openai_structured_response" \
    -w '%{http_code}' \
    "${OPENAI_BASE_URL}/chat/completions" \
    -H 'content-type: application/json' \
    -d "$openai_structured_request"
)"
if [[ "$openai_structured_status" != "400" ]]; then
  echo "expected structured-output request to return HTTP 400 until backend support lands, got ${openai_structured_status}" >&2
  cat "$openai_structured_response" >&2 || true
  exit 1
fi
assert_json "$openai_structured_response" '.error.code == "unsupported_model_feature"'

cleanup
SERVER_PID=""

echo "smoke: prompt exact-prefix hit and live-session reuse"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  "$STAGE_SERVER_BIN" serve-binary \
    --config "$PROMPT_CONFIG" \
    --activation-width 2048 \
    --activation-wire-dtype f16 \
    --max-inflight 4 \
    >"$PROMPT_LOG" 2>&1 &
SERVER_PID="$!"
if ! wait_for_tcp "127.0.0.1" "$PROMPT_PORT" "$SERVER_PID"; then
  echo "prompt binary stage did not become ready; log follows" >&2
  sed -n '1,240p' "$PROMPT_LOG" >&2 || true
  exit 1
fi

set +e
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  target/debug/skippy-prompt binary \
    --model-path "$DENSE_MODEL_PATH" \
    --tokenizer-model-path "$DENSE_MODEL_PATH" \
    --tokenizer-load-mode runtime-slice \
    --tokenizer-layer-start 0 \
    --tokenizer-layer-end "$DENSE_LAYER_END" \
    --first-stage-addr "$PROMPT_BIND" \
    --ctx-size "$PROMPT_CTX_SIZE" \
    --activation-width 2048 \
    --activation-wire-dtype f16 \
    --prefill-chunk-size "$PROMPT_PREFILL_CHUNK_SIZE" \
    --max-new-tokens "$PROMPT_MAX_NEW_TOKENS" \
    --session-id skippy-ci-smoke \
    --no-think \
    <"$PROMPT_IN" >"$PROMPT_OUT" 2>&1
PROMPT_STATUS=$?
set -e
if [[ "$PROMPT_STATUS" -ne 0 ]]; then
  echo "skippy-prompt smoke failed; prompt output follows" >&2
  sed -n '1,260p' "$PROMPT_OUT" >&2 || true
  echo "stage log follows" >&2
  sed -n '1,260p' "$PROMPT_LOG" >&2 || true
  exit "$PROMPT_STATUS"
fi

if ! grep -q 'reuse    exact_prefix=hit' "$PROMPT_OUT"; then
  echo "expected exact-prefix cache hit in prompt output" >&2
  sed -n '1,320p' "$PROMPT_OUT" >&2 || true
  exit 1
fi
if ! grep -q 'reuse    live_session=hit' "$PROMPT_OUT"; then
  echo "expected live-session reuse hit in prompt output" >&2
  sed -n '1,320p' "$PROMPT_OUT" >&2 || true
  exit 1
fi
cleanup
SERVER_PID=""

echo "skippy CI smoke passed"
echo "reports: $REPORT_DIR"
