#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPO_ID="${MTP_REPO_ID:-unsloth/Qwen3.6-27B-MTP-GGUF}"
MODEL_FILE="${MTP_MODEL_FILE:-Qwen3.6-27B-IQ4_XS.gguf}"
MODEL_ID="${MTP_MODEL_ID:-unsloth/Qwen3.6-27B-MTP-GGUF:IQ4_XS}"
OUT_DIR="${MTP_OUT_DIR:-$ROOT/experiments/mtp}"
MODEL_DIR="${MTP_MODEL_DIR:-$OUT_DIR/models}"
WORK_DIR="${MTP_WORK_DIR:-$OUT_DIR/work}"
PROMPT_CORPUS="${MTP_PROMPT_CORPUS:-$ROOT/crates/skippy-bench/corpora/speculative_coding_prompts.jsonl}"
PROMPT_LIMIT="${MTP_PROMPT_LIMIT:-12}"
MAX_TOKENS="${MTP_MAX_TOKENS:-128}"
CTX_SIZE="${MTP_CTX_SIZE:-8192}"
N_GPU_LAYERS="${MTP_N_GPU_LAYERS:--1}"
ACTIVATION_WIDTH="${MTP_ACTIVATION_WIDTH:-2048}"
ACTIVATION_WIRE_DTYPE="${MTP_ACTIVATION_WIRE_DTYPE:-f16}"
CONCURRENCY_DEPTH="${MTP_CONCURRENCY_DEPTH:-1}"
MTP_WINDOW="${MTP_WINDOW:-3}"
WARMUP_REQUESTS="${MTP_WARMUP_REQUESTS:-1}"
WARMUP_TOKENS="${MTP_WARMUP_TOKENS:-32}"
CORPUS_WARMUP_TOKENS="${MTP_CORPUS_WARMUP_TOKENS:-0}"
TEMPERATURE="${MTP_TEMPERATURE:-0}"
ENABLE_THINKING="${MTP_ENABLE_THINKING:-}"
PORT_BASE="${MTP_PORT_BASE:-29330}"
STARTUP_TIMEOUT_SECS="${MTP_STARTUP_TIMEOUT_SECS:-900}"
RUN_MODES="${MTP_RUN_MODES:-one-stage-baseline,one-stage-mtp,two-stage-baseline,two-stage-mtp}"
DOWNLOAD_MODEL="${MTP_DOWNLOAD_MODEL:-1}"
BUILD_BINARIES="${MTP_BUILD_BINARIES:-1}"
USE_STAGE_ARTIFACTS="${MTP_USE_STAGE_ARTIFACTS:-0}"
REMOTE_STAGE1_SSH="${MTP_REMOTE_STAGE1_SSH:-}"
REMOTE_STAGE1_ROOT="${MTP_REMOTE_STAGE1_ROOT:-mesh-llm}"
REMOTE_STAGE1_BIN="${MTP_REMOTE_STAGE1_BIN:-target/debug/skippy-server}"
REMOTE_STAGE1_MODEL_PATH="${MTP_REMOTE_STAGE1_MODEL_PATH:-$MODEL_DIR/$MODEL_FILE}"
REMOTE_STAGE1_ENDPOINT_HOST="${MTP_REMOTE_STAGE1_ENDPOINT_HOST:-}"
REMOTE_METRICS_HOST="${MTP_REMOTE_METRICS_HOST:-}"
REMOTE_METRICS_OTLP="${MTP_REMOTE_METRICS_OTLP:-}"

METRICS_SERVER_BIN="${MTP_METRICS_SERVER_BIN:-$ROOT/target/debug/metrics-server}"
SKIPPY_SERVER_BIN="${MTP_SKIPPY_SERVER_BIN:-$ROOT/target/debug/skippy-server}"
SKIPPY_BENCH_BIN="${MTP_SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
SKIPPY_PACKAGE_BIN="${MTP_SKIPPY_PACKAGE_BIN:-$ROOT/target/debug/skippy-model-package}"

usage() {
  cat <<EOF
Usage: $0 [--download-only] [--run MODES]

Skippy-local MTP experiment scaffold.

Modes:
  one-stage-baseline   Skippy single stage, no MTP
  one-stage-mtp        Skippy single stage with native MTP proposals
  two-stage-baseline   Skippy two local stage processes, no MTP
  two-stage-mtp        Skippy two local stage processes with final-stage MTP proposals
  all

Key environment overrides:
  MTP_MODEL_FILE=$MODEL_FILE
  MTP_PROMPT_LIMIT=$PROMPT_LIMIT
  MTP_MAX_TOKENS=$MAX_TOKENS
  MTP_WARMUP_REQUESTS=$WARMUP_REQUESTS
  MTP_WARMUP_TOKENS=$WARMUP_TOKENS
  MTP_CORPUS_WARMUP_TOKENS=$CORPUS_WARMUP_TOKENS
  MTP_TEMPERATURE=$TEMPERATURE
  MTP_ENABLE_THINKING=$ENABLE_THINKING
  MTP_CTX_SIZE=$CTX_SIZE
  MTP_ACTIVATION_WIDTH=$ACTIVATION_WIDTH
  MTP_WINDOW=$MTP_WINDOW
  MTP_USE_STAGE_ARTIFACTS=$USE_STAGE_ARTIFACTS
  MTP_REMOTE_STAGE1_SSH=skippy
  MTP_REMOTE_STAGE1_ENDPOINT_HOST=<skippy-lan-ip-or-hostname>
  MTP_REMOTE_STAGE1_ROOT=$REMOTE_STAGE1_ROOT
  MTP_REMOTE_STAGE1_MODEL_PATH=$REMOTE_STAGE1_MODEL_PATH
  MTP_REMOTE_METRICS_HOST=<local-lan-ip>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-only)
      RUN_MODES=""
      BUILD_BINARIES=0
      shift
      ;;
    --run)
      RUN_MODES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

need_cmd jq
need_cmd curl
mkdir -p "$OUT_DIR" "$MODEL_DIR" "$WORK_DIR"

if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  need_cmd hf
  echo "downloading $REPO_ID/$MODEL_FILE into $MODEL_DIR"
  hf download "$REPO_ID" "$MODEL_FILE" --local-dir "$MODEL_DIR"
fi

MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ "$BUILD_BINARIES" == "1" ]]; then
  just build
fi

if [[ -z "$RUN_MODES" ]]; then
  exit 0
fi

if [[ "$RUN_MODES" == "all" ]]; then
  RUN_MODES="one-stage-baseline,one-stage-mtp,two-stage-baseline,two-stage-mtp"
fi

for bin in "$METRICS_SERVER_BIN" "$SKIPPY_SERVER_BIN" "$SKIPPY_BENCH_BIN" "$SKIPPY_PACKAGE_BIN"; do
  if [[ ! -x "$bin" ]]; then
    echo "missing executable: $bin" >&2
    echo "run: just build" >&2
    exit 1
  fi
done

PLAN_JSON="$WORK_DIR/model-plan-2-stage.json"
"$SKIPPY_PACKAGE_BIN" plan "$MODEL_PATH" --stages 2 >"$PLAN_JSON"
LAYER_COUNT="$(jq -r '.layer_count' "$PLAN_JSON")"
SPLIT_LAYER="$(jq -r '.stages[0].layer_end' "$PLAN_JSON")"

wait_http_ok() {
  local url="$1"
  local timeout="$2"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start > timeout )); then
      echo "timed out waiting for $url" >&2
      return 1
    fi
    sleep 1
  done
}

warm_openai_model() {
  local openai_base="$1"
  local warmup_json="$2"
  local index

  if (( WARMUP_REQUESTS <= 0 )); then
    return 0
  fi

  local warmup_filter='
    {
      model: $model,
      messages: [{role: "user", content: $prompt}],
      max_tokens: $max_tokens,
      temperature: $temperature,
      stream: false,
      user: "mtp-warmup"
    }
  '

  if [[ -n "$ENABLE_THINKING" ]]; then
    warmup_filter+=" | .enable_thinking = \$enable_thinking"
    jq -n \
      --arg model "$MODEL_ID" \
      --arg prompt "Reply with exactly one short sentence confirming the model is ready." \
      --argjson max_tokens "$WARMUP_TOKENS" \
      --argjson temperature "$TEMPERATURE" \
      --argjson enable_thinking "$ENABLE_THINKING" \
      "$warmup_filter" >"$warmup_json"
  else
    jq -n \
      --arg model "$MODEL_ID" \
      --arg prompt "Reply with exactly one short sentence confirming the model is ready." \
      --argjson max_tokens "$WARMUP_TOKENS" \
      --argjson temperature "$TEMPERATURE" \
      "$warmup_filter" >"$warmup_json"
  fi

  for ((index = 1; index <= WARMUP_REQUESTS; index++)); do
    echo "warmup request $index/$WARMUP_REQUESTS -> $openai_base/chat/completions"
    curl -fsS \
      -H 'content-type: application/json' \
      --data-binary @"$warmup_json" \
      "$openai_base/chat/completions" >/dev/null
  done
}

warm_openai_corpus_prompt() {
  local openai_base="$1"
  local warmup_json="$2"
  local prompt
  local warmup_filter

  if (( CORPUS_WARMUP_TOKENS <= 0 )); then
    return 0
  fi

  prompt="$(jq -r 'select(length > 0) | .prompt' "$PROMPT_CORPUS" | head -n 1)"
  if [[ -z "$prompt" || "$prompt" == "null" ]]; then
    return 0
  fi

  warmup_filter='
    {
      model: $model,
      messages: [{role: "user", content: $prompt}],
      max_tokens: $max_tokens,
      temperature: $temperature,
      stream: false,
      user: "mtp-corpus-warmup"
    }
  '

  if [[ -n "$ENABLE_THINKING" ]]; then
    warmup_filter+=" | .enable_thinking = \$enable_thinking"
    jq -n \
      --arg model "$MODEL_ID" \
      --arg prompt "$prompt" \
      --argjson max_tokens "$CORPUS_WARMUP_TOKENS" \
      --argjson temperature "$TEMPERATURE" \
      --argjson enable_thinking "$ENABLE_THINKING" \
      "$warmup_filter" >"$warmup_json"
  else
    jq -n \
      --arg model "$MODEL_ID" \
      --arg prompt "$prompt" \
      --argjson max_tokens "$CORPUS_WARMUP_TOKENS" \
      --argjson temperature "$TEMPERATURE" \
      "$warmup_filter" >"$warmup_json"
  fi

  echo "corpus warmup request 1/1 -> $openai_base/chat/completions"
  curl -fsS \
    -H 'content-type: application/json' \
    --data-binary @"$warmup_json" \
    "$openai_base/chat/completions" >/dev/null
}

shell_quote() {
  printf "%q" "$1"
}

join_shell_words() {
  local out=""
  local word
  for word in "$@"; do
    if [[ -n "$out" ]]; then
      out+=" "
    fi
    out+="$(shell_quote "$word")"
  done
  printf "%s" "$out"
}

create_stage_config() {
  local path="$1"
  local run_id="$2"
  local stage_index="$3"
  local layer_start="$4"
  local layer_end="$5"
  local bind_addr="$6"
  local upstream_json="$7"
  local downstream_json="$8"
  local model_path="$9"
  local load_mode="${10}"
  local filter="${11}"

  jq -n \
    --arg run_id "$run_id" \
    --arg model_id "$MODEL_ID" \
    --arg model_path "$model_path" \
    --arg stage_id "stage-$stage_index" \
    --arg bind_addr "$bind_addr" \
    --arg load_mode "$load_mode" \
    --argjson upstream "$upstream_json" \
    --argjson downstream "$downstream_json" \
    --argjson stage_index "$stage_index" \
    --argjson layer_start "$layer_start" \
    --argjson layer_end "$layer_end" \
    --argjson ctx_size "$CTX_SIZE" \
    --argjson n_gpu_layers "$N_GPU_LAYERS" \
    --argjson filter "$filter" \
    '{
      run_id: $run_id,
      topology_id: "mtp-skippy-local",
      model_id: $model_id,
      model_path: $model_path,
      stage_id: $stage_id,
      stage_index: $stage_index,
      layer_start: $layer_start,
      layer_end: $layer_end,
      ctx_size: $ctx_size,
      lane_count: 1,
      n_gpu_layers: $n_gpu_layers,
      cache_type_k: "f16",
      cache_type_v: "f16",
      filter_tensors_on_load: $filter,
      load_mode: $load_mode,
      bind_addr: $bind_addr,
      upstream: $upstream,
      downstream: $downstream
    }' >"$path"
}

run_case() {
  local mode="$1"
  local ordinal="$2"
  local stage_count="$3"
  local mtp_enabled="$4"
  local run_id="mtp-skippy-${mode}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$OUT_DIR/$run_id"
  local metrics_port=$((PORT_BASE + 100 + ordinal))
  local otlp_port=$((PORT_BASE + 200 + ordinal))
  local openai_port=$((PORT_BASE + 300 + ordinal))
  local stage0_port=$((PORT_BASE + 400 + ordinal * 10))
  local stage1_port=$((stage0_port + 1))
  local metrics_http="http://127.0.0.1:$metrics_port"
  local metrics_otlp="http://127.0.0.1:$otlp_port"
  local openai_base="http://127.0.0.1:$openai_port/v1"
  local db="$run_dir/metrics.duckdb"
  local driver_json="$run_dir/driver-result.json"
  local metrics_json="$run_dir/metrics-report.json"
  local combined_json="$OUT_DIR/${run_id}.json"
  local pids=()

  mkdir -p "$run_dir"

  cleanup_case() {
    local pid
    for pid in "${pids[@]:-}"; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        kill "$pid" >/dev/null 2>&1 || true
      fi
    done
    wait >/dev/null 2>&1 || true
  }
  trap cleanup_case RETURN

  "$METRICS_SERVER_BIN" serve \
    --db "$db" \
    --http-addr "127.0.0.1:$metrics_port" \
    --otlp-grpc-addr "127.0.0.1:$otlp_port" \
    >"$run_dir/metrics-server.log" 2>&1 &
  pids+=("$!")

  wait_http_ok "$metrics_http/v1/runs/noop/status" 3 || true

  jq -n \
    --arg run_id "$run_id" \
    --arg mode "$mode" \
    --arg model_id "$MODEL_ID" \
    --arg model_path "$MODEL_PATH" \
    --argjson mtp_enabled "$mtp_enabled" \
    --argjson stage_count "$stage_count" \
    --argjson layer_count "$LAYER_COUNT" \
    --argjson split_layer "$SPLIT_LAYER" \
    '{
      run_id: $run_id,
      config: {
        experiment: "mtp-skippy-local",
        mode: $mode,
        model_id: $model_id,
        model_path: $model_path,
        mtp_enabled: $mtp_enabled,
        stage_count: $stage_count,
        layer_count: $layer_count,
        split_layer: $split_layer
      }
    }' >"$run_dir/create-run.json"
  curl -fsS -X POST "$metrics_http/v1/runs" \
    -H 'content-type: application/json' \
    --data-binary @"$run_dir/create-run.json" >/dev/null

  if [[ "$stage_count" == "1" ]]; then
    create_stage_config \
      "$run_dir/stage-0.json" "$run_id" 0 0 "$LAYER_COUNT" "127.0.0.1:$stage0_port" \
      'null' 'null' "$MODEL_PATH" "runtime-slice" false
    local openai_mtp_args=()
    if [[ "$mtp_enabled" == "true" ]]; then
      openai_mtp_args=(--openai-mtp --openai-mtp-window "$MTP_WINDOW")
    fi
    local stage0_cmd=(
      "$SKIPPY_SERVER_BIN" serve-binary
      --config "$run_dir/stage-0.json" \
      --activation-width "$ACTIVATION_WIDTH" \
      --activation-wire-dtype "$ACTIVATION_WIRE_DTYPE" \
      --metrics-otlp-grpc "$metrics_otlp" \
      --telemetry-level summary \
      --openai-bind-addr "127.0.0.1:$openai_port" \
      --openai-model-id "$MODEL_ID" \
      --openai-default-max-tokens "$MAX_TOKENS" \
      --openai-generation-concurrency "$CONCURRENCY_DEPTH"
    )
    if [[ "${#openai_mtp_args[@]}" -gt 0 ]]; then
      stage0_cmd+=("${openai_mtp_args[@]}")
    fi
    "${stage0_cmd[@]}" \
      >"$run_dir/stage-0.log" 2>&1 &
    pids+=("$!")
  else
    local stage0_model="$MODEL_PATH"
    local stage1_model="$MODEL_PATH"
    local stage0_load_mode="runtime-slice"
    local stage1_load_mode="runtime-slice"
    local stage_filter=false
    local stage1_bind_addr="127.0.0.1:$stage1_port"
    local stage1_endpoint="tcp://127.0.0.1:$stage1_port"
    local stage1_config_path="$run_dir/stage-1.json"
    local stage1_metrics_otlp="$metrics_otlp"

    if [[ "$USE_STAGE_ARTIFACTS" == "1" ]]; then
      if [[ -n "$REMOTE_STAGE1_SSH" ]]; then
        echo "MTP_USE_STAGE_ARTIFACTS=1 is not supported with MTP_REMOTE_STAGE1_SSH; use runtime-slice full GGUF on the remote node" >&2
        exit 1
      fi
      local package_dir="$WORK_DIR/${run_id}-stage-artifacts"
      "$SKIPPY_PACKAGE_BIN" write-stages "$MODEL_PATH" --stages 2 --out-dir "$package_dir"
      stage0_model="$package_dir/stage-000.gguf"
      stage1_model="$package_dir/stage-001.gguf"
      stage0_load_mode="artifact-slice"
      stage1_load_mode="artifact-slice"
      stage_filter=true
    fi

    if [[ -n "$REMOTE_STAGE1_SSH" ]]; then
      if [[ -z "$REMOTE_STAGE1_ENDPOINT_HOST" ]]; then
        echo "MTP_REMOTE_STAGE1_ENDPOINT_HOST is required when MTP_REMOTE_STAGE1_SSH is set" >&2
        exit 1
      fi
      if [[ -z "$REMOTE_METRICS_HOST" && -z "$REMOTE_METRICS_OTLP" ]]; then
        echo "MTP_REMOTE_METRICS_HOST is required when MTP_REMOTE_STAGE1_SSH is set; use the local machine LAN address, not 127.0.0.1" >&2
        exit 1
      fi
      stage1_model="$REMOTE_STAGE1_MODEL_PATH"
      stage1_bind_addr="0.0.0.0:$stage1_port"
      stage1_endpoint="tcp://$REMOTE_STAGE1_ENDPOINT_HOST:$stage1_port"
      if [[ -n "$REMOTE_METRICS_OTLP" ]]; then
        stage1_metrics_otlp="$REMOTE_METRICS_OTLP"
      else
        stage1_metrics_otlp="http://$REMOTE_METRICS_HOST:$otlp_port"
      fi
    fi

    create_stage_config \
      "$stage1_config_path" "$run_id" 1 "$SPLIT_LAYER" "$LAYER_COUNT" "$stage1_bind_addr" \
      "{\"stage_id\":\"stage-0\",\"stage_index\":0,\"endpoint\":\"driver\"}" 'null' \
      "$stage1_model" "$stage1_load_mode" "$stage_filter"
    local openai_mtp_args=()
    if [[ "$mtp_enabled" == "true" ]]; then
      openai_mtp_args=(--openai-mtp --openai-mtp-window "$MTP_WINDOW")
    fi
    if [[ -n "$REMOTE_STAGE1_SSH" ]]; then
      local remote_run_dir="$REMOTE_STAGE1_ROOT/experiments/mtp/$run_id"
      ssh "$REMOTE_STAGE1_SSH" "mkdir -p $(shell_quote "$remote_run_dir")"
      scp "$stage1_config_path" "$REMOTE_STAGE1_SSH:$remote_run_dir/stage-1.json" >/dev/null
      local remote_stage1_cmd
      remote_stage1_cmd="$(join_shell_words \
        "$REMOTE_STAGE1_BIN" serve-binary \
        --config "$remote_run_dir/stage-1.json" \
        --activation-width "$ACTIVATION_WIDTH" \
        --activation-wire-dtype "$ACTIVATION_WIRE_DTYPE" \
        --metrics-otlp-grpc "$stage1_metrics_otlp" \
        --telemetry-level summary)"
      ssh -tt "$REMOTE_STAGE1_SSH" "/bin/zsh -ilc 'cd $(shell_quote "$REMOTE_STAGE1_ROOT") && $remote_stage1_cmd 2>&1 | tee $(shell_quote "$remote_run_dir/stage-1.log")'" \
        >"$run_dir/stage-1.ssh.log" 2>&1 &
      pids+=("$!")
    else
      "$SKIPPY_SERVER_BIN" serve-binary \
        --config "$stage1_config_path" \
        --activation-width "$ACTIVATION_WIDTH" \
        --activation-wire-dtype "$ACTIVATION_WIRE_DTYPE" \
        --metrics-otlp-grpc "$metrics_otlp" \
        --telemetry-level summary \
        >"$run_dir/stage-1.log" 2>&1 &
      pids+=("$!")
    fi

    create_stage_config \
      "$run_dir/stage-0.json" "$run_id" 0 0 "$SPLIT_LAYER" "127.0.0.1:$stage0_port" \
      'null' "{\"stage_id\":\"stage-1\",\"stage_index\":1,\"endpoint\":\"$stage1_endpoint\"}" \
      "$stage0_model" "$stage0_load_mode" "$stage_filter"
    local stage0_cmd=(
      "$SKIPPY_SERVER_BIN" serve-binary
      --config "$run_dir/stage-0.json" \
      --activation-width "$ACTIVATION_WIDTH" \
      --activation-wire-dtype "$ACTIVATION_WIRE_DTYPE" \
      --metrics-otlp-grpc "$metrics_otlp" \
      --telemetry-level summary \
      --openai-bind-addr "127.0.0.1:$openai_port" \
      --openai-model-id "$MODEL_ID" \
      --openai-default-max-tokens "$MAX_TOKENS" \
      --openai-generation-concurrency "$CONCURRENCY_DEPTH"
    )
    if [[ "${#openai_mtp_args[@]}" -gt 0 ]]; then
      stage0_cmd+=("${openai_mtp_args[@]}")
    fi
    "${stage0_cmd[@]}" \
      >"$run_dir/stage-0.log" 2>&1 &
    pids+=("$!")
  fi

  wait_http_ok "http://127.0.0.1:$openai_port/v1/models" "$STARTUP_TIMEOUT_SECS"
  warm_openai_model "$openai_base" "$run_dir/warmup-request.json"
  warm_openai_corpus_prompt "$openai_base" "$run_dir/corpus-warmup-request.json"

  local bench_cmd=(
    "$SKIPPY_BENCH_BIN" chat-corpus
    --base-url "$openai_base"
    --model "$MODEL_ID"
    --prompt-corpus "$PROMPT_CORPUS"
    --prompt-limit "$PROMPT_LIMIT"
    --max-tokens "$MAX_TOKENS"
    --concurrency-depth "$CONCURRENCY_DEPTH"
    --stream
    --include-usage true
    --temperature "$TEMPERATURE"
    --output "$driver_json"
  )
  if [[ -n "$ENABLE_THINKING" ]]; then
    bench_cmd+=(--enable-thinking "$ENABLE_THINKING")
  fi
  "${bench_cmd[@]}"

  sleep 1
  curl -fsS -X POST "$metrics_http/v1/runs/$run_id/finalize" >/dev/null
  curl -fsS "$metrics_http/v1/runs/$run_id/report.json" >"$metrics_json"

  jq -n \
    --arg run_id "$run_id" \
    --arg mode "$mode" \
    --slurpfile metrics "$metrics_json" \
    --slurpfile driver "$driver_json" \
    '{
      run_id: $run_id,
      mode: $mode,
      status: "complete",
      metrics_report: $metrics[0],
      driver_result: $driver[0]
    }' >"$combined_json"
  echo "$combined_json"
}

IFS=',' read -r -a modes <<<"$RUN_MODES"
ordinal=0
for mode in "${modes[@]}"; do
  mode="${mode// /}"
  case "$mode" in
    one-stage-baseline) run_case "$mode" "$ordinal" 1 false ;;
    two-stage-baseline) run_case "$mode" "$ordinal" 2 false ;;
    one-stage-mtp) run_case "$mode" "$ordinal" 1 true ;;
    two-stage-mtp) run_case "$mode" "$ordinal" 2 true ;;
    *) echo "unknown mode: $mode" >&2; exit 1 ;;
  esac
  ordinal=$((ordinal + 1))
done
