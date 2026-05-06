#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${ROOT}/.deps/llama.cpp/build-stage-abi-metal" ]]; then
  DEFAULT_LLAMA_BUILD_DIR="${ROOT}/.deps/llama.cpp/build-stage-abi-metal"
else
  DEFAULT_LLAMA_BUILD_DIR="${ROOT}/.deps/llama.cpp/build-stage-abi-static"
fi

LLAMA_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$DEFAULT_LLAMA_BUILD_DIR}"
MODEL_REPO="${MODEL_REPO:-jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF}"
MODEL_FILE="${MODEL_FILE:-SmolLM2-135M-Instruct.Q4_K_M.gguf}"
MODEL_SELECTOR="${MODEL_SELECTOR:-Q4_K_M}"
MODEL_ID="${MODEL_ID:-${MODEL_REPO}:${MODEL_SELECTOR}}"
MODEL_PATH="${MODEL_PATH:-}"
RUN_ID="${RUN_ID:-spec-openai-$(date +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-${ROOT}/target/skippy-openai-spec-bench/${RUN_ID}}"
HOST="${HOST:-127.0.0.1}"
if [[ -z "${PORT:-}" ]]; then
  PORT="$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi
if [[ -z "${BINARY_PORT:-}" ]]; then
  BINARY_PORT="$(python3 - <<'PY'
import socket
for base in range(19000, 25000, 2):
    sockets = []
    try:
        for port in (base, base + 1):
            sock = socket.socket()
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
        print(base)
        break
    except OSError:
        pass
    finally:
        for sock in sockets:
            sock.close()
else:
    raise SystemExit("no free adjacent ports found")
PY
)"
fi
STAGE1_PORT="${STAGE1_PORT:-$((BINARY_PORT + 1))}"
BASE_URL="http://${HOST}:${PORT}/v1"
CTX_SIZE="${CTX_SIZE:-512}"
MAX_TOKENS="${MAX_TOKENS:-24}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
GENERATION_CONCURRENCY="${GENERATION_CONCURRENCY:-1}"
SPEC_WINDOW="${SPEC_WINDOW:-8}"
NGRAM_N="${NGRAM_N:-6}"
NGRAM_MIN_HITS="${NGRAM_MIN_HITS:-1}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
DRAFT_N_GPU_LAYERS="${DRAFT_N_GPU_LAYERS:-}"
WARM_REPEATS="${WARM_REPEATS:-6}"
MIXED_LIMIT="${MIXED_LIMIT:-8}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-600}"

if [[ -n "${MODES:-}" ]]; then
  read -r -a BENCH_MODES <<<"$MODES"
else
  BENCH_MODES=(baseline ngram ngram-adaptive)
  if [[ -n "$DRAFT_MODEL_PATH" ]]; then
    BENCH_MODES+=(draft draft-adaptive)
  fi
fi
MODES_JOINED="${BENCH_MODES[*]}"

STAGE0_PID=""
STAGE1_PID=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

cleanup() {
  stop_server
}
trap cleanup EXIT

check_port_free() {
  local port="$1"
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port ${port} is already listening" >&2
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
    exit 1
  fi
}

wait_ready() {
  local log="$1"
  for _ in {1..120}; do
    local models_json=""
    if models_json="$(curl -fsS --max-time 1 "${BASE_URL}/models" 2>/dev/null)" \
      && jq -e --arg model "$MODEL_ID" 'any(.data[]?; .id == $model)' <<<"$models_json" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$STAGE0_PID" >/dev/null 2>&1; then
      echo "stage0 serve-binary exited early; log follows" >&2
      sed -n '1,220p' "$log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "serve-binary did not become ready; log follows" >&2
  sed -n '1,220p' "$log" >&2 || true
  exit 1
}

wait_port_listening() {
  local port="$1"
  local pid="$2"
  local log="$3"
  for _ in {1..120}; do
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "serve-binary on port ${port} exited early; log follows" >&2
      sed -n '1,220p' "$log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "serve-binary on port ${port} did not listen; log follows" >&2
  sed -n '1,220p' "$log" >&2 || true
  exit 1
}

stop_server() {
  if [[ -n "$STAGE0_PID" ]] && kill -0 "$STAGE0_PID" >/dev/null 2>&1; then
    kill "$STAGE0_PID" >/dev/null 2>&1 || true
    wait "$STAGE0_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$STAGE1_PID" ]] && kill -0 "$STAGE1_PID" >/dev/null 2>&1; then
    kill "$STAGE1_PID" >/dev/null 2>&1 || true
    wait "$STAGE1_PID" >/dev/null 2>&1 || true
  fi
  STAGE0_PID=""
  STAGE1_PID=""
}

start_server() {
  local mode="$1"
  local log="$2"
  local stage1_log="${log%.log}-stage1.log"

  check_port_free "$PORT"
  check_port_free "$BINARY_PORT"
  check_port_free "$STAGE1_PORT"

  echo "starting ${mode} stage1 serve-binary on ${HOST}:${STAGE1_PORT}"
  SKIPPY_TELEMETRY_STDERR=1 \
  LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
    "${ROOT}/target/debug/skippy-server" serve-binary \
      --config "$STAGE1_CONFIG_PATH" \
      --bind-addr "${HOST}:${STAGE1_PORT}" \
      --activation-width "$ACTIVATION_WIDTH" \
      --telemetry-level debug \
      >"$stage1_log" 2>&1 &
  STAGE1_PID="$!"
  wait_port_listening "$STAGE1_PORT" "$STAGE1_PID" "$stage1_log"

  echo "starting ${mode} stage0 serve-binary on ${BASE_URL}"
  local -a command=(
    "${ROOT}/target/debug/skippy-server" serve-binary
    --config "$STAGE0_CONFIG_PATH"
    --bind-addr "${HOST}:${BINARY_PORT}"
    --activation-width "$ACTIVATION_WIDTH"
    --telemetry-level debug
    --openai-bind-addr "${HOST}:${PORT}"
    --openai-model-id "$MODEL_ID"
    --openai-default-max-tokens "$MAX_TOKENS"
    --openai-generation-concurrency "$GENERATION_CONCURRENCY"
  )
  case "$mode" in
    baseline)
      ;;
    ngram|ngram-adaptive|hybrid|hybrid-adaptive)
      command+=(
        --openai-ngram-speculative
        --openai-speculative-window "$SPEC_WINDOW"
        --openai-spec-ngram-size-n "$NGRAM_N"
        --openai-ngram-history-min-hits "$NGRAM_MIN_HITS"
      )
      ;;
    ngram-auto)
      command+=(
        --openai-ngram-auto
        --openai-speculative-window "$SPEC_WINDOW"
        --openai-spec-ngram-size-n "$NGRAM_N"
        --openai-ngram-history-min-hits "$NGRAM_MIN_HITS"
        --openai-adaptive-speculative-window
      )
      ;;
    draft|draft-adaptive)
      ;;
    *)
      echo "unknown benchmark mode: ${mode}" >&2
      exit 1
      ;;
  esac
  case "$mode" in
    draft|draft-adaptive|hybrid|hybrid-adaptive)
      if [[ -z "$DRAFT_MODEL_PATH" ]]; then
        echo "mode ${mode} requires DRAFT_MODEL_PATH" >&2
        exit 1
      fi
      command+=(
        --openai-draft-model-path "$DRAFT_MODEL_PATH"
        --openai-speculative-window "$SPEC_WINDOW"
      )
      if [[ -n "$DRAFT_N_GPU_LAYERS" ]]; then
        command+=("--openai-draft-n-gpu-layers=$DRAFT_N_GPU_LAYERS")
      fi
      ;;
  esac
  case "$mode" in
    *-adaptive)
      command+=(--openai-adaptive-speculative-window)
      ;;
  esac
  SKIPPY_TELEMETRY_STDERR=1 \
  LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
    "${command[@]}" \
      >"$log" 2>&1 &
  STAGE0_PID="$!"
  wait_ready "$log"
}

run_chat_corpus() {
  local mode="$1"
  local corpus_name="$2"
  local corpus_path="$3"
  local output_path="$4"

  echo "running ${mode}/${corpus_name} corpus"
  "${ROOT}/target/debug/skippy-bench" chat-corpus \
    --base-url "$BASE_URL" \
    --model "$MODEL_ID" \
    --prompt-corpus "$corpus_path" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency-depth "$GENERATION_CONCURRENCY" \
    --request-timeout-secs "$REQUEST_TIMEOUT_SECS" \
    --session-prefix "${RUN_ID}-${mode}-${corpus_name}" \
    --temperature 0 \
    --seed 7 \
    --output "$output_path" \
    >"${output_path%.json}.stdout.json"
}

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
if [[ -n "$DRAFT_MODEL_PATH" && ! -f "$DRAFT_MODEL_PATH" ]]; then
  echo "draft model path not found: $DRAFT_MODEL_PATH" >&2
  exit 1
fi

echo "building skippy benchmark binaries"
(cd "$ROOT" && LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" cargo build -p skippy-server -p skippy-model-package -p skippy-bench)

INSPECT_JSON="${WORK_DIR}/model-inspect.json"
LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" "${ROOT}/target/debug/skippy-model-package" inspect "$MODEL_PATH" >"$INSPECT_JSON"
LAYER_END="$(jq '[.tensors[] | select(.role == "layer") | .layer_index] | max + 1' "$INSPECT_JSON")"
if [[ -z "$LAYER_END" || "$LAYER_END" == "null" ]]; then
  echo "failed to infer layer_end from $MODEL_PATH" >&2
  exit 1
fi
SPLIT_LAYER="${SPLIT_LAYER:-$((LAYER_END / 2))}"
if (( SPLIT_LAYER <= 0 || SPLIT_LAYER >= LAYER_END )); then
  echo "invalid SPLIT_LAYER=${SPLIT_LAYER}; expected 0 < split < ${LAYER_END}" >&2
  exit 1
fi
ACTIVATION_WIDTH="${ACTIVATION_WIDTH:-$(jq '[.tensors[] | select(.layer_index == 0 and ((.name | endswith("attn_norm.weight")) or (.name | endswith("attention_norm.weight")) or (.name | endswith("input_layernorm.weight")) or (.name | endswith("ln_1.weight")))) | if .element_count > 0 then .element_count elif .ggml_type == 0 then (.byte_size / 4 | floor) elif .ggml_type == 1 then (.byte_size / 2 | floor) else empty end] | first // empty' "$INSPECT_JSON")}"
if [[ -z "$ACTIVATION_WIDTH" || "$ACTIVATION_WIDTH" == "null" ]]; then
  echo "failed to infer activation width from $MODEL_PATH; set ACTIVATION_WIDTH" >&2
  exit 1
fi

STAGE0_CONFIG_PATH="${WORK_DIR}/stage-0-openai-spec-bench.json"
STAGE1_CONFIG_PATH="${WORK_DIR}/stage-1-openai-spec-bench.json"
python3 - "$STAGE0_CONFIG_PATH" "$STAGE1_CONFIG_PATH" "$RUN_ID" "$MODEL_ID" "$MODEL_PATH" "$SPLIT_LAYER" "$LAYER_END" "$CTX_SIZE" "$N_GPU_LAYERS" "${HOST}:${BINARY_PORT}" "${HOST}:${STAGE1_PORT}" <<'PY'
import json
import sys

(
    stage0_path,
    stage1_path,
    run_id,
    model_id,
    model_path,
    split_layer,
    layer_end,
    ctx_size,
    n_gpu_layers,
    stage0_bind,
    stage1_bind,
) = sys.argv[1:]
split_layer = int(split_layer)
layer_end = int(layer_end)
common = {
    "run_id": run_id,
    "topology_id": "openai-spec-bench-local-split",
    "model_id": model_id,
    "model_path": model_path,
    "ctx_size": int(ctx_size),
    "n_gpu_layers": int(n_gpu_layers),
    "filter_tensors_on_load": True,
    "load_mode": "runtime-slice",
    "kv_server": None,
}
stage0 = {
    **common,
    "stage_id": "stage-0",
    "stage_index": 0,
    "layer_start": 0,
    "layer_end": split_layer,
    "bind_addr": stage0_bind,
    "upstream": None,
    "downstream": {
        "stage_id": "stage-1",
        "stage_index": 1,
        "endpoint": f"tcp://{stage1_bind}",
    },
}
stage1 = {
    **common,
    "stage_id": "stage-1",
    "stage_index": 1,
    "layer_start": split_layer,
    "layer_end": layer_end,
    "bind_addr": stage1_bind,
    "upstream": {
        "stage_id": "stage-0",
        "stage_index": 0,
        "endpoint": "driver",
    },
    "downstream": None,
}
for path, config in ((stage0_path, stage0), (stage1_path, stage1)):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
PY

MIXED_CORPUS="${WORK_DIR}/mixed.jsonl"
WARM_CORPUS="${WORK_DIR}/coding-warm.jsonl"
python3 - "$ROOT/evals/skippy-usecase-corpus.json" "$MIXED_CORPUS" "$WARM_CORPUS" "$MIXED_LIMIT" "$WARM_REPEATS" <<'PY'
import json
import sys

source_path, mixed_path, warm_path, mixed_limit, warm_repeats = sys.argv[1:]
mixed_limit = int(mixed_limit)
warm_repeats = int(warm_repeats)
with open(source_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
use_cases = data["use_cases"]

with open(mixed_path, "w", encoding="utf-8") as handle:
    for index, case in enumerate(use_cases[:mixed_limit]):
        row = {
            "id": f"mixed-{index}-{case['key']}",
            "category": case["key"],
            "session_group": f"mixed-{index}-{case['key']}",
            "prompt": case["prompt"],
        }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

coding_cases = [
    case for case in use_cases
    if case["key"] in {"coding_agent_loop", "issue_fixing", "code_refinement"}
]
if not coding_cases:
    coding_cases = use_cases[:1]
warm_suffix = (
    "\n\nRespond in this exact compact structure:\n"
    "Observation:\n"
    "Patch:\n"
    "Validation:\n"
)
with open(warm_path, "w", encoding="utf-8") as handle:
    for turn in range(warm_repeats):
        case = coding_cases[turn % len(coding_cases)]
        row = {
            "id": f"coding-warm-{turn:02d}-{case['key']}",
            "category": "coding_loop_warm",
            "session_group": "coding-loop-warm",
            "prompt": (
                f"Iteration {turn + 1} of the same coding session.\n"
                f"{case['prompt']}"
                f"{warm_suffix}"
            ),
        }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

for mode in "${BENCH_MODES[@]}"; do
  MODE_DIR="${WORK_DIR}/${mode}"
  mkdir -p "$MODE_DIR"
  for corpus_name in mixed coding-warm; do
    if [[ "$corpus_name" == "mixed" ]]; then
      corpus_path="$MIXED_CORPUS"
    else
      corpus_path="$WARM_CORPUS"
    fi
    SERVER_LOG="${MODE_DIR}/${corpus_name}-serve-binary.log"
    start_server "$mode" "$SERVER_LOG"
    run_chat_corpus "$mode" "$corpus_name" "$corpus_path" "${MODE_DIR}/${corpus_name}.json"
    stop_server
  done
done

SUMMARY_JSON="${WORK_DIR}/summary.json"
SUMMARY_MD="${WORK_DIR}/summary.md"
python3 - "$WORK_DIR" "$SUMMARY_JSON" "$SUMMARY_MD" "$MODES_JOINED" <<'PY'
import json
import math
import pathlib
import sys

work_dir = pathlib.Path(sys.argv[1])
summary_json = pathlib.Path(sys.argv[2])
summary_md = pathlib.Path(sys.argv[3])
modes = sys.argv[4].split()
corpora = ("mixed", "coding-warm")

def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

def proposal_bucket(count):
    count = int(number(count))
    if count <= 0:
        return "0"
    if count >= 4:
        return "4_plus"
    return str(count)

def bump_len_bucket(spec, prefix, bucket):
    key = f"{prefix}_{bucket}"
    if key in spec:
        spec[key] += 1

def parse_telemetry(path):
    spec = {
        "events": 0,
        "sources": {},
        "ngram_events": 0,
        "proposed": 0,
        "accepted": 0,
        "rejected_windows": 0,
        "verify_elapsed_ms": 0.0,
        "verify_stage0_compute_ms": 0.0,
        "verify_activation_encode_ms": 0.0,
        "verify_forward_write_ms": 0.0,
        "verify_chain_compute_ms": 0.0,
        "verify_chain_forward_write_ms": 0.0,
        "verify_chain_downstream_wait_ms": 0.0,
        "verify_chain_total_ms": 0.0,
        "verify_chain_stage_count": 0,
        "verify_chain_token_count": 0,
        "verify_checkpoint_ms": 0.0,
        "verify_checkpointed_windows": 0,
        "verify_skip_checkpoint_windows": 0,
        "ngram_skipped_single_token_verify_windows": 0,
        "verified_len_1": 0,
        "verified_len_2": 0,
        "verified_len_3": 0,
        "verified_len_4_plus": 0,
        "ngram_verified_len_1": 0,
        "ngram_verified_len_2": 0,
        "ngram_verified_len_3": 0,
        "ngram_verified_len_4_plus": 0,
        "ngram_skipped_len_1": 0,
        "ngram_skipped_len_2": 0,
        "ngram_skipped_len_3": 0,
        "ngram_skipped_len_4_plus": 0,
        "ngram_skip_reasons": {},
        "ngram_stop_reasons": {},
        "ngram_match_order_min": None,
        "ngram_match_order_max": 0,
        "repair_elapsed_ms": 0.0,
        "restore_decode_repairs": 0,
        "restore_decode_one_cost_repairs": 0,
        "restore_reverify_repairs": 0,
        "downstream_wait_ms": 0.0,
        "normal_decode_events": 0,
        "normal_decode_elapsed_ms": 0.0,
        "policy_disable_count_max": 0,
        "policy_grow_count_max": 0,
        "policy_shrink_count_max": 0,
        "policy_unproductive_windows_max": 0,
        "policy_accept_rate_last": None,
        "policy_quality_last": None,
        "policy_reason_last": None,
        "policy_auto_repeated_suffix_hits_max": 0,
        "ngram_mode_last": None,
        "ngram_auto_candidate_events": 0,
        "ngram_auto_reasons": {},
        "ngram_auto_candidate_reasons": {},
        "summary": {},
    }
    if not path.exists():
        return spec

    def source_stats(source):
        return spec["sources"].setdefault(source, {
            "windows": 0,
            "proposed": 0,
            "accepted": 0,
            "rejected_windows": 0,
            "verify_elapsed_ms": 0.0,
            "repair_elapsed_ms": 0.0,
            "downstream_wait_ms": 0.0,
            "len_1": 0,
            "len_2": 0,
            "len_3": 0,
            "len_4_plus": 0,
        })

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            attrs = event.get("attributes") or {}
            name = event.get("event")
            if name == "stage.openai_decode_verify_window" and "llama_stage.spec.proposal_source" in attrs:
                if "llama_stage.spec.ngram_mode" in attrs:
                    spec["ngram_mode_last"] = attrs["llama_stage.spec.ngram_mode"]
                if attrs.get("llama_stage.spec.ngram_auto_candidate") is True:
                    spec["ngram_auto_candidate_events"] += 1
                    reason = attrs.get("llama_stage.spec.ngram_auto_reason") or "unknown"
                    spec["ngram_auto_candidate_reasons"][reason] = spec["ngram_auto_candidate_reasons"].get(reason, 0) + 1
                if "llama_stage.spec.ngram_auto_reason" in attrs:
                    reason = attrs["llama_stage.spec.ngram_auto_reason"]
                    spec["ngram_auto_reasons"][reason] = spec["ngram_auto_reasons"].get(reason, 0) + 1
                spec["events"] += 1
                source = attrs.get("llama_stage.spec.proposal_source") or "unknown"
                proposed = int(number(attrs.get("llama_stage.spec.proposed")))
                accepted = int(number(attrs.get("llama_stage.spec.accepted")))
                rejected = attrs.get("llama_stage.spec.rejected") is True
                bucket = attrs.get("llama_stage.spec.proposal_len_bucket") or proposal_bucket(proposed)
                bucket_key = str(bucket).replace("+", "_plus")
                bump_len_bucket(spec, "verified_len", bucket_key)
                per_source = source_stats(source)
                per_source["windows"] += 1
                per_source["proposed"] += proposed
                per_source["accepted"] += accepted
                if rejected:
                    per_source["rejected_windows"] += 1
                if f"len_{bucket_key}" in per_source:
                    per_source[f"len_{bucket_key}"] += 1
                if source == "ngram":
                    spec["ngram_events"] += 1
                    bump_len_bucket(spec, "ngram_verified_len", bucket_key)
                    reason = attrs.get("llama_stage.spec.ngram_proposal_stop_reason") or "unknown"
                    spec["ngram_stop_reasons"][reason] = spec["ngram_stop_reasons"].get(reason, 0) + 1
                    match_min = int(number(attrs.get("llama_stage.spec.ngram_match_order_min")))
                    match_max = int(number(attrs.get("llama_stage.spec.ngram_match_order_max")))
                    if match_min > 0:
                        spec["ngram_match_order_min"] = (
                            match_min
                            if spec["ngram_match_order_min"] is None
                            else min(spec["ngram_match_order_min"], match_min)
                        )
                    spec["ngram_match_order_max"] = max(spec["ngram_match_order_max"], match_max)
                spec["proposed"] += proposed
                spec["accepted"] += accepted
                if rejected:
                    spec["rejected_windows"] += 1
                verify_elapsed = float(number(attrs.get("llama_stage.spec.verify_elapsed_ms")))
                repair_elapsed = float(number(attrs.get("llama_stage.spec.repair_elapsed_ms")))
                downstream_wait = float(number(attrs.get("llama_stage.downstream_wait_ms")))
                spec["verify_elapsed_ms"] += verify_elapsed
                per_source["verify_elapsed_ms"] += verify_elapsed
                per_source["repair_elapsed_ms"] += repair_elapsed
                per_source["downstream_wait_ms"] += downstream_wait
                spec["verify_stage0_compute_ms"] += float(number(attrs.get("llama_stage.stage0_compute_ms")))
                spec["verify_activation_encode_ms"] += float(number(attrs.get("llama_stage.activation_encode_ms")))
                spec["verify_forward_write_ms"] += float(number(attrs.get("llama_stage.forward_write_ms")))
                spec["verify_chain_compute_ms"] += float(number(attrs.get("llama_stage.spec.verify_chain_compute_ms")))
                spec["verify_chain_forward_write_ms"] += float(number(attrs.get("llama_stage.spec.verify_chain_forward_write_ms")))
                spec["verify_chain_downstream_wait_ms"] += float(number(attrs.get("llama_stage.spec.verify_chain_downstream_wait_ms")))
                spec["verify_chain_total_ms"] += float(number(attrs.get("llama_stage.spec.verify_chain_total_ms")))
                spec["verify_chain_stage_count"] += int(number(attrs.get("llama_stage.spec.verify_chain_stage_count")))
                spec["verify_chain_token_count"] += int(number(attrs.get("llama_stage.spec.verify_chain_token_count")))
                spec["verify_checkpoint_ms"] += float(number(attrs.get("llama_stage.spec.verify_checkpoint_ms")))
                if attrs.get("llama_stage.spec.verify_checkpointed") is True:
                    spec["verify_checkpointed_windows"] += 1
                else:
                    spec["verify_skip_checkpoint_windows"] += 1
                spec["repair_elapsed_ms"] += repair_elapsed
                spec["downstream_wait_ms"] += downstream_wait
                repair_path = attrs.get("llama_stage.spec.repair_path")
                if repair_path == "restore_decode":
                    spec["restore_decode_repairs"] += 1
                elif repair_path == "restore_decode_one_cost":
                    spec["restore_decode_one_cost_repairs"] += 1
                elif repair_path == "restore_reverify":
                    spec["restore_reverify_repairs"] += 1
                spec["policy_disable_count_max"] = max(
                    spec["policy_disable_count_max"],
                    int(number(attrs.get("llama_stage.spec.ngram_policy_disable_count"))),
                )
                spec["policy_grow_count_max"] = max(
                    spec["policy_grow_count_max"],
                    int(number(attrs.get("llama_stage.spec.ngram_policy_grow_count"))),
                )
                spec["policy_shrink_count_max"] = max(
                    spec["policy_shrink_count_max"],
                    int(number(attrs.get("llama_stage.spec.ngram_policy_shrink_count"))),
                )
                spec["policy_unproductive_windows_max"] = max(
                    spec["policy_unproductive_windows_max"],
                    int(number(attrs.get("llama_stage.spec.ngram_policy_unproductive_windows"))),
                )
                if "llama_stage.spec.ngram_policy_accept_rate" in attrs:
                    spec["policy_accept_rate_last"] = attrs["llama_stage.spec.ngram_policy_accept_rate"]
                if "llama_stage.spec.ngram_policy_quality" in attrs:
                    spec["policy_quality_last"] = attrs["llama_stage.spec.ngram_policy_quality"]
                if "llama_stage.spec.ngram_policy_reason" in attrs:
                    spec["policy_reason_last"] = attrs["llama_stage.spec.ngram_policy_reason"]
                spec["policy_auto_repeated_suffix_hits_max"] = max(
                    spec["policy_auto_repeated_suffix_hits_max"],
                    int(number(attrs.get("llama_stage.spec.ngram_policy_auto_repeated_suffix_hits"))),
                )
            elif name == "stage.openai_decode_token":
                if "llama_stage.spec.ngram_mode" in attrs:
                    spec["ngram_mode_last"] = attrs["llama_stage.spec.ngram_mode"]
                if attrs.get("llama_stage.spec.ngram_auto_candidate") is True:
                    spec["ngram_auto_candidate_events"] += 1
                    reason = attrs.get("llama_stage.spec.ngram_auto_reason") or "unknown"
                    spec["ngram_auto_candidate_reasons"][reason] = spec["ngram_auto_candidate_reasons"].get(reason, 0) + 1
                if "llama_stage.spec.ngram_auto_reason" in attrs:
                    reason = attrs["llama_stage.spec.ngram_auto_reason"]
                    spec["ngram_auto_reasons"][reason] = spec["ngram_auto_reasons"].get(reason, 0) + 1
                spec["normal_decode_events"] += 1
                spec["normal_decode_elapsed_ms"] += float(number(attrs.get("llama_stage.elapsed_ms")))
                if attrs.get("llama_stage.spec.ngram_skipped_single_token_verify") is True:
                    spec["ngram_skipped_single_token_verify_windows"] += 1
                    skipped = int(number(attrs.get("llama_stage.spec.ngram_skipped_proposed"))) or 1
                    bucket = attrs.get("llama_stage.spec.ngram_skipped_proposal_len_bucket") or proposal_bucket(skipped)
                    bump_len_bucket(spec, "ngram_skipped_len", str(bucket).replace("+", "_plus"))
                    reason = attrs.get("llama_stage.spec.ngram_skipped_reason") or "unknown"
                    spec["ngram_skip_reasons"][reason] = spec["ngram_skip_reasons"].get(reason, 0) + 1
                    stop_reason = attrs.get("llama_stage.spec.ngram_proposal_stop_reason") or "unknown"
                    spec["ngram_stop_reasons"][stop_reason] = spec["ngram_stop_reasons"].get(stop_reason, 0) + 1
                    match_min = int(number(attrs.get("llama_stage.spec.ngram_match_order_min")))
                    match_max = int(number(attrs.get("llama_stage.spec.ngram_match_order_max")))
                    if match_min > 0:
                        spec["ngram_match_order_min"] = (
                            match_min
                            if spec["ngram_match_order_min"] is None
                            else min(spec["ngram_match_order_min"], match_min)
                        )
                    spec["ngram_match_order_max"] = max(spec["ngram_match_order_max"], match_max)
                if "llama_stage.spec.ngram_policy_quality" in attrs:
                    spec["policy_quality_last"] = attrs["llama_stage.spec.ngram_policy_quality"]
                if "llama_stage.spec.ngram_policy_reason" in attrs:
                    spec["policy_reason_last"] = attrs["llama_stage.spec.ngram_policy_reason"]
                spec["policy_auto_repeated_suffix_hits_max"] = max(
                    spec["policy_auto_repeated_suffix_hits_max"],
                    int(number(attrs.get("llama_stage.spec.ngram_policy_auto_repeated_suffix_hits"))),
                )
            elif name == "stage.openai_generation_summary":
                for key, value in attrs.items():
                    if key.startswith("llama_stage.spec."):
                        spec["summary"][key] = value
                if "llama_stage.spec.ngram_mode" in attrs:
                    spec["ngram_mode_last"] = attrs["llama_stage.spec.ngram_mode"]
                if "llama_stage.spec.ngram_auto_reason" in attrs:
                    reason = attrs["llama_stage.spec.ngram_auto_reason"]
                    spec["ngram_auto_reasons"][reason] = spec["ngram_auto_reasons"].get(reason, 0) + 1
    spec["accept_rate"] = spec["accepted"] / spec["proposed"] if spec["proposed"] else None
    spec["verify_ms_per_proposed"] = spec["verify_elapsed_ms"] / spec["proposed"] if spec["proposed"] else None
    spec["verify_stage0_ms_per_window"] = spec["verify_stage0_compute_ms"] / spec["events"] if spec["events"] else None
    spec["verify_encode_ms_per_window"] = spec["verify_activation_encode_ms"] / spec["events"] if spec["events"] else None
    spec["verify_forward_write_ms_per_window"] = spec["verify_forward_write_ms"] / spec["events"] if spec["events"] else None
    spec["verify_chain_total_ms_per_proposed"] = (
        spec["verify_chain_total_ms"] / spec["proposed"] if spec["proposed"] else None
    )
    spec["verify_chain_compute_ms_per_window"] = (
        spec["verify_chain_compute_ms"] / spec["events"] if spec["events"] else None
    )
    spec["verify_chain_forward_write_ms_per_window"] = (
        spec["verify_chain_forward_write_ms"] / spec["events"] if spec["events"] else None
    )
    spec["verify_chain_downstream_wait_ms_per_window"] = (
        spec["verify_chain_downstream_wait_ms"] / spec["events"] if spec["events"] else None
    )
    spec["verify_chain_stages_per_window"] = (
        spec["verify_chain_stage_count"] / spec["events"] if spec["events"] else None
    )
    spec["verify_checkpoint_ms_per_window"] = spec["verify_checkpoint_ms"] / spec["events"] if spec["events"] else None
    spec["repair_ms_per_proposed"] = spec["repair_elapsed_ms"] / spec["proposed"] if spec["proposed"] else None
    spec["downstream_ms_per_window"] = spec["downstream_wait_ms"] / spec["events"] if spec["events"] else None
    spec["normal_decode_ms_per_token"] = (
        spec["normal_decode_elapsed_ms"] / spec["normal_decode_events"]
        if spec["normal_decode_events"]
        else None
    )
    for source in spec["sources"].values():
        source["accept_rate"] = (
            source["accepted"] / source["proposed"] if source["proposed"] else None
        )
        source["verify_ms_per_proposed"] = (
            source["verify_elapsed_ms"] / source["proposed"] if source["proposed"] else None
        )
        source["repair_ms_per_proposed"] = (
            source["repair_elapsed_ms"] / source["proposed"] if source["proposed"] else None
        )
        source["downstream_ms_per_window"] = (
            source["downstream_wait_ms"] / source["windows"] if source["windows"] else None
        )
    return spec

results = []
for mode in modes:
    mode_dir = work_dir / mode
    for corpus in corpora:
        report_path = mode_dir / f"{corpus}.json"
        server_log = mode_dir / f"{corpus}-serve-binary.log"
        telemetry = parse_telemetry(server_log)
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        summary = report["summary"]
        results.append({
            "mode": mode,
            "corpus": corpus,
            "errors": summary["errors"],
            "count": summary["count"],
            "completion_tok_s": summary["completion_tok_s"],
            "elapsed_ms_mean": summary["elapsed_ms_mean"],
            "elapsed_ms_p95": summary["elapsed_ms_p95"],
            "total_wall_ms": summary["total_wall_ms"],
            "telemetry": telemetry,
            "report": str(report_path),
            "server_log": str(server_log),
        })

by_key = {(row["mode"], row["corpus"]): row for row in results}
for corpus in corpora:
    base = by_key.get(("baseline", corpus))
    base_rate = (base or {}).get("completion_tok_s") or 0
    for mode in modes:
        row = by_key[(mode, corpus)]
        rate_value = row["completion_tok_s"] or 0
        row["speedup_vs_baseline"] = (
            rate_value / base_rate if mode != "baseline" and base_rate > 0 and rate_value > 0 else None
        )

summary_json.write_text(
    json.dumps({"modes": modes, "corpora": list(corpora), "results": results}, indent=2) + "\n",
    encoding="utf-8",
)

def fmt(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

def diagnosis(row):
    tel = row["telemetry"]
    if row["mode"] == "baseline":
        return "baseline"
    if tel["proposed"] == 0:
        if tel["ngram_skipped_single_token_verify_windows"]:
            return "single-token skipped"
        return "no proposals"
    accept_rate = tel.get("accept_rate") or 0
    if accept_rate < 0.35:
        return "proposal quality"
    normal = tel.get("normal_decode_ms_per_token")
    verify = tel.get("verify_ms_per_proposed")
    repair = tel.get("repair_ms_per_proposed") or 0
    if normal and verify and verify + repair > normal * 1.15:
        return "verifier cost"
    if tel["policy_disable_count_max"]:
        return "policy backoff"
    return "healthy"

def ngram_len_hist(tel):
    values = []
    for bucket in ("1", "2", "3", "4_plus"):
        values.append(
            tel[f"ngram_verified_len_{bucket}"] + tel[f"ngram_skipped_len_{bucket}"]
        )
    return "/".join(str(value) for value in values)

def proposal_len_hist(tel):
    values = []
    for bucket in ("1", "2", "3", "4_plus"):
        values.append(tel[f"verified_len_{bucket}"])
    return "/".join(str(value) for value in values)

def source_mix(tel):
    sources = tel.get("sources") or {}
    if not sources:
        return "fallback"
    values = []
    for source, stats in sorted(sources.items()):
        proposed = stats.get("proposed") or 0
        accepted = stats.get("accepted") or 0
        rate = stats.get("accept_rate")
        rate_label = "n/a" if rate is None else f"{rate:.2f}"
        values.append(f"{source}:{stats.get('windows', 0)}w/{proposed}p/{accepted}a/{rate_label}")
    return ",".join(values)

def skip_reasons(tel):
    reasons = tel.get("ngram_skip_reasons") or {}
    if not reasons:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in sorted(reasons.items()))

def stop_reasons(tel):
    reasons = tel.get("ngram_stop_reasons") or {}
    if not reasons:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in sorted(reasons.items()))

def match_order_range(tel):
    if tel.get("ngram_match_order_min") is None or not tel.get("ngram_match_order_max"):
        return "n/a"
    return f"{tel['ngram_match_order_min']}-{tel['ngram_match_order_max']}"

def ngram_auto_label(tel):
    mode = tel.get("ngram_mode_last")
    if mode != "auto":
        return mode or "n/a"
    reasons = tel.get("ngram_auto_candidate_reasons") or {}
    if not reasons:
        return f"auto:{tel.get('ngram_auto_candidate_events', 0)}c"
    reason = max(sorted(reasons), key=lambda key: reasons[key])
    return f"auto:{reason}:{tel.get('ngram_auto_candidate_events', 0)}c"

lines = [
    "# OpenAI Speculation Bench",
    "",
    f"Run directory: `{work_dir}`",
    "",
    f"Modes: `{', '.join(modes)}`",
    "",
    "| Corpus | Mode | Errors | Mean ms | p95 ms | tok/s | Speedup | Sources windows/proposed/accepted/rate | Proposed | Accepted | Accept rate | Verify ms/proposed | Chain verify ms/proposed | Chain compute ms/window | Chain downstream ms/window | Chain stages/window | Stage0 ms/window | Downstream ms/window | Proposal len 1/2/3/4+ | N-gram len 1/2/3/4+ | Match n | Stop reasons | Checkpointed/skipped | Fallback decode ms/token | N-gram one-token skips | Skipped reasons | N-gram auto | Policy quality | Policy reason | Auto suffix hits | Checkpoint ms/window | Repair ms/proposed | Decode-one repairs | Reverify repairs | Policy disables | Diagnosis |",
    "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
]
for corpus in corpora:
    for mode in modes:
        row = by_key[(mode, corpus)]
        tel = row["telemetry"]
        lines.append(
            "| {corpus} | {mode} | {errors}/{count} | {mean} | {p95} | {rate} | {speedup} | {sources} | {proposed} | {accepted} | {accept_rate} | {verify} | {chain_verify} | {chain_compute} | {chain_downstream} | {chain_stages} | {stage0} | {downstream} | {proposal_len_hist} | {ngram_len_hist} | {match_order} | {stop_reasons} | {checkpointed}/{skipped} | {fallback_decode} | {ngram_skips} | {skip_reasons} | {ngram_auto} | {policy_quality} | {policy_reason} | {auto_suffix_hits} | {checkpoint_ms} | {repair} | {decode_repairs} | {reverify_repairs} | {disables} | {diagnosis} |".format(
                corpus=corpus,
                mode=mode,
                errors=row["errors"],
                count=row["count"],
                mean=fmt(row["elapsed_ms_mean"]),
                p95=fmt(row["elapsed_ms_p95"]),
                rate=fmt(row["completion_tok_s"]),
                speedup=fmt(row.get("speedup_vs_baseline")),
                sources=source_mix(tel),
                proposed=tel["proposed"],
                accepted=tel["accepted"],
                accept_rate=fmt(tel.get("accept_rate")),
                verify=fmt(tel.get("verify_ms_per_proposed")),
                chain_verify=fmt(tel.get("verify_chain_total_ms_per_proposed")),
                chain_compute=fmt(tel.get("verify_chain_compute_ms_per_window")),
                chain_downstream=fmt(tel.get("verify_chain_downstream_wait_ms_per_window")),
                chain_stages=fmt(tel.get("verify_chain_stages_per_window")),
                stage0=fmt(tel.get("verify_stage0_ms_per_window")),
                downstream=fmt(tel.get("downstream_ms_per_window")),
                proposal_len_hist=proposal_len_hist(tel),
                ngram_len_hist=ngram_len_hist(tel),
                match_order=match_order_range(tel),
                stop_reasons=stop_reasons(tel),
                checkpointed=tel["verify_checkpointed_windows"],
                skipped=tel["verify_skip_checkpoint_windows"],
                fallback_decode=fmt(tel.get("normal_decode_ms_per_token")),
                ngram_skips=tel["ngram_skipped_single_token_verify_windows"],
                skip_reasons=skip_reasons(tel),
                ngram_auto=ngram_auto_label(tel),
                policy_quality=tel.get("policy_quality_last") or "n/a",
                policy_reason=tel.get("policy_reason_last") or "n/a",
                auto_suffix_hits=tel.get("policy_auto_repeated_suffix_hits_max") or 0,
                checkpoint_ms=fmt(tel.get("verify_checkpoint_ms_per_window")),
                repair=fmt(tel.get("repair_ms_per_proposed")),
                decode_repairs=tel["restore_decode_repairs"] + tel["restore_decode_one_cost_repairs"],
                reverify_repairs=tel["restore_reverify_repairs"],
                disables=tel["policy_disable_count_max"],
                diagnosis=diagnosis(row),
            )
        )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY

echo "OpenAI speculation bench complete"
echo "  work dir: $WORK_DIR"
echo "  summary:  $SUMMARY_MD"
