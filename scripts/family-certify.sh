#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
usage: scripts/family-certify.sh --family FAMILY --target-model TARGET_GGUF [options] [-- prompt flags...]

Runs a reproducible family certification pass and writes reports under
target/family-certify/<run-id>/<family>/<model-slug>/.

Required:
  --family FAMILY             model family label, for example qwen3-dense or gemma2
  --target-model GGUF         target GGUF to certify

Identity options:
  --model-id ID               public model coordinate, for example org/repo:Q4_K_M;
                              required for local paths outside the Hugging Face cache

Correctness options:
  --layer-end N               final layer for staged correctness lanes; default: 30
  --split-layer N             single split layer for single-step and dtype lanes
  --splits CSV                chain split layers, for example 10,20
  --activation-width N        hidden width for exact state-handoff lane
  --prompt TEXT               correctness prompt; default: Hello
  --ctx-size N                context size; default: 128
  --n-gpu-layers N            llama.cpp GPU layers; default: 999
  --wire-dtype DTYPE          default activation wire dtype; default: f16
  --wire-dtypes CSV           dtype matrix list; default: f32,f16,q8
  --allow-mismatch            allow mismatch in single-step and chain lanes
  --strict-dtype              make dtype-matrix mismatch a hard failure
  --skip-build                do not build correctness binaries first
  --skip-correctness          skip all correctness/state lanes
  --skip-dtype                skip dtype matrix
  --skip-state                skip state handoff

Speculative options:
  --draft-model GGUF          draft GGUF for draft speculative lanes
  --with-ngram                rejected in mesh-llm; standalone n-gram pools are not imported
  --spec-modes CSV            rejected in mesh-llm; use llama-spec-bench for draft checks
  --corpus JSONL              corpus path; default: target/bench-corpora/smoke/corpus.jsonl
  --corpus-limit N            prompt limit for llama-spec-bench
  --per-family-limit N        accepted for compatibility but not used by llama-spec-bench
  --spec-window N             speculative window; default: 8
  --max-tokens N              max new tokens per prompt; default: 24
  --decode-timeout N          decode timeout seconds; default: 120
  --qwen-state-baseline BYTES Qwen dense state baseline; default: 115388
  --state-reject-ratio N      reject exact state mobility over Nx baseline; default: 100
  --recurrent-ranges CSV      recurrent ranges, for example 0..12,16..24
  --recurrent-all             mark the whole certified layer range recurrent

Output options:
  --cert-root DIR             output root; default: target/family-certify
  --run-id ID                 output run id; default: timestamp
  --port-base N               base port; default: 19000 + random offset
  -h, --help                  show this help

Arguments after -- are currently ignored by the mesh-llm import.
EOF
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ROOT/$1" ;;
  esac
}

slugify() {
  printf '%s' "$1" | tr '/[:upper:]' '-[:lower:]' | tr -cs 'a-z0-9._-' '-'
}

family_id() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//'
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v arg '%q' "$arg"
    out+="${arg} "
  done
  printf '%s' "${out% }"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for family certification manifests" >&2
  exit 1
fi

FAMILY=""
TARGET_MODEL=""
MODEL_ID=""
DRAFT_MODEL=""
LAYER_END="30"
SPLIT_LAYER=""
SPLITS=""
ACTIVATION_WIDTH=""
PROMPT="Hello"
CTX_SIZE="128"
N_GPU_LAYERS="999"
WIRE_DTYPE="f16"
WIRE_DTYPES="f32,f16,q8"
ALLOW_MISMATCH=0
STRICT_DTYPE=0
SKIP_BUILD=0
SKIP_CORRECTNESS=0
SKIP_DTYPE=0
SKIP_STATE=0
WITH_NGRAM=0
SPEC_MODES=""
CORPUS="target/bench-corpora/smoke/corpus.jsonl"
CORPUS_LIMIT=""
PER_FAMILY_LIMIT=""
SPEC_WINDOW="8"
MAX_TOKENS="24"
DECODE_TIMEOUT="120"
QWEN_STATE_BASELINE_BYTES="115388"
STATE_REJECT_RATIO="100"
RECURRENT_RANGES=""
RECURRENT_ALL=0
CERT_ROOT="target/family-certify"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PORT_BASE="$((19000 + RANDOM % 1000))"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2 ;;
    --target-model|--model) TARGET_MODEL="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --draft-model) DRAFT_MODEL="$2"; shift 2 ;;
    --layer-end) LAYER_END="$2"; shift 2 ;;
    --split-layer) SPLIT_LAYER="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --activation-width) ACTIVATION_WIDTH="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --ctx-size) CTX_SIZE="$2"; shift 2 ;;
    --n-gpu-layers) N_GPU_LAYERS="$2"; shift 2 ;;
    --wire-dtype) WIRE_DTYPE="$2"; shift 2 ;;
    --wire-dtypes) WIRE_DTYPES="$2"; shift 2 ;;
    --allow-mismatch) ALLOW_MISMATCH=1; shift ;;
    --strict-dtype) STRICT_DTYPE=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-correctness) SKIP_CORRECTNESS=1; shift ;;
    --skip-dtype) SKIP_DTYPE=1; shift ;;
    --skip-state) SKIP_STATE=1; shift ;;
    --with-ngram) WITH_NGRAM=1; shift ;;
    --spec-modes) SPEC_MODES="$2"; shift 2 ;;
    --corpus) CORPUS="$2"; shift 2 ;;
    --corpus-limit) CORPUS_LIMIT="$2"; shift 2 ;;
    --per-family-limit) PER_FAMILY_LIMIT="$2"; shift 2 ;;
    --spec-window) SPEC_WINDOW="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --decode-timeout) DECODE_TIMEOUT="$2"; shift 2 ;;
    --qwen-state-baseline) QWEN_STATE_BASELINE_BYTES="$2"; shift 2 ;;
    --state-reject-ratio) STATE_REJECT_RATIO="$2"; shift 2 ;;
    --recurrent-ranges) RECURRENT_RANGES="$2"; shift 2 ;;
    --recurrent-all) RECURRENT_ALL=1; shift ;;
    --cert-root) CERT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 2 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$FAMILY" || -z "$TARGET_MODEL" ]]; then
  echo "--family and --target-model are required" >&2
  usage
  exit 2
fi

TARGET_MODEL_PATH="$(abs_path "$TARGET_MODEL")"
if [[ ! -f "$TARGET_MODEL_PATH" ]]; then
  echo "target model not found: $TARGET_MODEL_PATH" >&2
  exit 1
fi
DISPLAY_MODEL_ID="${MODEL_ID:-$FAMILY}"
if [[ -n "$DRAFT_MODEL" ]]; then
  DRAFT_MODEL_PATH="$(abs_path "$DRAFT_MODEL")"
  if [[ ! -f "$DRAFT_MODEL_PATH" ]]; then
    echo "draft model not found: $DRAFT_MODEL_PATH" >&2
    exit 1
  fi
else
  DRAFT_MODEL_PATH=""
fi

CERT_ROOT_PATH="$(abs_path "$CERT_ROOT")"
MODEL_SLUG="$(slugify "$(basename "$TARGET_MODEL_PATH" .gguf)")"
FAMILY_SLUG="$(slugify "$FAMILY")"
OUT_DIR="$CERT_ROOT_PATH/$RUN_ID/$FAMILY_SLUG/$MODEL_SLUG"
LOG_DIR="$OUT_DIR/logs"
REPORT_DIR="$OUT_DIR/reports"
COMMANDS_JSONL="$OUT_DIR/commands.jsonl"
SUMMARY_MD="$OUT_DIR/summary.md"
MANIFEST_JSON="$OUT_DIR/manifest.json"
CAPABILITY_DRAFT_JSON="$OUT_DIR/capability-draft.json"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
: > "$COMMANDS_JSONL"

record_event() {
  local name="$1"
  local status="$2"
  local exit_code="$3"
  local log="$4"
  local report="$5"
  local note="$6"
  jq -n \
    --arg name "$name" \
    --arg status "$status" \
    --argjson exit_code "$exit_code" \
    --arg log "$log" \
    --arg report "$report" \
    --arg note "$note" \
    '{
      name:$name,
      status:$status,
      exit_code:$exit_code,
      log:($log | if length > 0 then . else null end),
      report:($report | if length > 0 then . else null end),
      note:($note | if length > 0 then . else null end)
    } | with_entries(select(.value != null))' \
    >> "$COMMANDS_JSONL"
}

run_logged() {
  local name="$1"
  local report="$2"
  shift 2
  local log="$LOG_DIR/$name.log"
  local exit_code=0
  local command
  command="$(quote_cmd "$@")"
  {
    printf '+ %s\n\n' "$command"
    "$@"
  } >"$log" 2>&1 || exit_code=$?
  local status="pass"
  if (( exit_code != 0 )); then
    status="fail"
  fi
  record_event "$name" "$status" "$exit_code" "$log" "$report" ""
  printf '%s: %s (exit %s)\n' "$name" "$status" "$exit_code"
}

model_identity_json() {
  local model_id="$1"
  local model_path="$2"
  python3 - "$model_id" "$model_path" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

model_id, model_path = sys.argv[1:]
path = Path(model_path)
parts = path.parts
identity = {
    "model_id": model_id,
}

for index, part in enumerate(parts):
    if part.startswith("models--") and index + 3 < len(parts) and parts[index + 1] == "snapshots":
        repo = part.removeprefix("models--").replace("--", "/")
        revision = parts[index + 2]
        file = "/".join(parts[index + 3:])
        basename = Path(file).name
        distribution_id = basename[:-5] if basename.endswith(".gguf") else basename
        distribution_id = re.sub(r"-00001-of-[0-9]{5}$", "", distribution_id)
        selector = None
        if basename.endswith(".gguf"):
            stem = distribution_id
            match = re.search(r"(UD-[A-Z0-9_]+|IQ[0-9A-Z_]+|Q[0-9A-Z_]+|BF16|F16|F32)$", stem, re.I)
            if match:
                selector = match.group(1)
        identity.update({
            "source_repo": repo,
            "source_revision": revision,
            "source_file": file,
            "canonical_ref": f"{repo}@{revision}/{file}",
            "distribution_id": distribution_id,
        })
        if selector:
            identity["selector"] = selector
        break

print(json.dumps(identity, sort_keys=True))
PY
}

maybe_build() {
  if (( SKIP_BUILD != 0 )); then
    record_event "build" "skipped" 0 "" "" "--skip-build"
    return
  fi
  run_logged "build" "" \
    env LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$ROOT/.deps/llama.cpp/build-stage-abi-static}" \
    cargo build -p skippy-correctness -p skippy-server -p llama-spec-bench
}

correctness_common=(
  --model "$TARGET_MODEL_PATH"
  --layer-end "${LAYER_END:-30}"
  --ctx-size "$CTX_SIZE"
  --n-gpu-layers "$N_GPU_LAYERS"
  --prompt "$PROMPT"
  --stage-server-bin "$ROOT/target/debug/skippy-server"
)
if [[ -n "$MODEL_ID" ]]; then
  correctness_common+=(--model-id "$MODEL_ID")
fi

echo "family certification"
echo "  family: $FAMILY"
echo "  model_id: $DISPLAY_MODEL_ID"
echo "  target: $TARGET_MODEL_PATH"
echo "  output: $OUT_DIR"

maybe_build

if (( SKIP_CORRECTNESS != 0 )); then
  record_event "correctness" "skipped" 0 "" "" "--skip-correctness"
else
  if [[ -n "$SPLIT_LAYER" && -n "$LAYER_END" ]]; then
    single_args=(
      "$ROOT/target/debug/skippy-correctness"
      single-step
      "${correctness_common[@]}"
      --split-layer "$SPLIT_LAYER"
      --stage1-bind-addr "127.0.0.1:$((PORT_BASE + 1))"
      --activation-wire-dtype "$WIRE_DTYPE"
      --report-out "$REPORT_DIR/single-step.json"
    )
    if (( ALLOW_MISMATCH != 0 )); then
      single_args+=(--allow-mismatch)
    fi
    run_logged "single-step" "$REPORT_DIR/single-step.json" "${single_args[@]}"
  else
    record_event "single-step" "skipped" 0 "" "" "requires --split-layer and --layer-end"
  fi

  if [[ -n "$SPLITS" && -n "$LAYER_END" ]]; then
    IFS=',' read -r -a chain_split_parts <<< "$SPLITS"
    if (( ${#chain_split_parts[@]} != 2 )); then
      record_event "chain" "skipped" 0 "" "" "chain requires exactly two split indexes; single-step covers two-stage boundaries"
      printf 'chain: skipped (requires exactly two split indexes)\n'
    else
      chain_args=(
        "$ROOT/target/debug/skippy-correctness"
        chain
        "${correctness_common[@]}"
        --splits "$SPLITS"
        --stage1-bind-addr "127.0.0.1:$((PORT_BASE + 11))"
        --stage2-bind-addr "127.0.0.1:$((PORT_BASE + 12))"
        --activation-wire-dtype "$WIRE_DTYPE"
        --report-out "$REPORT_DIR/chain.json"
      )
      if (( ALLOW_MISMATCH != 0 )); then
        chain_args+=(--allow-mismatch)
      fi
      run_logged "chain" "$REPORT_DIR/chain.json" "${chain_args[@]}"
    fi
  else
    record_event "chain" "skipped" 0 "" "" "requires --splits and --layer-end"
  fi

  if (( SKIP_DTYPE != 0 )); then
    record_event "dtype-matrix" "skipped" 0 "" "" "--skip-dtype"
  elif [[ -n "$SPLIT_LAYER" && -n "$LAYER_END" ]]; then
    dtype_args=(
      "$ROOT/target/debug/skippy-correctness"
      dtype-matrix
      "${correctness_common[@]}"
      --split-layer "$SPLIT_LAYER"
      --stage1-bind-addr "127.0.0.1:$((PORT_BASE + 21))"
      --dtypes "$WIRE_DTYPES"
      --report-out "$REPORT_DIR/dtype-matrix.json"
    )
    if (( STRICT_DTYPE == 0 )); then
      dtype_args+=(--allow-mismatch)
    fi
    run_logged "dtype-matrix" "$REPORT_DIR/dtype-matrix.json" "${dtype_args[@]}"
  else
    record_event "dtype-matrix" "skipped" 0 "" "" "requires --split-layer and --layer-end"
  fi

  if (( SKIP_STATE != 0 )); then
    record_event "state-handoff" "skipped" 0 "" "" "--skip-state"
  elif [[ -n "$ACTIVATION_WIDTH" && -n "$LAYER_END" ]]; then
    state_args=(
      "$ROOT/target/debug/skippy-correctness"
      state-handoff
      "${correctness_common[@]}"
      --activation-width "$ACTIVATION_WIDTH"
      --source-bind-addr "127.0.0.1:$((PORT_BASE + 31))"
      --restore-bind-addr "127.0.0.1:$((PORT_BASE + 32))"
      --activation-wire-dtype "$WIRE_DTYPE"
      --report-out "$REPORT_DIR/state-handoff.json"
    )
    run_logged "state-handoff" "$REPORT_DIR/state-handoff.json" "${state_args[@]}"
  else
    record_event "state-handoff" "skipped" 0 "" "" "requires --activation-width and --layer-end"
  fi
fi

if [[ "$WITH_NGRAM" == "1" || -n "$SPEC_MODES" ]]; then
  echo "mesh-llm does not import standalone n-gram prompt-spec corpus modes" >&2
  exit 2
fi

if [[ -n "$DRAFT_MODEL_PATH" ]]; then
  SPEC_DIR="$OUT_DIR/speculative"
  mkdir -p "$SPEC_DIR"
  spec_args=(
    env
    "LLAMA_STAGE_BUILD_DIR=${LLAMA_STAGE_BUILD_DIR:-$ROOT/.deps/llama.cpp/build-stage-abi-static}"
    "$ROOT/target/debug/llama-spec-bench"
    --target-model-path "$TARGET_MODEL_PATH"
    --draft-model-path "$DRAFT_MODEL_PATH"
    --prompt-corpus "$(abs_path "$CORPUS")"
    --max-new-tokens "$MAX_TOKENS"
    --speculative-window "$SPEC_WINDOW"
    --ctx-size "$CTX_SIZE"
    --n-gpu-layers "$N_GPU_LAYERS"
    --json-out "$SPEC_DIR/spec-bench.json"
  )
  if [[ -n "$CORPUS_LIMIT" ]]; then
    spec_args+=(--prompt-limit "$CORPUS_LIMIT")
  fi
  run_logged "llama-spec-bench" "$SPEC_DIR/spec-bench.json" "${spec_args[@]}"
else
  record_event "llama-spec-bench" "skipped" 0 "" "" "requires --draft-model"
fi

json_or_null() {
  if [[ -f "$1" ]]; then
    jq -c '.' "$1"
  else
    printf 'null\n'
  fi
}

SINGLE_REPORT_JSON="$(json_or_null "$REPORT_DIR/single-step.json")"
CHAIN_REPORT_JSON="$(json_or_null "$REPORT_DIR/chain.json")"
DTYPE_REPORT_JSON="$(json_or_null "$REPORT_DIR/dtype-matrix.json")"
STATE_REPORT_JSON="$(json_or_null "$REPORT_DIR/state-handoff.json")"
TARGET_MODEL_IDENTITY_JSON="$(model_identity_json "$DISPLAY_MODEL_ID" "$TARGET_MODEL_PATH")"

jq -n \
  --arg schema_version "1" \
  --arg generated_by "scripts/family-certify.sh" \
  --arg family "$FAMILY" \
  --arg family_id "$(family_id "$FAMILY")" \
  --arg model_id "$DISPLAY_MODEL_ID" \
  --arg target_model "$TARGET_MODEL_PATH" \
  --argjson target_model_identity "$TARGET_MODEL_IDENTITY_JSON" \
  --arg layer_end "$LAYER_END" \
  --arg default_wire_dtype "$WIRE_DTYPE" \
  --arg qwen_state_baseline_bytes "$QWEN_STATE_BASELINE_BYTES" \
  --arg state_reject_ratio "$STATE_REJECT_RATIO" \
  --arg recurrent_ranges "$RECURRENT_RANGES" \
  --argjson recurrent_all "$RECURRENT_ALL" \
  --argjson single "$SINGLE_REPORT_JSON" \
  --argjson chain "$CHAIN_REPORT_JSON" \
  --argjson dtype "$DTYPE_REPORT_JSON" \
  --argjson state "$STATE_REPORT_JSON" \
  --argjson commands "$(jq -s '.' "$COMMANDS_JSONL")" \
  '
  def first_value(xs): xs | map(select(. != null)) | first;
  def q8_validation($dtype):
    if $dtype == null then "untested"
    else
      ([$dtype.results[]? | select(.split.wire_dtype == "q8") | .matches]) as $q8
      | if ($q8 | length) == 0 then "untested"
        elif all($q8[]; . == true) then "validated"
        else "rejected"
        end
    end;
  def state_mobility($state; $baseline; $reject_ratio):
    if $state == null then "untested"
    elif (($state.state_bytes // 0) > (($baseline | tonumber) * ($reject_ratio | tonumber))) then "rejected_too_large"
    elif $state.matches == true then "accepted"
    else "untested"
    end;
  def parse_ranges($csv):
    if $csv == "" then []
    else
      $csv
      | split(",")
      | map(capture("^(?<start>[0-9]+)(\\.\\.|-)(?<end>[0-9]+)$") | {start:(.start|tonumber), end:(.end|tonumber)})
    end;
  def family_split_constraints($family_id):
    if $family_id == "gemma4_e4b" then
      [{
        kind:"shared_kv_producer_consumer",
        range:{start:0,end:0},
        forbidden_boundaries:[12,14,24,28],
        reject_boundary_inside:false,
        reason:"known-bad Gemma4 E4B shared-KV producer/consumer boundary; keep this cut rejected unless KV replay or KV transfer is added"
      }]
    else [] end;
  def family_sidebands($family_id; $layer_count):
    if $family_id == "gemma4_e4b" then
      [{
        kind:"token_ids",
        first_required_layer:$layer_count,
        reason:"Gemma4 E4B downstream slices require token-id sideband to rebuild the auxiliary per-layer input path"
      }]
    else [] end;
  ($layer_end | tonumber) as $layer_count
  | first_value([
      $single.split.activation_width?,
      $chain.activation_width?,
      $dtype.results[]?.split.activation_width?
    ]) as $activation_width
  | first_value([
      $single.split.boundary.payload_bytes?,
      $chain.stages[]?.payload_bytes?,
      $dtype.results[]?.split.boundary.payload_bytes?
    ]) as $activation_payload_bytes
  | first_value([
      $single.split.boundary.wire_payload_bytes?,
      $chain.stages[]?.wire_payload_bytes?
    ]) as $default_wire_payload_bytes
  | first_value([
      $dtype.results[]? | select(.split.wire_dtype == "q8") | .split.boundary.wire_payload_bytes?
    ]) as $q8_wire_payload_bytes
  | (if $recurrent_all == 1 then [{start:0,end:$layer_count}] else parse_ranges($recurrent_ranges) end) as $ranges
  | q8_validation($dtype) as $q8
  | state_mobility($state; $qwen_state_baseline_bytes; $state_reject_ratio) as $state_mobility
  | {
      schema_version:($schema_version|tonumber),
      status:"draft",
      generated_by:$generated_by,
      family:$family,
      model_id:$model_id,
      target_model:$target_model,
      target_model_identity:$target_model_identity,
      planner_record:(
        if $activation_width == null then null
        else {
          family_id:$family_id,
          layer_count:$layer_count,
          activation_width:$activation_width,
          default_wire_dtype:$default_wire_dtype,
          q8_wire_validation:$q8,
          exact_state_mobility:$state_mobility,
          recurrent_ranges:$ranges,
          split_constraints:family_split_constraints($family_id),
          sidebands:family_sidebands($family_id; $layer_count)
        }
        end
      ),
      evidence:{
        activation:{
          activation_width:$activation_width,
          payload_bytes:$activation_payload_bytes,
          default_wire_payload_bytes:$default_wire_payload_bytes,
          q8_wire_payload_bytes:$q8_wire_payload_bytes
        },
        dtype_matrix:{
          q8_wire_validation:$q8,
          report:(
            if $dtype == null then null
            else {
              status:$dtype.status,
              mismatch_count:$dtype.mismatch_count,
              dtype_count:$dtype.dtype_count
            }
            end
          )
        },
        state_handoff:{
          exact_state_mobility:$state_mobility,
          qwen_state_baseline_bytes:($qwen_state_baseline_bytes|tonumber),
          state_reject_ratio:($state_reject_ratio|tonumber),
          report:(
            if $state == null then null
            else {
              status:$state.status,
              matches:$state.matches,
              state_bytes:$state.state_bytes,
              roundtrip_state_bytes:$state.roundtrip_state_bytes,
              vs_qwen_baseline:(($state.state_bytes // 0) / ($qwen_state_baseline_bytes|tonumber))
            }
            end
          )
        },
        commands:$commands
      }
    }
  ' > "$CAPABILITY_DRAFT_JSON"

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
jq -n \
  --arg run_id "$RUN_ID" \
  --arg family "$FAMILY" \
  --arg model_id "$DISPLAY_MODEL_ID" \
  --arg target_model "$TARGET_MODEL_PATH" \
  --argjson target_model_identity "$TARGET_MODEL_IDENTITY_JSON" \
  --arg draft_model "$DRAFT_MODEL_PATH" \
  --arg output_dir "$OUT_DIR" \
  --arg git_commit "$GIT_COMMIT" \
  --arg layer_end "$LAYER_END" \
  --arg split_layer "$SPLIT_LAYER" \
  --arg splits "$SPLITS" \
  --arg activation_width "$ACTIVATION_WIDTH" \
  --arg ctx_size "$CTX_SIZE" \
  --arg n_gpu_layers "$N_GPU_LAYERS" \
  --arg wire_dtype "$WIRE_DTYPE" \
  --arg wire_dtypes "$WIRE_DTYPES" \
  --arg corpus "$(abs_path "$CORPUS")" \
  --arg spec_modes "$SPEC_MODES" \
  --arg capability_draft "$CAPABILITY_DRAFT_JSON" \
  --argjson commands "$(jq -s '.' "$COMMANDS_JSONL")" \
  '{
    run_id:$run_id,
    family:$family,
    model_id:$model_id,
    target_model:$target_model,
    target_model_identity:$target_model_identity,
    draft_model:($draft_model | if length > 0 then . else null end),
    output_dir:$output_dir,
    git_commit:$git_commit,
    correctness:{
      layer_end:$layer_end,
      split_layer:$split_layer,
      splits:$splits,
      activation_width:$activation_width,
      ctx_size:$ctx_size,
      n_gpu_layers:$n_gpu_layers,
      wire_dtype:$wire_dtype,
      wire_dtypes:$wire_dtypes
    },
    speculative:{
      corpus:$corpus,
      modes:$spec_modes
    },
    capability_draft:$capability_draft,
    commands:$commands
  }' > "$MANIFEST_JSON"

{
  echo "# Family Certification"
  echo
  echo "| Field | Value |"
  echo "| --- | --- |"
  echo "| Family | \`$FAMILY\` |"
  echo "| Model ID | \`$DISPLAY_MODEL_ID\` |"
  echo "| Target | \`$TARGET_MODEL_PATH\` |"
  if [[ -n "$DRAFT_MODEL_PATH" ]]; then
    echo "| Draft | \`$DRAFT_MODEL_PATH\` |"
  fi
  echo "| Run ID | \`$RUN_ID\` |"
  echo "| Git commit | \`$GIT_COMMIT\` |"
  echo
  echo "## Command Results"
  echo
  echo "| Lane | Status | Exit | Report | Log | Note |"
  echo "| --- | --- | ---: | --- | --- | --- |"
  jq -r '
    . |
    "| \(.name) | \(.status) | \(.exit_code) | " +
    (if .report then "`\(.report)`" else "" end) + " | " +
    (if .log then "`\(.log)`" else "" end) + " | " +
    (if .note then .note else "" end) + " |"
  ' "$COMMANDS_JSONL"
  echo
  echo "## Files"
  echo
  echo "- Manifest: \`$MANIFEST_JSON\`"
  echo "- Capability draft: \`$CAPABILITY_DRAFT_JSON\`"
  echo "- Command log index: \`$COMMANDS_JSONL\`"
  if [[ -f "$OUT_DIR/speculative/summary.tsv" ]]; then
    echo "- Speculative summary: \`$OUT_DIR/speculative/summary.tsv\`"
  fi
  if [[ -f "$OUT_DIR/speculative/summary-by-family.tsv" ]]; then
    echo "- Speculative family summary: \`$OUT_DIR/speculative/summary-by-family.tsv\`"
  fi
} > "$SUMMARY_MD"

echo
echo "summary:  $SUMMARY_MD"
echo "manifest: $MANIFEST_JSON"
echo "capability draft: $CAPABILITY_DRAFT_JSON"

FAILED_COUNT="$(jq -s '[.[] | select(.status == "fail")] | length' "$COMMANDS_JSONL")"
if (( FAILED_COUNT > 0 )); then
  echo "failed lanes: $FAILED_COUNT"
  exit 1
fi
