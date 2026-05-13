#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-$ROOT/experiments/mtp/vanilla}"
MODEL="${MODEL:?MODEL is required}"
CORPUS="${CORPUS:?CORPUS is required}"
LLAMA="${LLAMA:?LLAMA is required}"
MAX_TOKENS="${MAX_TOKENS:-192}"
PROMPT_LIMIT="${PROMPT_LIMIT:-8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
WARMUP_TOKENS="${WARMUP_TOKENS:-64}"
CORPUS_WARMUP_TOKENS="${CORPUS_WARMUP_TOKENS:-192}"
WINDOWS="${WINDOWS:-1,2,3,4,5}"
BASELINE_MIN_TOK_S="${BASELINE_MIN_TOK_S:-17.3}"
COOLDOWN_SECS="${COOLDOWN_SECS:-60}"
MAX_GUARD_ATTEMPTS="${MAX_GUARD_ATTEMPTS:-3}"
MAX_LOAD_1="${MAX_LOAD_1:-4.0}"
MAX_BACKGROUND_CPU="${MAX_BACKGROUND_CPU:-25.0}"
QUIET_POLL_SECS="${QUIET_POLL_SECS:-15}"
QUIET_WAIT_MAX_SECS="${QUIET_WAIT_MAX_SECS:-900}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUMMARY_JSON="$OUT_DIR/guarded-vanilla-sweep-$RUN_TAG.jsonl"

mkdir -p "$OUT_DIR"

snapshot_host() {
  local label="$1"
  local path="$2"
  {
    echo "=== $label ==="
    date -u
    uptime || true
    echo "--- top cpu ---"
    ps -Ao pid,pcpu,pmem,command | sort -k2 -nr | head -14 || true
    echo "--- vm_stat ---"
    vm_stat || true
    echo "--- memory_pressure ---"
    memory_pressure -Q 2>/dev/null || true
    echo "--- pmset therm ---"
    pmset -g therm 2>/dev/null || true
    echo "--- pmset batt ---"
    pmset -g batt 2>/dev/null || true
  } >"$path"
}

current_load_1() {
  sysctl -n vm.loadavg | awk '{print $2}'
}

top_background_cpu() {
  (ps -Ao pcpu,command \
    | awk '
      NR == 1 { next }
      $0 ~ /(run_guarded_vanilla_mtp_sweep|run_vanilla_mtp_sweep|llama-server|python3|ssh|sshd|zsh|bash|awk|sort|head|ps -Ao)/ { next }
      { print $1; exit }
    ') || true
}

wait_for_quiet_host() {
  local label="$1"
  local waited=0
  local load_1
  local top_cpu

  while true; do
    load_1="$(current_load_1)"
    top_cpu="$(top_background_cpu)"
    top_cpu="${top_cpu:-0}"
    if awk -v load="$load_1" -v max_load="$MAX_LOAD_1" -v cpu="$top_cpu" -v max_cpu="$MAX_BACKGROUND_CPU" \
      'BEGIN { exit !((load <= max_load) && (cpu <= max_cpu)) }'; then
      echo "quiet host for $label: load_1=$load_1 top_background_cpu=$top_cpu"
      return 0
    fi
    echo "waiting for quiet host for $label: load_1=$load_1 top_background_cpu=$top_cpu"
    if (( waited >= QUIET_WAIT_MAX_SECS )); then
      echo "timed out waiting for quiet host for $label" >&2
      snapshot_host "quiet-timeout-$label" "$OUT_DIR/guard-${RUN_TAG}-quiet-timeout-${label}.txt"
      return 1
    fi
    sleep "$QUIET_POLL_SECS"
    waited=$((waited + QUIET_POLL_SECS))
  done
}

latest_json_for() {
  local condition="$1"
  ls -t "$OUT_DIR"/llama-server-"$condition"-*.json | awk 'NR == 1 { first = $0 } END { print first }'
}

json_tok_s() {
  jq -r '.summary.completion_tok_s' "$1"
}

json_mean_ms() {
  jq -r '.summary.mean_latency_ms' "$1"
}

run_runner() {
  local include_baseline="$1"
  local windows="$2"
  INCLUDE_BASELINE="$include_baseline" \
  WINDOWS="$windows" \
  OUT_DIR="$OUT_DIR" \
  MODEL="$MODEL" \
  CORPUS="$CORPUS" \
  LLAMA="$LLAMA" \
  MAX_TOKENS="$MAX_TOKENS" \
  PROMPT_LIMIT="$PROMPT_LIMIT" \
  WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  WARMUP_TOKENS="$WARMUP_TOKENS" \
  CORPUS_WARMUP_TOKENS="$CORPUS_WARMUP_TOKENS" \
    python3 "$OUT_DIR/run_vanilla_mtp_sweep.py"
}

run_baseline_guard() {
  local phase="$1"
  local window="$2"
  local attempt
  local json_path
  local tok_s

  for attempt in $(seq 1 "$MAX_GUARD_ATTEMPTS"); do
    wait_for_quiet_host "${phase}-w${window}-attempt${attempt}"
    snapshot_host "${phase}-w${window}-attempt${attempt}-before" \
      "$OUT_DIR/guard-${RUN_TAG}-${phase}-w${window}-attempt${attempt}-before.txt"
    run_runner 1 ""
    json_path="$(latest_json_for baseline_warm)"
    tok_s="$(json_tok_s "$json_path")"
    snapshot_host "${phase}-w${window}-attempt${attempt}-after" \
      "$OUT_DIR/guard-${RUN_TAG}-${phase}-w${window}-attempt${attempt}-after.txt"
    jq -n \
      --arg run_tag "$RUN_TAG" \
      --arg type "baseline_guard" \
      --arg phase "$phase" \
      --arg window "$window" \
      --arg attempt "$attempt" \
      --arg artifact "$json_path" \
      --argjson tok_s "$tok_s" \
      --argjson threshold "$BASELINE_MIN_TOK_S" \
      '{run_tag:$run_tag,type:$type,phase:$phase,window:($window|tonumber),attempt:($attempt|tonumber),artifact:$artifact,tok_s:$tok_s,threshold:$threshold,accepted:($tok_s >= $threshold)}' \
      | tee -a "$SUMMARY_JSON"
    awk -v tok="$tok_s" -v min="$BASELINE_MIN_TOK_S" 'BEGIN { exit !(tok >= min) }' && return 0
    echo "baseline guard $phase w$window attempt $attempt below threshold: $tok_s < $BASELINE_MIN_TOK_S" >&2
    sleep "$COOLDOWN_SECS"
  done
  return 1
}

run_window_condition() {
  local window="$1"
  local json_path
  local tok_s
  local mean_ms

  wait_for_quiet_host "condition-w${window}"
  snapshot_host "condition-w${window}-before" "$OUT_DIR/guard-${RUN_TAG}-condition-w${window}-before.txt"
  run_runner 0 "$window"
  json_path="$(latest_json_for "mtp_w${window}_warm")"
  tok_s="$(json_tok_s "$json_path")"
  mean_ms="$(json_mean_ms "$json_path")"
  snapshot_host "condition-w${window}-after" "$OUT_DIR/guard-${RUN_TAG}-condition-w${window}-after.txt"
  jq -n \
    --arg run_tag "$RUN_TAG" \
    --arg type "condition" \
    --arg window "$window" \
    --arg artifact "$json_path" \
    --argjson tok_s "$tok_s" \
    --argjson mean_ms "$mean_ms" \
    '{run_tag:$run_tag,type:$type,window:($window|tonumber),artifact:$artifact,tok_s:$tok_s,mean_ms:$mean_ms}' \
    | tee -a "$SUMMARY_JSON"
}

IFS=',' read -r -a window_list <<<"$WINDOWS"
for window in "${window_list[@]}"; do
  [[ -n "$window" ]] || continue
  echo "=== guarded vanilla MTP window $window ==="
  if ! run_baseline_guard pre "$window"; then
    echo "pre-baseline guard failed for window $window; stopping" >&2
    exit 2
  fi
  run_window_condition "$window"
  if ! run_baseline_guard post "$window"; then
    echo "post-baseline guard failed for window $window; stopping" >&2
    exit 3
  fi
  sleep "$COOLDOWN_SECS"
done

echo "$SUMMARY_JSON"
