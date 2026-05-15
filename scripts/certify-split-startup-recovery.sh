#!/usr/bin/env bash
# certify-split-startup-recovery.sh - certify split startup and worker-loss recovery.
#
# Usage:
#   scripts/certify-split-startup-recovery.sh <mesh-llm-binary> <model-ref-or-path>
#
# The harness starts one seed plus worker processes on localhost, waits for a
# multi-node split topology, kills an active downstream stage worker, and
# verifies the requested recovery outcome. It is intended for manual/release QA
# with real GGUFs or layer packages; it is not run by default in CI.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/certify-split-startup-recovery.sh <mesh-llm-binary> <model-ref-or-path>

Environment:
  MESH_SPLIT_CERT_WORKERS=2                 number of worker processes; recovery requires at least 2
  MESH_SPLIT_CERT_EXPECT=replacement        replacement|withdraw|local-fallback|any
  MESH_SPLIT_CERT_MAX_WAIT=420              seconds to wait for startup
  MESH_SPLIT_CERT_RECOVERY_MAX_WAIT=240     seconds to wait after killing a worker
  MESH_SPLIT_CERT_STABLE_PROBES=2           consecutive matching recovery observations required
  MESH_SPLIT_CERT_BASE_API_PORT=9460        first API port
  MESH_SPLIT_CERT_BASE_CONSOLE_PORT=3260    first management API port
  MESH_SPLIT_CERT_BASE_BIND_PORT=54600      first QUIC bind port
  MESH_SPLIT_CERT_DISCOVERY_MODE=           optional --mesh-discovery-mode value
  MESH_SPLIT_CERT_SEED_MAX_VRAM=10          --max-vram for the seed, in GB
  MESH_SPLIT_CERT_WORKER_MAX_VRAM=10        --max-vram for all workers, or comma list like 9,10
  MESH_SPLIT_CERT_CTX_SIZE=1024             optional --ctx-size
  MESH_SPLIT_CERT_DEVICE=Metal              optional --device override
  MESH_SPLIT_CERT_RUN_INFERENCE=0           set 1 to run a tiny chat completion before/after recovery
  MESH_SPLIT_CERT_KEEP_LOGS=0               set 1 to keep temp logs after success
  MESH_SPLIT_CERT_WORK_DIR=<dir>            directory for logs and result.jsonl
  MESH_SPLIT_CERT_PROCESS_ROOT=<dir>        short directory for per-process HOME/runtime roots

Examples:
  scripts/certify-split-startup-recovery.sh target/release/mesh-llm /models/qwen.gguf

  MESH_SPLIT_CERT_SEED_MAX_VRAM=10 \
  MESH_SPLIT_CERT_WORKER_MAX_VRAM=9,10 \
  scripts/certify-split-startup-recovery.sh target/release/mesh-llm Qwen3-32B-Q4_K_M
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MESH_LLM="${1:-}"
MODEL="${2:-}"
if [[ -z "$MESH_LLM" || -z "$MODEL" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -x "$MESH_LLM" ]]; then
    echo "FAIL prereq-binary: missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi

WORKERS="${MESH_SPLIT_CERT_WORKERS:-2}"
EXPECT="${MESH_SPLIT_CERT_EXPECT:-replacement}"
MAX_WAIT="${MESH_SPLIT_CERT_MAX_WAIT:-420}"
RECOVERY_MAX_WAIT="${MESH_SPLIT_CERT_RECOVERY_MAX_WAIT:-240}"
STABLE_PROBES="${MESH_SPLIT_CERT_STABLE_PROBES:-2}"
BASE_API_PORT="${MESH_SPLIT_CERT_BASE_API_PORT:-9460}"
BASE_CONSOLE_PORT="${MESH_SPLIT_CERT_BASE_CONSOLE_PORT:-3260}"
BASE_BIND_PORT="${MESH_SPLIT_CERT_BASE_BIND_PORT:-54600}"
DISCOVERY_MODE="${MESH_SPLIT_CERT_DISCOVERY_MODE:-}"
SEED_MAX_VRAM="${MESH_SPLIT_CERT_SEED_MAX_VRAM:-10}"
WORKER_MAX_VRAM="${MESH_SPLIT_CERT_WORKER_MAX_VRAM:-10}"
CTX_SIZE="${MESH_SPLIT_CERT_CTX_SIZE:-1024}"
DEVICE="${MESH_SPLIT_CERT_DEVICE:-}"
RUN_INFERENCE="${MESH_SPLIT_CERT_RUN_INFERENCE:-0}"
KEEP_LOGS="${MESH_SPLIT_CERT_KEEP_LOGS:-0}"

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 2 ]]; then
    echo "FAIL prereq-workers: recovery certification requires MESH_SPLIT_CERT_WORKERS >= 2" >&2
    exit 1
fi
case "$EXPECT" in
    replacement | withdraw | local-fallback | any) ;;
    *)
        echo "FAIL prereq-expect: MESH_SPLIT_CERT_EXPECT must be replacement, withdraw, local-fallback, or any" >&2
        exit 1
        ;;
esac
if ! [[ "$STABLE_PROBES" =~ ^[0-9]+$ ]] || [[ "$STABLE_PROBES" -lt 1 ]]; then
    echo "FAIL prereq-stable-probes: MESH_SPLIT_CERT_STABLE_PROBES must be >= 1" >&2
    exit 1
fi

if [[ -n "${MESH_SPLIT_CERT_WORK_DIR:-}" ]]; then
    WORK_DIR="$MESH_SPLIT_CERT_WORK_DIR"
    mkdir -p "$WORK_DIR"
else
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mesh-split-recovery.XXXXXX")"
fi
if [[ -n "${MESH_SPLIT_CERT_PROCESS_ROOT:-}" ]]; then
    PROCESS_ROOT="$MESH_SPLIT_CERT_PROCESS_ROOT"
    PROCESS_ROOT_OWNED=0
    mkdir -p "$PROCESS_ROOT"
else
    PROCESS_ROOT="$(mktemp -d "/tmp/mesh-split-proc.XXXXXX")"
    PROCESS_ROOT_OWNED=1
fi
RESULTS="${WORK_DIR}/result.jsonl"
: >"$RESULTS"

declare -a LABELS=()
declare -a PIDS=()
declare -a API_PORTS=()
declare -a CONSOLE_PORTS=()
declare -a BIND_PORTS=()
declare -a LOGS=()
declare -a HOMES=()
declare -a RUNTIME_ROOTS=()

FAILED=0
DRIVER_INDEX=0

log() {
    printf '%s\n' "$*"
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "FAIL prereq-command: missing required command: $command_name" >&2
        exit 1
    fi
}

record_result() {
    local status="$1"
    local name="$2"
    local detail="${3:-}"
    RESULT_STATUS="$status" RESULT_NAME="$name" RESULT_DETAIL="$detail" RESULT_FILE="$RESULTS" \
        python3 - <<'PY'
import json
import os
import time

record = {
    "ts": int(time.time()),
    "status": os.environ["RESULT_STATUS"],
    "name": os.environ["RESULT_NAME"],
}
detail = os.environ.get("RESULT_DETAIL", "")
if detail:
    record["detail"] = detail
with open(os.environ["RESULT_FILE"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
PY
    if [[ "$status" == "PASS" ]]; then
        log "PASS $name${detail:+: $detail}"
    else
        log "FAIL $name${detail:+: $detail}" >&2
        FAILED=1
    fi
}

require_command curl
require_command pgrep
require_command python3
require_command seq
MESH_LLM_VERSION="$("$MESH_LLM" --version 2>/dev/null || true)"
record_result "PASS" "prereq-binary" "path=${MESH_LLM} version=${MESH_LLM_VERSION:-unknown}"

status_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/status" 2>/dev/null || true
}

runtime_stages_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/runtime/stages" 2>/dev/null || true
}

v1_models_json() {
    local api_port="$1"
    curl -fsS --max-time 10 "http://127.0.0.1:${api_port}/v1/models" 2>/dev/null || true
}

query_status() {
    local json="$1"
    local action="$2"
    STATUS_JSON="$json" python3 - "$action" <<'PY'
import json
import os
import sys

action = sys.argv[1]
try:
    data = json.loads(os.environ.get("STATUS_JSON", "") or "{}")
except Exception:
    data = {}

if action == "token":
    print(data.get("token") or "")
elif action == "node_id":
    print(data.get("node_id") or "")
elif action == "peers":
    print(len(data.get("peers") or []))
elif action == "state":
    print(data.get("node_state") or "")
else:
    raise SystemExit(f"unknown status query: {action}")
PY
}

query_stages() {
    local json="$1"
    local action="$2"
    STAGES_JSON="$json" python3 - "$action" <<'PY'
import json
import os
import sys

action = sys.argv[1]
try:
    data = json.loads(os.environ.get("STAGES_JSON", "") or "{}")
except Exception:
    data = {}

topologies = data.get("topologies") or []
statuses = data.get("statuses") or data.get("stages") or []

def stage_nodes():
    nodes = []
    for topology in topologies:
        for stage in topology.get("stages") or []:
            node_id = stage.get("node_id")
            if node_id and node_id not in nodes:
                nodes.append(node_id)
    return nodes

if action == "topology_count":
    print(len(topologies))
elif action == "stage_count":
    print(sum(len(t.get("stages") or []) for t in topologies))
elif action == "unique_stage_nodes":
    print(len(stage_nodes()))
elif action == "stage_nodes":
    print("\n".join(stage_nodes()))
elif action == "first_run_id":
    print(topologies[0].get("run_id") if topologies else "")
elif action == "ready_status_count":
    print(sum(1 for s in statuses if str(s.get("state", "")).lower() == "ready"))
elif action == "summary":
    compact = []
    for topology in topologies:
        compact.append({
            "topology_id": topology.get("topology_id"),
            "run_id": topology.get("run_id"),
            "stages": [
                {
                    "stage_id": stage.get("stage_id"),
                    "stage_index": stage.get("stage_index"),
                    "node_id": stage.get("node_id"),
                    "layers": [stage.get("layer_start"), stage.get("layer_end")],
                }
                for stage in topology.get("stages") or []
            ],
        })
    print(json.dumps(compact, sort_keys=True))
else:
    raise SystemExit(f"unknown stages query: {action}")
PY
}

recovery_observation() {
    local stages_json="$1"
    local models_json="$2"
    local killed_short="$3"
    local old_run_id="$4"
    RECOVERY_STAGES_JSON="$stages_json" RECOVERY_MODELS_JSON="$models_json" KILLED_SHORT="$killed_short" OLD_RUN_ID="$old_run_id" \
        python3 - <<'PY'
import json
import os

try:
    stages = json.loads(os.environ.get("RECOVERY_STAGES_JSON", "") or "{}")
except Exception:
    stages = {}
try:
    models = json.loads(os.environ.get("RECOVERY_MODELS_JSON", "") or "{}")
except Exception:
    models = {}

killed = os.environ.get("KILLED_SHORT", "")
old_run_id = os.environ.get("OLD_RUN_ID", "")
topologies = stages.get("topologies") or []
model_count = len(models.get("data") or [])
nodes = []
stage_count = 0
run_ids = []
for topology in topologies:
    run_id = topology.get("run_id")
    if run_id and run_id not in run_ids:
        run_ids.append(run_id)
    for stage in topology.get("stages") or []:
        stage_count += 1
        node_id = stage.get("node_id") or ""
        if node_id and node_id not in nodes:
            nodes.append(node_id)

killed_present = any(node.startswith(killed) for node in nodes) if killed else False
unique_nodes = len(nodes)
has_new_run = not old_run_id or any(run_id != old_run_id for run_id in run_ids)
state = "pending"
if not killed_present and has_new_run and topologies and stage_count >= 2 and unique_nodes >= 2 and model_count >= 1:
    state = "replacement"
elif not killed_present and model_count >= 1 and unique_nodes <= 1:
    state = "local-fallback"
elif not killed_present and (model_count == 0 or stage_count == 0):
    state = "withdraw"

detail = (
    f"state={state} old_run_id={old_run_id or 'unknown'} run_ids={','.join(run_ids) or 'none'} "
    f"topologies={len(topologies)} stages={stage_count} "
    f"nodes={unique_nodes} models={model_count} killed_present={str(killed_present).lower()}"
)
print(f"{state}|{detail}")
PY
}

query_models() {
    local json="$1"
    MODELS_JSON="$json" python3 - <<'PY'
import json
import os

try:
    data = json.loads(os.environ.get("MODELS_JSON", "") or "{}")
except Exception:
    data = {}
print(len(data.get("data") or []))
PY
}

descendant_pids() {
    local pid="$1"
    local children
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    for child in $children; do
        descendant_pids "$child"
        printf '%s\n' "$child"
    done
}

kill_tree() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    local children
    children="$(descendant_pids "$pid" | sort -u || true)"
    kill "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill 2>/dev/null || true
    fi
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill -9 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

dump_logs() {
    log "--- split certification logs: ${WORK_DIR} ---"
    for idx in "${!LOGS[@]}"; do
        log "--- ${LABELS[$idx]} log tail (${LOGS[$idx]}) ---"
        tail -160 "${LOGS[$idx]}" 2>/dev/null || true
    done
    log "--- result.jsonl ---"
    cat "$RESULTS" 2>/dev/null || true
    log "--- end split certification logs ---"
}

cleanup() {
    local exit_code=$?
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        for pid in "${PIDS[@]}"; do
            kill_tree "$pid"
        done
    fi
    local alive=0
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        for pid in "${PIDS[@]}"; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done
    fi
    if [[ "$alive" -eq 0 ]]; then
        record_result "PASS" "cleanup" "processes=0 evidence=${WORK_DIR}"
    else
        record_result "FAIL" "cleanup" "processes_alive=${alive} evidence=${WORK_DIR}"
    fi
    if [[ "$exit_code" -ne 0 || "$FAILED" -ne 0 || "$KEEP_LOGS" == "1" ]]; then
        dump_logs
        log "process roots: ${PROCESS_ROOT}"
    elif [[ -z "${MESH_SPLIT_CERT_WORK_DIR:-}" ]]; then
        rm -rf "$WORK_DIR"
    fi
    if [[ "$PROCESS_ROOT_OWNED" == "1" && "$exit_code" -eq 0 && "$FAILED" -eq 0 && "$KEEP_LOGS" != "1" ]]; then
        rm -rf "$PROCESS_ROOT"
    fi
}
trap cleanup EXIT

worker_max_vram_for_index() {
    local index="$1"
    IFS=',' read -r -a values <<<"$WORKER_MAX_VRAM"
    if [[ "${#values[@]}" -eq 0 ]]; then
        printf '%s\n' "10"
    elif [[ -n "${values[$index]:-}" ]]; then
        printf '%s\n' "${values[$index]}"
    else
        local last_index
        last_index=$((${#values[@]} - 1))
        printf '%s\n' "${values[$last_index]}"
    fi
}

start_node() {
    local label="$1"
    local join_token="$2"
    local max_vram="$3"
    local api_port="$4"
    local console_port="$5"
    local bind_port="$6"

    local home="${PROCESS_ROOT}/${label}/h"
    local runtime_root="${PROCESS_ROOT}/${label}/r"
    local log_file="${WORK_DIR}/${label}.log"
    mkdir -p "$home" "$runtime_root"

    local -a args=()
    if [[ -n "$DISCOVERY_MODE" ]]; then
        args+=(--mesh-discovery-mode "$DISCOVERY_MODE")
    fi
    args+=(
        serve
        --model "$MODEL"
        --split
        --no-draft
        --ctx-size "$CTX_SIZE"
        --max-vram "$max_vram"
        --port "$api_port"
        --console "$console_port"
        --bind-port "$bind_port"
        --headless
    )
    if [[ -n "$join_token" ]]; then
        args+=(--join "$join_token")
    fi
    if [[ -n "$DEVICE" ]]; then
        args+=(--device "$DEVICE")
    fi

    log "START $label api=$api_port console=$console_port bind=$bind_port max_vram=${max_vram}GB"
    HOME="$home" \
        MESH_LLM_RUNTIME_ROOT="$runtime_root" \
        MESH_LLM_EPHEMERAL_KEY=1 \
        "$MESH_LLM" "${args[@]}" >"$log_file" 2>&1 &
    local pid=$!

    LABELS+=("$label")
    PIDS+=("$pid")
    API_PORTS+=("$api_port")
    CONSOLE_PORTS+=("$console_port")
    BIND_PORTS+=("$bind_port")
    LOGS+=("$log_file")
    HOMES+=("$home")
    RUNTIME_ROOTS+=("$runtime_root")
}

require_process_alive() {
    local label="$1"
    local pid="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        record_result "FAIL" "${label}-alive" "process exited unexpectedly"
        exit 1
    fi
}

wait_for_token() {
    local label="$1"
    local pid="$2"
    local console_port="$3"
    local token=""
    for i in $(seq 1 "$MAX_WAIT"); do
        require_process_alive "$label" "$pid"
        local status
        status="$(status_json "$console_port")"
        token="$(query_status "$status" token)"
        if [[ -n "$token" ]]; then
            record_result "PASS" "${label}-invite-token" "ready after ${i}s"
            printf '%s\n' "$token"
            return 0
        fi
        sleep 1
    done
    record_result "FAIL" "${label}-invite-token" "timed out after ${MAX_WAIT}s"
    exit 1
}

wait_for_peer_count() {
    local label="$1"
    local pid="$2"
    local console_port="$3"
    local expected="$4"
    for i in $(seq 1 "$MAX_WAIT"); do
        require_process_alive "$label" "$pid"
        local status peers
        status="$(status_json "$console_port")"
        peers="$(query_status "$status" peers)"
        if [[ "$peers" -ge "$expected" ]]; then
            record_result "PASS" "${label}-peer-count" "peers=${peers} after ${i}s"
            return 0
        fi
        sleep 1
    done
    record_result "FAIL" "${label}-peer-count" "timed out waiting for peers >= ${expected}"
    exit 1
}

wait_for_any_split_topology() {
    local label="$1"
    local max_wait="$2"
    for i in $(seq 1 "$max_wait"); do
        for node_index in "${!PIDS[@]}"; do
            local pid="${PIDS[$node_index]}"
            [[ -n "$pid" ]] || continue
            require_process_alive "${LABELS[$node_index]}" "$pid"
            local stages_json models_json topology_count stage_count unique_nodes model_count summary
            stages_json="$(runtime_stages_json "${CONSOLE_PORTS[$node_index]}")"
            topology_count="$(query_stages "$stages_json" topology_count)"
            stage_count="$(query_stages "$stages_json" stage_count)"
            unique_nodes="$(query_stages "$stages_json" unique_stage_nodes)"
            models_json="$(v1_models_json "${API_PORTS[$node_index]}")"
            model_count="$(query_models "$models_json")"
            if [[ "$topology_count" -ge 1 && "$stage_count" -ge 2 && "$unique_nodes" -ge 2 && "$model_count" -ge 1 ]]; then
                DRIVER_INDEX="$node_index"
                summary="$(query_stages "$stages_json" summary)"
                record_result "PASS" "${label}-split-topology-ready" "observer=${LABELS[$node_index]} topologies=${topology_count} stages=${stage_count} nodes=${unique_nodes} models=${model_count}"
                printf '%s\n' "$summary" >"${WORK_DIR}/${label}-topology.json"
                return 0
            fi
        done
        sleep 1
    done
    record_result "FAIL" "${label}-split-topology-ready" "timed out after ${max_wait}s"
    exit 1
}

run_chat_probe() {
    local label="$1"
    local api_port="$2"
    local payload="${WORK_DIR}/${label}-chat.json"
    local output="${WORK_DIR}/${label}-chat-response.json"
    local models_json model_id
    models_json="$(v1_models_json "$api_port")"
    model_id="$(
        MODELS_JSON="$models_json" python3 - <<'PY'
import json
import os

data = json.loads(os.environ.get("MODELS_JSON", "") or "{}").get("data") or []
print(data[0].get("id", "") if data else "")
PY
    )"
    if [[ -z "$model_id" ]]; then
        record_result "FAIL" "${label}-chat-probe" "no model returned by /v1/models"
        exit 1
    fi
    MODEL_ID="$model_id" PAYLOAD="$payload" python3 - <<'PY'
import json
import os

body = {
    "model": os.environ["MODEL_ID"],
    "messages": [{"role": "user", "content": "Reply with ok."}],
    "stream": False,
    "max_tokens": 8,
    "temperature": 0,
}
with open(os.environ["PAYLOAD"], "w", encoding="utf-8") as fh:
    json.dump(body, fh)
PY
    curl -fsS --max-time 180 \
        "http://127.0.0.1:${api_port}/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d @"$payload" \
        -o "$output"
    python3 - "$output" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    body = json.load(fh)
if body.get("object") != "chat.completion":
    raise SystemExit(f"unexpected object: {body.get('object')!r}")
if not body.get("choices"):
    raise SystemExit("missing choices")
PY
    record_result "PASS" "${label}-chat-probe" "model=${model_id}"
}

find_active_downstream_worker_stage() {
    local stages_json="$1"
    local excluded_worker_index="$2"
    shift
    shift
    local -a worker_shorts=("$@")
    STAGES_JSON="$stages_json" WORKER_SHORTS="$(IFS=,; echo "${worker_shorts[*]}")" EXCLUDED_WORKER_INDEX="$excluded_worker_index" python3 - <<'PY'
import json
import os

data = json.loads(os.environ.get("STAGES_JSON", "") or "{}")
worker_shorts = [item for item in os.environ.get("WORKER_SHORTS", "").split(",") if item]
try:
    excluded_worker_index = int(os.environ.get("EXCLUDED_WORKER_INDEX", "-1"))
except ValueError:
    excluded_worker_index = -1
fallback = None
for topology in data.get("topologies") or []:
    for stage in topology.get("stages") or []:
        if int(stage.get("stage_index") or 0) == 0:
            continue
        node_id = stage.get("node_id") or ""
        for idx, short in enumerate(worker_shorts):
            if node_id.startswith(short):
                if idx != excluded_worker_index:
                    print(idx)
                    raise SystemExit(0)
                if fallback is None:
                    fallback = idx
if fallback is not None:
    print(fallback)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

wait_for_expected_recovery() {
    local label="$1"
    local killed_short="$2"
    local expected="$3"
    local old_run_id="$4"
    local stable_count=0
    local last_detail=""
    for i in $(seq 1 "$RECOVERY_MAX_WAIT"); do
        local matched_state=""
        local matched_detail=""
        local matched_stages_json=""
        for node_index in "${!PIDS[@]}"; do
            local observer_pid="${PIDS[$node_index]}"
            [[ -n "$observer_pid" ]] || continue
            require_process_alive "${LABELS[$node_index]}" "$observer_pid"
            local stages_json models_json observation state detail
            stages_json="$(runtime_stages_json "${CONSOLE_PORTS[$node_index]}")"
            models_json="$(v1_models_json "${API_PORTS[$node_index]}")"
            observation="$(recovery_observation "$stages_json" "$models_json" "$killed_short" "$old_run_id")"
            state="${observation%%|*}"
            detail="${observation#*|}"
            last_detail="observer=${LABELS[$node_index]} ${detail}"
            if [[ "$state" == "$expected" || ( "$expected" == "any" && "$state" != "pending" ) ]]; then
                matched_state="$state"
                matched_detail="observer=${LABELS[$node_index]} killed_node=${killed_short} recovered_after=${i}s ${detail}"
                matched_stages_json="$stages_json"
                break
            fi
        done
        if [[ -n "$matched_state" ]]; then
            stable_count=$((stable_count + 1))
            if [[ "$stable_count" -ge "$STABLE_PROBES" ]]; then
                record_result "PASS" "${label}-recovery-${matched_state}" "$matched_detail"
                printf '%s\n' "$(query_stages "$matched_stages_json" summary)" >"${WORK_DIR}/${label}-recovered-topology.json"
                return 0
            fi
        else
            stable_count=0
        fi
        sleep 1
    done
    record_result "FAIL" "${label}-recovery-${expected}" "timed out after ${RECOVERY_MAX_WAIT}s killed_node=${killed_short} last=${last_detail}"
    exit 1
}

log "=== Split Worker Startup + Recovery Certification ==="
log "mesh-llm: $MESH_LLM"
log "model:    $MODEL"
log "workdir:  $WORK_DIR"
log "procs:    $PROCESS_ROOT"
log "workers:  $WORKERS"
log "expect:   $EXPECT"
if [[ -n "$DISCOVERY_MODE" ]]; then
    log "discovery: $DISCOVERY_MODE"
fi

start_node "seed" "" "$SEED_MAX_VRAM" "$BASE_API_PORT" "$BASE_CONSOLE_PORT" "$BASE_BIND_PORT"
TOKEN="$(wait_for_token "seed" "${PIDS[0]}" "$BASE_CONSOLE_PORT" | tail -n 1)"

for worker_index in $(seq 0 $((WORKERS - 1))); do
    api_port=$((BASE_API_PORT + worker_index + 1))
    console_port=$((BASE_CONSOLE_PORT + worker_index + 1))
    bind_port=$((BASE_BIND_PORT + worker_index + 1))
    start_node "worker-$((worker_index + 1))" "$TOKEN" "$(worker_max_vram_for_index "$worker_index")" "$api_port" "$console_port" "$bind_port"
done

wait_for_peer_count "seed" "${PIDS[0]}" "$BASE_CONSOLE_PORT" "$WORKERS"
wait_for_any_split_topology "startup" "$MAX_WAIT"

if [[ "$RUN_INFERENCE" == "1" ]]; then
    run_chat_probe "startup" "${API_PORTS[$DRIVER_INDEX]}"
fi

worker_shorts=()
for worker_index in $(seq 1 "$WORKERS"); do
    status="$(status_json "${CONSOLE_PORTS[$worker_index]}")"
    short_id="$(query_status "$status" node_id)"
    if [[ -z "$short_id" ]]; then
        record_result "FAIL" "worker-${worker_index}-node-id" "missing node_id in /api/status"
        exit 1
    fi
    worker_shorts+=("$short_id")
done

stages_json="$(runtime_stages_json "${CONSOLE_PORTS[$DRIVER_INDEX]}")"
old_run_id="$(query_stages "$stages_json" first_run_id)"
excluded_worker_index=-1
if [[ "$DRIVER_INDEX" -gt 0 ]]; then
    excluded_worker_index=$((DRIVER_INDEX - 1))
fi
kill_worker_zero_index="$(find_active_downstream_worker_stage "$stages_json" "$excluded_worker_index" "${worker_shorts[@]}")" || {
    record_result "FAIL" "active-downstream-worker-stage" "no downstream worker-owned stage found in topology"
    exit 1
}
kill_process_index=$((kill_worker_zero_index + 1))
killed_short="${worker_shorts[$kill_worker_zero_index]}"
record_result "PASS" "active-downstream-worker-stage" "observer=${LABELS[$DRIVER_INDEX]} worker_index=${kill_worker_zero_index} node=${killed_short} old_run_id=${old_run_id:-unknown}"

log "KILL worker-$((kill_worker_zero_index + 1)) pid=${PIDS[$kill_process_index]} node=${killed_short}"
kill_tree "${PIDS[$kill_process_index]}"
PIDS[$kill_process_index]=""

wait_for_expected_recovery "split" "$killed_short" "$EXPECT" "$old_run_id"

if [[ "$RUN_INFERENCE" == "1" ]]; then
    run_chat_probe "recovery" "${API_PORTS[$DRIVER_INDEX]}"
fi

record_result "PASS" "split-startup-recovery-certification" "results=${RESULTS}"
log "Split worker startup + recovery certification passed"
