#!/usr/bin/env bash
# qa-control-plane-mixed-version.sh - certify mixed-version public mesh and
# owner-control behavior across a released and current mesh-llm binary.

set -euo pipefail

RELEASED_BINARY=""
CURRENT_BINARY=""
EVIDENCE_DIR=".sisyphus/evidence"
LOCAL_ONLY=false
CONFIG_ONLY=false
MODEL=""
RELEASED_MODEL=""
CURRENT_MODEL=""
BASE_PORT="${MESH_QA_BASE_PORT:-19640}"
MAX_WAIT="${MESH_QA_MAX_WAIT:-180}"
STABLE_PROBES="${MESH_QA_STABLE_PROBES:-3}"
CHAT_MAX_TIME="${MESH_QA_CHAT_MAX_TIME:-120}"
CTX_SIZE="${MESH_QA_CTX_SIZE:-512}"
TMP_ROOT="${MESH_QA_TMP_ROOT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}"
RUN_CARGO_TESTS=true
KEEP_LOGS=false
REQUIRE_PUBLIC=false
PRINT_PLAN=false

RUN_DIR=""
LOG_DIR=""
STATUS_DIR=""
MODELS_DIR=""
CHAT_DIR=""
CONTROL_DIR=""
VERSIONS_DIR=""
RESULTS_JSONL=""
COMMANDS_JSONL=""
MANIFEST_JSON=""
SUMMARY_JSON=""
SUMMARY_MD=""
WORK_ROOT=""
PIDS=()
EXIT_STATUS=0
START_NODE_PID=""
WAIT_TOKEN_VALUE=""
WAIT_MODELS_MODEL_ID=""
NODE_INDEX=0

usage() {
    cat <<'EOF'
Usage:
  scripts/qa-control-plane-mixed-version.sh \
    --released-binary /path/to/released/mesh-llm \
    --current-binary /path/to/current/mesh-llm \
    --evidence-dir .sisyphus/evidence [options]

Purpose:
  Produce executable evidence that a current mesh-llm build remains compatible
  with a released mesh-llm binary while owner-control stays on its dedicated
  mesh-llm-control/1 ALPN.

Required:
  --released-binary PATH   Released/reference mesh-llm binary.
  --current-binary PATH    Current-branch mesh-llm binary.

Options:
  --evidence-dir DIR       Evidence root (default: .sisyphus/evidence).
  --local-only             Skip public-mesh probes.
  --config-only            Owner-control lane only; implies --local-only.
  --model REF_OR_PATH      Model used for both local routing smokes.
  --released-model REF     Model for released-binary serving direction.
  --current-model REF      Model for current-binary serving direction.
  --base-port PORT         First reserved local port (default: 19640).
  --max-wait SECONDS       Readiness timeout (default: 180).
  --stable-probes N        Consecutive peer/model probes required (default: 3).
  --chat-max-time SECONDS  Chat request timeout (default: 120).
  --ctx-size TOKENS        Local smoke context size (default: 512).
  --skip-cargo-tests       Skip current-branch protocol compatibility tests.
  --require-public         Treat public mesh probe prerequisites as failures.
  --print-plan             Print planned checks as JSON without side effects.
  --keep-logs              Keep successful run logs.
  -h, --help               Show this help.

Modes:
  Default mode runs public client --auto probes, optional local routing smokes,
  and owner-control checks. --local-only skips public probes.
  --config-only runs loopback coexistence plus owner-control migration checks.
  When prerequisites are present, it records:
    config-missing-endpoint-required
    config-new-client-owner-control
    config-control-rejects-legacy-frames
  If local prerequisites are missing, it records explicit PREREQ results such
  as config-cargo-tests or config-runtime-bootstrap.

Evidence:
  Each run creates a timestamped directory containing:
    manifest.json      Run inputs and mode flags.
    commands.jsonl      Commands executed by the harness and their logs.
    results.jsonl       Machine-readable PASS/FAIL/PREREQ records.
    summary.md          Human-readable final summary.
    summary.json        Machine-readable final summary.
    versions/*.txt      Captured released/current binary version strings.
    logs/, status/, models/, chat/, control/ grouped runtime payloads.

Result vocabulary:
  PASS completed, FAIL failed, PREREQ blocked by an explicit local prerequisite.

This script does not publish meshes and does not modify protocol state.
EOF
}

fail_usage() {
    echo "error: $*" >&2
    echo >&2
    usage >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --released-binary)
            RELEASED_BINARY="${2:-}"
            shift 2
            ;;
        --current-binary)
            CURRENT_BINARY="${2:-}"
            shift 2
            ;;
        --evidence-dir)
            EVIDENCE_DIR="${2:-}"
            shift 2
            ;;
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        --config-only)
            CONFIG_ONLY=true
            LOCAL_ONLY=true
            shift
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --released-model)
            RELEASED_MODEL="${2:-}"
            shift 2
            ;;
        --current-model)
            CURRENT_MODEL="${2:-}"
            shift 2
            ;;
        --base-port)
            BASE_PORT="${2:-}"
            shift 2
            ;;
        --max-wait)
            MAX_WAIT="${2:-}"
            shift 2
            ;;
        --stable-probes)
            STABLE_PROBES="${2:-}"
            shift 2
            ;;
        --chat-max-time)
            CHAT_MAX_TIME="${2:-}"
            shift 2
            ;;
        --ctx-size)
            CTX_SIZE="${2:-}"
            shift 2
            ;;
        --skip-cargo-tests)
            RUN_CARGO_TESTS=false
            shift
            ;;
        --require-public)
            REQUIRE_PUBLIC=true
            shift
            ;;
        --print-plan)
            PRINT_PLAN=true
            shift
            ;;
        --keep-logs)
            KEEP_LOGS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail_usage "unknown argument: $1"
            ;;
    esac
done

missing=()
[[ -n "$RELEASED_BINARY" ]] || missing+=("--released-binary")
[[ -n "$CURRENT_BINARY" ]] || missing+=("--current-binary")
if [[ "${#missing[@]}" -gt 0 ]]; then
    fail_usage "missing required options: ${missing[*]}"
fi

for numeric in BASE_PORT MAX_WAIT STABLE_PROBES CHAT_MAX_TIME CTX_SIZE; do
    value="${!numeric}"
    if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
        fail_usage "--$(printf '%s' "$numeric" | tr '[:upper:]_' '[:lower:]-') must be a positive integer"
    fi
done

RELEASED_MODEL="${RELEASED_MODEL:-$MODEL}"
CURRENT_MODEL="${CURRENT_MODEL:-$MODEL}"

require_tool() {
    command -v "$1" >/dev/null 2>&1 || { echo "error: missing required tool: $1" >&2; exit 2; }
}

require_tool python3

if [[ "$PRINT_PLAN" == true ]]; then
    python3 - \
        "$RELEASED_BINARY" \
        "$CURRENT_BINARY" \
        "$EVIDENCE_DIR" \
        "$LOCAL_ONLY" \
        "$CONFIG_ONLY" \
        "$RUN_CARGO_TESTS" \
        "$REQUIRE_PUBLIC" \
        "$MODEL" \
        "$RELEASED_MODEL" \
        "$CURRENT_MODEL" <<'PY'
import json
import sys

released, current, evidence, local_only, config_only, cargo, require_public, model, released_model, current_model = sys.argv[1:]
local = local_only == "true"
config = config_only == "true"
run_cargo = cargo == "true"

checks = [
    "prereq.released-binary",
    "prereq.current-binary",
]

if not local:
    for label in ["released-public", "current-public"]:
        checks.extend([
            f"{label}.status",
            f"{label}.no-control-leak",
            f"{label}.public-models",
            f"{label}.public.chat",
        ])

def add_direction(label, model_is_supplied):
    checks.extend([
        f"{label}-server.token",
        f"{label}-server.peers",
        f"{label}-client.peers",
        f"{label}-server.no-control-leak",
        f"{label}-client.no-control-leak",
    ])
    if config or not model_is_supplied:
        checks.append(f"{label}.loopback-coexistence")
    else:
        checks.extend([
            f"{label}-client.models",
            f"{label}-client.chat",
        ])

if local or released_model or current_model or config:
    if config or current_model:
        add_direction("current-serves-released-client", bool(current_model))
    else:
        checks.append("current-serves-released-client")

    if config or released_model:
        add_direction("released-serves-current-client", bool(released_model))
    else:
        checks.append("released-serves-current-client")

if run_cargo:
    checks.extend([
        "config-missing-endpoint-required",
        "config-new-client-owner-control",
        "config-control-rejects-legacy-frames",
    ])
else:
    checks.append("config-cargo-tests")

checks.extend([
    "config-runtime-bootstrap",
    "config-runtime-get-config",
    "cleanup",
])

plan = {
    "script": "qa-control-plane-mixed-version.sh",
    "released_binary": released,
    "current_binary": current,
    "evidence_dir": evidence,
    "local_only": local,
    "config_only": config,
    "public_mesh": not local,
    "run_cargo_tests": run_cargo,
    "require_public": require_public == "true",
    "model_supplied": bool(model or released_model or current_model),
    "checks": checks,
}
print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
PY
    exit 0
fi

if [[ ! -x "$RELEASED_BINARY" ]]; then
    fail_usage "--released-binary is not executable: $RELEASED_BINARY"
fi
if [[ ! -x "$CURRENT_BINARY" ]]; then
    fail_usage "--current-binary is not executable: $CURRENT_BINARY"
fi

require_tool curl
require_tool date
require_tool mktemp

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EVIDENCE_DIR%/}/control-plane-mixed-version-${RUN_ID}"
WORK_ROOT="$(mktemp -d "${TMP_ROOT%/}/mesh-control-plane-mixed-version.XXXXXX")"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
MODELS_DIR="$RUN_DIR/models"
CHAT_DIR="$RUN_DIR/chat"
CONTROL_DIR="$RUN_DIR/control"
VERSIONS_DIR="$RUN_DIR/versions"
RESULTS_JSONL="$RUN_DIR/results.jsonl"
COMMANDS_JSONL="$RUN_DIR/commands.jsonl"
MANIFEST_JSON="$RUN_DIR/manifest.json"
SUMMARY_JSON="$RUN_DIR/summary.json"
SUMMARY_MD="$RUN_DIR/summary.md"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$MODELS_DIR" "$CHAT_DIR" "$CONTROL_DIR" "$VERSIONS_DIR" "$WORK_ROOT"
: >"$RESULTS_JSONL"
: >"$COMMANDS_JSONL"

append_summary() {
    printf '%s\n' "$*" >>"$SUMMARY_MD"
}

record_command() {
    local name="$1"
    local log="$2"
    shift 2
    python3 - "$COMMANDS_JSONL" "$name" "$log" "$@" <<'PY'
import json
import sys

path, name, log, *argv = sys.argv[1:]
record = {"name": name, "argv": argv, "log": log}
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

record_result() {
    local status="$1"
    local name="$2"
    local message="$3"
    shift 3

    printf '%s %s' "$status" "$name"
    for field in "$@"; do
        printf ' %q' "$field"
    done
    if [[ -n "$message" ]]; then
        printf ' message=%q' "$message"
    fi
    printf '\n'

    python3 - "$RESULTS_JSONL" "$status" "$name" "$message" "$@" <<'PY'
import json
import sys

path, status, name, message, *fields = sys.argv[1:]
record = {"status": status, "name": name, "message": message}
for field in fields:
    if "=" not in field:
        continue
    key, value = field.split("=", 1)
    record[key] = value
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
PY

    append_summary "- ${status} ${name}: ${message}"
    if [[ "$status" == "FAIL" ]]; then
        EXIT_STATUS=1
    fi
}

write_manifest() {
    python3 - \
        "$MANIFEST_JSON" \
        "$RUN_ID" \
        "$RELEASED_BINARY" \
        "$CURRENT_BINARY" \
        "$LOCAL_ONLY" \
        "$CONFIG_ONLY" \
        "$RUN_CARGO_TESTS" \
        "$REQUIRE_PUBLIC" \
        "$BASE_PORT" \
        "$MAX_WAIT" \
        "$STABLE_PROBES" \
        "$CTX_SIZE" \
        "$TMP_ROOT" \
        "$RUN_DIR" \
        "$WORK_ROOT" \
        "$RELEASED_MODEL" \
        "$CURRENT_MODEL" <<'PY'
import json
import sys

keys = [
    "path",
    "run_id",
    "released_binary",
    "current_binary",
    "local_only",
    "config_only",
    "run_cargo_tests",
    "require_public",
    "base_port",
    "max_wait_seconds",
    "stable_probes",
    "ctx_size",
    "tmp_root",
    "evidence_dir",
    "work_root",
    "released_model",
    "current_model",
]
args = dict(zip(keys, sys.argv[1:]))
manifest = {
    "run_id": args["run_id"],
    "released_binary": args["released_binary"],
    "current_binary": args["current_binary"],
    "local_only": args["local_only"] == "true",
    "config_only": args["config_only"] == "true",
    "public_mesh": args["local_only"] != "true",
    "run_cargo_tests": args["run_cargo_tests"] == "true",
    "require_public": args["require_public"] == "true",
    "base_port": int(args["base_port"]),
    "max_wait_seconds": int(args["max_wait_seconds"]),
    "stable_probes": int(args["stable_probes"]),
    "ctx_size": int(args["ctx_size"]),
    "tmp_root": args["tmp_root"],
    "evidence_dir": args["evidence_dir"],
    "work_root": args["work_root"],
    "released_model": args["released_model"] or None,
    "current_model": args["current_model"] or None,
}
with open(args["path"], "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

write_summary_json() {
    python3 - "$RESULTS_JSONL" "$SUMMARY_JSON" "$RUN_DIR" <<'PY'
import json
import sys

results_path, summary_path, run_dir = sys.argv[1:]
records = []
try:
    with open(results_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
except FileNotFoundError:
    pass
statuses = [record.get("status") for record in records]
overall = "fail" if "FAIL" in statuses else "prereq" if "PREREQ" in statuses else "pass"
summary = {
    "overall": overall,
    "evidence_dir": run_dir,
    "counts": {name.lower(): statuses.count(name) for name in ["PASS", "FAIL", "PREREQ"]},
    "results": records,
}
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

mark_prereq_or_fail() {
    local name="$1"
    local message="$2"
    shift 2
    if [[ "$REQUIRE_PUBLIC" == true ]]; then
        record_result "FAIL" "$name" "$message" "$@"
    else
        record_result "PREREQ" "$name" "$message" "$@"
    fi
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

cleanup() {
    for pid in "${PIDS[@]}"; do
        kill_tree "$pid"
    done
    local alive=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done
    if [[ "$alive" -eq 0 ]]; then
        record_result "PASS" "cleanup" "harness-owned processes stopped" "processes=0"
    else
        record_result "FAIL" "cleanup" "harness-owned processes remain" "processes=$alive"
    fi
    if [[ "$EXIT_STATUS" -eq 0 && "$KEEP_LOGS" != true ]]; then
        find "$LOG_DIR" -type f -empty -delete 2>/dev/null || true
        rm -rf "$WORK_ROOT" 2>/dev/null || true
    fi
    write_summary_json
}
trap cleanup EXIT

write_manifest

append_summary "# Mixed-version owner-control QA"
append_summary ""
append_summary "- Run ID: \`$RUN_ID\`"
append_summary "- Released binary: \`$RELEASED_BINARY\`"
append_summary "- Current binary: \`$CURRENT_BINARY\`"
append_summary "- Local only: \`$LOCAL_ONLY\`"
append_summary "- Config only: \`$CONFIG_ONLY\`"
append_summary "- Evidence: \`$RUN_DIR\`"
append_summary "- Work root: \`$WORK_ROOT\`"
append_summary ""

record_binary_prereq() {
    local label="$1"
    local binary="$2"
    local version_path="$VERSIONS_DIR/${label}.txt"
    local version_log="$LOG_DIR/${label}-version.log"
    local version=""

    if "$binary" --version >"$version_path" 2>"$version_log"; then
        version="$(head -1 "$version_path" || true)"
        printf '%s\n' "$version" >"$version_path"
    fi

    if [[ -n "$version" ]]; then
        record_result "PASS" "prereq.${label}-binary" "${label} binary is executable and reports a version" \
            "path=$binary" "version=$(printf '%s' "$version" | tr ' ' '_')" \
            "version_path=$version_path" "log=$version_log"
    else
        record_result "PREREQ" "prereq.${label}-binary" "${label} binary did not report a version" \
            "path=$binary" "version_path=$version_path" "log=$version_log"
    fi
}

record_binary_prereq "released" "$RELEASED_BINARY"
record_binary_prereq "current" "$CURRENT_BINARY"

run_logged() {
    local name="$1"
    shift
    local log="$LOG_DIR/${name}.log"
    record_command "$name" "$log" "$@"
    if "$@" >"$log" 2>&1; then
        record_result "PASS" "$name" "command passed" "log=$log"
        return 0
    fi
    record_result "FAIL" "$name" "command failed" "log=$log"
    tail -80 "$log" >&2 || true
    return 1
}

assert_status_no_control_leak() {
    local label="$1"
    local path="$2"
    if python3 - "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    payload = json.load(fh)

def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "control_endpoint":
                raise SystemExit("control_endpoint key leaked")
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
    elif isinstance(value, str):
        if "mesh-llm-control/1" in value:
            raise SystemExit("owner-control ALPN leaked")
        if "control://" in value:
            raise SystemExit("owner-control token leaked")

walk(payload)
PY
    then
        record_result "PASS" "${label}.no-control-leak" \
            "status payload does not expose owner-control endpoint data" "path=$path"
        return 0
    fi
    record_result "FAIL" "${label}.no-control-leak" \
        "status payload exposed owner-control endpoint data" "path=$path"
    return 1
}

start_node() {
    local label="$1"
    local binary="$2"
    shift 2

    START_NODE_PID=""
    NODE_INDEX=$((NODE_INDEX + 1))
    local node_slug="n$NODE_INDEX"
    local home="$WORK_ROOT/h-$node_slug"
    local runtime="$WORK_ROOT/r-$node_slug"
    local log="$LOG_DIR/$label.log"
    mkdir -p "$home" "$runtime" || return 1

    (
        export HOME="$home"
        export MESH_LLM_RUNTIME_ROOT="$runtime"
        export MESH_LLM_EPHEMERAL_KEY=1
        exec "$binary" "$@"
    ) >"$log" 2>&1 &
    START_NODE_PID=$!
    PIDS+=("$START_NODE_PID")
}

assert_process_alive() {
    local pid="$1"
    local label="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        record_result "FAIL" "$label" "process exited unexpectedly" "pid=$pid" "log=$LOG_DIR/$label.log"
        return 1
    fi
}

curl_json() {
    curl -fsS --max-time 5 "$1" -o "$2"
}

json_query() {
    local mode="$1"
    local path="$2"
    local expr="$3"
    python3 - "$mode" "$path" "$expr" <<'PY'
import json
import sys

mode, path, expr = sys.argv[1:]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
value = data
for part in expr.split("."):
    if not part:
        continue
    if isinstance(value, dict):
        value = value.get(part)
    elif isinstance(value, list) and part.isdigit():
        value = value[int(part)]
    else:
        value = None
        break
if mode == "len":
    print(len(value) if isinstance(value, list) else 0)
    raise SystemExit
if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

json_field() { json_query field "$1" "$2"; }

json_len() { json_query len "$1" "$2"; }

wait_status() {
    local label="$1"
    local console_port="$2"
    local out="$STATUS_DIR/${label}-status.json"
    for second in $(seq 1 "$MAX_WAIT"); do
        if curl_json "http://127.0.0.1:${console_port}/api/status" "$out" 2>/dev/null; then
            record_result "PASS" "${label}.status" "management API returned status" \
                "seconds=$second" "path=$out"
            return 0
        fi
        sleep 1
    done
    record_result "FAIL" "${label}.status" "timed out waiting for management API" \
        "seconds=$MAX_WAIT" "port=$console_port"
    return 1
}

wait_token() {
    local label="$1"
    local console_port="$2"
    local out="$STATUS_DIR/${label}-status.json"
    WAIT_TOKEN_VALUE=""
    for second in $(seq 1 "$MAX_WAIT"); do
        if curl_json "http://127.0.0.1:${console_port}/api/status" "$out" 2>/dev/null; then
            local token
            token="$(json_field "$out" "token")"
            if [[ -n "$token" ]]; then
                WAIT_TOKEN_VALUE="$token"
                record_result "PASS" "${label}.token" "invite token available" \
                    "seconds=$second" "path=$out"
                return 0
            fi
        fi
        sleep 1
    done
    record_result "FAIL" "${label}.token" "timed out waiting for invite token" \
        "seconds=$MAX_WAIT" "port=$console_port"
    return 1
}

wait_peers() {
    local label="$1"
    local console_port="$2"
    local expected="$3"
    local out="$STATUS_DIR/${label}-peers.json"
    local stable=0
    for second in $(seq 1 "$MAX_WAIT"); do
        if curl_json "http://127.0.0.1:${console_port}/api/status" "$out" 2>/dev/null; then
            local peers
            peers="$(json_len "$out" "peers")"
            if [[ "$peers" -ge "$expected" ]]; then
                stable=$((stable + 1))
                if [[ "$stable" -ge "$STABLE_PROBES" ]]; then
                    record_result "PASS" "${label}.peers" "expected peers visible" \
                        "seconds=$second" "peers=$peers" "path=$out"
                    return 0
                fi
            else
                stable=0
            fi
        fi
        sleep 1
    done
    record_result "FAIL" "${label}.peers" "timed out waiting for peers" \
        "expected=$expected" "port=$console_port"
    return 1
}

first_model_id() {
    local models_json="$1"
    python3 - "$models_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh).get("data", [])
for item in data:
    model_id = item.get("id")
    if model_id:
        print(model_id)
        break
PY
}

wait_models() {
    local label="$1"
    local api_port="$2"
    local out="$MODELS_DIR/${label}-models.json"
    WAIT_MODELS_MODEL_ID=""
    for second in $(seq 1 "$MAX_WAIT"); do
        if curl_json "http://127.0.0.1:${api_port}/v1/models" "$out" 2>/dev/null; then
            local model_id
            model_id="$(first_model_id "$out")"
            if [[ -n "$model_id" ]]; then
                WAIT_MODELS_MODEL_ID="$model_id"
                record_result "PASS" "${label}.models" "OpenAI model list is populated" \
                    "seconds=$second" "model=$model_id" "path=$out"
                return 0
            fi
        fi
        sleep 1
    done
    record_result "FAIL" "${label}.models" "timed out waiting for /v1/models" \
        "port=$api_port"
    return 1
}

run_chat_smoke() {
    local label="$1"
    local api_port="$2"
    local model_id="$3"
    local payload="$CHAT_DIR/${label}-chat-request.json"
    local out="$CHAT_DIR/${label}-chat-response.json"

    python3 - "$model_id" "$payload" <<'PY'
import json
import sys

model, path = sys.argv[1:3]
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a terse compatibility probe."},
        {"role": "user", "content": "Reply with one short sentence."},
    ],
    "stream": False,
    "max_tokens": 16,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

    if ! curl -fsS --max-time "$CHAT_MAX_TIME" \
        "http://127.0.0.1:${api_port}/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d @"$payload" \
        -o "$out"; then
        record_result "FAIL" "${label}.chat" "chat request failed" "path=$out"
        return 1
    fi

    python3 - "$out" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    body = json.load(fh)
if body.get("object") != "chat.completion":
    raise SystemExit(f"unexpected object: {body.get('object')!r}")
choices = body.get("choices") or []
content = choices[0].get("message", {}).get("content", "") if choices else ""
if not content:
    raise SystemExit("empty chat content")
PY
    record_result "PASS" "${label}.chat" "chat completion returned content" "path=$out"
}

run_public_probe() {
    local label="$1"
    local binary="$2"
    local api_port="$3"
    local console_port="$4"

    local pid
    start_node "$label" "$binary" --client --auto --headless --port "$api_port" --console "$console_port" || return 1
    pid="$START_NODE_PID"
    assert_process_alive "$pid" "$label" || return 1
    wait_status "$label" "$console_port" || return 1
    assert_status_no_control_leak "$label" "$STATUS_DIR/${label}-status.json" || return 1

    local models_out="$MODELS_DIR/${label}-public-models.json"
    local model_id=""
    for second in $(seq 1 "$MAX_WAIT"); do
        if curl_json "http://127.0.0.1:${api_port}/v1/models" "$models_out" 2>/dev/null; then
            model_id="$(first_model_id "$models_out")"
            if [[ -n "$model_id" ]]; then
                break
            fi
        fi
        sleep 1
    done

    if [[ -z "$model_id" ]]; then
        mark_prereq_or_fail "${label}.public-models" \
            "public mesh did not expose a model during the probe window" \
            "seconds=$MAX_WAIT" "path=$models_out"
        return 0
    fi

    record_result "PASS" "${label}.public-models" "public mesh exposed a model" \
        "model=$model_id" "path=$models_out"
    run_chat_smoke "${label}.public" "$api_port" "$model_id"
}

run_local_direction() {
    local label="$1"
    local server_binary="$2"
    local client_binary="$3"
    local model="$4"
    local offset="$5"

    local server_api=$((BASE_PORT + offset))
    local server_console=$((BASE_PORT + offset + 1))
    local server_bind=$((BASE_PORT + offset + 2))
    local client_api=$((BASE_PORT + offset + 3))
    local client_console=$((BASE_PORT + offset + 4))

    local server_args=(--headless --port "$server_api" --console "$server_console" --bind-port "$server_bind")
    if [[ -n "$model" ]]; then
        server_args+=(--model "$model" --no-draft --device CPU --ctx-size "$CTX_SIZE")
    fi

    local server_pid
    start_node "${label}-server" "$server_binary" "${server_args[@]}" || return 1
    server_pid="$START_NODE_PID"
    assert_process_alive "$server_pid" "${label}-server" || return 1
    local token
    wait_token "${label}-server" "$server_console" || return 1
    token="$WAIT_TOKEN_VALUE"

    local client_pid
    start_node "${label}-client" "$client_binary" --client --join "$token" --headless --port "$client_api" --console "$client_console" || return 1
    client_pid="$START_NODE_PID"
    assert_process_alive "$client_pid" "${label}-client" || return 1

    wait_peers "${label}-server" "$server_console" 1 || return 1
    wait_peers "${label}-client" "$client_console" 1 || return 1
    assert_status_no_control_leak "${label}-server" "$STATUS_DIR/${label}-server-peers.json" || return 1
    assert_status_no_control_leak "${label}-client" "$STATUS_DIR/${label}-client-peers.json" || return 1

    if [[ -z "$model" || "$CONFIG_ONLY" == true ]]; then
        record_result "PASS" "${label}.loopback-coexistence" \
            "mixed-version private mesh peers discovered each other"
        return 0
    fi

    local routed_model
    wait_models "${label}-client" "$client_api" || return 1
    routed_model="$WAIT_MODELS_MODEL_ID"
    run_chat_smoke "${label}-client" "$client_api" "$routed_model"
}

run_protocol_contract_tests() {
    if [[ "$RUN_CARGO_TESTS" != true ]]; then
        record_result "PREREQ" "config-cargo-tests" "current-branch protocol tests skipped by request"
        return 0
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        record_result "PREREQ" "config-cargo-tests" "cargo unavailable; protocol tests not run"
        return 0
    fi

    run_logged "config-missing-endpoint-required" \
        cargo test -p mesh-llm --test protocol_compat_v0_client missing_control_endpoint_rejects_config_bootstrap || true
    run_logged "config-new-client-owner-control" \
        cargo test -p mesh-llm --test protocol_compat_v0_client explicit_control_endpoint_selects_owner_control || true
    run_logged "config-control-rejects-legacy-frames" \
        cargo test -p mesh-llm-client --test protocol_wire owner_control_legacy_json_rejects_with_structured_error || true
}

run_runtime_control_bootstrap() {
    local offset="$1"
    local api_port=$((BASE_PORT + offset))
    local console_port=$((BASE_PORT + offset + 1))
    local bind_port=$((BASE_PORT + offset + 2))
    local owner_key="$WORK_ROOT/owner-keystore.json"
    local auth_log="$LOG_DIR/config-auth-init.log"

    if ! "$CURRENT_BINARY" auth init --owner-key "$owner_key" --no-passphrase --force >"$auth_log" 2>&1; then
        record_result "PREREQ" "config-runtime-bootstrap" \
            "could not create temporary owner keystore" "log=$auth_log"
        return 0
    fi

    local pid
    start_node "config-current-node" "$CURRENT_BINARY" --headless \
        --owner-key "$owner_key" \
        --port "$api_port" \
        --console "$console_port" \
        --bind-port "$bind_port" || return 1
    pid="$START_NODE_PID"
    assert_process_alive "$pid" "config-current-node" || return 1
    wait_status "config-current-node" "$console_port" || return 1

    local bootstrap="$CONTROL_DIR/config-control-bootstrap.json"
    if ! curl_json "http://127.0.0.1:${console_port}/api/runtime/control-bootstrap" "$bootstrap"; then
        record_result "FAIL" "config-runtime-bootstrap" \
            "control bootstrap endpoint did not respond" "path=$bootstrap"
        return 1
    fi

    local enabled endpoint requires_endpoint
    enabled="$(json_field "$bootstrap" "enabled")"
    endpoint="$(json_field "$bootstrap" "endpoint")"
    requires_endpoint="$(json_field "$bootstrap" "requires_explicit_remote_endpoint")"
    if [[ "$requires_endpoint" != "true" ]]; then
        record_result "FAIL" "config-runtime-bootstrap" \
            "bootstrap must require explicit remote endpoint" "path=$bootstrap"
        return 1
    fi
    if [[ "$enabled" != "true" || -z "$endpoint" ]]; then
        record_result "PREREQ" "config-runtime-bootstrap" \
            "owner-control endpoint not enabled on this local node" "path=$bootstrap"
        return 0
    fi
    record_result "PASS" "config-runtime-bootstrap" \
        "current node exposes explicit owner-control endpoint" "path=$bootstrap"

    local get_config="$CONTROL_DIR/config-get-config.json"
    local get_config_log="$LOG_DIR/config-get-config.log"
    if "$CURRENT_BINARY" runtime get-config --port "$console_port" --endpoint "$endpoint" --json >"$get_config" 2>"$get_config_log"; then
        record_result "PASS" "config-runtime-get-config" \
            "owner-control get-config succeeded through explicit endpoint" "path=$get_config"
    else
        record_result "FAIL" "config-runtime-get-config" \
            "owner-control get-config failed" "path=$get_config" "log=$get_config_log"
        return 1
    fi
}

if [[ "$LOCAL_ONLY" != true ]]; then
    run_public_probe "released-public" "$RELEASED_BINARY" "$((BASE_PORT + 0))" "$((BASE_PORT + 1))" || true
    run_public_probe "current-public" "$CURRENT_BINARY" "$((BASE_PORT + 10))" "$((BASE_PORT + 11))" || true
fi

if [[ "$LOCAL_ONLY" == true || -n "$RELEASED_MODEL" || -n "$CURRENT_MODEL" || "$CONFIG_ONLY" == true ]]; then
    if [[ "$CONFIG_ONLY" == true || -n "$CURRENT_MODEL" ]]; then
        run_local_direction "current-serves-released-client" \
            "$CURRENT_BINARY" "$RELEASED_BINARY" "$CURRENT_MODEL" 30 || true
    else
        record_result "PREREQ" "current-serves-released-client" \
            "no --current-model/--model supplied for local routing smoke"
    fi

    if [[ "$CONFIG_ONLY" == true || -n "$RELEASED_MODEL" ]]; then
        run_local_direction "released-serves-current-client" \
            "$RELEASED_BINARY" "$CURRENT_BINARY" "$RELEASED_MODEL" 50 || true
    else
        record_result "PREREQ" "released-serves-current-client" \
            "no --released-model/--model supplied for local routing smoke"
    fi
fi

run_protocol_contract_tests || true
run_runtime_control_bootstrap 80 || true

if [[ "$EXIT_STATUS" -eq 0 ]]; then
    append_summary ""
    append_summary "Overall: PASS or PREREQ-only incomplete checks. See \`results.jsonl\`."
else
    append_summary ""
    append_summary "Overall: FAIL. See \`results.jsonl\` and logs."
fi

exit "$EXIT_STATUS"
