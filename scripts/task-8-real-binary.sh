#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN_NAME="${TASK8_PLAN_NAME:-version-admission-bounds}"
EVIDENCE_DIR="${TASK8_EVIDENCE_DIR:-$REPO_ROOT/.sisyphus/evidence}"
NOTEPAD_DIR="${TASK8_NOTEPAD_DIR:-$REPO_ROOT/.sisyphus/notepads/$PLAN_NAME}"
WORK_ROOT="${TASK8_WORK_ROOT:-/tmp/mesh-task8-real-binary}"
RUNTIME_ROOT="$WORK_ROOT/runtime"
if [[ -x "$REPO_ROOT/target/release/mesh-llm" ]]; then
  BIN_SRC="$REPO_ROOT/target/release/mesh-llm"
else
  BIN_SRC="$REPO_ROOT/target/debug/mesh-llm"
fi
export LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$REPO_ROOT/.deps/llama-build/build-stage-abi-cpu}"
STAMP_SEED_HEX="1111111111111111111111111111111111111111111111111111111111111111"
OWNER_KEY="$WORK_ROOT/owner.json"

BUILD_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-build.txt"
SIGNED_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-signed-accepted.txt"
UNSIGNED_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-unsigned-rejected.txt"
TAMPERED_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-tampered-rejected.txt"
VERSION_PROTO_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-version-protocol-rejected.txt"
PARTIAL_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-partial-bounds.txt"
TRANSITIVE_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-transitive.txt"
CLEANUP_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-cleanup.txt"

mkdir -p "$EVIDENCE_DIR" "$NOTEPAD_DIR" "$WORK_ROOT" "$RUNTIME_ROOT"

cleanup_all() {
  pkill -f "$WORK_ROOT/.*/mesh-llm" 2>/dev/null || true
  pkill -f "mesh-task8-real-binary" 2>/dev/null || true
}

trap cleanup_all EXIT

START_NODE_PID=""

json_get() {
  local file="$1"
  local expr="$2"
  python3 - "$file" "$expr" <<'PY'
import json, sys
path, expr = sys.argv[1:3]
with open(path, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
value = data
for part in expr.split('.'):
    if part:
        value = value[part]
if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

wait_for_status() {
  local port="$1"
  local out="$2"
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${port}/api/status" > "$out" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Never persist raw bootstrap tokens to disk. Evidence files record a stable
# SHA-256 fingerprint of the token instead, which is sufficient to correlate
# the scenario without exposing the join secret.
token_fingerprint() {
  local token="$1"
  if [[ -z "$token" ]]; then
    printf 'none'
    return
  fi
  local digest
  digest="$(printf '%s' "$token" | shasum -a 256 | awk '{print $1}')"
  printf 'sha256:%s' "$digest"
}

prepare_binary() {
  local name="$1"
  local node_dir="$WORK_ROOT/$name"
  mkdir -p "$node_dir"
  cp "$BIN_SRC" "$node_dir/mesh-llm"
  chmod +x "$node_dir/mesh-llm"
}

stamp_binary() {
  local name="$1"
  shift
  cargo run -p xtask -- release-attestation stamp --binary "$WORK_ROOT/$name/mesh-llm" --signing-seed-hex "$STAMP_SEED_HEX" "$@"
}

init_owner() {
  rm -f "$OWNER_KEY"
  "$BIN_SRC" auth init --owner-key "$OWNER_KEY" --force --no-passphrase >/dev/null
}

start_node() {
  local name="$1"
  local console="$2"
  local api="$3"
  local bind="$4"
  shift 4
  local node_dir="$WORK_ROOT/$name"
  local log="$node_dir/${name}.log"
  local status="$node_dir/status.json"
  local runtime_dir="$RUNTIME_ROOT/$name"
  mkdir -p "$runtime_dir"
  (
    export MESH_LLM_RUNTIME_ROOT="$runtime_dir"
    "$node_dir/mesh-llm" client --log-format json --headless --owner-required --owner-key "$OWNER_KEY" --console "$console" --port "$api" --bind-port "$bind" "$@" > "$log" 2>&1
  ) &
  local pid=$!
  echo "$pid" > "$node_dir/pid"
  wait_for_status "$console" "$status"
  START_NODE_PID="$pid"
}

stop_node() {
  local name="$1"
  local node_dir="$WORK_ROOT/$name"
  if [[ -f "$node_dir/pid" ]]; then
    local pid
    pid="$(cat "$node_dir/pid")"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

status_token() {
  local name="$1"
  json_get "$WORK_ROOT/$name/status.json" token
}

status_recent_reason() {
  local name="$1"
  python3 - "$WORK_ROOT/$name/status.json" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    data = json.load(fh)
recent = data.get('recent_mesh_rejections') or []
print(recent[0]['reason'] if recent else '')
PY
}

status_peer_count() {
  local name="$1"
  python3 - "$WORK_ROOT/$name/status.json" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    data = json.load(fh)
print(len(data.get('peers') or []))
PY
}

record_file_header() {
  local output="$1"
  shift
  {
    printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'work_root=%s\n' "$WORK_ROOT"
    for line in "$@"; do
      printf '%s\n' "$line"
    done
    printf '\n'
  } > "$output"
}

append_file() {
  local output="$1"
  shift
  for file in "$@"; do
    {
      printf '\n[file:%s]\n' "$file"
      cat "$file"
      printf '\n'
    } >> "$output"
  done
}

capture_node_state() {
  local output="$1"
  local name="$2"
  {
    printf '\n[node:%s]\n' "$name"
    printf 'pid=%s\n' "$(cat "$WORK_ROOT/$name/pid")"
    printf 'binary=%s\n' "$WORK_ROOT/$name/mesh-llm"
    printf 'attestation_path=%s\n' "$WORK_ROOT/$name/mesh-llm.attestation.json"
    printf 'status_json=\n'
    cat "$WORK_ROOT/$name/status.json"
    printf '\nlog_tail=\n'
    python3 - "$WORK_ROOT/$name/${name}.log" <<'PY'
import sys
from collections import deque
with open(sys.argv[1], 'r', encoding='utf-8', errors='replace') as fh:
    for line in deque(fh, maxlen=40):
        print(line, end='')
PY
    printf '\n'
  } >> "$output"
}

run_build_evidence() {
  prepare_binary signed-a
  stamp_binary signed-a --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/signed-a-summary.json"
  "$WORK_ROOT/signed-a/mesh-llm" --version > "$WORK_ROOT/version.txt"
  record_file_header "$BUILD_EVIDENCE" \
    "binary_path=$WORK_ROOT/signed-a/mesh-llm" \
    "version=$(tr -d '\n' < "$WORK_ROOT/version.txt")" \
    "owner_key=$OWNER_KEY"
  append_file "$BUILD_EVIDENCE" "$WORK_ROOT/signed-a-summary.json"
}

run_signed_acceptance() {
  prepare_binary signed-host
  prepare_binary signed-joiner
  stamp_binary signed-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/signed-host-summary.json"
  stamp_binary signed-joiner --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/signed-joiner-summary.json"
  local signer_key
  signer_key="$(json_get "$WORK_ROOT/signed-host-summary.json" signer_key_id)"
  start_node signed-host 3411 9411 7841 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.65.0 --max-node-version 0.65.9 --min-protocol-version 1 --max-protocol-version 1
  local host_pid="$START_NODE_PID"
  local token
  token="$(status_token signed-host)"
  start_node signed-joiner 3412 9412 7842 --join "$token"
  local joiner_pid="$START_NODE_PID"
  sleep 4
  wait_for_status 3411 "$WORK_ROOT/signed-host/status.json"
  wait_for_status 3412 "$WORK_ROOT/signed-joiner/status.json"
  record_file_header "$SIGNED_EVIDENCE" \
    "scenario=signed accepted" \
    "host_pid=$host_pid" \
    "joiner_pid=$joiner_pid" \
    "join_token_fingerprint=$(token_fingerprint "$token")" \
    "host_peer_count=$(status_peer_count signed-host)" \
    "joiner_peer_count=$(status_peer_count signed-joiner)"
  append_file "$SIGNED_EVIDENCE" "$WORK_ROOT/signed-host-summary.json" "$WORK_ROOT/signed-joiner-summary.json"
  capture_node_state "$SIGNED_EVIDENCE" signed-host
  capture_node_state "$SIGNED_EVIDENCE" signed-joiner
  stop_node signed-joiner
  stop_node signed-host
}

run_unsigned_rejection() {
  prepare_binary unsigned-host
  prepare_binary unsigned-joiner
  stamp_binary unsigned-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/unsigned-host-summary.json"
  local signer_key
  signer_key="$(json_get "$WORK_ROOT/unsigned-host-summary.json" signer_key_id)"
  start_node unsigned-host 3421 9421 7851 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.65.0 --max-node-version 0.65.9 --min-protocol-version 1 --max-protocol-version 1 >/dev/null
  local token
  token="$(status_token unsigned-host)"
  start_node unsigned-joiner 3422 9422 7852 --join "$token" >/dev/null
  sleep 4
  wait_for_status 3421 "$WORK_ROOT/unsigned-host/status.json"
  wait_for_status 3422 "$WORK_ROOT/unsigned-joiner/status.json"
  record_file_header "$UNSIGNED_EVIDENCE" \
    "scenario=unsigned rejected" \
    "join_token_fingerprint=$(token_fingerprint "$token")" \
    "host_recent_reason=$(status_recent_reason unsigned-host)" \
    "joiner_recent_reason=$(status_recent_reason unsigned-joiner)"
  append_file "$UNSIGNED_EVIDENCE" "$WORK_ROOT/unsigned-host-summary.json"
  capture_node_state "$UNSIGNED_EVIDENCE" unsigned-host
  capture_node_state "$UNSIGNED_EVIDENCE" unsigned-joiner
  stop_node unsigned-joiner
  stop_node unsigned-host
}

run_tampered_rejection() {
  prepare_binary tampered-host
  prepare_binary tampered-joiner
  stamp_binary tampered-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/tampered-host-summary.json"
  stamp_binary tampered-joiner --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/tampered-joiner-summary.json"
  python3 - "$WORK_ROOT/tampered-joiner/mesh-llm.attestation.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
data['signature'] = []
with open(path, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2)
PY
  local signer_key
  signer_key="$(json_get "$WORK_ROOT/tampered-host-summary.json" signer_key_id)"
  start_node tampered-host 3431 9431 7861 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.65.0 --max-node-version 0.65.9 --min-protocol-version 1 --max-protocol-version 1 >/dev/null
  local token
  token="$(status_token tampered-host)"
  start_node tampered-joiner 3432 9432 7862 --join "$token" >/dev/null
  sleep 4
  wait_for_status 3431 "$WORK_ROOT/tampered-host/status.json"
  wait_for_status 3432 "$WORK_ROOT/tampered-joiner/status.json"
  record_file_header "$TAMPERED_EVIDENCE" \
    "scenario=tampered rejected" \
    "join_token_fingerprint=$(token_fingerprint "$token")" \
    "host_recent_reason=$(status_recent_reason tampered-host)" \
    "joiner_recent_reason=$(status_recent_reason tampered-joiner)"
  append_file "$TAMPERED_EVIDENCE" "$WORK_ROOT/tampered-host-summary.json" "$WORK_ROOT/tampered-joiner-summary.json"
  capture_node_state "$TAMPERED_EVIDENCE" tampered-host
  capture_node_state "$TAMPERED_EVIDENCE" tampered-joiner
  stop_node tampered-joiner
  stop_node tampered-host
}

run_version_protocol_rejection() {
  prepare_binary version-host
  prepare_binary version-joiner
  prepare_binary protocol-host
  prepare_binary protocol-joiner
  stamp_binary version-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/version-host-summary.json"
  stamp_binary version-joiner --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/version-joiner-summary.json"
  stamp_binary protocol-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/protocol-host-summary.json"
  stamp_binary protocol-joiner --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/protocol-joiner-summary.json"
  local signer_key
  signer_key="$(json_get "$WORK_ROOT/version-host-summary.json" signer_key_id)"

  start_node version-host 3441 9441 7871 --require-release-attestation --release-signer-key "$signer_key" --max-node-version 0.65.0 >/dev/null
  local version_token
  version_token="$(status_token version-host)"
  start_node version-joiner 3442 9442 7872 --join "$version_token" >/dev/null
  sleep 4
  wait_for_status 3441 "$WORK_ROOT/version-host/status.json"
  wait_for_status 3442 "$WORK_ROOT/version-joiner/status.json"

  start_node protocol-host 3443 9443 7873 --require-release-attestation --release-signer-key "$signer_key" --min-protocol-version 2 --max-protocol-version 2 >/dev/null
  local protocol_token
  protocol_token="$(status_token protocol-host)"
  start_node protocol-joiner 3444 9444 7874 --join "$protocol_token" >/dev/null
  sleep 4
  wait_for_status 3443 "$WORK_ROOT/protocol-host/status.json"
  wait_for_status 3444 "$WORK_ROOT/protocol-joiner/status.json"

  record_file_header "$VERSION_PROTO_EVIDENCE" \
    "scenario=version and protocol rejection" \
    "version_host_recent_reason=$(status_recent_reason version-host)" \
    "protocol_host_recent_reason=$(status_recent_reason protocol-host)"
  append_file "$VERSION_PROTO_EVIDENCE" "$WORK_ROOT/version-host-summary.json" "$WORK_ROOT/version-joiner-summary.json" "$WORK_ROOT/protocol-host-summary.json" "$WORK_ROOT/protocol-joiner-summary.json"
  capture_node_state "$VERSION_PROTO_EVIDENCE" version-host
  capture_node_state "$VERSION_PROTO_EVIDENCE" version-joiner
  capture_node_state "$VERSION_PROTO_EVIDENCE" protocol-host
  capture_node_state "$VERSION_PROTO_EVIDENCE" protocol-joiner
  stop_node version-joiner
  stop_node version-host
  stop_node protocol-joiner
  stop_node protocol-host
}

run_partial_bounds() {
  prepare_binary minonly-host
  prepare_binary minonly-joiner
  prepare_binary maxonly-host
  prepare_binary maxonly-joiner
  stamp_binary minonly-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/minonly-host-summary.json"
  stamp_binary minonly-joiner --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/minonly-joiner-summary.json"
  stamp_binary maxonly-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/maxonly-host-summary.json"
  stamp_binary maxonly-joiner --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/maxonly-joiner-summary.json"
  local signer_key
  signer_key="$(json_get "$WORK_ROOT/minonly-host-summary.json" signer_key_id)"
  start_node minonly-host 3451 9451 7881 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.65.0 --min-protocol-version 1 >/dev/null
  local min_token
  min_token="$(status_token minonly-host)"
  start_node minonly-joiner 3452 9452 7882 --join "$min_token" >/dev/null
  sleep 4
  wait_for_status 3451 "$WORK_ROOT/minonly-host/status.json"
  wait_for_status 3452 "$WORK_ROOT/minonly-joiner/status.json"

  start_node maxonly-host 3453 9453 7883 --require-release-attestation --release-signer-key "$signer_key" --max-node-version 0.65.9 --max-protocol-version 1 >/dev/null
  local max_token
  max_token="$(status_token maxonly-host)"
  start_node maxonly-joiner 3454 9454 7884 --join "$max_token" >/dev/null
  sleep 4
  wait_for_status 3453 "$WORK_ROOT/maxonly-host/status.json"
  wait_for_status 3454 "$WORK_ROOT/maxonly-joiner/status.json"

  record_file_header "$PARTIAL_EVIDENCE" \
    "scenario=partial bounds" \
    "minonly_host_peer_count=$(status_peer_count minonly-host)" \
    "maxonly_host_peer_count=$(status_peer_count maxonly-host)"
  append_file "$PARTIAL_EVIDENCE" "$WORK_ROOT/minonly-host-summary.json" "$WORK_ROOT/minonly-joiner-summary.json" "$WORK_ROOT/maxonly-host-summary.json" "$WORK_ROOT/maxonly-joiner-summary.json"
  capture_node_state "$PARTIAL_EVIDENCE" minonly-host
  capture_node_state "$PARTIAL_EVIDENCE" minonly-joiner
  capture_node_state "$PARTIAL_EVIDENCE" maxonly-host
  capture_node_state "$PARTIAL_EVIDENCE" maxonly-joiner
  stop_node minonly-joiner
  stop_node minonly-host
  stop_node maxonly-joiner
  stop_node maxonly-host
}

run_transitive() {
  prepare_binary transitive-host
  prepare_binary transitive-bridge
  prepare_binary transitive-client
  stamp_binary transitive-host --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/transitive-host-summary.json"
  stamp_binary transitive-bridge --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/transitive-bridge-summary.json"
  stamp_binary transitive-client --protocol-min 1 --protocol-max 1 > "$WORK_ROOT/transitive-client-summary.json"
  local signer_key
  signer_key="$(json_get "$WORK_ROOT/transitive-host-summary.json" signer_key_id)"
  start_node transitive-host 3461 9461 7891 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.65.0 --max-node-version 0.65.9 --min-protocol-version 1 --max-protocol-version 1 >/dev/null
  local host_token
  host_token="$(status_token transitive-host)"
  start_node transitive-bridge 3462 9462 7892 --join "$host_token" >/dev/null
  sleep 4
  local bridge_token
  bridge_token="$(status_token transitive-bridge)"
  rm -f "$WORK_ROOT/transitive-client/mesh-llm.attestation.json"
  start_node transitive-client 3463 9463 7893 --join "$bridge_token" >/dev/null
  sleep 5
  wait_for_status 3461 "$WORK_ROOT/transitive-host/status.json"
  wait_for_status 3462 "$WORK_ROOT/transitive-bridge/status.json"
  wait_for_status 3463 "$WORK_ROOT/transitive-client/status.json"
  record_file_header "$TRANSITIVE_EVIDENCE" \
    "scenario=transitive non-admission" \
    "host_peer_count=$(status_peer_count transitive-host)" \
    "bridge_peer_count=$(status_peer_count transitive-bridge)" \
    "client_peer_count=$(status_peer_count transitive-client)" \
    "client_recent_reason=$(status_recent_reason transitive-client)"
  append_file "$TRANSITIVE_EVIDENCE" "$WORK_ROOT/transitive-host-summary.json" "$WORK_ROOT/transitive-bridge-summary.json" "$WORK_ROOT/transitive-client-summary.json"
  capture_node_state "$TRANSITIVE_EVIDENCE" transitive-host
  capture_node_state "$TRANSITIVE_EVIDENCE" transitive-bridge
  capture_node_state "$TRANSITIVE_EVIDENCE" transitive-client
  stop_node transitive-client
  stop_node transitive-bridge
  stop_node transitive-host
}

run_cleanup_evidence() {
  record_file_header "$CLEANUP_EVIDENCE" "cleanup=pkill -f mesh-task8-real-binary || true"
  cleanup_all
  {
    printf 'cleanup_check_command=ps -eo pid=,comm=,args=\n'
    ps -eo pid=,comm=,args= | python3 - <<'PY'
import sys
count = 0
for line in sys.stdin:
    parts = line.strip().split(None, 2)
    if len(parts) >= 2 and parts[1] == 'mesh-llm':
        print(line, end='')
        count += 1
print(f'remaining_mesh_llm_binary_process_count={count}')
PY
  } >> "$CLEANUP_EVIDENCE"
}

append_notepads() {
  cat >> "$NOTEPAD_DIR/learnings.md" <<'EOF'

## Task 8 real-binary validation
- Runtime startup now loads a sibling `mesh-llm.attestation.json` sidecar so compiled binaries can advertise real `ReleaseBuildAttestation` payloads without a test-only injection path.
- `cargo run -p xtask -- release-attestation stamp` produces deterministic test-signer sidecars and reports canonical attestation hashes for evidence capture.
EOF
  cat >> "$NOTEPAD_DIR/issues.md" <<'EOF'

## Task 8 real-binary issues
- Current requirement evaluation cryptographically verifies the release-attestation signature from `signer_key_id`, and malformed or tampered attestation can surface as `BuildProofInvalid` during admission.
EOF
  cat >> "$NOTEPAD_DIR/problems.md" <<'EOF'

## Task 8 real-binary problems
- Scenario 7 wording says an "unsigned/unverified" Node C should stay unadmitted until direct verification. With current behavior an actually unsigned direct join is rejected immediately and cannot later promote without adding attestation first, so the evidence captures unsigned non-admission through the bridge path and records that limitation explicitly.
EOF
}

run_selected() {
  local target="${1:-all}"
  cleanup_all
  init_owner
  case "$target" in
    all)
      run_build_evidence
      run_signed_acceptance
      run_unsigned_rejection
      run_tampered_rejection
      run_version_protocol_rejection
      run_partial_bounds
      run_transitive
      run_cleanup_evidence
      append_notepads
      ;;
    unsigned)
      run_unsigned_rejection
      ;;
    tampered)
      run_tampered_rejection
      ;;
    version-protocol)
      run_version_protocol_rejection
      ;;
    partial-bounds)
      run_partial_bounds
      ;;
    transitive)
      run_transitive
      ;;
    cleanup)
      run_cleanup_evidence
      append_notepads
      ;;
    *)
      printf 'unknown scenario: %s\n' "$target" >&2
      return 1
      ;;
  esac
}

run_selected "$@"
