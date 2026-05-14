#!/usr/bin/env bash
set -euo pipefail

# plan-clippy-batches.sh: deterministically binpack Rust crates for PR clippy.
#
# Usage:
#   bash scripts/plan-clippy-batches.sh --all [--bins 3]
#   bash scripts/plan-clippy-batches.sh --crates-json '["mesh-llm","model-ref"]' [--bins 3]
#
# The script emits a JSON array of matrix entries:
#   [{"idx":0,"weight":10,"crates":["mesh-llm"]}, ...]

WORKSPACE_MEMBERS=(
  "mesh-llm"
  "mesh-llm-gpu-bench"
  "mesh-llm-host-runtime"
  "mesh-llm-identity"
  "mesh-llm-protocol"
  "mesh-llm-routing"
  "mesh-llm-system"
  "mesh-llm-types"
  "mesh-llm-ui"
  "mesh-llm-plugin"
  "mesh-llm-client"
  "mesh-api"
  "mesh-host-core"
  "mesh-api-ffi"
  "mesh-llm-test-harness"
  "model-ref"
  "model-artifact"
  "model-hf"
  "model-resolver"
  "skippy-protocol"
  "skippy-coordinator"
  "skippy-topology"
  "skippy-cache"
  "skippy-metrics"
  "openai-frontend"
  "skippy-ffi"
  "skippy-runtime"
  "skippy-server"
  "metrics-server"
  "skippy-model-package"
  "model-package"
  "skippy-correctness"
  "llama-spec-bench"
  "skippy-bench"
  "skippy-prompt"
  "xtask"
)

usage() {
  cat >&2 <<'EOF'
usage: plan-clippy-batches.sh (--all | --crates-json JSON) [--bins N]
EOF
}

mode=""
crates_json="[]"
bins=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      mode="all"
      shift
      ;;
    --crates-json)
      mode="json"
      crates_json="${2:-}"
      shift 2
      ;;
    --bins)
      bins="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$mode" ]]; then
  usage
  exit 2
fi

if ! [[ "$bins" =~ ^[1-9][0-9]*$ ]]; then
  echo "bins must be a positive integer: $bins" >&2
  exit 2
fi

if [[ "$mode" == "all" ]]; then
  crates_json=$(printf '%s\n' "${WORKSPACE_MEMBERS[@]}" | jq -Rsc 'split("\n") | map(select(length > 0))')
else
  jq -e 'type == "array"' >/dev/null <<<"$crates_json"
fi

python3 - "$bins" "$crates_json" <<'PY'
import json
import sys

bins = int(sys.argv[1])
crates = json.loads(sys.argv[2])

# Static, deterministic weights approximate clippy cost from recent PR runs.
# Unknown crates intentionally default to 1 so new crates still get scheduled.
weights = {
    "mesh-llm": 10,
    "mesh-llm-host-runtime": 8,
    "mesh-llm-client": 6,
    "mesh-api-ffi": 5,
    "skippy-server": 5,
    "skippy-runtime": 5,
    "skippy-correctness": 5,
    "model-package": 5,
    "skippy-bench": 4,
    "skippy-model-package": 4,
    "openai-frontend": 4,
    "model-artifact": 4,
    "model-hf": 4,
    "model-resolver": 3,
    "mesh-api": 3,
    "mesh-llm-gpu-bench": 3,
    "llama-spec-bench": 3,
    "skippy-prompt": 3,
    "mesh-llm-system": 3,
    "mesh-llm-routing": 2,
    "mesh-llm-protocol": 2,
    "mesh-llm-types": 2,
    "mesh-llm-ui": 2,
    "mesh-llm-plugin": 2,
    "mesh-host-core": 2,
    "skippy-protocol": 2,
    "skippy-topology": 2,
    "skippy-cache": 2,
    "skippy-metrics": 2,
    "metrics-server": 2,
    "mesh-llm-identity": 1,
    "mesh-llm-test-harness": 1,
    "model-ref": 1,
    "skippy-coordinator": 1,
    "xtask": 1,
}

deduped = []
seen = set()
for crate in crates:
    if crate == "skippy-ffi":
        continue
    if crate not in seen:
        seen.add(crate)
        deduped.append(crate)

if not deduped:
    print(json.dumps([{"idx": 0, "weight": 0, "crates": []}], separators=(",", ":")))
    raise SystemExit(0)

indexed = [(crate, weights.get(crate, 1), idx) for idx, crate in enumerate(deduped)]
indexed.sort(key=lambda item: (-item[1], item[2], item[0]))

buckets = [{"idx": idx, "weight": 0, "crates": []} for idx in range(bins)]
for crate, weight, _ in indexed:
    target = min(buckets, key=lambda bucket: (bucket["weight"], bucket["idx"]))
    target["crates"].append(crate)
    target["weight"] += weight

# Preserve all bins so matrix naming is stable even when one bin is empty.
print(json.dumps(buckets, separators=(",", ":")))
PY
