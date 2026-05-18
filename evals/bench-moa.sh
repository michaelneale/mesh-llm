#!/usr/bin/env bash
# bench-moa.sh — quick latency benchmark for MoA fan-out.
#
# Sends N chat completion requests with model="mesh" to a running mesh-llm
# endpoint and reports p50/p95/p99 wall-clock latency.
#
# Usage:
#   ./evals/bench-moa.sh                          # 20 requests, localhost:9337
#   N=50 ./evals/bench-moa.sh                     # 50 requests
#   BASE_URL=http://host:9337 ./evals/bench-moa.sh
#   PROMPT="why is the sky blue?" ./evals/bench-moa.sh
#
# Outputs per-request lines (idx, ms, model, reducer_used, worker_count) and a
# summary block. Failed requests are reported but excluded from percentiles.
#
# Run on the same machine as a serving host so wall-clock RTT is dominated by
# inference + arbitration, not network.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:9337}"
N="${N:-20}"
PROMPT="${PROMPT:-Briefly: what is 17 squared?}"
MAX_TOKENS="${MAX_TOKENS:-128}"

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required" >&2
  exit 1
fi

# Probe: ensure mesh is up and has ≥2 models (else MoA returns 503).
echo "Probing ${BASE_URL%/}/v1/models …"
models_json="$(curl -fsS "${BASE_URL%/}/v1/models" || true)"
if [[ -z "$models_json" ]]; then
  echo "ERROR: cannot reach ${BASE_URL}/v1/models" >&2
  exit 1
fi
model_count="$(jq '.data | length' <<<"$models_json")"
echo "  endpoint: $BASE_URL"
echo "  models:   $model_count"
if [[ "$model_count" -lt 2 ]]; then
  echo "WARN: MoA requires ≥2 models; got $model_count — expect 503s" >&2
fi

OUT_DIR="${OUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/moa-bench.XXXXXX")}"
mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/results.tsv"
printf 'idx\tstatus\tms\tcontent_len\n' >"$RESULTS"

echo "Running $N requests against $BASE_URL …"
echo "  prompt: $PROMPT"
echo "  output: $OUT_DIR"
echo

body=$(jq -nc --arg p "$PROMPT" --argjson mt "$MAX_TOKENS" \
  '{model:"mesh", messages:[{role:"user", content:$p}], max_tokens:$mt, stream:false}')

for i in $(seq 1 "$N"); do
  start_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
  http_code="$(curl -sS -o "$OUT_DIR/resp.$i.json" \
    -w '%{http_code}' \
    -H 'content-type: application/json' \
    -X POST "${BASE_URL%/}/v1/chat/completions" \
    -d "$body" || echo "000")"
  end_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

  content_len=0
  if [[ "$http_code" == "200" ]]; then
    content_len="$(jq -r '.choices[0].message.content // "" | length' "$OUT_DIR/resp.$i.json")"
  fi
  printf '%d\t%s\t%d\t%s\n' "$i" "$http_code" "$elapsed_ms" "$content_len" >>"$RESULTS"
  printf '  [%3d/%d] %s %6dms %5d chars\n' "$i" "$N" "$http_code" "$elapsed_ms" "$content_len"
done

echo
echo "Summary:"
python3 - "$RESULTS" <<'PY'
import statistics, sys
path = sys.argv[1]
ok, fail = [], 0
with open(path) as f:
    next(f)  # header
    for line in f:
        idx, status, ms, _ = line.strip().split("\t")
        if status == "200":
            ok.append(int(ms))
        else:
            fail += 1
total = len(ok) + fail
print(f"  total:    {total}")
print(f"  ok:       {len(ok)}")
print(f"  failed:   {fail}")
if ok:
    ok_sorted = sorted(ok)
    def pct(p):
        k = max(0, min(len(ok_sorted) - 1, int(round(p/100*(len(ok_sorted)-1)))))
        return ok_sorted[k]
    print(f"  min:      {ok_sorted[0]} ms")
    print(f"  p50:      {pct(50)} ms")
    print(f"  p95:      {pct(95)} ms")
    print(f"  p99:      {pct(99)} ms")
    print(f"  max:      {ok_sorted[-1]} ms")
    print(f"  mean:     {round(statistics.mean(ok))} ms")
    if len(ok) > 1:
        print(f"  stdev:    {round(statistics.stdev(ok))} ms")
PY
echo
echo "Raw results: $RESULTS"
