#!/usr/bin/env bash
# bench-moa.sh — latency + path benchmark for MoA fan-out.
#
# Sends N chat completion requests with model="mesh" to a running mesh-llm
# endpoint and reports p50/p95/p99 wall-clock latency plus a histogram of
# the gateway paths actually taken (fanout vs early-exit vs tool-result),
# reducer engagement, and hedge rate.
#
# Usage:
#   ./evals/bench-moa.sh                          # 20 requests, localhost:9337
#   N=50 ./evals/bench-moa.sh                     # 50 requests
#   BASE_URL=http://host:9337 ./evals/bench-moa.sh
#   PROMPT="why is the sky blue?" ./evals/bench-moa.sh
#
# Per-request output shows the gateway-emitted x-moa-* headers so you can
# eyeball individual turns. The summary block aggregates them. Failed
# requests are reported but excluded from percentiles and from path/reducer
# aggregates.
#
# Run on the same machine as a serving host so wall-clock RTT is dominated
# by inference + arbitration, not network.
#
# Requires the gateway to emit x-moa-* response headers (added alongside
# the introduction of TurnKind / reducer_attempts in TurnResult). Older
# mesh-llm binaries will simply show blanks in the path/reducer columns.

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
printf 'idx\tstatus\tms\tcontent_len\tturn\tworkers\tworkers_ok\treducer\treducer_attempts\n' >"$RESULTS"

echo "Running $N requests against $BASE_URL …"
echo "  prompt: $PROMPT"
echo "  output: $OUT_DIR"
echo

body=$(jq -nc --arg p "$PROMPT" --argjson mt "$MAX_TOKENS" \
  '{model:"mesh", messages:[{role:"user", content:$p}], max_tokens:$mt, stream:false}')

# Case-insensitive header extractor: x-moa-* values land lowercase per HTTP
# but some proxies preserve case. Match both.
extract_header() {
  local file="$1" name="$2"
  # Strip CRs, find "name: value", return value trimmed.
  awk -v IGNORECASE=1 -v name="$name" '
    BEGIN{ FS=": " }
    /^[A-Za-z0-9-]+: / {
      if (tolower($1) == tolower(name)) {
        sub(/\r$/, "", $0);
        # rebuild value (could contain ": ")
        $1 = "";
        sub(/^ /, "", $0);
        print $0;
        exit
      }
    }
  ' "$file"
}

for i in $(seq 1 "$N"); do
  start_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
  http_code="$(curl -sS -o "$OUT_DIR/resp.$i.json" \
    -D "$OUT_DIR/hdr.$i.txt" \
    -w '%{http_code}' \
    -H 'content-type: application/json' \
    -X POST "${BASE_URL%/}/v1/chat/completions" \
    -d "$body" || echo "000")"
  end_ns="$(python3 -c 'import time; print(time.monotonic_ns())')"
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

  content_len=0
  turn=""; workers=""; workers_ok=""; reducer=""; reducer_attempts=""
  if [[ "$http_code" == "200" ]]; then
    content_len="$(jq -r '.choices[0].message.content // "" | length' "$OUT_DIR/resp.$i.json")"
    turn="$(extract_header "$OUT_DIR/hdr.$i.txt" "x-moa-turn")"
    workers="$(extract_header "$OUT_DIR/hdr.$i.txt" "x-moa-workers")"
    workers_ok="$(extract_header "$OUT_DIR/hdr.$i.txt" "x-moa-workers-ok")"
    reducer="$(extract_header "$OUT_DIR/hdr.$i.txt" "x-moa-reducer")"
    reducer_attempts="$(extract_header "$OUT_DIR/hdr.$i.txt" "x-moa-reducer-attempts")"
  fi
  printf '%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$i" "$http_code" "$elapsed_ms" "$content_len" \
    "$turn" "$workers" "$workers_ok" "$reducer" "$reducer_attempts" \
    >>"$RESULTS"

  # Compact per-request line. Show turn/reducer when present, blank otherwise.
  turn_disp="${turn:--}"
  red_disp="-"
  if [[ -n "$reducer" ]]; then
    if [[ "$reducer" == "true" ]]; then
      red_disp="reducer×${reducer_attempts:-?}"
    else
      red_disp="no-reducer"
    fi
  fi
  workers_disp="-"
  if [[ -n "$workers" ]]; then
    workers_disp="${workers_ok:-?}/${workers}"
  fi
  printf '  [%3d/%d] %s %6dms %5d chars  %-11s %-7s %s\n' \
    "$i" "$N" "$http_code" "$elapsed_ms" "$content_len" \
    "$turn_disp" "$workers_disp" "$red_disp"
done

echo
echo "Summary:"
python3 - "$RESULTS" <<'PY'
import statistics, sys
from collections import Counter

path = sys.argv[1]
ok_rows, fail = [], 0
with open(path) as f:
    header = next(f).rstrip("\n").split("\t")
    cols = {name: i for i, name in enumerate(header)}
    for line in f:
        row = line.rstrip("\n").split("\t")
        # pad short rows defensively
        while len(row) < len(header):
            row.append("")
        if row[cols["status"]] == "200":
            ok_rows.append(row)
        else:
            fail += 1

total = len(ok_rows) + fail
print(f"  total:    {total}")
print(f"  ok:       {len(ok_rows)}")
print(f"  failed:   {fail}")

if not ok_rows:
    sys.exit(0)

# Latency percentiles
ms_vals = sorted(int(r[cols["ms"]]) for r in ok_rows)
def pct(p):
    k = max(0, min(len(ms_vals) - 1, int(round(p / 100 * (len(ms_vals) - 1)))))
    return ms_vals[k]
print()
print("  Latency (ms):")
print(f"    min:    {ms_vals[0]}")
print(f"    p50:    {pct(50)}")
print(f"    p95:    {pct(95)}")
print(f"    p99:    {pct(99)}")
print(f"    max:    {ms_vals[-1]}")
print(f"    mean:   {round(statistics.mean(ms_vals))}")
if len(ms_vals) > 1:
    print(f"    stdev:  {round(statistics.stdev(ms_vals))}")

# Path aggregates — only rows that have non-empty x-moa-turn
tagged = [r for r in ok_rows if r[cols["turn"]]]
if not tagged:
    print()
    print("  (no x-moa-* headers seen — server may predate observability headers)")
    sys.exit(0)

print()
print(f"  Gateway paths  ({len(tagged)}/{len(ok_rows)} tagged):")
turn_counts = Counter(r[cols["turn"]] for r in tagged)
for kind in ("fanout", "early-exit", "tool-result", "failed"):
    c = turn_counts.get(kind, 0)
    if c:
        pct_share = 100.0 * c / len(tagged)
        print(f"    {kind:<12} {c:>4}  ({pct_share:5.1f}%)")
# Any unexpected labels
for kind, c in turn_counts.items():
    if kind not in ("fanout", "early-exit", "tool-result", "failed"):
        print(f"    {kind:<12} {c:>4}  (unrecognized turn kind)")

# Reducer engagement
reducer_used = sum(1 for r in tagged if r[cols["reducer"]] == "true")
print()
print("  Reducer:")
print(f"    invoked:  {reducer_used}/{len(tagged)}  ({100.0 * reducer_used / len(tagged):.1f}%)")
if reducer_used:
    attempts = [
        int(r[cols["reducer_attempts"]])
        for r in tagged
        if r[cols["reducer"]] == "true" and r[cols["reducer_attempts"]].isdigit()
    ]
    if attempts:
        hedged = sum(1 for a in attempts if a >= 2)
        print(f"    hedged:   {hedged}/{len(attempts)}  ({100.0 * hedged / len(attempts):.1f}% of reducer turns)")
        print(f"    avg attempts: {statistics.mean(attempts):.2f}")
        print(f"    max attempts: {max(attempts)}")

# Worker fan-out width (when reported)
widths = [int(r[cols["workers"]]) for r in tagged if r[cols["workers"]].isdigit()]
if widths:
    print()
    print("  Worker fan-out:")
    print(f"    avg width: {statistics.mean(widths):.2f}")
    width_counts = Counter(widths)
    for w in sorted(width_counts):
        print(f"    {w:>2} workers: {width_counts[w]}")

# Latency split by path — useful for spotting "reducer turns are 4× slower"
print()
print("  Latency p50 by path (ms):")
by_kind = {}
for r in tagged:
    by_kind.setdefault(r[cols["turn"]], []).append(int(r[cols["ms"]]))
for kind in ("fanout", "early-exit", "tool-result"):
    vals = by_kind.get(kind, [])
    if vals:
        s = sorted(vals)
        m = s[len(s) // 2]
        print(f"    {kind:<12} p50={m:>5}  (n={len(vals)})")
PY
echo
echo "Raw results: $RESULTS"
