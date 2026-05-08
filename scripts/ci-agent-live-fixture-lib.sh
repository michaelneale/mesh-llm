#!/usr/bin/env bash

agent_smoke_normalize_v1_base() {
    local base_url="${1:?base URL required}"
    base_url="${base_url%/}"
    if [[ "$base_url" != */v1 ]]; then
        base_url="${base_url}/v1"
    fi
    printf '%s\n' "$base_url"
}

agent_smoke_pick_model() {
    local base_url="${1:?base URL required}"
    local requested="${2:-}"

    if [[ -n "$requested" ]]; then
        printf '%s\n' "$requested"
        return 0
    fi

    curl -sf "${base_url%/}/models" |
        python3 -c 'import json,sys
data=json.load(sys.stdin).get("data", [])
preferred=("minimax", "glm", "qwen", "coder", "hermes")
ids=[item.get("id","") for item in data if item.get("id")]
for needle in preferred:
    for model_id in ids:
        if needle in model_id.lower():
            print(model_id)
            raise SystemExit
print(ids[0] if ids else "")'
}

agent_smoke_write_fixture() {
    local work_dir="${1:?work dir required}"
    mkdir -p "${work_dir}/facts" "${work_dir}/src" "${work_dir}/notes" "${work_dir}/tests"

    cat >"${work_dir}/README.md" <<'EOF'
# Agent Smoke Fixture

This repository is intentionally tiny. The smoke test should answer only from
files on disk, not from the prompt text.
EOF

    cat >"${work_dir}/facts/signal.md" <<'EOF'
# Runtime Signal

CODEWORD=signal-7429

Question seed:
Which file names contain the word signal?
EOF

    cat >"${work_dir}/src/matrix.txt" <<'EOF'
checksum: FS-319-DELTA
numbers: 2 3 4 5
hint: the prime sum is computed from the numbers line
EOF

    cat >"${work_dir}/src/smoke_calc.py" <<'EOF'
from __future__ import annotations


def parse_codeword(path: str) -> str:
    """Return the CODEWORD value from a small key/value markdown file."""
    raise NotImplementedError("ci smoke fixture")


def prime_sum_from_matrix(path: str) -> int:
    """Return the sum of prime numbers from the numbers line in matrix.txt."""
    raise NotImplementedError("ci smoke fixture")
EOF

    cat >"${work_dir}/tests/test_smoke_calc.py" <<'EOF'
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke_calc import parse_codeword, prime_sum_from_matrix


ROOT = Path(__file__).resolve().parents[1]


class SmokeCalcTests(unittest.TestCase):
    def test_parse_codeword(self):
        self.assertEqual(parse_codeword(str(ROOT / "facts" / "signal.md")), "signal-7429")

    def test_prime_sum_from_matrix(self):
        self.assertEqual(prime_sum_from_matrix(str(ROOT / "src" / "matrix.txt")), 10)


if __name__ == "__main__":
    unittest.main()
EOF

    cat >"${work_dir}/notes/manifest.txt" <<'EOF'
tracked files:
- facts/signal.md
- src/matrix.txt
- src/smoke_calc.py
- tests/test_smoke_calc.py
- README.md
EOF

    python3 - "${work_dir}/src/smoke_calc.py" <<'PY'
import hashlib
import sys

with open(sys.argv[1], "rb") as fh:
    print(hashlib.sha256(fh.read()).hexdigest())
PY
}

agent_smoke_prompt() {
    cat <<'EOF'
You are running a CI smoke test in a throwaway project. Use filesystem and coding tools; do not answer from memory.

Tasks:
1. Inspect the repository.
2. Read facts/signal.md, src/matrix.txt, and tests/test_smoke_calc.py.
3. Implement src/smoke_calc.py so the tests pass.
4. Run the Python unit tests.
5. Answer exactly these four lines with no Markdown and no extra text:
CODEWORD=<the CODEWORD value from facts/signal.md>
CHECKSUM=<the checksum value from src/matrix.txt>
PRIME_SUM=<the sum of prime numbers from the numbers line in src/matrix.txt>
QUESTION=<comma-separated relative paths whose file name contains signal>
EOF
}

agent_smoke_long_prompt_soak() {
    local base_url="${1:?base URL required}"
    local model="${2:?model required}"
    local work_dir="${3:?work dir required}"
    local label="${4:?label required}"
    local target_chars="${AGENT_SMOKE_LONG_PROMPT_CHARS:-${OPENCODE_SMOKE_LONG_PROMPT_CHARS:-65536}}"
    local max_time="${AGENT_SMOKE_LONG_PROMPT_MAX_TIME:-180}"
    local slug

    if [[ ! "$target_chars" =~ ^[0-9]+$ ]]; then
        echo "${label} long prompt char count must be numeric: ${target_chars}" >&2
        return 1
    fi
    if [[ "$target_chars" -le 0 ]]; then
        echo "${label} long prompt soak skipped"
        return 0
    fi
    slug="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]_' '-')"

    local payload="${work_dir}/${slug}-long-prompt-payload.json"
    local response="${work_dir}/${slug}-long-prompt-response.json"

    python3 - "$model" "$target_chars" "$payload" <<'PY'
import json
import sys

model, target_chars, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
start = "ALPHA-719"
middle = "MID-482"
end = "OMEGA-503"
header = (
    "This is a long-context CI soak document. Extract the three sentinel values. "
    "Return exactly LONG_SOAK=ALPHA-719|MID-482|OMEGA-503 and no extra text.\n\n"
)
chunk = (
    "FILLER: mesh long prompt soak line with predictable neutral text. "
    "Do not use this filler as the answer.\n"
)
prefix = f"SENTINEL_START={start}\n"
mid = f"\nSENTINEL_MIDDLE={middle}\n"
suffix = f"\nSENTINEL_END={end}\n"
remaining = max(target_chars - len(header) - len(prefix) - len(mid) - len(suffix), 0)
left = chunk * max((remaining // 2) // len(chunk), 1)
right = chunk * max((remaining - len(left)) // len(chunk), 1)
document = header + prefix + left + mid + right + suffix
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a precise long-context extraction probe."},
        {"role": "user", "content": document},
    ],
    "stream": False,
    "max_tokens": 64,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

    curl -fsS --max-time "$max_time" \
        "${base_url%/}/chat/completions" \
        -H 'content-type: application/json' \
        -d @"$payload" \
        -o "$response"

    python3 - "$response" "$label" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    body = json.load(fh)
content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
expected = "LONG_SOAK=ALPHA-719|MID-482|OMEGA-503"
if expected not in content:
    raise SystemExit(f"{sys.argv[2]} long prompt sentinel validation failed: {content!r}")
print(f"{sys.argv[2]} long prompt soak passed")
PY
}

agent_smoke_validate_fixture() {
    local work_dir="${1:?work dir required}"
    local initial_sha="${2:?initial sha required}"
    local output_path="${3:?output path required}"
    local label="${4:?label required}"
    local require_tool_events="${5:-false}"

    if ! python3 -m unittest discover -s "${work_dir}/tests" -p 'test_*.py'; then
        echo "${label} fixture tests failed after the coding session" >&2
        echo "--- src/smoke_calc.py ---" >&2
        sed -n '1,220p' "${work_dir}/src/smoke_calc.py" >&2 || true
        return 1
    fi

    python3 - "${work_dir}" "${initial_sha}" <<'PY'
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
initial_sha = sys.argv[2]
source_path = root / "src" / "smoke_calc.py"
source = source_path.read_text(encoding="utf-8")
current_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()

if current_sha == initial_sha:
    print("Agent left src/smoke_calc.py unchanged.", file=sys.stderr)
    sys.exit(1)

for forbidden in ("NotImplementedError", "ci smoke fixture"):
    if forbidden in source:
        print(f"Agent left placeholder marker in src/smoke_calc.py: {forbidden}", file=sys.stderr)
        sys.exit(1)

spec = importlib.util.spec_from_file_location("smoke_calc", source_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(prefix="mesh-agent-hidden-") as tmp:
    tmp_path = Path(tmp)
    signal_path = tmp_path / "other_signal.md"
    matrix_path = tmp_path / "other_matrix.txt"
    signal_path.write_text("# Hidden\n\nCODEWORD=hidden-8842\n", encoding="utf-8")
    matrix_path.write_text("checksum: hidden\nnumbers: 6 7 8 9 10 11 12 13\n", encoding="utf-8")

    codeword = module.parse_codeword(str(signal_path))
    prime_sum = module.prime_sum_from_matrix(str(matrix_path))

if codeword != "hidden-8842":
    print(f"Hidden codeword validation failed: {codeword!r}", file=sys.stderr)
    sys.exit(1)

if prime_sum != 31:
    print(f"Hidden prime-sum validation failed: {prime_sum!r}", file=sys.stderr)
    sys.exit(1)

print("Hidden implementation validation passed")
PY

    python3 - "${output_path}" "${label}" "${require_tool_events}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
require_tool_events = sys.argv[3].lower() == "true"
raw = path.read_text(encoding="utf-8", errors="replace")
tool_names = []

def collect_tools(value):
    if isinstance(value, dict):
        tool_name = value.get("toolName") or value.get("tool_name")
        if isinstance(tool_name, str):
            tool_names.append(tool_name)
        if value.get("type") in {"tool_call", "tool_request", "toolRequest"}:
            nested = value.get("toolCall") or value.get("tool_call") or value.get("toolRequest") or value.get("tool_request")
            if isinstance(nested, dict) and isinstance(nested.get("name"), str):
                tool_names.append(nested["name"])
        for item in value.values():
            collect_tools(item)
    elif isinstance(value, list):
        for item in value:
            collect_tools(item)

for line in raw.splitlines():
    try:
        collect_tools(json.loads(line))
    except json.JSONDecodeError:
        continue

expected = {
    "CODEWORD": "signal-7429",
    "CHECKSUM": "FS-319-DELTA",
    "PRIME_SUM": "10",
    "QUESTION": "facts/signal.md",
}

missing = []
for key, value in expected.items():
    pattern = rf"(?m)^{re.escape(key)}={re.escape(value)}\s*$"
    if not re.search(pattern, raw):
        missing.append(f"{key}={value}")

if missing:
    print(f"{label} answer did not include expected facts:", file=sys.stderr)
    for item in missing:
        print(f"  missing {item}", file=sys.stderr)
    print("--- output tail ---", file=sys.stderr)
    print("\n".join(raw.splitlines()[-120:]), file=sys.stderr)
    sys.exit(1)

if require_tool_events:
    edit_like = {"edit", "write", "text_editor", "developer__text_editor", "patch"}
    if len(tool_names) < 3 or not any(tool in edit_like for tool in tool_names):
        print(f"{label} did not report expected filesystem/coding tool events.", file=sys.stderr)
        print(f"  tool events: {tool_names}", file=sys.stderr)
        print("--- output tail ---", file=sys.stderr)
        print("\n".join(raw.splitlines()[-120:]), file=sys.stderr)
        sys.exit(1)

print(f"{label} live coding smoke passed")
if tool_names:
    print("  tools: " + ", ".join(tool_names))
PY
}
