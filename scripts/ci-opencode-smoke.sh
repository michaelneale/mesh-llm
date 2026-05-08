#!/usr/bin/env bash
# ci-opencode-smoke.sh - exercise OpenCode's filesystem tools with a coding model.

set -euo pipefail

if [[ -n "${MESH_OPENCODE_BASE_URL:-}" ]]; then
    MESH_BASE_URL="$MESH_OPENCODE_BASE_URL"
elif [[ -n "${MESH_CLIENT_API_BASE:-}" ]]; then
    MESH_BASE_URL="${MESH_CLIENT_API_BASE%/}/v1"
else
    MESH_BASE_URL="http://127.0.0.1:9337/v1"
fi
MESH_MODEL="${MESH_OPENCODE_MODEL:-${MESH_SDK_MODEL_ID:-}}"
MODEL="${OPENCODE_SMOKE_MODEL:-}"
TIMEOUT_SECONDS="${OPENCODE_SMOKE_TIMEOUT:-300}"
WORK_DIR="${OPENCODE_SMOKE_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-opencode-smoke.XXXXXX")}"
OUTPUT_JSONL="${OPENCODE_SMOKE_OUTPUT:-${WORK_DIR}/opencode-output.jsonl}"
TURN1_JSONL="${OPENCODE_SMOKE_TURN1_OUTPUT:-${WORK_DIR}/opencode-turn1.jsonl}"
TURN2_JSONL="${OPENCODE_SMOKE_TURN2_OUTPUT:-${WORK_DIR}/opencode-turn2.jsonl}"
ERROR_LOG="${OPENCODE_SMOKE_ERROR_LOG:-${WORK_DIR}/opencode-stderr.log}"
SURFACE_LOG="${OPENCODE_SMOKE_SURFACE_LOG:-${WORK_DIR}/openai-surface.jsonl}"
SURFACE_PROXY_LOG="${OPENCODE_SMOKE_SURFACE_PROXY_LOG:-${WORK_DIR}/openai-surface-proxy.log}"
SURFACE_CAPTURE="${OPENCODE_SMOKE_CAPTURE_SURFACE:-true}"
LONG_PROMPT_CHARS="${OPENCODE_SMOKE_LONG_PROMPT_CHARS:-65536}"

if ! command -v opencode >/dev/null 2>&1; then
    echo "opencode is not installed or is not on PATH" >&2
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    MODELS_JSON="$(curl -sf "${MESH_BASE_URL%/}/models" 2>/dev/null || true)"
    if [[ -z "$MODELS_JSON" ]]; then
        echo "::notice::Skipping OpenCode smoke because mesh endpoint is not reachable at ${MESH_BASE_URL%/}/models."
        echo "::notice::Start mesh-llm or set MESH_OPENCODE_BASE_URL to an OpenAI-compatible mesh /v1 endpoint."
        exit 0
    fi

    if [[ -z "$MESH_MODEL" ]]; then
        MESH_MODEL="$(
            printf '%s' "$MODELS_JSON" | python3 -c 'import json,sys
data=json.load(sys.stdin).get("data", [])
preferred=("minimax", "glm", "qwen", "coder", "hermes")
ids=[item.get("id","") for item in data if item.get("id")]
for needle in preferred:
    for model_id in ids:
        if needle in model_id.lower():
            print(model_id)
            raise SystemExit
print(ids[0] if ids else "")' 2>/dev/null || echo ""
        )"
    fi

    if [[ -z "$MESH_MODEL" ]]; then
        echo "Mesh endpoint returned no models from ${MESH_BASE_URL%/}/models" >&2
        exit 1
    fi

    MODEL="mesh/${MESH_MODEL}"
fi

CONFIG_BASE_URL="$MESH_BASE_URL"
SURFACE_PROXY_PID=""
cleanup_surface_proxy() {
    if [[ -n "$SURFACE_PROXY_PID" ]]; then
        kill "$SURFACE_PROXY_PID" 2>/dev/null || true
        wait "$SURFACE_PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup_surface_proxy EXIT

if [[ "$SURFACE_CAPTURE" == "true" && "$MODEL" == mesh/* ]]; then
    SURFACE_READY="${WORK_DIR}/openai-surface-proxy.ready"
    python3 -u - "$MESH_BASE_URL" "$SURFACE_LOG" "$SURFACE_READY" >"$SURFACE_PROXY_LOG" 2>&1 <<'PY' &
import http.server
import json
import socketserver
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

upstream = sys.argv[1].rstrip("/")
upstream_path = urlsplit(upstream).path.rstrip("/")
log_path = sys.argv[2]
ready_path = Path(sys.argv[3])

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        forward_path = self.path
        if upstream_path and forward_path.startswith(upstream_path + "/"):
            forward_path = forward_path[len(upstream_path):]
        target = upstream + forward_path
        parsed_body = None
        if body:
            try:
                parsed_body = json.loads(body)
            except Exception:
                parsed_body = None

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "method": self.command,
                "path": self.path,
                "body": parsed_body,
                "headers": {
                    "content-type": self.headers.get("content-type"),
                    "accept": self.headers.get("accept"),
                },
            }) + "\n")

        headers = {
            key: value for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding", "connection"}
        }
        request = urllib.request.Request(target, data=body if self.command == "POST" else None, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                data = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in {"content-length", "connection", "transfer-encoding", "content-encoding"}:
                        continue
                    self.send_header(key, value)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as err:
            data = err.read()
            self.send_response(err.code)
            for key, value in err.headers.items():
                if key.lower() in {"content-length", "connection", "transfer-encoding", "content-encoding"}:
                    continue
                self.send_header(key, value)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

server = Server(("127.0.0.1", 0), Handler)
ready_path.write_text(f"http://127.0.0.1:{server.server_port}/v1", encoding="utf-8")
print(f"http://127.0.0.1:{server.server_port}/v1", flush=True)
server.serve_forever()
PY
    SURFACE_PROXY_PID=$!

    for _ in $(seq 1 100); do
        if [[ -s "$SURFACE_READY" ]]; then
            CONFIG_BASE_URL="$(cat "$SURFACE_READY")"
            break
        fi
        if ! kill -0 "$SURFACE_PROXY_PID" 2>/dev/null; then
            echo "OpenAI surface capture proxy exited unexpectedly" >&2
            cat "$SURFACE_PROXY_LOG" >&2 || true
            exit 1
        fi
        sleep 0.1
    done

    if [[ "$CONFIG_BASE_URL" == "$MESH_BASE_URL" ]]; then
        echo "Timed out starting OpenAI surface capture proxy" >&2
        cat "$SURFACE_PROXY_LOG" >&2 || true
        exit 1
    fi
fi

if [[ -z "${OPENCODE_CONFIG_CONTENT:-}" ]]; then
    if [[ "$MODEL" == mesh/* ]]; then
        export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
        OPENCODE_CONFIG_CONTENT="$(
            python3 - "$CONFIG_BASE_URL" "$MESH_MODEL" <<'PY'
import json
import sys

base_url, model = sys.argv[1:3]
print(json.dumps({
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "mesh": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "mesh-llm",
            "options": {
                "baseURL": base_url.rstrip("/"),
            },
            "models": {
                model: {
                    "name": model,
                    "limit": {
                        "context": 32768,
                        "output": 4096,
                    },
                },
            },
        },
    },
    "permission": {
        "bash": "allow",
        "read": "allow",
        "grep": "allow",
        "glob": "allow",
        "edit": "allow",
        "webfetch": "deny",
        "websearch": "deny",
        "question": "deny",
        "todowrite": "deny",
    },
}))
PY
        )"
        export OPENCODE_CONFIG_CONTENT
    else
        export OPENCODE_CONFIG_CONTENT='{
          "$schema": "https://opencode.ai/config.json",
          "permission": {
            "bash": "allow",
            "read": "allow",
            "grep": "allow",
            "glob": "allow",
            "edit": "allow",
            "webfetch": "deny",
            "websearch": "deny",
            "question": "deny",
            "todowrite": "deny"
          }
        }'
    fi
fi

export OPENCODE_DISABLE_AUTOUPDATE="${OPENCODE_DISABLE_AUTOUPDATE:-true}"
export OPENCODE_DISABLE_PRUNE="${OPENCODE_DISABLE_PRUNE:-true}"
export OPENCODE_DISABLE_LSP_DOWNLOAD="${OPENCODE_DISABLE_LSP_DOWNLOAD:-true}"

mkdir -p "${WORK_DIR}/facts" "${WORK_DIR}/src" "${WORK_DIR}/notes" "${WORK_DIR}/tests"

cat >"${WORK_DIR}/README.md" <<'EOF'
# OpenCode Smoke Fixture

This repository is intentionally tiny. The smoke test should answer only from
files on disk, not from the prompt text.
EOF

cat >"${WORK_DIR}/facts/signal.md" <<'EOF'
# Runtime Signal

CODEWORD=signal-7429

Question seed:
Which file names contain the word signal?
EOF

cat >"${WORK_DIR}/src/matrix.txt" <<'EOF'
checksum: FS-319-DELTA
numbers: 2 3 4 5
hint: the prime sum is computed from the numbers line
EOF

cat >"${WORK_DIR}/src/smoke_calc.py" <<'EOF'
from __future__ import annotations


def parse_codeword(path: str) -> str:
    """Return the CODEWORD value from a small key/value markdown file."""
    raise NotImplementedError("ci smoke fixture")


def prime_sum_from_matrix(path: str) -> int:
    """Return the sum of prime numbers from the numbers line in matrix.txt."""
    raise NotImplementedError("ci smoke fixture")
EOF
INITIAL_IMPL_SHA="$(
    python3 - "${WORK_DIR}/src/smoke_calc.py" <<'PY'
import hashlib
import sys

with open(sys.argv[1], "rb") as fh:
    print(hashlib.sha256(fh.read()).hexdigest())
PY
)"

cat >"${WORK_DIR}/tests/test_smoke_calc.py" <<'EOF'
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

cat >"${WORK_DIR}/notes/manifest.txt" <<'EOF'
tracked files:
- facts/signal.md
- src/matrix.txt
- src/smoke_calc.py
- tests/test_smoke_calc.py
- README.md
EOF

TURN1_PROMPT='You are running turn 1 of a CI smoke test in a throwaway project. Use filesystem tools, inspect the repository, read the tests, and implement src/smoke_calc.py so the tests are intended to pass. This is a real coding task: edit the file, do not just describe the edit. Keep the implementation small and dependency-free. End your response with TURN1_DONE.'

TURN2_PROMPT='This is turn 2 of the same CI smoke test. Continue from the prior work. Run the Python tests, fix src/smoke_calc.py if anything fails, then answer exactly these four lines with no Markdown and no extra text:
CODEWORD=<the CODEWORD value from facts/signal.md>
CHECKSUM=<the checksum value from src/matrix.txt>
PRIME_SUM=<the sum of prime numbers from the numbers line in src/matrix.txt>
QUESTION=<comma-separated relative paths whose file name contains signal>'

echo "=== CI OpenCode Smoke Test ==="
echo "  model:      ${MODEL}"
if [[ "$MODEL" == mesh/* ]]; then
    echo "  mesh:       ${MESH_BASE_URL%/}"
    if [[ "$SURFACE_CAPTURE" == "true" ]]; then
        echo "  capture:    ${CONFIG_BASE_URL%/}"
    fi
fi
echo "  opencode:   $(opencode --version 2>/dev/null || echo unknown)"
echo "  work dir:   ${WORK_DIR}"
echo "  output:     ${OUTPUT_JSONL}"

if [[ "$SURFACE_CAPTURE" == "true" && "$MODEL" == mesh/* ]]; then
    curl -sf "${CONFIG_BASE_URL%/}/models" >/dev/null
    SURFACE_PROBE_PAYLOAD="${WORK_DIR}/openai-surface-probe.json"
    SURFACE_PROBE_RESPONSE="${WORK_DIR}/openai-surface-probe-response.json"
    python3 - "$MESH_MODEL" "$SURFACE_PROBE_PAYLOAD" <<'PY'
import json
import sys

model, path = sys.argv[1:3]
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a brief CI compatibility probe."},
        {"role": "user", "content": "Reply with ok, or call the tool if needed."},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_fixture_fact",
                "description": "Return one known fact from the smoke fixture.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "enum": ["codeword", "checksum"]},
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "stream": False,
    "max_tokens": 8,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY
    curl -fsS --max-time 120 \
        "${CONFIG_BASE_URL%/}/chat/completions" \
        -H 'content-type: application/json' \
        -d @"$SURFACE_PROBE_PAYLOAD" \
        -o "$SURFACE_PROBE_RESPONSE"
    python3 - "$SURFACE_PROBE_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    body = json.load(fh)
if body.get("object") != "chat.completion":
    raise SystemExit(f"unexpected probe response object: {body.get('object')!r}")
if not body.get("choices"):
    raise SystemExit("probe response had no choices")
PY

    if [[ "$LONG_PROMPT_CHARS" -gt 0 ]]; then
        LONG_PROMPT_PAYLOAD="${WORK_DIR}/openai-long-prompt-probe.json"
        LONG_PROMPT_RESPONSE="${WORK_DIR}/openai-long-prompt-probe-response.json"
        python3 - "$MESH_MODEL" "$LONG_PROMPT_CHARS" "$LONG_PROMPT_PAYLOAD" <<'PY'
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
        curl -fsS --max-time 180 \
            "${CONFIG_BASE_URL%/}/chat/completions" \
            -H 'content-type: application/json' \
            -d @"$LONG_PROMPT_PAYLOAD" \
            -o "$LONG_PROMPT_RESPONSE"
        python3 - "$LONG_PROMPT_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    body = json.load(fh)
content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
expected = "LONG_SOAK=ALPHA-719|MID-482|OMEGA-503"
if expected not in content:
    raise SystemExit(f"long prompt sentinel validation failed: {content!r}")
print("Long prompt soak passed")
PY
    fi
fi

BASE_RUN_ARGS=(run --format json --model "${MODEL}" --dir "${WORK_DIR}")
if [[ -n "${OPENCODE_SMOKE_VARIANT:-}" ]]; then
    BASE_RUN_ARGS+=(--variant "${OPENCODE_SMOKE_VARIANT}")
fi

if command -v timeout >/dev/null 2>&1; then
    OPENCODE_COMMAND=(timeout "${TIMEOUT_SECONDS}" opencode)
else
    OPENCODE_COMMAND=(opencode)
fi

if ! "${OPENCODE_COMMAND[@]}" "${BASE_RUN_ARGS[@]}" "${TURN1_PROMPT}" >"${TURN1_JSONL}" 2>"${ERROR_LOG}"; then
    echo "OpenCode smoke turn 1 failed" >&2
    echo "--- opencode stderr ---" >&2
    tail -120 "${ERROR_LOG}" >&2 || true
    echo "--- opencode json output ---" >&2
    tail -120 "${TURN1_JSONL}" >&2 || true
    exit 1
fi

SESSION_ID="$(
    python3 - "${TURN1_JSONL}" <<'PY'
import json
import sys

for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    session = event.get("sessionID")
    if isinstance(session, str) and session:
        print(session)
        raise SystemExit
print("")
PY
)"

if [[ -z "$SESSION_ID" ]]; then
    echo "OpenCode turn 1 did not emit a sessionID" >&2
    tail -120 "${TURN1_JSONL}" >&2 || true
    exit 1
fi

if ! "${OPENCODE_COMMAND[@]}" "${BASE_RUN_ARGS[@]}" --session "$SESSION_ID" "${TURN2_PROMPT}" >"${TURN2_JSONL}" 2>>"${ERROR_LOG}"; then
    echo "OpenCode smoke turn 2 failed" >&2
    echo "--- opencode stderr ---" >&2
    tail -120 "${ERROR_LOG}" >&2 || true
    echo "--- opencode json output ---" >&2
    tail -120 "${TURN2_JSONL}" >&2 || true
    exit 1
fi

cat "${TURN1_JSONL}" "${TURN2_JSONL}" >"${OUTPUT_JSONL}"

if ! python3 -m unittest discover -s "${WORK_DIR}/tests" -p 'test_*.py'; then
    echo "OpenCode smoke fixture tests failed after the coding session" >&2
    echo "--- src/smoke_calc.py ---" >&2
    sed -n '1,220p' "${WORK_DIR}/src/smoke_calc.py" >&2 || true
    exit 1
fi

python3 - "${WORK_DIR}" "${INITIAL_IMPL_SHA}" <<'PY'
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
    print("OpenCode left src/smoke_calc.py unchanged.", file=sys.stderr)
    sys.exit(1)

for forbidden in ("NotImplementedError", "ci smoke fixture"):
    if forbidden in source:
        print(f"OpenCode left placeholder marker in src/smoke_calc.py: {forbidden}", file=sys.stderr)
        sys.exit(1)

spec = importlib.util.spec_from_file_location("smoke_calc", source_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(prefix="mesh-opencode-hidden-") as tmp:
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

python3 - "${OUTPUT_JSONL}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text(encoding="utf-8", errors="replace")
tool_names = []
text_chunks = []

def text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from text_values(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"text", "content", "message", "delta"}:
                yield from text_values(item)
            elif isinstance(item, (dict, list)):
                yield from text_values(item)

for line in raw.splitlines():
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        text_chunks.append(line)
        continue

    if event.get("type") == "tool_use":
        part = event.get("part") or {}
        tool = part.get("tool") or part.get("name")
        if isinstance(tool, str):
            tool_names.append(tool)
    if event.get("type") in {"text", "message", "assistant"}:
        text_chunks.extend(text_values(event))

answer_text = "\n".join(text_chunks) + "\n" + raw
expected = {
    "CODEWORD": "signal-7429",
    "CHECKSUM": "FS-319-DELTA",
    "PRIME_SUM": "10",
    "QUESTION": "facts/signal.md",
}

missing = []
for key, value in expected.items():
    pattern = rf"(?m)^{re.escape(key)}={re.escape(value)}\s*$"
    if not re.search(pattern, answer_text):
        missing.append(f"{key}={value}")

filesystem_tools = {"bash", "read", "grep", "glob", "edit", "write", "apply_patch"}
used_filesystem_tools = [tool for tool in tool_names if tool in filesystem_tools]
edit_tools = [tool for tool in tool_names if tool in {"edit", "write", "apply_patch"}]

if missing:
    print("OpenCode answer did not include expected facts:", file=sys.stderr)
    for item in missing:
        print(f"  missing {item}", file=sys.stderr)
    print("--- output tail ---", file=sys.stderr)
    print("\n".join(raw.splitlines()[-80:]), file=sys.stderr)
    sys.exit(1)

if len(tool_names) < 4 or len(used_filesystem_tools) < 4 or not edit_tools:
    print("OpenCode did not report the expected multi-turn filesystem/coding tool calls.", file=sys.stderr)
    print(f"  tool events: {tool_names}", file=sys.stderr)
    print("--- output tail ---", file=sys.stderr)
    print("\n".join(raw.splitlines()[-80:]), file=sys.stderr)
    sys.exit(1)

print("OpenCode multi-turn coding smoke passed")
print("  tools: " + ", ".join(tool_names))
PY

if [[ "$SURFACE_CAPTURE" == "true" && "$MODEL" == mesh/* ]]; then
    python3 - "$SURFACE_LOG" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
long_prompt_chars = int(os.environ.get("OPENCODE_SMOKE_LONG_PROMPT_CHARS", "65536") or "0")
events = []
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass

models_gets = [event for event in events if event.get("method") == "GET" and event.get("path", "").rstrip("/") == "/v1/models"]
chat_posts = [
    event for event in events
    if event.get("method") == "POST" and event.get("path", "").split("?", 1)[0].rstrip("/") == "/v1/chat/completions"
]
bodies = [event.get("body") or {} for event in chat_posts]

def message_roles(body):
    return [message.get("role") for message in body.get("messages", []) if isinstance(message, dict)]

def has_assistant_tool_calls(body):
    for message in body.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls"):
            return True
    return False

def has_tool_result(body):
    return any(isinstance(message, dict) and message.get("role") == "tool" for message in body.get("messages", []))

required = {
    "GET /v1/models": bool(models_gets),
    "POST /v1/chat/completions": bool(chat_posts),
    "streaming chat request": any(body.get("stream") is True for body in bodies),
    "non-stream chat request": any(body.get("stream") is False for body in bodies),
    "tools schema": any(isinstance(body.get("tools"), list) and body["tools"] for body in bodies),
    "tool_choice field": any("tool_choice" in body for body in bodies),
    "parallel_tool_calls field": any("parallel_tool_calls" in body for body in bodies),
    "system/developer instructions": any(any(role in {"system", "developer"} for role in message_roles(body)) for body in bodies),
    "user message": any("user" in message_roles(body) for body in bodies),
    "assistant tool-call history": any(has_assistant_tool_calls(body) for body in bodies),
    "tool-result message": any(has_tool_result(body) for body in bodies),
    "multi-message history": any(len(body.get("messages", [])) >= 4 for body in bodies),
}
if long_prompt_chars > 0:
    required["long prompt request"] = any(
        len(json.dumps(body, separators=(",", ":"))) >= long_prompt_chars
        for body in bodies
    )

missing = [name for name, ok in required.items() if not ok]
if missing:
    print("OpenAI agent surface validation failed.", file=sys.stderr)
    for name in missing:
        print(f"  missing {name}", file=sys.stderr)
    print(f"  captured events: {len(events)}", file=sys.stderr)
    for body in bodies:
        print(json.dumps({
            "model": body.get("model"),
            "stream": body.get("stream"),
            "tool_count": len(body.get("tools", []) or []),
            "has_tool_choice": "tool_choice" in body,
            "has_parallel_tool_calls": "parallel_tool_calls" in body,
            "roles": message_roles(body),
            "messages": len(body.get("messages", [])),
            "assistant_tool_calls": has_assistant_tool_calls(body),
            "tool_result": has_tool_result(body),
        }), file=sys.stderr)
    sys.exit(1)

print("OpenAI agent surface validation passed")
print(f"  captured requests: models={len(models_gets)} chat={len(chat_posts)}")
PY
fi
