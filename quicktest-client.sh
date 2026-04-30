#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/target/release/mesh-llm"

# Optional overrides:
#   QUICKTEST_JOIN_TOKEN      Explicit join token (highest priority)
#   QUICKTEST_TOKEN_FILE      Cached token path (default: ~/.mesh-llm/quicktest-client.token)
#   QUICKTEST_DISCOVER_WAIT   LAN discover window in seconds (default: 6)
#   QUICKTEST_RESTART_DELAY   Delay between restarts in seconds (default: 3)
#   QUICKTEST_OFFLINE         1/0, default 1
#   QUICKTEST_HEADLESS        1/0, default 0
#   QUICKTEST_LOG_FORMAT      pretty/json, default pretty
TOKEN_FILE="${QUICKTEST_TOKEN_FILE:-$HOME/.mesh-llm/quicktest-client.token}"
DISCOVER_WAIT="${QUICKTEST_DISCOVER_WAIT:-6}"
RESTART_DELAY="${QUICKTEST_RESTART_DELAY:-3}"
OFFLINE="${QUICKTEST_OFFLINE:-1}"
HEADLESS="${QUICKTEST_HEADLESS:-0}"
LOG_FORMAT="${QUICKTEST_LOG_FORMAT:-pretty}"

stop_requested=0
runtime_pid=""

on_signal() {
	stop_requested=1
	if [[ -n "$runtime_pid" ]]; then
		kill -TERM "$runtime_pid" 2>/dev/null || true
	fi
}

trap on_signal INT TERM

if [[ ! -x "$BIN" ]]; then
	echo "mesh-llm binary not found at: $BIN" >&2
	echo "Build first with: just build" >&2
	exit 1
fi

mkdir -p "$(dirname "$TOKEN_FILE")"

discover_lan_token() {
	local out token
	out="$($BIN discover --lan --auto --wait-secs "$DISCOVER_WAIT" 2>/dev/null || true)"

	token="$(printf '%s\n' "$out" | sed -n 's/^[[:space:]]*token:[[:space:]]*//p' | head -n1)"
	if [[ -z "$token" ]]; then
		token="$(printf '%s\n' "$out" | awk '{for (i = 1; i <= NF; i++) { if ($i ~ /^eyJ/) { print $i; exit } }}')"
	fi

	# Ignore shortened preview tokens from non-auto discovery output.
	if [[ "$token" == *"..."* ]]; then
		token=""
	fi

	printf '%s' "$token"
}

load_token() {
	if [[ -n "${QUICKTEST_JOIN_TOKEN:-}" ]]; then
		printf '%s' "$QUICKTEST_JOIN_TOKEN"
		return
	fi

	if [[ -s "$TOKEN_FILE" ]]; then
		tr -d '[:space:]' < "$TOKEN_FILE"
		return
	fi

	printf ''
}

save_token() {
	local token="$1"
	printf '%s\n' "$token" > "$TOKEN_FILE"
}

run_client() {
	local token="$1"
	local args=()

	args+=(serve --client --listen-all --log-format "$LOG_FORMAT")
	if [[ "$OFFLINE" == "1" ]]; then
		args+=(--offline)
	fi
	if [[ "$HEADLESS" == "1" ]]; then
		args+=(--headless)
	fi
	args+=(--join "$token")

	echo "[quicktest-client] starting gateway client"
	echo "[quicktest-client] token source: ${TOKEN_FILE}"
	"$BIN" "${args[@]}" &
	runtime_pid=$!
	wait "$runtime_pid"
	local code=$?
	runtime_pid=""
	return "$code"
}

echo "[quicktest-client] binary: $BIN"
echo "[quicktest-client] offline: $OFFLINE, headless: $HEADLESS, log-format: $LOG_FORMAT"
echo "[quicktest-client] token file: $TOKEN_FILE"

while [[ "$stop_requested" -eq 0 ]]; do
	token="$(load_token)"

	if [[ -z "$token" ]]; then
		echo "[quicktest-client] no cached token, discovering LAN mesh..."
		token="$(discover_lan_token)"
		if [[ -n "$token" ]]; then
			save_token "$token"
			echo "[quicktest-client] discovered and cached token"
		else
			echo "[quicktest-client] no LAN token found, retrying in ${RESTART_DELAY}s"
			sleep "$RESTART_DELAY"
			continue
		fi
	fi

	set +e
	run_client "$token"
	exit_code=$?
	set -e

	if [[ "$stop_requested" -eq 1 ]]; then
		break
	fi

	echo "[quicktest-client] mesh-llm exited with code $exit_code"

	# If a join token is stale/unreachable, force rediscovery on next loop.
	if [[ -z "${QUICKTEST_JOIN_TOKEN:-}" ]]; then
		: > "$TOKEN_FILE"
		echo "[quicktest-client] cleared cached token to force LAN rediscovery"
	fi

	echo "[quicktest-client] restarting in ${RESTART_DELAY}s"
	sleep "$RESTART_DELAY"
done

echo "[quicktest-client] shutdown complete"
