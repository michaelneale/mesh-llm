#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_DIR="${ROOT}/docker/skippy-wan-lab"
ENV_FILE="${ENV_FILE:-${LAB_DIR}/.env}"
LINK_ENV_FILE="${LINK_ENV_FILE:-${LAB_DIR}/.env.link}"

load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

hf_home_dir() {
  if [[ -n "${HF_HOME:-}" ]]; then
    printf '%s\n' "$HF_HOME"
  else
    printf '%s\n' "${HOME}/.cache/huggingface"
  fi
}

if [[ ! -f "$ENV_FILE" ]]; then
  printf '[skippy-wan-lab-prompt] %s does not exist; run docker/skippy-wan-lab/up.sh first\n' "$ENV_FILE" >&2
  exit 64
fi

load_env_file "$ENV_FILE"
load_env_file "$LINK_ENV_FILE"

export HF_CACHE_MOUNT="${HF_CACHE_MOUNT:-$(hf_home_dir)}"

compose_args=(
  --env-file "$ENV_FILE"
)
if [[ -f "$LINK_ENV_FILE" ]]; then
  compose_args+=(--env-file "$LINK_ENV_FILE")
fi
compose_args+=(
  -f "${LAB_DIR}/docker-compose.yml"
)

exec docker compose "${compose_args[@]}" exec stage0 skippy-wan-lab prompt "$@"
