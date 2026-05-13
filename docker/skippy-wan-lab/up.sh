#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_DIR="${ROOT}/docker/skippy-wan-lab"
ENV_FILE="${ENV_FILE:-${LAB_DIR}/.env}"
LINK_ENV_FILE="${LINK_ENV_FILE:-${LAB_DIR}/.env.link}"

log() {
  printf '[skippy-wan-lab-up] %s\n' "$*" >&2
}

load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

parse_hf_package_ref() {
  local ref="$1"
  python3 - "$ref" <<'PY'
import sys

value = sys.argv[1]
if not value.startswith("hf://"):
    raise SystemExit("MODEL_PACKAGE_REF must start with hf://")
rest = value[len("hf://"):]
revision = "main"
if "@" in rest:
    rest, revision = rest.rsplit("@", 1)
elif ":" in rest:
    rest, revision = rest.rsplit(":", 1)
if "/" not in rest or not rest or not revision:
    raise SystemExit(f"invalid MODEL_PACKAGE_REF: {value}")
print(rest)
print(revision)
PY
}

hf_home_dir() {
  if [[ -n "${HF_HOME:-}" ]]; then
    printf '%s\n' "$HF_HOME"
  else
    printf '%s\n' "${HOME}/.cache/huggingface"
  fi
}

hf_snapshot_dir() {
  local repo="$1"
  local revision="$2"
  local hf_home="$3"
  local repo_dir="${hf_home}/hub/models--${repo//\//--}"
  local commit=""

  if [[ -f "${repo_dir}/refs/${revision}" ]]; then
    commit="$(cat "${repo_dir}/refs/${revision}")"
  elif [[ -d "${repo_dir}/snapshots/${revision}" ]]; then
    commit="$revision"
  fi

  if [[ -n "$commit" && -d "${repo_dir}/snapshots/${commit}" ]]; then
    printf '%s\n' "${repo_dir}/snapshots/${commit}"
    return 0
  fi

  return 1
}

verify_package_cache() {
  local snapshot="$1"
  python3 - "$snapshot" <<'PY'
import json
import os
import sys

snapshot = sys.argv[1]
manifest_path = os.path.join(snapshot, "model-package.json")
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

expected_layers = int(os.environ.get("LAYER_COUNT") or manifest["layer_count"])
expected_width = int(os.environ.get("ACTIVATION_WIDTH") or manifest["activation_width"])
if manifest["layer_count"] != expected_layers:
    raise SystemExit(f"manifest layer_count {manifest['layer_count']} != expected {expected_layers}")
if manifest.get("activation_width") != expected_width:
    raise SystemExit(f"manifest activation_width {manifest.get('activation_width')} != expected {expected_width}")

required = [
    manifest["shared"]["metadata"],
    manifest["shared"]["embeddings"],
    manifest["shared"]["output"],
    *manifest["layers"],
    *manifest.get("projectors", []),
]

missing = []
for artifact in required:
    path = os.path.join(snapshot, artifact["path"])
    if not os.path.exists(path):
        missing.append(artifact["path"])
        continue
    size = os.path.getsize(path)
    expected = int(artifact["artifact_bytes"])
    if size != expected:
        missing.append(f"{artifact['path']} (expected {expected} bytes, got {size})")

if missing:
    raise SystemExit("missing or incomplete cached package artifacts:\n" + "\n".join(missing))

print(json.dumps({
    "model_id": manifest["model_id"],
    "layer_count": manifest["layer_count"],
    "activation_width": manifest.get("activation_width"),
    "artifact_count": len(required),
}))
PY
}

ensure_hf_package() {
  local package_ref="${MODEL_PACKAGE_REF:-hf://meshllm/gemma-4-26B-A4B-it-UD-Q4_K_M-layers}"
  mapfile -t parsed < <(parse_hf_package_ref "$package_ref")
  local repo="${parsed[0]}"
  local revision="${parsed[1]}"
  local hf_home
  hf_home="$(hf_home_dir)"

  if ! command -v hf >/dev/null 2>&1; then
    log "the Hugging Face CLI 'hf' is required; install huggingface_hub first"
    log "  python3 -m pip install -U huggingface_hub"
    exit 69
  fi

  log "ensuring HF package is cached on host: ${repo}@${revision}"
  hf download "$repo" \
    --revision "$revision" \
    --include model-package.json \
    --include 'shared/*' \
    --include 'layers/*' \
    --include 'projectors/*' \
    >/dev/null

  local snapshot
  if ! snapshot="$(hf_snapshot_dir "$repo" "$revision" "$hf_home")"; then
    log "could not find ${repo}@${revision} under host HF cache ${hf_home}"
    log "check HF_HOME or run 'hf cache list --filter ${repo}'"
    exit 67
  fi
  log "verifying host HF cache snapshot: ${snapshot}"
  verify_package_cache "$snapshot" >&2

  export HF_CACHE_MOUNT="$hf_home"
}

if [[ ! -f "$ENV_FILE" ]]; then
  log "creating ${ENV_FILE} from .env.example"
  cp "${LAB_DIR}/.env.example" "$ENV_FILE"
fi

load_env_file "$ENV_FILE"
load_env_file "$LINK_ENV_FILE"

ensure_hf_package

compose_args=(
  --env-file "$ENV_FILE"
)
if [[ -f "$LINK_ENV_FILE" ]]; then
  compose_args+=(--env-file "$LINK_ENV_FILE")
fi
compose_args+=(
  -f "${LAB_DIR}/docker-compose.yml"
)

log "starting Docker lab"
exec docker compose "${compose_args[@]}" up --build "$@"
