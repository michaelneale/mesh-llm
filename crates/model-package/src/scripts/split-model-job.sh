#!/bin/bash
set -euo pipefail

# This script runs inside an HF Job container.
# It clones mesh-llm, builds the splitter, splits the model, validates, and publishes.
#
# Environment variables (set by mesh-llm model-package job spec):
#   SOURCE_REPO, SOURCE_FILE, TARGET_REPO, MODEL_ID, SOURCE_REVISION
#   MESH_LLM_REF — git ref to build from (default: main)
#   CATALOG_CREATE_PR — "true" to open a PR for catalog updates (non-org members)
#   HF_TOKEN — injected as a secret by HF Jobs
#
# Volumes:
#   /source  — source GGUF repo (read-only mount)
#   /bucket  — writable storage bucket for script + large package workspace

MESH_LLM_REF="${MESH_LLM_REF:-main}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Layer Package Split Job                                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Source: ${SOURCE_REPO}/${SOURCE_FILE}"
echo "║  Target: ${TARGET_REPO}"
echo "║  Model:  ${MODEL_ID}"
echo "║  Build:  mesh-llm @ ${MESH_LLM_REF}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Build tools ──────────────────────────────────────────────────────────
echo "=== [1/9] Installing build dependencies ==="
apt-get update -qq && apt-get install -y -qq \
    cmake git curl build-essential pkg-config libssl-dev \
    python3-pip python3-venv > /dev/null 2>&1

echo "=== [2/9] Installing Rust ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
source /root/.cargo/env

echo "=== [3/9] Cloning mesh-llm and building skippy-model-package ==="
git clone --filter=blob:none https://github.com/Mesh-LLM/mesh-llm.git /tmp/build
cd /tmp/build
if git ls-remote --exit-code --heads origin "$MESH_LLM_REF" >/dev/null 2>&1 || \
   git ls-remote --exit-code --tags origin "$MESH_LLM_REF" >/dev/null 2>&1; then
    git fetch --depth 1 origin "$MESH_LLM_REF"
    git checkout --detach FETCH_HEAD
elif git cat-file -e "$MESH_LLM_REF^{commit}" 2>/dev/null; then
    git checkout --detach "$MESH_LLM_REF"
else
    git fetch --depth 1 origin "$MESH_LLM_REF"
    git checkout --detach FETCH_HEAD
fi

# Full clone needed for git-am patches in prepare-llama
sed -i 's/--filter=blob:none //' scripts/prepare-llama.sh
echo "  Running prepare-llama.sh..."
scripts/prepare-llama.sh pinned 2>&1 | tail -5
echo "  Running build-llama.sh..."
scripts/build-llama.sh 2>&1 | tail -5

# Locate the llama.cpp build directory (build-llama.sh puts it here)
LLAMA_BUILD_DIR=".deps/llama-build/build-stage-abi-cpu"
echo "  Verifying llama.cpp build at $LLAMA_BUILD_DIR..."
find "$LLAMA_BUILD_DIR" -name "*.a" 2>/dev/null | head -10 || echo "  WARNING: no .a files found"

# Build the splitter binary
echo "  Building skippy-model-package..."
SKIPPY_LLAMA_BUILD_DIR="$LLAMA_BUILD_DIR" \
    cargo build --release -p skippy-model-package 2>&1 | tail -20
SLICER=/tmp/build/target/release/skippy-model-package
if [ ! -f "$SLICER" ]; then
    echo "ERROR: Build failed — binary not found at $SLICER"
    echo "Retrying with full output..."
    SKIPPY_LLAMA_BUILD_DIR=.deps/llama.cpp/build-stage-abi-static \
        cargo build --release -p skippy-model-package 2>&1
    exit 1
fi
echo "  ✓ Built: $SLICER"

# ─── Split ────────────────────────────────────────────────────────────────
echo ""
echo "=== [4/9] Splitting model ==="
SOURCE_PATH="/source/${SOURCE_FILE}"
JOB_WORK_ROOT="${JOB_WORK_ROOT:-/bucket/job-work}"
if [ -z "${JOB_WORK_DIR:-}" ]; then
    SAFE_TARGET_REPO="$(printf '%s' "$TARGET_REPO" | tr -c '[:alnum:]._-' '_')"
    JOB_WORK_DIR="${JOB_WORK_ROOT}/${SAFE_TARGET_REPO}-$(date +%Y%m%d%H%M%S)-$$"
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-true}"
else
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-false}"
fi
PACKAGE_DIR="${PACKAGE_DIR:-${JOB_WORK_DIR}/package}"
export JOB_WORK_DIR PACKAGE_DIR

cleanup_job_work_dir() {
    if [ "${CLEANUP_JOB_WORK_DIR}" = "true" ] && [ -n "${JOB_WORK_DIR:-}" ]; then
        echo "Cleaning job work dir: ${JOB_WORK_DIR}"
        rm -rf "$JOB_WORK_DIR"
    fi
}
trap cleanup_job_work_dir EXIT

mkdir -p "$PACKAGE_DIR"

if [ ! -f "$SOURCE_PATH" ]; then
    echo "ERROR: Source file not found at $SOURCE_PATH"
    echo ""
    echo "Available GGUF files in /source:"
    find /source -name "*.gguf" -type f | sort | head -30
    exit 1
fi

echo "  Source: $SOURCE_PATH ($(du -h "$SOURCE_PATH" | cut -f1))"
echo "  Package workspace: $PACKAGE_DIR"
time $SLICER write-package "$SOURCE_PATH" \
    --out-dir "$PACKAGE_DIR" \
    --model-id "$MODEL_ID" \
    --source-repo "$SOURCE_REPO" \
    --source-revision "${SOURCE_REVISION:-main}" \
    --source-file "$SOURCE_FILE"

LAYER_COUNT=$(ls "$PACKAGE_DIR"/layers/ | wc -l)
TOTAL_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
echo "  ✓ Split into $LAYER_COUNT layers ($TOTAL_SIZE total)"

# ─── Validate ─────────────────────────────────────────────────────────────
echo ""
echo "=== [5/9] Validating package ==="
time $SLICER validate-package "$SOURCE_PATH" "$PACKAGE_DIR"
echo "  ✓ Validation passed — all tensors accounted for"

# ─── Publish ──────────────────────────────────────────────────────────────
echo ""
echo "=== [6/9] Publishing to HuggingFace ==="
python3 -m venv /tmp/venv > /dev/null
/tmp/venv/bin/pip install -q huggingface_hub

/tmp/venv/bin/python3 << PYTHON
from huggingface_hub import HfApi
import os, json

api = HfApi(token=os.environ['HF_TOKEN'])
target_repo = os.environ['TARGET_REPO']
source_repo = os.environ['SOURCE_REPO']
model_id = os.environ.get('MODEL_ID', '')

# Create repo (idempotent)
api.create_repo(target_repo, exist_ok=True)

# Upload the entire package
api.upload_folder(
    repo_id=target_repo,
    folder_path=os.environ['PACKAGE_DIR'],
    commit_message=f'Layer package from {source_repo} ({model_id})',
)

# Print summary
manifest = json.load(open(os.path.join(os.environ['PACKAGE_DIR'], 'model-package.json')))
print(f'  ✓ Published: https://huggingface.co/{target_repo}')
print(f'    Model:  {manifest["model_id"]}')
print(f'    Layers: {manifest["layer_count"]}')
print(f'    Schema: {manifest["schema_version"]}')
PYTHON

# ─── Update catalog ───────────────────────────────────────────────────────
echo ""
echo "=== [7/9] Updating meshllm/catalog ==="
/tmp/venv/bin/python3 << 'PYTHON'
from huggingface_hub import HfApi
import os, json, tempfile

api = HfApi(token=os.environ['HF_TOKEN'])
source_repo = os.environ['SOURCE_REPO']
target_repo = os.environ['TARGET_REPO']
source_file = os.environ['SOURCE_FILE']
source_revision = os.environ.get('SOURCE_REVISION', 'main')
model_id = os.environ.get('MODEL_ID', '')
package_dir = os.environ['PACKAGE_DIR']

# Read manifest for metadata
manifest = json.load(open(os.path.join(package_dir, 'model-package.json')))
layer_count = manifest['layer_count']

# Determine catalog entry path: entries/<owner>/<repo-name>.json
owner, repo_name = source_repo.split('/', 1)
entry_path = f"entries/{owner}/{repo_name}.json"

# Try to fetch existing entry
catalog_repo = "meshllm/catalog"
try:
    existing_path = api.hf_hub_download(
        repo_id=catalog_repo,
        filename=entry_path,
        repo_type="dataset",
    )
    entry = json.load(open(existing_path))
except Exception:
    # Create new entry
    entry = {"schema_version": 1, "source_repo": source_repo, "variants": {}}

# Build variant name from source file stem (not MODEL_ID).
# For "UD-Q4_K_XL/Qwen3-32B-UD-Q4_K_XL-00001-of-00002.gguf" → "Qwen3-32B-UD-Q4_K_XL"
import re
file_stem = source_file.split('/')[-1].replace('.gguf', '')
# Strip shard suffix like "-00001-of-00002"
variant_name = re.sub(r'-\d{5}-of-\d{5}$', '', file_stem)

package_entry = {
    "type": "layer-package",
    "repo": target_repo,
    "layer_count": layer_count,
}

# Handle both dict-style and list-style variants
variants = entry.get("variants", {})
if isinstance(variants, dict):
    # Dict-keyed by variant name (existing catalog format)
    if variant_name in variants:
        packages = variants[variant_name].get("packages", [])
        packages = [p for p in packages if p.get("repo") != target_repo]
        packages.append(package_entry)
        variants[variant_name]["packages"] = packages
    else:
        variants[variant_name] = {
            "source": {
                "repo": source_repo,
                "file": source_file,
                "revision": source_revision,
            },
            "curated": {
                "name": variant_name,
                "size": f"{layer_count} layers",
                "description": f"Layer package for {model_id}",
            },
            "packages": [package_entry],
        }
    entry["variants"] = variants
else:
    # List-style (fallback)
    existing_variant = None
    for v in variants:
        if v.get("curated", {}).get("name") == variant_name:
            existing_variant = v
            break
    if existing_variant:
        packages = existing_variant.get("packages", [])
        packages = [p for p in packages if p.get("repo") != target_repo]
        packages.append(package_entry)
        existing_variant["packages"] = packages
    else:
        variants.append({
            "source": {
                "repo": source_repo,
                "file": source_file,
                "revision": source_revision,
            },
            "curated": {
                "name": variant_name,
                "size": f"{layer_count} layers",
                "description": f"Layer package for {model_id}",
            },
            "packages": [package_entry],
        })

# Write and upload
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(entry, f, indent=2)
    tmp_path = f.name

create_pr = os.environ.get('CATALOG_CREATE_PR', 'false').lower() == 'true'

api.upload_file(
    repo_id=catalog_repo,
    path_or_fileobj=tmp_path,
    path_in_repo=entry_path,
    repo_type="dataset",
    commit_message=f"Add layer package for {model_id} ({target_repo})",
    create_pr=create_pr,
)
print(f"  ✓ Catalog updated: {catalog_repo}/{entry_path}")
print(f"    Variant: {variant_name}")
print(f"    Package: {target_repo} ({layer_count} layers)")
PYTHON

# ─── Model Card ────────────────────────────────────────────────────────────
echo ""
echo "=== [8/9] Uploading model card ==="
/tmp/venv/bin/python3 << 'PYTHON'
from huggingface_hub import HfApi
from pathlib import Path
import hashlib
import json
import os

package_dir = Path(os.environ["PACKAGE_DIR"])
manifest_path = package_dir / "model-package.json"
manifest = json.loads(manifest_path.read_text())

source_repo = os.environ["SOURCE_REPO"]
source_file = os.environ["SOURCE_FILE"]
source_revision = os.environ.get("SOURCE_REVISION", "main")
target_repo = os.environ["TARGET_REPO"]
model_id = os.environ.get("MODEL_ID", manifest.get("model_id", target_repo))
mesh_llm_ref = os.environ.get("MESH_LLM_REF", "main")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def fmt_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024

def artifact_bytes(artifact: dict) -> int:
    return int(artifact.get("artifact_bytes") or 0)

def md_cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")

def link(label: str, url: str) -> str:
    return f"[{md_cell(label)}]({url})"

def code(value) -> str:
    return f"`{md_cell(value)}`"

def yaml_quote(value: str) -> str:
    return json.dumps(value)

def infer_model_family(name: str) -> str:
    lowered = name.lower()
    for family in ["Qwen3", "Qwen2.5", "DeepSeek", "Kimi", "Gemma", "GLM", "Llama"]:
        if family.lower() in lowered:
            return family
    return name.split("-")[0] if name else "Unknown"

def infer_parameter_scale(name: str) -> str:
    import re
    match = re.search(r"(?i)(\d+(?:\.\d+)?[BM](?:-A\d+(?:\.\d+)?B)?)", name)
    return match.group(1) if match else "not recorded"

def infer_quantization(name: str, source_path: str) -> str:
    import re
    combined = f"{name}/{source_path}"
    patterns = [
        r"UD-Q\d+_[A-Z]+(?:_[A-Z]+)?",
        r"Q\d+_[A-Z]+(?:_[A-Z]+)?",
        r"IQ\d+_[A-Z]+(?:_[A-Z]+)?",
        r"BF16",
        r"F16",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            return match.group(0)
    return "not recorded"

shared = manifest.get("shared", {})
layers = manifest.get("layers", [])
projectors = manifest.get("projectors", [])
manifest_hash = sha256(manifest_path)
total_bytes = sum(artifact_bytes(artifact) for artifact in shared.values())
total_bytes += sum(artifact_bytes(layer) for layer in layers)
total_bytes += sum(artifact_bytes(projector) for projector in projectors)

source_model = manifest.get("source_model", {})
display_name = source_model.get("distribution_id") or model_id
model_family = infer_model_family(display_name)
parameter_scale = infer_parameter_scale(display_name)
quantization = infer_quantization(display_name, source_file)
activation_width = manifest.get("activation_width") or "not recorded"
skippy_abi = manifest.get("skippy_abi_version") or "not recorded"
source_sha = source_model.get("sha256") or "not recorded"
canonical_ref = source_model.get("canonical_ref") or f"{source_repo}@{source_revision}/{source_file}"

file_rows = [
    ("Manifest", "model-package.json", "Package schema, source identity, checksums", manifest_hash),
]
for label, key in [
    ("Metadata", "metadata"),
    ("Embeddings", "embeddings"),
    ("Output head", "output"),
]:
    artifact = shared.get(key)
    if artifact:
        file_rows.append((
            label,
            artifact.get("path", f"shared/{key}.gguf"),
            f"{artifact.get('tensor_count', 'unknown')} tensors, {fmt_bytes(artifact_bytes(artifact))}",
            artifact.get("sha256", "not recorded"),
        ))
if layers:
    layer_bytes = sum(artifact_bytes(layer) for layer in layers)
    layer_tensors = sum(int(layer.get("tensor_count") or 0) for layer in layers)
    file_rows.append((
        "Transformer layers",
        "layers/layer-*.gguf",
        f"{len(layers)} layer artifacts, {layer_tensors} tensors, {fmt_bytes(layer_bytes)}",
        "see model-package.json",
    ))
for projector in projectors:
    file_rows.append((
        "Projector",
        projector.get("path", "projectors/projector.gguf"),
        f"{projector.get('kind', 'multimodal')} projector, {fmt_bytes(artifact_bytes(projector))}",
        projector.get("sha256", "not recorded"),
    ))

rows = [
    ("Source model", link(source_repo, f"https://huggingface.co/{source_repo}")),
    ("Model id", code(model_id)),
    ("Family", model_family),
    ("Parameter scale", parameter_scale),
    ("Quantization", code(quantization)),
    ("Layer count", manifest.get("layer_count", len(layers))),
    ("Activation width", activation_width),
    ("Package size", fmt_bytes(total_bytes)),
    ("Source file", code(source_file)),
    ("Package repo", link(target_repo, f"https://huggingface.co/{target_repo}")),
]

readme = f"""---
library_name: mesh-llm
base_model:
- {yaml_quote(source_repo)}
pipeline_tag: text-generation
tags:
- gguf
- mesh-llm
- layer-package
- skippy
- distributed-inference
- local-inference
- openai-compatible
---

<div align="center">
  <a href="https://www.meshllm.cloud">
    <img src="https://github.com/Mesh-LLM/mesh-llm/raw/main/docs/mesh-llm-logo.svg" alt="Mesh LLM" width="220">
  </a>

  <h1>{display_name}</h1>

  <p>
    <strong>Distributed GGUF inference package for Mesh LLM</strong>
  </p>

  <p>
    <a href="https://www.meshllm.cloud"><img alt="Website" src="https://img.shields.io/badge/Website-meshllm.cloud-111111?style=for-the-badge"></a>
    <a href="https://github.com/Mesh-LLM/mesh-llm"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Mesh--LLM-24292f?style=for-the-badge&logo=github"></a>
    <a href="https://discord.gg/rs6fmc63eN"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white"></a>
  </p>
</div>

GGUF layer package for running **{display_name}** across a local Mesh LLM cluster.

This package is derived from [{source_repo}](https://huggingface.co/{source_repo}) and keeps the original GGUF distribution split into per-layer artifacts for distributed inference.

## Highlights

| Run locally | Pool multiple machines | OpenAI-compatible | Package variant |
|---|---|---|---|
| Private inference on your hardware | Split layers across peers | Serve `/v1/chat/completions` locally | `{quantization}` layer package |

## Model Overview

| Property | Value |
|---|---|
"""

for key, value in rows:
    readme += f"| **{md_cell(key)}** | {md_cell(value)} |\n"

readme += f"""
## Recommended Use

- Local and private inference with Mesh LLM.
- Multi-machine serving when the full GGUF is too large for one host.
- OpenAI-compatible chat/completions workflows through Mesh LLM's local API.

For upstream architecture details, chat template guidance, sampling recommendations, license terms, and benchmark notes, see the source model card: [{source_repo}](https://huggingface.co/{source_repo}).

## Quickstart

```bash
# Run this on each machine that should contribute memory/compute.
mesh-llm serve --model "{target_repo}" --split
```

```bash
# Check the mesh and discover the OpenAI-compatible model name.
curl -s http://localhost:3131/api/status
curl -s http://localhost:3131/v1/models
```

```bash
# Send an OpenAI-compatible chat request.
curl -s http://localhost:3131/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model_id}",
    "messages": [{{"role": "user", "content": "Write a tiny hello-world function in Rust."}}],
    "max_tokens": 128
  }}'
```

## Package Variant

| Property | Value |
|---|---|
"""

for key, value in [
    ("Format", code(manifest.get("format", "layer-package"))),
    ("Canonical source ref", code(canonical_ref)),
    ("Source revision", code(source_revision)),
    ("Source SHA-256", code(source_sha)),
    ("Skippy ABI", code(skippy_abi)),
    ("Package manifest SHA-256", code(manifest_hash)),
]:
    readme += f"| **{md_cell(key)}** | {md_cell(value)} |\n"

readme += f"""
## What Is Included

| Artifact | Path | Contents | SHA-256 |
|---|---|---|---|
"""

for label, path, contents, checksum in file_rows:
    readme += f"| {md_cell(label)} | {code(path)} | {md_cell(contents)} | {code(checksum)} |\n"

readme += f"""
## Validation

Generated by the Mesh LLM HF Jobs splitter from `mesh-llm` ref `{mesh_llm_ref}` and validated before upload:

```bash
skippy-model-package validate-package "/source/{source_file}" "{package_dir}"
```

## Links

- Source model: [{source_repo}](https://huggingface.co/{source_repo})
- Mesh LLM website: [meshllm.cloud](https://www.meshllm.cloud)
- Mesh LLM: [github.com/Mesh-LLM/mesh-llm](https://github.com/Mesh-LLM/mesh-llm)
- Discord: [discord.gg/rs6fmc63eN](https://discord.gg/rs6fmc63eN)
- Package catalog: [meshllm/catalog](https://huggingface.co/datasets/meshllm/catalog)
- Package format: [layer-package-repos.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/specs/layer-package-repos.md)
"""

Path("/tmp/README.md").write_text(readme)

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj="/tmp/README.md",
    path_in_repo="README.md",
    repo_id=target_repo,
    repo_type="model",
)
print("  ✓ Model card uploaded")
PYTHON

# ─── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== [9/9] Done ==="
echo ""
echo "  Published:  https://huggingface.co/${TARGET_REPO}"
echo "  Layers:     ${LAYER_COUNT}"
echo "  Total size: ${TOTAL_SIZE}"
echo ""
echo "  Use with mesh-llm:"
echo "    mesh-llm serve --model ${TARGET_REPO} --split"
