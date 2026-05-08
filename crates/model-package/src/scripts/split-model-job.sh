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
PACKAGE_DIR="/tmp/package"
mkdir -p "$PACKAGE_DIR"

if [ ! -f "$SOURCE_PATH" ]; then
    echo "ERROR: Source file not found at $SOURCE_PATH"
    echo ""
    echo "Available GGUF files in /source:"
    find /source -name "*.gguf" -type f | sort | head -30
    exit 1
fi

echo "  Source: $SOURCE_PATH ($(du -h "$SOURCE_PATH" | cut -f1))"
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
    folder_path='/tmp/package',
    commit_message=f'Layer package from {source_repo} ({model_id})',
)

# Print summary
manifest = json.load(open('/tmp/package/model-package.json'))
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

# Read manifest for metadata
manifest = json.load(open('/tmp/package/model-package.json'))
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
ACTIVATION_WIDTH=$(python3 -c "import json; m=json.load(open('/tmp/package/model-package.json')); print(m.get('activation_width', 'unknown'))")

cat > /tmp/README.md << EOF
---
library_name: mesh-llm
base_model: ${SOURCE_REPO}
tags:
- mesh-llm
- layer-package
- skippy
- distributed-inference
---

# ${MODEL_ID} — Layer Package for Mesh LLM

Pre-split layer package for distributed inference with [Mesh LLM](https://github.com/Mesh-LLM/mesh-llm).

**Source model:** [${SOURCE_REPO}](https://huggingface.co/${SOURCE_REPO})

## Details

| Property | Value |
|---|---|
| **Source** | [${SOURCE_REPO}](https://huggingface.co/${SOURCE_REPO}) |
| **Source file** | \`${SOURCE_FILE}\` |
| **Layers** | ${LAYER_COUNT} |
| **Total size** | ${TOTAL_SIZE} |
| **Activation width** | ${ACTIVATION_WIDTH} |
| **Format** | Per-layer GGUF (layer-package) |

## Usage

\`\`\`bash
# Each node downloads only its assigned layers:
mesh-llm serve --model "${TARGET_REPO}" --split
\`\`\`

Nodes discover each other on the local network, plan the topology based on available RAM, and each downloads only its portion.

## Structure

\`\`\`
model-package.json          # Manifest (layer count, checksums, metadata)
shared/metadata.gguf        # Model metadata & vocabulary
shared/embeddings.gguf      # Token embedding weights
shared/output.gguf          # Output head weights
layers/layer-000.gguf       # Per-layer transformer weights
layers/layer-001.gguf
...
layers/layer-$(printf "%03d" $((LAYER_COUNT - 1))).gguf
\`\`\`

## Links

- **Source model:** [${SOURCE_REPO}](https://huggingface.co/${SOURCE_REPO})
- [Mesh LLM](https://github.com/Mesh-LLM/mesh-llm) — distributed inference runtime
- [Splitter tool](https://github.com/Mesh-LLM/hf-mesh-skippy-splitter) — HF Jobs layer splitter
- [Model catalog](https://huggingface.co/datasets/meshllm/catalog) — registry of available packages
EOF

/tmp/venv/bin/python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj='/tmp/README.md',
    path_in_repo='README.md',
    repo_id=os.environ['TARGET_REPO'],
    repo_type='model',
)
print('  ✓ Model card uploaded')
"

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
