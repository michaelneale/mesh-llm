#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"

exec "$repo_root/target/release/mesh-llm" serve --offline --model unsloth/gemma-4-31B-it-GGUF:UD-IQ2_XXS