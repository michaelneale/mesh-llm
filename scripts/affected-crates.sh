#!/usr/bin/env bash
set -euo pipefail

# affected-crates.sh: Compute which Rust crates are affected by changed files
# Usage:
#   bash scripts/affected-crates.sh --stdin < changed_files.txt
#   bash scripts/affected-crates.sh file1.rs file2.rs ...

# Hardcoded workspace members (fallback for fail-open)
WORKSPACE_MEMBERS=(
  "mesh-llm"
  "mesh-llm-gpu-bench"
  "mesh-llm-host-runtime"
  "mesh-llm-identity"
  "mesh-llm-protocol"
  "mesh-llm-routing"
  "mesh-llm-system"
  "mesh-llm-types"
  "mesh-llm-ui"
  "mesh-llm-plugin"
  "mesh-llm-client"
  "mesh-api"
  "mesh-host-core"
  "mesh-api-ffi"
  "mesh-llm-test-harness"
  "model-ref"
  "model-artifact"
  "model-hf"
  "model-resolver"
  "skippy-protocol"
  "skippy-coordinator"
  "skippy-topology"
  "skippy-cache"
  "skippy-metrics"
  "openai-frontend"
  "skippy-ffi"
  "skippy-runtime"
  "skippy-server"
  "metrics-server"
  "skippy-model-package"
  "model-package"
  "skippy-correctness"
  "llama-spec-bench"
  "skippy-bench"
  "skippy-prompt"
  "xtask"
)

# Fail-open handler: emit all_rust=true with full workspace
fail_open() {
  local exit_code=$?
  echo "WARNING: affected-crates.sh encountered an error (exit=$exit_code), falling back to all_rust=true" >&2
  
  # Build full workspace list as JSON array
  local all_crates_json="["
  for i in "${!WORKSPACE_MEMBERS[@]}"; do
    if [ $i -gt 0 ]; then
      all_crates_json+=","
    fi
    all_crates_json+="\"${WORKSPACE_MEMBERS[$i]}\""
  done
  all_crates_json+="]"
  
  # Emit fallback JSON
  cat <<EOF
{
  "affected": $all_crates_json,
  "test_crates": [],
  "batches": [[], [], []],
  "all_rust": true,
  "ui_changed": false
}
EOF
  exit 0
}

trap 'fail_open' ERR

main() {
  # Parse input: --stdin or positional args
  local -a changed_files=()
  
  if [[ "${1:-}" == "--stdin" ]]; then
    while IFS= read -r line; do
      [[ -n "$line" ]] && changed_files+=("$line")
    done
  else
    changed_files=("$@")
  fi
  
  # Check for escalation paths or __force_all__ sentinel
  local escalate=false
  local ui_changed=false
  
  for file in "${changed_files[@]}"; do
    # __force_all__ sentinel
    if [[ "$file" == "__force_all__" ]]; then
      escalate=true
      break
    fi
    
    # Escalation patterns
    if [[ "$file" =~ ^Cargo\.lock$ ]] || \
       [[ "$file" =~ ^Cargo\.toml$ ]] || \
       [[ "$file" =~ ^third_party/llama\.cpp/ ]] || \
       [[ "$file" =~ ^scripts/(build-llama|prepare-llama|build-mac|build-windows|skippy-ci-smoke)\. ]] || \
       [[ "$file" =~ ^Justfile$ ]] || \
       [[ "$file" =~ ^\.github/cache-version\.txt$ ]] || \
       [[ "$file" =~ ^\.github/workflows/ci\.yml$ ]] || \
       [[ "$file" =~ ^\.github/workflows/pr_builds\.yml$ ]] || \
       [[ "$file" =~ ^\.github/workflows/pr_quality\.yml$ ]] || \
       [[ "$file" =~ ^\.github/workflows/pr_docker\.yml$ ]] || \
       [[ "$file" =~ ^\.github/workflows/pr_cleanup\.yml$ ]] || \
       [[ "$file" =~ ^\.github/workflows/smoke\.yml$ ]] || \
       [[ "$file" =~ ^scripts/affected-crates\.sh$ ]] || \
       [[ "$file" =~ ^scripts/plan-clippy-batches\.sh$ ]] || \
       [[ "$file" =~ ^rust-toolchain(\.toml)?$ ]]; then
      escalate=true
      break
    fi
    
    # UI changed detection
    if [[ "$file" =~ ^crates/mesh-llm-ui/ ]]; then
      ui_changed=true
    fi
  done
  
  # If escalation, return all_rust=true
  if [[ "$escalate" == true ]]; then
    local all_crates_json="["
    for i in "${!WORKSPACE_MEMBERS[@]}"; do
      if [ $i -gt 0 ]; then
        all_crates_json+=","
      fi
      all_crates_json+="\"${WORKSPACE_MEMBERS[$i]}\""
    done
    all_crates_json+="]"
    
    cat <<EOF
{
  "affected": $all_crates_json,
  "test_crates": [],
  "batches": [[], [], []],
  "all_rust": true,
  "ui_changed": $([[ "$ui_changed" == true ]] && echo "true" || echo "false")
}
EOF
    return 0
  fi
  
  # Normal path: use cargo metadata to build crate map and reverse-dep graph
  
  # Step 1: Get crate → manifest_path mapping (no deps)
  local metadata_no_deps
  metadata_no_deps=$(cargo metadata --format-version=1 --no-deps 2>/dev/null)

  local workspace_root
  workspace_root=$(echo "$metadata_no_deps" | jq -r '.workspace_root')

  local -A crate_to_dir=()
  while IFS='|' read -r crate_name manifest_path; do
    [[ -z "$crate_name" ]] && continue
    dir="${manifest_path%/Cargo.toml}"
    crate_to_dir["$crate_name"]="$dir"
  done < <(echo "$metadata_no_deps" | jq -r '.packages[] | "\(.name)|\(.manifest_path)"')
  
  # Step 2: Build reverse-dep graph: for each crate, list which crates depend on it
  local -A reverse_deps=()
  local metadata_json
  metadata_json=$(cargo metadata --format-version=1 2>/dev/null)
  
  # Build name→id mapping
  local -A name_to_id=()
  while IFS='|' read -r name id; do
    [[ -z "$name" ]] && continue
    name_to_id["$name"]="$id"
  done < <(echo "$metadata_json" | jq -r '.packages[] | "\(.name)|\(.id)"')
  
  # Build reverse deps: for each node, add it as a reverse dep of its dependencies
  while IFS='|' read -r node_id dep_id; do
    [[ -z "$node_id" ]] && continue
    # Find crate name for dep_id
    local dep_name=""
    for name in "${!name_to_id[@]}"; do
      if [[ "${name_to_id[$name]}" == "$dep_id" ]]; then
        dep_name="$name"
        break
      fi
    done
    [[ -z "$dep_name" ]] && continue
    
    # Find crate name for node_id
    local node_name=""
    for name in "${!name_to_id[@]}"; do
      if [[ "${name_to_id[$name]}" == "$node_id" ]]; then
        node_name="$name"
        break
      fi
    done
    [[ -z "$node_name" ]] && continue
    
    # Add node_name as reverse dep of dep_name
    if [[ -z "${reverse_deps[$dep_name]:-}" ]]; then
      reverse_deps["$dep_name"]="$node_name"
    else
      reverse_deps["$dep_name"]="${reverse_deps[$dep_name]} $node_name"
    fi
  done < <(echo "$metadata_json" | jq -r '.resolve.nodes[] | "\(.id)|\(.dependencies[]?)"')
  
  # Step 3: Match changed files to owning crates
  local -a test_crates=()
  
  for file in "${changed_files[@]}"; do
    # Skip non-Rust files (docs, config, etc.)
    if [[ ! "$file" =~ ^crates/ ]] && [[ ! "$file" =~ ^tools/ ]]; then
      continue
    fi
    
    # Skip UI crate files (they don't affect Rust builds)
    if [[ "$file" =~ ^crates/mesh-llm-ui/ ]]; then
      continue
    fi
    
    # Find longest-prefix match in crate_to_dir
    local best_crate=""
    local best_len=0
    
    for crate_name in "${!crate_to_dir[@]}"; do
      local crate_dir="${crate_to_dir[$crate_name]}"
      # Convert absolute manifest dirs to paths relative to the cargo workspace root.
      local crate_rel="$crate_dir"
      if [[ "$crate_rel" == "$workspace_root" ]]; then
        crate_rel=""
      elif [[ "$crate_rel" == "$workspace_root/"* ]]; then
        crate_rel="${crate_rel#"$workspace_root/"}"
      fi
      
      if [[ -n "$crate_rel" ]] && { [[ "$file" == "$crate_rel" ]] || [[ "$file" == "$crate_rel/"* ]]; }; then
        local len=${#crate_rel}
        if [[ $len -gt $best_len ]]; then
          best_crate="$crate_name"
          best_len=$len
        fi
      fi
    done
    
    if [[ -n "$best_crate" ]] && [[ ! " ${test_crates[@]} " =~ " ${best_crate} " ]]; then
      test_crates+=("$best_crate")
    fi
  done
  
  # Step 4: BFS from test_crates through reverse-dep graph
  local -a affected=()
  local -a queue=("${test_crates[@]}")
  local -A visited=()
  
  while [[ ${#queue[@]} -gt 0 ]]; do
    local current="${queue[0]}"
    queue=("${queue[@]:1}")
    
    if [[ -n "${visited[$current]:-}" ]]; then
      continue
    fi
    visited[$current]=1
    affected+=("$current")
    
    # Get reverse deps of current
    local deps="${reverse_deps[$current]:-}"
    for dep in $deps; do
      [[ -z "$dep" ]] && continue
      if [[ -z "${visited[$dep]:-}" ]]; then
        queue+=("$dep")
      fi
    done
  done
  
  # Step 5: Topologically sort affected crates and bucket into 3 batches
  # For simplicity, use depth in reverse-dep graph (distance from leaves)
  local -A depth_map=()
  
  for crate in "${affected[@]}"; do
    local max_depth=0
    local deps="${reverse_deps[$crate]:-}"
    
    for dep in $deps; do
      [[ -z "$dep" ]] && continue
      local dep_depth=${depth_map[$dep]:-0}
      if [[ $dep_depth -gt $max_depth ]]; then
        max_depth=$dep_depth
      fi
    done
    
    depth_map[$crate]=$((max_depth + 1))
  done
  
  # Bucket by depth % 3
  local -a batch0=()
  local -a batch1=()
  local -a batch2=()
  
  for crate in "${affected[@]}"; do
    local d=${depth_map[$crate]:-0}
    local bucket=$((d % 3))
    
    case $bucket in
      0) batch0+=("$crate") ;;
      1) batch1+=("$crate") ;;
      2) batch2+=("$crate") ;;
    esac
  done
  
  # Step 6: Emit JSON output using jq
  jq -n \
    --argjson affected "$(printf '%s\n' "${affected[@]}" | jq -Rs 'split("\n") | map(select(length > 0))')" \
    --argjson test_crates "$(printf '%s\n' "${test_crates[@]}" | jq -Rs 'split("\n") | map(select(length > 0))')" \
    --argjson batch0 "$(printf '%s\n' "${batch0[@]}" | jq -Rs 'split("\n") | map(select(length > 0))')" \
    --argjson batch1 "$(printf '%s\n' "${batch1[@]}" | jq -Rs 'split("\n") | map(select(length > 0))')" \
    --argjson batch2 "$(printf '%s\n' "${batch2[@]}" | jq -Rs 'split("\n") | map(select(length > 0))')" \
    --arg ui_changed "$([[ "$ui_changed" == true ]] && echo "true" || echo "false")" \
    '{
      affected: $affected,
      test_crates: $test_crates,
      batches: [$batch0, $batch1, $batch2],
      all_rust: false,
      ui_changed: ($ui_changed == "true")
    }'
}

main "$@"
