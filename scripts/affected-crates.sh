#!/usr/bin/env bash
# scripts/affected-crates.sh — Affected crate detection via cargo metadata graph walk.
set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: $(basename "$0") [--stdin] [file ...]

Detect workspace crates affected by changed files using cargo metadata.

Options:
  --stdin   Read newline-separated file paths from stdin instead of CLI args

Output (JSON to stdout):
{
  "affected": ["crate-a", "crate-b"],
  "batches": [["leaf-crate"], ["intermediate-crate"]],
  "all_rust_changed": false,
  "ui_changed": false
}
EOF
  exit 1
}

###############################################################################
# Parse inputs: CLI args or stdin (--stdin flag)
###############################################################################
read_from_stdin=false
files=()

for arg in "$@"; do
  case "$arg" in
    --help|-h) usage ;;
    --stdin)   read_from_stdin=true ;;
    *)         files+=("$arg") ;;
  esac
done

if $read_from_stdin; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] && files+=("$line")
  done
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo '{"affected":[],"batches":[],"all_rust_changed":false,"ui_changed":false}'
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

###############################################################################
# Check for special-case files that trigger all_rust_changed or ui_changed
###############################################################################
all_rust_changed=false
ui_changed=false
ROOT_CARGO_TOML="$WORKSPACE_ROOT/Cargo.toml"

for f in "${files[@]}"; do
  if [[ "$f" != /* ]]; then
    f="$WORKSPACE_ROOT/$f"
  fi

  case "$f" in
    */Cargo.lock|*/Cargo.bazel.lock)       all_rust_changed=true ;;
    */third_party/llama.cpp/*)             all_rust_changed=true ;;
    */scripts/build-*.sh)                  all_rust_changed=true ;;
    */Justfile|*/justfile)                 all_rust_changed=true ;;
    "$WORKSPACE_ROOT/Cargo.toml")          all_rust_changed=true ;;
  esac

  if [[ "$(realpath "$f" 2>/dev/null)" == "$(realpath "$ROOT_CARGO_TOML")" ]]; then
    all_rust_changed=true
  fi

  [[ "$f" == */crates/mesh-llm-ui/* ]] && ui_changed=true
done

if $all_rust_changed; then
  all_crates=$(cargo metadata --format-version=1 --no-deps \
    --manifest-path "$ROOT_CARGO_TOML" 2>/dev/null \
    | jq -r '[.packages[] | .name] | sort' \
    || echo '["mesh-llm"]')

  echo "{\"affected\":$all_crates,\"batches\":[],\"all_rust_changed\":true,\"ui_changed\":$ui_changed}"
  exit 0
fi

###############################################################################
# Run cargo metadata and build workspace crate directory map
###############################################################################
METADATA=$(cargo metadata --format-version=1 \
  --manifest-path "$ROOT_CARGO_TOML" 2>/dev/null \
  || { echo '{"affected":[],"batches":[],"all_rust_changed":true,"ui_changed":false}' >&2; exit 0; })

CRATE_DIRS=$(echo "$METADATA" | jq -r '
  [ .packages[] | select(.source == null) ] as $pkgs |
  ($pkgs | map({key: .name, value: (.manifest_path | split("/") | .[:-1] | join("/"))}) | from_entries)
')

ALL_CRATES=$(echo "$CRATE_DIRS" | jq -r 'keys[]')

###############################################################################
# Map changed files to workspace crates by directory prefix matching
###############################################################################
changed_crates=()

for f in "${files[@]}"; do
  if [[ "$f" != /* ]]; then
    f="$WORKSPACE_ROOT/$f"
  fi
  
  rel="${f#$WORKSPACE_ROOT/}"
  
  for crate in $ALL_CRATES; do
    crate_dir=$(echo "$CRATE_DIRS" | jq -r --arg c "$crate" '.[$c] // empty')
    [[ -z "$crate_dir" ]] && continue
    
    crate_rel="${crate_dir#$WORKSPACE_ROOT/}"
    
    if [[ "$rel" == "$crate_rel/"* || "$rel" == "$crate_rel" ]]; then
      changed_crates+=("$crate")
      break
    fi
  done
done

IFS=$'\n' read -r -d '' -a changed_crates < <(printf '%s\n' "${changed_crates[@]}" | sort -u && printf '\0') || true

# Filter out mesh-llm-ui from Rust affected list when only UI assets changed.
# The ui_changed flag already captures UI file changes for workflow conditions.
if [[ ${#changed_crates[@]} -eq 1 && "${changed_crates[0]}" == "mesh-llm-ui" ]]; then
  changed_crates=()
fi

if [[ ${#changed_crates[@]} -eq 0 ]]; then
  # Fail-open: if no Rust crates detected, default to all crates for safety.
  all_crates=$(cargo metadata --format-version=1 --no-deps \
    --manifest-path "$ROOT_CARGO_TOML" 2>/dev/null \
    | jq -r '[.packages[] | .name] | sort' \
    || echo '["mesh-llm"]')

  if [[ "$all_rust_changed" == "true" ]]; then
    echo "{\"affected\":$all_crates,\"batches\":[],\"all_rust_changed\":true,\"ui_changed\":$ui_changed}"
  else
    # ui_changed is the only signal — don't trigger Rust checks for pure UI PRs.
    if [[ "$ui_changed" == "true" ]]; then
      echo "{\"affected\":[],\"batches\":[],\"all_rust_changed\":false,\"ui_changed\":true}"
    else
      # No recognized changes at all — fail open to all crates.
      echo "{\"affected\":$all_crates,\"batches\":[],\"all_rust_changed\":true,\"ui_changed\":false}"
    fi
  fi
  exit 0
fi

###############################################################################
# BFS: compute transitive reverse dependencies (all crates depending on changed ones)
###############################################################################
AFFECTED=$(echo "$METADATA" | jq --argjson changed "$(printf '%s\n' "${changed_crates[@]}" | jq -R . | jq -s .)" '
  [ .packages[] | select(.source == null) ] as $pkgs |
  
  [$pkgs[] | {name}] | map(.name) as $member_names |
  
  ($pkgs | map({
    key: .name,
    value: [.dependencies[] | select(.name as $d | $member_names | index($d)) | .name] | unique
  }) | from_entries) as $deps |

  {affected: ($changed | sort), queue: ($changed | sort)} |
  until (.queue | length == 0;
    (
      .queue as $current |
      [ $deps | to_entries[] | select(.value | any(. == ($current[]))) | .key ] | unique
    ) as $new_deps |
    (($new_deps - .affected) | sort) as $to_add |
    . + {affected: (.affected + $to_add), queue: $to_add}
  ) | .affected
')

###############################################################################
# Topological sort: group affected crates into dependency-depth batches
###############################################################################
BATCHES=$(echo "$METADATA" | jq --argjson affected "$(echo "$AFFECTED")" '
  [ .packages[] | select(.source == null) ] as $pkgs |
  
  [$pkgs[] | {name}] | map(.name) as $member_names |
  
  ($pkgs | map({
    key: .name,
    value: [.dependencies[] | select(.name as $d | $member_names | index($d)) | .name] | unique
  }) | from_entries) as $deps |

  ($affected | map({key: ., value: [($deps[.] // [])[] | select(. as $d | $affected | index($d))]}) | from_entries) as $filtered_deps |

  {remaining: ($affected | sort), assigned: [], batches: []} |
  until (.remaining | length == 0;
    (
      .assigned as $asgn |
      .remaining as $left |
      [ $filtered_deps | to_entries[] |
        select(.key as $k | $left | index($k)) |
        select(([.value[] | . as $v | ($asgn | any(. == $v))] | all) or (.value | length) == 0) |
        .key
      ] | sort
    ) as $batch |

    if ($batch | length) == 0 then
      . + {batches: (.batches + [.remaining]), remaining: []}
    else
      . + {assigned: (.assigned + $batch), batches: (.batches + [$batch]), remaining: (.remaining - $batch)}
    end
  ) | .batches
')

###############################################################################
# Output JSON result
###############################################################################
jq -n \
  --argjson affected "$AFFECTED" \
  --argjson batches "$BATCHES" \
  --argjson all_rust "$all_rust_changed" \
  --argjson ui "$ui_changed" \
  '{affected: $affected, batches: $batches, all_rust_changed: $all_rust, ui_changed: $ui}'
