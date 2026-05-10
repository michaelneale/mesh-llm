#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: scripts/release-version.sh <version|vversion>" >&2
    exit 1
fi

raw_version="$1"
version="${raw_version#v}"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "invalid version: $raw_version" >&2
    echo "expected semantic version like 0.49.0, 0.49.0-rc.1, v0.49.0, or v0.49.0-rc.1" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

require_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "missing required file: $file" >&2
        exit 1
    fi
}

update_lib_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/pub const VERSION: &str = "\K[^"]+(?=";)/'"$next"'/g' "$file")"
    if [[ "$before" == "$after" ]]; then
        if grep -Eq 'pub const VERSION: &str = "'"$next"'";' "$file"; then
            return
        fi
        echo "failed to update VERSION constant in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

update_manifest_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    if perl -0777 -ne 'exit((/\[package\][^[]*?\nversion\.workspace\s*=\s*true/s) ? 0 : 1)' "$file"; then
        return
    fi
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(\[package\][^[]*?\nversion\s*=\s*")[^"]+(")/${1}'"$next"'$2/s' "$file")"
    if [[ "$before" == "$after" ]]; then
        if perl -0777 -ne 'exit((/\[package\][^[]*?\nversion\s*=\s*"'"$next"'"/s) ? 0 : 1)' "$file"; then
            return
        fi
        echo "failed to update [package].version in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

update_workspace_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(\[workspace\.package\][^[]*?\nversion\s*=\s*")[^"]+(")/${1}'"$next"'$2/s' "$file")"
    if [[ "$before" == "$after" ]]; then
        if perl -0777 -ne 'exit((/\[workspace\.package\][^[]*?\nversion\s*=\s*"'"$next"'"/s) ? 0 : 1)' "$file"; then
            return
        fi
        echo "failed to update [workspace.package].version in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

update_mesh_client_dependency_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(mesh-client\s*=\s*\{[^}]*package\s*=\s*"mesh-llm-client"[^}]*version\s*=\s*")[^"]+(")/${1}'"$next"'$2/s' "$file")"
    if [[ "$before" == "$after" ]]; then
        return
    fi
    printf '%s\n' "$after" >"$file"
}

update_gradle_project_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(\nversion\s*=\s*")[^"]+(")/${1}'"$next"'$2/s' "$file")"
    if [[ "$before" == "$after" ]]; then
        if perl -0777 -ne 'exit((/\nversion\s*=\s*"'"$next"'"/s) ? 0 : 1)' "$file"; then
            return
        fi
        echo "failed to update Gradle project version in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

manifests=()
while IFS= read -r manifest; do
    manifests+=("$manifest")
done < <(
    cd "$REPO_ROOT"
    git ls-files \
        'crates/*/Cargo.toml' \
        'tools/*/Cargo.toml' \
        | sort -u
)

if [[ "${#manifests[@]}" -eq 0 ]]; then
    echo "no Cargo.toml manifests found under crates/ or tools/" >&2
    exit 1
fi

versioned_files=()

workspace_manifest="$REPO_ROOT/Cargo.toml"
require_file "$workspace_manifest"
update_workspace_version "$workspace_manifest" "$version"
versioned_files+=("$workspace_manifest")

lib_file="$REPO_ROOT/crates/mesh-llm-host-runtime/src/lib.rs"
require_file "$lib_file"
update_lib_version "$lib_file" "$version"
versioned_files+=("$lib_file")

for relative_manifest in "${manifests[@]}"; do
    manifest="$REPO_ROOT/$relative_manifest"
    require_file "$manifest"
    update_manifest_version "$manifest" "$version"
    update_mesh_client_dependency_version "$manifest" "$version"
    versioned_files+=("$manifest")
done

kotlin_build_file="$REPO_ROOT/sdk/kotlin/build.gradle.kts"
require_file "$kotlin_build_file"
update_gradle_project_version "$kotlin_build_file" "$version"
versioned_files+=("$kotlin_build_file")

echo "Refreshing Cargo.lock workspace package versions..."
(cd "$REPO_ROOT" && cargo metadata --format-version 1 >/dev/null)

versioned_files+=("$REPO_ROOT/Cargo.lock")

echo "Updated release version to $version:"
for file in "${versioned_files[@]}"; do
    echo "  $file"
done
