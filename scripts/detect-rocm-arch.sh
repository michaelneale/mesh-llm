#!/usr/bin/env bash
# detect-rocm-arch.sh — detect AMDGPU_TARGETS values for ROCm builds
#
# Outputs a semicolon-separated list of gfx targets, e.g. "gfx942;gfx90a".

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

ARCHES=()

add_arch() {
    local arch="$1"
    [[ "$arch" =~ ^gfx[0-9a-z]+$ ]] || return 0
    for existing in "${ARCHES[@]:-}"; do
        [[ "$existing" == "$arch" ]] && return 0
    done
    ARCHES+=("$arch")
}

if command -v amdgpu-arch &>/dev/null; then
    while IFS= read -r arch; do
        arch="${arch//[[:space:]]/}"
        [[ -n "$arch" ]] && add_arch "$arch"
    done < <(amdgpu-arch 2>/dev/null || true)
fi

if [[ ${#ARCHES[@]} -eq 0 ]] && command -v rocminfo &>/dev/null; then
    while IFS= read -r arch; do
        arch="${arch//[[:space:]]/}"
        [[ -n "$arch" ]] && add_arch "$arch"
    done < <(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' || true)
fi

if [[ ${#ARCHES[@]} -eq 0 ]] && command -v rocm-smi &>/dev/null; then
    while IFS= read -r series; do
        case "$series" in
            *MI300X*|*MI300*)
                add_arch gfx942
                ;;
            *MI250*|*MI210*|*MI200*)
                add_arch gfx90a
                ;;
            *MI100*)
                add_arch gfx908
                ;;
            *RX\ 7900*|*Navi31*)
                add_arch gfx1100
                ;;
            *RX\ 7800*|*RX\ 7700*|*Navi32*)
                add_arch gfx1101
                ;;
            *RX\ 7600*|*Navi33*)
                add_arch gfx1102
                ;;
        esac
    done < <(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*Card series:[[:space:]]*//p' || true)
fi

if [[ ${#ARCHES[@]} -eq 0 ]]; then
    die "Could not detect ROCm architecture automatically.
Pass the arch explicitly:
  just build backend=rocm rocm_arch=gfx942

Common values:
  gfx942  MI300X / MI300
  gfx90a  MI250 / MI210
  gfx908  MI100
  gfx1100 Radeon RX 7900 / Navi31
  gfx1101 Radeon RX 7800 / RX 7700 / Navi32
  gfx1102 Radeon RX 7600 / Navi33"
fi

(IFS=';'; echo "${ARCHES[*]}")
