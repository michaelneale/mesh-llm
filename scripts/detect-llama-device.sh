#!/usr/bin/env bash
# detect-llama-device.sh — pick the best llama.cpp device string for this host/binary

set -euo pipefail

probe_available_devices() {
    local binary="${1:-}"
    [[ -n "$binary" && -x "$binary" ]] || return 0

    local output
    output="$("$binary" -d __mesh_llm_probe_invalid__ -p 0 2>&1 || true)"

    awk '
        BEGIN { in_devices = 0 }
        {
            line = $0
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
            if (line == "available devices:") {
                in_devices = 1
                next
            }
            if (!in_devices || line == "") {
                next
            }
            split(line, parts, ":")
            name = parts[1]
            if (name ~ /^[[:alnum:]]+$/) {
                print name
            }
        }
    ' <<<"$output"
}

detect_host_device() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo MTL0
        return 0
    fi

    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q '[^[:space:]]'; then
            echo CUDA0
            return 0
        fi
    fi

    if command -v tegrastats &>/dev/null; then
        echo CUDA0
        return 0
    fi

    if command -v rocm-smi &>/dev/null; then
        if rocm-smi --showproductname 2>/dev/null | grep -q '^GPU\['; then
            echo ROCm0
            return 0
        fi
    fi

    if command -v rocminfo &>/dev/null; then
        if rocminfo 2>/dev/null | grep -q 'gfx'; then
            echo ROCm0
            return 0
        fi
    fi

    if command -v vulkaninfo &>/dev/null; then
        if vulkaninfo --summary >/dev/null 2>&1; then
            echo Vulkan0
            return 0
        fi
    fi

    echo CPU
}

pick_preferred_available_device() {
    local available=("$@")
    local candidate

    for candidate in MTL0 CUDA0 ROCm0 HIP0 Vulkan0 CPU; do
        local device
        for device in "${available[@]}"; do
            if [[ "$device" == "$candidate" ]]; then
                echo "$candidate"
                return 0
            fi
        done
    done

    [[ ${#available[@]} -gt 0 ]] && echo "${available[0]}"
}

main() {
    local binary="${1:-}"
    local detected
    detected="$(detect_host_device)"

    if [[ -n "$binary" ]]; then
        local available=()
        local line
        while IFS= read -r line; do
            [[ -n "$line" ]] && available+=("$line")
        done < <(probe_available_devices "$binary")
        if [[ ${#available[@]} -gt 0 ]]; then
            local device
            for device in "${available[@]}"; do
                if [[ "$device" == "$detected" ]]; then
                    echo "$detected"
                    return 0
                fi
            done

            pick_preferred_available_device "${available[@]}"
            return 0
        fi
    fi

    echo "$detected"
}

main "$@"
