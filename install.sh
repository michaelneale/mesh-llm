#!/usr/bin/env bash

set -euo pipefail

REPO="${MESH_LLM_INSTALL_REPO:-Mesh-LLM/mesh-llm}"
REPO_REF="${MESH_LLM_INSTALL_REF:-main}"
INSTALL_DIR="${MESH_LLM_INSTALL_DIR:-$HOME/.local/bin}"
INSTALL_FLAVOR="${MESH_LLM_INSTALL_FLAVOR:-}"
INSTALL_PRERELEASE="${MESH_LLM_INSTALL_PRERELEASE:-0}"
INSTALL_SERVICE="${MESH_LLM_INSTALL_SERVICE:-0}"
INSTALL_SERVICE_ARGS="${MESH_LLM_INSTALL_SERVICE_ARGS:-}"
INSTALL_SERVICE_START="${MESH_LLM_INSTALL_SERVICE_START:-1}"

SERVICE_NAME="mesh-llm"
SERVICE_LABEL="com.mesh-llm.mesh-llm"
MESH_CONFIG_FILE="$HOME/.mesh-llm/config.toml"
SERVICE_CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/mesh-llm"
SERVICE_ENV_FILE="$SERVICE_CONFIG_DIR/service.env"
SERVICE_RUNNER="$SERVICE_CONFIG_DIR/run-service.sh"
SYSTEMD_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SYSTEMD_UNIT_PATH="$SYSTEMD_UNIT_DIR/$SERVICE_NAME.service"
LAUNCHD_AGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCHD_PLIST_PATH="$LAUNCHD_AGENT_DIR/$SERVICE_LABEL.plist"
LAUNCHD_LOG_DIR="$HOME/Library/Logs/mesh-llm"
LAUNCHD_STDOUT_LOG="$LAUNCHD_LOG_DIR/stdout.log"
LAUNCHD_STDERR_LOG="$LAUNCHD_LOG_DIR/stderr.log"
DIST_DIR="dist"
SYSTEMD_TEMPLATE_PATH="$DIST_DIR/$SERVICE_NAME.service"
LAUNCHD_TEMPLATE_PATH="$DIST_DIR/$SERVICE_LABEL.plist"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

path_contains_install_dir() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) return 0 ;;
        *) return 1 ;;
    esac
}

bool_is_true() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<EOF
Usage: install.sh [--pre-release] [--service] [--no-start-service]

Options:
  --pre-release              Install the latest published GitHub prerelease instead of the latest stable release.
  --service                  Install a per-user background service for this platform.
  --no-start-service         Install the service files but do not start them yet.
  -h, --help                 Show this help text.

Environment overrides:
  MESH_LLM_INSTALL_DIR
  MESH_LLM_INSTALL_FLAVOR
  MESH_LLM_INSTALL_PRERELEASE=1
  MESH_LLM_INSTALL_REF=main
  MESH_LLM_INSTALL_SERVICE=1
  MESH_LLM_INSTALL_SERVICE_START=0
EOF
}

parse_args() {
    while (($# > 0)); do
        case "$1" in
            --pre-release)
                INSTALL_PRERELEASE=1
                ;;
            --service)
                INSTALL_SERVICE=1
                ;;
            --service-args)
                echo "error: background services now run \`mesh-llm serve\` and load startup models from $MESH_CONFIG_FILE" >&2
                echo "Add startup models under [[models]] instead of passing custom service args." >&2
                exit 1
                ;;
            --no-start-service)
                INSTALL_SERVICE_START=0
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "error: unknown argument: $1" >&2
                echo >&2
                usage >&2
                exit 1
                ;;
        esac
        shift
    done
}

platform_id() {
    printf "%s/%s\n" "$(uname -s)" "$(uname -m)"
}

probe_nvidia() {
    command -v nvidia-smi >/dev/null 2>&1 ||
        command -v nvcc >/dev/null 2>&1 ||
        [[ -e /dev/nvidiactl ]] ||
        [[ -d /proc/driver/nvidia/gpus ]]
}

probe_rocm() {
    command -v rocm-smi >/dev/null 2>&1 ||
        command -v rocminfo >/dev/null 2>&1 ||
        command -v hipcc >/dev/null 2>&1 ||
        [[ -x /opt/rocm/bin/hipcc ]]
}

probe_vulkan() {
    if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1; then
        return 0
    fi
    if command -v glslc >/dev/null 2>&1; then
        if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists vulkan 2>/dev/null; then
            return 0
        fi
        if [[ -f /usr/include/vulkan/vulkan.h || -f /usr/local/include/vulkan/vulkan.h ]]; then
            return 0
        fi
        if [[ -n "${VULKAN_SDK:-}" ]]; then
            return 0
        fi
    fi
    return 1
}

supported_flavors() {
    case "$(platform_id)" in
        Darwin/arm64)
            echo "metal"
            ;;
        Linux/x86_64)
            echo "cpu cuda rocm vulkan"
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

recommended_flavor() {
    case "$(platform_id)" in
        Darwin/arm64)
            echo "metal"
            ;;
        Linux/x86_64)
            if probe_nvidia; then
                echo "cuda"
            elif probe_rocm; then
                echo "rocm"
            elif probe_vulkan; then
                echo "vulkan"
            else
                echo "cpu"
            fi
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

recommendation_reason() {
    case "$(recommended_flavor)" in
        metal)
            echo "Apple Silicon host detected."
            ;;
        cuda)
            echo "NVIDIA tooling or devices were detected."
            ;;
        rocm)
            echo "ROCm/HIP tooling was detected."
            ;;
        vulkan)
            echo "Vulkan tooling was detected."
            ;;
        cpu)
            echo "No supported GPU runtime was detected."
            ;;
    esac
}

validate_flavor() {
    local flavor="$1"
    local supported
    for supported in $(supported_flavors); do
        if [[ "$supported" == "$flavor" ]]; then
            return 0
        fi
    done
    echo "error: unsupported flavor '$flavor' for $(platform_id)" >&2
    exit 1
}

choose_flavor() {
    local recommended
    recommended="$(recommended_flavor)"

    if [[ -n "$INSTALL_FLAVOR" ]]; then
        validate_flavor "$INSTALL_FLAVOR"
        echo "$INSTALL_FLAVOR"
        return 0
    fi

    if [[ ! -t 0 || ! -t 1 ]]; then
        echo "$recommended"
        return 0
    fi

    local flavors
    flavors=($(supported_flavors))

    if [[ ${#flavors[@]} -eq 1 ]]; then
        echo "$recommended"
        return 0
    fi

    echo "Mesh LLM installer"
    echo "Platform: $(platform_id)"
    echo "Recommended flavor: $recommended"
    echo "Reason: $(recommendation_reason)"
    echo
    echo "Available flavors:"

    local index=1
    local flavor
    for flavor in "${flavors[@]}"; do
        if [[ "$flavor" == "$recommended" ]]; then
            echo "  $index. $flavor (recommended)"
        else
            echo "  $index. $flavor"
        fi
        index=$((index + 1))
    done

    echo
    local reply
    read -r -p "Install which flavor? [$recommended] " reply
    reply="${reply:-$recommended}"

    if [[ "$reply" =~ ^[0-9]+$ ]]; then
        local selection=$((reply - 1))
        if (( selection >= 0 && selection < ${#flavors[@]} )); then
            reply="${flavors[$selection]}"
        fi
    fi

    validate_flavor "$reply"
    echo "$reply"
}

asset_name() {
    local flavor="$1"
    case "$(platform_id)" in
        Darwin/arm64)
            echo "mesh-llm-aarch64-apple-darwin.tar.gz"
            ;;
        Linux/x86_64)
            case "$flavor" in
                cpu) echo "mesh-llm-x86_64-unknown-linux-gnu.tar.gz" ;;
                cuda) echo "mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz" ;;
                rocm) echo "mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz" ;;
                vulkan) echo "mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz" ;;
                *)
                    echo "error: unsupported Linux flavor '$flavor'" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

latest_prerelease_tag() {
    local api_url="https://api.github.com/repos/${REPO}/releases?per_page=20"
    local releases_page_url="https://github.com/${REPO}/releases"
    local response
    local -a curl_args=(
        -fsSL
        -H 'Accept: application/vnd.github+json'
        -H 'X-GitHub-Api-Version: 2022-11-28'
    )

    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        curl_args+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    elif [[ -n "${GH_TOKEN:-}" ]]; then
        curl_args+=(-H "Authorization: Bearer ${GH_TOKEN}")
    fi

    local tag
    if response="$(curl "${curl_args[@]}" "$api_url" 2>/dev/null)"; then
        local compact
        compact="$(printf '%s' "$response" | tr -d '\n\r\t ')"

        tag="$(
            printf '%s' "$compact" |
                sed 's/},{/}\
{/g' |
                awk '
                    /"prerelease":true/ && !/"draft":true/ {
                        if (match($0, /"tag_name":"[^"]+"/)) {
                            value = substr($0, RSTART, RLENGTH)
                            sub(/^"tag_name":"/, "", value)
                            sub(/"$/, "", value)
                            print value
                            exit
                        }
                    }
                '
        )"
    fi

    if [[ -z "${tag:-}" ]]; then
        if ! response="$(curl -fsSL "$releases_page_url" 2>/dev/null)"; then
            echo "error: could not query GitHub releases for ${REPO}" >&2
            return 1
        fi

        tag="$(
            printf '%s' "$response" |
                tr '\n' ' ' |
                sed "s#<a href=\"/${REPO}/releases/tag/#\\
TAG:#g; s#Pre-release#\\
PRE-RELEASE#g" |
                awk '
                    /TAG:/ {
                        tag = $0
                        sub(/^.*TAG:/, "", tag)
                        sub(/".*$/, "", tag)
                    }
                    /PRE-RELEASE/ && tag != "" {
                        print tag
                        exit
                    }
                '
        )"
    fi

    if [[ -z "$tag" ]]; then
        echo "error: could not find a published prerelease for ${REPO}" >&2
        return 1
    fi

    printf '%s\n' "$tag"
}

release_url() {
    local asset="$1"
    if bool_is_true "$INSTALL_PRERELEASE"; then
        local tag
        if ! tag="$(latest_prerelease_tag)"; then
            return 1
        fi
        printf 'https://github.com/%s/releases/download/%s/%s\n' "$REPO" "$tag" "$asset"
        return 0
    fi

    printf 'https://github.com/%s/releases/latest/download/%s\n' "$REPO" "$asset"
}

stale_binary_names() {
    cat <<'EOF'
mesh-llm
rpc-server
llama-server
llama-moe-split
rpc-server-cpu
llama-server-cpu
rpc-server-cuda
llama-server-cuda
rpc-server-rocm
llama-server-rocm
rpc-server-vulkan
llama-server-vulkan
rpc-server-metal
llama-server-metal
EOF
}

remove_stale_binaries() {
    mkdir -p "$INSTALL_DIR"
    local name
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        rm -f "$INSTALL_DIR/$name"
    done < <(stale_binary_names)
}

install_bundle() {
    local bundle_dir="$1"
    remove_stale_binaries

    local file
    for file in "$bundle_dir"/*; do
        mv -f "$file" "$INSTALL_DIR/"
    done
}

systemd_escape_assignment_value() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '%s' "$value"
}

systemd_quote_token() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$/$$}"
    value="${value//%/%%}"
    printf '"%s"' "$value"
}

local_template_path() {
    local rel_path="$1"
    local source_path="${BASH_SOURCE[0]-}"
    local script_dir

    if [[ -z "$source_path" || "$source_path" != */* ]]; then
        return 1
    fi

    script_dir="$(cd "$(dirname "$source_path")" && pwd)"
    [[ -f "$script_dir/$rel_path" ]] || return 1
    printf '%s\n' "$script_dir/$rel_path"
}

template_stream() {
    local rel_path="$1"
    local local_path

    if local_path="$(local_template_path "$rel_path")"; then
        cat "$local_path"
        return 0
    fi

    curl -fsSL "https://raw.githubusercontent.com/${REPO}/${REPO_REF}/${rel_path}"
}

render_template_to_file() {
    local template_path="$1"
    local output_path="$2"
    shift 2

    local -a replacements=("$@")
    local -a env_vars=()
    local pair
    local key
    local value
    for pair in "${replacements[@]}"; do
        key="${pair%%=*}"
        value="${pair#*=}"
        env_vars+=("TPL_${key}=${value}")
    done

    env "${env_vars[@]}" awk '
        BEGIN {
            split("ARGS_METADATA ENV_LINES", multiline_keys, " ");
            for (i in multiline_keys) {
                multiline[multiline_keys[i]] = 1;
            }
        }
        {
            line = $0;
            for (name in multiline) {
                marker = "@" name "@";
                if (line == marker) {
                    print ENVIRON["TPL_" name];
                    next;
                }
            }
            for (var in ENVIRON) {
                if (index(var, "TPL_") != 1) {
                    continue;
                }
                name = substr(var, 5);
                if (name in multiline) {
                    continue;
                }
                marker = "@" name "@";
                gsub(marker, ENVIRON[var], line);
            }
            print line;
        }
    ' < <(template_stream "$template_path") > "$output_path"
}

ensure_service_env_file() {
    if [[ -f "$SERVICE_ENV_FILE" ]]; then
        return
    fi

    mkdir -p "$(dirname "$SERVICE_ENV_FILE")"
    {
        echo "# Optional environment variables for mesh-llm."
        echo "# Use plain KEY=value lines."
        echo "# Example:"
        echo "# RUST_LOG=mesh_inference=debug"
    } > "$SERVICE_ENV_FILE"
}

write_service_runner() {
    mkdir -p "$(dirname "$SERVICE_RUNNER")"

    cat > "$SERVICE_RUNNER" <<EOF
#!/usr/bin/env bash

set -euo pipefail

BIN="$INSTALL_DIR/mesh-llm"
ENV_FILE="$SERVICE_ENV_FILE"

if [[ ! -x "\$BIN" ]]; then
    echo "mesh-llm binary not found or not executable: \$BIN" >&2
    exit 1
fi

if [[ -f "\$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    . "\$ENV_FILE"
    set +a
fi

exec "\$BIN" serve
EOF

    chmod +x "$SERVICE_RUNNER"
}

ensure_launchd_service_files() {
    mkdir -p "$SERVICE_CONFIG_DIR"
    ensure_service_env_file
    write_service_runner
}

install_systemd_service() {
    need_cmd systemctl
    mkdir -p "$SERVICE_CONFIG_DIR" "$SYSTEMD_UNIT_DIR"
    ensure_service_env_file
    local exec_line
    exec_line="ExecStart=$(systemd_quote_token "$INSTALL_DIR/mesh-llm") serve"

    render_template_to_file "$SYSTEMD_TEMPLATE_PATH" "$SYSTEMD_UNIT_PATH" \
        "ARGS_METADATA=# mesh-llm serve (startup models come from $MESH_CONFIG_FILE)" \
        "SERVICE_ENV_FILE=$SERVICE_ENV_FILE" \
        "ENV_LINES=" \
        "EXEC_LINE=$exec_line"

    systemctl --user daemon-reload || true

    if bool_is_true "$INSTALL_SERVICE_START"; then
        if systemctl --user enable "$SERVICE_NAME.service" &&
            (systemctl --user restart "$SERVICE_NAME.service" ||
                systemctl --user start "$SERVICE_NAME.service"); then
            echo "Installed and started systemd user service: $SERVICE_NAME.service"
        else
            echo "Installed $SYSTEMD_UNIT_PATH" >&2
            echo "warning: could not start the systemd user service automatically." >&2
            echo "Start it with: systemctl --user enable --now $SERVICE_NAME.service" >&2
        fi
    else
        echo "Installed $SYSTEMD_UNIT_PATH"
        echo "Start it with: systemctl --user enable --now $SERVICE_NAME.service"
    fi

    echo "Command: $exec_line"
    echo "Optional env: $SERVICE_ENV_FILE"
    echo "Edit startup models: $MESH_CONFIG_FILE"
    echo "Logs: journalctl --user -u $SERVICE_NAME.service -f"
    echo "Boot without login (optional): sudo loginctl enable-linger \$USER"
}

install_launchd_service() {
    need_cmd launchctl
    ensure_launchd_service_files
    mkdir -p "$LAUNCHD_AGENT_DIR" "$LAUNCHD_LOG_DIR"

    render_template_to_file "$LAUNCHD_TEMPLATE_PATH" "$LAUNCHD_PLIST_PATH" \
        "SERVICE_LABEL=$SERVICE_LABEL" \
        "SERVICE_RUNNER=$SERVICE_RUNNER" \
        "HOME_DIR=$HOME" \
        "STDOUT_LOG=$LAUNCHD_STDOUT_LOG" \
        "STDERR_LOG=$LAUNCHD_STDERR_LOG"

    local launch_domain="gui/$(id -u)"
    if bool_is_true "$INSTALL_SERVICE_START"; then
        launchctl bootout "$launch_domain" "$LAUNCHD_PLIST_PATH" >/dev/null 2>&1 || true
        if launchctl bootstrap "$launch_domain" "$LAUNCHD_PLIST_PATH"; then
            launchctl enable "$launch_domain/$SERVICE_LABEL" >/dev/null 2>&1 || true
            launchctl kickstart -k "$launch_domain/$SERVICE_LABEL" >/dev/null 2>&1 || true
            echo "Installed and started launchd agent: $SERVICE_LABEL"
        else
            echo "Installed $LAUNCHD_PLIST_PATH" >&2
            echo "warning: could not start the launchd agent automatically." >&2
            echo "Start it with: launchctl bootstrap $launch_domain $LAUNCHD_PLIST_PATH" >&2
        fi
    else
        echo "Installed $LAUNCHD_PLIST_PATH"
        echo "Start it with: launchctl bootstrap $launch_domain $LAUNCHD_PLIST_PATH"
    fi

    echo "Startup models: $MESH_CONFIG_FILE"
    echo "Optional env: $SERVICE_ENV_FILE"
    echo "Logs: $LAUNCHD_STDOUT_LOG and $LAUNCHD_STDERR_LOG"
}

install_service() {
    case "$(uname -s)" in
        Darwin)
            install_launchd_service
            ;;
        Linux)
            install_systemd_service
            ;;
        *)
            echo "error: service install is not supported on $(uname -s)" >&2
            exit 1
            ;;
    esac
}

main() {
    parse_args "$@"
    if [[ -n "$INSTALL_SERVICE_ARGS" ]]; then
        echo "error: background services now run \`mesh-llm serve\` and load startup models from $MESH_CONFIG_FILE" >&2
        echo "Add startup models under [[models]] instead of using MESH_LLM_INSTALL_SERVICE_ARGS." >&2
        exit 1
    fi
    need_cmd curl
    need_cmd tar
    need_cmd mktemp

    local flavor
    flavor="$(choose_flavor)"
    local asset
    asset="$(asset_name "$flavor")"
    local url
    if ! url="$(release_url "$asset")"; then
        exit 1
    fi

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local tmp_dir_escaped
    printf -v tmp_dir_escaped '%q' "$tmp_dir"
    trap "rm -rf -- $tmp_dir_escaped" EXIT

    local archive="$tmp_dir/$asset"
    echo "Installing flavor: $flavor"
    if bool_is_true "$INSTALL_PRERELEASE"; then
        echo "Release channel: prerelease"
    else
        echo "Release channel: stable"
    fi
    echo "Downloading $url"
    curl -fsSL "$url" -o "$archive"

    tar -xzf "$archive" -C "$tmp_dir"

    if [[ ! -d "$tmp_dir/mesh-bundle" ]]; then
        echo "error: release archive did not contain mesh-bundle/" >&2
        exit 1
    fi

    install_bundle "$tmp_dir/mesh-bundle"

    echo "Installed $asset to $INSTALL_DIR"

    if bool_is_true "$INSTALL_SERVICE"; then
        echo
        install_service
    fi

    if ! path_contains_install_dir; then
        echo
        echo "$INSTALL_DIR is not on your PATH."
        echo "Add it with one of these commands:"
        echo
        echo "bash:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        echo
        echo "zsh:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc"
        echo "  source ~/.zshrc"
    fi
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
