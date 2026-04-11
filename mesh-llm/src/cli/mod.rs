use clap::{Parser, Subcommand};
use std::ffi::OsString;
use std::path::PathBuf;

use crate::cli::benchmark::BenchmarkCommand;
use crate::cli::runtime::RuntimeCommand;
use crate::crypto::TrustPolicy;

#[derive(Subcommand, Debug)]
pub(crate) enum TrustCommand {
    /// Add an owner to the local trust store allowlist.
    Add {
        /// Owner ID to trust.
        owner_id: String,
        /// Optional human label for this owner.
        #[arg(long)]
        label: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Remove an owner from the local trust store allowlist.
    Remove {
        /// Owner ID to remove.
        owner_id: String,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Show the current trust store contents.
    List {
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum AuthCommand {
    /// Generate a new owner keypair and save to keystore.
    Init {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Overwrite an existing keystore.
        #[arg(long)]
        force: bool,
        /// Skip passphrase prompt (store keys unencrypted).
        #[arg(long, conflicts_with = "keychain")]
        no_passphrase: bool,
        /// Store a random unlock passphrase in the OS keychain (macOS Keychain,
        /// Windows Credential Manager, Linux Secret Service). New keystores
        /// already default to this when a backend is available; use this flag
        /// to force it when overwriting an existing keystore.
        #[arg(long)]
        keychain: bool,
    },
    /// Show current owner identity status.
    Status {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Path to the node ownership certificate.
        #[arg(long)]
        node_ownership: Option<PathBuf>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Sign the current node identity with the existing owner keystore.
    SignNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
    },
    /// Renew the local node ownership certificate in place.
    RenewNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
    },
    /// Verify a node ownership certificate.
    VerifyNode {
        /// Path to the signed node certificate.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Override the node ID to verify against.
        #[arg(long)]
        node_id: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
        /// Override trust policy used for verification.
        #[arg(long = "verify-trust-policy", value_enum)]
        trust_policy: Option<TrustPolicy>,
    },
    /// Rotate the local node identity key.
    RotateNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
        /// Revoke the current certificate and node ID in the local trust store first.
        #[arg(long)]
        revoke_current: bool,
        /// Optional revocation reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Revoke an owner in the local trust store.
    RevokeOwner {
        /// Owner ID to revoke.
        owner_id: String,
        /// Optional reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Revoke a node certificate or node ID in the local trust store.
    RevokeNode {
        /// Certificate ID to revoke.
        #[arg(long)]
        cert_id: Option<String>,
        /// Node endpoint ID to revoke.
        #[arg(long)]
        node_id: Option<String>,
        /// Optional reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Rotate the existing owner keystore identity.
    RotateOwner {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Skip passphrase prompt (store keys unencrypted).
        #[arg(long)]
        no_passphrase: bool,
        /// Overwrite an existing backup file if present.
        #[arg(long)]
        force: bool,
    },
    /// Manage the local trust store.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
}

pub(crate) mod benchmark;
pub(crate) mod commands;
pub mod models;
pub(crate) mod runtime;

#[derive(Parser, Debug)]
#[command(
    name = "mesh-llm",
    version = crate::VERSION,
    about = "Pool GPUs over the internet for LLM inference",
    after_help = "Preferred runtime entrypoints:\n  mesh-llm serve\n  mesh-llm serve --model Qwen3-8B-Q4_K_M\n  mesh-llm client --auto\n  mesh-llm gpus\n\n`mesh-llm serve` loads startup models from ~/.mesh-llm/config.toml.\nRun with --help-advanced for all options."
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Command>,

    /// Show all options (including advanced/niche ones).
    #[arg(long, hide = true)]
    pub(crate) help_advanced: bool,

    /// Join a mesh via invite token (can repeat).
    #[arg(long, short)]
    pub(crate) join: Vec<String>,

    /// Discover a mesh via Nostr and join it.
    #[arg(long, default_missing_value = "", num_args = 0..=1)]
    pub(crate) discover: Option<String>,

    /// Auto-join the best mesh found via Nostr.
    #[arg(long)]
    pub(crate) auto: bool,

    /// Model to serve (path, catalog name, HF exact ref, or HuggingFace URL).
    #[arg(long)]
    pub(crate) model: Vec<PathBuf>,

    /// Raw local GGUF file to serve directly (repeatable).
    #[arg(long)]
    pub(crate) gguf: Vec<PathBuf>,

    /// Explicit mmproj sidecar to pass to llama-server for the primary served model.
    #[arg(long, hide = true)]
    pub(crate) mmproj: Option<PathBuf>,

    /// API port (default: 9337).
    #[arg(long, default_value = "9337")]
    pub(crate) port: u16,

    /// Run as a client — no GPU, no model needed.
    #[arg(long)]
    pub(crate) client: bool,

    /// Web console port (default: 3131).
    #[arg(long, default_value = "3131")]
    pub(crate) console: u16,

    /// Publish this mesh for discovery by others.
    #[arg(long)]
    pub(crate) publish: bool,

    /// Name for this mesh (shown in discovery).
    #[arg(long)]
    pub(crate) mesh_name: Option<String>,

    /// Region tag, e.g. "US", "EU", "AU" (shown in discovery).
    #[arg(long)]
    pub(crate) region: Option<String>,

    /// Enable blackboard on public meshes (on by default for private meshes).
    #[arg(long)]
    pub(crate) blackboard: bool,

    /// Your display name on the blackboard.
    #[arg(long)]
    pub(crate) name: Option<String>,

    /// Internal plugin service mode.
    #[arg(long, hide = true)]
    pub(crate) plugin: Option<String>,

    /// Update mesh-llm before continuing for release-bundle installs if a newer bundled release is available.
    #[arg(long, global = true)]
    pub(crate) auto_update: bool,

    // ── Advanced options (hidden from default --help) ─────────────
    /// Draft model for speculative decoding.
    #[arg(long, hide = true)]
    pub(crate) draft: Option<PathBuf>,

    /// Max draft tokens (default: 8).
    #[arg(long, default_value = "8", hide = true)]
    pub(crate) draft_max: u16,

    /// Disable automatic draft model detection.
    #[arg(long, hide = true)]
    pub(crate) no_draft: bool,

    /// Force tensor split even if the model fits on one node.
    #[arg(long, hide = true)]
    pub(crate) split: bool,

    /// Override context size (tokens). Default: auto-scaled to available VRAM.
    #[arg(long, hide = true)]
    pub(crate) ctx_size: Option<u32>,

    /// Cap VRAM used for planning, local-fit decisions, and mesh advertisement (GB).
    #[arg(long)]
    pub(crate) max_vram: Option<f64>,

    /// Enumerate host hardware (GPU name, hostname) at startup.
    #[arg(long, hide = true)]
    pub(crate) enumerate_host: bool,

    /// Path to rpc-server, llama-server, and llama-moe-split binaries.
    #[arg(long, hide = true)]
    pub(crate) bin_dir: Option<PathBuf>,

    /// Override which bundled llama.cpp flavor to use.
    #[arg(long, value_enum)]
    pub(crate) llama_flavor: Option<crate::inference::launch::BinaryFlavor>,

    /// Device for rpc-server (e.g. MTL0, CUDA0, HIP0, Vulkan0, CPU).
    #[arg(long, hide = true)]
    pub(crate) device: Option<String>,

    /// Tensor split ratios for llama-server (e.g. "0.8,0.2").
    #[arg(long, hide = true)]
    pub(crate) tensor_split: Option<String>,

    /// Override iroh relay URLs.
    #[arg(long, hide = true)]
    pub(crate) relay: Vec<String>,

    /// Bind QUIC to a fixed UDP port (for NAT port forwarding).
    #[arg(long, hide = true)]
    pub(crate) bind_port: Option<u16>,

    /// Bind to 0.0.0.0 (for containers/Fly.io).
    #[arg(long, hide = true)]
    pub(crate) listen_all: bool,

    /// Stop advertising when N clients connected.
    #[arg(long, hide = true)]
    pub(crate) max_clients: Option<usize>,

    /// Custom Nostr relay URLs.
    #[arg(long, hide = true)]
    pub(crate) nostr_relay: Vec<String>,

    /// Ignored (backward compat).
    #[arg(long, hide = true)]
    pub(crate) no_console: bool,

    /// Optional path to the mesh-llm config file.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,

    /// Path to the owner keystore used to attest this node.
    #[arg(long)]
    pub(crate) owner_key: Option<PathBuf>,

    /// Fail startup if owner attestation cannot be loaded or signed.
    #[arg(long)]
    pub(crate) owner_required: bool,

    /// Optional human label attached to this node certificate.
    #[arg(long)]
    pub(crate) node_label: Option<String>,

    /// Override peer ownership trust policy.
    #[arg(long, value_enum)]
    pub(crate) trust_policy: Option<TrustPolicy>,

    /// Add trusted owner IDs on top of the local trust store.
    #[arg(long)]
    pub(crate) trust_owner: Vec<String>,

    /// Internal: set when this node joined via Nostr discovery (not --join).
    #[arg(skip)]
    pub(crate) nostr_discovery: bool,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Manage model storage, migration, and update checks.
    Models {
        #[command(subcommand)]
        command: models::ModelsCommand,
    },
    /// Download a model from the catalog
    Download {
        /// Model name (e.g. "Qwen2.5-32B-Instruct-Q4_K_M" or just "32b")
        name: Option<String>,
        /// Also download the recommended draft model for speculative decoding
        #[arg(long)]
        draft: bool,
    },
    /// Update mesh-llm to the latest bundled release and exit.
    Update,
    /// Inspect local GPUs, stable IDs, and cached bandwidth.
    #[command(alias = "gpu")]
    Gpus,
    /// Inspect and manage local runtime-served models.
    #[command(hide = true)]
    Runtime {
        #[command(subcommand)]
        command: Option<RuntimeCommand>,
    },
    /// Load a local model into a running mesh-llm instance.
    Load {
        /// Model name/path/url to load
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Unload a local model from a running mesh-llm instance.
    #[command(alias = "drop")]
    Unload {
        /// Model name to unload
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Show local model status on a running mesh-llm instance.
    Status {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Discover meshes on Nostr and optionally auto-join one.
    Discover {
        /// Filter by model name (substring match)
        #[arg(long)]
        model: Option<String>,
        /// Filter by minimum VRAM (GB)
        #[arg(long)]
        min_vram: Option<f64>,
        /// Filter by region
        #[arg(long)]
        region: Option<String>,
        /// Print the invite token of the best match (for piping to --join)
        #[arg(long)]
        auto: bool,
        /// Nostr relay URLs (default: see DEFAULT_RELAYS)
        #[arg(long)]
        relay: Vec<String>,
    },
    /// Rotate all identity keys (node + Nostr).
    #[command(hide = true)]
    RotateKey,
    /// Launch Goose with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "goose")]
    Goose {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Launch Claude Code with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "claude")]
    Claude {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Stop all running mesh-llm, llama-server, and rpc-server processes.
    Stop,
    /// Blackboard — post, search, and read messages shared across the mesh.
    ///
    /// Post a message:   mesh-llm blackboard "your message here"
    /// Show feed:        mesh-llm blackboard
    /// Search:           mesh-llm blackboard --search "query"
    /// From a peer:      mesh-llm blackboard --from tyler
    /// MCP server:       mesh-llm client --join <token> blackboard --mcp
    /// Install skill:    mesh-llm blackboard install-skill
    ///
    /// Conventions: prefix messages with QUESTION:, STATUS:, FINDING:, TIP: etc.
    /// Search picks these up naturally via multi-term OR matching.
    #[command(name = "blackboard")]
    Blackboard {
        /// Message to post (if provided).
        text: Option<String>,
        /// Search the blackboard.
        #[arg(long)]
        search: Option<String>,
        /// Filter by author name.
        #[arg(long)]
        from: Option<String>,
        /// Only show items from the last N hours (default: 24).
        #[arg(long)]
        since: Option<f64>,
        /// Max items to show (default: 20).
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Console/API port of the running mesh-llm instance.
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Run as an MCP server over stdio (for agent integration).
        #[arg(long)]
        mcp: bool,
    },
    /// Plugin management.
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
    /// Benchmark and compare model/runtime strategies.
    #[command(hide = true)]
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCommand,
    },
    /// Manage owner identity and keystore.
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum PluginCommand {
    /// Compatibility shim for the old install workflow.
    Install {
        /// Plugin name.
        name: String,
    },
    /// List auto-registered and configured plugins.
    List,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeSurface {
    Serve,
    Client,
}

#[derive(Clone, Debug)]
pub(crate) struct NormalizedRuntimeArgs {
    pub(crate) original: Vec<OsString>,
    pub(crate) normalized: Vec<OsString>,
    pub(crate) explicit_surface: Option<RuntimeSurface>,
}

pub(crate) fn normalize_runtime_surface_args<I, S>(args: I) -> NormalizedRuntimeArgs
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let original: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let mut normalized = original.clone();
    let mut explicit_surface = None;

    match original.get(1).and_then(|arg| arg.to_str()) {
        Some("serve") => match original.get(2).and_then(|arg| arg.to_str()) {
            Some(arg) if arg.starts_with('-') => {
                normalized.remove(1);
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            None => {
                normalized[1] = OsString::from("--help");
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            _ => {}
        },
        Some("client") => {
            normalized.remove(1);
            normalized.insert(1, OsString::from("--client"));
            explicit_surface = Some(RuntimeSurface::Client);
        }
        _ => {}
    }

    NormalizedRuntimeArgs {
        original,
        normalized,
        explicit_surface,
    }
}

pub(crate) fn legacy_runtime_surface_warning(
    cli: &Cli,
    original_args: &[OsString],
    explicit_surface: Option<RuntimeSurface>,
) -> Option<String> {
    if explicit_surface.is_some() || cli.command.is_some() {
        return None;
    }

    if cli.client {
        return Some(format!(
            "⚠️ top-level `--client` now maps to `mesh-llm client`.\n  Please use: {}",
            suggested_client_command(original_args)
        ));
    }

    if !cli.model.is_empty() || !cli.gguf.is_empty() || cli.mmproj.is_some() {
        return Some(format!(
            "⚠️ top-level serving flags now map to `mesh-llm serve`.\n  Please use: {}",
            suggested_serve_command(original_args)
        ));
    }

    None
}

fn suggested_serve_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len() + 1);
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("serve"));
    args.extend(original_args.iter().skip(1).cloned());
    shell_join(&args)
}

fn suggested_client_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len());
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("client"));
    let mut skipped_client = false;
    for arg in original_args.iter().skip(1) {
        if !skipped_client && arg.to_string_lossy() == "--client" {
            skipped_client = true;
            continue;
        }
        args.push(arg.clone());
    }
    shell_join(&args)
}

fn shell_join(args: &[OsString]) -> String {
    args.iter()
        .map(|arg| shell_display(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_display(arg: &OsString) -> String {
    let text = arg.to_string_lossy();
    if text.is_empty() {
        "\"\"".into()
    } else if text
        .chars()
        .any(|ch| ch.is_whitespace() || matches!(ch, '"' | '\'' | '\\'))
    {
        format!("{text:?}")
    } else {
        text.into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn normalize_runtime_surface_args_rewrites_serve_invocation() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--auto",
            "--model",
            "Qwen3-8B-Q4_K_M",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_bare_serve_rewrites_to_help() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--help"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_rewrites_client_invocation() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "client", "--auto", "--port", "9337"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--client", "--auto", "--port", "9337"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_keeps_non_runtime_subcommands() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "download", "foo"]);

        assert_eq!(normalized.explicit_surface, None);
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "download", "foo"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_serve_flags() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm serve --auto --model Qwen3-8B-Q4_K_M"));
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_client_flag() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "--auto", "--client"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm client --auto"));
    }

    #[test]
    fn explicit_runtime_surface_suppresses_legacy_warning() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "client", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        assert!(legacy_runtime_surface_warning(
            &cli,
            &normalized.original,
            normalized.explicit_surface
        )
        .is_none());
    }

    #[test]
    fn auth_status_accepts_owner_key_locally() {
        let cli = Cli::parse_from(["mesh-llm", "auth", "status", "--owner-key", "owner.json"]);

        match cli.command.expect("auth command expected") {
            Command::Auth {
                command: AuthCommand::Status { owner_key, .. },
            } => {
                assert_eq!(owner_key, Some(PathBuf::from("owner.json")));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn auth_status_rejects_runtime_only_owner_required_flag() {
        let err = Cli::try_parse_from(["mesh-llm", "auth", "status", "--owner-required"])
            .expect_err("runtime-only flag should be rejected for auth status");

        let rendered = err.to_string();
        assert!(rendered.contains("--owner-required"));
    }
}
