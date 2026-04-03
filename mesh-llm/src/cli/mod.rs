use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::cli::benchmark::BenchmarkCommand;
use crate::cli::runtime::RuntimeCommand;

pub(crate) mod benchmark;
pub(crate) mod commands;
pub mod models;
pub(crate) mod runtime;

#[derive(Parser, Debug)]
#[command(
    name = "mesh-llm",
    version = crate::VERSION,
    about = "Pool GPUs over the internet for LLM inference",
    after_help = "Run with --help-advanced for all options."
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

    /// Disable startup self-update for this process.
    #[arg(long, hide = true)]
    pub(crate) no_self_update: bool,

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

    /// Limit VRAM advertised to the mesh (GB).
    #[arg(long, hide = true)]
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
    #[arg(long, hide = true)]
    pub(crate) config: Option<PathBuf>,

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
    /// MCP server:       mesh-llm --client --join <token> blackboard --mcp
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
