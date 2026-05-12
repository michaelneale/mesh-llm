use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub(crate) enum RuntimeCommand {
    /// Show local model status on a running mesh-llm instance.
    Status {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Show the local-only owner-control bootstrap policy for a running mesh-llm instance.
    Bootstrap {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
    /// Fetch config from a remote owner-control endpoint through the local management API.
    GetConfig {
        /// Explicit owner-control endpoint token for the target node.
        #[arg(long)]
        endpoint: String,
        /// Console/API port of the local mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
    /// Refresh local inventory on a remote owner-control endpoint through the local management API.
    RefreshInventory {
        /// Explicit owner-control endpoint token for the target node.
        #[arg(long)]
        endpoint: String,
        /// Console/API port of the local mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
    /// Apply config to a remote owner-control endpoint through the local management API.
    ApplyConfig {
        /// Explicit owner-control endpoint token for the target node.
        #[arg(long)]
        endpoint: String,
        /// Expected remote config revision for CAS.
        #[arg(long)]
        expected_revision: u64,
        /// TOML config file to apply remotely.
        #[arg(long)]
        config: PathBuf,
        /// Console/API port of the local mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
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
}
