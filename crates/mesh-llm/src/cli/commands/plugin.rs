use anyhow::Result;

use crate::cli::{Cli, PluginCommand};
use crate::plugin;
use crate::runtime;

pub(crate) async fn run_plugin_command(command: &PluginCommand, cli: &Cli) -> Result<()> {
    match command {
        PluginCommand::Install { name } if name == plugin::BLACKBOARD_PLUGIN_ID => {
            eprintln!("Blackboard is auto-registered by mesh-llm. Nothing to install.");
            eprintln!("Disable it with [[plugin]] name = \"blackboard\" enabled = false in the config if needed.");
        }
        PluginCommand::Install { name } if name == plugin::BLOBSTORE_PLUGIN_ID => {
            eprintln!("Blobstore is auto-registered by mesh-llm. Nothing to install.");
            eprintln!("Disable it with [[plugin]] name = \"blobstore\" enabled = false in the config if needed.");
        }
        PluginCommand::Install { name } => {
            let config = plugin::config_path(cli.config.as_deref())?;
            anyhow::bail!(
                "Plugins are configured as executables in {}. No install step exists for '{}'.",
                config.display(),
                name
            );
        }
        PluginCommand::List => {
            let resolved = runtime::load_resolved_plugins(cli)?;
            for spec in resolved.externals {
                println!(
                    "{}\tkind=external\tcommand={}\targs={}",
                    spec.name,
                    spec.command,
                    spec.args.join(" ")
                );
            }
        }
    }
    Ok(())
}
