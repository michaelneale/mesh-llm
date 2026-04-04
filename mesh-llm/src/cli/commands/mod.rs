mod benchmark;
mod blackboard;
mod discover;
mod download;
mod integrations;
mod models;
mod plugin;
mod runtime;
mod update;

use anyhow::Result;

use crate::cli::commands::benchmark::dispatch_benchmark_command;
use crate::cli::commands::blackboard::{install_skill, run_blackboard};
use crate::cli::commands::discover::{run_discover, run_stop};
use crate::cli::commands::download::dispatch_download_command;
use crate::cli::commands::integrations::{run_claude, run_goose};
use crate::cli::commands::models::dispatch_models_command;
use crate::cli::commands::plugin::run_plugin_command;
use crate::cli::commands::runtime::{dispatch_runtime_command, run_drop, run_load, run_status};
use crate::cli::commands::update::run_update;
use crate::cli::{Cli, Command};
use crate::network::nostr;

pub(crate) async fn dispatch(cli: &Cli) -> Result<bool> {
    let Some(cmd) = cli.command.as_ref() else {
        return Ok(false);
    };
    match cmd {
        Command::Models { command } => {
            dispatch_models_command(command).await?;
            Ok(())
        }
        Command::Download { name, draft } => {
            dispatch_download_command(name.as_deref(), *draft).await
        }
        Command::Update => run_update(cli).await,
        Command::Runtime { command } => dispatch_runtime_command(command.as_ref()).await,
        Command::Load { name, port } => run_load(name, *port).await,
        Command::Unload { name, port } => run_drop(name, *port).await,
        Command::Status { port } => run_status(*port).await,
        Command::Stop => run_stop(),
        Command::Discover {
            model,
            min_vram,
            region,
            auto,
            relay,
        } => {
            run_discover(
                model.clone(),
                *min_vram,
                region.clone(),
                *auto,
                relay.clone(),
            )
            .await
        }
        Command::RotateKey => nostr::rotate_keys().map_err(Into::into),
        Command::Goose { model, port } => run_goose(model.clone(), *port).await,
        Command::Claude { model, port } => run_claude(model.clone(), *port).await,
        Command::Blackboard {
            text,
            search,
            from,
            since,
            limit,
            port,
            mcp,
        } => {
            if *mcp {
                crate::runtime::run_plugin_mcp(cli).await
            } else if text.as_deref() == Some("install-skill") {
                install_skill()
            } else {
                run_blackboard(
                    text.clone(),
                    search.clone(),
                    from.clone(),
                    *since,
                    *limit,
                    *port,
                )
                .await
            }
        }
        Command::Plugin { command } => run_plugin_command(command, cli).await,
        Command::Benchmark { command } => dispatch_benchmark_command(command).await,
    }?;
    Ok(true)
}
