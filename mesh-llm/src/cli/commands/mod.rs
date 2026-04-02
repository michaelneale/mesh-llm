mod blackboard;
mod discover;
mod integrations;
mod models;
mod plugin;

use anyhow::Result;

use crate::cli::commands::blackboard::{install_skill, run_blackboard};
use crate::cli::commands::discover::{run_discover, run_stop};
use crate::cli::commands::integrations::{run_claude, run_goose};
use crate::cli::commands::models::dispatch_models_command;
use crate::cli::commands::plugin::run_plugin_command;
use crate::cli::runtime::{run_drop, run_load, run_status, RuntimeCommand};
use crate::cli::{Cli, Command};
use crate::{nostr, runtime};

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
            match name {
                Some(query) => {
                    let model = crate::models::catalog::find_model(query).ok_or_else(|| {
                        anyhow::anyhow!(
                            "No model matching '{}' in catalog. Run `mesh-llm download` to list.",
                            query
                        )
                    })?;
                    crate::models::catalog::download_model(model).await?;
                    if *draft {
                        if let Some(draft_name) = model.draft.as_deref() {
                            let draft_model = crate::models::catalog::find_model(draft_name)
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Draft model '{}' not found in catalog",
                                        draft_name
                                    )
                                })?;
                            crate::models::catalog::download_model(draft_model).await?;
                        } else {
                            eprintln!("⚠ No draft model available for {}", model.name);
                        }
                    }
                }
                None => crate::models::catalog::list_models(),
            }
            Ok(())
        }
        Command::Runtime { command } => match command {
            Some(RuntimeCommand::Status { port }) => run_status(*port).await,
            Some(RuntimeCommand::Load { name, port }) => run_load(name, *port).await,
            Some(RuntimeCommand::Unload { name, port }) => run_drop(name, *port).await,
            None => run_status(3131).await,
        },
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
                runtime::run_plugin_mcp(cli).await
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
    }?;
    Ok(true)
}
