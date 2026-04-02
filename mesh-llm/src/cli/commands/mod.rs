use anyhow::Result;

use crate::app;
use crate::cli::models::dispatch_models_command;
use crate::cli::runtime::{run_drop, run_load, run_status, RuntimeCommand};
use crate::cli::{Cli, Command};
use crate::{models, nostr};

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
                    let model = models::catalog::find_model(query).ok_or_else(|| {
                        anyhow::anyhow!(
                            "No model matching '{}' in catalog. Run `mesh-llm download` to list.",
                            query
                        )
                    })?;
                    models::catalog::download_model(model).await?;
                    if *draft {
                        if let Some(draft_name) = model.draft.as_deref() {
                            let draft_model =
                                models::catalog::find_model(draft_name).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Draft model '{}' not found in catalog",
                                        draft_name
                                    )
                                })?;
                            models::catalog::download_model(draft_model).await?;
                        } else {
                            eprintln!("⚠ No draft model available for {}", model.name);
                        }
                    }
                }
                None => models::catalog::list_models(),
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
        Command::Stop => app::run_stop(),
        Command::Discover {
            model,
            min_vram,
            region,
            auto,
            relay,
        } => {
            app::run_discover(
                model.clone(),
                *min_vram,
                region.clone(),
                *auto,
                relay.clone(),
            )
            .await
        }
        Command::RotateKey => nostr::rotate_keys().map_err(Into::into),
        Command::Goose { model, port } => app::run_goose(model.clone(), *port).await,
        Command::Claude { model, port } => app::run_claude(model.clone(), *port).await,
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
                app::run_plugin_mcp(cli).await
            } else if text.as_deref() == Some("install-skill") {
                app::install_skill()
            } else {
                app::run_blackboard(
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
        Command::Plugin { command } => app::run_plugin_command(command, cli).await,
    }?;
    Ok(true)
}
