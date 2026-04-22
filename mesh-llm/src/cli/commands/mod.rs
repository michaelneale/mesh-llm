mod auth;
mod benchmark;
mod blackboard;
mod discover;
mod download;
mod gpus;
mod integrations;
mod models;
mod moe;
mod plugin;
mod runtime;
mod update;

use anyhow::Result;

use crate::cli::commands::benchmark::dispatch_benchmark_command;
use crate::cli::commands::blackboard::{install_skill, run_blackboard};
use crate::cli::commands::discover::{run_discover, run_stop};
use crate::cli::commands::download::dispatch_download_command;
use crate::cli::commands::gpus::dispatch_gpu_command;
use crate::cli::commands::integrations::{run_claude, run_goose, run_opencode};
use crate::cli::commands::models::dispatch_models_command;
use crate::cli::commands::moe::dispatch_moe_command;
use crate::cli::commands::plugin::run_plugin_command;
use crate::cli::commands::runtime::{dispatch_runtime_command, run_drop, run_load, run_status};
use crate::cli::commands::update::run_update;
use crate::cli::{AuthCommand, Cli, Command};
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
        Command::Update { .. } => run_update(cli).await,
        Command::Gpus { json, command } => {
            dispatch_gpu_command(*json, command.as_ref())?;
            Ok(())
        }
        Command::Moe { command } => dispatch_moe_command(command, cli).await,
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
        Command::RotateKey => nostr::rotate_keys(),
        Command::Goose { model, port } => run_goose(model.clone(), *port).await,
        Command::Claude { model, port } => run_claude(model.clone(), *port).await,
        Command::Opencode { model, port, write } => {
            run_opencode(model.clone(), *port, *write).await
        }
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
        Command::Auth { command } => match command {
            AuthCommand::Init {
                owner_key,
                force,
                no_passphrase,
                keychain,
            } => auth::run_init(owner_key.clone(), *force, *no_passphrase, *keychain),
            AuthCommand::Status {
                owner_key,
                node_key,
                node_ownership,
                trust_store,
            } => auth::run_status(
                owner_key.clone(),
                node_key.clone(),
                node_ownership.clone(),
                trust_store.clone(),
            ),
            AuthCommand::SignNode {
                owner_key,
                node_key,
                out,
                hostname_hint,
                node_label,
                expires_in_hours,
            } => auth::run_sign_node(
                owner_key.clone(),
                node_key.clone(),
                out.clone(),
                node_label.clone(),
                hostname_hint.clone(),
                *expires_in_hours,
            ),
            AuthCommand::RenewNode {
                owner_key,
                node_key,
                out,
                hostname_hint,
                node_label,
                expires_in_hours,
            } => auth::run_renew_node(
                owner_key.clone(),
                node_key.clone(),
                out.clone(),
                node_label.clone(),
                hostname_hint.clone(),
                *expires_in_hours,
            ),
            AuthCommand::VerifyNode {
                file,
                node_id,
                trust_store,
                trust_policy,
            } => auth::run_verify_node(
                file.clone(),
                node_id.clone(),
                trust_store.clone(),
                *trust_policy,
            ),
            AuthCommand::RotateNode {
                owner_key,
                node_key,
                out,
                hostname_hint,
                node_label,
                expires_in_hours,
                revoke_current,
                reason,
                trust_store,
            } => auth::run_rotate_node(
                owner_key.clone(),
                node_key.clone(),
                out.clone(),
                node_label.clone(),
                hostname_hint.clone(),
                *expires_in_hours,
                *revoke_current,
                reason.clone(),
                trust_store.clone(),
            ),
            AuthCommand::RevokeOwner {
                owner_id,
                reason,
                trust_store,
            } => auth::run_revoke_owner(owner_id.clone(), reason.clone(), trust_store.clone()),
            AuthCommand::RevokeNode {
                cert_id,
                node_id,
                reason,
                trust_store,
            } => auth::run_revoke_node(
                cert_id.clone(),
                node_id.clone(),
                reason.clone(),
                trust_store.clone(),
            ),
            AuthCommand::RotateOwner {
                owner_key,
                no_passphrase,
                force,
            } => auth::run_rotate_owner(owner_key.clone(), *no_passphrase, *force),
            AuthCommand::Trust { command } => auth::run_trust_command(command),
        },
    }?;
    Ok(true)
}
