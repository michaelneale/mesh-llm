use anyhow::Result;

use crate::cli::Cli;
use crate::cli::Command;
use crate::system::autoupdate;

pub async fn run_update(cli: &Cli) -> Result<()> {
    let requested_version = match &cli.command {
        Some(Command::Update { version }) => version.as_deref(),
        _ => None,
    };
    autoupdate::run_update_command(autoupdate::UpdateCommandOptions {
        llama_flavor: cli.llama_flavor,
        requested_version,
        current_version: crate::VERSION,
    })
    .await
}
