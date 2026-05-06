use anyhow::Result;

use crate::cli::Cli;
use crate::system::autoupdate;

pub async fn run_update(cli: &Cli) -> Result<()> {
    autoupdate::run_update_command(cli).await
}
