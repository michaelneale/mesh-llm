use anyhow::Result;

use crate::cli::NetworkCommand;

pub(crate) async fn dispatch_network_command(command: &NetworkCommand) -> Result<()> {
    match command {
        NetworkCommand::Doctor {
            bind_port,
            relay,
            json,
        } => run_network_doctor(*bind_port, relay, *json).await,
    }
}

async fn run_network_doctor(
    bind_port: Option<u16>,
    relay_urls: &[String],
    json: bool,
) -> Result<()> {
    let report = crate::mesh::network_doctor(bind_port, relay_urls).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", crate::mesh::format_network_doctor(&report));
    }
    Ok(())
}
