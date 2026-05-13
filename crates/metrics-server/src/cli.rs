use std::{net::SocketAddr, path::PathBuf};

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Skippy staged-runtime benchmark telemetry service")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    Serve(ServeArgs),
    EmitFixture(EmitFixtureArgs),
}

#[derive(Args)]
pub(crate) struct ServeArgs {
    #[arg(long, default_value = "metrics.sqlite")]
    pub(crate) db: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub(crate) http_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:4317")]
    pub(crate) otlp_grpc_addr: SocketAddr,
    #[arg(long)]
    pub(crate) debug_retain_raw_otlp: bool,
}

#[derive(Args)]
pub(crate) struct EmitFixtureArgs {
    #[arg(long, default_value = "http://127.0.0.1:4317")]
    pub(crate) otlp_grpc_addr: String,
    #[arg(long)]
    pub(crate) run_id: String,
    #[arg(long, default_value = "fixture-request-1")]
    pub(crate) request_id: String,
    #[arg(long, default_value = "fixture-session-1")]
    pub(crate) session_id: String,
    #[arg(long, default_value = "stage-0")]
    pub(crate) stage_id: String,
}
