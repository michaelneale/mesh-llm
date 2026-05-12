use anyhow::Result;
use clap::Parser;

use skippy_server::cli::{Cli, Command};
use skippy_server::{
    binary_transport::serve_binary, config::example_config, frontend::serve_openai, http::serve,
};

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve(args) => serve(args).await,
        Command::ServeBinary(args) => serve_binary(args).await,
        Command::ServeOpenAi(args) => serve_openai(args).await,
        Command::ExampleConfig => {
            println!("{}", serde_json::to_string_pretty(&example_config())?);
            Ok(())
        }
    }
}
