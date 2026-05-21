#![recursion_limit = "256"]

mod api;
mod cli;
pub mod crypto;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
mod plugins;
mod protocol;
mod runtime;
mod runtime_data;
mod system;

pub mod proto {
    pub use mesh_llm_protocol::proto::*;
}

pub(crate) use plugins::blackboard;

use anyhow::Result;
use std::time::Duration;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn run() -> Result<()> {
    runtime::run().await
}

pub async fn run_main() -> i32 {
    match run().await {
        Ok(()) => 0,
        Err(err) => {
            let _ = cli::output::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}

#[cfg(test)]
include!("exact_test_wrappers.rs");
