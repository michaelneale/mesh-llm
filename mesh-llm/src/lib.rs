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
mod system;

pub mod proto {
    pub mod node {
        include!(concat!(env!("OUT_DIR"), "/meshllm.node.v1.rs"));
    }
}

pub(crate) use plugins::blackboard;

use anyhow::Result;

pub const VERSION: &str = "0.60.0-rc.1";

pub async fn run() -> Result<()> {
    runtime::run().await
}
