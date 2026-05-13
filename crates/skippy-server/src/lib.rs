//! skippy-server library interface.
//!
//! Exposes the stage serving loop for in-process embedding by mesh-llm
//! or other host runtimes.

pub mod binary_transport;
pub mod cli;
pub mod config;
pub mod embedded;
pub mod frontend;
pub mod http;
pub mod kv_integration;
pub mod kv_proto;
pub mod package;
pub mod runtime_state;
pub mod telemetry;

// Re-export key types for consumers
pub use binary_transport::serve_binary;
pub use cli::ServeBinaryArgs;
pub use embedded::{
    start_binary_stage, start_embedded_openai, start_openai_backend, start_stage_http,
    EmbeddedRuntimeOptions, EmbeddedRuntimeStatus, EmbeddedServerHandle, EmbeddedServerStatus,
    EmbeddedState, SkippyRuntimeHandle,
};
pub use frontend::{
    embedded_openai_backend, EmbeddedOpenAiArgs, EmbeddedOpenAiBackend, CONTEXT_BUDGET_MAX_TOKENS,
};
pub use skippy_protocol::StageConfig;
