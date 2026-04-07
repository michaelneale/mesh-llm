//! Native MLX inference backend.
//!
//! Runs quantized safetensors models directly on Apple Silicon Metal
//! via mlx-rs (Rust bindings to MLX C). No Python, no subprocess.
//!
//! This module provides `start_mlx_server()` as a drop-in replacement
//! for `launch::start_llama_server()` — same contract (port + death_rx),
//! so the proxy and election machinery work unchanged.

pub mod model;
pub mod sampling;
pub mod server;
pub mod template;

pub use model::{is_mlx_model_dir, mlx_model_dir};
pub use server::start_mlx_server;
