mod output;
mod runner;

#[cfg(any(feature = "cuda", feature = "hip", feature = "intel"))]
mod capture;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "hip")]
mod hip;

#[cfg(feature = "intel")]
mod intel;

#[cfg(target_os = "macos")]
mod metal;

pub use output::BenchmarkOutput;
pub use runner::{
    parse_benchmark_output, run_benchmark, runner_for, BenchmarkBackend, BenchmarkRunner,
};
