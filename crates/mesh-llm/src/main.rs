#![recursion_limit = "256"]

/// Default Tokio worker thread stack size: 8 MB.
///
/// The standard Tokio default is 2 MB, which is too small for several spawned
/// futures in this codebase (startup_local_model_loop, api_proxy,
/// nostr_rediscovery, publish_loop, handle_mesh_request). Box::pin on the
/// largest spawn sites keeps most futures on the heap, but 8 MB provides a
/// safety margin for any remaining large futures or deep call chains. This
/// matches the macOS main-thread default.
///
/// Override with MESH_TOKIO_STACK_SIZE for CI or debugging (e.g. set to
/// 1048576 to catch regressions with a 1 MB clamp).
const DEFAULT_WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;

fn main() {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();

    let stack_size = std::env::var("MESH_TOKIO_STACK_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WORKER_STACK_SIZE);
    builder.thread_stack_size(stack_size);

    let runtime = builder.build().expect("build tokio runtime");
    std::process::exit(runtime.block_on(mesh_llm::run_main()));
}
