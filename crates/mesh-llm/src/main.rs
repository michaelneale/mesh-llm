#![recursion_limit = "256"]

const TOKIO_THREAD_STACK_SIZE_BYTES: usize = 64 * 1024 * 1024;

fn main() {
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        std::env::set_var("RUST_MIN_STACK", TOKIO_THREAD_STACK_SIZE_BYTES.to_string());
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(TOKIO_THREAD_STACK_SIZE_BYTES)
        .build()
        .expect("build mesh-llm tokio runtime");

    std::process::exit(runtime.block_on(mesh_llm::run_main()));
}
