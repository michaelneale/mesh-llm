#![recursion_limit = "256"]

fn main() {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();

    // Allow CI and developers to constrain the Tokio worker thread stack size
    // so that oversized spawned futures are caught before they overflow in
    // production. The default Tokio worker stack is 2 MB; with Box::pin on
    // large spawned futures, workers should need far less.
    if let Ok(val) = std::env::var("MESH_TOKIO_STACK_SIZE") {
        if let Ok(size) = val.parse::<usize>() {
            builder.thread_stack_size(size);
        }
    }

    let runtime = builder.build().expect("build tokio runtime");
    std::process::exit(runtime.block_on(mesh_llm::run_main()));
}
