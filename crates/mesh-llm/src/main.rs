#![recursion_limit = "256"]

fn main() {
    // Scrub dangerous env vars (DYLD_INSERT_LIBRARIES, LD_PRELOAD, etc.)
    // BEFORE the tokio runtime spawns worker threads. std::env::remove_var is
    // unsafe when other threads may be reading the environment concurrently.
    mesh_llm::scrub_env_pre_thread();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(async {
            std::process::exit(mesh_llm::run_main().await);
        });
}
