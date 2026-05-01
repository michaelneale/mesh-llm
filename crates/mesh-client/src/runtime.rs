//! Dedicated-thread tokio runtime for mesh-client.
//!
//! The tokio runtime is owned by a dedicated OS thread so that its `Drop`
//! never executes inside a tokio context. The public `CoreRuntime` holds only
//! a `Handle` plus a shutdown channel to the owning thread, which means
//! dropping `CoreRuntime` from arbitrary call sites (including from inside a
//! tokio task) only signals the owning thread and joins it — the real
//! `tokio::runtime::Runtime::drop` always runs on the dedicated thread, which
//! is itself not inside a tokio context.

use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("failed to spawn dedicated tokio thread")]
    ThreadSpawnFailed,
    #[error("failed to receive runtime handle from dedicated thread")]
    HandleRecvFailed,
}

pub struct CoreRuntime {
    handle: tokio::runtime::Handle,
    shutdown_tx: std::sync::mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CoreRuntime {
    pub fn new() -> Result<Self, RuntimeError> {
        let (handle_tx, handle_rx) = std::sync::mpsc::channel();
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("mesh-client-tokio".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .worker_threads(2)
                    .thread_name("mesh-core-worker")
                    .build()
                    .expect("tokio runtime build");
                handle_tx.send(rt.handle().clone()).expect("send handle");
                let _ = shutdown_rx.recv();
                rt.shutdown_timeout(Duration::from_secs(5));
            })
            .map_err(|_| RuntimeError::ThreadSpawnFailed)?;
        let handle = handle_rx
            .recv()
            .map_err(|_| RuntimeError::HandleRecvFailed)?;
        Ok(Self {
            handle,
            shutdown_tx,
            thread: Some(thread),
        })
    }

    pub fn handle(&self) -> &tokio::runtime::Handle {
        &self.handle
    }
}

impl Drop for CoreRuntime {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
