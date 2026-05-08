use anyhow::{Context, Result};
use std::io::Write;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

pub(crate) fn clear_stderr_line() -> Result<()> {
    if crate::cli::output::json_mode_enabled() {
        return Ok(());
    }
    eprint!("\r\x1b[2K");
    std::io::stderr()
        .flush()
        .context("Flush terminal progress clear")?;
    Ok(())
}

pub(crate) struct SpinnerHandle {
    done: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl SpinnerHandle {
    pub(crate) fn finish(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let _ = clear_stderr_line();
    }
}

impl Drop for SpinnerHandle {
    fn drop(&mut self) {
        self.finish();
    }
}

pub(crate) fn start_spinner(message: &str) -> SpinnerHandle {
    if crate::cli::output::json_mode_enabled() {
        return SpinnerHandle {
            done: Arc::new(AtomicBool::new(true)),
            thread: None,
        };
    }
    let done = Arc::new(AtomicBool::new(false));
    let done_thread = Arc::clone(&done);
    let message = Arc::new(Mutex::new(message.to_string()));
    let message_thread = Arc::clone(&message);
    let thread = thread::spawn(move || {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut index = 0usize;
        while !done_thread.load(Ordering::Relaxed) {
            let current = message_thread
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_else(|_| "Working".to_string());
            eprint!("\r\x1b[2K{} {}", frames[index % frames.len()], current);
            let _ = std::io::stderr().flush();
            index += 1;
            thread::sleep(Duration::from_millis(120));
        }
    });
    SpinnerHandle {
        done,
        thread: Some(thread),
    }
}

pub(crate) struct DeterminateProgressLine {
    prefix: String,
}

impl DeterminateProgressLine {
    pub(crate) fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    pub(crate) fn draw_counts(
        &self,
        label: &str,
        current: usize,
        total: usize,
        detail: Option<&str>,
    ) -> Result<()> {
        if crate::cli::output::json_mode_enabled() {
            return Ok(());
        }
        let percent = if total > 0 {
            (current as f64 / total as f64) * 100.0
        } else {
            100.0
        };
        let detail = detail.unwrap_or("");
        eprint!(
            "\r\x1b[2K{} {} {:>5.1}% [{}/{}]{}",
            self.prefix, label, percent, current, total, detail
        );
        std::io::stderr()
            .flush()
            .context("Flush determinate progress")?;
        Ok(())
    }
}
