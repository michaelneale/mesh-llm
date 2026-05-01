use mesh_client::runtime::CoreRuntime;
use std::time::Duration;

fn thread_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "kern.num_threads"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .unwrap_or(0)
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if line.starts_with("Threads:") {
                return line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

#[test]
fn runtime_100_create_drop_no_thread_leak() {
    let before = thread_count();
    for _ in 0..100 {
        let rt = CoreRuntime::new().expect("create runtime");
        drop(rt);
    }
    let after = thread_count();
    let diff = (after as i64 - before as i64).abs();
    println!("threads_before={before} threads_after={after} diff={diff}");
    assert!(
        diff <= 3,
        "thread leak detected: before={before} after={after}"
    );
}

#[test]
fn runtime_drop_from_tokio_task_no_panic() {
    let rt = CoreRuntime::new().expect("create runtime");
    let handle = rt.handle().clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.block_on(async {
            drop(rt);
        });
    }));
    assert!(result.is_ok(), "drop from tokio task panicked");
}

#[test]
fn runtime_pending_tasks_shutdown_within_5s() {
    let rt = CoreRuntime::new().expect("create runtime");
    rt.handle().spawn(async {
        tokio::time::sleep(Duration::from_secs(100)).await;
    });
    let start = std::time::Instant::now();
    drop(rt);
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(6),
        "shutdown took too long: {elapsed:?}"
    );
}
