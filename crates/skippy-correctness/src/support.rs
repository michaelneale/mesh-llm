use std::{
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use skippy_protocol::binary::{recv_ready, WireActivationDType};

pub struct ChildGuard {
    child: Child,
}

impl ChildGuard {
    pub fn spawn(mut command: Command) -> Result<Self> {
        let child = command
            .spawn()
            .with_context(|| format!("failed to spawn {:?}", command))?;
        Ok(Self { child })
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn connect_ready(addr: SocketAddr, timeout_secs: u64) -> Result<TcpStream> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match TcpStream::connect(addr) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                match recv_ready(&mut stream) {
                    Ok(()) => return Ok(stream),
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

pub fn activation_width(frame: &skippy_runtime::ActivationFrame) -> Result<i32> {
    if frame.desc.token_count == 0 {
        bail!("activation frame token_count is zero");
    }
    let bytes_per_token = frame
        .payload
        .len()
        .checked_div(frame.desc.token_count as usize)
        .context("activation token_count overflow")?;
    if !bytes_per_token.is_multiple_of(4) {
        bail!("activation payload is not F32 aligned");
    }
    i32::try_from(bytes_per_token / 4).context("activation width exceeds i32")
}

pub fn parse_wire_dtype(value: &str) -> Result<WireActivationDType> {
    match value {
        "fp32" | "f32" => Ok(WireActivationDType::F32),
        "fp16" | "f16" => Ok(WireActivationDType::F16),
        "q8" | "int8" | "i8" => Ok(WireActivationDType::Q8),
        _ => bail!("unsupported activation wire dtype {value}"),
    }
}

pub fn generate_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis();
    format!("correctness-{millis}")
}

pub fn temp_config_path_for(run_id: &str, stage_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{run_id}-{stage_id}.json"))
}
