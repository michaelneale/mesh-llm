#![forbid(unsafe_code)]

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("mesh-llm binary not found at {0}")]
    BinaryNotFound(PathBuf),
    #[error("mesh-llm startup timed out after 60s")]
    StartupTimeout,
    #[error("invalid /api/status response: {0}")]
    InvalidStatusResponse(String),
    #[error("model load failed: {0}")]
    ModelLoadFailed(String),
}

pub struct FixtureMesh {
    invite_token: String,
    child: Child,
    _port: u16,
}

impl FixtureMesh {
    pub fn new(model: &str) -> Result<Self, FixtureError> {
        let binary = find_mesh_llm_binary()?;
        let port = pick_port();
        let child = Command::new(&binary)
            .args(["serve", "--model", model, "--port", &port.to_string()])
            .spawn()
            .map_err(|e| FixtureError::ModelLoadFailed(e.to_string()))?;
        let invite_token = wait_for_status(port, Duration::from_secs(60))?;
        Ok(Self {
            invite_token,
            child,
            _port: port,
        })
    }

    pub fn invite_token(&self) -> &str {
        &self.invite_token
    }
}

impl Drop for FixtureMesh {
    fn drop(&mut self) {
        // Forcefully terminate the child process, then wait up to 5s for it to exit.
        let _ = self.child.kill();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                _ => break,
            }
        }
    }
}

fn find_mesh_llm_binary() -> Result<PathBuf, FixtureError> {
    // Check $MESH_LLM_BIN env var first, then target/release/mesh-llm
    if let Ok(bin) = std::env::var("MESH_LLM_BIN") {
        let path = PathBuf::from(bin);
        if path.exists() {
            return Ok(path);
        }
    }
    // Walk up from current dir to find workspace root
    let mut dir = std::env::current_dir().unwrap_or_default();
    for _ in 0..5 {
        let candidate = dir.join("target/release/mesh-llm");
        if candidate.exists() {
            return Ok(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    Err(FixtureError::BinaryNotFound(PathBuf::from(
        "target/release/mesh-llm",
    )))
}

fn pick_port() -> u16 {
    // Bind to port 0 to get an OS-assigned port, then release it
    TcpListener::bind("127.0.0.1:0")
        .map(|l| l.local_addr().unwrap().port())
        .unwrap_or(19337)
}

fn wait_for_status(port: u16, timeout: Duration) -> Result<String, FixtureError> {
    let url = format!("http://127.0.0.1:{}/api/status", port);
    let deadline = Instant::now() + timeout;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| FixtureError::InvalidStatusResponse(e.to_string()))?;
    loop {
        if Instant::now() > deadline {
            return Err(FixtureError::StartupTimeout);
        }
        match client.get(&url).send() {
            Ok(resp) if resp.status().is_success() => {
                let body: serde_json::Value = resp
                    .json()
                    .map_err(|e| FixtureError::InvalidStatusResponse(e.to_string()))?;
                let token = body["token"]
                    .as_str()
                    .ok_or_else(|| FixtureError::InvalidStatusResponse("missing token".into()))?
                    .to_string();
                return Ok(token);
            }
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    }
}
