use anyhow::{Context, Result};

use crate::cli::runtime::RuntimeCommand;

pub(crate) async fn dispatch_runtime_command(command: Option<&RuntimeCommand>) -> Result<()> {
    match command {
        Some(RuntimeCommand::Status { port }) => run_status(*port).await,
        Some(RuntimeCommand::Load { name, port }) => run_load(name, *port).await,
        Some(RuntimeCommand::Unload { name, port }) => run_drop(name, *port).await,
        None => run_status(3131).await,
    }
}

pub(crate) async fn run_drop(model_name: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let encoded = percent_encode_path_segment(model_name);
    let url = format!("http://127.0.0.1:{port}/api/runtime/models/{encoded}");
    let resp = client
        .delete(&url)
        .send()
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?;
    display_runtime_result(resp, model_name, "Unloaded").await
}

pub(crate) async fn run_load(model_name: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/models");
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"model": model_name}))
        .send()
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?;
    display_runtime_result(resp, model_name, "Loaded").await
}

async fn display_runtime_result(
    resp: reqwest::Response,
    model_name: &str,
    verb: &str,
) -> Result<()> {
    let action_inf = if verb == "Loaded" { "load" } else { "unload" };
    let is_success = resp.status().is_success();
    let body = resp.json::<serde_json::Value>().await.ok();
    if is_success {
        for line in runtime_success_lines(model_name, verb, body.as_ref()) {
            eprintln!("{line}");
        }
    } else {
        eprintln!("❌ Failed to {action_inf} runtime model");
        eprintln!();
        eprintln!("Model: {model_name}");
        let reason = body
            .as_ref()
            .and_then(|value| value["error"].as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown error".to_string());
        eprintln!("Reason: {reason}");
    }
    Ok(())
}

fn runtime_success_lines(
    model_name: &str,
    verb: &str,
    body: Option<&serde_json::Value>,
) -> Vec<String> {
    let response_model_key = match verb {
        "Loaded" => "loaded",
        "Unloaded" => "dropped",
        _ => "model",
    };
    let display_model = body
        .and_then(|value| value[response_model_key].as_str())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(model_name);
    let instance_id = body
        .and_then(|value| value["instance_id"].as_str())
        .filter(|value| !value.trim().is_empty());

    let mut lines = vec![
        format!("✅ {verb} runtime model"),
        String::new(),
        format!("Model: {display_model}"),
    ];
    if let Some(instance_id) = instance_id {
        lines.push(format!("Instance: {instance_id}"));
    }
    lines.push("Scope: Local node".to_string());
    lines
}

/// Percent-encode a string for use as a URL path segment.
/// Unreserved characters (A-Z a-z 0-9 - _ . ~) are passed through unchanged;
/// all other bytes are encoded as %XX.
fn percent_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b => {
                out.push('%');
                out.push(
                    char::from_digit((b >> 4) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
                out.push(
                    char::from_digit((b & 0xf) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    out
}

pub(crate) async fn run_status(port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let runtime_body = fetch_runtime_payload(&client, port, "/api/runtime").await?;
    let processes_body = fetch_runtime_payload(&client, port, "/api/runtime/processes").await?;

    let models = runtime_body["models"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid runtime status payload"))?;
    let processes = processes_body["processes"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid runtime process payload"))?;

    println!("⚙️  Runtime");
    println!();

    if models.is_empty() {
        println!("📦 Models served locally: 0");
        println!();
        println!("No local models are currently being served.");
        return Ok(());
    }

    println!("📦 Models served locally: {}", models.len());
    println!();

    println!(
        "{:<42} {:<12} {:<8} {:<10} {:<8} {:<6}",
        "Model", "Instance", "Backend", "State", "Pid", "Port"
    );
    for model in models {
        let name = model["name"].as_str().unwrap_or("unknown");
        let instance = model["instance_id"].as_str().unwrap_or("-");
        let backend = display_backend_label(model["backend"].as_str().unwrap_or("unknown"));
        let status = display_runtime_state(model["status"].as_str().unwrap_or("unknown"));
        let pid = find_pid(processes, model)
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        let port = model["port"]
            .as_u64()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<42} {:<12} {:<8} {:<10} {:<8} {:<6}",
            name, instance, backend, status, pid, port
        );
    }

    Ok(())
}

fn display_runtime_state(value: &str) -> &'static str {
    match value {
        "ready" => "Ready",
        "starting" => "Starting",
        "stopped" => "Stopped",
        _ => "Unknown",
    }
}

fn display_backend_label(value: &str) -> &'static str {
    match value {
        "llama" => "Llama",
        _ => "Unknown",
    }
}

async fn fetch_runtime_payload(
    client: &reqwest::Client,
    port: u16,
    path: &str,
) -> Result<serde_json::Value> {
    let url = format!("http://127.0.0.1:{port}{path}");
    client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!("Can't connect to mesh-llm console on port {port}. Is it running?")
        })?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await
        .map_err(Into::into)
}

fn find_pid(processes: &[serde_json::Value], model: &serde_json::Value) -> Option<u64> {
    let name = model["name"].as_str()?;
    let instance_id = model["instance_id"].as_str();
    let port = model["port"].as_u64();
    processes
        .iter()
        .find(|process| {
            if let Some(instance_id) = instance_id {
                return process["instance_id"].as_str() == Some(instance_id);
            }
            process["name"].as_str() == Some(name)
                && port
                    .map(|port| process["port"].as_u64() == Some(port))
                    .unwrap_or(true)
        })
        .and_then(|process| process["pid"].as_u64())
}

#[cfg(test)]
mod tests {
    use super::runtime_success_lines;
    use serde_json::json;

    #[test]
    fn runtime_success_lines_print_loaded_instance_id() {
        let body = json!({
            "loaded": "Qwen3-8B",
            "instance_id": "runtime-2",
        });

        assert_eq!(
            runtime_success_lines("fallback", "Loaded", Some(&body)),
            vec![
                "✅ Loaded runtime model".to_string(),
                String::new(),
                "Model: Qwen3-8B".to_string(),
                "Instance: runtime-2".to_string(),
                "Scope: Local node".to_string(),
            ]
        );
    }

    #[test]
    fn runtime_success_lines_print_unloaded_instance_id() {
        let body = json!({
            "dropped": "Qwen3-8B",
            "instance_id": "runtime-2",
        });

        assert_eq!(
            runtime_success_lines("fallback", "Unloaded", Some(&body)),
            vec![
                "✅ Unloaded runtime model".to_string(),
                String::new(),
                "Model: Qwen3-8B".to_string(),
                "Instance: runtime-2".to_string(),
                "Scope: Local node".to_string(),
            ]
        );
    }

    #[test]
    fn runtime_success_lines_omit_missing_instance_id() {
        assert_eq!(
            runtime_success_lines("Qwen3-8B", "Loaded", None),
            vec![
                "✅ Loaded runtime model".to_string(),
                String::new(),
                "Model: Qwen3-8B".to_string(),
                "Scope: Local node".to_string(),
            ]
        );
    }
}
