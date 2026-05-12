//! MoA gateway against mesh-llm or any OpenAI-compatible multi-model endpoint.
//!
//! Discovers available models from the endpoint, fans out to all of them,
//! and runs a comparative test.
//!
//! Usage:
//!   # Against local mesh-llm proxy (default)
//!   cargo run -p moa-gateway --bin moa-mesh
//!
//!   # Against ollama
//!   cargo run -p moa-gateway --bin moa-mesh -- --url http://localhost:11434/v1
//!
//!   # Against any OpenAI-compatible endpoint
//!   cargo run -p moa-gateway --bin moa-mesh -- --url http://gpu-box:8000/v1
//!
//! The gateway treats each discovered model as a separate endpoint at the
//! same base URL.  It fans out to all models in parallel, arbitrates, and
//! returns one response.  This works because mesh-llm's proxy routes by
//! the `model` field in the request body.

use moa_gateway::{Endpoint, Gateway, GatewayConfig};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "moa_gateway=info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let base_url = args
        .iter()
        .position(|a| a == "--url")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "http://localhost:9337/v1".to_string());

    println!("MoA Mesh Gateway");
    println!("Endpoint: {base_url}");
    println!();

    // Discover models
    let endpoints = match discover_models(&base_url).await {
        Ok(eps) => eps,
        Err(e) => {
            eprintln!("Failed to discover models at {base_url}: {e}");
            eprintln!();
            eprintln!("Start mesh-llm first:");
            eprintln!("  mesh-llm client --auto    # join public mesh");
            eprintln!("  mesh-llm serve --auto     # serve + join");
            eprintln!();
            eprintln!("Or point to ollama:");
            eprintln!(
                "  cargo run -p moa-gateway --bin moa-mesh -- --url http://localhost:11434/v1"
            );
            std::process::exit(1);
        }
    };

    if endpoints.is_empty() {
        eprintln!("No models found at {base_url}");
        std::process::exit(1);
    }

    println!(
        "Found {} models: [{}]",
        endpoints.len(),
        endpoints
            .iter()
            .map(|e| e.model.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();

    if endpoints.len() < 2 {
        eprintln!(
            "Need at least 2 models for MoA. Only found: {}",
            endpoints[0].model
        );
        std::process::exit(1);
    }

    // ── Run tests ────────────────────────────────────────────────

    let config = GatewayConfig {
        endpoints: endpoints.clone(),
        worker_timeout: Duration::from_secs(60),
        reducer_timeout: Duration::from_secs(90),
    };

    // Test 1: Factual — MoA vs each model solo
    println!("━━━ Test 1: Factual — MoA vs solo ━━━");
    println!();

    let q = json!({
        "messages": [{"role": "user", "content": "What year was the first iPhone released? One sentence."}]
    });

    // Solo
    let http = reqwest::Client::new();
    for ep in &endpoints {
        let start = Instant::now();
        let mut body = q.clone();
        body["model"] = json!(ep.model);
        body["max_tokens"] = json!(128);
        body["stream"] = json!(false);
        body["temperature"] = json!(0.3);

        match http
            .post(format!("{}/chat/completions", ep.base_url))
            .json(&body)
            .timeout(Duration::from_secs(60))
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(r) = resp.json::<Value>().await {
                    let content = r["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("[empty]");
                    println!(
                        "  Solo {} ({}ms): {}",
                        ep.model,
                        start.elapsed().as_millis(),
                        trunc(content, 200),
                    );
                }
            }
            Err(e) => println!("  Solo {} FAILED: {e}", ep.model),
        }
    }

    // MoA
    let mut gw = Gateway::new(config.clone());
    let start = Instant::now();
    let result = gw.turn(&q).await;
    let content = result.response_body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("[empty]");
    let ok_count = result
        .worker_summaries
        .iter()
        .filter(|w| w.succeeded)
        .count();
    println!(
        "  MoA ({}ms, {}/{} workers, reducer={}): {}",
        start.elapsed().as_millis(),
        ok_count,
        result.worker_summaries.len(),
        result.reducer_used,
        trunc(content, 200),
    );
    println!();

    // Test 2: Tool calling
    println!("━━━ Test 2: Tool calling ━━━");
    println!();

    let tool_q = json!({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": "Search for the latest news about AI regulation"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"]
                }
            }
        }]
    });

    let mut gw2 = Gateway::new(config.clone());
    let start = Instant::now();
    let result = gw2.turn(&tool_q).await;
    let msg = &result.response_body["choices"][0]["message"];
    let has_tc = msg
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    if has_tc {
        let tc = &msg["tool_calls"][0];
        println!(
            "  ✅ tool_call: {}({}) — {}ms, reducer={}",
            tc["function"]["name"],
            tc["function"]["arguments"],
            start.elapsed().as_millis(),
            result.reducer_used,
        );
    } else {
        let content = msg["content"].as_str().unwrap_or("[empty]");
        println!(
            "  ⚠️  Text answer (no tool call): {} — {}ms",
            trunc(content, 150),
            start.elapsed().as_millis(),
        );
    }
    for ws in &result.worker_summaries {
        println!(
            "    {} ({}): {:?} conf={:?} {}ms",
            ws.model,
            ws.role.label(),
            ws.output_kind,
            ws.confidence,
            ws.elapsed_ms,
        );
    }
    println!();

    // Test 3: Multi-turn with tool lifecycle
    println!("━━━ Test 3: Multi-turn tool lifecycle ━━━");
    println!();

    let mut gw3 = Gateway::new(config.clone());

    // Turn 1
    let t1 = json!({
        "messages": [
            {"role": "system", "content": "You are helpful. Use tools when needed."},
            {"role": "user", "content": "What's the weather in Sydney?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }]
    });

    let r1 = gw3.turn(&t1).await;
    let msg1 = &r1.response_body["choices"][0]["message"];
    let has_tc1 = msg1
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    if has_tc1 {
        let tc = &msg1["tool_calls"][0];
        let call_id = tc["id"].as_str().unwrap_or("call_1");
        println!(
            "  Turn 1: ✅ {}({}) — {}ms",
            tc["function"]["name"], tc["function"]["arguments"], r1.elapsed_ms,
        );

        // Turn 2: feed tool result
        let t2 = json!({
            "messages": [
                {"role": "system", "content": "You are helpful. Use tools when needed."},
                {"role": "user", "content": "What's the weather in Sydney?"},
                {"role": "assistant", "content": null, "tool_calls": [tc.clone()]},
                {"role": "tool", "tool_call_id": call_id, "content": "Sydney: 18°C, partly cloudy, humidity 65%"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
                }
            }]
        });

        let r2 = gw3.turn(&t2).await;
        let c2 = r2.response_body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[empty]");
        println!(
            "  Turn 2: {} — {}ms (reducer={})",
            trunc(c2, 200),
            r2.elapsed_ms,
            r2.reducer_used
        );

        // Turn 3: follow-up using context (should know we asked about Sydney)
        let t3 = json!({
            "messages": [
                {"role": "system", "content": "You are helpful. Use tools when needed."},
                {"role": "user", "content": "What's the weather in Sydney?"},
                {"role": "assistant", "content": null, "tool_calls": [tc.clone()]},
                {"role": "tool", "tool_call_id": call_id, "content": "Sydney: 18°C, partly cloudy, humidity 65%"},
                {"role": "assistant", "content": c2},
                {"role": "user", "content": "Should I bring a jacket?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
                }
            }]
        });

        let r3 = gw3.turn(&t3).await;
        let c3 = r3.response_body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[empty]");
        let contextual = c3.to_lowercase().contains("18")
            || c3.to_lowercase().contains("cool")
            || c3.to_lowercase().contains("jacket")
            || c3.to_lowercase().contains("sydney");
        println!(
            "  Turn 3: {} — {}ms, context retained: {}",
            trunc(c3, 200),
            r3.elapsed_ms,
            if contextual { "✅" } else { "⚠️" },
        );
    } else {
        let c1 = msg1["content"].as_str().unwrap_or("[empty]");
        println!("  Turn 1: ⚠️ No tool call: {}", trunc(c1, 200));
    }
    println!();

    println!("Done.");
}

async fn discover_models(base_url: &str) -> Result<Vec<Endpoint>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/models"))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("can't reach {base_url}/models: {e}"))?;
    let body: Value = resp.json().await.map_err(|e| format!("bad json: {e}"))?;

    let models: Vec<Endpoint> = body["data"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?;
            // Skip cloud/remote models and moa itself
            if id.contains("cloud") || id == "moa" {
                return None;
            }
            Some(Endpoint {
                base_url: base_url.to_string(),
                model: id.to_string(),
            })
        })
        .collect();

    Ok(models)
}

fn trunc(s: &str, max: usize) -> String {
    let stripped = strip_think(s);
    if stripped.len() > max {
        format!("{}...", &stripped[..max])
    } else {
        stripped
    }
}

fn strip_think(s: &str) -> String {
    let mut r = s.to_string();
    while let Some(start) = r.find("<think>") {
        if let Some(end) = r[start..].find("</think>") {
            r = format!("{}{}", &r[..start], &r[start + end + 8..]);
        } else {
            r = r[..start].to_string();
            break;
        }
    }
    r.trim().to_string()
}
