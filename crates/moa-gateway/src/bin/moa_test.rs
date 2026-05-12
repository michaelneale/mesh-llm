//! MoA gateway test harness.
//!
//! Runs comparative tests: MoA (fan-out to multiple models) vs solo (single
//! model).  Proves whether the ensemble is smarter and whether tool calling
//! works coherently through the gateway.
//!
//! Usage:
//!   cargo run -p moa-gateway --bin moa_test
//!
//! Expects ollama running at localhost:11434 with multiple models.

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

    // Discover available models from ollama
    let endpoints = match discover_ollama_models().await {
        Ok(eps) => eps,
        Err(e) => {
            eprintln!("Failed to discover ollama models: {e}");
            eprintln!("Make sure ollama is running: ollama serve");
            std::process::exit(1);
        }
    };

    if endpoints.len() < 2 {
        eprintln!(
            "Need at least 2 models for MoA testing, found {}",
            endpoints.len()
        );
        eprintln!("Pull more models: ollama pull qwen3:4b && ollama pull llama3.2:3b");
        std::process::exit(1);
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          Mixture-of-Agents Gateway Test                 ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Models: {:<47}║",
        endpoints
            .iter()
            .map(|e| e.model.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Test 1: Knowledge question (MoA vs solo) ─────────────────

    println!("━━━ Test 1: Knowledge — is MoA smarter than solo? ━━━");
    println!();
    let knowledge_q = json!({
        "messages": [
            {"role": "user", "content": "What is the capital of Myanmar? Answer in one sentence."}
        ]
    });
    run_comparison(&endpoints, &knowledge_q, "Knowledge").await;

    // ── Test 2: Reasoning (MoA vs solo) ──────────────────────────

    println!("━━━ Test 2: Reasoning — does ensemble help? ━━━");
    println!();
    let reasoning_q = json!({
        "messages": [
            {"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning step by step."}
        ]
    });
    run_comparison(&endpoints, &reasoning_q, "Reasoning").await;

    // ── Test 3: Tool calling coherence ───────────────────────────

    println!("━━━ Test 3: Tool calling — does the gateway produce valid tool calls? ━━━");
    println!();
    let tool_q = json!({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": "What's the current weather in Tokyo?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    });
    run_tool_test(&endpoints, &tool_q).await;

    // ── Test 4: Multi-turn tool lifecycle ─────────────────────────

    println!("━━━ Test 4: Tool lifecycle — tool call → result → answer ━━━");
    println!();
    run_tool_lifecycle_test(&endpoints).await;

    // ── Test 5: Coding question ──────────────────────────────────

    println!("━━━ Test 5: Code — can the ensemble write better code? ━━━");
    println!();
    let code_q = json!({
        "messages": [
            {"role": "user", "content": "Write a Python function that checks if a string is a palindrome, handling Unicode properly. Just the function, no explanation."}
        ]
    });
    run_comparison(&endpoints, &code_q, "Code").await;
}

async fn discover_ollama_models() -> Result<Vec<Endpoint>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get("http://localhost:11434/v1/models")
        .send()
        .await
        .map_err(|e| format!("can't reach ollama: {e}"))?;
    let body: Value = resp.json().await.map_err(|e| format!("bad json: {e}"))?;

    let models: Vec<Endpoint> = body["data"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?;
            // Skip cloud/remote models
            if id.contains("cloud") {
                return None;
            }
            Some(Endpoint {
                base_url: "http://localhost:11434/v1".to_string(),
                model: id.to_string(),
            })
        })
        .collect();

    Ok(models)
}

/// Run a question through MoA and each model solo, compare.
async fn run_comparison(endpoints: &[Endpoint], body: &Value, label: &str) {
    // Solo runs: ask each model individually
    let http = reqwest::Client::new();
    for ep in endpoints {
        let start = Instant::now();
        let mut solo_body = body.clone();
        solo_body["model"] = json!(ep.model);
        solo_body["max_tokens"] = json!(256);
        solo_body["stream"] = json!(false);
        solo_body["temperature"] = json!(0.3);

        match http
            .post(format!("{}/chat/completions", ep.base_url))
            .json(&solo_body)
            .timeout(Duration::from_secs(60))
            .send()
            .await
        {
            Ok(resp) => {
                let elapsed = start.elapsed().as_millis();
                if let Ok(r) = resp.json::<Value>().await {
                    let content = r["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("[no content]");
                    println!(
                        "  Solo {} ({}ms):\n    {}",
                        ep.model,
                        elapsed,
                        truncate(content, 200)
                    );
                }
            }
            Err(e) => println!("  Solo {} FAILED: {e}", ep.model),
        }
        println!();
    }

    // MoA run
    let config = GatewayConfig {
        endpoints: endpoints.to_vec(),
        worker_timeout: Duration::from_secs(60),
        reducer_timeout: Duration::from_secs(90),
    };
    let mut gateway = Gateway::new(config);
    let start = Instant::now();
    let result = gateway.turn(body).await;
    let elapsed = start.elapsed().as_millis();

    let content = result.response_body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("[no content]");
    let workers_ok = result
        .worker_summaries
        .iter()
        .filter(|w| w.succeeded)
        .count();
    println!(
        "  MoA ({}ms, {}/{} workers, reducer={}): \n    {}",
        elapsed,
        workers_ok,
        result.worker_summaries.len(),
        result.reducer_used,
        truncate(content, 300),
    );
    println!();
    println!("  [{label} comparison complete]");
    println!();
}

/// Test that MoA produces a valid tool call.
async fn run_tool_test(endpoints: &[Endpoint], body: &Value) {
    let config = GatewayConfig {
        endpoints: endpoints.to_vec(),
        worker_timeout: Duration::from_secs(60),
        reducer_timeout: Duration::from_secs(90),
    };
    let mut gateway = Gateway::new(config);
    let result = gateway.turn(body).await;

    let msg = &result.response_body["choices"][0]["message"];
    let has_tool_calls = msg
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let has_content = msg
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false);

    if has_tool_calls {
        let tc = &msg["tool_calls"][0];
        println!("  ✅ MoA produced a tool call:");
        println!("     function: {}", tc["function"]["name"]);
        println!("     arguments: {}", tc["function"]["arguments"]);
        println!(
            "     ({}ms, reducer={})",
            result.elapsed_ms, result.reducer_used
        );
    } else if has_content {
        let content = msg["content"].as_str().unwrap_or("");
        println!("  ⚠️  MoA produced a text answer instead of tool call:");
        println!("     {}", truncate(content, 200));
        println!(
            "     ({}ms, reducer={})",
            result.elapsed_ms, result.reducer_used
        );
    } else {
        println!("  ❌ MoA produced neither tool call nor content");
        println!("     raw: {}", result.response_body);
    }

    // Show worker details
    for ws in &result.worker_summaries {
        println!(
            "     worker {} ({}): ok={}, {:?}, conf={:?}",
            ws.model,
            ws.role.label(),
            ws.succeeded,
            ws.output_kind,
            ws.confidence,
        );
    }
    println!();
}

/// Test the full tool lifecycle: query → tool_call → tool_result → final answer.
async fn run_tool_lifecycle_test(endpoints: &[Endpoint]) {
    let config = GatewayConfig {
        endpoints: endpoints.to_vec(),
        worker_timeout: Duration::from_secs(60),
        reducer_timeout: Duration::from_secs(90),
    };
    let mut gateway = Gateway::new(config);

    // Turn 1: user asks a question that needs a tool
    let turn1 = json!({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
            {"role": "user", "content": "Look up the current price of Bitcoin"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }]
    });

    println!("  Turn 1: User asks about Bitcoin price (expects tool call)");
    let r1 = gateway.turn(&turn1).await;
    let msg1 = &r1.response_body["choices"][0]["message"];
    let has_tc = msg1
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    if has_tc {
        let tc = &msg1["tool_calls"][0];
        let call_id = tc["id"].as_str().unwrap_or("call_1");
        println!(
            "    ✅ tool_call: {}({})",
            tc["function"]["name"], tc["function"]["arguments"]
        );

        // Turn 2: Simulate tool result coming back
        let turn2 = json!({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
                {"role": "user", "content": "Look up the current price of Bitcoin"},
                {"role": "assistant", "content": null, "tool_calls": [tc.clone()]},
                {"role": "tool", "tool_call_id": call_id, "content": "Bitcoin is currently trading at $104,250 USD, up 2.3% in the last 24 hours."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
                }
            }]
        });

        println!("  Turn 2: Tool result returned (expects final answer)");
        let r2 = gateway.turn(&turn2).await;
        let content = r2.response_body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[no content]");
        let has_tc2 = r2.response_body["choices"][0]["message"]
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);

        if has_tc2 {
            println!("    ⚠️  Gateway proposed another tool call instead of answering");
        } else {
            println!("    ✅ Final answer: {}", truncate(content, 200));
        }
        println!("    ({}ms, reducer={})", r2.elapsed_ms, r2.reducer_used);
    } else {
        let content = msg1["content"].as_str().unwrap_or("[empty]");
        println!(
            "    ⚠️  No tool call, got text answer: {}",
            truncate(content, 200)
        );
    }
    println!();
}

fn truncate(s: &str, max: usize) -> String {
    // Also strip <think> tags for display
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
