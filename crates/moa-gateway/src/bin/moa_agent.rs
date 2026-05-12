//! Agentic workload simulation for the MoA gateway.
//!
//! Simulates a coding agent that uses tools across many turns:
//!   1. Read a file
//!   2. Analyze what's wrong
//!   3. Search for a fix
//!   4. Edit the file
//!   5. Run tests
//!   6. Fix test failures
//!   7. Summarize
//!
//! This exercises:
//!   - Many sequential turns (7+) with accumulating context
//!   - Repeated tool use with different tools
//!   - Context that must carry across turns (file contents, error messages)
//!   - Workers getting summaries, not raw 20-turn histories
//!   - The gateway's tool lifecycle under sustained load
//!
//! Usage:
//!   cargo run -p moa-gateway --bin moa-agent
//!   cargo run -p moa-gateway --bin moa-agent -- --url http://localhost:11434/v1

use moa_gateway::{Endpoint, Gateway, GatewayConfig, TurnResult};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const TOOLS: &str = r#"[
  {"type":"function","function":{"name":"read_file","description":"Read the contents of a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"}},"required":["path"]}}},
  {"type":"function","function":{"name":"edit_file","description":"Edit a file by replacing old text with new text","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"old_text":{"type":"string","description":"Text to find"},"new_text":{"type":"string","description":"Replacement text"}},"required":["path","old_text","new_text"]}}},
  {"type":"function","function":{"name":"run_command","description":"Run a shell command and return stdout/stderr","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Command to run"}},"required":["command"]}}},
  {"type":"function","function":{"name":"search_code","description":"Search for a pattern across files","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Search pattern"},"directory":{"type":"string","description":"Directory to search"}},"required":["pattern"]}}}
]"#;

struct AgentSim {
    gateway: Gateway,
    messages: Vec<Value>,
    tools: Value,
    total_turns: usize,
    tool_calls_made: usize,
    reducer_invocations: usize,
    start: Instant,
}

impl AgentSim {
    fn new(endpoints: Vec<Endpoint>) -> Self {
        let config = GatewayConfig {
            endpoints,
            worker_timeout: Duration::from_secs(60),
            reducer_timeout: Duration::from_secs(90),
        };
        let tools: Value = serde_json::from_str(TOOLS).unwrap();
        Self {
            gateway: Gateway::new(config),
            messages: vec![json!({"role": "system", "content":
                "You are a senior software engineer. You have access to tools for reading files, editing files, running commands, and searching code. Use them to complete the task. Think step by step. Only call one tool at a time."
            })],
            tools,
            total_turns: 0,
            tool_calls_made: 0,
            reducer_invocations: 0,
            start: Instant::now(),
        }
    }

    /// Send a user message and get the response.
    async fn user_turn(&mut self, text: &str) -> TurnResult {
        self.messages.push(json!({"role": "user", "content": text}));
        let body = json!({
            "messages": self.messages,
            "tools": self.tools,
        });
        let result = self.gateway.turn(&body).await;
        self.total_turns += 1;

        let msg = result.response_body["choices"][0]["message"].clone();
        self.messages.push(msg);

        if result.reducer_used {
            self.reducer_invocations += 1;
        }

        result
    }

    /// Simulate a tool result coming back from the environment.
    async fn tool_result(&mut self, call_id: &str, content: &str) -> TurnResult {
        self.messages
            .push(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
        let body = json!({
            "messages": self.messages,
            "tools": self.tools,
        });
        let result = self.gateway.turn(&body).await;
        self.total_turns += 1;
        self.tool_calls_made += 1;

        let msg = result.response_body["choices"][0]["message"].clone();
        self.messages.push(msg);

        if result.reducer_used {
            self.reducer_invocations += 1;
        }

        result
    }

    fn extract_tool_call(result: &TurnResult) -> Option<(String, String, Value)> {
        let tc = result
            .response_body
            .pointer("/choices/0/message/tool_calls/0")?;
        let id = tc.get("id")?.as_str()?.to_string();
        let name = tc.pointer("/function/name")?.as_str()?.to_string();
        let args: Value = tc
            .pointer("/function/arguments")
            .and_then(|a| a.as_str())
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(json!({}));
        Some((id, name, args))
    }

    fn extract_content(result: &TurnResult) -> String {
        result.response_body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[empty]")
            .to_string()
    }

    fn message_count(&self) -> usize {
        self.messages.len()
    }
}

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
        .unwrap_or_else(|| "http://localhost:11434/v1".to_string());

    // Accept --models flag to select specific models (comma-separated)
    let model_filter: Option<Vec<String>> = args
        .iter()
        .position(|a| a == "--models")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').map(|m| m.trim().to_string()).collect());

    let mut endpoints = match discover_models(&base_url).await {
        Ok(eps) if eps.len() >= 2 => eps,
        Ok(eps) => {
            eprintln!(
                "Need ≥2 models, found {}: {:?}",
                eps.len(),
                eps.iter().map(|e| &e.model).collect::<Vec<_>>()
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to discover models at {base_url}: {e}");
            std::process::exit(1);
        }
    };

    if let Some(filter) = model_filter {
        endpoints.retain(|e| filter.iter().any(|f| e.model.contains(f)));
        if endpoints.len() < 2 {
            eprintln!(
                "After filtering, need ≥2 models but have {}",
                endpoints.len()
            );
            std::process::exit(1);
        }
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       MoA Gateway — Agentic Workload Test              ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Models: {:<47}║",
        endpoints
            .iter()
            .map(|e| e.model.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("║  Tools:  read_file, edit_file, run_command, search_code  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let mut sim = AgentSim::new(endpoints);

    // ── Step 1: User asks the agent to fix a bug ─────────────────

    println!("═══ Step 1: User reports a bug ═══");
    let r1 = sim.user_turn(
        "There's a bug in src/auth.py — the login function doesn't hash passwords before comparing. \
         Can you read the file and fix it?"
    ).await;
    print_turn(1, &r1, &sim);

    // Expect: tool call to read_file
    let (call_id, tool_name, _args) = match AgentSim::extract_tool_call(&r1) {
        Some(tc) => tc,
        None => {
            println!(
                "  ⚠️  No tool call — got text: {}",
                trunc(&AgentSim::extract_content(&r1), 200)
            );
            println!("  Continuing anyway...\n");
            // Give it a nudge
            let r1b = sim
                .user_turn("Please use the read_file tool to read src/auth.py first.")
                .await;
            print_turn(1, &r1b, &sim);
            match AgentSim::extract_tool_call(&r1b) {
                Some(tc) => tc,
                None => {
                    println!("  ❌ Still no tool call. Aborting.");
                    return;
                }
            }
        }
    };
    println!("  → Tool: {tool_name}");

    // ── Step 2: Simulate file read result ────────────────────────

    println!("\n═══ Step 2: Tool result — file contents ═══");
    let file_contents = r#"import os
from flask import Flask, request, jsonify

app = Flask(__name__)

USERS = {
    "alice": "password123",
    "bob": "hunter2",
}

def login(username, password):
    """Authenticate a user."""
    if username not in USERS:
        return {"error": "User not found"}, 401
    if USERS[username] == password:  # BUG: comparing plaintext!
        token = os.urandom(32).hex()
        return {"token": token}, 200
    return {"error": "Invalid password"}, 401

@app.route("/login", methods=["POST"])
def login_route():
    data = request.get_json()
    return jsonify(*login(data["username"], data["password"]))
"#;
    let r2 = sim.tool_result(&call_id, file_contents).await;
    print_turn(2, &r2, &sim);

    // Expect: either a text analysis or an edit_file tool call
    if let Some((call_id2, tool_name2, args2)) = AgentSim::extract_tool_call(&r2) {
        println!("  → Tool: {tool_name2}");

        // ── Step 3: Edit result ──────────────────────────────────
        println!("\n═══ Step 3: Tool result — edit applied ═══");
        let edit_result = if tool_name2 == "edit_file" {
            let old = args2
                .get("old_text")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("Successfully replaced {} bytes in src/auth.py", old.len())
        } else if tool_name2 == "search_code" {
            "src/auth.py:15:    if USERS[username] == password:  # BUG: comparing plaintext!"
                .to_string()
        } else {
            format!("{tool_name2} completed successfully")
        };
        let r3 = sim.tool_result(&call_id2, &edit_result).await;
        print_turn(3, &r3, &sim);

        // Continue the chain...
        continue_agent_loop(&mut sim, &r3, 4).await;
    } else {
        // It gave a text analysis — ask it to proceed with the fix
        println!(
            "  → Analysis: {}",
            trunc(&AgentSim::extract_content(&r2), 200)
        );

        println!("\n═══ Step 3: Ask agent to apply the fix ═══");
        let r3 = sim
            .user_turn(
                "Good analysis. Now please use edit_file to fix the password comparison. \
             Use bcrypt or hashlib to hash before comparing.",
            )
            .await;
        print_turn(3, &r3, &sim);
        continue_agent_loop(&mut sim, &r3, 4).await;
    }

    // ── Final summary ────────────────────────────────────────────

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    Summary                             ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Total turns:         {:>4}                              ║",
        sim.total_turns
    );
    println!(
        "║  Tool calls made:     {:>4}                              ║",
        sim.tool_calls_made
    );
    println!(
        "║  Reducer invocations: {:>4}                              ║",
        sim.reducer_invocations
    );
    println!(
        "║  Messages in context: {:>4}                              ║",
        sim.message_count()
    );
    println!(
        "║  Total time:          {:>4}s                             ║",
        sim.start.elapsed().as_secs()
    );
    println!("╚══════════════════════════════════════════════════════════╝");
}

/// Continue the agent loop: keep processing tool calls until the agent
/// gives a text answer or we hit a limit.
async fn continue_agent_loop(sim: &mut AgentSim, last_result: &TurnResult, mut step: usize) {
    let mut current = last_result.response_body.clone();
    let mut recent_tools: Vec<String> = Vec::new(); // for loop detection
    let mut asked_for_tests = false;

    loop {
        if step > 10 {
            println!("\n  ⚠️  Hit step limit (10), stopping.");
            break;
        }

        // Check if the last response was a tool call
        let tc = current.pointer("/choices/0/message/tool_calls/0").cloned();

        match tc {
            Some(tc_val) => {
                let call_id = tc_val.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let name = tc_val
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let args: Value = tc_val
                    .pointer("/function/arguments")
                    .and_then(|a| a.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(json!({}));

                // Loop detection: if the same tool+args repeated 2x, nudge the model
                let sig = format!(
                    "{name}:{}",
                    serde_json::to_string(&args).unwrap_or_default()
                );
                let repeat_count = recent_tools.iter().filter(|s| **s == sig).count();
                recent_tools.push(sig);

                if repeat_count >= 2 {
                    println!("\n  ⚠️  Loop detected ({name} called 3x with same args), nudging.");
                    let r = sim
                        .user_turn(
                            "You've called the same tool multiple times with the same arguments. \
                         Please proceed with the next step or summarize what you've done.",
                        )
                        .await;
                    print_turn(step, &r, sim);
                    current = r.response_body;
                    step += 1;
                    continue;
                }

                // Simulate tool execution
                let tool_output = simulate_tool(name, &args, step);

                println!("\n═══ Step {step}: Tool result — {name} ═══");
                let r = sim.tool_result(call_id, &tool_output).await;
                print_turn(step, &r, sim);
                current = r.response_body;
                step += 1;
            }
            None => {
                // Text answer — we're done (or need a nudge)
                let content = current["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("[empty]");
                if content.len() < 10 || content == "[empty]" {
                    println!("\n  ⚠️  Empty response, stopping.");
                    break;
                }

                // Ask for tests once if the agent hasn't mentioned them
                if !asked_for_tests && step <= 8 {
                    asked_for_tests = true;
                    println!("\n═══ Step {step}: Ask agent to run tests ═══");
                    let r = sim
                        .user_turn("Good. Now run the tests to make sure the fix works: use run_command with 'pytest src/test_auth.py -v'")
                        .await;
                    print_turn(step, &r, sim);
                    current = r.response_body;
                    step += 1;
                } else {
                    // Agent gave a final summary or we've already asked for tests
                    println!("\n  ✅ Agent finished with text response.");
                    break;
                }
            }
        }
    }
}

/// Simulate tool execution with realistic outputs.
/// Tracks state to give different results on repeated calls.
fn simulate_tool(name: &str, args: &Value, step: usize) -> String {
    static EDIT_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    match name {
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let edits = EDIT_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            if path.contains("test") {
                r#"import pytest
from auth import login, USERS
import hashlib

def test_login_valid():
    result, status = login("alice", "password123")
    assert status == 200
    assert "token" in result

def test_login_wrong_password():
    result, status = login("alice", "wrongpass")
    assert status == 401

def test_login_unknown_user():
    result, status = login("nobody", "pass")
    assert status == 401
"#
                .to_string()
            } else if edits > 0 {
                // After edits, show the fixed version
                r#"import os
import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

USERS = {
    "alice": _hash("password123"),
    "bob": _hash("hunter2"),
}

def login(username, password):
    """Authenticate a user."""
    if username not in USERS:
        return {"error": "User not found"}, 401
    if USERS[username] == _hash(password):
        token = os.urandom(32).hex()
        return {"token": token}, 200
    return {"error": "Invalid password"}, 401

@app.route("/login", methods=["POST"])
def login_route():
    data = request.get_json()
    return jsonify(*login(data["username"], data["password"]))
"#
                .to_string()
            } else {
                // Original buggy version
                r#"import os
from flask import Flask, request, jsonify

app = Flask(__name__)

USERS = {
    "alice": "password123",
    "bob": "hunter2",
}

def login(username, password):
    """Authenticate a user."""
    if username not in USERS:
        return {"error": "User not found"}, 401
    if USERS[username] == password:  # BUG: comparing plaintext!
        token = os.urandom(32).hex()
        return {"token": token}, 200
    return {"error": "Invalid password"}, 401

@app.route("/login", methods=["POST"])
def login_route():
    data = request.get_json()
    return jsonify(*login(data["username"], data["password"]))
"#
                .to_string()
            }
        }
        "edit_file" => {
            EDIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("file");
            let old = args.get("old_text").and_then(|v| v.as_str()).unwrap_or("");
            let new = args.get("new_text").and_then(|v| v.as_str()).unwrap_or("");
            format!(
                "✅ Edited {path}: replaced {} bytes with {} bytes.",
                old.len(),
                new.len()
            )
        }
        "run_command" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let edits = EDIT_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            if cmd.contains("pytest") {
                if edits == 0 || step <= 4 {
                    // Tests fail: stored passwords aren't hashed yet
                    r#"============================= test session starts ==============================
collected 3 items

src/test_auth.py::test_login_valid FAILED
src/test_auth.py::test_login_wrong_password PASSED
src/test_auth.py::test_login_unknown_user PASSED

=================================== FAILURES ===================================
___________________________ test_login_valid ____________________________

    def test_login_valid():
>       result, status = login("alice", "password123")
E       AssertionError: Expected 200 but got 401
E       The stored password hash doesn't match — USERS dict still has plaintext

=========================== short test summary info ============================
FAILED src/test_auth.py::test_login_valid
========================= 1 failed, 2 passed in 0.03s ========================="#.to_string()
                } else {
                    // After fix: tests pass
                    r#"============================= test session starts ==============================
collected 3 items

src/test_auth.py::test_login_valid PASSED
src/test_auth.py::test_login_wrong_password PASSED
src/test_auth.py::test_login_unknown_user PASSED

============================== 3 passed in 0.02s ==============================="#.to_string()
                }
            } else {
                format!("$ {cmd}\nCommand completed successfully.")
            }
        }
        "search_code" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
            format!(
                "src/auth.py:15:    if USERS[username] == password:  # matches '{pattern}'\n\
                 src/auth.py:7:    \"alice\": \"password123\",  # plaintext password"
            )
        }
        _ => format!("{name} completed"),
    }
}

fn print_turn(_step: usize, result: &TurnResult, sim: &AgentSim) {
    let has_tc = result
        .response_body
        .pointer("/choices/0/message/tool_calls/0")
        .is_some();
    let content = result.response_body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");

    if has_tc {
        let tc = &result.response_body["choices"][0]["message"]["tool_calls"][0];
        let name = tc
            .pointer("/function/name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let args = tc
            .pointer("/function/arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("{}");
        println!("  🔧 tool_call: {name}({})", trunc(args, 100),);
    } else {
        println!("  💬 {}", trunc(&strip_think(content), 300));
    }
    println!(
        "  ⏱  {}ms | workers: {}/{} | reducer: {} | msgs: {} | turn: {}",
        result.elapsed_ms,
        result
            .worker_summaries
            .iter()
            .filter(|w| w.succeeded)
            .count(),
        result.worker_summaries.len(),
        if result.reducer_used { "yes" } else { "no" },
        sim.message_count(),
        sim.total_turns,
    );
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
    let s = strip_think(s);
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s
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
