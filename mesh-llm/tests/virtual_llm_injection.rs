//! Integration test: virtual LLM injection framing.
//!
//! Spins up a mock hook server + llama-server with `--mesh-port`,
//! sends queries, and checks whether the model incorporates injected hints.
//!
//! Requires:
//!   - `llama.cpp/build/bin/llama-server` (mesh-hooks build)
//!   - A small GGUF model (auto-detected from HuggingFace cache)
//!
//! Run: `cargo test -p mesh-llm --test virtual_llm_injection -- --nocapture`
//!
//! Skipped automatically if llama-server or model file is missing.

use axum::{extract::Json, routing::post, Router};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::time::Duration;

// ── Helpers ──────────────────────────────────────────────────────────────

fn find_llama_server() -> Option<PathBuf> {
    // Walk up from CARGO_MANIFEST_DIR to find llama.cpp/build/bin/llama-server
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest.parent()?;
    let bin = repo_root.join("llama.cpp/build/bin/llama-server");
    if bin.is_file() {
        Some(bin)
    } else {
        None
    }
}

fn find_small_model() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;

    // CI puts models here
    for name in &["SmolLM2-135M-Instruct-Q8_0.gguf"] {
        let p = PathBuf::from(format!("{home}/.models/{name}"));
        if p.is_file() {
            return Some(p);
        }
    }

    // Local dev — check HuggingFace cache
    let hub = PathBuf::from(format!("{home}/.cache/huggingface/hub"));
    let candidates = [
        ("models--unsloth--Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
        (
            "models--unsloth--gemma-4-E4B-it-GGUF",
            "gemma-4-E4B-it-Q4_K_M.gguf",
        ),
    ];

    for (model_dir, filename) in &candidates {
        let snapshots = hub.join(model_dir).join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let candidate = entry.path().join(filename);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Wait for llama-server to be ready on the given port.
async fn wait_for_ready(port: u16, timeout_secs: u64) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if tokio::time::Instant::now() > deadline {
            return false;
        }
        match reqwest::get(format!("http://127.0.0.1:{port}/health")).await {
            Ok(resp) => {
                if let Ok(body) = resp.json::<Value>().await {
                    if body["status"] == "ok" {
                        return true;
                    }
                }
            }
            Err(_) => {}
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Send a chat completion and return the assistant's content + reasoning.
async fn chat(port: u16, question: &str, max_tokens: u32) -> (String, String) {
    let client = reqwest::Client::new();
    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    });

    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&body)
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .expect("request failed")
        .json::<Value>()
        .await
        .expect("parse failed");

    let msg = &resp["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("").to_string();
    let reasoning = msg["reasoning_content"].as_str().unwrap_or("").to_string();
    (content, reasoning)
}

// ── Framing variants ────────────────────────────────────────────────────

fn framing_current(hint: &str) -> String {
    format!(
        "\nHere is relevant information to help answer: {hint}\n\n\
         Now answer the user's question directly:\n"
    )
}

fn framing_reference(hint: &str) -> String {
    format!("\n\nReference answer: {hint}\n\nUse the reference above to provide an accurate response.\n")
}

fn framing_assistant_draft(hint: &str) -> String {
    format!(
        "\n\nHere is a draft answer from another assistant:\n{hint}\n\n\
         Incorporate this into your response. Answer concisely:\n"
    )
}

fn framing_rag(hint: &str) -> String {
    format!(
        "\n\nContext from knowledge base:\n{hint}\n\n\
         Based on the above context, answer the question:\n"
    )
}

// ── Test cases ──────────────────────────────────────────────────────────

struct TestCase {
    question: &'static str,
    hint: &'static str,
    /// Substrings: answer is correct if ANY of these appear (case-insensitive).
    accept: &'static [&'static str],
}

const CASES: &[TestCase] = &[
    TestCase {
        question: "What is the population of Nauru?",
        hint: "The population of Nauru is approximately 12,500 people as of 2023.",
        accept: &["12,500", "12500", "12.5"],
    },
    TestCase {
        question: "Who won the 1953 Pulitzer Prize for Fiction?",
        hint:
            "Ernest Hemingway won the 1953 Pulitzer Prize for Fiction for The Old Man and the Sea.",
        accept: &["hemingway", "old man and the sea"],
    },
    TestCase {
        question: "What is the capital of Burkina Faso?",
        hint: "The capital of Burkina Faso is Ouagadougou.",
        accept: &["ouagadougou"],
    },
    TestCase {
        question: "Translate to Swahili: The meeting has been rescheduled to Thursday",
        hint: "Mkutano umesogezwa hadi Alhamisi.",
        accept: &["mkutano", "alhamisi"],
    },
    // Easy question — the hint is correct, model should still get it right
    TestCase {
        question: "What is 2+2?",
        hint: "2 + 2 = 4.",
        accept: &["4"],
    },
];

// ── Mock hook server ────────────────────────────────────────────────────

#[derive(Clone)]
struct MockState {
    /// The inject text to return for post_prefill / mid_generation hooks.
    inject_text: Arc<Mutex<String>>,
    /// Log of hooks received.
    hook_log: Arc<Mutex<Vec<String>>>,
}

async fn handle_hook(
    axum::extract::State(state): axum::extract::State<MockState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let hook = payload["hook"].as_str().unwrap_or("");
    state.hook_log.lock().unwrap().push(hook.to_string());

    let inject = state.inject_text.lock().unwrap().clone();

    match hook {
        // Always inject on post_prefill (Hook 2) — we're testing the framing.
        // In production this only fires when entropy > threshold, but with
        // --mesh-hook-debug the threshold is 0.5 so it fires on most queries.
        "post_prefill" if !inject.is_empty() => Json(json!({ "action": "inject", "text": inject })),
        _ => Json(json!({ "action": "none" })),
    }
}

async fn start_mock(state: MockState) -> (u16, tokio::task::JoinHandle<()>) {
    let app = Router::new()
        .route("/mesh/hook", post(handle_hook))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    (port, handle)
}

// ── Main test ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_injection_framing() {
    // Skip if prerequisites are missing
    let llama_bin = match find_llama_server() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: llama-server not found (build llama.cpp first)");
            return;
        }
    };
    let model_path = match find_small_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no small GGUF model found in HuggingFace cache");
            return;
        }
    };

    eprintln!("Using llama-server: {}", llama_bin.display());
    eprintln!("Using model: {}", model_path.display());

    // Start mock hook server
    let state = MockState {
        inject_text: Arc::new(Mutex::new(String::new())),
        hook_log: Arc::new(Mutex::new(Vec::new())),
    };
    let (mock_port, mock_handle) = start_mock(state.clone()).await;
    eprintln!("Mock hook server on port {mock_port}");

    // Find a free port for llama-server
    let llama_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let llama_port = llama_listener.local_addr().unwrap().port();
    drop(llama_listener);

    // Start llama-server
    let mut child = Command::new(&llama_bin)
        .args([
            "-m",
            model_path.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            &llama_port.to_string(),
            "--mesh-port",
            &mock_port.to_string(),
            "--mesh-hook-debug",
            "-ngl",
            "99",
            "--no-warmup",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("failed to start llama-server");

    // Wait for ready
    assert!(
        wait_for_ready(llama_port, 60).await,
        "llama-server didn't become ready"
    );
    eprintln!("llama-server ready on port {llama_port}");

    let framings: Vec<(&str, fn(&str) -> String)> = vec![
        ("current", framing_current),
        ("reference", framing_reference),
        ("assistant_draft", framing_assistant_draft),
        ("rag", framing_rag),
    ];

    // ── Baseline: no injection ──────────────────────────────────────
    eprintln!("\n=== BASELINE (no injection) ===");
    let mut baseline_results = Vec::new();
    for tc in CASES {
        *state.inject_text.lock().unwrap() = String::new();
        state.hook_log.lock().unwrap().clear();

        let (content, reasoning) = chat(llama_port, tc.question, 300).await;
        let combined = format!("{content} {reasoning}").to_lowercase();
        let passed = tc
            .accept
            .iter()
            .any(|a| combined.contains(&a.to_lowercase()));
        baseline_results.push(passed);

        let display = if content.is_empty() {
            format!(
                "(reasoning only) {}",
                &reasoning.chars().take(100).collect::<String>()
            )
        } else {
            content.chars().take(100).collect::<String>()
        };
        eprintln!("  {} {}", if passed { "✅" } else { "❌" }, tc.question);
        eprintln!("     → {display}");
    }

    // ── Test each framing ───────────────────────────────────────────
    let mut framing_results: Vec<(&str, Vec<bool>)> = Vec::new();

    for (name, framing_fn) in &framings {
        eprintln!("\n=== FRAMING: {name} ===");
        let mut results = Vec::new();

        for tc in CASES {
            let inject = framing_fn(tc.hint);
            *state.inject_text.lock().unwrap() = inject;
            state.hook_log.lock().unwrap().clear();

            let (content, reasoning) = chat(llama_port, tc.question, 300).await;
            let hooks = state.hook_log.lock().unwrap().clone();
            let combined = format!("{content} {reasoning}").to_lowercase();
            let passed = tc
                .accept
                .iter()
                .any(|a| combined.contains(&a.to_lowercase()));
            results.push(passed);

            let hook_str = if hooks.contains(&"post_prefill".to_string()) {
                "🔗"
            } else {
                "⚪"
            };
            let display = if content.is_empty() {
                format!(
                    "(reasoning only) {}",
                    &reasoning.chars().take(100).collect::<String>()
                )
            } else {
                content.chars().take(100).collect::<String>()
            };
            eprintln!(
                "  {} {hook_str} {}",
                if passed { "✅" } else { "❌" },
                tc.question
            );
            eprintln!("     → {display}");
        }

        framing_results.push((name, results));
    }

    // ── Summary ─────────────────────────────────────────────────────
    eprintln!("\n=== SUMMARY ===");
    let baseline_score: usize = baseline_results.iter().filter(|&&b| b).count();
    eprintln!("  baseline:         {baseline_score}/{}", CASES.len());
    for (name, results) in &framing_results {
        let score: usize = results.iter().filter(|&&b| b).count();
        eprintln!("  {name:20}{score}/{}", CASES.len());
    }

    // ── Assertions ──────────────────────────────────────────────────
    // At least one framing should beat baseline
    let best_framing_score = framing_results
        .iter()
        .map(|(_, r)| r.iter().filter(|&&b| b).count())
        .max()
        .unwrap_or(0);

    eprintln!("\nBaseline: {baseline_score}, Best framing: {best_framing_score}");

    // The injection should be helping — at least the best framing should
    // get more right than baseline. If baseline already gets everything
    // right, we can't distinguish, so skip the assertion.
    if baseline_score < CASES.len() {
        assert!(
            best_framing_score > baseline_score,
            "No framing improved over baseline ({baseline_score}/{}) — \
             injection is not working",
            CASES.len()
        );
    }

    // Cleanup
    child.kill().await.ok();
    mock_handle.abort();
}
