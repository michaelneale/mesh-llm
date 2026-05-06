use std::{
    fs,
    io::{BufRead, BufReader},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::cli::ChatCorpusArgs;

#[derive(Debug, Clone)]
struct PromptCase {
    index: usize,
    prompt_id: Option<String>,
    category: Option<String>,
    length_bucket: Option<String>,
    session_group: Option<String>,
    prompt: String,
    messages: Option<Value>,
}

#[derive(Serialize)]
struct ChatCorpusReport {
    base_url: String,
    model: String,
    endpoint: &'static str,
    stream: bool,
    include_usage: bool,
    max_tokens: u32,
    concurrency_depth: usize,
    request_timeout_secs: u64,
    request_count: usize,
    prompt_corpus: Option<String>,
    prompt_limit: Option<usize>,
    sampling: SamplingReport,
    results: Vec<ChatCorpusResult>,
    summary: ChatCorpusSummary,
}

#[derive(Default, Serialize)]
struct SamplingReport {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<u64>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatCorpusResult {
    sequence: usize,
    prompt_id: Option<String>,
    category: Option<String>,
    length_bucket: Option<String>,
    session_id: String,
    prompt_chars: usize,
    elapsed_ms: f64,
    ttft_ms: Option<f64>,
    completion_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    total_tokens: Option<u64>,
    finish_reason: Option<String>,
    output_chars: usize,
    error: Option<String>,
    api_error_code: Option<String>,
}

#[derive(Default, Serialize)]
struct ChatCorpusSummary {
    count: usize,
    errors: usize,
    elapsed_ms_min: Option<f64>,
    elapsed_ms_mean: Option<f64>,
    elapsed_ms_p50: Option<f64>,
    elapsed_ms_p95: Option<f64>,
    elapsed_ms_p99: Option<f64>,
    ttft_ms_p50: Option<f64>,
    ttft_ms_p95: Option<f64>,
    ttft_ms_p99: Option<f64>,
    completion_tokens: u64,
    total_tokens: u64,
    total_wall_ms: f64,
    completion_tok_s: Option<f64>,
    total_tok_s: Option<f64>,
}

pub fn chat_corpus(args: ChatCorpusArgs) -> Result<()> {
    if args.concurrency_depth == 0 {
        anyhow::bail!("--concurrency-depth must be greater than zero");
    }
    if args.max_tokens == 0 {
        anyhow::bail!("--max-tokens must be greater than zero");
    }

    let prompts = Arc::new(prompt_cases(&args)?);
    let client = Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs))
        .build()
        .context("failed to build HTTP client")?;
    let next = Arc::new(AtomicUsize::new(0));
    let results = Arc::new(Mutex::new(Vec::with_capacity(prompts.len())));
    let started = Instant::now();
    let args = Arc::new(args);

    let mut workers = Vec::with_capacity(args.concurrency_depth);
    for _ in 0..args.concurrency_depth {
        let client = client.clone();
        let prompts = Arc::clone(&prompts);
        let next = Arc::clone(&next);
        let results = Arc::clone(&results);
        let args = Arc::clone(&args);
        workers.push(thread::spawn(move || loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(prompt_case) = prompts.get(index) else {
                break;
            };
            let result = run_case(&client, &args, prompt_case);
            results.lock().expect("results mutex poisoned").push(result);
        }));
    }

    for worker in workers {
        worker.join().expect("chat corpus worker panicked");
    }
    let total_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut results = Arc::try_unwrap(results)
        .expect("results still shared")
        .into_inner()
        .expect("results mutex poisoned");
    results.sort_by_key(|result| result.sequence);

    let report = ChatCorpusReport {
        base_url: args.base_url.trim_end_matches('/').to_string(),
        model: args.model.clone(),
        endpoint: "/v1/chat/completions",
        stream: args.stream,
        include_usage: args.include_usage,
        max_tokens: args.max_tokens,
        concurrency_depth: args.concurrency_depth,
        request_timeout_secs: args.request_timeout_secs,
        request_count: prompts.len(),
        prompt_corpus: args
            .prompt_corpus
            .as_ref()
            .map(|path| path.display().to_string()),
        prompt_limit: args.prompt_limit,
        sampling: SamplingReport {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            seed: args.seed,
            enable_thinking: args.enable_thinking,
            reasoning_effort: args.reasoning_effort.clone(),
        },
        summary: summarize(&results, total_wall_ms),
        results,
    };

    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output.as_ref() {
        fs::write(output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

fn run_case(client: &Client, args: &ChatCorpusArgs, prompt_case: &PromptCase) -> ChatCorpusResult {
    let session_id = prompt_case
        .session_group
        .clone()
        .unwrap_or_else(|| format!("{}-{}", args.session_prefix, prompt_case.index));
    let started = Instant::now();
    let request = request_body(args, prompt_case, &session_id);
    let response = client
        .post(format!(
            "{}/chat/completions",
            args.base_url.trim_end_matches('/')
        ))
        .json(&request)
        .send();
    match response {
        Ok(response) if args.stream => {
            parse_stream_response(response, started, prompt_case, session_id)
        }
        Ok(response) => parse_json_response(response, started, prompt_case, session_id),
        Err(error) => error_result(prompt_case, session_id, started, error.to_string(), None),
    }
}

fn request_body(args: &ChatCorpusArgs, prompt_case: &PromptCase, session_id: &str) -> Value {
    let messages = prompt_case.messages.clone().unwrap_or_else(|| {
        json!([
            {
                "role": "user",
                "content": prompt_case.prompt,
            }
        ])
    });
    let mut object = Map::new();
    object.insert("model".to_string(), Value::String(args.model.clone()));
    object.insert("messages".to_string(), messages);
    object.insert("max_tokens".to_string(), json!(args.max_tokens));
    object.insert("stream".to_string(), json!(args.stream));
    object.insert("user".to_string(), Value::String(session_id.to_string()));
    if args.stream && args.include_usage {
        object.insert(
            "stream_options".to_string(),
            json!({
                "include_usage": true,
            }),
        );
    }
    if let Some(temperature) = args.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = args.top_p {
        object.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(top_k) = args.top_k {
        object.insert("top_k".to_string(), json!(top_k));
    }
    if let Some(seed) = args.seed {
        object.insert("seed".to_string(), json!(seed));
    }
    if let Some(enable_thinking) = args.enable_thinking {
        object.insert("enable_thinking".to_string(), json!(enable_thinking));
    }
    if let Some(reasoning_effort) = args.reasoning_effort.as_ref() {
        object.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort.clone()),
        );
    }
    Value::Object(object)
}

fn parse_json_response(
    response: reqwest::blocking::Response,
    started: Instant,
    prompt_case: &PromptCase,
    session_id: String,
) -> ChatCorpusResult {
    let status = response.status();
    let body = response.json::<Value>();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match body {
        Ok(value) if status.is_success() => {
            let usage = value.get("usage");
            ChatCorpusResult {
                sequence: prompt_case.index,
                prompt_id: prompt_case.prompt_id.clone(),
                category: prompt_case.category.clone(),
                length_bucket: prompt_case.length_bucket.clone(),
                session_id,
                prompt_chars: prompt_case.prompt.chars().count(),
                elapsed_ms,
                ttft_ms: None,
                completion_tokens: usage.and_then(|usage| usage_u64(usage, "completion_tokens")),
                prompt_tokens: usage.and_then(|usage| usage_u64(usage, "prompt_tokens")),
                total_tokens: usage.and_then(|usage| usage_u64(usage, "total_tokens")),
                finish_reason: value
                    .pointer("/choices/0/finish_reason")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                output_chars: value
                    .pointer("/choices/0/message/content")
                    .and_then(Value::as_str)
                    .map(str::chars)
                    .map(Iterator::count)
                    .unwrap_or_default(),
                error: None,
                api_error_code: None,
            }
        }
        Ok(value) => error_result(
            prompt_case,
            session_id,
            started,
            format!("chat completions request failed with status {status}: {value}"),
            api_error_code(&value),
        ),
        Err(error) => error_result(
            prompt_case,
            session_id,
            started,
            format!("failed to parse chat completions JSON response: {error}"),
            None,
        ),
    }
}

fn parse_stream_response(
    response: reqwest::blocking::Response,
    started: Instant,
    prompt_case: &PromptCase,
    session_id: String,
) -> ChatCorpusResult {
    let status = response.status();
    if !status.is_success() {
        return parse_json_response(response, started, prompt_case, session_id);
    }

    let mut reader = BufReader::new(response);
    let mut line = String::new();
    let mut first_token_ms = None;
    let mut output_chars = 0usize;
    let mut completion_tokens = None;
    let mut prompt_tokens = None;
    let mut total_tokens = None;
    let mut finish_reason = None;
    let mut error = None;
    let mut api_error_code_value = None;

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(read_error) => {
                error = Some(format!("failed reading SSE response: {read_error}"));
                break;
            }
        }
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            error = Some(format!("failed parsing SSE data: {data}"));
            break;
        };
        if value.get("error").is_some() {
            api_error_code_value = api_error_code(&value);
            error = Some(format!("chat completions stream error: {value}"));
            break;
        }
        if let Some(content) = value
            .pointer("/choices/0/delta/content")
            .or_else(|| value.pointer("/choices/0/text"))
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        {
            if first_token_ms.is_none() {
                first_token_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            }
            output_chars += content.chars().count();
        }
        if finish_reason.is_none() {
            finish_reason = value
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
        }
        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            completion_tokens = usage_u64(usage, "completion_tokens");
            prompt_tokens = usage_u64(usage, "prompt_tokens");
            total_tokens = usage_u64(usage, "total_tokens");
        }
    }

    ChatCorpusResult {
        sequence: prompt_case.index,
        prompt_id: prompt_case.prompt_id.clone(),
        category: prompt_case.category.clone(),
        length_bucket: prompt_case.length_bucket.clone(),
        session_id,
        prompt_chars: prompt_case.prompt.chars().count(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        ttft_ms: first_token_ms,
        completion_tokens,
        prompt_tokens,
        total_tokens,
        finish_reason,
        output_chars,
        error,
        api_error_code: api_error_code_value,
    }
}

fn error_result(
    prompt_case: &PromptCase,
    session_id: String,
    started: Instant,
    error: String,
    api_error_code: Option<String>,
) -> ChatCorpusResult {
    ChatCorpusResult {
        sequence: prompt_case.index,
        prompt_id: prompt_case.prompt_id.clone(),
        category: prompt_case.category.clone(),
        length_bucket: prompt_case.length_bucket.clone(),
        session_id,
        prompt_chars: prompt_case.prompt.chars().count(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        ttft_ms: None,
        completion_tokens: None,
        prompt_tokens: None,
        total_tokens: None,
        finish_reason: None,
        output_chars: 0,
        error: Some(error),
        api_error_code,
    }
}

fn api_error_code(value: &Value) -> Option<String> {
    value
        .pointer("/error/code")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn usage_u64(usage: &Value, field: &str) -> Option<u64> {
    usage.get(field).and_then(Value::as_u64)
}

fn prompt_cases(args: &ChatCorpusArgs) -> Result<Vec<PromptCase>> {
    let mut prompts = match args.prompt_corpus.as_ref() {
        Some(path) => {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let mut cases = Vec::new();
            for (line_index, line) in text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                cases.push(prompt_case_from_line(cases.len(), line).with_context(|| {
                    format!(
                        "read prompt corpus line {} in {}",
                        line_index + 1,
                        path.display()
                    )
                })?);
            }
            cases
        }
        None => vec![PromptCase {
            index: 0,
            prompt_id: None,
            category: None,
            length_bucket: None,
            session_group: None,
            prompt: args.prompt.clone(),
            messages: None,
        }],
    };
    if let Some(limit) = args.prompt_limit {
        prompts.truncate(limit);
    }
    if prompts.is_empty() {
        anyhow::bail!("benchmark prompt set is empty");
    }
    for (index, prompt) in prompts.iter_mut().enumerate() {
        prompt.index = index;
    }
    Ok(prompts)
}

fn prompt_case_from_line(index: usize, line: &str) -> Result<PromptCase> {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return Ok(PromptCase {
            index,
            prompt_id: None,
            category: None,
            length_bucket: None,
            session_group: None,
            prompt: line.to_string(),
            messages: None,
        });
    };
    let prompt_id = value
        .get("id")
        .or_else(|| value.get("prompt_id"))
        .and_then(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .or_else(|| value.as_i64().map(|id| id.to_string()))
        });
    let category = value
        .get("category")
        .or_else(|| value.get("family"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let length_bucket = value
        .get("length_bucket")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let session_group = value
        .get("session_group")
        .or_else(|| value.get("session_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    if let Some(messages) = value.get("messages").and_then(Value::as_array) {
        let prompt = messages
            .iter()
            .filter_map(|message| message.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(PromptCase {
            index,
            prompt_id,
            category,
            length_bucket,
            session_group,
            prompt,
            messages: Some(Value::Array(messages.clone())),
        });
    }
    let prompt = if let Some(prompt) = value.get("prompt").and_then(Value::as_str) {
        prompt.to_string()
    } else if let Some(turns) = value.get("turns").and_then(Value::as_array) {
        turns
            .iter()
            .find_map(Value::as_str)
            .context("turns did not contain a string prompt")?
            .to_string()
    } else {
        anyhow::bail!("JSONL row must include prompt, turns, or messages");
    };
    Ok(PromptCase {
        index,
        prompt_id,
        category,
        length_bucket,
        session_group,
        prompt,
        messages: None,
    })
}

fn summarize(results: &[ChatCorpusResult], total_wall_ms: f64) -> ChatCorpusSummary {
    let count = results.len();
    let errors = results
        .iter()
        .filter(|result| result.error.is_some())
        .count();
    let mut elapsed = results
        .iter()
        .filter(|result| result.error.is_none())
        .map(|result| result.elapsed_ms)
        .collect::<Vec<_>>();
    let mut ttft = results
        .iter()
        .filter(|result| result.error.is_none())
        .filter_map(|result| result.ttft_ms)
        .collect::<Vec<_>>();
    let completion_tokens = results
        .iter()
        .filter_map(|result| result.completion_tokens)
        .sum::<u64>();
    let total_tokens = results
        .iter()
        .filter_map(|result| result.total_tokens)
        .sum::<u64>();
    elapsed.sort_by(f64::total_cmp);
    ttft.sort_by(f64::total_cmp);
    ChatCorpusSummary {
        count,
        errors,
        elapsed_ms_min: elapsed.first().copied(),
        elapsed_ms_mean: mean(&elapsed),
        elapsed_ms_p50: percentile(&elapsed, 0.50),
        elapsed_ms_p95: percentile(&elapsed, 0.95),
        elapsed_ms_p99: percentile(&elapsed, 0.99),
        ttft_ms_p50: percentile(&ttft, 0.50),
        ttft_ms_p95: percentile(&ttft, 0.95),
        ttft_ms_p99: percentile(&ttft, 0.99),
        completion_tokens,
        total_tokens,
        total_wall_ms,
        completion_tok_s: rate(completion_tokens, total_wall_ms),
        total_tok_s: rate(total_tokens, total_wall_ms),
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values.get(index).copied()
}

fn rate(tokens: u64, wall_ms: f64) -> Option<f64> {
    (tokens > 0 && wall_ms > 0.0).then(|| tokens as f64 / (wall_ms / 1000.0))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn default_args() -> ChatCorpusArgs {
        ChatCorpusArgs {
            base_url: "http://127.0.0.1:9337/v1".to_string(),
            model: "org/repo:Q4_K_M".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            max_tokens: 512,
            concurrency_depth: 1,
            stream: false,
            include_usage: true,
            request_timeout_secs: 600,
            output: None,
            session_prefix: "chat-corpus-test".to_string(),
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            enable_thinking: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn prompt_cases_load_supported_jsonl_shapes() {
        let mut args = default_args();
        args.prompt_corpus = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpora/chat_corpus_fixture.jsonl"),
        );

        let cases = prompt_cases(&args).expect("fixture corpus should load");

        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].prompt_id.as_deref(), Some("plain-1"));
        assert_eq!(cases[0].category.as_deref(), Some("qa"));
        assert_eq!(cases[0].length_bucket.as_deref(), Some("short"));
        assert!(cases[0].messages.is_none());

        assert_eq!(cases[1].prompt_id.as_deref(), Some("messages-1"));
        assert_eq!(
            cases[1].prompt,
            "You are concise.\nPatch the request parser."
        );
        assert!(cases[1].messages.is_some());

        assert_eq!(cases[2].prompt_id.as_deref(), Some("turns-1"));
        assert_eq!(cases[2].prompt, "Explain the queueing risk.");
        assert_eq!(
            cases.iter().map(|case| case.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn request_body_preserves_messages_and_explicit_benchmark_knobs() {
        let mut args = default_args();
        args.stream = true;
        args.temperature = Some(0.125);
        args.top_p = Some(0.875);
        args.top_k = Some(40);
        args.seed = Some(7);
        args.enable_thinking = Some(false);
        args.reasoning_effort = Some("low".to_string());

        let prompt_case = prompt_case_from_line(
            0,
            r#"{"id":"messages-1","messages":[{"role":"system","content":"You are concise."},{"role":"user","content":"Patch the request parser."}]}"#,
        )
        .expect("messages row should parse");

        let body = request_body(&args, &prompt_case, "session-7");

        assert_eq!(body["model"], "org/repo:Q4_K_M");
        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["stream"], true);
        assert_eq!(body["user"], "session-7");
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["temperature"].as_f64(), Some(0.125));
        assert_eq!(body["top_p"].as_f64(), Some(0.875));
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["seed"], 7);
        assert_eq!(body["enable_thinking"], false);
        assert_eq!(body["reasoning_effort"], "low");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "Patch the request parser.");
    }

    #[test]
    fn request_body_omits_stream_usage_when_disabled() {
        let mut args = default_args();
        args.stream = true;
        args.include_usage = false;
        let prompt_case = prompt_case_from_line(0, "plain text prompt").expect("plain prompt");

        let body = request_body(&args, &prompt_case, "session-1");

        assert!(body.get("stream_options").is_none());
        assert_eq!(body["messages"][0]["content"], "plain text prompt");
    }

    #[test]
    fn prompt_case_preserves_session_group_for_openai_user() {
        let prompt_case = prompt_case_from_line(
            0,
            r#"{"id":"warm-1","session_group":"repo:turns","prompt":"Patch the loop."}"#,
        )
        .expect("session group row should parse");

        assert_eq!(prompt_case.session_group.as_deref(), Some("repo:turns"));

        let args = default_args();
        let body = request_body(
            &args,
            &prompt_case,
            prompt_case.session_group.as_deref().unwrap(),
        );

        assert_eq!(body["user"], "repo:turns");
    }
}
