use anyhow::{Context, Result};

use crate::{cli::shell, runtime};
use url::Url;

const OPENCODE_PROVIDER_ID: &str = "mesh";
const OPENCODE_CONFIG_ENV: &str = "OPENCODE_CONFIG_CONTENT";
const OPENCODE_API_KEY_ENV: &str = "OPENAI_API_KEY";
const OPENCODE_API_KEY_VALUE: &str = "dummy";
const OPENCODE_INSTALL_HINT: &str = "curl -fsSL https://opencode.ai/install | bash";

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenCodeLaunchSpec {
    provider_id: &'static str,
    model: String,
    config_content: String,
    api_key_env: &'static str,
    api_key_value: &'static str,
    install_hint: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenCodeTarget {
    input: String,
    api_base_url: String,
    api_models_url: String,
    management_models_url: String,
    auto_start_local_mesh: bool,
    local_port: Option<u16>,
}

fn is_loopback_or_localhost(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

fn normalize_mesh_host(host: &str) -> Result<OpenCodeTarget> {
    normalize_mesh_host_with_label(host, "mesh host")
}

fn normalize_mesh_host_with_label(host: &str, label: &str) -> Result<OpenCodeTarget> {
    const DEFAULT_API_PORT: u16 = 9337;
    const DEFAULT_MANAGEMENT_PORT: u16 = 3131;

    let trimmed = host.trim();
    if trimmed.is_empty() {
        anyhow::bail!("{label} cannot be empty");
    }

    let has_scheme = trimmed.contains("://");
    let normalized_host = if has_scheme {
        trimmed.to_string()
    } else if trimmed.parse::<u16>().is_ok() {
        format!("127.0.0.1:{trimmed}")
    } else {
        trimmed.to_string()
    };
    let mut parsed = if has_scheme {
        Url::parse(&normalized_host).with_context(|| format!("Invalid {label} URL '{trimmed}'"))?
    } else {
        Url::parse(&format!("http://{normalized_host}"))
            .with_context(|| format!("Invalid {label} '{trimmed}'"))?
    };

    let host_name = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("{label} '{trimmed}' is missing a hostname"))?
        .to_string();

    let is_local_host = is_loopback_or_localhost(&host_name);
    let should_default_api_port =
        parsed.port().is_none() && (!has_scheme || (is_local_host && parsed.scheme() == "http"));
    if should_default_api_port {
        parsed
            .set_port(Some(DEFAULT_API_PORT))
            .map_err(|_| anyhow::anyhow!("Invalid {label} '{trimmed}'"))?;
    }

    parsed.set_query(None);
    parsed.set_fragment(None);

    let mut api_base = parsed.clone();
    api_base.set_path("/v1");

    let mut api_models = api_base.clone();
    api_models.set_path("/v1/models");

    let mut management = parsed.clone();
    if !has_scheme || should_default_api_port || (is_local_host && parsed.scheme() == "http") {
        management
            .set_port(Some(DEFAULT_MANAGEMENT_PORT))
            .map_err(|_| anyhow::anyhow!("Invalid {label} '{trimmed}'"))?;
    }
    management.set_path("/api/models");

    let auto_start_local_mesh = is_local_host && parsed.scheme() == "http";

    Ok(OpenCodeTarget {
        input: trimmed.to_string(),
        api_base_url: api_base.to_string(),
        api_models_url: api_models.to_string(),
        management_models_url: management.to_string(),
        auto_start_local_mesh,
        local_port: api_base.port_or_known_default(),
    })
}

fn normalize_opencode_host(host: &str) -> Result<OpenCodeTarget> {
    normalize_mesh_host_with_label(host, "OpenCode host")
}

fn build_opencode_launch_spec(
    model_names: &[String],
    resolved_model: &str,
    api_base_url: &str,
) -> OpenCodeLaunchSpec {
    build_opencode_launch_spec_with_limits(
        model_names,
        resolved_model,
        api_base_url,
        &std::collections::HashMap::new(),
    )
}

fn build_opencode_launch_spec_with_limits(
    model_names: &[String],
    resolved_model: &str,
    api_base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> OpenCodeLaunchSpec {
    let mut models = serde_json::Map::new();
    for model in model_names {
        let mut model_obj = serde_json::Map::new();
        model_obj.insert("name".to_string(), serde_json::json!(model));

        if let Some(&Some(ctx_len)) = context_lengths.get(model) {
            let limit = serde_json::json!({
                "context": ctx_len,
                "output": ctx_len,
            });
            model_obj.insert("limit".to_string(), limit);
        }

        models.insert(model.clone(), serde_json::Value::Object(model_obj));
    }

    // Build provider object with explicit field order: name, npm, options, then models
    let mut mesh_provider = serde_json::Map::new();
    mesh_provider.insert("name".to_string(), serde_json::json!("mesh-llm"));
    mesh_provider.insert(
        "npm".to_string(),
        serde_json::json!("@ai-sdk/openai-compatible"),
    );
    mesh_provider.insert(
        "options".to_string(),
        serde_json::json!({
            "baseURL": api_base_url,
        }),
    );
    mesh_provider.insert("models".to_string(), serde_json::Value::Object(models));

    let config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            OPENCODE_PROVIDER_ID: serde_json::Value::Object(mesh_provider),
        }
    });

    OpenCodeLaunchSpec {
        provider_id: OPENCODE_PROVIDER_ID,
        model: format!("{OPENCODE_PROVIDER_ID}/{resolved_model}"),
        config_content: config.to_string(),
        api_key_env: OPENCODE_API_KEY_ENV,
        api_key_value: OPENCODE_API_KEY_VALUE,
        install_hint: OPENCODE_INSTALL_HINT,
    }
}

fn opencode_missing_binary_guidance(
    chosen: &str,
    host: &str,
    spec: &OpenCodeLaunchSpec,
) -> Vec<String> {
    vec![
        "opencode not found in PATH".to_string(),
        spec.install_hint.to_string(),
        "Then rerun through mesh-llm:".to_string(),
        format!("  mesh-llm opencode --host {host} --model {chosen}"),
        "mesh-llm injects OPENCODE_CONFIG_CONTENT automatically when launching OpenCode."
            .to_string(),
    ]
}

fn pi_missing_binary_guidance(model_arg: &str) -> Vec<String> {
    vec![
        "pi not found in PATH.".to_string(),
        "Install: npm install -g @mariozechner/pi-coding-agent".to_string(),
        "Or run manually:".to_string(),
        format!("  pi --model {}", shell::single_quote(model_arg)),
    ]
}

fn cleanup_mesh_child(mesh_child: &mut Option<std::process::Child>) {
    if let Some(ref mut child) = mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = child.kill();
        let _ = child.wait();
    }
}

async fn fetch_mesh_models(
    client: &reqwest::Client,
    models_url: &str,
    requested_model: &Option<String>,
) -> Result<(Vec<String>, String)> {
    let resp = client
        .get(models_url)
        .send()
        .await
        .with_context(|| format!("Failed to reach mesh target at {models_url}"))?;

    let body = resp
        .error_for_status()
        .with_context(|| format!("mesh target returned an error for {models_url}"))?
        .json::<serde_json::Value>()
        .await
        .with_context(|| format!("Failed to parse model list from {models_url}"))?;

    let models: Vec<String> = body["data"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|m| m["id"].as_str().map(String::from))
        .collect();

    if models.is_empty() {
        anyhow::bail!(
            "mesh target at {models_url} has no models yet (or could not be reached).\n\
             Ensure at least one serving peer is available on the mesh."
        );
    }

    let chosen = if let Some(ref model) = requested_model {
        if !models.iter().any(|name| name == model) {
            anyhow::bail!(
                "Model '{}' not available. Available: {}",
                model,
                models.join(", ")
            );
        }
        model.clone()
    } else {
        let available: Vec<(&str, f64, crate::models::ModelCapabilities)> = models
            .iter()
            .map(|name| {
                let caps = crate::models::installed_model_capabilities(name);
                (name.as_str(), 0.0, caps)
            })
            .collect();
        let agentic = crate::network::router::Classification {
            category: crate::network::router::Category::Code,
            complexity: crate::network::router::Complexity::Deep,
            needs_tools: true,
            has_media_inputs: false,
        };
        crate::network::router::pick_model_classified(&agentic, &available)
            .map(|s| s.to_string())
            .unwrap_or_else(|| models[0].clone())
    };

    eprintln!("   Models: {}", models.join(", "));
    eprintln!("   Using: {chosen}");

    Ok((models, chosen))
}

pub(crate) async fn run_goose(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (models, chosen, mut mesh_child) = runtime::check_mesh(&client, port, &model).await?;

    let goose_config_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".config")
        .join("goose")
        .join("custom_providers");
    std::fs::create_dir_all(&goose_config_dir)?;

    let provider_models: Vec<serde_json::Value> = models
        .iter()
        .map(|name| serde_json::json!({"name": name, "context_limit": 65536}))
        .collect();

    let provider = serde_json::json!({
        "name": "mesh",
        "engine": "openai",
        "display_name": "mesh-llm",
        "description": "Distributed LLM inference via mesh-llm",
        "api_key_env": "",
        "base_url": format!("http://localhost:{port}"),
        "models": provider_models,
        "timeout_seconds": 600,
        "supports_streaming": true,
        "requires_auth": false
    });

    let provider_path = goose_config_dir.join("mesh.json");
    std::fs::write(&provider_path, serde_json::to_string_pretty(&provider)?)?;
    eprintln!("✅ Wrote {}", provider_path.display());

    let goose_app = std::path::Path::new("/Applications/Goose.app");
    if goose_app.exists() {
        eprintln!("🪿 Launching Goose.app...");
        std::process::Command::new("open")
            .arg("-a")
            .arg(goose_app)
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen)
            .spawn()?;
        if mesh_child.is_some() {
            eprintln!(
                "ℹ️  mesh-llm node running in background (kill manually or use `mesh-llm stop`)"
            );
        }
    } else {
        eprintln!("🪿 Launching goose session...");
        let status = std::process::Command::new("goose")
            .arg("session")
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen)
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => eprintln!("goose exited with {s}"),
            Err(_) => {
                eprintln!("goose not found. Install: https://github.com/block/goose");
                eprintln!("Or run manually:");
                eprintln!("  GOOSE_PROVIDER=mesh GOOSE_MODEL={chosen} goose session");
            }
        }
        if let Some(ref mut c) = mesh_child {
            eprintln!("🧹 Stopping mesh-llm node we started...");
            let _ = c.kill();
            let _ = c.wait();
        }
    }
    Ok(())
}

pub(crate) async fn run_claude(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (_models, chosen, mut mesh_child) = runtime::check_mesh(&client, port, &model).await?;

    let base_url = format!("http://127.0.0.1:{port}");
    let settings = serde_json::json!({
        "env": {
            "ANTHROPIC_BASE_URL": &base_url,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": &chosen,
            "CLAUDE_CODE_SUBAGENT_MODEL": &chosen,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "DISABLE_PROMPT_CACHING": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1"
        },
        "attribution": {
            "commit": "",
            "pr": ""
        },
        "prefersReducedMotion": true,
        "terminalProgressBarEnabled": false
    });
    let settings_json = serde_json::to_string(&settings)?;

    eprintln!("🚀 Launching Claude Code with {chosen} → {base_url}\n");
    let status = std::process::Command::new("claude")
        .args(["--model", &chosen, "--settings", &settings_json])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("claude exited with {s}"),
        Err(_) => {
            eprintln!("claude not found. Install: https://docs.anthropic.com/en/docs/claude-code");
            eprintln!("Or run manually:");
            eprintln!("  ANTHROPIC_BASE_URL={base_url} ANTHROPIC_API_KEY= claude --model {chosen}");
        }
    }
    if let Some(ref mut c) = mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = c.kill();
        let _ = c.wait();
    }
    Ok(())
}

fn resolve_pi_models_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".pi")
        .join("agent")
        .join("models.json")
}

#[cfg(test)]
fn build_pi_provider_config(model_names: &[String], api_base_url: &str) -> serde_json::Value {
    build_pi_provider_config_with_limits(
        model_names,
        api_base_url,
        &std::collections::HashMap::new(),
    )
}

fn build_pi_provider_config_with_limits(
    model_names: &[String],
    api_base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> serde_json::Value {
    let models: Vec<serde_json::Value> = model_names
        .iter()
        .map(|name| {
            let mut model = serde_json::Map::new();
            model.insert("id".to_string(), serde_json::json!(name));
            model.insert("name".to_string(), serde_json::json!(name));

            if let Some(&Some(ctx_len)) = context_lengths.get(name) {
                model.insert("contextWindow".to_string(), serde_json::json!(ctx_len));
                model.insert("maxTokens".to_string(), serde_json::json!(ctx_len));
            }

            serde_json::Value::Object(model)
        })
        .collect();

    let mut provider = serde_json::Map::new();
    provider.insert("api".to_string(), serde_json::json!("openai-completions"));
    provider.insert("apiKey".to_string(), serde_json::json!("mesh"));
    provider.insert("baseUrl".to_string(), serde_json::json!(api_base_url));
    provider.insert("models".to_string(), serde_json::Value::Array(models));

    serde_json::Value::Object(provider)
}

fn load_existing_config(path: &std::path::Path) -> Result<serde_json::Value> {
    if !path.exists() {
        return Ok(serde_json::json!({}));
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse {} as JSON", path.display()))?;

    if !config.is_object() {
        anyhow::bail!("Expected {} to contain a JSON object", path.display());
    }

    Ok(config)
}

fn provider_map_mut<'a>(
    config: &'a mut serde_json::Value,
    field_name: &str,
    path: &std::path::Path,
) -> Result<&'a mut serde_json::Map<String, serde_json::Value>> {
    let config_object = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Expected {} to contain a JSON object", path.display()))?;
    let providers = config_object
        .entry(field_name.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));

    providers.as_object_mut().ok_or_else(|| {
        anyhow::anyhow!(
            "Expected '{}' in {} to be a JSON object",
            field_name,
            path.display()
        )
    })
}

fn merge_provider(
    config: &mut serde_json::Value,
    field_name: &str,
    provider_id: &str,
    provider: serde_json::Value,
    path: &std::path::Path,
) -> Result<()> {
    provider_map_mut(config, field_name, path)?.insert(provider_id.to_string(), provider);
    Ok(())
}

fn write_pi_config_with_limits(
    model_names: &[String],
    api_base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> Result<()> {
    let models_path = resolve_pi_models_path();
    write_pi_config_to_path_with_limits(&models_path, model_names, api_base_url, context_lengths)
}

#[cfg(test)]
fn write_pi_config_to_path(
    models_path: &std::path::Path,
    model_names: &[String],
    api_base_url: &str,
) -> Result<()> {
    write_pi_config_to_path_with_limits(
        models_path,
        model_names,
        api_base_url,
        &std::collections::HashMap::new(),
    )
}

fn write_pi_config_to_path_with_limits(
    models_path: &std::path::Path,
    model_names: &[String],
    api_base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> Result<()> {
    std::fs::create_dir_all(models_path.parent().expect("models path must have parent"))?;

    let mut config = load_existing_config(models_path)?;
    let provider = build_pi_provider_config_with_limits(model_names, api_base_url, context_lengths);
    merge_provider(&mut config, "providers", "mesh", provider, models_path)?;

    std::fs::write(models_path, serde_json::to_string_pretty(&config)?)?;
    eprintln!(
        "✅ Wrote mesh provider to {} ({} models)",
        models_path.display(),
        model_names.len()
    );

    Ok(())
}

#[cfg(test)]
fn write_pi_config_for_test(
    models_path: &std::path::Path,
    model_names: &[String],
    host: &str,
) -> Result<()> {
    let target = normalize_mesh_host(host)?;
    write_pi_config_to_path(models_path, model_names, &target.api_base_url)
}

pub(crate) async fn run_pi(model: Option<String>, host: &str, write: bool) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let target = normalize_mesh_host(host)?;

    let (models, chosen, mut mesh_child) = if target.auto_start_local_mesh {
        let port = target
            .local_port
            .ok_or_else(|| anyhow::anyhow!("Pi host '{}' is missing a usable port", host))?;
        let (models, chosen, child) = runtime::check_mesh(&client, port, &model).await?;
        (models, chosen, child)
    } else {
        let (models, chosen) = fetch_mesh_models(&client, &target.api_models_url, &model).await?;
        (models, chosen, None)
    };

    let context_lengths = fetch_model_context_lengths(&client, &target.management_models_url).await;
    let result = run_pi_with_mesh(
        &models,
        &chosen,
        &target.api_base_url,
        &context_lengths,
        write,
    );

    cleanup_mesh_child(&mut mesh_child);

    result
}

fn run_pi_with_mesh(
    model_names: &[String],
    chosen: &str,
    base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
    write: bool,
) -> Result<()> {
    write_pi_config_with_limits(model_names, base_url, context_lengths)?;

    if write {
        return Ok(());
    }

    let model_arg = format!("mesh/{chosen}");
    eprintln!("🚀 Launching pi with {chosen} → {base_url}\n");
    let status = std::process::Command::new("pi")
        .args(["--model", &model_arg])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("pi exited with {s}"),
        Err(_) => {
            for line in pi_missing_binary_guidance(&model_arg) {
                eprintln!("{line}");
            }
        }
    }

    Ok(())
}

pub(crate) async fn run_opencode(model: Option<String>, host: &str, write: bool) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let target = normalize_opencode_host(host)?;

    let (models, chosen, mut mesh_child) = if target.auto_start_local_mesh {
        let port = target
            .local_port
            .ok_or_else(|| anyhow::anyhow!("OpenCode host '{}' is missing a usable port", host))?;
        let (models, chosen, child) = runtime::check_mesh(&client, port, &model).await?;
        (models, chosen, child)
    } else {
        let (models, chosen) = fetch_mesh_models(&client, &target.api_models_url, &model).await?;
        (models, chosen, None)
    };

    let result = if write {
        write_opencode_config(&client, &models, &chosen, &target).await
    } else {
        let spec = build_opencode_launch_spec(&models, &chosen, &target.api_base_url);

        eprintln!(
            "🚀 Launching OpenCode with {} → {}\n",
            chosen, target.api_base_url
        );
        let status = std::process::Command::new("opencode")
            .args(["-m", &spec.model])
            .env(OPENCODE_CONFIG_ENV, &spec.config_content)
            .env(spec.api_key_env, spec.api_key_value)
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => eprintln!("opencode exited with {s}"),
            Err(_) => {
                for line in opencode_missing_binary_guidance(&chosen, &target.input, &spec) {
                    eprintln!("{line}");
                }
            }
        }
        Ok(())
    };

    cleanup_mesh_child(&mut mesh_child);

    result
}

fn resolve_opencode_config_path() -> Result<std::path::PathBuf> {
    let home_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .to_path_buf();
    resolve_opencode_config_path_from_home(&home_dir)
}

fn resolve_opencode_config_path_from_home(
    home_dir: &std::path::Path,
) -> Result<std::path::PathBuf> {
    let config_dir = home_dir.join(".config").join("opencode");

    std::fs::create_dir_all(&config_dir)?;

    let json_path = config_dir.join("opencode.json");
    let jsonc_path = config_dir.join("opencode.jsonc");

    if json_path.exists() {
        return Ok(json_path);
    }
    if jsonc_path.exists() {
        anyhow::bail!(
            "Found {} but mesh-llm only writes opencode.json. Rename or migrate it to {} and rerun `mesh-llm opencode --write`.",
            jsonc_path.display(),
            json_path.display()
        );
    }

    Ok(json_path)
}

fn merge_mesh_provider(
    config: &mut serde_json::Value,
    mesh_provider: serde_json::Value,
    config_path: &std::path::Path,
) -> Result<()> {
    merge_provider(config, "provider", "mesh", mesh_provider, config_path)
}

async fn fetch_model_context_lengths(
    client: &reqwest::Client,
    management_models_url: &str,
) -> std::collections::HashMap<String, Option<u32>> {
    let mut context_map = std::collections::HashMap::new();

    if let Ok(resp) = client.get(management_models_url).send().await {
        if let Ok(body) = resp.json::<serde_json::Value>().await {
            for model in body["mesh_models"].as_array().unwrap_or(&vec![]) {
                let name = model["name"].as_str().map(String::from);
                let ctx_len = model["context_length"].as_u64().map(|v| v as u32);
                if let Some(n) = name {
                    context_map.insert(n, ctx_len);
                }
            }
        }
    }

    context_map
}

async fn write_opencode_config(
    client: &reqwest::Client,
    model_names: &[String],
    resolved_model: &str,
    target: &OpenCodeTarget,
) -> Result<()> {
    let config_path = resolve_opencode_config_path()?;
    write_opencode_config_to_path(client, model_names, resolved_model, target, &config_path).await
}

async fn write_opencode_config_to_path(
    client: &reqwest::Client,
    model_names: &[String],
    resolved_model: &str,
    target: &OpenCodeTarget,
    config_path: &std::path::Path,
) -> Result<()> {
    std::fs::create_dir_all(config_path.parent().expect("config path must have parent"))?;

    let existing_config = load_existing_config(config_path)?;

    let context_lengths = fetch_model_context_lengths(client, &target.management_models_url).await;

    let spec = build_opencode_launch_spec_with_limits(
        model_names,
        resolved_model,
        &target.api_base_url,
        &context_lengths,
    );
    let config_value: serde_json::Value = serde_json::from_str(&spec.config_content)?;
    let mesh_provider = config_value["provider"]["mesh"].clone();

    // Merge schema if needed (for display in ordered format)
    let mut merged_config = existing_config.clone();
    if merged_config.get("$schema").is_none() {
        if let Some(schema) = config_value.get("$schema") {
            merged_config
                .as_object_mut()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Expected {} to contain a JSON object",
                        config_path.display()
                    )
                })?
                .insert("$schema".to_string(), schema.clone());
        }
    }

    merge_mesh_provider(&mut merged_config, mesh_provider.clone(), config_path)?;

    let formatted_json = serde_json::to_string_pretty(&merged_config)?;
    std::fs::write(config_path, &formatted_json)?;

    eprintln!(
        "✅ Wrote {} ({} models)",
        config_path.display(),
        model_names.len()
    );

    Ok(())
}

#[cfg(test)]
pub(crate) async fn write_opencode_config_for_test(
    config_path: &std::path::Path,
    models: &[String],
    host: &str,
) -> Result<(), anyhow::Error> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let target = normalize_opencode_host(host)?;
    write_opencode_config_to_path(
        &client,
        models,
        &models.first().cloned().unwrap_or_default(),
        &target,
        config_path,
    )
    .await
}

#[cfg(test)]
pub(crate) fn build_mesh_provider_spec_for_test(
    models: &[String],
    host: &str,
) -> serde_json::Value {
    let target = normalize_opencode_host(host).expect("valid OpenCode host");
    let spec = build_opencode_launch_spec(
        models,
        &models.first().cloned().unwrap_or_default(),
        &target.api_base_url,
    );
    let config_value: serde_json::Value =
        serde_json::from_str(&spec.config_content).expect("valid JSON");
    config_value["provider"]["mesh"].clone()
}

#[cfg(test)]
mod tests {
    use super::{
        build_mesh_provider_spec_for_test, build_opencode_launch_spec,
        build_opencode_launch_spec_with_limits, build_pi_provider_config,
        build_pi_provider_config_with_limits, cleanup_mesh_child, normalize_opencode_host,
        opencode_missing_binary_guidance, pi_missing_binary_guidance,
        resolve_opencode_config_path_from_home, write_opencode_config_for_test,
        write_pi_config_for_test, write_pi_config_to_path, OPENCODE_INSTALL_HINT,
    };

    const LOCAL_OPENCODE_HOST: &str = "127.0.0.1:9337";

    fn write_config(
        config_path: &std::path::Path,
        models: &[String],
        host: &str,
    ) -> anyhow::Result<()> {
        tokio::runtime::Runtime::new()
            .expect("test runtime")
            .block_on(write_opencode_config_for_test(config_path, models, host))
    }

    #[test]
    fn opencode_launch_spec_uses_mesh_provider_and_v1_base_url() {
        let spec = build_opencode_launch_spec(
            &[
                "GLM-4.7-Flash-Q4_K_M".to_string(),
                "bartowski/DeepSeek-R1.gguf".to_string(),
            ],
            "GLM-4.7-Flash-Q4_K_M",
            "http://127.0.0.1:9337/v1",
        );
        let config: serde_json::Value =
            serde_json::from_str(&spec.config_content).expect("valid OpenCode config JSON");

        assert_eq!(spec.provider_id, "mesh");
        assert_eq!(spec.api_key_env, "OPENAI_API_KEY");
        assert_eq!(spec.api_key_value, "dummy");
        assert_eq!(config["$schema"], "https://opencode.ai/config.json");
        assert_eq!(
            config["provider"]["mesh"]["npm"],
            "@ai-sdk/openai-compatible"
        );
        assert_eq!(config["provider"]["mesh"]["name"], "mesh-llm");
        assert_eq!(
            config["provider"]["mesh"]["options"]["baseURL"],
            "http://127.0.0.1:9337/v1"
        );
        // apiKey should NOT be in persisted config (handled at runtime via env var)
        assert!(
            config["provider"]["mesh"]["options"]
                .get("apiKey")
                .is_none(),
            "apiKey should not be in options for persisted config"
        );
        assert_eq!(
            config["provider"]["mesh"]["models"]["GLM-4.7-Flash-Q4_K_M"]["name"],
            "GLM-4.7-Flash-Q4_K_M"
        );
        assert_eq!(
            config["provider"]["mesh"]["models"]["bartowski/DeepSeek-R1.gguf"]["name"],
            "bartowski/DeepSeek-R1.gguf"
        );
        assert_eq!(
            config["provider"]["mesh"]["models"]
                .as_object()
                .map(|m| m.len()),
            Some(2)
        );
    }

    #[test]
    fn opencode_launch_spec_uses_mesh_prefixed_model() {
        let spec = build_opencode_launch_spec(
            &[
                "GLM-4.7-Flash-Q4_K_M".to_string(),
                "bartowski/DeepSeek-R1.gguf".to_string(),
            ],
            "bartowski/DeepSeek-R1.gguf",
            "http://127.0.0.1:8080/v1",
        );

        assert_eq!(spec.provider_id, "mesh");
        assert_eq!(spec.model, "mesh/bartowski/DeepSeek-R1.gguf");
    }

    #[test]
    fn opencode_install_hint_mentions_official_install_url() {
        assert!(OPENCODE_INSTALL_HINT.contains("https://opencode.ai/install"));
        assert_eq!(
            OPENCODE_INSTALL_HINT,
            "curl -fsSL https://opencode.ai/install | bash"
        );
    }

    #[test]
    fn opencode_missing_binary_reports_official_install_hint() {
        let spec = build_opencode_launch_spec(
            &[
                "GLM-4.7-Flash-Q4_K_M".to_string(),
                "bartowski/DeepSeek-R1.gguf".to_string(),
            ],
            "GLM-4.7-Flash-Q4_K_M",
            "http://127.0.0.1:9337/v1",
        );
        let lines =
            opencode_missing_binary_guidance("GLM-4.7-Flash-Q4_K_M", LOCAL_OPENCODE_HOST, &spec);

        assert_eq!(lines[0], "opencode not found in PATH");
        assert_eq!(lines[1], OPENCODE_INSTALL_HINT);
        assert_eq!(lines[2], "Then rerun through mesh-llm:");
        assert_eq!(
            lines[3],
            "  mesh-llm opencode --host 127.0.0.1:9337 --model GLM-4.7-Flash-Q4_K_M"
        );
        assert_eq!(
            lines[4],
            "mesh-llm injects OPENCODE_CONFIG_CONTENT automatically when launching OpenCode."
        );
    }

    #[test]
    fn pi_missing_binary_guidance_quotes_model_argument() {
        let lines = pi_missing_binary_guidance("mesh/Qwen's 3.6 27B");

        assert_eq!(lines[0], "pi not found in PATH.");
        assert_eq!(
            lines[1],
            "Install: npm install -g @mariozechner/pi-coding-agent"
        );
        assert_eq!(lines[2], "Or run manually:");
        assert_eq!(lines[3], "  pi --model 'mesh/Qwen'\"'\"'s 3.6 27B'");
    }

    #[test]
    fn test_write_creates_new_config_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");

        assert!(!config_path.exists());

        let models = vec!["qwen2.5-3b".to_string(), "glm-4.7-flash".to_string()];

        let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

        assert!(
            result.is_ok(),
            "write_opencode_config should succeed on new file"
        );
        assert!(config_path.exists(), "config file should be created");

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert_eq!(parsed["$schema"], "https://opencode.ai/config.json");
        assert!(parsed["provider"]["mesh"].is_object());
    }

    #[test]
    fn test_write_merges_with_existing_providers() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");

        let existing_config = serde_json::json!({
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "anthropic": {
                    "npm": "@ai-sdk/anthropic",
                    "name": "Anthropic",
                    "options": {
                        "apiKey": "{env:ANTHROPIC_API_KEY}"
                    },
                    "models": {
                        "claude-3-sonnet": { "name": "claude-3-sonnet" }
                    }
                },
                "openai": {
                    "npm": "@ai-sdk/openai",
                    "name": "OpenAI",
                    "options": {
                        "apiKey": "{env:OPENAI_API_KEY}"
                    },
                    "models": {
                        "gpt-4o": { "name": "gpt-4o" }
                    }
                }
            }
        });

        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&existing_config).unwrap(),
        )
        .expect("failed to write initial config");

        let models = vec!["qwen2.5-3b".to_string()];

        let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

        assert!(result.is_ok(), "merge should succeed");

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert_eq!(parsed["$schema"], "https://opencode.ai/config.json");
        assert!(
            parsed["provider"]["anthropic"].is_object(),
            "anthropic provider should be preserved"
        );
        assert!(
            parsed["provider"]["openai"].is_object(),
            "openai provider should be preserved"
        );
        assert!(
            parsed["provider"]["mesh"].is_object(),
            "mesh provider should be added"
        );
        assert_eq!(
            parsed["provider"]["anthropic"]["name"], "Anthropic",
            "anthropic name should be unchanged"
        );
    }

    #[test]
    fn test_write_overwrites_mesh_provider() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");

        let existing_config = serde_json::json!({
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "mesh": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "mesh-llm-old",
                    "options": {
                        "baseURL": "http://127.0.0.1:8080/v1",
                        "apiKey": "{env:OPENAI_API_KEY}"
                    },
                    "models": {
                        "old-model": { "name": "old-model" }
                    }
                }
            }
        });

        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&existing_config).unwrap(),
        )
        .expect("failed to write initial config");

        let models = vec!["qwen2.5-3b".to_string(), "deepseek-r1".to_string()];

        let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

        assert!(result.is_ok(), "overwrite should succeed");

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert_eq!(
            parsed["provider"]["mesh"]["name"], "mesh-llm",
            "mesh name should be updated"
        );
        assert_eq!(
            parsed["provider"]["mesh"]["options"]["baseURL"], "http://127.0.0.1:9337/v1",
            "baseURL should be updated to new port"
        );
        assert!(
            parsed["provider"]["mesh"]["models"]["old-model"].is_null(),
            "old model should be removed"
        );
        assert_eq!(
            parsed["provider"]["mesh"]["models"]["qwen2.5-3b"]["name"], "qwen2.5-3b",
            "new model should be present"
        );
        assert_eq!(
            parsed["provider"]["mesh"]["models"]["deepseek-r1"]["name"], "deepseek-r1",
            "second new model should be present"
        );
    }

    #[test]
    fn test_build_mesh_provider_spec_generates_correct_format() {
        let models = vec![
            "Qwen2.5-3B-Q4_K_M".to_string(),
            "bartowski/GLM-4.7-Flash-Q4_K_M".to_string(),
        ];
        let spec = build_mesh_provider_spec_for_test(&models, LOCAL_OPENCODE_HOST);

        assert!(spec.is_object(), "should return a JSON object");

        assert_eq!(
            spec["npm"], "@ai-sdk/openai-compatible",
            "npm package should match opencode format"
        );
        assert_eq!(spec["name"], "mesh-llm", "name field should be mesh-llm");
        assert!(spec["options"].is_object(), "options should be an object");
        assert_eq!(
            spec["options"]["baseURL"], "http://127.0.0.1:9337/v1",
            "baseURL should include /v1 suffix and correct port"
        );
        // apiKey is not persisted in config (handled at runtime via env var)
        assert!(
            spec["options"].get("apiKey").is_none(),
            "apiKey should not be in options for persisted config"
        );
        assert!(spec["models"].is_object(), "models should be an object");
        assert_eq!(
            spec["models"]["Qwen2.5-3B-Q4_K_M"]["name"], "Qwen2.5-3B-Q4_K_M",
            "model name should match input"
        );
        assert_eq!(
            spec["models"]["bartowski/GLM-4.7-Flash-Q4_K_M"]["name"],
            "bartowski/GLM-4.7-Flash-Q4_K_M",
            "model with slash in name should work correctly"
        );
    }

    #[test]
    fn test_write_handles_empty_models_list() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");

        let models: Vec<String> = vec![];

        let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

        assert!(result.is_ok(), "should succeed with empty models list");
        assert!(config_path.exists(), "config file should still be created");

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert!(
            parsed["provider"]["mesh"]["models"].is_object(),
            "models field should exist even when empty"
        );
        assert_eq!(
            parsed["provider"]["mesh"]["models"]
                .as_object()
                .map(|m| m.len())
                .unwrap_or(0),
            0,
            "models object should be empty"
        );
    }

    #[test]
    fn test_write_handles_special_characters_in_model_names() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");

        let models = vec![
            "model-with-dashes".to_string(),
            "model_with_underscores".to_string(),
            "ModelWithCamelCase".to_string(),
            "bartowski/model-v2.5-Q4_K_M.gguf".to_string(),
            "1-model-starting-with-number".to_string(),
        ];

        let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

        assert!(
            result.is_ok(),
            "should succeed with special character model names"
        );

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        for model in &models {
            assert!(
                !parsed["provider"]["mesh"]["models"][model].is_null(),
                "model '{}' should be present in config",
                model
            );
            assert_eq!(
                parsed["provider"]["mesh"]["models"][model]["name"], *model,
                "model name should match exactly"
            );
        }
    }

    #[test]
    fn test_write_preserves_existing_file_schema() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");

        let existing_config = serde_json::json!({
            "$schema": "https://opencode.ai/config.json",
            "$customField": "preserve-me",
            "provider": {}
        });

        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&existing_config).unwrap(),
        )
        .expect("failed to write initial config");

        let models = vec!["qwen".to_string()];

        let result = write_config(&config_path, &models, LOCAL_OPENCODE_HOST);

        assert!(result.is_ok());

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert_eq!(
            parsed["$schema"], "https://opencode.ai/config.json",
            "schema should be preserved"
        );
        assert_eq!(
            parsed["$customField"], "preserve-me",
            "custom fields at root level should be preserved"
        );
    }

    #[test]
    fn pi_provider_config_lists_all_mesh_models_with_models_key_last() {
        let models = vec!["Qwen 3.6 27B".to_string(), "Qwen 3.5 4B".to_string()];
        let provider = build_pi_provider_config(&models, "http://localhost:9337/v1");

        assert_eq!(provider["api"], "openai-completions");
        assert_eq!(provider["apiKey"], "mesh");
        assert_eq!(provider["baseUrl"], "http://localhost:9337/v1");
        assert_eq!(provider["models"].as_array().map(Vec::len), Some(2));
        assert_eq!(provider["models"][0]["id"], "Qwen 3.6 27B");
        assert_eq!(provider["models"][0]["name"], "Qwen 3.6 27B");
        assert_eq!(provider["models"][1]["id"], "Qwen 3.5 4B");
        assert_eq!(provider["models"][1]["name"], "Qwen 3.5 4B");

        let key_order: Vec<&str> = provider
            .as_object()
            .expect("provider is object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(key_order.last(), Some(&"models"));
    }

    #[test]
    fn pi_provider_config_includes_context_window_and_max_tokens_when_known() {
        let models = vec![
            "Qwen3.6-27B-UD-Q4_K_XL".to_string(),
            "Qwen3.5-4B-UD-Q4_K_XL".to_string(),
            "Unknown-Model".to_string(),
        ];
        let mut context_lengths = std::collections::HashMap::new();
        context_lengths.insert("Qwen3.6-27B-UD-Q4_K_XL".to_string(), Some(262144));
        context_lengths.insert("Qwen3.5-4B-UD-Q4_K_XL".to_string(), Some(65536));
        context_lengths.insert("Unknown-Model".to_string(), None);

        let provider = build_pi_provider_config_with_limits(
            &models,
            "http://carrack.patio51.com:9337/v1",
            &context_lengths,
        );

        assert_eq!(provider["models"][0]["contextWindow"], 262144);
        assert_eq!(provider["models"][0]["maxTokens"], 262144);
        assert_eq!(provider["models"][1]["contextWindow"], 65536);
        assert_eq!(provider["models"][1]["maxTokens"], 65536);
        assert!(
            provider["models"][2]["contextWindow"].is_null(),
            "model with unknown context_length should omit contextWindow"
        );
        assert!(
            provider["models"][2]["maxTokens"].is_null(),
            "model with unknown context_length should omit maxTokens"
        );

        let key_order: Vec<&str> = provider
            .as_object()
            .expect("provider is object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(key_order.last(), Some(&"models"));
    }

    #[test]
    fn pi_write_creates_provider_and_preserves_other_providers() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("models.json");
        let existing_config = serde_json::json!({
            "providers": {
                "anthropic": {
                    "api": "anthropic",
                    "apiKey": "preserve-me",
                    "models": [{ "id": "claude" }]
                }
            }
        });
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&existing_config).unwrap(),
        )
        .expect("failed to write initial config");

        let models = vec!["Qwen 3.6 27B".to_string(), "Qwen 3.5 4B".to_string()];
        write_pi_config_to_path(&config_path, &models, "http://localhost:9337/v1")
            .expect("pi write should succeed");

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert_eq!(parsed["providers"]["anthropic"]["apiKey"], "preserve-me");
        assert_eq!(parsed["providers"]["mesh"]["api"], "openai-completions");
        assert_eq!(
            parsed["providers"]["mesh"]["baseUrl"],
            "http://localhost:9337/v1"
        );
        assert_eq!(
            parsed["providers"]["mesh"]["models"]
                .as_array()
                .map(Vec::len),
            Some(2)
        );
        assert!(
            !parsed["providers"]["mesh"]["models"]
                .as_array()
                .expect("models is array")
                .iter()
                .any(|model| model["id"] == "auto"),
            "pi --write should list mesh models, not add a synthetic auto model"
        );
    }

    #[test]
    fn pi_write_uses_normalized_remote_host_as_base_url() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("models.json");
        let models = vec![
            "Qwen3.5-4B-UD-Q4_K_XL".to_string(),
            "Qwen3.6-27B-UD-Q4_K_XL".to_string(),
        ];

        write_pi_config_for_test(
            &config_path,
            &models,
            "https://carrack.patio51.com:9443/custom/path",
        )
        .expect("pi write should succeed with a full remote URL");

        let content = std::fs::read_to_string(&config_path).expect("failed to read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert_eq!(
            parsed["providers"]["mesh"]["baseUrl"],
            "https://carrack.patio51.com:9443/v1"
        );
        assert_eq!(parsed["providers"]["mesh"]["models"][0]["id"], models[0]);
        assert_eq!(parsed["providers"]["mesh"]["models"][1]["id"], models[1]);

        let key_order: Vec<&str> = parsed["providers"]["mesh"]
            .as_object()
            .expect("provider is object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(key_order.last(), Some(&"models"));
    }

    #[test]
    fn pi_write_rejects_invalid_json_without_clobbering_config() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("models.json");
        std::fs::write(&config_path, "not-json").expect("failed to write invalid config");

        let err = write_pi_config_to_path(
            &config_path,
            &["Qwen 3.6 27B".to_string()],
            "http://localhost:9337/v1",
        )
        .expect_err("invalid JSON should fail");

        assert!(err.to_string().contains("Failed to parse"));
        assert_eq!(
            std::fs::read_to_string(&config_path).expect("failed to reread config"),
            "not-json"
        );
    }

    #[test]
    fn pi_write_rejects_non_object_providers() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("models.json");
        std::fs::write(&config_path, r#"{"providers": []}"#)
            .expect("failed to write invalid providers config");

        let err = write_pi_config_to_path(
            &config_path,
            &["Qwen 3.6 27B".to_string()],
            "http://localhost:9337/v1",
        )
        .expect_err("array providers should fail");

        assert!(err.to_string().contains("providers"));
        assert!(err.to_string().contains("object"));
    }

    #[test]
    fn opencode_write_rejects_non_object_provider() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"provider": []}"#)
            .expect("failed to write invalid provider config");

        let result = write_config(&config_path, &["qwen".to_string()], LOCAL_OPENCODE_HOST);

        let err = result.expect_err("array provider should fail");
        assert!(err.to_string().contains("provider"));
        assert!(err.to_string().contains("object"));
    }

    #[test]
    fn test_build_opencode_launch_spec_with_limits_includes_context_length() {
        let mut context_lengths = std::collections::HashMap::new();
        context_lengths.insert("Qwen3.5-27B".to_string(), Some(262144));
        context_lengths.insert("Gemma-7B".to_string(), Some(8192));
        context_lengths.insert("Llama-3B".to_string(), None);

        let models = vec![
            "Qwen3.5-27B".to_string(),
            "Gemma-7B".to_string(),
            "Llama-3B".to_string(),
        ];

        let spec = build_opencode_launch_spec_with_limits(
            &models,
            "Qwen3.5-27B",
            "http://127.0.0.1:9337/v1",
            &context_lengths,
        );
        let config: serde_json::Value =
            serde_json::from_str(&spec.config_content).expect("valid JSON");

        assert_eq!(
            config["provider"]["mesh"]["models"]["Qwen3.5-27B"]["name"],
            "Qwen3.5-27B"
        );
        assert_eq!(
            config["provider"]["mesh"]["models"]["Qwen3.5-27B"]["limit"]["context"],
            262144
        );
        assert_eq!(
            config["provider"]["mesh"]["models"]["Qwen3.5-27B"]["limit"]["output"],
            262144
        );

        assert_eq!(
            config["provider"]["mesh"]["models"]["Gemma-7B"]["name"],
            "Gemma-7B"
        );
        assert_eq!(
            config["provider"]["mesh"]["models"]["Gemma-7B"]["limit"]["context"],
            8192
        );
        assert_eq!(
            config["provider"]["mesh"]["models"]["Gemma-7B"]["limit"]["output"],
            8192
        );

        assert_eq!(
            config["provider"]["mesh"]["models"]["Llama-3B"]["name"],
            "Llama-3B"
        );
        assert!(
            config["provider"]["mesh"]["models"]["Llama-3B"]["limit"].is_null(),
            "model with None context_length should not have limit field"
        );
    }

    #[test]
    fn opencode_host_normalization_defaults_bare_host_ports_and_management_lookup() {
        let target = normalize_opencode_host("mesh.example.com").expect("valid host");

        assert_eq!(target.api_base_url, "http://mesh.example.com:9337/v1");
        assert_eq!(
            target.api_models_url,
            "http://mesh.example.com:9337/v1/models"
        );
        assert_eq!(
            target.management_models_url,
            "http://mesh.example.com:3131/api/models"
        );
        assert!(!target.auto_start_local_mesh);
    }

    #[test]
    fn opencode_host_normalization_treats_bare_port_as_loopback_api_port() {
        let target = normalize_opencode_host("9443").expect("valid port-only host");

        assert_eq!(target.api_base_url, "http://127.0.0.1:9443/v1");
        assert_eq!(target.api_models_url, "http://127.0.0.1:9443/v1/models");
        assert_eq!(
            target.management_models_url,
            "http://127.0.0.1:3131/api/models"
        );
        assert!(target.auto_start_local_mesh);
        assert_eq!(target.local_port, Some(9443));
    }

    #[test]
    fn opencode_host_normalization_defaults_scheme_loopback_to_mesh_ports() {
        let localhost = normalize_opencode_host("http://localhost").expect("valid localhost URL");
        let loopback = normalize_opencode_host("http://127.0.0.1").expect("valid loopback URL");

        assert_eq!(localhost.api_base_url, "http://localhost:9337/v1");
        assert_eq!(localhost.api_models_url, "http://localhost:9337/v1/models");
        assert_eq!(
            localhost.management_models_url,
            "http://localhost:3131/api/models"
        );
        assert!(localhost.auto_start_local_mesh);
        assert_eq!(localhost.local_port, Some(9337));

        assert_eq!(loopback.api_base_url, "http://127.0.0.1:9337/v1");
        assert_eq!(
            loopback.management_models_url,
            "http://127.0.0.1:3131/api/models"
        );
        assert!(loopback.auto_start_local_mesh);
        assert_eq!(loopback.local_port, Some(9337));
    }

    #[test]
    fn opencode_host_normalization_uses_management_port_for_explicit_loopback_api_urls() {
        let localhost =
            normalize_opencode_host("http://localhost:9337").expect("valid localhost URL");
        let loopback =
            normalize_opencode_host("http://127.0.0.1:9443").expect("valid loopback URL");

        assert_eq!(localhost.api_base_url, "http://localhost:9337/v1");
        assert_eq!(
            localhost.management_models_url,
            "http://localhost:3131/api/models"
        );
        assert!(localhost.auto_start_local_mesh);
        assert_eq!(localhost.local_port, Some(9337));

        assert_eq!(loopback.api_base_url, "http://127.0.0.1:9443/v1");
        assert_eq!(
            loopback.management_models_url,
            "http://127.0.0.1:3131/api/models"
        );
        assert!(loopback.auto_start_local_mesh);
        assert_eq!(loopback.local_port, Some(9443));
    }

    #[test]
    fn opencode_host_validation_mentions_opencode_host() {
        let err = normalize_opencode_host("   ").expect_err("empty host should fail");

        assert!(err.to_string().contains("OpenCode host"));
        assert!(!err.to_string().contains("mesh host"));
    }

    #[test]
    fn opencode_host_normalization_does_not_auto_start_https_loopback() {
        let target = normalize_opencode_host("https://localhost:9337").expect("valid HTTPS URL");

        assert_eq!(target.api_base_url, "https://localhost:9337/v1");
        assert_eq!(
            target.management_models_url,
            "https://localhost:9337/api/models"
        );
        assert!(!target.auto_start_local_mesh);
        assert_eq!(target.local_port, Some(9337));
    }

    #[test]
    fn context_length_lookup_is_best_effort_and_returns_empty_map_on_failure() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(50))
            .build()
            .expect("client should build");

        let context_lengths = tokio::runtime::Runtime::new()
            .expect("test runtime")
            .block_on(super::fetch_model_context_lengths(
                &client,
                "http://127.0.0.1:9/api/models",
            ));

        assert!(context_lengths.is_empty());
    }

    #[test]
    fn opencode_host_normalization_preserves_full_url_origin() {
        let target = normalize_opencode_host("https://mesh.example.com:9443/custom/path")
            .expect("valid URL");

        assert_eq!(target.api_base_url, "https://mesh.example.com:9443/v1");
        assert_eq!(
            target.management_models_url,
            "https://mesh.example.com:9443/api/models"
        );
        assert!(!target.auto_start_local_mesh);
    }

    #[test]
    fn opencode_host_normalization_marks_loopback_targets_for_auto_start() {
        let localhost = normalize_opencode_host("127.0.0.1").expect("valid loopback host");
        let remote = normalize_opencode_host("https://mesh.example.com").expect("valid host");

        assert!(localhost.auto_start_local_mesh);
        assert_eq!(localhost.local_port, Some(9337));
        assert!(!remote.auto_start_local_mesh);
    }

    #[test]
    fn resolve_opencode_config_path_rejects_jsonc_only_configs() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let config_dir = temp_dir.path().join(".config").join("opencode");
        std::fs::create_dir_all(&config_dir).expect("failed to create config dir");
        let jsonc_path = config_dir.join("opencode.jsonc");
        std::fs::write(&jsonc_path, "{/* comments */}").expect("failed to write jsonc config");

        let err = resolve_opencode_config_path_from_home(temp_dir.path())
            .expect_err("jsonc-only config should be rejected");
        let rendered = err.to_string();

        assert!(rendered.contains("only writes opencode.json"));
        assert!(rendered.contains("Rename or migrate"));
    }

    #[test]
    fn cleanup_mesh_child_stops_spawned_process() {
        let mut child = Some(
            std::process::Command::new("sleep")
                .arg("30")
                .spawn()
                .expect("failed to spawn test child"),
        );

        cleanup_mesh_child(&mut child);

        assert!(child.is_some());
        let status = child
            .as_mut()
            .expect("child handle retained")
            .try_wait()
            .expect("wait should succeed");
        assert!(status.is_some(), "child should be exited after cleanup");
    }
}
