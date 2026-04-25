use anyhow::{Context, Result};

use crate::runtime;
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

fn normalize_opencode_host(host: &str) -> Result<OpenCodeTarget> {
    const DEFAULT_API_PORT: u16 = 9337;
    const DEFAULT_MANAGEMENT_PORT: u16 = 3131;

    let trimmed = host.trim();
    if trimmed.is_empty() {
        anyhow::bail!("OpenCode host cannot be empty");
    }

    let has_scheme = trimmed.contains("://");
    let mut parsed = if has_scheme {
        Url::parse(trimmed).with_context(|| format!("Invalid OpenCode host URL '{trimmed}'"))?
    } else {
        Url::parse(&format!("http://{trimmed}"))
            .with_context(|| format!("Invalid OpenCode host '{trimmed}'"))?
    };

    let host_name = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("OpenCode host '{trimmed}' is missing a hostname"))?
        .to_string();

    if !has_scheme && parsed.port().is_none() {
        parsed
            .set_port(Some(DEFAULT_API_PORT))
            .map_err(|_| anyhow::anyhow!("Invalid OpenCode host '{trimmed}'"))?;
    }

    parsed.set_query(None);
    parsed.set_fragment(None);

    let mut api_base = parsed.clone();
    api_base.set_path("/v1");

    let mut api_models = api_base.clone();
    api_models.set_path("/v1/models");

    let mut management = parsed.clone();
    if !has_scheme {
        management
            .set_port(Some(DEFAULT_MANAGEMENT_PORT))
            .map_err(|_| anyhow::anyhow!("Invalid OpenCode host '{trimmed}'"))?;
    }
    management.set_path("/api/models");

    let auto_start_local_mesh = is_loopback_or_localhost(&host_name);

    Ok(OpenCodeTarget {
        input: trimmed.to_string(),
        api_base_url: api_base.to_string(),
        api_models_url: api_models.to_string(),
        management_models_url: management.to_string(),
        auto_start_local_mesh,
        local_port: api_base.port_or_known_default(),
    })
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

fn cleanup_mesh_child(mesh_child: &mut Option<std::process::Child>) {
    if let Some(ref mut child) = mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = child.kill();
        let _ = child.wait();
    }
}

async fn fetch_opencode_models(
    client: &reqwest::Client,
    models_url: &str,
    requested_model: &Option<String>,
) -> Result<(Vec<String>, String)> {
    let resp = client
        .get(models_url)
        .send()
        .await
        .with_context(|| format!("Failed to reach OpenCode target at {models_url}"))?;

    let body = resp
        .error_for_status()
        .with_context(|| format!("OpenCode target returned an error for {models_url}"))?
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
            "OpenCode target at {models_url} has no models yet (or could not be reached).\n\
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

pub(crate) async fn run_pi(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (_models, chosen, mut mesh_child) = runtime::check_mesh(&client, port, &model).await?;

    let base_url = format!("http://localhost:{port}/v1");

    // Write/merge mesh provider into ~/.pi/agent/models.json
    let pi_agent_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".pi")
        .join("agent");
    std::fs::create_dir_all(&pi_agent_dir)?;

    let models_path = pi_agent_dir.join("models.json");
    let mut config = if models_path.exists() {
        let content = std::fs::read_to_string(&models_path)?;
        serde_json::from_str::<serde_json::Value>(&content)
            .unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let providers = config
        .as_object_mut()
        .expect("config must be an object")
        .entry("providers".to_string())
        .or_insert_with(|| serde_json::json!({}));

    let mut model_entries = vec![serde_json::json!({ "id": "auto" })];
    if model.is_some() && chosen != "auto" {
        model_entries.push(serde_json::json!({ "id": &chosen }));
    }

    providers
        .as_object_mut()
        .expect("providers must be an object")
        .insert(
            "mesh".to_string(),
            serde_json::json!({
                "api": "openai-completions",
                "apiKey": "mesh",
                "baseUrl": &base_url,
                "models": model_entries
            }),
        );

    std::fs::write(&models_path, serde_json::to_string_pretty(&config)?)?;
    eprintln!("✅ Wrote mesh provider to {}", models_path.display());

    let model_arg = if model.is_some() {
        format!("mesh/{chosen}")
    } else {
        "mesh/auto".to_string()
    };
    eprintln!("🚀 Launching pi with {chosen} → {base_url}\n");
    let status = std::process::Command::new("pi")
        .args(["--model", &model_arg])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("pi exited with {s}"),
        Err(_) => {
            eprintln!("pi not found in PATH.");
            eprintln!("Install: npm install -g @mariozechner/pi-coding-agent");
            eprintln!("Or run manually:");
            eprintln!("  pi --model {model_arg}");
        }
    }
    cleanup_mesh_child(&mut mesh_child);
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
        let (models, chosen) =
            fetch_opencode_models(&client, &target.api_models_url, &model).await?;
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

fn load_existing_config(path: &std::path::Path) -> Result<serde_json::Value> {
    if !path.exists() {
        return Ok(serde_json::json!({}));
    }

    let content = std::fs::read_to_string(path)?;

    serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", path.display(), e))
}

fn merge_mesh_provider(config: &mut serde_json::Value, mesh_provider: serde_json::Value) {
    let provider = config
        .as_object_mut()
        .expect("config must be an object")
        .entry("provider".to_string())
        .or_insert_with(|| serde_json::Map::new().into())
        .as_object_mut()
        .expect("provider must be an object");

    provider.insert("mesh".to_string(), mesh_provider);
}

async fn fetch_model_context_lengths(
    client: &reqwest::Client,
    management_models_url: &str,
) -> Result<std::collections::HashMap<String, Option<u32>>> {
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

    Ok(context_map)
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

    let context_lengths =
        fetch_model_context_lengths(client, &target.management_models_url).await?;

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
                .expect("config is object")
                .insert("$schema".to_string(), schema.clone());
        }
    }

    merge_mesh_provider(&mut merged_config, mesh_provider.clone());

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
        build_opencode_launch_spec_with_limits, cleanup_mesh_child, normalize_opencode_host,
        opencode_missing_binary_guidance, resolve_opencode_config_path_from_home,
        write_opencode_config_for_test, OPENCODE_INSTALL_HINT,
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
