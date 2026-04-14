use anyhow::Result;

use crate::runtime;

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

fn build_opencode_launch_spec(
    model_names: &[String],
    resolved_model: &str,
    port: u16,
) -> OpenCodeLaunchSpec {
    let mut models = serde_json::Map::new();
    for model in model_names {
        models.insert(
            model.clone(),
            serde_json::json!({
                "name": model,
            }),
        );
    }

    let config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            OPENCODE_PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "mesh-llm",
                "options": {
                    "baseURL": format!("http://127.0.0.1:{port}/v1"),
                    "apiKey": format!("{{env:{OPENCODE_API_KEY_ENV}}}"),
                },
                "models": models,
            }
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
    port: u16,
    spec: &OpenCodeLaunchSpec,
) -> Vec<String> {
    vec![
        "opencode not found in PATH".to_string(),
        spec.install_hint.to_string(),
        "Manual rerun examples:".to_string(),
        format!("  mesh-llm opencode --port {port} --model {chosen}"),
        format!(
            "  {}='{}' {}='{}' opencode -m {}",
            OPENCODE_CONFIG_ENV,
            spec.config_content,
            spec.api_key_env,
            spec.api_key_value,
            spec.model,
        ),
    ]
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

pub(crate) async fn run_opencode(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (models, chosen, mut mesh_child) = runtime::check_mesh(&client, port, &model).await?;
    let spec = build_opencode_launch_spec(&models, &chosen, port);

    eprintln!(
        "🚀 Launching OpenCode with {} → http://127.0.0.1:{port}/v1\n",
        chosen
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
            for line in opencode_missing_binary_guidance(&chosen, port, &spec) {
                eprintln!("{line}");
            }
        }
    }
    if let Some(ref mut c) = mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = c.kill();
        let _ = c.wait();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_opencode_launch_spec, opencode_missing_binary_guidance, OPENCODE_INSTALL_HINT,
    };

    #[test]
    fn opencode_launch_spec_uses_mesh_provider_and_v1_base_url() {
        let spec = build_opencode_launch_spec(
            &[
                "GLM-4.7-Flash-Q4_K_M".to_string(),
                "bartowski/DeepSeek-R1.gguf".to_string(),
            ],
            "GLM-4.7-Flash-Q4_K_M",
            9337,
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
        assert_eq!(
            config["provider"]["mesh"]["options"]["apiKey"],
            "{env:OPENAI_API_KEY}"
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
            8080,
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
            9337,
        );
        let lines = opencode_missing_binary_guidance("GLM-4.7-Flash-Q4_K_M", 9337, &spec);

        assert_eq!(lines[0], "opencode not found in PATH");
        assert_eq!(lines[1], OPENCODE_INSTALL_HINT);
        assert_eq!(lines[2], "Manual rerun examples:");
        assert_eq!(
            lines[3],
            "  mesh-llm opencode --port 9337 --model GLM-4.7-Flash-Q4_K_M"
        );
        assert!(lines[4].contains("opencode -m mesh/GLM-4.7-Flash-Q4_K_M"));
        assert!(lines[4].contains("OPENCODE_CONFIG_CONTENT='{"));
        assert!(lines[4].contains("OPENAI_API_KEY='dummy'"));
    }
}
