//! CLI command implementations — standalone commands that operate on a running
//! mesh node or configure external tools (Goose, Claude, etc.).
//!
//! These are invoked from `cli/commands/mod.rs` via the `app::` namespace.

use crate::cli::{Cli, PluginCommand};
use crate::{blackboard, mesh, nostr, plugin, router};
use anyhow::{Context, Result};

pub(crate) async fn run_discover(
    model: Option<String>,
    min_vram: Option<f64>,
    region: Option<String>,
    auto_join: bool,
    relays: Vec<String>,
) -> Result<()> {
    let relays = super::nostr_relays(&relays);

    let filter = nostr::MeshFilter {
        model,
        min_vram_gb: min_vram,
        region,
    };

    eprintln!("🔍 Searching Nostr relays for mesh-llm meshes...");
    let meshes = nostr::discover(&relays, &filter, None).await?;

    if meshes.is_empty() {
        eprintln!("No meshes found.");
        if filter.model.is_some() || filter.min_vram_gb.is_some() || filter.region.is_some() {
            eprintln!("Try broader filters or check relays.");
        }
        return Ok(());
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let last_mesh_id = mesh::load_last_mesh_id();
    eprintln!("Found {} mesh(es):\n", meshes.len());
    for (i, mesh) in meshes.iter().enumerate() {
        let score = nostr::score_mesh(mesh, now, last_mesh_id.as_deref());
        let age = now.saturating_sub(mesh.published_at);
        let freshness = if age < 120 {
            "fresh"
        } else if age < 300 {
            "ok"
        } else {
            "stale"
        };
        let capacity = if mesh.listing.max_clients > 0 {
            format!(
                "{}/{} clients",
                mesh.listing.client_count, mesh.listing.max_clients
            )
        } else {
            format!("{} clients", mesh.listing.client_count)
        };
        eprintln!(
            "  [{}] {} (score: {}, {}, {})",
            i + 1,
            mesh,
            score,
            freshness,
            capacity
        );
        let token = &mesh.listing.invite_token;
        let display_token = if token.len() > 40 {
            format!("{}...{}", &token[..20], &token[token.len() - 12..])
        } else {
            token.clone()
        };
        if !mesh.listing.on_disk.is_empty() {
            eprintln!("      on disk: {}", mesh.listing.on_disk.join(", "));
        }
        eprintln!("      token: {}", display_token);
        eprintln!();
    }

    if auto_join {
        let best = &meshes[0];
        eprintln!("Auto-joining best match: {}", best);
        eprintln!("\nRun:");
        eprintln!("  mesh-llm --join {}", best.listing.invite_token);
        // Print the full token so it can be piped
        println!("{}", best.listing.invite_token);
    } else {
        eprintln!("To join a mesh:");
        eprintln!("  mesh-llm --join <token>");
        eprintln!("\nOr use `mesh-llm discover --join` to auto-join the best match.");
    }

    Ok(())
}

/// Drop a model from the mesh by sending a control request to the running instance.
pub(crate) fn run_stop() -> Result<()> {
    let mut killed = 0u32;
    for name in &["llama-server", "rpc-server", "mesh-llm"] {
        if crate::launch::terminate_process_by_name(name) {
            eprintln!("🧹 Stopped {name}");
            killed += 1;
        }
    }
    if killed == 0 {
        eprintln!("Nothing running.");
    }
    Ok(())
}

/// Ensure mesh-llm is running on `port`, then return (available_models, chosen_model, spawned_child).
///
/// Launcher behavior: if nothing is listening yet, auto-start `mesh-llm --client --auto`
/// (client node — tunnels to mesh peers without publishing to Nostr).
/// Returns the child process handle if we spawned one, so callers can clean up on exit.
async fn check_mesh(
    client: &reqwest::Client,
    port: u16,
    model: &Option<String>,
) -> Result<(Vec<String>, String, Option<std::process::Child>)> {
    let url = format!("http://127.0.0.1:{port}/v1/models");

    // If no local mesh API is up, start a full auto-join node in the background.
    let mut child: Option<std::process::Child> = None;
    if client.get(&url).send().await.is_err() {
        eprintln!("🔍 No mesh-llm on port {port} — starting background auto-join node...");
        let exe = std::env::current_exe().unwrap_or_else(|_| "mesh-llm".into());
        child = Some(
            std::process::Command::new(&exe)
                .args(["--client", "--auto", "--port", &port.to_string()])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to start mesh-llm node")?,
        );
    }

    // Wait for API/models readiness.
    let mut models: Vec<String> = Vec::new();
    for i in 0..40 {
        if let Ok(resp) = client.get(&url).send().await {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                models = body["data"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect();
                if !models.is_empty() {
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        if i % 5 == 4 {
            eprintln!(
                "   Waiting for mesh/models... ({:.0}s)",
                (i + 1) as f64 * 3.0
            );
        }
    }

    if models.is_empty() {
        // Clean up the child we spawned before bailing
        if let Some(mut c) = child {
            let _ = c.kill();
        }
        anyhow::bail!(
            "mesh-llm on port {port} has no models yet (or could not be reached).\n\
             Ensure at least one serving peer is available on the mesh."
        );
    }

    let chosen = if let Some(ref m) = model {
        if !models.iter().any(|n| n == m) {
            if let Some(mut c) = child {
                let _ = c.kill();
                let _ = c.wait();
            }
            anyhow::bail!(
                "Model '{}' not available. Available: {}",
                m,
                models.join(", ")
            );
        }
        m.clone()
    } else {
        // Pick the strongest tool-capable model for agentic work.
        let available: Vec<(&str, f64)> = models.iter().map(|n| (n.as_str(), 0.0)).collect();
        let agentic = router::Classification {
            category: router::Category::Code,
            complexity: router::Complexity::Deep,
            needs_tools: true,
        };
        router::pick_model_classified(&agentic, &available)
            .map(|s| s.to_string())
            .unwrap_or_else(|| models[0].clone())
    };
    eprintln!("   Models: {}", models.join(", "));
    eprintln!("   Using: {chosen}");
    Ok((models, chosen, child))
}

pub(crate) async fn run_goose(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (models, chosen, mut _mesh_child) = check_mesh(&client, port, &model).await?;

    // Write custom provider JSON
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

    // Launch Goose
    let goose_app = std::path::Path::new("/Applications/Goose.app");
    if goose_app.exists() {
        eprintln!("🪿 Launching Goose.app...");
        std::process::Command::new("open")
            .arg("-a")
            .arg(goose_app)
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen)
            .spawn()?;
        // Goose.app is a GUI — can't wait for it. Mesh stays running.
        if _mesh_child.is_some() {
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
        // CLI goose exited — clean up mesh if we started it
        if let Some(ref mut c) = _mesh_child {
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
    let (_models, chosen, mut _mesh_child) = check_mesh(&client, port, &model).await?;

    // Configure and launch Claude Code
    // llama-server natively serves the Anthropic /v1/messages API, and
    // mesh-llm's TCP tunnel passes it through transparently. No proxy needed.
    let base_url = format!("http://127.0.0.1:{port}");
    // Settings optimized for local LLMs.
    // CLAUDE_CODE_ATTRIBUTION_HEADER=0 is critical — without it, Claude Code
    // prepends a changing attribution header that invalidates the KV cache on
    // every request, making inference ~90% slower. See:
    // https://unsloth.ai/docs/basics/claude-code
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
    // Claude exited — clean up mesh if we started it
    if let Some(ref mut c) = _mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = c.kill();
        let _ = c.wait();
    }
    Ok(())
}

pub(crate) async fn run_blackboard(
    text: Option<String>,
    search: Option<String>,
    from: Option<String>,
    since_hours: Option<f64>,
    limit: usize,
    port: u16,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let base = format!("http://127.0.0.1:{port}");

    // Quick connectivity check
    let status_resp = client.get(format!("{base}/api/status")).send().await;
    if status_resp.is_err() {
        eprintln!("No mesh-llm node running on port {port}.");
        eprintln!();
        eprintln!("Blackboard requires a running mesh node:");
        eprintln!("  Private mesh:  mesh-llm --client  (share the join token printed out)");
        eprintln!("  Join a mesh:   mesh-llm --client --join <token>");
        eprintln!("  Public mesh:   mesh-llm --client --auto");
        eprintln!();
        eprintln!("See https://github.com/michaelneale/mesh-llm for setup guide.");
        std::process::exit(1);
    }

    // Check if blackboard is enabled on this node
    let feed_check = client
        .get(format!("{base}/api/blackboard/feed?limit=1"))
        .send()
        .await;
    if let Ok(resp) = feed_check {
        if resp.status().as_u16() == 404 {
            eprintln!("Mesh is running but blackboard is disabled on that node.");
            eprintln!("Re-enable it in the mesh config if you want to use the blackboard plugin.");
            std::process::exit(1);
        }
    }

    // Default: 24h for feed/search, override with --since
    let default_hours = 24.0;
    let since_secs = {
        let hours = since_hours.unwrap_or(default_hours);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub((hours * 3600.0) as u64)
    };

    // Post a message
    if let Some(msg) = text {
        // PII check
        let issues = blackboard::pii_check(&msg);
        if !issues.is_empty() {
            eprintln!("⚠️  PII/secret issues detected:");
            for issue in &issues {
                eprintln!("   • {issue}");
            }
            eprintln!("Scrubbing and posting...");
        }
        let clean = blackboard::pii_scrub(&msg);

        let body = serde_json::json!({ "text": clean });
        let resp = client
            .post(format!("{base}/api/blackboard/post"))
            .json(&body)
            .send()
            .await
            .context("Cannot reach mesh-llm — is it running?")?;
        if resp.status().is_success() {
            let item: blackboard::BlackboardItem = resp.json().await?;
            eprintln!("📝 Posted (id: {:x})", item.id);
        } else {
            let err = resp.text().await.unwrap_or_default();
            eprintln!("Error: {err}");
        }
        return Ok(());
    }

    // Search
    if let Some(q) = search {
        let resp = client
            .get(format!("{base}/api/blackboard/search"))
            .query(&[
                ("q", q.as_str()),
                ("limit", &limit.to_string()),
                ("since", &since_secs.to_string()),
            ])
            .send()
            .await
            .context("Cannot reach mesh-llm — is it running?")?;
        let items: Vec<blackboard::BlackboardItem> = resp.json().await?;
        if items.is_empty() {
            eprintln!("No results.");
        } else {
            print_blackboard_items(&items);
        }
        return Ok(());
    }

    // Feed (optionally filtered by peer)
    let mut params = vec![
        ("limit", limit.to_string()),
        ("since", since_secs.to_string()),
    ];
    if let Some(ref f) = from {
        params.push(("from", f.clone()));
    }
    let resp = client
        .get(format!("{base}/api/blackboard/feed"))
        .query(&params)
        .send()
        .await
        .context("Cannot reach mesh-llm — is it running?")?;
    let items: Vec<blackboard::BlackboardItem> = resp.json().await?;
    if items.is_empty() {
        eprintln!("Blackboard is empty.");
    } else {
        print_blackboard_items(&items);
    }
    Ok(())
}

fn print_blackboard_items(items: &[blackboard::BlackboardItem]) {
    for item in items {
        let time = chrono_format(item.timestamp);
        println!("{:x} │ {} │ {}", item.id, time, item.from);
        // Indent the text
        for line in item.text.lines() {
            println!("  {line}");
        }
        println!();
    }
}

pub(crate) async fn run_plugin_command(command: &PluginCommand, cli: &Cli) -> Result<()> {
    match command {
        PluginCommand::Install { name } if name == plugin::BLACKBOARD_PLUGIN_ID => {
            eprintln!("Blackboard is auto-registered by mesh-llm. Nothing to install.");
            eprintln!("Disable it with [[plugin]] name = \"blackboard\" enabled = false in the config if needed.");
        }
        PluginCommand::Install { name } => {
            let config = plugin::config_path(cli.config.as_deref())?;
            anyhow::bail!(
                "Plugins are configured as executables in {}. No install step exists for '{}'.",
                config.display(),
                name
            );
        }
        PluginCommand::List => {
            let resolved = super::load_resolved_plugins(cli)?;
            for spec in resolved.externals {
                println!(
                    "{}\tkind=external\tcommand={}\targs={}",
                    spec.name,
                    spec.command,
                    spec.args.join(" ")
                );
            }
        }
    }
    Ok(())
}

fn chrono_format(ts: u64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let ago = now.saturating_sub(ts);
    if ago < 60 {
        format!("{ago}s ago")
    } else if ago < 3600 {
        format!("{}m ago", ago / 60)
    } else if ago < 86400 {
        format!("{}h ago", ago / 3600)
    } else {
        format!("{}d ago", ago / 86400)
    }
}

pub(crate) fn install_skill() -> Result<()> {
    let skill_content = include_str!("../../skills/blackboard/SKILL.md");
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    let skill_dir = home.join(".agents").join("skills").join("blackboard");
    std::fs::create_dir_all(&skill_dir)?;
    let skill_path = skill_dir.join("SKILL.md");
    std::fs::write(&skill_path, skill_content)?;
    eprintln!("✅ Installed blackboard skill to {}", skill_path.display());
    eprintln!("   Works with pi, Goose, and other agents that read ~/.agents/skills/");
    eprintln!(
        "   Make sure mesh-llm is running and the blackboard plugin is not disabled in config."
    );
    Ok(())
}
