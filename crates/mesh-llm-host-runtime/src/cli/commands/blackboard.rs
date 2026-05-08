use anyhow::{Context, Result};

use crate::blackboard;

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

    let status_resp = client.get(format!("{base}/api/status")).send().await;
    if status_resp.is_err() {
        eprintln!("No mesh-llm node running on port {port}.");
        eprintln!();
        eprintln!("Blackboard requires a running mesh node:");
        eprintln!("  Private mesh:  mesh-llm client  (share the join token printed out)");
        eprintln!("  Join a mesh:   mesh-llm client --join <token>");
        eprintln!("  Public mesh:   mesh-llm client --auto");
        eprintln!();
        eprintln!("See https://github.com/Mesh-LLM/mesh-llm for setup guide.");
        std::process::exit(1);
    }

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

    let default_hours = 24.0;
    let since_secs = {
        let hours = since_hours.unwrap_or(default_hours);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub((hours * 3600.0) as u64)
    };

    if let Some(msg) = text {
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
        for line in item.text.lines() {
            println!("  {line}");
        }
        println!();
    }
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
    let skill_content = include_str!("../../../skills/blackboard/SKILL.md");
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
