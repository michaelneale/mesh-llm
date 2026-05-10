use crate::cli::output::{emit_event, OutputEvent};
use crate::cli::Cli;
use crate::mesh;
use crate::network::{nostr, router};
use anyhow::{Context, Result};
use std::cmp::Reverse;

/// Health probe: try QUIC connect to the mesh's bootstrap node.
/// Returns Ok if reachable within 10s, Err if not.
/// Re-discover meshes via Nostr when all peers are lost.
/// Only runs for --auto nodes that originally discovered via Nostr.
/// Checks every 30s; if 0 peers for 90s straight, re-discovers and joins.
pub(super) async fn nostr_rediscovery(
    node: mesh::Node,
    nostr_relays: Vec<String>,
    _relay_urls: Vec<String>,
    mesh_name: Option<String>,
) {
    const CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
    const GRACE_PERIOD: std::time::Duration = std::time::Duration::from_secs(90);

    tokio::time::sleep(std::time::Duration::from_secs(30)).await;

    let mut alone_since: Option<std::time::Instant> = None;

    loop {
        tokio::time::sleep(CHECK_INTERVAL).await;

        let peers = node.peers().await;
        if !peers.is_empty() {
            if alone_since.is_some() {
                tracing::debug!("Nostr rediscovery: peers recovered, resetting timer");
                alone_since = None;
            }
            continue;
        }

        let now = std::time::Instant::now();
        let start = *alone_since.get_or_insert(now);

        if now.duration_since(start) < GRACE_PERIOD {
            tracing::debug!(
                "Nostr rediscovery: 0 peers for {}s (grace: {}s)",
                now.duration_since(start).as_secs(),
                GRACE_PERIOD.as_secs()
            );
            continue;
        }

        let _ = emit_event(OutputEvent::DiscoveryStarting {
            source: "Nostr re-discovery".to_string(),
        });

        let filter = nostr::MeshFilter::default();
        let meshes = match nostr::discover(&nostr_relays, &filter, None).await {
            Ok(m) => m,
            Err(e) => {
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: "Nostr re-discovery failed".to_string(),
                    detail: Some(e.to_string()),
                });
                alone_since = Some(std::time::Instant::now());
                continue;
            }
        };

        let filtered: Vec<_> = if let Some(ref name) = mesh_name {
            meshes
                .iter()
                .filter(|m| {
                    m.listing
                        .name
                        .as_ref()
                        .map(|n| n.eq_ignore_ascii_case(name))
                        .unwrap_or(false)
                })
                .collect()
        } else {
            meshes.iter().collect()
        };

        if filtered.is_empty() {
            let name_hint = mesh_name.as_deref().unwrap_or("any");
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: format!("No meshes found on Nostr matching \"{name_hint}\" — will retry"),
                detail: None,
            });
            alone_since = Some(std::time::Instant::now());
            continue;
        }

        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last_mesh_id = mesh::load_last_mesh_id();

        let mut candidates: Vec<_> = filtered
            .iter()
            .map(|m| (*m, nostr::score_mesh(m, now_ts, last_mesh_id.as_deref())))
            .collect();
        candidates.sort_by_key(|b| Reverse(b.1));

        let our_mesh_id = node.mesh_id().await;

        let mut rejoined = false;
        for (mesh, _score) in &candidates {
            if let (Some(ref ours), Some(ref theirs)) = (&our_mesh_id, &mesh.listing.mesh_id) {
                if ours == theirs {
                    continue;
                }
            }
            let mesh_label = mesh
                .listing
                .name
                .as_deref()
                .unwrap_or("unnamed")
                .to_string();
            let _ = emit_event(OutputEvent::MeshFound {
                mesh: mesh_label.clone(),
                peers: mesh.listing.node_count,
                region: None,
            });
            let token = &mesh.listing.invite_token;
            match node.join(token).await {
                Ok(()) => {
                    let _ = emit_event(OutputEvent::DiscoveryJoined { mesh: mesh_label });
                    rejoined = true;
                }
                Err(e) => {
                    let _ = emit_event(OutputEvent::DiscoveryFailed {
                        message: format!(
                            "Failed to re-join mesh {}",
                            mesh.listing.name.as_deref().unwrap_or("unnamed")
                        ),
                        detail: Some(e.to_string()),
                    });
                }
            }
            if rejoined {
                break;
            }
        }

        if rejoined {
            alone_since = None;
        } else {
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: "Could not re-join any mesh — will retry".to_string(),
                detail: None,
            });
            alone_since = Some(std::time::Instant::now());
        }
    }
}

/// Helper for StartNew path — configure CLI to start a new mesh.
pub(super) fn start_new_mesh(
    cli: &mut Cli,
    models: &[String],
    my_vram_gb: f64,
    has_startup_models: bool,
) {
    let primary = models.first().cloned().unwrap_or_default();
    if !has_startup_models && cli.model.is_empty() {
        cli.model.push(primary.clone().into());
    }
    let detail = if has_startup_models {
        "using configured startup models".to_string()
    } else {
        format!("serving: {primary}")
    };
    let discovery = if cli.publish {
        "publishing for discovery"
    } else {
        "mesh is private — add --publish to advertise it for discovery"
    };
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Starting a new mesh — {detail} — capacity: {:.0}GB — {discovery}",
            my_vram_gb
        ),
        context: None,
    });
}

pub(crate) fn nostr_relays(cli_relays: &[String]) -> Vec<String> {
    if cli_relays.is_empty() {
        nostr::DEFAULT_RELAYS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cli_relays.to_vec()
    }
}

/// Ensure mesh-llm is running on `port`, then return (available_models, chosen_model, spawned_child).
///
/// Launcher behavior: if nothing is listening yet, auto-start `mesh-llm client --auto`
/// (client node — tunnels to mesh peers without publishing to Nostr).
/// Returns the child process handle if we spawned one, so callers can clean up on exit.
pub(crate) async fn check_mesh(
    client: &reqwest::Client,
    port: u16,
    model: &Option<String>,
) -> Result<(Vec<String>, String, Option<std::process::Child>)> {
    let url = format!("http://127.0.0.1:{port}/v1/models");

    let mut child: Option<std::process::Child> = None;
    if client.get(&url).send().await.is_err() {
        let _ = emit_event(OutputEvent::Info {
            message: format!("No mesh-llm on port {port} — starting background auto-join node"),
            context: None,
        });
        let exe = std::env::current_exe().unwrap_or_else(|_| "mesh-llm".into());
        child = Some(
            std::process::Command::new(&exe)
                .args(["client", "--auto", "--port", &port.to_string()])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to start mesh-llm node")?,
        );
    }

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
            let _ = emit_event(OutputEvent::Info {
                message: format!("Waiting for mesh/models... ({:.0}s)", (i + 1) as f64 * 3.0),
                context: Some(format!("port={port}")),
            });
        }
    }

    if models.is_empty() {
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
        let available: Vec<(&str, f64, crate::models::ModelCapabilities)> = models
            .iter()
            .map(|n| {
                let caps = crate::models::installed_model_capabilities(n);
                (n.as_str(), 0.0, caps)
            })
            .collect();
        let agentic = router::Classification {
            category: router::Category::Code,
            complexity: router::Complexity::Deep,
            needs_tools: true,
            has_media_inputs: false,
        };
        router::pick_model_classified(&agentic, &available)
            .map(|s| s.to_string())
            .unwrap_or_else(|| models[0].clone())
    };
    let _ = emit_event(OutputEvent::Info {
        message: format!("Models: {}", models.join(", ")),
        context: Some(format!("port={port}")),
    });
    let _ = emit_event(OutputEvent::Info {
        message: format!("Using: {chosen}"),
        context: Some(format!("port={port}")),
    });
    Ok((models, chosen, child))
}
