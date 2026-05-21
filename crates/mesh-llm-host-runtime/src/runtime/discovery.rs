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
        run_rediscovery_tick(
            &node,
            &nostr_relays,
            mesh_name.as_deref(),
            GRACE_PERIOD,
            &mut alone_since,
        )
        .await;
    }
}

async fn run_rediscovery_tick(
    node: &mesh::Node,
    nostr_relays: &[String],
    mesh_name: Option<&str>,
    grace_period: std::time::Duration,
    alone_since: &mut Option<std::time::Instant>,
) {
    if reset_rediscovery_timer_if_peers_recovered(node, alone_since).await {
        return;
    }

    if rediscovery_grace_period_active(alone_since, grace_period) {
        return;
    }

    let _ = emit_event(OutputEvent::DiscoveryStarting {
        source: "Nostr re-discovery".to_string(),
    });

    let Some(meshes) = discover_rediscovery_meshes(nostr_relays, alone_since).await else {
        return;
    };

    let filtered = filter_rediscovery_meshes(&meshes, mesh_name);
    if filtered.is_empty() {
        report_no_rediscovery_meshes(mesh_name, alone_since);
        return;
    }

    let candidates = rank_rediscovery_candidates(&filtered);
    let our_mesh_id = node.mesh_id().await;
    if try_rejoin_rediscovery_candidates(node, &candidates, our_mesh_id.as_deref()).await {
        *alone_since = None;
    } else {
        report_rediscovery_retry(alone_since);
    }
}

async fn reset_rediscovery_timer_if_peers_recovered(
    node: &mesh::Node,
    alone_since: &mut Option<std::time::Instant>,
) -> bool {
    if node.peers().await.is_empty() {
        return false;
    }
    if alone_since.is_some() {
        tracing::debug!("Nostr rediscovery: peers recovered, resetting timer");
        *alone_since = None;
    }
    true
}

fn rediscovery_grace_period_active(
    alone_since: &mut Option<std::time::Instant>,
    grace_period: std::time::Duration,
) -> bool {
    let now = std::time::Instant::now();
    let start = *alone_since.get_or_insert(now);
    let elapsed = now.duration_since(start);
    if elapsed >= grace_period {
        return false;
    }
    tracing::debug!(
        "Nostr rediscovery: 0 peers for {}s (grace: {}s)",
        elapsed.as_secs(),
        grace_period.as_secs()
    );
    true
}

async fn discover_rediscovery_meshes(
    nostr_relays: &[String],
    alone_since: &mut Option<std::time::Instant>,
) -> Option<Vec<nostr::DiscoveredMesh>> {
    let filter = nostr::MeshFilter::default();
    match nostr::discover(nostr_relays, &filter, None).await {
        Ok(meshes) => Some(meshes),
        Err(err) => {
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: "Nostr re-discovery failed".to_string(),
                detail: Some(err.to_string()),
            });
            *alone_since = Some(std::time::Instant::now());
            None
        }
    }
}

fn filter_rediscovery_meshes<'a>(
    meshes: &'a [nostr::DiscoveredMesh],
    mesh_name: Option<&str>,
) -> Vec<&'a nostr::DiscoveredMesh> {
    match mesh_name {
        Some(name) => meshes
            .iter()
            .filter(|mesh| rediscovery_mesh_name_matches(mesh, name))
            .collect(),
        None => meshes.iter().collect(),
    }
}

fn rediscovery_mesh_name_matches(mesh: &nostr::DiscoveredMesh, name: &str) -> bool {
    mesh.listing
        .name
        .as_ref()
        .map(|candidate| candidate.eq_ignore_ascii_case(name))
        .unwrap_or(false)
}

fn report_no_rediscovery_meshes(
    mesh_name: Option<&str>,
    alone_since: &mut Option<std::time::Instant>,
) {
    let name_hint = mesh_name.unwrap_or("any");
    let _ = emit_event(OutputEvent::DiscoveryFailed {
        message: format!("No meshes found on Nostr matching \"{name_hint}\" — will retry"),
        detail: None,
    });
    *alone_since = Some(std::time::Instant::now());
}

fn rank_rediscovery_candidates<'a>(
    meshes: &[&'a nostr::DiscoveredMesh],
) -> Vec<(&'a nostr::DiscoveredMesh, i64)> {
    let now_ts = current_unix_secs();
    let last_mesh_id = mesh::load_last_mesh_id();
    let mut candidates: Vec<_> = meshes
        .iter()
        .map(|mesh| {
            (
                *mesh,
                nostr::score_mesh(mesh, now_ts, last_mesh_id.as_deref()),
            )
        })
        .collect();
    candidates.sort_by_key(|candidate| Reverse(candidate.1));
    candidates
}

async fn try_rejoin_rediscovery_candidates(
    node: &mesh::Node,
    candidates: &[(&nostr::DiscoveredMesh, i64)],
    our_mesh_id: Option<&str>,
) -> bool {
    for (mesh, _score) in candidates {
        if rediscovery_candidate_is_current_mesh(mesh, our_mesh_id) {
            continue;
        }
        if try_rejoin_rediscovery_mesh(node, mesh).await {
            return true;
        }
    }
    false
}

fn rediscovery_candidate_is_current_mesh(
    mesh: &nostr::DiscoveredMesh,
    our_mesh_id: Option<&str>,
) -> bool {
    match (our_mesh_id, mesh.listing.mesh_id.as_deref()) {
        (Some(ours), Some(theirs)) => ours == theirs,
        _ => false,
    }
}

async fn try_rejoin_rediscovery_mesh(node: &mesh::Node, mesh: &nostr::DiscoveredMesh) -> bool {
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
    match node.join(&mesh.listing.invite_token).await {
        Ok(()) => {
            let _ = emit_event(OutputEvent::DiscoveryJoined { mesh: mesh_label });
            true
        }
        Err(err) => {
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: format!(
                    "Failed to re-join mesh {}",
                    mesh.listing.name.as_deref().unwrap_or("unnamed")
                ),
                detail: Some(err.to_string()),
            });
            false
        }
    }
}

fn report_rediscovery_retry(alone_since: &mut Option<std::time::Instant>) {
    let _ = emit_event(OutputEvent::DiscoveryFailed {
        message: "Could not re-join any mesh — will retry".to_string(),
        detail: None,
    });
    *alone_since = Some(std::time::Instant::now());
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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
        // Pre-startup path: no live routing metrics yet, so candidates
        // are scored as cold (uniform weight).
        let available: Vec<router::RoutingCandidate<'_>> = models
            .iter()
            .map(|n| {
                let caps = crate::models::installed_model_capabilities(n);
                router::RoutingCandidate::unscored(n.as_str(), caps)
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
