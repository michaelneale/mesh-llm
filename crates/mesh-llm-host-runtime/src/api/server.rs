use super::{
    assets::{respond_console_asset, respond_console_index},
    http::{http_body_text, respond_error},
    routes::dispatch_request,
    MeshApi,
};
use crate::{inference::election, network::proxy};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::watch,
};

// ── Server ──

pub(crate) async fn start_with_listener(
    port: u16,
    state: MeshApi,
    mut target_rx: watch::Receiver<election::InferenceTarget>,
    listen_all: bool,
    headless: bool,
    existing_listener: Option<TcpListener>,
) {
    state.set_headless(headless).await;
    // Watch election target changes
    let state2 = state.clone();
    tokio::spawn(async move {
        loop {
            if target_rx.changed().await.is_err() {
                break;
            }
            let target = target_rx.borrow().clone();
            match target {
                election::InferenceTarget::Local(port) => {
                    state2.set_llama_port(Some(port)).await;
                }
                election::InferenceTarget::Remote(_) => {
                    let mut inner = state2.inner.lock().await;
                    inner.llama_ready = true;
                    inner.llama_port = None;
                    inner
                        .runtime_data_producer
                        .publish_runtime_status(|runtime_status| {
                            let mut changed = false;
                            if !runtime_status.llama_ready {
                                runtime_status.llama_ready = true;
                                changed = true;
                            }
                            if runtime_status.llama_port.is_some() {
                                runtime_status.llama_port = None;
                                changed = true;
                            }
                            changed
                        });
                }
                election::InferenceTarget::None => {
                    state2.set_llama_port(None).await;
                }
            }
        }
    });

    // Push status when peers join/leave.
    let mut peer_rx = {
        let inner = state.inner.lock().await;
        inner.node.peer_change_rx.clone()
    };
    let state3 = state.clone();
    tokio::spawn(async move {
        loop {
            if peer_rx.changed().await.is_err() {
                break;
            }
            state3.push_status().await;
        }
    });

    // Push status when in-flight request count changes.
    let mut inflight_rx = {
        let inner = state.inner.lock().await;
        inner.node.inflight_change_rx()
    };
    let state4 = state.clone();
    tokio::spawn(async move {
        loop {
            if inflight_rx.changed().await.is_err() {
                break;
            }
            state4.push_status().await;
        }
    });

    // One-shot check for newer public release (for UI footer indicator).
    let state5 = state.clone();
    tokio::spawn(async move {
        let Some(latest) = crate::system::autoupdate::latest_release_version().await else {
            return;
        };
        if !crate::system::autoupdate::version_newer(&latest, crate::VERSION) {
            return;
        }
        {
            let mut inner = state5.inner.lock().await;
            inner.latest_version = Some(latest);
        }
        state5.push_status().await;
    });

    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match existing_listener {
        Some(listener) => listener,
        None => match TcpListener::bind(format!("{addr}:{port}")).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Management API: failed to bind :{port}: {e}");
                return;
            }
        },
    };
    let management_url = listener
        .local_addr()
        .map(|addr| format!("http://{addr}"))
        .unwrap_or_else(|err| {
            tracing::warn!("Management API: failed to read listener address: {err}");
            format!("http://localhost:{port}")
        });
    tracing::info!("Management API on {management_url}");

    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = Box::pin(handle_request(stream, &state)).await {
                tracing::debug!("API connection error: {e}");
            }
        });
    }
}

// ── Request dispatch ──

pub(crate) fn is_ui_only_route(path: &str) -> bool {
    matches!(
        path,
        "/" | "/dashboard" | "/dashboard/" | "/chat" | "/chat/"
    ) || path.starts_with("/chat/")
        || path.starts_with("/assets/")
        || matches!(path.rsplit('.').next(), Some("png" | "ico" | "webmanifest"))
        || (path.ends_with(".json") && !path.starts_with("/api/"))
}

pub(crate) async fn handle_request(mut stream: TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let request = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        proxy::read_http_request(&mut stream),
    )
    .await
    {
        Ok(Ok(request)) => request,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Ok(()), // read timeout — health check probe, just close
    };
    let req = String::from_utf8_lossy(&request.raw);
    let method = request.method.as_str();
    let path = request.path.as_str();
    let path_only = path.split('?').next().unwrap_or(path);
    let body = http_body_text(&request.raw);

    if method == "GET" && state.is_headless().await && is_ui_only_route(path_only) {
        respond_error(&mut stream, 404, "Not found").await?;
        return Ok(());
    }

    match (method, path_only) {
        // ── Dashboard UI ──
        ("GET", "/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", "/dashboard") | ("GET", "/chat") | ("GET", "/dashboard/") | ("GET", "/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", p) if p.starts_with("/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        // ── Frontend static assets (bundled UI dist) ──
        ("GET", p)
            if p.starts_with("/assets/")
                || matches!(p.rsplit('.').next(), Some("png" | "ico" | "webmanifest"))
                || (p.ends_with(".json") && !p.starts_with("/api/")) =>
        {
            if !respond_console_asset(&mut stream, p).await? {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        _ => {
            if !dispatch_request(
                &mut stream,
                state,
                method,
                path,
                path_only,
                body,
                req.as_ref(),
                &request.raw,
            )
            .await?
            {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }
    }
    Ok(())
}
