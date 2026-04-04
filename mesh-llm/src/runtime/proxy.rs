use crate::api;
use crate::inference::{election, pipeline};
use crate::mesh;
use crate::network::{affinity, proxy, router};

/// Model-aware API proxy. Parses the "model" field from POST request bodies
/// and routes to the correct host. Falls back to the first available target
/// if model is not specified or not found.
pub(super) async fn api_proxy(
    node: mesh::Node,
    port: u16,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    let listener = match existing_listener {
        Some(l) => l,
        None => {
            let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
            match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!("Failed to bind API proxy to port {port}: {e}");
                    return;
                }
            }
        }
    };

    loop {
        let (tcp_stream, addr) = match listener.accept().await {
            Ok(r) => r,
            Err(_) => break,
        };
        let _ = tcp_stream.set_nodelay(true);

        let targets = target_rx.borrow().clone();
        let node = node.clone();
        let affinity = affinity.clone();
        let control_tx = control_tx.clone();
        tokio::spawn(async move {
            let mut tcp_stream = tcp_stream;
            let plugin_manager = node.plugin_manager().await;
            match proxy::read_http_request_with_plugin_manager(
                &mut tcp_stream,
                plugin_manager.as_ref(),
            )
            .await
            {
                Ok(request) => {
                    let body_json = request.body_json.as_ref();
                    if proxy::is_models_list_request(&request.method, &request.path) {
                        let models: Vec<String> = targets.targets.keys().cloned().collect();
                        let _ = proxy::send_models_list(tcp_stream, &models).await;
                        return;
                    }

                    let path = request.path.split('?').next().unwrap_or(&request.path);
                    if request.method == "POST" && path == "/mesh/load" {
                        if let Some(ref spec) = request.model_name {
                            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                            let _ = control_tx.send(api::RuntimeControlRequest::Load {
                                spec: spec.clone(),
                                resp: resp_tx,
                            });
                            match resp_rx.await {
                                Ok(Ok(loaded)) => {
                                    let _ = proxy::send_json_ok(
                                        tcp_stream,
                                        &serde_json::json!({"loaded": loaded}),
                                    )
                                    .await;
                                }
                                Ok(Err(e)) => {
                                    let msg = e.to_string();
                                    let code = api::classify_runtime_error(&msg);
                                    let _ = proxy::send_error(tcp_stream, code, &msg).await;
                                }
                                Err(_) => {
                                    let _ = proxy::send_503(tcp_stream).await;
                                }
                            }
                        } else {
                            let _ = proxy::send_400(tcp_stream, "missing 'model' field").await;
                        }
                        return;
                    }

                    if proxy::is_drop_request(&request.method, &request.path) {
                        if let Some(ref name) = request.model_name {
                            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                            let _ = control_tx.send(api::RuntimeControlRequest::Unload {
                                model: name.clone(),
                                resp: resp_tx,
                            });
                            match resp_rx.await {
                                Ok(Ok(())) => {
                                    let _ = proxy::send_json_ok(
                                        tcp_stream,
                                        &serde_json::json!({"dropped": name}),
                                    )
                                    .await;
                                }
                                Ok(Err(e)) => {
                                    let msg = e.to_string();
                                    let code = api::classify_runtime_error(&msg);
                                    let _ = proxy::send_error(tcp_stream, code, &msg).await;
                                }
                                Err(_) => {
                                    let _ = proxy::send_503(tcp_stream).await;
                                }
                            }
                        } else {
                            let _ = proxy::send_400(tcp_stream, "missing 'model' field").await;
                        }
                        return;
                    }

                    let (effective_model, classification) = if request.model_name.is_none()
                        || request.model_name.as_deref() == Some("auto")
                    {
                        if let Some(body_json) = body_json {
                            let cl = router::classify(body_json);
                            let available: Vec<(&str, f64)> = targets
                                .targets
                                .keys()
                                .map(|name| (name.as_str(), 0.0))
                                .collect();
                            let picked = router::pick_model_classified(&cl, &available);
                            if let Some(name) = picked {
                                tracing::info!(
                                    "router: {:?}/{:?} tools={} → {name}",
                                    cl.category,
                                    cl.complexity,
                                    cl.needs_tools
                                );
                                (Some(name.to_string()), Some(cl))
                            } else {
                                (None, Some(cl))
                            }
                        } else {
                            (None, None)
                        }
                    } else {
                        (request.model_name.clone(), None)
                    };

                    if let Some(ref name) = effective_model {
                        node.record_request(name);
                    }

                    let use_pipeline = classification
                        .as_ref()
                        .map(|cl| pipeline::should_pipeline(cl))
                        .unwrap_or(false)
                        && request.response_adapter == proxy::ResponseAdapter::None;

                    if use_pipeline {
                        if let Some(ref strong_name) = effective_model {
                            let planner = targets
                                .targets
                                .iter()
                                .find(|(name, target_vec)| {
                                    *name != strong_name
                                        && target_vec.iter().any(|t| {
                                            matches!(t, election::InferenceTarget::Local(_))
                                        })
                                })
                                .and_then(|(name, target_vec)| {
                                    target_vec.iter().find_map(|t| match t {
                                        election::InferenceTarget::Local(p) => {
                                            Some((name.clone(), *p))
                                        }
                                        _ => None,
                                    })
                                });

                            let strong_local_port =
                                targets.targets.get(strong_name.as_str()).and_then(|tv| {
                                    tv.iter().find_map(|t| match t {
                                        election::InferenceTarget::Local(p) => Some(*p),
                                        _ => None,
                                    })
                                });

                            if let (Some((planner_name, planner_port)), Some(strong_port)) =
                                (planner, strong_local_port)
                            {
                                if let Some(body_json) = request.body_json.clone() {
                                    tracing::info!(
                                        "pipeline: {planner_name} (plan) → {strong_name} (execute)"
                                    );
                                    if matches!(
                                        proxy::pipeline_proxy_local(
                                            &mut tcp_stream,
                                            &request.path,
                                            body_json,
                                            planner_port,
                                            &planner_name,
                                            strong_port,
                                            &node,
                                        )
                                        .await,
                                        proxy::PipelineProxyResult::Handled
                                    ) {
                                        proxy::release_request_objects(
                                            &node,
                                            &request.request_object_request_ids,
                                        )
                                        .await;
                                        return;
                                    }
                                }
                                tracing::warn!(
                                    "pipeline: falling back to direct proxy for {strong_name}"
                                );
                            }
                        }
                    }

                    let target = if targets.moe.is_some() {
                        let session_hint = request
                            .session_hint
                            .clone()
                            .unwrap_or_else(|| format!("{addr}"));
                        targets
                            .get_moe_target(&session_hint)
                            .unwrap_or(first_available_target(&targets))
                    } else if let Some(ref name) = effective_model {
                        if targets.candidates(name).is_empty() {
                            tracing::debug!("Model '{}' not found, trying first available", name);
                            first_available_target(&targets)
                        } else {
                            let routed = proxy::route_model_request(
                                node.clone(),
                                tcp_stream,
                                &targets,
                                name,
                                body_json,
                                &request.raw,
                                request.response_adapter,
                                &affinity,
                            )
                            .await;
                            proxy::release_request_objects(
                                &node,
                                &request.request_object_request_ids,
                            )
                            .await;
                            debug_assert!(routed);
                            return;
                        }
                    } else {
                        first_available_target(&targets)
                    };

                    let _ = proxy::route_to_target(
                        node.clone(),
                        tcp_stream,
                        target,
                        &request.raw,
                        request.response_adapter,
                    )
                    .await;
                    proxy::release_request_objects(&node, &request.request_object_request_ids)
                        .await;
                }
                Err(err) => {
                    let _ = proxy::send_400(tcp_stream, &err.to_string()).await;
                    return;
                }
            };
        });
    }
}

/// Bootstrap proxy: runs during GPU startup, tunnels all requests to mesh hosts.
/// Returns the TcpListener when signaled to stop (so api_proxy can take it over).
pub(super) async fn bootstrap_proxy(
    node: mesh::Node,
    port: u16,
    mut stop_rx: tokio::sync::mpsc::Receiver<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Bootstrap proxy: failed to bind to port {port}: {e}");
            return;
        }
    };
    eprintln!("⚡ API ready (bootstrap): http://localhost:{port}");
    eprintln!("  Requests tunneled to mesh while GPU loads...");

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (tcp_stream, _addr) = match accept {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let _ = tcp_stream.set_nodelay(true);
                let node = node.clone();
                let affinity = affinity.clone();
                tokio::spawn(proxy::handle_mesh_request(node, tcp_stream, true, affinity));
            }
            resp_tx = stop_rx.recv() => {
                if let Some(tx) = resp_tx {
                    eprintln!("⚡ Bootstrap proxy handing off to full API proxy");
                    let _ = tx.send(listener);
                }
                return;
            }
        }
    }
}

fn first_available_target(targets: &election::ModelTargets) -> election::InferenceTarget {
    for hosts in targets.targets.values() {
        for target in hosts {
            if !matches!(target, election::InferenceTarget::None) {
                return target.clone();
            }
        }
    }
    election::InferenceTarget::None
}

#[cfg(test)]
mod tests;
