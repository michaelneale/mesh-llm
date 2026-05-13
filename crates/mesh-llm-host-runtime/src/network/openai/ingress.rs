use crate::api;
use crate::cli::output::{emit_event, OutputEvent};
use crate::inference::{election, pipeline};
use crate::mesh;
use crate::network::affinity;
use crate::network::openai::transport as proxy;
use crate::network::router;

/// Model-aware API proxy. Parses the "model" field from POST request bodies
/// and routes to the correct host. Falls back to the first available target
/// if model is not specified or not found.
pub(crate) async fn api_proxy(
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
        let (tcp_stream, _addr) = match listener.accept().await {
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
                Ok(mut request) => {
                    if proxy::is_models_list_request(&request.method, &request.path) {
                        let mut models = callable_models(&targets);
                        models.extend(node.models_being_served().await);
                        if let Some(plugin_manager) = plugin_manager.as_ref() {
                            if let Ok(mut external_models) = plugin_manager.inference_models().await
                            {
                                models.append(&mut external_models);
                            }
                        }
                        models.sort();
                        models.dedup();
                        // Offer "moa" virtual model when ≥2 distinct models
                        // are available for mixture-of-agents fan-out.
                        if models.len() >= 2 && !models.contains(&"moa".to_string()) {
                            models.push("moa".to_string());
                        }
                        let descriptors = node.served_model_descriptors().await;
                        let _ = proxy::send_models_list_with_descriptors(
                            tcp_stream,
                            &models,
                            &descriptors,
                        )
                        .await;
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
                                        &serde_json::json!({
                                            "loaded": loaded.model,
                                            "instance_id": loaded.instance_id,
                                        }),
                                    )
                                    .await;
                                }
                                Ok(Err(e)) => {
                                    let msg = e.to_string();
                                    let code = api::classify_runtime_error(&msg);
                                    let _ = proxy::send_error(tcp_stream, code, &msg).await;
                                }
                                Err(_) => {
                                    let _ =
                                        proxy::send_503(tcp_stream, "runtime load channel closed")
                                            .await;
                                }
                            }
                        } else {
                            let _ = proxy::send_400(tcp_stream, "missing 'model' field").await;
                        }
                        return;
                    }

                    let mut callable = callable_models(&targets);
                    for name in node.models_being_served().await {
                        if !callable.iter().any(|existing| existing == &name) {
                            callable.push(name);
                        }
                    }
                    callable.sort();
                    let descriptors = node.served_model_descriptors().await;
                    proxy::rewrite_public_model_alias(&mut request, &callable, &descriptors);

                    if proxy::is_drop_request(&request.method, &request.path) {
                        if let Some(ref name) = request.model_name {
                            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                            let _ = control_tx.send(api::RuntimeControlRequest::Unload {
                                target: name.clone(),
                                resp: resp_tx,
                            });
                            match resp_rx.await {
                                Ok(Ok(dropped)) => {
                                    let _ = proxy::send_json_ok(
                                        tcp_stream,
                                        &serde_json::json!({
                                            "dropped": dropped.model,
                                            "instance_id": dropped.instance_id,
                                        }),
                                    )
                                    .await;
                                }
                                Ok(Err(e)) => {
                                    let msg = e.to_string();
                                    let code = api::classify_runtime_error(&msg);
                                    let _ = proxy::send_error(tcp_stream, code, &msg).await;
                                }
                                Err(_) => {
                                    let _ = proxy::send_503(
                                        tcp_stream,
                                        "runtime unload channel closed",
                                    )
                                    .await;
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
                        request.ensure_body_json();
                        if let Some(body_json) = request.body_json.as_ref() {
                            let cl = router::classify(body_json);
                            let media = router::media_requirements(body_json);
                            let mut available_models = callable.clone();
                            if let Some(plugin_manager) = plugin_manager.as_ref() {
                                if let Ok(external_models) = plugin_manager.inference_models().await
                                {
                                    for name in external_models {
                                        if !available_models
                                            .iter()
                                            .any(|existing| existing == &name)
                                        {
                                            available_models.push(name);
                                        }
                                    }
                                }
                            }
                            let available: Vec<(&str, f64, crate::models::ModelCapabilities)> =
                                available_models
                                    .iter()
                                    .map(|name| {
                                        let caps =
                                            proxy::capabilities_for_model(name, &descriptors);
                                        (name.as_str(), 0.0, caps)
                                    })
                                    .collect();
                            let Some(available) =
                                router::filter_media_compatible_candidates(&available, &media)
                            else {
                                let _ = proxy::send_error(
                                    tcp_stream,
                                    422,
                                    "no served model can satisfy the requested media inputs",
                                )
                                .await;
                                proxy::release_request_objects(
                                    &node,
                                    &request.request_object_request_ids,
                                )
                                .await;
                                return;
                            };
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
                    // Enable mesh hooks for auto-routed requests. When the
                    // smart router picks the model, hooks allow the local
                    // model to consult peers during inference (e.g. caption
                    // images via a vision peer, get a second opinion on
                    // uncertain answers).
                    if request.model_name.is_none() || request.model_name.as_deref() == Some("auto")
                    {
                        proxy::inject_mesh_hooks_flag(&mut request.raw, true);
                        if let Some(model) = effective_model.as_deref() {
                            proxy::rewrite_model_field(&mut request, model);
                        }
                    }

                    let required_tokens = proxy::request_budget_tokens_from_parts(
                        request.body_len_bytes,
                        request.completion_tokens,
                    );

                    if let Some(ref name) = effective_model {
                        node.record_request(name);
                    }

                    // ── MoA intercept ─────────────────────────────────
                    // When model == "moa", fan out to all available models
                    // via the MoA gateway and return the arbitrated result.
                    // Uses the `callable` list which includes both local
                    // targets and mesh-served models.
                    if effective_model.as_deref() == Some("moa") {
                        request.ensure_body_json();
                        if let Some(body_json) = request.body_json.clone() {
                            let moa_models: Vec<String> = callable
                                .iter()
                                .filter(|m| m.as_str() != "moa")
                                .cloned()
                                .collect();
                            if moa_models.len() >= 2 {
                                // Strip stream flag — MoA workers are non-streaming;
                                // we synthesize SSE frames from the final response
                                // when the client asked for streaming.
                                let was_streaming = body_json
                                    .get("stream")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                let mut moa_body = body_json;
                                moa_body.as_object_mut().map(|o| o.remove("stream"));

                                let moa_result =
                                    handle_moa_request(port, &moa_models, moa_body).await;

                                if was_streaming {
                                    let _ = send_moa_as_sse(tcp_stream, &moa_result).await;
                                } else {
                                    let _ = proxy::send_json_ok(tcp_stream, &moa_result).await;
                                }
                                return;
                            }
                        }
                        // Fallback: not enough models or no body
                        let _ = proxy::send_503(
                            tcp_stream,
                            "MoA requires ≥2 models available in the mesh",
                        )
                        .await;
                        return;
                    }
                    // ── end MoA ──────────────────────────────────────

                    let use_pipeline = classification
                        .as_ref()
                        .map(pipeline::should_pipeline)
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
                                request.ensure_body_json();
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

                    let target = if let Some(ref name) = effective_model {
                        if !has_available_candidates(&targets, name) {
                            let remote_hosts = node.hosts_for_model(name).await;
                            if !remote_hosts.is_empty() {
                                let mut mesh_targets = targets.clone();
                                mesh_targets.targets.insert(
                                    name.clone(),
                                    remote_hosts
                                        .into_iter()
                                        .map(election::InferenceTarget::Remote)
                                        .collect(),
                                );
                                let routed = proxy::route_model_request(
                                    node.clone(),
                                    tcp_stream,
                                    &mesh_targets,
                                    name,
                                    &request,
                                    required_tokens,
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
                            if let Some(plugin_manager) = plugin_manager.as_ref() {
                                match plugin_manager.inference_endpoint_for_model(name).await {
                                    Ok(Some(endpoint)) => {
                                        let routed = proxy::route_http_endpoint_request(
                                            &node,
                                            Some(name),
                                            &mut tcp_stream,
                                            &endpoint.address,
                                            &request.raw,
                                            &request.path,
                                            request.response_adapter,
                                        )
                                        .await;
                                        proxy::release_request_objects(
                                            &node,
                                            &request.request_object_request_ids,
                                        )
                                        .await;
                                        if !routed {
                                            let _ = proxy::send_503(
                                                tcp_stream,
                                                &format!(
                                                    "plugin endpoint for model '{name}' failed"
                                                ),
                                            )
                                            .await;
                                        }
                                        return;
                                    }
                                    Ok(None) => {
                                        tracing::debug!(
                                            "Model '{}' not found, trying first available",
                                            name
                                        );
                                        first_available_target(&targets)
                                    }
                                    Err(err) => {
                                        tracing::warn!(
                                            "API proxy: failed to resolve external endpoint for model '{}': {}",
                                            name,
                                            err
                                        );
                                        first_available_target(&targets)
                                    }
                                }
                            } else {
                                tracing::debug!(
                                    "Model '{}' not found, trying first available",
                                    name
                                );
                                first_available_target(&targets)
                            }
                        } else {
                            if targets.candidates(name).len() > 1 {
                                request.ensure_body_json();
                            }
                            let routed = proxy::route_model_request(
                                node.clone(),
                                tcp_stream,
                                &targets,
                                name,
                                &request,
                                required_tokens,
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
                        effective_model.as_deref(),
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
                }
            };
        });
    }
}

/// Bootstrap proxy: runs during GPU startup, tunnels all requests to mesh hosts.
/// Returns the TcpListener when signaled to stop (so api_proxy can take it over).
pub(crate) async fn bootstrap_proxy(
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
    let _ = emit_event(OutputEvent::Info {
        message: format!("API ready (bootstrap): http://localhost:{port}"),
        context: Some("bootstrap_proxy".to_string()),
    });
    let _ = emit_event(OutputEvent::Info {
        message: "Requests tunneled to mesh while GPU loads...".to_string(),
        context: Some("bootstrap_proxy".to_string()),
    });

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
                tokio::spawn(Box::pin(proxy::handle_mesh_request(node, tcp_stream, true, affinity)));
            }
            resp_tx = stop_rx.recv() => {
                if let Some(tx) = resp_tx {
                    let _ = emit_event(OutputEvent::Info {
                        message: "Bootstrap proxy handing off to full API proxy".to_string(),
                        context: Some("bootstrap_proxy".to_string()),
                    });
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

fn has_available_candidates(targets: &election::ModelTargets, model: &str) -> bool {
    targets
        .candidates(model)
        .iter()
        .any(|target| !matches!(target, election::InferenceTarget::None))
}

/// Wrap a completed MoA chat-completion response as SSE frames so streaming
/// clients (like Goose) can consume it.  Emits one delta chunk with the full
/// content, then a `finish_reason: stop` chunk, then `[DONE]`.
async fn send_moa_as_sse(
    mut stream: tokio::net::TcpStream,
    response: &serde_json::Value,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;

    let id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-moa");
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("moa");
    let raw_content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    // Strip <think>...</think> tags so streaming clients get clean content
    let content = strip_think_from_content(raw_content);

    // Check if this is a tool_calls response
    let tool_calls = response
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
        .cloned();

    let finish_reason = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };

    // Content or tool_calls chunk
    let delta = if let Some(ref tcs) = tool_calls {
        // For tool_calls, emit them in the delta
        serde_json::json!({
            "role": "assistant",
            "tool_calls": tcs.iter().enumerate().map(|(i, tc)| {
                serde_json::json!({
                    "index": i,
                    "id": tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_0"),
                    "type": "function",
                    "function": tc.get("function").cloned().unwrap_or(serde_json::json!({})),
                })
            }).collect::<Vec<_>>()
        })
    } else {
        serde_json::json!({ "role": "assistant", "content": content })
    };

    let chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": null,
        }]
    });
    let data = format!("data: {}\n\n", chunk);
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;

    // Stop chunk
    let stop = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }]
    });
    let data = format!("data: {}\n\n", stop);
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;

    // Done
    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;

    // Terminating chunk
    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// Strip `<think>...</think>` tags and orphan `</think>` from content.
fn strip_think_from_content(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</think>".len()..]
            );
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    result = result.replace("</think>", "");
    result.trim().to_string()
}

/// Handle a MoA (mixture-of-agents) request by fanning out to all available
/// models through the local API proxy and arbitrating the responses.
///
/// Each worker endpoint points back at `localhost:{api_port}/v1` with a
/// different model name — the existing proxy handles routing each one to
/// the correct local or remote backend.  This avoids duplicating any
/// routing/tunnel logic.
async fn handle_moa_request(
    api_port: u16,
    model_names: &[String],
    body: serde_json::Value,
) -> serde_json::Value {
    let base_url = format!("http://localhost:{api_port}/v1");

    // Dedup model aliases — e.g. "unsloth/GLM-4.7-Flash-GGUF" and
    // "unsloth/GLM-4.7-Flash-GGUF@main:Q4_K_M" are the same model.
    // Keep the shorter name (canonical) and skip aliases.
    let mut seen_bases = std::collections::HashSet::new();
    let mut endpoints = Vec::new();
    let mut sorted_names = model_names.to_vec();
    sorted_names.sort_by_key(|n| n.len());
    for name in &sorted_names {
        let base = name
            .split('@')
            .next()
            .unwrap_or(name)
            .to_lowercase()
            .replace("-gguf", "")
            .replace("unsloth/", "")
            .replace("meshllm/", "");
        if seen_bases.insert(base) {
            endpoints.push(moa_gateway::Endpoint {
                base_url: base_url.clone(),
                model: name.clone(),
            });
        }
    }

    let config = moa_gateway::GatewayConfig {
        endpoints,
        worker_timeout: std::time::Duration::from_secs(30),
        reducer_timeout: std::time::Duration::from_secs(45),
    };

    let mut gateway = moa_gateway::Gateway::new(config);
    let result = gateway.turn(&body).await;

    tracing::info!(
        "moa: {}ms, {}/{} workers, reducer={}",
        result.elapsed_ms,
        result
            .worker_summaries
            .iter()
            .filter(|w| w.succeeded)
            .count(),
        result.worker_summaries.len(),
        result.reducer_used,
    );

    result.response_body
}

pub(crate) fn callable_models(targets: &election::ModelTargets) -> Vec<String> {
    let mut models: Vec<String> = targets
        .targets
        .iter()
        .filter(|(_, hosts)| {
            hosts
                .iter()
                .any(|target| !matches!(target, election::InferenceTarget::None))
        })
        .map(|(name, _)| name.clone())
        .collect();
    models.sort();
    models
}
