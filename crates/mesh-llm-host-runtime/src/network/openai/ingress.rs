use crate::api;
use crate::cli::output::{emit_event, OutputEvent};
use crate::inference::{election, pipeline};
use crate::mesh;
use crate::network::affinity;
use crate::network::openai::transport as proxy;
use crate::network::router;
use mesh_mixture_of_agents as moa;

enum AutoRouteResolution {
    Continue {
        effective_model: Option<String>,
        classification: Option<router::Classification>,
    },
    MediaUnsupported,
}

enum MissingModelRouteResult {
    Routed,
    Fallback(tokio::net::TcpStream),
}

struct IngressRouteContext<'a> {
    node: &'a mesh::Node,
    targets: &'a election::ModelTargets,
    affinity: &'a affinity::AffinityRouter,
    plugin_manager: Option<&'a crate::plugin::PluginManager>,
}

struct ProxyConnectionContext<'a> {
    route: IngressRouteContext<'a>,
    control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
}

struct AutoRouteDecision {
    effective_model: Option<String>,
    classification: Option<router::Classification>,
    required_tokens: Option<u32>,
}

async fn bind_api_proxy_listener(
    port: u16,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
) -> Option<tokio::net::TcpListener> {
    match existing_listener {
        Some(listener) => Some(listener),
        None => {
            let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
            match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
                Ok(listener) => Some(listener),
                Err(error) => {
                    tracing::error!("Failed to bind API proxy to port {port}: {error}");
                    None
                }
            }
        }
    }
}

async fn send_runtime_control_response<T, F>(
    tcp_stream: tokio::net::TcpStream,
    response: Result<Result<T, anyhow::Error>, tokio::sync::oneshot::error::RecvError>,
    closed_reason: &str,
    ok_response: F,
) where
    F: FnOnce(T) -> serde_json::Value,
{
    match response {
        Ok(Ok(value)) => {
            let _ = proxy::send_json_ok(tcp_stream, &ok_response(value)).await;
        }
        Ok(Err(error)) => {
            let message = error.to_string();
            let code = api::classify_runtime_error(&message);
            let _ = proxy::send_error(tcp_stream, code, &message).await;
        }
        Err(_) => {
            let _ = proxy::send_503(tcp_stream, closed_reason).await;
        }
    }
}

async fn handle_mesh_load_request(
    tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    control_tx: &tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
) {
    if let Some(spec) = request.model_name.as_ref() {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let _ = control_tx.send(api::RuntimeControlRequest::Load {
            spec: spec.clone(),
            resp: resp_tx,
        });
        send_runtime_control_response(
            tcp_stream,
            resp_rx.await,
            "runtime load channel closed",
            |loaded| {
                serde_json::json!({
                    "loaded": loaded.model,
                    "instance_id": loaded.instance_id,
                })
            },
        )
        .await;
    } else {
        let _ = proxy::send_400(tcp_stream, "missing 'model' field").await;
    }
}

async fn handle_mesh_unload_request(
    tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    control_tx: &tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
) {
    if let Some(name) = request.model_name.as_ref() {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let _ = control_tx.send(api::RuntimeControlRequest::Unload {
            target: name.clone(),
            resp: resp_tx,
        });
        send_runtime_control_response(
            tcp_stream,
            resp_rx.await,
            "runtime unload channel closed",
            |dropped| {
                serde_json::json!({
                    "dropped": dropped.model,
                    "instance_id": dropped.instance_id,
                })
            },
        )
        .await;
    } else {
        let _ = proxy::send_400(tcp_stream, "missing 'model' field").await;
    }
}

async fn handle_models_list_request(
    tcp_stream: tokio::net::TcpStream,
    node: &mesh::Node,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
) {
    let mut models = callable_models(targets);
    models.extend(node.models_being_served().await);
    if let Some(plugin_manager) = plugin_manager {
        if let Ok(mut external_models) = plugin_manager.inference_models().await {
            models.append(&mut external_models);
        }
    }
    models.sort();
    models.dedup();
    let descriptors = node.served_model_descriptors().await;
    let _ = proxy::send_models_list_with_descriptors(tcp_stream, &models, &descriptors).await;
}

async fn collect_available_models_for_auto_route(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
) -> Vec<String> {
    let mut available_models = callable_models(targets);
    for name in node.models_being_served().await {
        if !available_models.iter().any(|existing| existing == &name) {
            available_models.push(name);
        }
    }
    if let Some(plugin_manager) = plugin_manager {
        if let Ok(external_models) = plugin_manager.inference_models().await {
            for name in external_models {
                if !available_models.iter().any(|existing| existing == &name) {
                    available_models.push(name);
                }
            }
        }
    }
    available_models
}

async fn resolve_auto_routed_model(
    node: &mesh::Node,
    request: &mut proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
    descriptors: &[crate::mesh::ServedModelDescriptor],
) -> AutoRouteResolution {
    if request.model_name.is_some() && request.model_name.as_deref() != Some("auto") {
        return AutoRouteResolution::Continue {
            effective_model: request.model_name.clone(),
            classification: None,
        };
    }

    request.ensure_body_json();
    let Some(body_json) = request.body_json.as_ref() else {
        return AutoRouteResolution::Continue {
            effective_model: None,
            classification: None,
        };
    };

    let classification = router::classify(body_json);
    let media = router::media_requirements(body_json);
    let available_models =
        collect_available_models_for_auto_route(node, targets, plugin_manager).await;
    let metrics = node.routing_metrics();
    let available: Vec<router::RoutingCandidate<'_>> = available_models
        .iter()
        .map(|name| {
            let caps = proxy::capabilities_for_model(name, descriptors);
            let (tps_hint, throughput_samples) = metrics
                .tps_for_model(name)
                .map(|(t, s)| (Some(t), s))
                .unwrap_or((None, 0));
            router::RoutingCandidate {
                name: name.as_str(),
                caps,
                tps_hint,
                throughput_samples,
            }
        })
        .collect();
    let Some(available) = router::filter_media_compatible_candidates(&available, &media) else {
        proxy::release_request_objects(node, &request.request_object_request_ids).await;
        return AutoRouteResolution::MediaUnsupported;
    };

    let effective_model = router::pick_model_classified(&classification, &available).map(|name| {
        tracing::info!(
            "router: {:?}/{:?} tools={} → {name}",
            classification.category,
            classification.complexity,
            classification.needs_tools
        );
        name.to_string()
    });

    AutoRouteResolution::Continue {
        effective_model,
        classification: Some(classification),
    }
}

fn maybe_enable_auto_route_hooks(
    request: &mut proxy::BufferedHttpRequest,
    effective_model: Option<&str>,
) {
    if request.model_name.is_none() || request.model_name.as_deref() == Some("auto") {
        proxy::inject_mesh_hooks_flag(&mut request.raw, true);
        if let Some(model) = effective_model {
            proxy::rewrite_model_field(request, model);
        }
    }
}

async fn try_pipeline_proxy(
    node: &mesh::Node,
    tcp_stream: &mut tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    strong_name: &str,
) -> bool {
    let Some((planner_name, planner_port, strong_port)) =
        pipeline_local_ports(targets, strong_name)
    else {
        return false;
    };

    request.ensure_body_json();
    let Some(body_json) = request.body_json.clone() else {
        warn_pipeline_fallback(strong_name);
        return false;
    };

    tracing::info!("pipeline: {planner_name} (plan) → {strong_name} (execute)");
    let handled = matches!(
        proxy::pipeline_proxy_local(
            tcp_stream,
            &request.path,
            body_json,
            planner_port,
            &planner_name,
            strong_port,
            node,
        )
        .await,
        proxy::PipelineProxyResult::Handled
    );
    if !handled {
        warn_pipeline_fallback(strong_name);
    }
    handled
}

fn pipeline_local_ports(
    targets: &election::ModelTargets,
    strong_name: &str,
) -> Option<(String, u16, u16)> {
    let (planner_name, planner_port) = targets
        .targets
        .iter()
        .find(|(name, target_vec)| {
            *name != strong_name
                && target_vec
                    .iter()
                    .any(|target| matches!(target, election::InferenceTarget::Local(_)))
        })
        .and_then(|(name, target_vec)| {
            target_vec.iter().find_map(|target| match target {
                election::InferenceTarget::Local(port) => Some((name.clone(), *port)),
                _ => None,
            })
        })?;
    let strong_port = targets.targets.get(strong_name).and_then(|target_vec| {
        target_vec.iter().find_map(|target| match target {
            election::InferenceTarget::Local(port) => Some(*port),
            _ => None,
        })
    })?;
    Some((planner_name, planner_port, strong_port))
}

fn warn_pipeline_fallback(strong_name: &str) {
    tracing::warn!("pipeline: falling back to direct proxy for {strong_name}");
}

async fn route_missing_local_model(
    tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    required_tokens: Option<u32>,
) -> MissingModelRouteResult {
    if let Some(mesh_targets) = remote_mesh_targets(ctx, model_name).await {
        let routed = proxy::route_model_request(
            ctx.node.clone(),
            tcp_stream,
            &mesh_targets,
            model_name,
            request,
            required_tokens,
            ctx.affinity,
        )
        .await;
        debug_assert!(routed);
        return MissingModelRouteResult::Routed;
    }

    if ctx.plugin_manager.is_some() {
        return try_route_plugin_model(ctx, tcp_stream, request, model_name).await;
    }

    tracing::debug!("Model '{}' not found, trying first available", model_name);
    MissingModelRouteResult::Fallback(tcp_stream)
}

async fn remote_mesh_targets(
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
) -> Option<election::ModelTargets> {
    let remote_hosts = ctx.node.hosts_for_model(model_name).await;
    if remote_hosts.is_empty() {
        return None;
    }
    let mut mesh_targets = ctx.targets.clone();
    mesh_targets.targets.insert(
        model_name.to_string(),
        remote_hosts
            .into_iter()
            .map(election::InferenceTarget::Remote)
            .collect(),
    );
    Some(mesh_targets)
}

async fn try_route_plugin_model(
    ctx: &IngressRouteContext<'_>,
    mut tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    model_name: &str,
) -> MissingModelRouteResult {
    let plugin_manager = ctx
        .plugin_manager
        .expect("plugin route called without plugin manager");
    match plugin_manager
        .inference_endpoint_for_model(model_name)
        .await
    {
        Ok(Some(endpoint)) => {
            let routed = proxy::route_http_endpoint_request(
                ctx.node,
                Some(model_name),
                &mut tcp_stream,
                &endpoint.address,
                &request.raw,
                &request.path,
                request.response_adapter,
            )
            .await;
            if !routed {
                let _ = proxy::send_503(
                    tcp_stream,
                    &format!("plugin endpoint for model '{model_name}' failed"),
                )
                .await;
            }
            MissingModelRouteResult::Routed
        }
        Ok(None) => MissingModelRouteResult::Fallback(tcp_stream),
        Err(error) => {
            tracing::warn!(
                "API proxy: failed to resolve external endpoint for model '{}': {}",
                model_name,
                error
            );
            MissingModelRouteResult::Fallback(tcp_stream)
        }
    }
}

async fn route_request(
    tcp_stream: tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    effective_model: Option<&str>,
    required_tokens: Option<u32>,
) {
    let mut tcp_stream = Some(tcp_stream);
    let target = if let Some(model_name) = effective_model {
        if !has_available_candidates(ctx.targets, model_name) {
            match route_missing_local_model(
                tcp_stream
                    .take()
                    .expect("route_request stream already taken"),
                request,
                ctx,
                model_name,
                required_tokens,
            )
            .await
            {
                MissingModelRouteResult::Routed => return,
                MissingModelRouteResult::Fallback(stream) => tcp_stream = Some(stream),
            }
            first_available_target(ctx.targets)
        } else {
            if ctx.targets.candidates(model_name).len() > 1 {
                request.ensure_body_json();
            }
            let routed = proxy::route_model_request(
                ctx.node.clone(),
                tcp_stream
                    .take()
                    .expect("route_request stream already taken"),
                ctx.targets,
                model_name,
                request,
                required_tokens,
                ctx.affinity,
            )
            .await;
            debug_assert!(routed);
            return;
        }
    } else {
        first_available_target(ctx.targets)
    };

    let _ = proxy::route_to_target(
        ctx.node.clone(),
        tcp_stream.expect("route_request stream already taken"),
        effective_model,
        target,
        &request.raw,
        request.response_adapter,
    )
    .await;
}

async fn prepare_auto_route_decision(
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    descriptors: &[crate::mesh::ServedModelDescriptor],
) -> Result<AutoRouteDecision, ()> {
    match resolve_auto_routed_model(
        ctx.node,
        request,
        ctx.targets,
        ctx.plugin_manager,
        descriptors,
    )
    .await
    {
        AutoRouteResolution::Continue {
            effective_model,
            classification,
        } => {
            maybe_enable_auto_route_hooks(request, effective_model.as_deref());
            let required_tokens = proxy::request_budget_tokens_from_parts(
                request.body_len_bytes,
                request.completion_tokens,
            );
            if let Some(name) = effective_model.as_ref() {
                ctx.node.record_request(name);
            }
            Ok(AutoRouteDecision {
                effective_model,
                classification,
                required_tokens,
            })
        }
        AutoRouteResolution::MediaUnsupported => Err(()),
    }
}

async fn send_media_unsupported(tcp_stream: tokio::net::TcpStream) {
    let _ = proxy::send_error(
        tcp_stream,
        422,
        "no served model can satisfy the requested media inputs",
    )
    .await;
}

fn callable_models_with_local_served(
    targets: &election::ModelTargets,
    local_models: Vec<String>,
) -> Vec<String> {
    let mut callable = callable_models(targets);
    for name in local_models {
        if !callable.iter().any(|existing| existing == &name) {
            callable.push(name);
        }
    }
    callable.sort();
    callable
}

async fn maybe_handle_control_request(
    tcp_stream: tokio::net::TcpStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
) -> Result<(), tokio::net::TcpStream> {
    if proxy::is_models_list_request(&request.method, &request.path) {
        handle_models_list_request(
            tcp_stream,
            ctx.route.node,
            ctx.route.targets,
            ctx.route.plugin_manager,
        )
        .await;
        return Ok(());
    }

    let path = request.path.split('?').next().unwrap_or(&request.path);
    if request.method == "POST" && path == "/mesh/load" {
        handle_mesh_load_request(tcp_stream, request, ctx.control_tx).await;
        return Ok(());
    }

    Err(tcp_stream)
}

async fn try_pipeline_route(
    tcp_stream: &mut tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    decision: &AutoRouteDecision,
) -> bool {
    let use_pipeline = decision
        .classification
        .as_ref()
        .map(pipeline::should_pipeline)
        .unwrap_or(false)
        && request.response_adapter == proxy::ResponseAdapter::None;
    if !use_pipeline {
        return false;
    }
    let Some(strong_name) = decision.effective_model.as_deref() else {
        return false;
    };
    try_pipeline_proxy(ctx.node, tcp_stream, request, ctx.targets, strong_name).await
}

enum MoaInterceptResult {
    /// MoA handled the request; the response has been written and the stream
    /// is consumed.
    Handled,
    /// Not an MoA request — caller should continue with normal routing,
    /// reusing the returned stream.
    NotMoa(tokio::net::TcpStream),
}

/// Dispatch to the MoA gateway when `model == "mesh"`. Self-gates on the
/// effective model so the call site is unconditional.
async fn try_handle_moa_intercept(
    tcp_stream: tokio::net::TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
    decision: &AutoRouteDecision,
) -> MoaInterceptResult {
    if decision.effective_model.as_deref() != Some(moa::VIRTUAL_MODEL_NAME) {
        return MoaInterceptResult::NotMoa(tcp_stream);
    }
    // try_handle_moa self-gates on the model name; the Some(_) return path
    // means it declined to handle despite our outer gate — log and treat as
    // handled (we can't reliably re-use the stream after the handoff).
    if let Some(_unused_stream) = crate::network::openai::moa_gateway::try_handle_moa(
        ctx.route.node,
        tcp_stream,
        request,
        decision.effective_model.as_deref(),
        Some(ctx.route.targets),
    )
    .await
    {
        tracing::error!("moa: try_handle_moa returned unused stream despite outer model gate");
    }
    proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
    MoaInterceptResult::Handled
}

async fn handle_buffered_api_request(
    tcp_stream: tokio::net::TcpStream,
    mut request: proxy::BufferedHttpRequest,
    ctx: ProxyConnectionContext<'_>,
) {
    let tcp_stream = match maybe_handle_control_request(tcp_stream, &request, &ctx).await {
        Ok(()) => return,
        Err(tcp_stream) => tcp_stream,
    };

    let local_models = ctx.route.node.models_being_served().await;
    let callable = callable_models_with_local_served(ctx.route.targets, local_models);
    let descriptors = ctx.route.node.served_model_descriptors().await;
    proxy::rewrite_public_model_alias(&mut request, &callable, &descriptors);

    if proxy::is_drop_request(&request.method, &request.path) {
        handle_mesh_unload_request(tcp_stream, &request, ctx.control_tx).await;
        return;
    }

    let decision = match prepare_auto_route_decision(&mut request, &ctx.route, &descriptors).await {
        Ok(decision) => decision,
        Err(()) => {
            send_media_unsupported(tcp_stream).await;
            return;
        }
    };

    let tcp_stream = match try_handle_moa_intercept(tcp_stream, &mut request, &ctx, &decision).await
    {
        MoaInterceptResult::Handled => return,
        MoaInterceptResult::NotMoa(stream) => stream,
    };

    let mut tcp_stream = tcp_stream;
    if try_pipeline_route(&mut tcp_stream, &mut request, &ctx.route, &decision).await {
        proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
        return;
    }

    route_request(
        tcp_stream,
        &mut request,
        &ctx.route,
        decision.effective_model.as_deref(),
        decision.required_tokens,
    )
    .await;
    proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
}

async fn handle_api_proxy_connection(
    node: mesh::Node,
    mut tcp_stream: tokio::net::TcpStream,
    targets: election::ModelTargets,
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    affinity: affinity::AffinityRouter,
) {
    let plugin_manager = node.plugin_manager().await;
    match proxy::read_http_request_with_plugin_manager(&mut tcp_stream, plugin_manager.as_ref())
        .await
    {
        Ok(request) => {
            let route = IngressRouteContext {
                node: &node,
                targets: &targets,
                affinity: &affinity,
                plugin_manager: plugin_manager.as_ref(),
            };
            handle_buffered_api_request(
                tcp_stream,
                request,
                ProxyConnectionContext {
                    route,
                    control_tx: &control_tx,
                },
            )
            .await;
        }
        Err(error) => {
            let _ = proxy::send_400(tcp_stream, &error.to_string()).await;
        }
    }
}

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
    let Some(listener) = bind_api_proxy_listener(port, existing_listener, listen_all).await else {
        return;
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
            handle_api_proxy_connection(node, tcp_stream, targets, control_tx, affinity).await;
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
