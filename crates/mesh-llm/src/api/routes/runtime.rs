use super::super::{
    http::{respond_error, respond_json, respond_runtime_error},
    status::decode_runtime_model_path,
    MeshApi, RuntimeControlRequest,
};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match (method, path_only) {
        ("GET", "/api/status") => handle_status(stream, state).await,
        ("GET", "/api/models") => handle_models(stream, state).await,
        ("GET", "/api/runtime") => handle_runtime_status(stream, state).await,
        ("GET", "/api/runtime/llama") => handle_runtime_llama(stream, state).await,
        ("GET", "/api/runtime/events") => handle_runtime_events(stream, state).await,
        ("GET", "/api/runtime/endpoints") => handle_runtime_endpoints(stream, state).await,
        ("GET", "/api/runtime/processes") => handle_runtime_processes(stream, state).await,
        ("GET", "/api/runtime/stages") => handle_runtime_stages(stream, state).await,
        ("POST", "/api/runtime/models") => handle_load_model(stream, state, body).await,
        ("DELETE", p) if p.starts_with("/api/runtime/models/") => {
            handle_unload_model(stream, state, p).await
        }
        ("GET", "/api/events") => handle_events(stream, state).await,
        _ => Ok(()),
    }
}

async fn handle_runtime_stages(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_stages()).await {
        Ok(runtime_stages) => respond_json(stream, 200, &runtime_stages).await,
        Err(_) => respond_error(stream, 503, "Runtime stage status temporarily unavailable").await,
    }
}

async fn handle_status(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.status()).await {
        Ok(status) => respond_json(stream, 200, &status).await,
        Err(_) => respond_error(stream, 503, "Status temporarily unavailable").await,
    }
}

async fn handle_models(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let mesh_models = state.mesh_models().await;
    respond_json(
        stream,
        200,
        &serde_json::json!({ "mesh_models": mesh_models }),
    )
    .await
}

async fn handle_runtime_status(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_status()).await {
        Ok(runtime_status) => respond_json(stream, 200, &runtime_status).await,
        Err(_) => respond_error(stream, 503, "Runtime status temporarily unavailable").await,
    }
}

async fn handle_runtime_processes(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_processes()).await {
        Ok(runtime_processes) => respond_json(stream, 200, &runtime_processes).await,
        Err(_) => {
            respond_error(
                stream,
                503,
                "Runtime process status temporarily unavailable",
            )
            .await
        }
    }
}

async fn handle_runtime_llama(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_llama()).await {
        Ok(runtime_llama) => respond_json(stream, 200, &runtime_llama).await,
        Err(_) => {
            respond_error(
                stream,
                503,
                "Runtime llama snapshot temporarily unavailable",
            )
            .await
        }
    }
}

async fn handle_runtime_events(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;

    let mut subscription = {
        state
            .inner
            .lock()
            .await
            .runtime_data_collector
            .clone()
            .subscribe()
    };
    let mut last_sent_json = None;

    let runtime_llama = state.runtime_llama().await;
    if let Ok(json) = serde_json::to_string(&runtime_llama) {
        stream
            .write_all(format!("data: {json}\n\n").as_bytes())
            .await?;
        last_sent_json = Some(json);
    }

    loop {
        tokio::select! {
            changed = subscription.changed() => {
                match changed {
                    Ok(()) => {
                        let subscription_state = *subscription.borrow_and_update();
                        if !subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::RUNTIME) {
                            continue;
                        }
                        let runtime_llama = state.runtime_llama().await;
                        let Ok(json) = serde_json::to_string(&runtime_llama) else {
                            continue;
                        };
                        if last_sent_json.as_deref() == Some(json.as_str()) {
                            continue;
                        }
                        if stream.write_all(format!("data: {json}\n\n").as_bytes()).await.is_err() {
                            break;
                        }
                        last_sent_json = Some(json);
                    }
                    Err(_) => break,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                if stream.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn handle_runtime_endpoints(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match state.runtime_endpoints().await {
        Ok(endpoints) => {
            respond_json(stream, 200, &serde_json::json!({ "endpoints": endpoints })).await
        }
        Err(err) => respond_error(stream, 500, &err.to_string()).await,
    }
}

async fn handle_load_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };

    let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);
    match parsed {
        Ok(val) => {
            let spec = val["model"].as_str().unwrap_or("").to_string();
            if spec.is_empty() {
                respond_error(stream, 400, "Missing 'model' field").await?;
            } else {
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                let _ = control_tx.send(RuntimeControlRequest::Load {
                    spec,
                    resp: resp_tx,
                });
                match resp_rx.await {
                    Ok(Ok(loaded)) => {
                        respond_json(stream, 201, &serde_json::json!({ "loaded": loaded })).await?;
                    }
                    Ok(Err(e)) => {
                        respond_runtime_error(stream, &e.to_string()).await?;
                    }
                    Err(_) => {
                        respond_error(stream, 503, "Runtime control unavailable").await?;
                    }
                }
            }
        }
        Err(_) => {
            respond_error(stream, 400, "Invalid JSON body").await?;
        }
    }
    Ok(())
}

async fn handle_unload_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };
    let Some(model_name) = decode_runtime_model_path(path) else {
        return respond_error(stream, 400, "Missing model path").await;
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::Unload {
        model: model_name.clone(),
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(())) => {
            respond_json(stream, 200, &serde_json::json!({ "dropped": model_name })).await?;
        }
        Ok(Err(e)) => {
            respond_runtime_error(stream, &e.to_string()).await?;
        }
        Err(_) => {
            respond_error(stream, 503, "Runtime control unavailable").await?;
        }
    }
    Ok(())
}

async fn handle_events(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;

    let status = state.status().await;
    if let Ok(json) = serde_json::to_string(&status) {
        stream
            .write_all(format!("data: {json}\n\n").as_bytes())
            .await?;
    }

    let mut subscription = {
        state
            .inner
            .lock()
            .await
            .runtime_data_collector
            .clone()
            .subscribe()
    };

    loop {
        tokio::select! {
            changed = subscription.changed() => {
                match changed {
                    Ok(()) => {
                        let subscription_state = *subscription.borrow_and_update();
                        let interesting = subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::STATUS)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::MODELS)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::ROUTING)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::PROCESSES)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::INVENTORY)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::PLUGINS);
                        if !interesting {
                            continue;
                        }
                        let status = state.status().await;
                        let Ok(json) = serde_json::to_string(&status) else {
                            continue;
                        };
                        if stream.write_all(format!("data: {json}\n\n").as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                if stream.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
            }
        }
    }

    Ok(())
}
