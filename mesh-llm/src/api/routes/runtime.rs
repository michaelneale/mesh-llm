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
        ("GET", "/api/runtime") => handle_runtime_status(stream, state).await,
        ("GET", "/api/runtime/processes") => handle_runtime_processes(stream, state).await,
        ("POST", "/api/runtime/models") => handle_load_model(stream, state, body).await,
        ("DELETE", p) if p.starts_with("/api/runtime/models/") => {
            handle_unload_model(stream, state, p).await
        }
        ("GET", "/api/events") => handle_events(stream, state).await,
        _ => Ok(()),
    }
}

async fn handle_status(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.status()).await {
        Ok(status) => respond_json(stream, 200, &status).await,
        Err(_) => respond_error(stream, 503, "Status temporarily unavailable").await,
    }
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

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    state.inner.lock().await.sse_clients.push(tx);

    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(data) => {
                        if stream.write_all(data.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    None => break,
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
