use crate::network::openai::transport;
use anyhow::Result;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Semaphore, TryAcquireError};
use tokio::task::JoinSet;
use tokio::time::{sleep, Duration};

/// Maximum concurrent requests allowed through the backend proxy.
///
/// Must match the `--parallel` slot count passed to llama-server in
/// `inference/launch.rs`. When inflight requests reach this cap, the proxy
/// drops new TCP connections instead of forwarding them, which prevents
/// llama.cpp's internal `queue_tasks_deferred` from growing unboundedly.
///
/// Clients observe the dropped connection as a connect/EOF failure, which
/// the OpenAI transport layer already classifies as `RetryableUnavailable`
/// and retries against the next candidate host/model.
pub(crate) const BACKEND_PROXY_MAX_INFLIGHT: usize = 4;

#[derive(Debug)]
pub(crate) struct BackendProxyHandle {
    port: u16,
    stop_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl BackendProxyHandle {
    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) async fn shutdown(self) {
        let _ = self.stop_tx.send(true);
        let _ = self.task.await;
    }
}

pub(crate) async fn start_backend_proxy(llama_port: u16) -> Result<BackendProxyHandle> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let (stop_tx, mut stop_rx) = watch::channel(false);

    // Local semaphore gates concurrent forwarded connections at the cap.
    // A permit is acquired in the accept loop, *before* forwarding to
    // llama-server, and held for the full connection lifetime. When all
    // permits are in use, new connections are dropped immediately —
    // preventing llama.cpp's `queue_tasks_deferred` from growing unbounded.
    //
    // The gate lives here (not on the mesh Node counter) so that it can't
    // race with upstream callers that increment the counter only *after*
    // a successful connect.
    let slots = Arc::new(Semaphore::new(BACKEND_PROXY_MAX_INFLIGHT));

    let task = tokio::spawn(async move {
        let mut connections = JoinSet::new();
        loop {
            tokio::select! {
                changed = stop_rx.changed() => {
                    if changed.is_err() || *stop_rx.borrow() {
                        break;
                    }
                }
                Some(result) = connections.join_next(), if !connections.is_empty() => {
                    if let Err(err) = result {
                        if !err.is_cancelled() {
                            tracing::debug!("backend proxy connection task failed: {err}");
                        }
                    }
                }
                accept_result = listener.accept() => match accept_result {
                    Ok((stream, _)) => {
                        let permit = match Arc::clone(&slots).try_acquire_owned() {
                            Ok(p) => p,
                            Err(TryAcquireError::NoPermits) => {
                                tracing::warn!(
                                    cap = BACKEND_PROXY_MAX_INFLIGHT,
                                    "backend proxy at capacity; dropping incoming connection"
                                );
                                drop(stream);
                                continue;
                            }
                            Err(TryAcquireError::Closed) => {
                                // Semaphore is only closed on shutdown; bail.
                                drop(stream);
                                break;
                            }
                        };
                        connections.spawn(async move {
                            let _permit = permit;
                            if let Err(err) = handle_connection(stream, llama_port).await {
                                tracing::debug!("backend proxy request failed: {err}");
                            }
                        });
                    }
                    Err(err) => {
                        tracing::warn!("backend proxy accept error: {err}");
                        sleep(Duration::from_millis(50)).await;
                    }
                }
            }
        }

        connections.abort_all();
        while let Some(result) = connections.join_next().await {
            if let Err(err) = result {
                if !err.is_cancelled() {
                    tracing::debug!("backend proxy shutdown join failed: {err}");
                }
            }
        }
    });

    Ok(BackendProxyHandle {
        port,
        stop_tx,
        task,
    })
}

async fn handle_connection(mut stream: TcpStream, llama_port: u16) -> Result<()> {
    let _ = stream.set_nodelay(true);
    let request = match transport::read_http_request(&mut stream).await {
        Ok(request) => request,
        Err(err) => {
            let _ = transport::send_400(stream, &err.to_string()).await;
            return Ok(());
        }
    };

    let mut upstream = match TcpStream::connect(format!("127.0.0.1:{llama_port}")).await {
        Ok(stream) => stream,
        Err(err) => {
            tracing::warn!("failed to connect to llama backend on port {llama_port}: {err}");
            let _ = transport::send_503(stream, "llama backend unavailable").await;
            return Ok(());
        }
    };
    let _ = upstream.set_nodelay(true);

    if let Err(err) = upstream.write_all(&request.raw).await {
        tracing::warn!("failed to write request to llama backend on port {llama_port}: {err}");
        let _ = transport::send_503(stream, "llama backend unavailable").await;
        return Ok(());
    }

    // The buffered request is already written to upstream; relay the response back to the client.
    let _ = tokio::io::copy(&mut upstream, &mut stream).await;
    let _ = stream.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio::time::timeout;

    /// Starts a minimal dummy HTTP server that echoes a fixed response for every connection.
    async fn start_dummy_upstream(response: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut conn, _)) = listener.accept().await {
                let response = response;
                tokio::spawn(async move {
                    // Drain the incoming request so the client gets its response.
                    let mut buf = [0u8; 4096];
                    let _ = conn.read(&mut buf).await;
                    let _ = conn.write_all(response.as_bytes()).await;
                });
            }
        });
        port
    }

    async fn start_recording_upstream(response: &'static str) -> (u16, oneshot::Receiver<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (request_tx, request_rx) = oneshot::channel();
        tokio::spawn(async move {
            if let Ok((mut conn, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let read = conn.read(&mut buf).await.unwrap_or(0);
                let _ = request_tx.send(buf[..read].to_vec());
                let _ = conn.write_all(response.as_bytes()).await;
            }
        });
        (port, request_rx)
    }

    async fn start_stalled_upstream() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut conn, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = conn.read(&mut buf).await;
                    sleep(Duration::from_secs(60)).await;
                });
            }
        });
        port
    }

    #[tokio::test]
    async fn test_backend_proxy_forwards_request_and_response() {
        let upstream_response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let (upstream_port, request_rx) = start_recording_upstream(upstream_response).await;

        let proxy = start_backend_proxy(upstream_port).await.unwrap();
        let proxy_port = proxy.port();

        let mut client = TcpStream::connect(format!("127.0.0.1:{proxy_port}"))
            .await
            .unwrap();
        let req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        client.write_all(req).await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let response = String::from_utf8_lossy(&buf);
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "unexpected response: {response:?}"
        );
        assert!(response.contains("{}"), "body not forwarded: {response:?}");
        let forwarded = String::from_utf8(request_rx.await.unwrap()).unwrap();
        assert!(
            forwarded.contains("Connection: close\r\n"),
            "forwarded request should force connection close: {forwarded:?}"
        );

        proxy.shutdown().await;
    }

    #[tokio::test]
    async fn test_backend_proxy_shutdown_stops_accepting() {
        let upstream_response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let upstream_port = start_dummy_upstream(upstream_response).await;

        let proxy = start_backend_proxy(upstream_port).await.unwrap();
        let proxy_port = proxy.port();

        // Confirm proxy accepts connections before shutdown.
        TcpStream::connect(format!("127.0.0.1:{proxy_port}"))
            .await
            .expect("should connect before shutdown");

        proxy.shutdown().await;

        // After shutdown the accept loop is aborted; new connections should be refused.
        // Give the OS a moment to release the port then verify connection is refused.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let result = TcpStream::connect(format!("127.0.0.1:{proxy_port}")).await;
        assert!(
            result.is_err(),
            "connection should be refused after shutdown"
        );
    }

    #[tokio::test]
    async fn test_backend_proxy_shutdown_aborts_inflight_connections() {
        let upstream_port = start_stalled_upstream().await;

        let proxy = start_backend_proxy(upstream_port).await.unwrap();
        let proxy_port = proxy.port();

        let mut client = TcpStream::connect(format!("127.0.0.1:{proxy_port}"))
            .await
            .unwrap();
        let req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        client.write_all(req).await.unwrap();

        proxy.shutdown().await;

        let mut buf = [0u8; 1];
        let read = timeout(Duration::from_secs(1), client.read(&mut buf))
            .await
            .expect("client read should complete after shutdown");
        match read {
            Ok(0) => {}
            Ok(_) => panic!("expected backend proxy shutdown to close the client stream"),
            Err(err) => assert!(
                matches!(
                    err.kind(),
                    std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::BrokenPipe
                ),
                "unexpected shutdown read error: {err}"
            ),
        }
    }

    #[tokio::test]
    async fn test_backend_proxy_returns_503_when_upstream_unavailable() {
        // Use a port that has nothing listening on it.
        let dead_port = {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            l.local_addr().unwrap().port()
            // listener dropped — port freed
        };

        let proxy = start_backend_proxy(dead_port).await.unwrap();
        let proxy_port = proxy.port();

        let mut client = TcpStream::connect(format!("127.0.0.1:{proxy_port}"))
            .await
            .unwrap();
        let req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        client.write_all(req).await.unwrap();

        let mut buf = Vec::new();
        client.read_to_end(&mut buf).await.unwrap();
        let response = String::from_utf8_lossy(&buf);
        assert!(
            response.starts_with("HTTP/1.1 503"),
            "expected 503, got: {response:?}"
        );

        proxy.shutdown().await;
    }

    #[tokio::test]
    async fn test_backend_proxy_drops_connection_when_at_capacity() {
        // A stalled upstream never responds, so each forwarded request
        // holds its permit indefinitely. Fire enough clients to exhaust
        // the cap, then assert that the next client gets dropped without
        // a response instead of being queued.
        let upstream_port = start_stalled_upstream().await;
        let proxy = start_backend_proxy(upstream_port).await.unwrap();
        let proxy_port = proxy.port();

        let req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";

        // Saturate the permits. Keep the streams alive for the duration
        // of the test so the proxy doesn't free any slot.
        let mut saturating = Vec::new();
        for _ in 0..BACKEND_PROXY_MAX_INFLIGHT {
            let mut s = TcpStream::connect(format!("127.0.0.1:{proxy_port}"))
                .await
                .unwrap();
            s.write_all(req).await.unwrap();
            saturating.push(s);
        }

        // Give the accept loop a moment to pick each one up and acquire
        // a permit. Without this sleep the 5th client below could race
        // the accept of the 4th, permits wouldn't yet be exhausted, and
        // it would get forwarded to the stalled upstream instead of
        // dropped.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // The (cap + 1)-th connection should be accepted at the OS
        // level, then immediately dropped by the proxy. The client
        // observes EOF before any response bytes.
        let mut overflow = TcpStream::connect(format!("127.0.0.1:{proxy_port}"))
            .await
            .unwrap();
        let _ = overflow.write_all(req).await;
        let mut buf = Vec::new();
        let read = timeout(Duration::from_secs(2), overflow.read_to_end(&mut buf)).await;
        match read {
            Ok(Ok(_)) => assert!(
                buf.is_empty(),
                "expected no response bytes from dropped connection, got: {:?}",
                String::from_utf8_lossy(&buf)
            ),
            Ok(Err(_)) => {}
            Err(_) => panic!("proxy did not drop overflow connection within timeout"),
        }

        drop(saturating);
        proxy.shutdown().await;
    }
}
