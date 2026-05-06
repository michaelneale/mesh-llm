use base64::Engine as _;
use iroh::endpoint::{QuicTransportConfig, RelayMode};
use iroh::{Endpoint, SecretKey};
use mesh_client::network::http_parse::read_http_request;
use mesh_client::network::transport::TransportIo;
use mesh_client::network::transport_iroh::IrohBiStream;
use mesh_client::protocol::{
    read_len_prefixed, write_len_prefixed, ALPN_V1, NODE_PROTOCOL_GENERATION, STREAM_GOSSIP,
    STREAM_TOPOLOGY_SUBSCRIBE, STREAM_TUNNEL_HTTP,
};
use prost::Message as _;
use std::sync::Arc;
use tokio::sync::watch;

async fn write_topology_snapshot(
    send: &mut iroh::endpoint::SendStream,
    models: &[String],
    endpoint_id: &[u8],
) {
    let peer = mesh_client::proto::node::PeerAnnouncement {
        endpoint_id: endpoint_id.to_vec(),
        role: mesh_client::proto::node::NodeRole::Host as i32,
        http_port: Some(9337),
        serving_models: models.to_vec(),
        hosted_models: models.to_vec(),
        hosted_models_known: Some(true),
        serialized_addr: Vec::new(),
        ..Default::default()
    };
    let snapshot = mesh_client::proto::node::GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        peers: vec![peer],
        sender_id: endpoint_id.to_vec(),
    };
    write_len_prefixed(send, &snapshot.encode_to_vec())
        .await
        .expect("write topology snapshot");
}

pub async fn spawn_mock_mesh(models: &[&str], chat_response: &str) -> String {
    spawn_mock_mesh_controlled(models, chat_response)
        .await
        .invite_token
}

#[allow(dead_code)]
pub struct MockMeshHandle {
    pub invite_token: String,
    models_tx: watch::Sender<Vec<String>>,
}

#[allow(dead_code)]
impl MockMeshHandle {
    pub fn set_models(&self, models: &[&str]) {
        let next = models.iter().map(|model| (*model).to_string()).collect();
        let _ = self.models_tx.send(next);
    }
}

pub async fn spawn_mock_mesh_controlled(models: &[&str], chat_response: &str) -> MockMeshHandle {
    let transport_config = QuicTransportConfig::builder()
        .max_concurrent_bidi_streams(64u32.into())
        .build();
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_V1.to_vec()])
        .relay_mode(RelayMode::Disabled)
        .transport_config(transport_config)
        .bind()
        .await
        .expect("bind test endpoint");

    let token_json = serde_json::to_vec(&endpoint.addr()).expect("serialize endpoint addr");
    let invite_token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(token_json);
    let endpoint_id = endpoint.id().as_bytes().to_vec();
    let chat_response = chat_response.to_string();
    let initial_models = models
        .iter()
        .map(|id| (*id).to_string())
        .collect::<Vec<_>>();
    let (models_tx, models_rx) = watch::channel(initial_models);

    tokio::spawn(async move {
        loop {
            let incoming = match endpoint.accept().await {
                Some(incoming) => incoming,
                None => break,
            };
            let mut accepting = incoming.accept().expect("accept incoming");
            let _alpn = accepting.alpn().await.expect("read alpn");
            let connection = accepting.await.expect("finish accept");
            let chat_response = chat_response.clone();
            let endpoint_id = endpoint_id.clone();
            let models_rx = Arc::new(models_rx.clone());

            tokio::spawn(async move {
                loop {
                    let (send, mut recv) = match connection.accept_bi().await {
                        Ok(streams) => streams,
                        Err(_) => break,
                    };

                    let mut stream_type = [0u8; 1];
                    recv.read_exact(&mut stream_type)
                        .await
                        .expect("read stream type");
                    if stream_type[0] == STREAM_GOSSIP {
                        let mut send = send;
                        let mut recv = recv;
                        let _request = read_len_prefixed(&mut recv)
                            .await
                            .expect("read gossip request");
                        let response = mesh_client::proto::node::GossipFrame {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            peers: vec![],
                            sender_id: endpoint_id.clone(),
                        };
                        write_len_prefixed(&mut send, &response.encode_to_vec())
                            .await
                            .expect("write gossip response");
                        send.finish().expect("shutdown gossip stream");
                        continue;
                    }

                    if stream_type[0] == STREAM_TOPOLOGY_SUBSCRIBE {
                        let endpoint_id = endpoint_id.clone();
                        let models_rx = models_rx.clone();
                        tokio::spawn(async move {
                            let mut send = send;
                            let mut recv = recv;
                            let _request = read_len_prefixed(&mut recv)
                                .await
                                .expect("read topology subscribe");
                            let mut models_rx = (*models_rx).clone();

                            let initial_models = models_rx.borrow().clone();
                            write_topology_snapshot(&mut send, &initial_models, &endpoint_id).await;

                            while models_rx.changed().await.is_ok() {
                                let current_models = models_rx.borrow().clone();
                                write_topology_snapshot(&mut send, &current_models, &endpoint_id)
                                    .await;
                            }
                            let _ = send.finish();
                        });
                        continue;
                    }

                    assert_eq!(stream_type[0], STREAM_TUNNEL_HTTP);

                    let current_models = models_rx.borrow().clone();
                    let model_entries = current_models
                        .iter()
                        .map(|id| format!("{{\"id\":\"{id}\"}}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    let chat_response = chat_response.clone();
                    tokio::spawn(async move {
                        let mut stream = IrohBiStream::new(send, recv);
                        let request = read_http_request(&mut stream)
                            .await
                            .expect("read tunneled HTTP request");
                        let response = match (request.method.as_str(), request.path.as_str()) {
                            ("GET", "/v1/models") => format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{{\"data\":[{}]}}",
                                model_entries
                            )
                            .into_bytes(),
                            ("GET", "/api/events") => {
                                b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nunexpected GET /api/events".to_vec()
                            }
                            ("POST", "/v1/chat/completions") => format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{{\"choices\":[{{\"message\":{{\"content\":\"{}\"}}}}]}}",
                                chat_response
                            )
                            .into_bytes(),
                            (method, path) => format!(
                                "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nunexpected {method} {path}"
                            )
                            .into_bytes(),
                        };

                        stream.write_all(&response).await.expect("write response");
                        stream.shutdown().await.expect("shutdown response stream");
                    });
                }
            });
        }
    });

    MockMeshHandle {
        invite_token,
        models_tx,
    }
}
