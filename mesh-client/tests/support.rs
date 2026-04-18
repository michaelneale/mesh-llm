use base64::Engine as _;
use iroh::endpoint::{QuicTransportConfig, RelayMode};
use iroh::{Endpoint, SecretKey};
use mesh_client::network::http_parse::read_http_request;
use mesh_client::network::transport::TransportIo;
use mesh_client::network::transport_iroh::IrohBiStream;
use mesh_client::protocol::{ALPN_V1, STREAM_TUNNEL_HTTP};

pub async fn spawn_mock_mesh(models: &[&str], chat_response: &str) -> String {
    let transport_config = QuicTransportConfig::builder()
        .max_concurrent_bidi_streams(64u32.into())
        .build();
    let endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V1.to_vec()])
        .relay_mode(RelayMode::Disabled)
        .transport_config(transport_config)
        .bind()
        .await
        .expect("bind test endpoint");

    let token_json = serde_json::to_vec(&endpoint.addr()).expect("serialize endpoint addr");
    let invite_token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(token_json);
    let model_entries = models
        .iter()
        .map(|id| format!("{{\"id\":\"{id}\"}}"))
        .collect::<Vec<_>>()
        .join(",");
    let chat_response = chat_response.to_string();

    tokio::spawn(async move {
        loop {
            let incoming = match endpoint.accept().await {
                Some(incoming) => incoming,
                None => break,
            };
            let mut accepting = incoming.accept().expect("accept incoming");
            let _alpn = accepting.alpn().await.expect("read alpn");
            let connection = accepting.await.expect("finish accept");
            let model_entries = model_entries.clone();
            let chat_response = chat_response.clone();

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
                    assert_eq!(stream_type[0], STREAM_TUNNEL_HTTP);

                    let model_entries = model_entries.clone();
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

    invite_token
}
