use mesh_client::network::http_parse::{is_models_list_request, read_http_request};
use mesh_client::network::transport::MockTransportIo;

#[tokio::test]
async fn parse_models_request() {
    let data = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n".to_vec();
    let mut mock = MockTransportIo::new(data);
    let req = read_http_request(&mut mock).await.unwrap();
    assert!(is_models_list_request(&req.method, &req.path));
}

#[tokio::test]
async fn parse_chat_request() {
    let body = r#"{"model":"test","messages":[]}"#;
    let data = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    )
    .into_bytes();
    let mut mock = MockTransportIo::new(data);
    let req = read_http_request(&mut mock).await.unwrap();
    assert!(!is_models_list_request(&req.method, &req.path));
}

#[tokio::test]
async fn parse_chat_request_extracts_model_name() {
    let body = r#"{"model":"llama3","messages":[]}"#;
    let data = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    )
    .into_bytes();
    let mut mock = MockTransportIo::new(data);
    let req = read_http_request(&mut mock).await.unwrap();
    assert_eq!(req.model_name.as_deref(), Some("llama3"));
}

#[tokio::test]
async fn models_list_request_rejects_non_get() {
    assert!(!is_models_list_request("POST", "/v1/models"));
    assert!(is_models_list_request("GET", "/v1/models"));
    assert!(is_models_list_request("GET", "/models"));
}
