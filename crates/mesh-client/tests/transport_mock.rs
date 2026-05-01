use mesh_client::network::transport::{MockTransportIo, TransportIo};

#[tokio::test]
async fn mock_transport_roundtrip() {
    let data = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec();
    let mut mock = MockTransportIo::new(data.clone());
    let mut buf = vec![0u8; data.len()];
    mock.read_exact(&mut buf).await.unwrap();
    assert_eq!(buf, data);
}

#[tokio::test]
async fn mock_transport_write_captured() {
    let mut mock = MockTransportIo::new(Vec::new());
    mock.write_all(b"hello").await.unwrap();
    mock.write_all(b" world").await.unwrap();
    assert_eq!(mock.write_buf, b"hello world");
}

#[tokio::test]
async fn mock_transport_read_variable() {
    let data = b"some bytes".to_vec();
    let mut mock = MockTransportIo::new(data.clone());
    let mut buf = [0u8; 4];
    let n = mock.read(&mut buf).await.unwrap();
    assert_eq!(n, 4);
    assert_eq!(&buf[..n], b"some");
}

#[tokio::test]
async fn mock_transport_shutdown_ok() {
    let mut mock = MockTransportIo::new(Vec::new());
    mock.shutdown().await.unwrap();
}
