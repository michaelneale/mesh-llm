use mesh_client::network::transport::MockTransportIo;
use mesh_client::network::tunnel::relay_with_rewrite;

#[tokio::test]
async fn relay_transfers_bytes() {
    let data = b"GET /v1/models HTTP/1.1\r\n\r\n".to_vec();
    let mut incoming = MockTransportIo::new(data.clone());
    let mut outgoing = MockTransportIo::new(vec![]);
    let bytes = relay_with_rewrite(&mut incoming, &mut outgoing)
        .await
        .unwrap();
    assert_eq!(bytes, data.len() as u64);
    assert_eq!(outgoing.write_buf, data);
}
