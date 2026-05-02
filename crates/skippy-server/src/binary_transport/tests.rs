use super::{prepare_binary_stage_connection, tensor_name_requires_recurrent_state};
use std::{
    io,
    net::{TcpListener, TcpStream},
    os::fd::AsRawFd,
    thread,
    time::Duration,
};

#[test]
fn accepted_binary_stage_connection_is_blocking() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    let client = thread::spawn(move || TcpStream::connect(addr).unwrap());

    let (stream, _) = loop {
        match listener.accept() {
            Ok(conn) => break conn,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("accept failed: {error}"),
        }
    };
    stream.set_nonblocking(true).unwrap();
    prepare_binary_stage_connection(&stream).unwrap();

    let flags = unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_GETFL) };
    assert_ne!(flags, -1);
    assert_eq!(flags & libc::O_NONBLOCK, 0);
    drop(client.join().unwrap());
}

#[test]
fn recurrent_tensor_names_disable_kv_only_cache() {
    assert!(tensor_name_requires_recurrent_state("blk.0.ssm_a"));
    assert!(tensor_name_requires_recurrent_state(
        "blk.0.ssm_conv1d.weight"
    ));
    assert!(tensor_name_requires_recurrent_state(
        "blk.0.time_mix_k.weight"
    ));
    assert!(tensor_name_requires_recurrent_state(
        "blk.0.rwkv_gate.weight"
    ));
    assert!(!tensor_name_requires_recurrent_state("blk.0.attn_q.weight"));
    assert!(!tensor_name_requires_recurrent_state(
        "blk.0.ffn_down.weight"
    ));
}
