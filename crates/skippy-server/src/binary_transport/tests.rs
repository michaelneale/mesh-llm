use super::{prepare_binary_stage_connection, restore_prefill_decode_as_decode_message};
use std::{
    io,
    net::{TcpListener, TcpStream},
    os::fd::AsRawFd,
    thread,
    time::Duration,
};

use skippy_protocol::binary::{
    StageSamplingConfig, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
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
fn restore_prefill_decode_as_decode_preserves_chat_metadata() {
    let metadata = r#"{"grammar":"chat"}"#;
    let sampling = StageSamplingConfig {
        flags: 1,
        seed: 42,
        ..StageSamplingConfig::default()
    };
    let mut state = StageStateHeader::new(
        WireMessageKind::TryRestorePrefillDecode,
        WireActivationDType::F16,
    );
    state.prompt_token_count = 4;
    state.decode_step = 0;
    state.current_token = 104;

    let message = StageWireMessage {
        kind: WireMessageKind::TryRestorePrefillDecode,
        pos_start: 3,
        token_count: 1,
        state,
        request_id: 11,
        session_id: 13,
        sampling: Some(sampling.clone()),
        chat_sampling_metadata: Some(metadata.to_string()),
        tokens: vec![101, 102, 103, 104],
        positions: Vec::new(),
        activation: vec![1, 2, 3, 4],
        raw_bytes: Vec::new(),
    };

    let decode = restore_prefill_decode_as_decode_message(&message, 104);

    assert_eq!(decode.kind, WireMessageKind::DecodeEmbd);
    assert_eq!(decode.token_count, 1);
    assert_eq!(decode.tokens, vec![104]);
    assert_eq!(decode.sampling, Some(sampling));
    assert_eq!(decode.chat_sampling_metadata.as_deref(), Some(metadata));
    assert!(decode.activation.is_empty());
    assert!(decode.positions.is_empty());
}
