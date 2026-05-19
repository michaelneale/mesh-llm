mod activation;
mod codec;
mod types;

pub use activation::{
    activation_payload_multiplier_from_state_flags, activation_wire_bytes,
    activation_wire_bytes_with_state_flags, encode_f32_activation_payload,
    encode_f32_activation_payload_with_state_flags,
};
pub use codec::{
    read_stage_message, recv_ready, recv_reply, send_ready, send_reply_ack,
    send_reply_ack_with_stats, send_reply_predicted, send_reply_predicted_tokens_with_stats,
    send_reply_predicted_with_stats, write_stage_message,
};
pub use types::{
    activation_frame_flags_from_state_flags, activation_state_flags_from_frame_flags, state_flags,
    StageLogitBias, StageReply, StageReplyStats, StageSamplingConfig, StageStateHeader,
    StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind, WireStagePhase,
    ACTIVATION_FLAG_GEMMA3N_ALTUP, ACTIVATION_FLAG_RWKV7_V_FIRST, LLAMA_TOKEN_NULL,
    MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES,
    MAX_STAGE_DECODED_ACTIVATION_BYTES, MAX_STAGE_LOGIT_BIAS, MAX_STAGE_PREDICTED_TOKENS,
    MAX_STAGE_SIDEBAND_VALUES, MAX_STAGE_STATE_IMPORT_BYTES, READY_MAGIC,
    STAGE_LOGIT_BIAS_WIRE_BYTES, STAGE_SAMPLING_CONFIG_BASE_BYTES, STAGE_STATE_HEADER_BYTES,
    STAGE_STATE_VERSION, STAGE_WIRE_FIXED_HEADER_BYTES,
};

pub(crate) fn invalid_data(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

pub(crate) fn invalid_input(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_i32(bytes: &mut Vec<u8>, value: i32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_state_header(bytes: &mut Vec<u8>, state: StageStateHeader) {
        push_i32(bytes, state.version);
        push_i32(bytes, state.seq_id);
        push_i32(bytes, state.phase);
        push_i32(bytes, state.flags);
        push_i32(bytes, state.checkpoint_generation);
        push_i32(bytes, state.prompt_token_count);
        push_i32(bytes, state.decode_step);
        push_i32(bytes, state.current_token);
        push_i32(bytes, state.source_stage_index);
        push_i32(bytes, state.reserved);
    }

    fn stage_frame_prefix(
        kind: WireMessageKind,
        token_count: i32,
        token_sideband_count: i32,
        position_sideband_count: i32,
        state: StageStateHeader,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, kind as i32);
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, token_count);
        push_i32(&mut bytes, token_sideband_count);
        push_i32(&mut bytes, position_sideband_count);
        push_state_header(&mut bytes, state);
        push_u64(&mut bytes, 7);
        push_u64(&mut bytes, 11);
        bytes
    }

    fn assert_invalid_data<T: std::fmt::Debug>(result: std::io::Result<T>, expected: &str) {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), expected);
    }

    #[test]
    fn ready_round_trips() {
        let mut bytes = Vec::new();
        send_ready(&mut bytes).unwrap();
        recv_ready(Cursor::new(bytes)).unwrap();
    }

    #[test]
    fn reply_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted(&mut bytes, 42).unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        assert_eq!(reply.predicted_tokens, vec![42]);
    }

    #[test]
    fn token_vector_reply_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted_tokens_with_stats(&mut bytes, &[1, 2, 3], StageReplyStats::default())
            .unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedTokens);
        assert_eq!(reply.predicted, 1);
        assert_eq!(reply.predicted_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn reply_rejects_predicted_token_count_over_limit() {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, WireReplyKind::PredictedTokens as i32);
        push_i32(&mut bytes, 1);
        push_i32(
            &mut bytes,
            i32::try_from(MAX_STAGE_PREDICTED_TOKENS + 1).unwrap(),
        );

        assert_invalid_data(
            recv_reply(Cursor::new(bytes)),
            "predicted token count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_round_trips_f32() {
        let mut state =
            StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32);
        state.prompt_token_count = 1;
        state.decode_step = 0;
        state.current_token = 11;
        state.source_stage_index = 0;
        let activation = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 1,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 11,
            sampling: Some(StageSamplingConfig {
                flags: 1,
                seed: 42,
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                logit_bias: vec![StageLogitBias {
                    token_id: 123,
                    bias: -50.0,
                }],
                ..StageSamplingConfig::default()
            }),
            chat_sampling_metadata: None,
            tokens: vec![11],
            positions: Vec::new(),
            activation: activation.clone(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::DecodeEmbd);
        assert_eq!(decoded.tokens, vec![11]);
        assert_eq!(decoded.activation, activation);
        assert_eq!(decoded.state.source_stage_index, 0);
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.session_id, 11);
        assert_ne!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert_eq!(decoded.state.flags & state_flags::CHAT_SAMPLING_METADATA, 0);
        assert_eq!(decoded.chat_sampling_metadata, None);
        let sampling = decoded.sampling.expect("sampling extension round-tripped");
        assert_eq!(sampling.seed, 42);
        assert_eq!(sampling.top_k, 40);
        assert_eq!(sampling.logit_bias.len(), 1);
        assert_eq!(sampling.logit_bias[0].token_id, 123);
        assert_eq!(sampling.logit_bias[0].bias, -50.0);
    }

    #[test]
    fn generation_config_round_trips_sampling_metadata() {
        let message = StageWireMessage::configure_generation(
            WireActivationDType::F32,
            7,
            11,
            123,
            Some(StageSamplingConfig {
                flags: 1,
                seed: 42,
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                ..StageSamplingConfig::default()
            }),
            Some("{\"grammar\":\"root ::= \\\"x\\\"\"}".to_string()),
        );
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::ConfigureGeneration);
        assert_eq!(decoded.token_count, 0);
        assert_eq!(decoded.tokens, Vec::<i32>::new());
        assert_eq!(decoded.activation, Vec::<u8>::new());
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.session_id, 11);
        assert_eq!(decoded.state.prompt_token_count, 123);
        assert_ne!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert_ne!(decoded.state.flags & state_flags::CHAT_SAMPLING_METADATA, 0);
        assert_eq!(
            decoded.chat_sampling_metadata.as_deref(),
            Some("{\"grammar\":\"root ::= \\\"x\\\"\"}")
        );
        let sampling = decoded.sampling.expect("sampling extension round-tripped");
        assert_eq!(sampling.seed, 42);
        assert_eq!(sampling.top_k, 40);
    }

    #[test]
    fn stage_message_rejects_sampling_metadata_length_over_limit() {
        let mut state = StageStateHeader::new(
            WireMessageKind::ConfigureGeneration,
            WireActivationDType::F32,
        );
        state.flags |= state_flags::CHAT_SAMPLING_METADATA;
        let mut bytes = stage_frame_prefix(WireMessageKind::ConfigureGeneration, 0, 0, 0, state);
        push_u32(
            &mut bytes,
            u32::try_from(MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES + 1).unwrap(),
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "chat sampling metadata length exceeds maximum",
        );
    }

    #[test]
    fn driver_origin_message_round_trips_without_activation() {
        let mut state =
            StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32);
        state.prompt_token_count = 2;
        state.current_token = 22;
        state.source_stage_index = -1;
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 2,
            state,
            request_id: 13,
            session_id: 17,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![11, 22],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.tokens, vec![11, 22]);
        assert!(decoded.activation.is_empty());
        assert_eq!(decoded.state.source_stage_index, -1);
        assert_eq!(decoded.request_id, 13);
        assert_eq!(decoded.session_id, 17);
        assert_eq!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert!(decoded.sampling.is_none());
    }

    #[test]
    fn stage_message_rejects_token_sideband_count_over_limit() {
        let mut state =
            StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32);
        state.source_stage_index = -1;
        let bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            0,
            i32::try_from(MAX_STAGE_SIDEBAND_VALUES + 1).unwrap(),
            0,
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "token sideband count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_rejects_position_sideband_count_over_limit() {
        let mut state =
            StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32);
        state.source_stage_index = -1;
        let bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            0,
            0,
            i32::try_from(MAX_STAGE_SIDEBAND_VALUES + 1).unwrap(),
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "position sideband count exceeds maximum",
        );
    }

    #[test]
    fn prefill_wire_overhead_is_fixed_and_bounded() {
        let mut state =
            StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32);
        state.prompt_token_count = 128;
        state.current_token = 127;
        state.source_stage_index = -1;
        let tokens: Vec<i32> = (0..128).collect();
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: tokens.len() as i32,
            state,
            request_id: u64::MAX - 1,
            session_id: u64::MAX,
            sampling: None,
            chat_sampling_metadata: None,
            tokens,
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();

        assert_eq!(STAGE_STATE_HEADER_BYTES, 40);
        assert_eq!(STAGE_SAMPLING_CONFIG_BASE_BYTES, 40);
        assert_eq!(STAGE_WIRE_FIXED_HEADER_BYTES, 76);
        assert_eq!(
            bytes.len(),
            STAGE_WIRE_FIXED_HEADER_BYTES + message.tokens.len() * 4
        );
        const { assert!(STAGE_WIRE_FIXED_HEADER_BYTES <= 80) };
    }

    #[test]
    fn session_control_messages_are_fixed_header_only() {
        for kind in [
            WireMessageKind::CheckpointSession,
            WireMessageKind::RestoreSession,
            WireMessageKind::TrimSession,
        ] {
            let message = StageWireMessage {
                kind,
                pos_start: 0,
                token_count: 0,
                state: StageStateHeader::new(kind, WireActivationDType::F32),
                request_id: 23,
                session_id: 29,
                sampling: None,
                chat_sampling_metadata: None,
                tokens: Vec::new(),
                positions: Vec::new(),
                activation: Vec::new(),
                raw_bytes: Vec::new(),
            };
            let mut bytes = Vec::new();
            write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
            assert_eq!(bytes.len(), STAGE_WIRE_FIXED_HEADER_BYTES);
            let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
            assert_eq!(decoded.kind, kind);
            assert_eq!(decoded.request_id, 23);
            assert_eq!(decoded.session_id, 29);
            assert!(decoded.tokens.is_empty());
            assert!(decoded.activation.is_empty());
        }
    }

    #[test]
    fn state_import_message_round_trips_raw_bytes() {
        let state = StageStateHeader::new(WireMessageKind::StateImport, WireActivationDType::F32);
        let message = StageWireMessage {
            kind: WireMessageKind::StateImport,
            pos_start: 0,
            token_count: 4,
            state,
            request_id: 31,
            session_id: 37,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::StateImport);
        assert_eq!(decoded.raw_bytes, vec![1, 2, 3, 4]);
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn state_import_rejects_raw_byte_count_over_limit() {
        let state = StageStateHeader::new(WireMessageKind::StateImport, WireActivationDType::F32);
        let bytes = stage_frame_prefix(
            WireMessageKind::StateImport,
            i32::try_from(MAX_STAGE_STATE_IMPORT_BYTES + 1).unwrap(),
            0,
            0,
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "state import byte count exceeds maximum",
        );
    }

    #[test]
    fn state_import_writer_rejects_raw_byte_count_mismatch() {
        let state = StageStateHeader::new(WireMessageKind::StateImport, WireActivationDType::F32);
        let message = StageWireMessage {
            kind: WireMessageKind::StateImport,
            pos_start: 0,
            token_count: 8,
            state,
            request_id: 31,
            session_id: 37,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        let error = write_stage_message(&mut bytes, &message, WireActivationDType::F32)
            .expect_err("mismatched state import byte count should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), "state import raw byte count mismatch");
    }

    #[test]
    fn state_export_message_round_trips_without_payload() {
        let state = StageStateHeader::new(WireMessageKind::StateExport, WireActivationDType::F32);
        let message = StageWireMessage {
            kind: WireMessageKind::StateExport,
            pos_start: 0,
            token_count: 0,
            state,
            request_id: 41,
            session_id: 43,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::StateExport);
        assert!(decoded.raw_bytes.is_empty());
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn stage_message_rejects_activation_payload_over_limit() {
        let mut state =
            StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32);
        state.source_stage_index = 0;
        state.flags |= state_flags::GEMMA3N_ALTUP_SIDEBAND;
        let token_count = i32::try_from(MAX_STAGE_ACTIVATION_BYTES / 4 / 4 / 1024 + 1).unwrap();
        let bytes = stage_frame_prefix(WireMessageKind::DecodeEmbd, token_count, 0, 0, state);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 1024),
            "activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_rejects_f16_activation_when_decoded_payload_exceeds_limit() {
        let mut state =
            StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F16);
        state.source_stage_index = 0;
        let n_embd = 65_536;
        let token_count =
            i32::try_from(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 / n_embd as usize + 1).unwrap();
        let wire_bytes = activation_wire_bytes_with_state_flags(
            WireActivationDType::F16,
            token_count,
            n_embd,
            0,
        )
        .unwrap();
        assert!(wire_bytes <= MAX_STAGE_ACTIVATION_BYTES);
        let bytes = stage_frame_prefix(WireMessageKind::DecodeEmbd, token_count, 0, 0, state);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), n_embd),
            "decoded activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_rejects_q8_activation_when_decoded_payload_exceeds_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::Q8);
        state.source_stage_index = 0;
        let n_embd = 65_536;
        let token_count =
            i32::try_from(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 / n_embd as usize + 1).unwrap();
        let wire_bytes =
            activation_wire_bytes_with_state_flags(WireActivationDType::Q8, token_count, n_embd, 0)
                .unwrap();
        assert!(wire_bytes <= MAX_STAGE_ACTIVATION_BYTES);
        let bytes = stage_frame_prefix(WireMessageKind::DecodeEmbd, token_count, 0, 0, state);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), n_embd),
            "decoded activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_rejects_q8_sideband_activation_when_decoded_payload_exceeds_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::Q8);
        state.source_stage_index = 0;
        state.flags |= state_flags::GEMMA3N_ALTUP_SIDEBAND;
        let n_embd = 65_536;
        let token_count =
            i32::try_from(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 / 4 / n_embd as usize + 1)
                .unwrap();
        let wire_bytes = activation_wire_bytes_with_state_flags(
            WireActivationDType::Q8,
            token_count,
            n_embd,
            state.flags,
        )
        .unwrap();
        assert!(wire_bytes <= MAX_STAGE_ACTIVATION_BYTES);
        let bytes = stage_frame_prefix(WireMessageKind::DecodeEmbd, token_count, 0, 0, state);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), n_embd),
            "decoded activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn activation_encoding_rejects_decoded_payload_over_limit_before_compression() {
        let n_embd = 65_536;
        let token_count =
            i32::try_from(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 / n_embd as usize + 1).unwrap();

        assert_invalid_data(
            encode_f32_activation_payload(WireActivationDType::F16, token_count, n_embd, &[]),
            "decoded activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn q8_payload_decodes_to_f32_bytes() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0.5_f32.to_le_bytes());
        payload.extend_from_slice(&[2_u8, 254_u8]);
        let decoded = activation::decode_q8_to_f32_bytes(&payload, 1, 2).unwrap();
        let first = f32::from_le_bytes(decoded[0..4].try_into().unwrap());
        let second = f32::from_le_bytes(decoded[4..8].try_into().unwrap());
        assert_eq!(first, 1.0);
        assert_eq!(second, -1.0);
    }

    #[test]
    fn f32_payload_encodes_to_q8_and_decodes() {
        let mut input = Vec::new();
        input.extend_from_slice(&1.0_f32.to_le_bytes());
        input.extend_from_slice(&(-1.0_f32).to_le_bytes());
        let encoded = encode_f32_activation_payload(WireActivationDType::Q8, 1, 2, &input).unwrap();
        let decoded = activation::decode_q8_to_f32_bytes(&encoded, 1, 2).unwrap();
        let first = f32::from_le_bytes(decoded[0..4].try_into().unwrap());
        let second = f32::from_le_bytes(decoded[4..8].try_into().unwrap());
        assert!((first - 1.0).abs() < 0.01);
        assert!((second + 1.0).abs() < 0.01);
    }

    #[test]
    fn rwkv7_sideband_activation_round_trips() {
        let mut state =
            StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32);
        state.source_stage_index = 0;
        state.flags |= state_flags::RWKV7_V_FIRST_SIDEBAND;
        let mut activation = Vec::new();
        for value in [1.0_f32, 2.0, 3.0, 4.0] {
            activation.extend_from_slice(&value.to_le_bytes());
        }
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![42],
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.activation.len(), 16);
        assert_eq!(
            activation_frame_flags_from_state_flags(decoded.state.flags),
            ACTIVATION_FLAG_RWKV7_V_FIRST
        );
        assert_eq!(
            decoded.activation_f32_payload(2).unwrap(),
            message.activation
        );
    }

    #[test]
    fn gemma3n_altup_sideband_activation_round_trips() {
        let mut state =
            StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32);
        state.source_stage_index = 0;
        state.flags |= state_flags::GEMMA3N_ALTUP_SIDEBAND;
        let mut activation = Vec::new();
        for value in 0..8 {
            activation.extend_from_slice(&(value as f32).to_le_bytes());
        }
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![42],
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message, WireActivationDType::F32).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.activation.len(), 32);
        assert_eq!(
            activation_frame_flags_from_state_flags(decoded.state.flags),
            ACTIVATION_FLAG_GEMMA3N_ALTUP
        );
        assert_eq!(
            decoded.activation_f32_payload(2).unwrap(),
            message.activation
        );
    }
}
