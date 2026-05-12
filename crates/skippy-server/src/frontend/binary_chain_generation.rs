use super::*;

impl StageOpenAiBackend {
    pub(super) fn generate_binary_chain_tokens(
        &self,
        request: BinaryChainGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        let wire_sampling = wire_sampling_config(request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let connect_timer = PhaseTimer::start();
        let mut stream =
            connect_endpoint_ready(request.first_stage_addr, request.startup_timeout_secs)
                .map_err(openai_backend_error)?;
        let mut connect_attrs = self.openai_attrs(request.ids);
        connect_attrs.insert(
            "llama_stage.first_stage_addr".to_string(),
            json!(request.first_stage_addr),
        );
        self.emit_openai_phase(
            "stage.openai_downstream_connect",
            connect_timer,
            connect_attrs,
        );
        let result = (|| {
            let prefill_token_count = request.prompt_token_ids.len().saturating_sub(1);
            let prefill_timer = PhaseTimer::start();
            let mut prefill_chunks = 0usize;
            let mut prefill_min_chunk_size = usize::MAX;
            let mut prefill_max_chunk_size = 0usize;
            let mut prefill_planner = request.prefill_chunk_policy.planner();
            if prefill_token_count > 0 {
                let prefill_tokens = &request.prompt_token_ids[..prefill_token_count];
                let mut pos_start = 0usize;
                let mut chunk_index = 0usize;
                while pos_start < prefill_tokens.len() {
                    if request
                        .cancellation
                        .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                    {
                        return Ok(());
                    }
                    let chunk_size = prefill_planner.chunk_size_for(chunk_index);
                    let end = pos_start
                        .saturating_add(chunk_size)
                        .min(prefill_tokens.len());
                    let chunk = &prefill_tokens[pos_start..end];
                    prefill_min_chunk_size = prefill_min_chunk_size.min(chunk.len());
                    prefill_max_chunk_size = prefill_max_chunk_size.max(chunk.len());
                    send_prefill_chunk(
                        &mut stream,
                        request.wire_dtype,
                        OpenAiPrefillChunk {
                            seq_id: chunk_index,
                            pos_start,
                            prefill_token_count,
                            tokens: chunk,
                            request_id,
                            session_id,
                        },
                    )
                    .map_err(openai_backend_error)?;
                    prefill_planner.advance_without_observation();
                    prefill_chunks += 1;
                    pos_start = end;
                    chunk_index += 1;
                }
            }
            let mut prefill_attrs = self.openai_attrs(request.ids);
            prefill_attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill_token_count),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunks),
            );
            attrs_insert_prefill_chunk_policy(
                &mut prefill_attrs,
                request.prefill_chunk_policy,
                prefill_min_chunk_size,
                prefill_max_chunk_size,
            );
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, prefill_attrs);

            if let Some(message) = generation_config_message(
                request.wire_dtype,
                request_id,
                session_id,
                request.prompt_token_ids.len(),
                wire_sampling.clone(),
                request.chat_sampling_metadata,
            )? {
                write_stage_message(&mut stream, &message, request.wire_dtype)
                    .map_err(openai_io_error)?;
                let reply = recv_reply(&mut stream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::Ack {
                    return Err(OpenAiError::backend(format!(
                        "expected generation config ACK, got {:?}",
                        reply.kind
                    )));
                }
            }

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            for decode_step in 0..request.max_tokens {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let mut state =
                    StageStateHeader::new(WireMessageKind::DecodeEmbd, request.wire_dtype);
                state.seq_id = 0;
                state.prompt_token_count = i32::try_from(request.prompt_token_ids.len())
                    .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
                state.decode_step = i32::try_from(decode_step)
                    .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
                state.current_token = current;
                state.source_stage_index = -1;
                let message = StageWireMessage {
                    kind: WireMessageKind::DecodeEmbd,
                    pos_start: i32::try_from(prefill_token_count + decode_step as usize)
                        .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
                    token_count: 1,
                    state,
                    request_id,
                    session_id,
                    sampling: wire_sampling.clone(),
                    chat_sampling_metadata: None,
                    tokens: vec![current],
                    positions: Vec::new(),
                    activation: Vec::new(),
                    raw_bytes: Vec::new(),
                };
                write_stage_message(&mut stream, &message, request.wire_dtype)
                    .map_err(openai_io_error)?;
                let reply = recv_reply(&mut stream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::PredictedToken {
                    return Err(OpenAiError::backend(format!(
                        "expected predicted-token reply, got {:?}",
                        reply.kind
                    )));
                }
                current = reply.predicted;
                decoded_tokens += 1;
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            let mut decode_attrs = self.openai_attrs(request.ids);
            decode_attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            self.emit_openai_phase("stage.openai_decode", decode_timer, decode_attrs);
            Ok(())
        })();
        let stop_result = write_stage_message(
            &mut stream,
            &StageWireMessage::stop_with_identity(request.wire_dtype, request_id, session_id),
            request.wire_dtype,
        )
        .and_then(|_| recv_reply(&mut stream).map(|reply| reply.kind))
        .and_then(|kind| {
            if kind == WireReplyKind::Ack {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected stop ACK, got {kind:?}"),
                ))
            }
        });
        if result.is_ok() {
            stop_result.map_err(openai_io_error)?;
        }
        result?;
        Ok(GenerationCacheStats::default())
    }
}
