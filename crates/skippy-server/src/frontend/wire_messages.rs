use super::*;

pub(super) struct DecodeMessageArgs {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) current: i32,
    pub(super) sampling: Option<WireSamplingConfig>,
}

pub(super) fn embedded_decode_message(
    wire_dtype: WireActivationDType,
    args: DecodeMessageArgs,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    state.decode_step = i32::try_from(args.decode_step)
        .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
    state.current_token = args.current;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
        token_count: 1,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: None,
        tokens: vec![args.current],
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) struct VerifySpanMessageArgs<'a> {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) tokens: &'a [i32],
    pub(super) checkpoint: bool,
}

pub(super) fn embedded_verify_message(
    wire_dtype: WireActivationDType,
    args: VerifySpanMessageArgs<'_>,
) -> OpenAiResult<StageWireMessage> {
    if args.tokens.is_empty() {
        return Err(OpenAiError::backend(
            "verify span requires at least one token",
        ));
    }
    let mut state = StageStateHeader::new(WireMessageKind::VerifySpan, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    state.decode_step = i32::try_from(args.decode_step)
        .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
    state.current_token = args.tokens[0];
    state.source_stage_index = -1;
    if !args.checkpoint {
        state.flags |= state_flags::SKIP_VERIFY_CHECKPOINT;
    }
    Ok(StageWireMessage {
        kind: WireMessageKind::VerifySpan,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("verify span position exceeds i32"))?,
        token_count: i32::try_from(args.tokens.len())
            .map_err(|_| OpenAiError::backend("verify span exceeds i32"))?,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: args.tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) fn embedded_session_control_message(
    wire_dtype: WireActivationDType,
    kind: WireMessageKind,
    request_id: u64,
    session_id: u64,
) -> StageWireMessage {
    StageWireMessage {
        kind,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(kind, wire_dtype),
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

pub(super) struct OpenAiPrefillChunk<'a> {
    pub(super) seq_id: usize,
    pub(super) pos_start: usize,
    pub(super) prefill_token_count: usize,
    pub(super) tokens: &'a [i32],
    pub(super) request_id: u64,
    pub(super) session_id: u64,
}

pub(super) fn send_prefill_chunk(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    chunk: OpenAiPrefillChunk<'_>,
) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id = i32::try_from(chunk.seq_id).context("prefill seq exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(chunk.prefill_token_count).context("prefill token count exceeds i32")?;
    state.current_token = *chunk.tokens.last().context("prefill chunk is empty")?;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: i32::try_from(chunk.pos_start).context("prefill chunk position exceeds i32")?,
        token_count: i32::try_from(chunk.tokens.len())
            .context("prefill token count exceeds i32")?,
        state,
        request_id: chunk.request_id,
        session_id: chunk.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: chunk.tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype).context("send prefill chunk")?;
    let reply = recv_reply(&mut *stream).context("receive prefill chunk ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

pub(super) fn generation_config_message(
    wire_dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    sampling: Option<WireSamplingConfig>,
    chat_sampling_metadata: Option<&str>,
) -> OpenAiResult<Option<StageWireMessage>> {
    let Some(metadata) = chat_sampling_metadata else {
        return Ok(None);
    };
    let prompt_token_count = i32::try_from(prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    Ok(Some(StageWireMessage::configure_generation(
        wire_dtype,
        request_id,
        session_id,
        prompt_token_count,
        sampling,
        Some(metadata.to_string()),
    )))
}

pub(super) fn embedded_prefill_message(
    wire_dtype: WireActivationDType,
    chunk: OpenAiPrefillChunk<'_>,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id =
        i32::try_from(chunk.seq_id).map_err(|_| OpenAiError::backend("prefill seq exceeds i32"))?;
    state.prompt_token_count = i32::try_from(chunk.prefill_token_count)
        .map_err(|_| OpenAiError::backend("prefill token count exceeds i32"))?;
    state.current_token = *chunk
        .tokens
        .last()
        .ok_or_else(|| OpenAiError::backend("prefill chunk is empty"))?;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: i32::try_from(chunk.pos_start)
            .map_err(|_| OpenAiError::backend("prefill chunk position exceeds i32"))?,
        token_count: i32::try_from(chunk.tokens.len())
            .map_err(|_| OpenAiError::backend("prefill token count exceeds i32"))?,
        state,
        request_id: chunk.request_id,
        session_id: chunk.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: chunk.tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) fn embedded_prefix_cache_message(
    kind: WireMessageKind,
    wire_dtype: WireActivationDType,
    tokens: &[i32],
    request_id: u64,
    session_id: u64,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(kind, wire_dtype);
    state.prompt_token_count = i32::try_from(tokens.len())
        .map_err(|_| OpenAiError::backend("prefix token count exceeds i32"))?;
    state.current_token = tokens.last().copied().unwrap_or(LLAMA_TOKEN_NULL);
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind,
        pos_start: 0,
        token_count: i32::try_from(tokens.len())
            .map_err(|_| OpenAiError::backend("prefix token count exceeds i32"))?,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) struct RestorePrefillDecodeMessageArgs<'a> {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) prefix_tokens: &'a [i32],
    pub(super) current: i32,
    pub(super) sampling: Option<WireSamplingConfig>,
    pub(super) chat_sampling_metadata: Option<&'a str>,
}

pub(super) fn embedded_restore_prefill_decode_message(
    wire_dtype: WireActivationDType,
    args: RestorePrefillDecodeMessageArgs<'_>,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::TryRestorePrefillDecode, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    state.decode_step = i32::try_from(args.decode_step)
        .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
    state.current_token = args.current;
    state.source_stage_index = -1;
    let mut tokens = Vec::with_capacity(args.prefix_tokens.len().saturating_add(1));
    tokens.extend_from_slice(args.prefix_tokens);
    tokens.push(args.current);
    Ok(StageWireMessage {
        kind: WireMessageKind::TryRestorePrefillDecode,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
        token_count: 1,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: args.chat_sampling_metadata.map(str::to_string),
        tokens,
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) fn openai_stage_mask(stage_index: u32) -> i64 {
    if stage_index < 63 {
        1_i64 << stage_index
    } else {
        0
    }
}

pub(super) struct MultimodalPrefillArgs {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) token_count: usize,
    pub(super) positions: Vec<i32>,
    pub(super) sampling: Option<WireSamplingConfig>,
    pub(super) final_chunk: bool,
}

pub(super) fn multimodal_prefill_message(
    wire_dtype: WireActivationDType,
    args: MultimodalPrefillArgs,
) -> OpenAiResult<StageWireMessage> {
    let kind = if args.final_chunk {
        WireMessageKind::PrefillFinalEmbd
    } else {
        WireMessageKind::PrefillEmbd
    };
    let mut state = StageStateHeader::new(kind, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("multimodal prefill token count exceeds i32"))?;
    state.current_token = LLAMA_TOKEN_NULL;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("multimodal prefill position exceeds i32"))?,
        token_count: i32::try_from(args.token_count)
            .map_err(|_| OpenAiError::backend("multimodal prefill token count exceeds i32"))?,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: args.positions,
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}
