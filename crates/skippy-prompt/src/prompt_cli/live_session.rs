fn live_resident_prefix_matches(
    resident_tokens: &[i32],
    token_ids: &[i32],
    prefill_token_count: usize,
) -> bool {
    !resident_tokens.is_empty()
        && resident_tokens.len() <= prefill_token_count
        && token_ids
            .get(..resident_tokens.len())
            .is_some_and(|prefix| prefix == resident_tokens)
}

fn should_try_exact_prefix_restore(
    live_enabled: bool,
    prefill_start: usize,
    prefill_token_count: usize,
) -> bool {
    !live_enabled
        && prefill_start == 0
        && prefill_token_count >= PROMPT_EXACT_PREFIX_RESTORE_MIN_TOKENS
}

struct Stats {
    prompt_tokens: usize,
    prefill_tokens: usize,
    prefill_chunks: usize,
    generated_tokens: usize,
    tokenize_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    wallblock_ms: f64,
    first_time_to_token_ms: f64,
    tpot_ms: f64,
    tpot_after_first_ms: f64,
    reply_stats: StageReplyStats,
    speculative_stats: SpeculativeStats,
    session_reuse: PromptSessionReuseStats,
}

fn stage_chain_error_context(args: &BinaryReplArgs) -> String {
    let mut message = format!(
        "target stage chain did not respond within {}s; a remote stage may have exited or stopped forwarding",
        args.decode_timeout_secs.max(1)
    );
    if let Some(hint) = args.diagnostics_hint.as_deref() {
        message.push('\n');
        message.push_str(hint);
    }
    message
}

fn reset_live_prompt_runtime(
    live: &mut PromptLiveSession,
    args: &BinaryReplArgs,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    wire_session_id: u64,
) -> Result<()> {
    if live.stream.is_none() {
        live.stream = Some(
            connect_ready(&args.first_stage_addr, args.startup_timeout_secs)
                .context("first binary stage did not become ready")?,
        );
    }
    if let Some(stream) = live.stream.as_mut() {
        let io_timeout = Duration::from_secs(args.decode_timeout_secs.max(1));
        stream.set_read_timeout(Some(io_timeout)).ok();
        stream.set_write_timeout(Some(io_timeout)).ok();
        stop_prompt_stream(stream, wire_dtype, request_id, wire_session_id, args)
            .context("reset live prompt runtime session")?;
    }
    live.resident_tokens.clear();
    live.dirty = false;
    eprintln!("request {prompt_index}: live session runtime reset");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rematerialize_live_transcript(
    stream: &mut TcpStream,
    tokenizer: &StageModel,
    chat_template_model: Option<&StageModel>,
    args: &BinaryReplArgs,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    wire_session_id: u64,
    live_messages: &[ChatTemplateMessage],
    token_ids: &[i32],
    generated: &[i32],
    assistant_raw_text: &str,
    generation_reached_eog: bool,
) -> Result<()> {
    if !generation_reached_eog {
        return Ok(());
    }
    let mut closed_messages = live_messages.to_vec();
    closed_messages.push(ChatTemplateMessage::new("assistant", assistant_raw_text));
    let transcript_tokens =
        live_transcript_tokens(tokenizer, chat_template_model, args, &closed_messages)?;
    let mut runtime_tokens = token_ids.to_vec();
    runtime_tokens.extend_from_slice(generated);
    let common_prefix = common_token_prefix_len(&runtime_tokens, &transcript_tokens);
    if common_prefix < runtime_tokens.len() {
        send_trim_session(
            stream,
            wire_dtype,
            prompt_index,
            request_id,
            wire_session_id,
            common_prefix,
        )
        .with_context(|| stage_chain_error_context(args))?;
    }
    let suffix = &transcript_tokens[common_prefix..];
    for (chunk_index, chunk) in suffix.chunks(args.prefill_chunk_size).enumerate() {
        if chunk.is_empty() {
            continue;
        }
        let pos_start = common_prefix + chunk_index * args.prefill_chunk_size;
        send_prefill_chunk(
            stream,
            wire_dtype,
            ReplPrefillChunk {
                prompt_index,
                request_id,
                session_id: wire_session_id,
                pos_start,
                prefill_token_count: transcript_tokens.len(),
                tokens: chunk,
            },
        )
        .with_context(|| stage_chain_error_context(args))?;
    }
    Ok(())
}

fn live_transcript_tokens(
    tokenizer: &StageModel,
    chat_template_model: Option<&StageModel>,
    args: &BinaryReplArgs,
    messages: &[ChatTemplateMessage],
) -> Result<Vec<i32>> {
    let left = live_transcript_probe_tokens(tokenizer, chat_template_model, args, messages, "A")?;
    let right = live_transcript_probe_tokens(tokenizer, chat_template_model, args, messages, "B")?;
    let prefix_len = common_token_prefix_len(&left, &right);
    Ok(left[..prefix_len].to_vec())
}

fn live_transcript_probe_tokens(
    tokenizer: &StageModel,
    chat_template_model: Option<&StageModel>,
    args: &BinaryReplArgs,
    messages: &[ChatTemplateMessage],
    probe_user: &str,
) -> Result<Vec<i32>> {
    let mut probed_messages = messages.to_vec();
    probed_messages.push(ChatTemplateMessage::new("user", probe_user));
    let probed_prompt = format_messages_for_model_with_options(
        tokenizer,
        chat_template_model,
        &probed_messages,
        args,
        true,
    )?;
    tokenizer
        .tokenize(&probed_prompt, true)
        .with_context(|| format!("tokenize probed live prompt {probed_prompt:?}"))
}

fn stop_live_prompt_session(
    live: &mut PromptLiveSession,
    args: &BinaryReplArgs,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    wire_session_id: u64,
) -> Result<()> {
    let Some(stream) = live.stream.as_mut() else {
        return Ok(());
    };
    let prompt_index_bytes = prompt_index.to_le_bytes();
    let request_id = stable_wire_id(&[b"prompt-live-stop".as_slice(), &prompt_index_bytes]);
    stop_prompt_stream(stream, wire_dtype, request_id, wire_session_id, args)
        .context("stop live prompt session")?;
    live.resident_tokens.clear();
    live.stream.take();
    Ok(())
}

fn stop_prompt_stream(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    request_id: u64,
    wire_session_id: u64,
    args: &BinaryReplArgs,
) -> Result<()> {
    write_stage_message(
        &mut *stream,
        &StageWireMessage::stop_with_identity(wire_dtype, request_id, wire_session_id),
        wire_dtype,
    )
    .with_context(|| stage_chain_error_context(args))?;
    let stop_reply = recv_reply(stream).with_context(|| stage_chain_error_context(args))?;
    if stop_reply.kind != WireReplyKind::Ack {
        bail!("expected stop ACK, got {:?}", stop_reply.kind);
    }
    Ok(())
}

fn send_trim_session(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    token_count: usize,
) -> Result<SessionControlReply> {
    let started = Instant::now();
    let mut state = StageStateHeader::new(WireMessageKind::TrimSession, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind: WireMessageKind::TrimSession,
        pos_start: 0,
        token_count: i32::try_from(token_count).context("trim token count exceeds i32")?,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send trim session to {token_count} token(s)"))?;
    let reply = recv_reply(&mut *stream)
        .with_context(|| format!("receive trim session ACK for {token_count} token(s)"))?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected trim-session ACK, got {:?}", reply.kind);
    }
    Ok(SessionControlReply {
        stats: reply.stats,
        elapsed_ms: elapsed_ms(started),
    })
}

fn common_token_prefix_len(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn token_window(tokens: &[i32], center: usize) -> Vec<i32> {
    let start = center.saturating_sub(4);
    let end = (center + 8).min(tokens.len());
    tokens[start..end].to_vec()
}
