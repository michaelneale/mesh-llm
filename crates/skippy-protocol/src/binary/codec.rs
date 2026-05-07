use std::io::{self, Read, Write};

use super::{
    activation::activation_wire_bytes_with_state_flags, invalid_data, invalid_input,
    StageLogitBias, StageReply, StageReplyStats, StageSamplingConfig, StageStateHeader,
    StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind, MAX_STAGE_LOGIT_BIAS,
    READY_MAGIC, STAGE_STATE_VERSION,
};

pub fn send_ready(mut writer: impl Write) -> io::Result<()> {
    write_i32(&mut writer, READY_MAGIC)
}

pub fn recv_ready(mut reader: impl Read) -> io::Result<()> {
    let magic = read_i32(&mut reader)?;
    if magic != READY_MAGIC {
        return Err(invalid_data("stage ready magic mismatch"));
    }
    Ok(())
}

pub fn send_reply_ack(mut writer: impl Write) -> io::Result<()> {
    send_reply_ack_with_stats(&mut writer, StageReplyStats::default())
}

pub fn send_reply_ack_with_stats(mut writer: impl Write, stats: StageReplyStats) -> io::Result<()> {
    write_i32(&mut writer, WireReplyKind::Ack as i32)?;
    write_i32(&mut writer, 0)?;
    write_i32(&mut writer, 0)?;
    write_reply_stats(&mut writer, stats)
}

pub fn send_reply_predicted(mut writer: impl Write, predicted: i32) -> io::Result<()> {
    send_reply_predicted_with_stats(&mut writer, predicted, StageReplyStats::default())
}

pub fn send_reply_predicted_with_stats(
    mut writer: impl Write,
    predicted: i32,
    stats: StageReplyStats,
) -> io::Result<()> {
    write_i32(&mut writer, WireReplyKind::PredictedToken as i32)?;
    write_i32(&mut writer, predicted)?;
    write_i32(&mut writer, 1)?;
    write_i32(&mut writer, predicted)?;
    write_reply_stats(&mut writer, stats)
}

pub fn send_reply_predicted_tokens_with_stats(
    mut writer: impl Write,
    predicted_tokens: &[i32],
    stats: StageReplyStats,
) -> io::Result<()> {
    let predicted = predicted_tokens.first().copied().unwrap_or(0);
    write_i32(&mut writer, WireReplyKind::PredictedTokens as i32)?;
    write_i32(&mut writer, predicted)?;
    write_i32(
        &mut writer,
        i32::try_from(predicted_tokens.len())
            .map_err(|_| invalid_input("too many predicted tokens"))?,
    )?;
    for token in predicted_tokens {
        write_i32(&mut writer, *token)?;
    }
    write_reply_stats(&mut writer, stats)
}

pub fn recv_reply(mut reader: impl Read) -> io::Result<StageReply> {
    let kind = WireReplyKind::try_from(read_i32(&mut reader)?)?;
    let predicted = read_i32(&mut reader)?;
    let predicted_count = read_i32(&mut reader)?;
    if predicted_count < 0 {
        return Err(invalid_data("negative predicted token count"));
    }
    let mut predicted_tokens = Vec::with_capacity(predicted_count as usize);
    for _ in 0..predicted_count {
        predicted_tokens.push(read_i32(&mut reader)?);
    }
    let stats = read_reply_stats(&mut reader)?;
    Ok(StageReply {
        kind,
        predicted,
        predicted_tokens,
        stats,
    })
}

pub fn write_stage_message(
    mut writer: impl Write,
    message: &StageWireMessage,
    dtype: WireActivationDType,
) -> io::Result<()> {
    // Wire v4 fixed prefix, little-endian:
    // kind, pos_start, token_count, token_sideband_count, position_sideband_count (5 x i32);
    // StageStateHeader (10 x i32); request_id, session_id (2 x u64);
    // optional StageSamplingConfig follows when state_flags::SAMPLING is set.
    // Token sideband, raw StateImport bytes, or activation bytes follow this
    // prefix, so prefill overhead stays independent of ID string length.
    write_i32(&mut writer, message.kind as i32)?;
    write_i32(&mut writer, message.pos_start)?;
    write_i32(&mut writer, message.token_count)?;
    write_i32(
        &mut writer,
        i32::try_from(message.tokens.len()).map_err(|_| invalid_input("too many tokens"))?,
    )?;
    write_i32(
        &mut writer,
        i32::try_from(message.positions.len())
            .map_err(|_| invalid_input("too many position sideband values"))?,
    )?;

    let mut state = message.state;
    state.reserved = dtype as i32;
    if message.sampling.is_some() {
        state.flags |= super::state_flags::SAMPLING;
    } else {
        state.flags &= !super::state_flags::SAMPLING;
    }
    if message.chat_sampling_metadata.is_some() {
        state.flags |= super::state_flags::CHAT_SAMPLING_METADATA;
    } else {
        state.flags &= !super::state_flags::CHAT_SAMPLING_METADATA;
    }
    write_state_header(&mut writer, state)?;
    write_u64(&mut writer, message.request_id)?;
    write_u64(&mut writer, message.session_id)?;
    if let Some(sampling) = message.sampling.as_ref() {
        write_sampling_config(&mut writer, sampling)?;
    }
    if let Some(metadata) = message.chat_sampling_metadata.as_ref() {
        let bytes = metadata.as_bytes();
        write_u32(
            &mut writer,
            u32::try_from(bytes.len())
                .map_err(|_| invalid_input("chat sampling metadata is too large"))?,
        )?;
        writer.write_all(bytes)?;
    }

    if message.kind == WireMessageKind::StateImport {
        writer.write_all(&message.raw_bytes)?;
        return Ok(());
    }
    for token in &message.tokens {
        write_i32(&mut writer, *token)?;
    }
    for position in &message.positions {
        write_i32(&mut writer, *position)?;
    }
    writer.write_all(&message.activation)?;
    Ok(())
}

pub fn read_stage_message(mut reader: impl Read, n_embd: i32) -> io::Result<StageWireMessage> {
    let kind = WireMessageKind::try_from(read_i32(&mut reader)?)?;
    let pos_start = read_i32(&mut reader)?;
    let token_count = read_i32(&mut reader)?;
    let token_sideband_count = read_i32(&mut reader)?;
    let position_sideband_count = read_i32(&mut reader)?;
    let state = read_state_header(&mut reader)?;
    if state.version != STAGE_STATE_VERSION {
        return Err(invalid_data("unsupported stage state version"));
    }
    let request_id = read_u64(&mut reader)?;
    let session_id = read_u64(&mut reader)?;
    let sampling = if (state.flags & super::state_flags::SAMPLING) != 0 {
        Some(read_sampling_config(&mut reader)?)
    } else {
        None
    };
    let chat_sampling_metadata = if (state.flags & super::state_flags::CHAT_SAMPLING_METADATA) != 0
    {
        let len = usize::try_from(read_u32(&mut reader)?)
            .map_err(|_| invalid_data("chat sampling metadata length overflows usize"))?;
        let mut bytes = vec![0_u8; len];
        reader.read_exact(&mut bytes)?;
        Some(
            String::from_utf8(bytes)
                .map_err(|_| invalid_data("chat sampling metadata is not UTF-8"))?,
        )
    } else {
        None
    };
    let dtype = state.dtype()?;
    if kind == WireMessageKind::Stop {
        return Ok(StageWireMessage {
            kind,
            pos_start,
            token_count,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        });
    }
    if token_count < 0 || token_sideband_count < 0 || position_sideband_count < 0 {
        return Err(invalid_data("negative wire count"));
    }
    if kind == WireMessageKind::StateImport {
        let mut raw_bytes = vec![0; token_count as usize];
        reader.read_exact(&mut raw_bytes)?;
        return Ok(StageWireMessage {
            kind,
            pos_start,
            token_count,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes,
        });
    }

    let mut tokens = Vec::with_capacity(token_sideband_count as usize);
    for _ in 0..token_sideband_count {
        tokens.push(read_i32(&mut reader)?);
    }
    let mut positions = Vec::with_capacity(position_sideband_count as usize);
    for _ in 0..position_sideband_count {
        positions.push(read_i32(&mut reader)?);
    }
    let activation_bytes =
        if state.source_stage_index < 0 || kind.is_activationless_prefix_cache_control() {
            0
        } else {
            activation_wire_bytes_with_state_flags(dtype, token_count, n_embd, state.flags)?
        };
    let mut activation = vec![0; activation_bytes];
    if activation_bytes > 0 {
        reader.read_exact(&mut activation)?;
    }
    Ok(StageWireMessage {
        kind,
        pos_start,
        token_count,
        state,
        request_id,
        session_id,
        sampling,
        chat_sampling_metadata,
        tokens,
        positions,
        activation,
        raw_bytes: Vec::new(),
    })
}

fn write_state_header(mut writer: impl Write, state: StageStateHeader) -> io::Result<()> {
    write_i32(&mut writer, state.version)?;
    write_i32(&mut writer, state.seq_id)?;
    write_i32(&mut writer, state.phase)?;
    write_i32(&mut writer, state.flags)?;
    write_i32(&mut writer, state.checkpoint_generation)?;
    write_i32(&mut writer, state.prompt_token_count)?;
    write_i32(&mut writer, state.decode_step)?;
    write_i32(&mut writer, state.current_token)?;
    write_i32(&mut writer, state.source_stage_index)?;
    write_i32(&mut writer, state.reserved)
}

fn read_state_header(mut reader: impl Read) -> io::Result<StageStateHeader> {
    Ok(StageStateHeader {
        version: read_i32(&mut reader)?,
        seq_id: read_i32(&mut reader)?,
        phase: read_i32(&mut reader)?,
        flags: read_i32(&mut reader)?,
        checkpoint_generation: read_i32(&mut reader)?,
        prompt_token_count: read_i32(&mut reader)?,
        decode_step: read_i32(&mut reader)?,
        current_token: read_i32(&mut reader)?,
        source_stage_index: read_i32(&mut reader)?,
        reserved: read_i32(&mut reader)?,
    })
}

fn write_sampling_config(mut writer: impl Write, sampling: &StageSamplingConfig) -> io::Result<()> {
    write_u32(&mut writer, sampling.flags)?;
    write_u32(&mut writer, sampling.seed)?;
    write_f32(&mut writer, sampling.temperature)?;
    write_f32(&mut writer, sampling.top_p)?;
    write_i32(&mut writer, sampling.top_k)?;
    write_f32(&mut writer, sampling.presence_penalty)?;
    write_f32(&mut writer, sampling.frequency_penalty)?;
    write_f32(&mut writer, sampling.repeat_penalty)?;
    write_i32(&mut writer, sampling.penalty_last_n)?;
    let count = sampling.logit_bias.len().min(MAX_STAGE_LOGIT_BIAS);
    write_u32(&mut writer, count as u32)?;
    for bias in sampling.logit_bias.iter().take(count) {
        write_i32(&mut writer, bias.token_id)?;
        write_f32(&mut writer, bias.bias)?;
    }
    Ok(())
}

fn read_sampling_config(mut reader: impl Read) -> io::Result<StageSamplingConfig> {
    let mut sampling = StageSamplingConfig {
        flags: read_u32(&mut reader)?,
        seed: read_u32(&mut reader)?,
        temperature: read_f32(&mut reader)?,
        top_p: read_f32(&mut reader)?,
        top_k: read_i32(&mut reader)?,
        presence_penalty: read_f32(&mut reader)?,
        frequency_penalty: read_f32(&mut reader)?,
        repeat_penalty: read_f32(&mut reader)?,
        penalty_last_n: read_i32(&mut reader)?,
        logit_bias: Vec::new(),
    };
    let logit_bias_count = usize::try_from(read_u32(&mut reader)?)
        .map_err(|_| invalid_data("logit bias count overflows usize"))?;
    if logit_bias_count > MAX_STAGE_LOGIT_BIAS {
        return Err(invalid_data("logit bias count exceeds maximum"));
    }
    sampling.logit_bias.reserve(logit_bias_count);
    for _ in 0..logit_bias_count {
        sampling.logit_bias.push(StageLogitBias {
            token_id: read_i32(&mut reader)?,
            bias: read_f32(&mut reader)?,
        });
    }
    Ok(sampling)
}

fn write_reply_stats(mut writer: impl Write, stats: StageReplyStats) -> io::Result<()> {
    write_i64(&mut writer, stats.kv_lookup_hits)?;
    write_i64(&mut writer, stats.kv_lookup_misses)?;
    write_i64(&mut writer, stats.kv_lookup_errors)?;
    write_i64(&mut writer, stats.kv_imported_pages)?;
    write_i64(&mut writer, stats.kv_imported_tokens)?;
    write_i64(&mut writer, stats.kv_recorded_pages)?;
    write_i64(&mut writer, stats.kv_recorded_bytes)?;
    write_i64(&mut writer, stats.kv_hit_stage_mask)?;
    write_i64(&mut writer, stats.kv_record_stage_mask)?;
    write_i64(&mut writer, stats.checkpoint_flush_us)?;
    write_i64(&mut writer, stats.checkpoint_prefill_drain_us)?;
    write_i64(&mut writer, stats.checkpoint_local_us)?;
    write_i64(&mut writer, stats.checkpoint_downstream_write_us)?;
    write_i64(&mut writer, stats.checkpoint_downstream_wait_us)?;
    write_i64(&mut writer, stats.checkpoint_total_us)?;
    write_i64(&mut writer, stats.checkpoint_prefill_drained_replies)?;
    write_i64(&mut writer, stats.restore_flush_us)?;
    write_i64(&mut writer, stats.restore_prefill_drain_us)?;
    write_i64(&mut writer, stats.restore_local_us)?;
    write_i64(&mut writer, stats.restore_downstream_write_us)?;
    write_i64(&mut writer, stats.restore_downstream_wait_us)?;
    write_i64(&mut writer, stats.restore_total_us)?;
    write_i64(&mut writer, stats.restore_prefill_drained_replies)?;
    write_i64(&mut writer, stats.verify_span_compute_us)?;
    write_i64(&mut writer, stats.verify_span_forward_write_us)?;
    write_i64(&mut writer, stats.verify_span_downstream_wait_us)?;
    write_i64(&mut writer, stats.verify_span_total_us)?;
    write_i64(&mut writer, stats.verify_span_stage_count)?;
    write_i64(&mut writer, stats.verify_span_request_count)?;
    write_i64(&mut writer, stats.verify_span_token_count)?;
    write_i64(&mut writer, stats.verify_span_max_tokens)?;
    write_i64(&mut writer, stats.verify_span_checkpointed_requests)?;
    write_i64(&mut writer, stats.verify_span_skip_checkpoint_requests)
}

fn read_reply_stats(mut reader: impl Read) -> io::Result<StageReplyStats> {
    Ok(StageReplyStats {
        kv_lookup_hits: read_i64(&mut reader)?,
        kv_lookup_misses: read_i64(&mut reader)?,
        kv_lookup_errors: read_i64(&mut reader)?,
        kv_imported_pages: read_i64(&mut reader)?,
        kv_imported_tokens: read_i64(&mut reader)?,
        kv_recorded_pages: read_i64(&mut reader)?,
        kv_recorded_bytes: read_i64(&mut reader)?,
        kv_hit_stage_mask: read_i64(&mut reader)?,
        kv_record_stage_mask: read_i64(&mut reader)?,
        checkpoint_flush_us: read_i64(&mut reader)?,
        checkpoint_prefill_drain_us: read_i64(&mut reader)?,
        checkpoint_local_us: read_i64(&mut reader)?,
        checkpoint_downstream_write_us: read_i64(&mut reader)?,
        checkpoint_downstream_wait_us: read_i64(&mut reader)?,
        checkpoint_total_us: read_i64(&mut reader)?,
        checkpoint_prefill_drained_replies: read_i64(&mut reader)?,
        restore_flush_us: read_i64(&mut reader)?,
        restore_prefill_drain_us: read_i64(&mut reader)?,
        restore_local_us: read_i64(&mut reader)?,
        restore_downstream_write_us: read_i64(&mut reader)?,
        restore_downstream_wait_us: read_i64(&mut reader)?,
        restore_total_us: read_i64(&mut reader)?,
        restore_prefill_drained_replies: read_i64(&mut reader)?,
        verify_span_compute_us: read_i64(&mut reader)?,
        verify_span_forward_write_us: read_i64(&mut reader)?,
        verify_span_downstream_wait_us: read_i64(&mut reader)?,
        verify_span_total_us: read_i64(&mut reader)?,
        verify_span_stage_count: read_i64(&mut reader)?,
        verify_span_request_count: read_i64(&mut reader)?,
        verify_span_token_count: read_i64(&mut reader)?,
        verify_span_max_tokens: read_i64(&mut reader)?,
        verify_span_checkpointed_requests: read_i64(&mut reader)?,
        verify_span_skip_checkpoint_requests: read_i64(&mut reader)?,
    })
}

fn read_i32(mut reader: impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn write_i32(mut writer: impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(mut reader: impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn write_u32(mut writer: impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_f32(mut reader: impl Read) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn write_f32(mut writer: impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i64(mut reader: impl Read) -> io::Result<i64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn write_i64(mut writer: impl Write, value: i64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u64(mut reader: impl Read) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_u64(mut writer: impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
