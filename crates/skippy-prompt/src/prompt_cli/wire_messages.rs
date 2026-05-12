struct DecodeStepReply {
    predicted: i32,
    stats: StageReplyStats,
    elapsed_ms: f64,
}

struct VerifySpanReply {
    predicted_tokens: Vec<i32>,
    stats: StageReplyStats,
    write_ms: f64,
    wait_ms: f64,
    elapsed_ms: f64,
}

struct SessionControlReply {
    stats: StageReplyStats,
    elapsed_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn send_decode_step(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    prefill_token_count: usize,
    decode_index: usize,
    current: i32,
) -> Result<DecodeStepReply> {
    let decode_started = Instant::now();
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?;
    state.decode_step = i32::try_from(decode_index).context("decode step exceeds i32")?;
    state.current_token = current;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: i32::try_from(prefill_token_count + decode_index)
            .context("decode position exceeds i32")?,
        token_count: 1,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![current],
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send decode step {decode_index}"))?;
    let reply = recv_reply(&mut *stream)
        .with_context(|| format!("receive decode step {decode_index} reply"))?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    Ok(DecodeStepReply {
        predicted: reply.predicted,
        stats: reply.stats,
        elapsed_ms: elapsed_ms(decode_started),
    })
}

#[allow(clippy::too_many_arguments)]
fn send_verify_span(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    pos_start: usize,
    decode_index: usize,
    tokens: &[i32],
    checkpoint: bool,
) -> Result<VerifySpanReply> {
    if tokens.is_empty() {
        bail!("verify span requires at least one token");
    }
    let verify_started = Instant::now();
    let mut state = StageStateHeader::new(WireMessageKind::VerifySpan, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?;
    state.decode_step = i32::try_from(decode_index).context("decode step exceeds i32")?;
    state.current_token = tokens[0];
    state.source_stage_index = -1;
    if !checkpoint {
        state.flags |= state_flags::SKIP_VERIFY_CHECKPOINT;
    }
    let message = StageWireMessage {
        kind: WireMessageKind::VerifySpan,
        pos_start: i32::try_from(pos_start).context("verify span position exceeds i32")?,
        token_count: i32::try_from(tokens.len()).context("verify span exceeds i32")?,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    let write_started = Instant::now();
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send verify span at decode step {decode_index}"))?;
    let write_ms = elapsed_ms(write_started);
    let wait_started = Instant::now();
    let reply = recv_reply(&mut *stream)
        .with_context(|| format!("receive verify span {decode_index} reply"))?;
    let wait_ms = elapsed_ms(wait_started);
    if reply.kind != WireReplyKind::PredictedTokens {
        bail!("expected predicted-tokens reply, got {:?}", reply.kind);
    }
    Ok(VerifySpanReply {
        predicted_tokens: reply.predicted_tokens,
        stats: reply.stats,
        write_ms,
        wait_ms,
        elapsed_ms: elapsed_ms(verify_started),
    })
}

fn print_stats(stats: Stats) {
    let prompt_tps = if stats.prefill_ms > 0.0 {
        stats.prefill_tokens as f64 / (stats.prefill_ms / 1000.0)
    } else {
        0.0
    };
    let decode_tps = if stats.tpot_ms > 0.0 {
        1000.0 / stats.tpot_ms
    } else {
        0.0
    };

    let chunk_label = if stats.prefill_chunks == 1 {
        "chunk"
    } else {
        "chunks"
    };
    eprintln!("stats:");
    eprintln!(
        "  tokens   prompt={} prefill={} ({} {}) generated={}",
        stats.prompt_tokens,
        stats.prefill_tokens,
        stats.prefill_chunks,
        chunk_label,
        stats.generated_tokens
    );
    eprintln!(
        "  time     tokenize={:.2}ms prefill={:.2}ms decode={:.2}ms wallblock={:.2}ms",
        stats.tokenize_ms, stats.prefill_ms, stats.decode_ms, stats.wallblock_ms
    );
    eprintln!(
        "  speed    prefill={:.2} tok/s decode={:.2} tok/s tpot={:.2}ms steady_tpot={:.2}ms",
        prompt_tps, decode_tps, stats.tpot_ms, stats.tpot_after_first_ms
    );
    eprintln!("  latency  ttft={:.2}ms", stats.first_time_to_token_ms);
    if stats.session_reuse.outcome != "disabled" {
        eprintln!(
            "  reuse    live_session={} reused_tokens={} appended_prefill_tokens={} resident_tokens={}->{}",
            stats.session_reuse.outcome,
            stats.session_reuse.reused_tokens,
            stats.session_reuse.appended_prefill_tokens,
            stats.session_reuse.resident_tokens_before,
            stats.session_reuse.resident_tokens_after
        );
    }
    if !stats.reply_stats.is_empty() {
        let lookups = stats.reply_stats.kv_lookup_hits + stats.reply_stats.kv_lookup_misses;
        let hit_rate = if lookups > 0 {
            100.0 * stats.reply_stats.kv_lookup_hits as f64 / lookups as f64
        } else {
            0.0
        };
        let hit_stage_count = stage_mask_count(stats.reply_stats.kv_hit_stage_mask);
        let cached_prompt_tokens_est = (stats.reply_stats.kv_imported_tokens.max(0) as u64)
            .checked_div(hit_stage_count)
            .unwrap_or(0);
        let exact_prefix_reuse = if stats.reply_stats.kv_lookup_hits > 0 {
            "hit"
        } else if stats.reply_stats.kv_lookup_misses > 0 {
            "miss"
        } else if stats.reply_stats.kv_lookup_errors > 0 {
            "error"
        } else {
            "not_checked"
        };
        let record_status = if stats.reply_stats.kv_recorded_pages > 0 {
            "recorded"
        } else {
            "not_recorded"
        };
        eprintln!(
            "  reuse    exact_prefix={} cached_prompt_tokens_est={} stage_imported_tokens={} record={}",
            exact_prefix_reuse,
            cached_prompt_tokens_est,
            stats.reply_stats.kv_imported_tokens.max(0),
            record_status
        );
        eprintln!(
            "  kv       lookup hit={} miss={} hit_rate={:.1}% error={} imported_pages={} imported_tokens={}",
            stats.reply_stats.kv_lookup_hits,
            stats.reply_stats.kv_lookup_misses,
            hit_rate,
            stats.reply_stats.kv_lookup_errors,
            stats.reply_stats.kv_imported_pages,
            stats.reply_stats.kv_imported_tokens
        );
        eprintln!(
            "  kv       recorded_pages={} recorded_bytes={} hit_stages={} record_stages={}",
            stats.reply_stats.kv_recorded_pages,
            format_bytes(stats.reply_stats.kv_recorded_bytes.max(0) as u64),
            format_stage_mask(stats.reply_stats.kv_hit_stage_mask),
            format_stage_mask(stats.reply_stats.kv_record_stage_mask)
        );
        if stats.reply_stats.restore_total_us > 0 {
            eprintln!(
                "  kv       restore_ms total={:.2} flush={:.2} prefill_drain={:.2} local={:.2} downstream_write={:.2} downstream_wait={:.2} drained_replies={}",
                us_to_ms(stats.reply_stats.restore_total_us),
                us_to_ms(stats.reply_stats.restore_flush_us),
                us_to_ms(stats.reply_stats.restore_prefill_drain_us),
                us_to_ms(stats.reply_stats.restore_local_us),
                us_to_ms(stats.reply_stats.restore_downstream_write_us),
                us_to_ms(stats.reply_stats.restore_downstream_wait_us),
                stats.reply_stats.restore_prefill_drained_replies
            );
        }
        if stats.reply_stats.checkpoint_total_us > 0 {
            eprintln!(
                "  kv       checkpoint_ms total={:.2} flush={:.2} prefill_drain={:.2} local={:.2} downstream_write={:.2} downstream_wait={:.2} drained_replies={}",
                us_to_ms(stats.reply_stats.checkpoint_total_us),
                us_to_ms(stats.reply_stats.checkpoint_flush_us),
                us_to_ms(stats.reply_stats.checkpoint_prefill_drain_us),
                us_to_ms(stats.reply_stats.checkpoint_local_us),
                us_to_ms(stats.reply_stats.checkpoint_downstream_write_us),
                us_to_ms(stats.reply_stats.checkpoint_downstream_wait_us),
                stats.reply_stats.checkpoint_prefill_drained_replies
            );
        }
    } else {
        eprintln!("  reuse    exact_prefix=not_reported record=not_reported");
        eprintln!("  kv       no lookup/record events reported");
    }
    if stats.speculative_stats.windows > 0 {
        let acceptance = if stats.speculative_stats.draft_tokens == 0 {
            0.0
        } else {
            100.0 * stats.speculative_stats.accepted_tokens as f64
                / stats.speculative_stats.draft_tokens as f64
        };
        eprintln!(
            "  spec     windows={} proposed={} accepted={} rejected={} accept_rate={:.1}%",
            stats.speculative_stats.windows,
            stats.speculative_stats.draft_tokens,
            stats.speculative_stats.accepted_tokens,
            stats.speculative_stats.rejected_tokens,
            acceptance
        );
        let avg_reject_pos = if stats.speculative_stats.rejected_windows == 0 {
            0.0
        } else {
            stats.speculative_stats.first_reject_position_sum as f64
                / stats.speculative_stats.rejected_windows as f64
        };
        eprintln!(
            "  spec     full_accept_windows={} accepted_stop_windows={} rejected_windows={} early_reject_windows={} tail_reject_windows={} early_reject_stop_windows={} repair_required_windows={} avg_reject_pos={:.2}",
            stats.speculative_stats.full_accept_windows,
            stats.speculative_stats.accepted_stop_windows,
            stats.speculative_stats.rejected_windows,
            stats.speculative_stats.early_reject_windows,
            stats.speculative_stats.tail_reject_windows,
            stats.speculative_stats.early_reject_stop_windows,
            stats.speculative_stats.repair_required_windows,
            avg_reject_pos
        );
        if stats.speculative_stats.adaptive_window_max > 0 {
            let avg_window = stats.speculative_stats.adaptive_window_sum as f64
                / stats.speculative_stats.windows.max(1) as f64;
            eprintln!(
                "  spec     window_policy={} start={} final={} max={} avg={:.2} min={} max_seen={} grows={} shrinks={}",
                if stats.speculative_stats.adaptive_window_enabled {
                    "adaptive"
                } else {
                    "fixed"
                },
                stats.speculative_stats.adaptive_window_start,
                stats.speculative_stats.adaptive_window_final,
                stats.speculative_stats.adaptive_window_max,
                avg_window,
                stats.speculative_stats.adaptive_window_min,
                stats.speculative_stats.adaptive_window_max_seen,
                stats.speculative_stats.adaptive_window_grows,
                stats.speculative_stats.adaptive_window_shrinks
            );
        }
        if stats.speculative_stats.primary_verify_requests > 0 {
            let avg_span_ms = stats.speculative_stats.primary_verify_elapsed_ms
                / stats.speculative_stats.primary_verify_requests as f64;
            let ms_per_token = stats.speculative_stats.primary_verify_elapsed_ms
                / stats.speculative_stats.primary_verify_tokens.max(1) as f64;
            let primary_stage_total_ms = us_to_ms(stats.speculative_stats.primary_verify_total_us);
            let client_unaccounted_ms = (stats.speculative_stats.primary_verify_elapsed_ms
                - primary_stage_total_ms)
                .max(0.0);
            eprintln!(
                "  spec     verify_wall_ms requests={} tokens={} elapsed={:.2} write={:.2} wait={:.2} avg_span={:.2} ms_per_token={:.2} client_unaccounted={:.2}",
                stats.speculative_stats.primary_verify_requests,
                stats.speculative_stats.primary_verify_tokens,
                stats.speculative_stats.primary_verify_elapsed_ms,
                stats.speculative_stats.primary_verify_write_ms,
                stats.speculative_stats.primary_verify_wait_ms,
                avg_span_ms,
                ms_per_token,
                client_unaccounted_ms
            );
            let primary_verify_tok_s = if stats.speculative_stats.primary_verify_elapsed_ms > 0.0 {
                1000.0 * stats.speculative_stats.primary_verify_tokens as f64
                    / stats.speculative_stats.primary_verify_elapsed_ms
            } else {
                0.0
            };
            eprintln!(
                "  spec     verify_primary_breakdown_ms total={:.2} compute={:.2} forward={:.2} downstream_wait={:.2} stages={} verify_tok_s={:.2}",
                primary_stage_total_ms,
                us_to_ms(stats.speculative_stats.primary_verify_compute_us),
                us_to_ms(stats.speculative_stats.primary_verify_forward_write_us),
                us_to_ms(stats.speculative_stats.primary_verify_downstream_wait_us),
                stats.speculative_stats.primary_verify_stage_count,
                primary_verify_tok_s
            );
        }
        if stats.speculative_stats.recovery_restores > 0 {
            if stats.speculative_stats.checkpoint_ms > 0.0 {
                eprintln!(
                    "  spec     checkpoint_ms={:.2} recovery_restores={} recovery_decode_repairs={} recovery_decode_ms={:.2} recovery_reverify_tokens={} recovery_ms={:.2}",
                    stats.speculative_stats.checkpoint_ms,
                    stats.speculative_stats.recovery_restores,
                    stats.speculative_stats.recovery_decode_repairs,
                    stats.speculative_stats.recovery_decode_elapsed_ms,
                    stats.speculative_stats.recovery_reverify_tokens,
                    stats.speculative_stats.recovery_ms
                );
            } else {
                eprintln!(
                    "  spec     recovery_restores={} recovery_decode_repairs={} recovery_decode_ms={:.2} recovery_reverify_tokens={} recovery_ms={:.2}",
                    stats.speculative_stats.recovery_restores,
                    stats.speculative_stats.recovery_decode_repairs,
                    stats.speculative_stats.recovery_decode_elapsed_ms,
                    stats.speculative_stats.recovery_reverify_tokens,
                    stats.speculative_stats.recovery_ms
                );
            }
        } else if stats.speculative_stats.checkpoint_ms > 0.0 {
            eprintln!(
                "  spec     checkpoint_ms={:.2}",
                stats.speculative_stats.checkpoint_ms
            );
        }
        if stats.reply_stats.checkpoint_total_us > 0 {
            eprintln!(
                "  spec     checkpoint_breakdown_ms total={:.2} flush={:.2} prefill_drain={:.2} local={:.2} downstream_write={:.2} downstream_wait={:.2} drained_replies={}",
                us_to_ms(stats.reply_stats.checkpoint_total_us),
                us_to_ms(stats.reply_stats.checkpoint_flush_us),
                us_to_ms(stats.reply_stats.checkpoint_prefill_drain_us),
                us_to_ms(stats.reply_stats.checkpoint_local_us),
                us_to_ms(stats.reply_stats.checkpoint_downstream_write_us),
                us_to_ms(stats.reply_stats.checkpoint_downstream_wait_us),
                stats.reply_stats.checkpoint_prefill_drained_replies
            );
        }
        if stats.reply_stats.restore_total_us > 0 {
            eprintln!(
                "  spec     restore_breakdown_ms total={:.2} flush={:.2} prefill_drain={:.2} local={:.2} downstream_write={:.2} downstream_wait={:.2} drained_replies={}",
                us_to_ms(stats.reply_stats.restore_total_us),
                us_to_ms(stats.reply_stats.restore_flush_us),
                us_to_ms(stats.reply_stats.restore_prefill_drain_us),
                us_to_ms(stats.reply_stats.restore_local_us),
                us_to_ms(stats.reply_stats.restore_downstream_write_us),
                us_to_ms(stats.reply_stats.restore_downstream_wait_us),
                stats.reply_stats.restore_prefill_drained_replies
            );
        }
        if stats.reply_stats.verify_span_total_us > 0 {
            let verify_total_ms = us_to_ms(stats.reply_stats.verify_span_total_us);
            let verify_tok_s = if verify_total_ms > 0.0 {
                1000.0 * stats.speculative_stats.draft_tokens as f64 / verify_total_ms
            } else {
                0.0
            };
            eprintln!(
                "  spec     verify_breakdown_ms total={:.2} compute={:.2} forward={:.2} downstream_wait={:.2} stages={} proposed_tok_s={:.2}",
                verify_total_ms,
                us_to_ms(stats.reply_stats.verify_span_compute_us),
                us_to_ms(stats.reply_stats.verify_span_forward_write_us),
                us_to_ms(stats.reply_stats.verify_span_downstream_wait_us),
                stats.reply_stats.verify_span_stage_count,
                verify_tok_s
            );
            let protocol_avg_span = if stats.reply_stats.verify_span_request_count > 0 {
                stats.reply_stats.verify_span_token_count as f64
                    / stats.reply_stats.verify_span_request_count as f64
            } else {
                0.0
            };
            eprintln!(
                "  spec     verify_batch_stats protocol_requests={} protocol_tokens={} max_span={} avg_span={:.2} checkpointed_requests={} skip_checkpoint_requests={}",
                stats.reply_stats.verify_span_request_count,
                stats.reply_stats.verify_span_token_count,
                stats.reply_stats.verify_span_max_tokens,
                protocol_avg_span,
                stats.reply_stats.verify_span_checkpointed_requests,
                stats.reply_stats.verify_span_skip_checkpoint_requests
            );
        }
        if stats.speculative_stats.recovery_reverify_elapsed_ms > 0.0 {
            eprintln!(
                "  spec     recovery_reverify_breakdown_ms elapsed={:.2} write={:.2} wait={:.2} stage_compute={:.2} stage_forward={:.2} stage_downstream_wait={:.2} stages={}",
                stats.speculative_stats.recovery_reverify_elapsed_ms,
                stats.speculative_stats.recovery_reverify_write_ms,
                stats.speculative_stats.recovery_reverify_wait_ms,
                us_to_ms(stats.speculative_stats.recovery_reverify_compute_us),
                us_to_ms(stats.speculative_stats.recovery_reverify_forward_write_us),
                us_to_ms(stats.speculative_stats.recovery_reverify_downstream_wait_us),
                stats.speculative_stats.recovery_reverify_stage_count
            );
        }
    }
}

fn send_session_control(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    kind: WireMessageKind,
) -> Result<SessionControlReply> {
    if !kind.is_session_control() {
        bail!("session control requires a session-control message kind");
    }
    let started = Instant::now();
    let mut state = StageStateHeader::new(kind, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind,
        pos_start: 0,
        token_count: 0,
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
        .with_context(|| format!("send session control {kind:?}"))?;
    let reply =
        recv_reply(&mut *stream).with_context(|| format!("receive session control {kind:?}"))?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected session-control ACK, got {:?}", reply.kind);
    }
    Ok(SessionControlReply {
        stats: reply.stats,
        elapsed_ms: elapsed_ms(started),
    })
}

fn send_try_restore_prefill(
    stream: &mut TcpStream,
    wire_dtype: WireActivationDType,
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    tokens: &[i32],
) -> Result<SessionControlReply> {
    if tokens.is_empty() {
        bail!("exact-prefix restore requires at least one token");
    }
    let started = Instant::now();
    let mut state = StageStateHeader::new(WireMessageKind::TryRestorePrefill, wire_dtype);
    state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(tokens.len()).context("prefix token count exceeds i32")?;
    state.current_token = tokens.last().copied().unwrap_or(LLAMA_TOKEN_NULL);
    state.source_stage_index = -1;
    let message = StageWireMessage {
        kind: WireMessageKind::TryRestorePrefill,
        pos_start: 0,
        token_count: i32::try_from(tokens.len()).context("prefix token count exceeds i32")?,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype)
        .context("send exact-prefix restore request")?;
    let reply = recv_reply(&mut *stream).context("receive exact-prefix restore ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected exact-prefix restore ACK, got {:?}", reply.kind);
    }
    Ok(SessionControlReply {
        stats: reply.stats,
        elapsed_ms: elapsed_ms(started),
    })
}

struct ReplPrefillChunk<'a> {
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    pos_start: usize,
    prefill_token_count: usize,
    tokens: &'a [i32],
}

fn send_prefill_chunk(
    stream: &mut std::net::TcpStream,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    chunk: ReplPrefillChunk<'_>,
) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id = i32::try_from(chunk.prompt_index).context("prompt index exceeds i32")?;
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
    write_stage_message(&mut *stream, &message, wire_dtype)
        .with_context(|| format!("send prefill chunk at {}", chunk.pos_start))?;
    let reply = recv_reply(&mut *stream)
        .with_context(|| format!("receive prefill chunk ACK at {}", chunk.pos_start))?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn us_to_ms(us: i64) -> f64 {
    us as f64 / 1000.0
}

fn nonzero_min(current: usize, value: usize) -> usize {
    if current == 0 {
        value
    } else {
        current.min(value)
    }
}

fn default_session_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("prompt-{}-{millis}", std::process::id())
}

fn stable_wire_id(parts: &[&[u8]]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let id = u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("8-byte digest prefix"),
    );
    if id == 0 {
        1
    } else {
        id
    }
}
