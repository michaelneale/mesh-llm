struct PromptRun<'a> {
    args: &'a BinaryReplArgs,
    tokenizer: &'a StageModel,
    chat_template_model: Option<&'a StageModel>,
    draft: Option<&'a mut DraftRunner>,
    ngram: Option<&'a mut NgramSource>,
    interrupt: &'a Arc<PromptInterruptState>,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    session_id: &'a str,
    wire_session_id: u64,
    prompt_index: usize,
    prompt: &'a str,
    live_session: Option<&'a mut PromptLiveSession>,
}

fn run_prompt(run: PromptRun<'_>) -> Result<()> {
    let PromptRun {
        args,
        tokenizer,
        chat_template_model,
        mut draft,
        mut ngram,
        interrupt,
        wire_dtype,
        session_id,
        wire_session_id,
        prompt_index,
        prompt,
        mut live_session,
    } = run;

    if args.prefill_chunk_size == 0 {
        bail!("prefill_chunk_size must be greater than zero");
    }
    interrupt.begin_request();
    let wall_started = Instant::now();
    let tokenize_started = Instant::now();
    let live_enabled = live_session.is_some() && !args.raw_prompt;
    let mut live_messages = Vec::new();
    let prompt_for_model = if let Some(live) = live_session.as_ref().filter(|_| !args.raw_prompt) {
        live_messages = live.messages.clone();
        live_messages.push(ChatTemplateMessage::new("user", prompt));
        format_messages_for_model(tokenizer, chat_template_model, &live_messages, args)?
    } else {
        format_prompt_for_model(tokenizer, chat_template_model, prompt, args)?
    };
    let token_ids = tokenizer
        .tokenize(&prompt_for_model, true)
        .with_context(|| format!("tokenize prompt {prompt_for_model:?}"))?;
    let tokenize_ms = elapsed_ms(tokenize_started);
    if token_ids.is_empty() {
        bail!("prompt produced no tokens");
    }
    let max_new_tokens =
        effective_prompt_max_new_tokens(args.max_new_tokens, args.ctx_size, token_ids.len())?;

    let prefill_token_count = if token_ids.len() == 1 {
        1
    } else {
        token_ids.len().saturating_sub(1)
    };
    eprintln!(
        "request {prompt_index}: prompt_tokens={} prefill_tokens={} max_new_tokens={}",
        token_ids.len(),
        prefill_token_count,
        max_new_tokens
    );
    let prompt_index_bytes = prompt_index.to_le_bytes();
    let request_id = stable_wire_id(&[session_id.as_bytes(), &prompt_index_bytes]);

    let mut session_reuse = PromptSessionReuseStats::default();
    let mut one_shot_stream = None;
    if let Some(live) = live_session.as_deref_mut() {
        let resident_before = live.resident_tokens.len();
        session_reuse.resident_tokens_before = resident_before;
        if live.dirty {
            eprintln!("request {prompt_index}: resetting dirty live session");
            reset_live_prompt_runtime(
                live,
                args,
                wire_dtype,
                prompt_index,
                request_id,
                wire_session_id,
            )?;
            session_reuse.outcome = "reset";
        }
        let prefix_match =
            live_resident_prefix_matches(&live.resident_tokens, &token_ids, prefill_token_count);
        if !live.resident_tokens.is_empty() && !prefix_match {
            let common_prefix = common_token_prefix_len(&live.resident_tokens, &token_ids);
            eprintln!(
                "request {prompt_index}: live session prefix mismatch common_prefix={} resident_len={} prompt_len={} resident_tail={:?} prompt_tail={:?}",
                common_prefix,
                live.resident_tokens.len(),
                token_ids.len(),
                token_window(&live.resident_tokens, common_prefix),
                token_window(&token_ids, common_prefix)
            );
            if common_prefix > 0 {
                let stream = live
                    .stream
                    .as_mut()
                    .context("live prompt stream was not connected")?;
                send_trim_session(
                    stream,
                    wire_dtype,
                    prompt_index,
                    request_id,
                    wire_session_id,
                    common_prefix,
                )
                .with_context(|| stage_chain_error_context(args))?;
                live.resident_tokens.truncate(common_prefix);
                session_reuse.outcome = "partial";
            } else {
                eprintln!(
                    "request {prompt_index}: live session prefix mismatch; resetting runtime lane"
                );
                reset_live_prompt_runtime(
                    live,
                    args,
                    wire_dtype,
                    prompt_index,
                    request_id,
                    wire_session_id,
                )?;
                session_reuse.outcome = "mismatch";
            }
        }
        if live.stream.is_none() {
            eprintln!(
                "request {prompt_index}: connecting to {}",
                args.first_stage_addr
            );
            live.stream = Some(
                connect_ready(&args.first_stage_addr, args.startup_timeout_secs)
                    .context("first binary stage did not become ready")?,
            );
            eprintln!("request {prompt_index}: connected");
            if session_reuse.outcome == "disabled" {
                session_reuse.outcome = "miss";
            }
        } else if prefix_match {
            session_reuse.outcome = "hit";
        } else if session_reuse.outcome == "disabled" {
            session_reuse.outcome = "miss";
        }
        session_reuse.reused_tokens = if prefix_match || session_reuse.outcome == "partial" {
            live.resident_tokens.len()
        } else {
            0
        };
    } else {
        eprintln!(
            "request {prompt_index}: connecting to {}",
            args.first_stage_addr
        );
        one_shot_stream = Some(
            connect_ready(&args.first_stage_addr, args.startup_timeout_secs)
                .context("first binary stage did not become ready")?,
        );
        eprintln!("request {prompt_index}: connected");
    }
    let mut prefill_start = session_reuse.reused_tokens.min(prefill_token_count);
    let stream = if let Some(live) = live_session.as_deref_mut() {
        live.stream
            .as_mut()
            .context("live prompt stream was not connected")?
    } else {
        one_shot_stream
            .as_mut()
            .context("one-shot prompt stream was not connected")?
    };
    let _interrupt_guard = interrupt.activate(stream)?;
    let io_timeout = Duration::from_secs(args.decode_timeout_secs.max(1));
    stream.set_read_timeout(Some(io_timeout)).ok();
    stream.set_write_timeout(Some(io_timeout)).ok();
    let mut reply_stats = StageReplyStats::default();
    if should_try_exact_prefix_restore(live_enabled, prefill_start, prefill_token_count) {
        let restore_tokens = &token_ids[..prefill_token_count];
        eprintln!(
            "request {prompt_index}: checking exact-prefix cache tokens={}",
            restore_tokens.len()
        );
        let restore = send_try_restore_prefill(
            stream,
            wire_dtype,
            prompt_index,
            request_id,
            wire_session_id,
            restore_tokens,
        )
        .with_context(|| stage_chain_error_context(args))?;
        let hit_stage_count = stage_mask_count(restore.stats.kv_hit_stage_mask);
        let cached_prompt_tokens_est = (restore.stats.kv_imported_tokens.max(0) as u64)
            .checked_div(hit_stage_count)
            .unwrap_or(0);
        let restored_full_prefix = restore.stats.kv_lookup_hits > 0
            && restore.stats.kv_lookup_misses == 0
            && restore.stats.kv_lookup_errors == 0
            && cached_prompt_tokens_est >= prefill_token_count as u64;
        if restored_full_prefix {
            prefill_start = prefill_token_count;
            eprintln!(
                "request {prompt_index}: exact-prefix cache hit stages={} imported_tokens={} elapsed_ms={:.2}",
                format_stage_mask(restore.stats.kv_hit_stage_mask),
                restore.stats.kv_imported_tokens.max(0),
                restore.elapsed_ms
            );
        } else {
            eprintln!(
                "request {prompt_index}: exact-prefix cache miss hit={} miss={} error={} stages={} elapsed_ms={:.2}",
                restore.stats.kv_lookup_hits,
                restore.stats.kv_lookup_misses,
                restore.stats.kv_lookup_errors,
                format_stage_mask(restore.stats.kv_hit_stage_mask),
                restore.elapsed_ms
            );
        }
        reply_stats.merge(restore.stats);
    }
    let prefill_started = Instant::now();
    let mut prefill_chunk_count = 0usize;
    let prefill_tokens = &token_ids[prefill_start..prefill_token_count];
    session_reuse.appended_prefill_tokens = prefill_tokens.len();
    if !prefill_tokens.is_empty() {
        for (chunk_index, chunk) in prefill_tokens.chunks(args.prefill_chunk_size).enumerate() {
            if interrupt.interrupt_requested() {
                bail!("prompt interrupted");
            }
            prefill_chunk_count += 1;
            let pos_start = prefill_start + chunk_index * args.prefill_chunk_size;
            eprintln!(
                "request {prompt_index}: prefill chunk {} tokens={} pos={}",
                prefill_chunk_count - 1,
                chunk.len(),
                pos_start
            );
            send_prefill_chunk(
                stream,
                wire_dtype,
                ReplPrefillChunk {
                    prompt_index,
                    request_id,
                    session_id: wire_session_id,
                    pos_start,
                    prefill_token_count,
                    tokens: chunk,
                },
            )
            .with_context(|| stage_chain_error_context(args))?;
        }
    }
    let prefill_ms = elapsed_ms(prefill_started);
    eprintln!(
        "request {prompt_index}: prefill complete chunks={} elapsed_ms={:.2}",
        prefill_chunk_count, prefill_ms
    );

    let mut current = *token_ids.last().expect("checked non-empty tokens");
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut decode_ms = 0.0;
    let mut first_decode_ms = None;
    let mut first_time_to_token_ms = None;
    let mut saw_visible_output = false;
    let mut speculative_stats = SpeculativeStats::default();
    let mut assistant_raw_text = String::new();
    let mut generation_reached_eog = false;
    let mut context_tokens = token_ids.clone();
    if let Some(draft) = draft.as_deref_mut() {
        draft.reset_to_context(&context_tokens)?;
    }
    if let Some(ngram) = ngram.as_deref_mut() {
        ngram.observe_sequence(session_id, &context_tokens)?;
    }
    let max_speculative_window = args.speculative_window.max(1);
    let mut adaptive_window = if args.adaptive_speculative_window {
        max_speculative_window.min(4)
    } else {
        max_speculative_window
    };
    if draft.is_some() || ngram.is_some() {
        speculative_stats.adaptive_window_max = max_speculative_window;
        speculative_stats.adaptive_window_start = adaptive_window;
        speculative_stats.adaptive_window_enabled = args.adaptive_speculative_window;
    }

    while generated.len() < max_new_tokens {
        if interrupt.interrupt_requested() {
            bail!("prompt interrupted");
        }
        if generated.is_empty() {
            eprintln!("request {prompt_index}: waiting for first decode token");
        }

        let remaining = max_new_tokens - generated.len();
        let proposal_limit = remaining.min(adaptive_window);
        let draft_tokens = match ngram.as_deref_mut() {
            Some(ngram) => ngram.propose(session_id, &context_tokens, proposal_limit)?,
            None => Vec::new(),
        };
        let draft_tokens = if draft_tokens.is_empty() {
            match draft.as_deref_mut() {
                Some(draft) if draft.window > 0 => {
                    draft.propose(current, proposal_limit.min(draft.window))?
                }
                _ => Vec::new(),
            }
        } else {
            draft_tokens
        };

        if draft_tokens.is_empty() {
            let decode_index = generated.len();
            let reply = send_decode_step(
                stream,
                wire_dtype,
                prompt_index,
                request_id,
                wire_session_id,
                token_ids.len(),
                prefill_token_count,
                decode_index,
                current,
            )
            .with_context(|| stage_chain_error_context(args))?;
            decode_ms += reply.elapsed_ms;
            first_decode_ms.get_or_insert(reply.elapsed_ms);
            reply_stats.merge(reply.stats);
            current = reply.predicted;
            generated.push(current);
            context_tokens.push(current);
            if let Some(ngram) = ngram.as_deref_mut() {
                ngram.observe_accepted(session_id, &context_tokens)?;
            }
            first_time_to_token_ms.get_or_insert_with(|| elapsed_ms(wall_started));
            if tokenizer.token_is_eog(current)? {
                generation_reached_eog = true;
                break;
            }
            let piece = tokenizer.detokenize(&[current])?;
            assistant_raw_text.push_str(&piece);
            if piece.chars().any(|ch| !ch.is_whitespace()) {
                saw_visible_output = true;
            }
            print!("{piece}");
            io::stdout().flush().ok();
            continue;
        }

        speculative_stats.windows += 1;
        speculative_stats.draft_tokens += draft_tokens.len();
        speculative_stats.adaptive_window_sum += adaptive_window;
        speculative_stats.adaptive_window_min =
            nonzero_min(speculative_stats.adaptive_window_min, adaptive_window);
        speculative_stats.adaptive_window_max_seen = speculative_stats
            .adaptive_window_max_seen
            .max(adaptive_window);
        let decode_index = generated.len();
        let verify_inputs = verify_inputs_for_proposals(current, &draft_tokens);
        let reply = send_verify_span(
            stream,
            wire_dtype,
            prompt_index,
            request_id,
            wire_session_id,
            token_ids.len(),
            prefill_token_count + decode_index,
            decode_index,
            &verify_inputs,
            true,
        )
        .with_context(|| stage_chain_error_context(args))?;
        decode_ms += reply.elapsed_ms;
        first_decode_ms.get_or_insert(reply.elapsed_ms);
        speculative_stats.observe_primary_verify(&reply, verify_inputs.len());
        reply_stats.merge(reply.stats);
        first_time_to_token_ms.get_or_insert_with(|| elapsed_ms(wall_started));
        let decision = classify_verify_span(
            &draft_tokens,
            &reply.predicted_tokens,
            generated.len(),
            max_new_tokens,
            |token| tokenizer.token_is_eog(token),
        )?;
        speculative_stats.observe_verify_decision(
            decision,
            &mut adaptive_window,
            args.adaptive_speculative_window,
            max_speculative_window,
        );

        let mut commit_tokens = reply.predicted_tokens[..decision.commit_count].to_vec();

        if decision.requires_repair() {
            let repair_input_count = decision
                .repair_input_count
                .context("missing rejected span index")?;
            speculative_stats.recovery_restores += 1;
            let restore = send_session_control(
                stream,
                wire_dtype,
                prompt_index,
                request_id,
                wire_session_id,
                WireMessageKind::RestoreSession,
            )
            .with_context(|| stage_chain_error_context(args))?;
            decode_ms += restore.elapsed_ms;
            speculative_stats.recovery_ms += restore.elapsed_ms;
            speculative_stats.recovery_restore_ms += restore.elapsed_ms;
            reply_stats.merge(restore.stats);

            if repair_input_count == 1 {
                let repair = send_decode_step(
                    stream,
                    wire_dtype,
                    prompt_index,
                    request_id,
                    wire_session_id,
                    token_ids.len(),
                    prefill_token_count,
                    decode_index,
                    current,
                )
                .with_context(|| stage_chain_error_context(args))?;
                commit_tokens = vec![repair.predicted];
                reply_stats.merge(repair.stats);
                decode_ms += repair.elapsed_ms;
                speculative_stats.recovery_decode_repairs += 1;
                speculative_stats.recovery_ms += repair.elapsed_ms;
                speculative_stats.recovery_decode_elapsed_ms += repair.elapsed_ms;
            } else {
                let repair_inputs = &verify_inputs[..repair_input_count];
                let repair = send_verify_span(
                    stream,
                    wire_dtype,
                    prompt_index,
                    request_id,
                    wire_session_id,
                    token_ids.len(),
                    prefill_token_count + decode_index,
                    decode_index,
                    repair_inputs,
                    false,
                )
                .with_context(|| stage_chain_error_context(args))?;
                commit_tokens = repaired_commit_tokens(
                    &draft_tokens,
                    decision.accepted_before_reject,
                    repair_input_count,
                    &repair.predicted_tokens,
                )?;
                reply_stats.merge(repair.stats);
                decode_ms += repair.elapsed_ms;
                speculative_stats.recovery_reverify_tokens += repair_inputs.len();
                speculative_stats.recovery_ms += repair.elapsed_ms;
                speculative_stats.recovery_reverify_elapsed_ms += repair.elapsed_ms;
                speculative_stats.recovery_reverify_write_ms += repair.write_ms;
                speculative_stats.recovery_reverify_wait_ms += repair.wait_ms;
                speculative_stats.recovery_reverify_compute_us +=
                    repair.stats.verify_span_compute_us;
                speculative_stats.recovery_reverify_forward_write_us +=
                    repair.stats.verify_span_forward_write_us;
                speculative_stats.recovery_reverify_downstream_wait_us +=
                    repair.stats.verify_span_downstream_wait_us;
                speculative_stats.recovery_reverify_stage_count +=
                    repair.stats.verify_span_stage_count;
            }
        }
        let mut reached_eog = false;
        for predicted in commit_tokens {
            current = predicted;
            generated.push(current);
            context_tokens.push(current);
            if let Some(ngram) = ngram.as_deref_mut() {
                ngram.observe_accepted(session_id, &context_tokens)?;
            }
            if tokenizer.token_is_eog(current)? {
                reached_eog = true;
                generation_reached_eog = true;
            }
            let piece = tokenizer.detokenize(&[current])?;
            assistant_raw_text.push_str(&piece);
            if piece.chars().any(|ch| !ch.is_whitespace()) {
                saw_visible_output = true;
            }
            print!("{piece}");
            io::stdout().flush().ok();
            if reached_eog || generated.len() >= max_new_tokens {
                break;
            }
        }
        speculative_stats.adaptive_window_final = adaptive_window;
        if decision.rejected() || reached_eog {
            if let Some(draft) = draft.as_deref_mut() {
                draft.reset_to_context(&context_tokens)?;
            }
        }
        if reached_eog {
            break;
        }
    }
    println!();

    if !saw_visible_output {
        eprintln!(
            "warning: generated no visible non-whitespace text; first generated token ids: {:?}",
            generated.iter().take(16).collect::<Vec<_>>()
        );
    }
    if live_enabled {
        rematerialize_live_transcript(
            stream,
            tokenizer,
            chat_template_model,
            args,
            wire_dtype,
            prompt_index,
            request_id,
            wire_session_id,
            &live_messages,
            &token_ids,
            &generated,
            &assistant_raw_text,
            generation_reached_eog,
        )?;
    }

    if !live_enabled {
        stop_prompt_stream(stream, wire_dtype, request_id, wire_session_id, args)?;
    }

    let wallblock_ms = elapsed_ms(wall_started);
    let generated_tokens = generated.len();
    let tpot_ms = if generated_tokens == 0 {
        0.0
    } else {
        decode_ms / generated_tokens as f64
    };
    let tpot_after_first_ms = if generated_tokens <= 1 {
        0.0
    } else {
        (decode_ms - first_decode_ms.unwrap_or(0.0)) / (generated_tokens - 1) as f64
    };
    let mut resident_tokens_after = 0usize;
    if let Some(live) = live_session {
        live.messages = live_messages;
        if let Some(last) = live.messages.last_mut() {
            if last.role == "user" {
                live.messages
                    .push(ChatTemplateMessage::new("assistant", &assistant_raw_text));
            }
        }
        live.resident_tokens =
            live_transcript_tokens(tokenizer, chat_template_model, args, &live.messages)?;
        live.dirty = !generation_reached_eog;
        resident_tokens_after = live.resident_tokens.len();
    }
    session_reuse.resident_tokens_after = resident_tokens_after;
    print_stats(Stats {
        prompt_tokens: token_ids.len(),
        prefill_tokens: prefill_token_count,
        prefill_chunks: prefill_chunk_count,
        generated_tokens,
        tokenize_ms,
        prefill_ms,
        decode_ms,
        wallblock_ms,
        first_time_to_token_ms: first_time_to_token_ms.unwrap_or(0.0),
        tpot_ms,
        tpot_after_first_ms,
        reply_stats,
        speculative_stats,
        session_reuse,
    });

    Ok(())
}
