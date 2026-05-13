use super::*;

impl StageOpenAiBackend {
    pub(super) fn generate_embedded_stage_zero_tokens(
        &self,
        request: EmbeddedStageZeroGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        if request.config.downstream.is_none() {
            return self.generate_local_tokens(
                LocalGeneration {
                    prompt_token_ids: request.prompt_token_ids,
                    max_tokens: request.max_tokens,
                    sampling: request.sampling,
                    chat_sampling_metadata: request.chat_sampling_metadata,
                    hook_request: request.hook_request,
                    hook_runtime: request.hook_runtime,
                    cancellation: request.cancellation,
                    ids: request.ids,
                },
                on_token,
            );
        }

        let wire_sampling = wire_sampling_config(request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let lane_pool = request
            .lane_pool
            .as_ref()
            .ok_or_else(|| OpenAiError::backend("embedded stage 0 has no downstream lane pool"))?;
        let mut lane = lane_pool.checkout(request.ids)?;
        let mut cache_stats = GenerationCacheStats::default();

        let result = (|| {
            let downstream = &mut lane.stream;
            let prefill_token_count = request.prompt_token_ids.len().saturating_sub(1);
            let prefill_timer = PhaseTimer::start();
            let mut prefill_chunks = 0usize;
            let mut prefill_min_chunk_size = usize::MAX;
            let mut prefill_max_chunk_size = 0usize;
            let mut prefill_stage0_compute_ms = 0.0;
            let mut prefill_runtime_lock_wait_ms = 0.0;
            let mut prefill_runtime_lock_wait_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_hold_ms = 0.0;
            let mut prefill_runtime_lock_hold_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_acquires = 0usize;
            let mut prefill_runtime_sessions_before = None;
            let mut prefill_runtime_sessions_after = None;
            let mut prefill_forward_write_ms = 0.0;
            let mut prefill_output_activation_bytes = 0usize;
            let mut prefill_forward_activation_bytes = 0usize;
            let mut prefill_downstream_wait_ms = 0.0;
            let mut pending_prefill_replies = 0usize;
            let mut prefill_credit_wait_count = 0usize;
            let mut prefill_deferred_replies_drained = 0usize;
            let mut prefill_pending_replies_max = 0usize;
            let mut prefill_stage0_cache_hits = 0usize;
            let mut prefill_stage0_cache_misses = 0usize;
            let mut prefill_stage0_cache_errors = 0usize;
            let mut prefill_chain_cache_restored = false;
            let mut prefill_chain_restored_tokens = 0usize;
            let mut prefill_chain_cache_stats = StageReplyStats::default();
            let mut prefill_stage0_full_recorded = false;
            let mut fused_first_decode = None;
            let mut prefill_planner = request.prefill_chunk_policy.planner();
            if prefill_token_count > 0 {
                let prefill_tokens = &request.prompt_token_ids[..prefill_token_count];
                if request.max_tokens > 0 && request.draft.is_none() {
                    let current = *request
                        .prompt_token_ids
                        .last()
                        .expect("checked non-empty prompt");
                    if let Some(fused) = self.try_restore_embedded_split_prefill_and_decode(
                        &request,
                        &session_key,
                        downstream,
                        prefill_tokens,
                        current,
                        wire_sampling.clone(),
                    )? {
                        prefill_chain_cache_restored = true;
                        prefill_chain_restored_tokens = prefill_token_count;
                        prefill_chain_cache_stats = fused.reply_stats;
                        cache_stats.cached_prompt_tokens = saturating_u32(prefill_token_count);
                        cache_stats.matched_prefix_tokens = saturating_u32(prefill_token_count);
                        cache_stats.suffix_prefill_tokens = 0;
                        cache_stats.hit_kind = Some("chain_fused_exact_prefix");
                        fused_first_decode = Some(fused);
                    }
                }
                if !prefill_chain_cache_restored {
                    if let Some(restore) = self.try_restore_embedded_split_prefill(
                        &request,
                        &session_key,
                        downstream,
                        prefill_tokens,
                    )? {
                        prefill_chain_restored_tokens = restore.restored_tokens;
                        prefill_chain_cache_restored =
                            prefill_chain_restored_tokens >= prefill_tokens.len();
                        prefill_chain_cache_stats = restore.stats;
                        cache_stats.cached_prompt_tokens =
                            saturating_u32(prefill_chain_restored_tokens);
                        cache_stats.matched_prefix_tokens =
                            saturating_u32(prefill_chain_restored_tokens);
                        cache_stats.suffix_prefill_tokens = saturating_u32(
                            prefill_tokens
                                .len()
                                .saturating_sub(prefill_chain_restored_tokens),
                        );
                        cache_stats.hit_kind = Some("chain_prefix");
                    }
                }
                let mut pos_start = prefill_chain_restored_tokens.min(prefill_tokens.len());
                let mut chunk_index = 0usize;
                while pos_start < prefill_tokens.len() {
                    if request
                        .cancellation
                        .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                    {
                        drain_embedded_prefill_replies(
                            downstream,
                            &mut pending_prefill_replies,
                            &mut prefill_chain_cache_stats,
                        )?;
                        return Ok(());
                    }
                    let chunk_size = prefill_planner.chunk_size_for(chunk_index);
                    let end = pos_start
                        .saturating_add(chunk_size)
                        .min(prefill_tokens.len());
                    let chunk = &prefill_tokens[pos_start..end];
                    prefill_min_chunk_size = prefill_min_chunk_size.min(chunk.len());
                    prefill_max_chunk_size = prefill_max_chunk_size.max(chunk.len());
                    let message = embedded_prefill_message(
                        request.wire_dtype,
                        OpenAiPrefillChunk {
                            seq_id: chunk_index,
                            pos_start,
                            prefill_token_count,
                            tokens: chunk,
                            request_id,
                            session_id,
                        },
                    )?;
                    let stage0_timer = PhaseTimer::start();
                    let pending_prefill_replies_before = pending_prefill_replies;
                    let mut output = self.restore_embedded_stage0_prefill(
                        &session_key,
                        request.ids,
                        pos_start as u64,
                        chunk,
                        request.activation_width,
                    )?;
                    if output.is_some() {
                        prefill_stage0_cache_hits += 1;
                    } else {
                        prefill_stage0_cache_misses += usize::from(pos_start == 0);
                    }
                    let output = match output.take() {
                        Some(output) => output,
                        None => {
                            let lock_timer = PhaseTimer::start();
                            let mut runtime = self
                                .runtime
                                .lock()
                                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                            let lock_wait_ms = lock_timer.elapsed_ms();
                            prefill_runtime_lock_wait_ms += lock_wait_ms;
                            prefill_runtime_lock_wait_max_ms =
                                prefill_runtime_lock_wait_max_ms.max(lock_wait_ms);
                            prefill_runtime_lock_acquires += 1;
                            let lock_hold_timer = PhaseTimer::start();
                            prefill_runtime_sessions_before
                                .get_or_insert_with(|| runtime.session_stats());
                            let output = run_binary_stage_message(
                                &mut runtime,
                                &session_key,
                                &message,
                                chunk,
                                None,
                                false,
                            )
                            .map_err(openai_backend_error)?
                            .2;
                            prefill_runtime_sessions_after = Some(runtime.session_stats());
                            let lock_hold_ms = lock_hold_timer.elapsed_ms();
                            prefill_runtime_lock_hold_ms += lock_hold_ms;
                            prefill_runtime_lock_hold_max_ms =
                                prefill_runtime_lock_hold_max_ms.max(lock_hold_ms);
                            output
                        }
                    };
                    if let Err(error) = self.record_embedded_stage0_prefill(
                        &session_key,
                        request.ids,
                        pos_start as u64,
                        chunk,
                        request.activation_width,
                        &output,
                    ) {
                        prefill_stage0_cache_errors += 1;
                        let mut attrs = self.openai_attrs(request.ids);
                        attrs.insert(
                            "skippy.kv.decision".to_string(),
                            json!("stage0_record_error"),
                        );
                        attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                        self.telemetry
                            .emit("stage.openai_kv_record_decision", attrs);
                    }
                    let chunk_stage0_compute_ms = stage0_timer.elapsed_ms();
                    prefill_stage0_compute_ms += chunk_stage0_compute_ms;
                    let forwarded = forwarded_stage_message(
                        request.config,
                        &message,
                        &output,
                        request.wire_dtype,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?;
                    prefill_output_activation_bytes =
                        prefill_output_activation_bytes.saturating_add(output.payload.len());
                    prefill_forward_activation_bytes =
                        prefill_forward_activation_bytes.saturating_add(forwarded.activation.len());
                    let write_timer = PhaseTimer::start();
                    write_stage_message_conditioned(
                        &mut *downstream,
                        &forwarded,
                        request.wire_dtype,
                        request.downstream_wire_condition,
                    )
                    .map_err(openai_io_error)?;
                    let chunk_forward_write_ms = write_timer.elapsed_ms();
                    prefill_forward_write_ms += chunk_forward_write_ms;
                    let mut chunk_downstream_wait_ms = 0.0;
                    let mut chunk_deferred_replies_drained = 0usize;
                    let mut chunk_credit_wait_count = 0usize;
                    if request.prefill_reply_credit_limit == 0 {
                        let wait_timer = PhaseTimer::start();
                        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                        chunk_downstream_wait_ms = wait_timer.elapsed_ms();
                        if reply.kind != WireReplyKind::Ack {
                            return Err(OpenAiError::backend(format!(
                                "expected prefill ACK from downstream, got {:?}",
                                reply.kind
                            )));
                        }
                        prefill_chain_cache_stats.merge(reply.stats);
                    } else {
                        while pending_prefill_replies >= request.prefill_reply_credit_limit {
                            prefill_credit_wait_count = prefill_credit_wait_count.saturating_add(1);
                            chunk_credit_wait_count = chunk_credit_wait_count.saturating_add(1);
                            let drained = drain_one_embedded_prefill_reply(
                                downstream,
                                &mut pending_prefill_replies,
                                &mut prefill_chain_cache_stats,
                            )?;
                            prefill_deferred_replies_drained = prefill_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_deferred_replies_drained = chunk_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_downstream_wait_ms += drained.downstream_wait_ms;
                        }
                        pending_prefill_replies = pending_prefill_replies.saturating_add(1);
                        prefill_pending_replies_max =
                            prefill_pending_replies_max.max(pending_prefill_replies);
                    }
                    prefill_downstream_wait_ms += chunk_downstream_wait_ms;
                    prefill_planner.observe(PrefillChunkObservation {
                        compute_ms: chunk_stage0_compute_ms,
                        forward_write_ms: chunk_forward_write_ms,
                        downstream_wait_ms: chunk_downstream_wait_ms,
                    });
                    let mut chunk_attrs = self.openai_attrs(request.ids);
                    chunk_attrs
                        .insert("llama_stage.message_kind".to_string(), json!("PrefillEmbd"));
                    chunk_attrs.insert("llama_stage.seq_id".to_string(), json!(chunk_index));
                    chunk_attrs.insert("llama_stage.pos_start".to_string(), json!(pos_start));
                    chunk_attrs.insert("llama_stage.token_count".to_string(), json!(chunk.len()));
                    chunk_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(chunk_stage0_compute_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(chunk_forward_write_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(chunk_downstream_wait_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.output_activation_bytes".to_string(),
                        json!(output.payload.len()),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_activation_bytes".to_string(),
                        json!(forwarded.activation.len()),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_limit".to_string(),
                        json!(request.prefill_reply_credit_limit),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_before".to_string(),
                        json!(pending_prefill_replies_before),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_after".to_string(),
                        json!(pending_prefill_replies),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_wait_count".to_string(),
                        json!(chunk_credit_wait_count),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_deferred_replies_drained".to_string(),
                        json!(chunk_deferred_replies_drained),
                    );
                    self.telemetry.emit_debug_span(
                        "stage.openai_prefill_chunk",
                        chunk_attrs,
                        stage0_timer.start_unix_nanos,
                        now_unix_nanos() as u64,
                    );
                    prefill_chunks += 1;
                    pos_start = end;
                    chunk_index += 1;
                }
                let drained = drain_embedded_prefill_replies(
                    downstream,
                    &mut pending_prefill_replies,
                    &mut prefill_chain_cache_stats,
                )?;
                prefill_deferred_replies_drained =
                    prefill_deferred_replies_drained.saturating_add(drained.drained_replies);
                prefill_downstream_wait_ms += drained.downstream_wait_ms;
                if !prefill_chain_cache_restored {
                    prefill_stage0_full_recorded = self.record_embedded_stage0_full_prefill(
                        &session_key,
                        request.ids,
                        prefill_tokens,
                    )?;
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
            prefill_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(prefill_stage0_compute_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(prefill_runtime_lock_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(prefill_runtime_lock_wait_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(prefill_runtime_lock_hold_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(prefill_runtime_lock_hold_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(prefill_runtime_lock_acquires),
            );
            if let Some(stats) = prefill_runtime_sessions_before.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = prefill_runtime_sessions_after.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            prefill_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(prefill_forward_write_ms),
            );
            prefill_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(prefill_output_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(prefill_forward_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(prefill_downstream_wait_ms),
            );
            prefill_attrs.insert(
                "skippy.prefill_credit_limit".to_string(),
                json!(request.prefill_reply_credit_limit),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_max".to_string(),
                json!(prefill_pending_replies_max),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_after".to_string(),
                json!(pending_prefill_replies),
            );
            prefill_attrs.insert(
                "skippy.prefill_credit_wait_count".to_string(),
                json!(prefill_credit_wait_count),
            );
            prefill_attrs.insert(
                "skippy.prefill_deferred_replies_drained".to_string(),
                json!(prefill_deferred_replies_drained),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_hits".to_string(),
                json!(prefill_stage0_cache_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_misses".to_string(),
                json!(prefill_stage0_cache_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_errors".to_string(),
                json!(prefill_stage0_cache_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_full_recorded".to_string(),
                json!(prefill_stage0_full_recorded),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_restored".to_string(),
                json!(prefill_chain_cache_restored),
            );
            prefill_attrs.insert(
                "skippy.kv.matched_prefix_tokens".to_string(),
                json!(prefill_chain_restored_tokens),
            );
            prefill_attrs.insert(
                "skippy.kv.suffix_prefill_tokens".to_string(),
                json!(prefill_token_count.saturating_sub(prefill_chain_restored_tokens)),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hits".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_misses".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_errors".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hit_stage_mask".to_string(),
                json!(prefill_chain_cache_stats.kv_hit_stage_mask),
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
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::Ack {
                    return Err(OpenAiError::backend(format!(
                        "expected generation config ACK from downstream, got {:?}",
                        reply.kind
                    )));
                }
            }

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_wait_max_ms = 0.0_f64;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_hold_max_ms = 0.0_f64;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_runtime_sessions_before = None;
            let mut decode_runtime_sessions_after = None;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_forward_activation_encode_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;
            let mut decode_downstream_wait_ms = 0.0;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            let mut context_tokens = request.prompt_token_ids.to_vec();
            let mut fused_reached_stop = false;
            if let Some(fused) = fused_first_decode.take() {
                current = fused.predicted;
                decoded_tokens = 1;
                context_tokens.push(current);
                decode_stage0_compute_ms += fused.execution.stage0_compute_ms;
                decode_runtime_lock_wait_ms += fused.execution.runtime_lock_wait_ms;
                decode_runtime_lock_wait_max_ms =
                    decode_runtime_lock_wait_max_ms.max(fused.execution.runtime_lock_wait_ms);
                decode_runtime_lock_hold_ms += fused.execution.runtime_lock_hold_ms;
                decode_runtime_lock_hold_max_ms =
                    decode_runtime_lock_hold_max_ms.max(fused.execution.runtime_lock_hold_ms);
                decode_runtime_lock_acquires += 1;
                decode_forward_activation_encode_ms += fused.execution.activation_encode_ms;
                decode_output_activation_bytes = decode_output_activation_bytes
                    .saturating_add(fused.execution.output_activation_bytes);
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(fused.execution.forward_activation_bytes);
                decode_forward_write_ms += fused.execution.forward_write_ms;
                decode_downstream_wait_ms += fused.execution.downstream_wait_ms;
                let mut token_attrs = self.openai_attrs(request.ids);
                token_attrs.insert("llama_stage.decode_step".to_string(), json!(0));
                token_attrs.insert(
                    "llama_stage.decode_token_phase".to_string(),
                    json!("fused-restore"),
                );
                token_attrs.insert(
                    "llama_stage.message_kind".to_string(),
                    json!("TryRestorePrefillDecode"),
                );
                token_attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(fused.elapsed_ms),
                );
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(fused.execution.stage0_compute_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(fused.execution.runtime_lock_wait_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(fused.execution.runtime_lock_hold_ms),
                );
                token_attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(fused.execution.output_activation_bytes),
                );
                token_attrs.insert(
                    "llama_stage.forward_activation_bytes".to_string(),
                    json!(fused.execution.forward_activation_bytes),
                );
                token_attrs.insert(
                    "llama_stage.activation_encode_ms".to_string(),
                    json!(fused.execution.activation_encode_ms),
                );
                token_attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(fused.execution.forward_write_ms),
                );
                token_attrs.insert(
                    "llama_stage.downstream_wait_ms".to_string(),
                    json!(fused.execution.downstream_wait_ms),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                self.telemetry
                    .emit_debug("stage.openai_decode_token", token_attrs);
                fused_reached_stop = on_token(current)? == TokenControl::Stop;
            }
            let max_speculative_window = request.speculative_window.max(1);
            let mut adaptive_window = if request.adaptive_speculative_window {
                max_speculative_window.min(4)
            } else {
                max_speculative_window
            };
            let mut speculative_stats = OpenAiSpeculativeStats {
                adaptive_window_start: adaptive_window,
                adaptive_window_final: adaptive_window,
                adaptive_window_max: max_speculative_window,
                adaptive_window_min: if request.draft.is_some() {
                    adaptive_window
                } else {
                    0
                },
                adaptive_window_max_seen: adaptive_window,
                adaptive_window_enabled: request.adaptive_speculative_window,
                ..OpenAiSpeculativeStats::default()
            };
            let mut draft_guard = match request.draft.as_ref() {
                Some(draft) if request.speculative_window > 0 => {
                    let draft_reset_timer = PhaseTimer::start();
                    let mut draft = draft
                        .lock()
                        .map_err(|_| OpenAiError::backend("draft model lock poisoned"))?;
                    draft
                        .reset_to_context(&context_tokens)
                        .map_err(openai_backend_error)?;
                    speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                    let mut attrs = self.openai_attrs(request.ids);
                    attrs.insert(
                        "llama_stage.draft_model_path".to_string(),
                        json!(draft.path.display().to_string()),
                    );
                    attrs.insert(
                        "llama_stage.speculative_window".to_string(),
                        json!(draft.window),
                    );
                    attrs.insert(
                        "llama_stage.adaptive_speculative_window".to_string(),
                        json!(request.adaptive_speculative_window),
                    );
                    self.emit_openai_phase("stage.openai_draft_reset", draft_reset_timer, attrs);
                    Some(draft)
                }
                _ => None,
            };
            for decode_step in decoded_tokens as u32..request.max_tokens {
                if fused_reached_stop {
                    break;
                }
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let token_timer = PhaseTimer::start();
                if draft_guard.is_some() {
                    let remaining = request.max_tokens as usize - decoded_tokens;
                    if remaining == 0 {
                        break;
                    }
                    let mut proposal_source = "none";
                    let proposal_limit = remaining.min(adaptive_window);
                    let propose_timer = PhaseTimer::start();
                    let mut draft_tokens = Vec::new();
                    if draft_tokens.is_empty() {
                        if let Some(draft) = draft_guard.as_deref_mut() {
                            let proposal_limit = proposal_limit.min(draft.window);
                            draft_tokens = draft
                                .propose(current, proposal_limit)
                                .map_err(openai_backend_error)?;
                            if !draft_tokens.is_empty() {
                                proposal_source = "draft-model";
                            }
                        }
                    }
                    let draft_propose_ms = propose_timer.elapsed_ms();
                    speculative_stats.draft_propose_ms += draft_propose_ms;
                    if !draft_tokens.is_empty() {
                        let verify_inputs = verify_inputs_for_proposals(current, &draft_tokens);
                        let message = embedded_verify_message(
                            request.wire_dtype,
                            VerifySpanMessageArgs {
                                request_id,
                                session_id,
                                prompt_token_count: request.prompt_token_ids.len(),
                                pos_start: prefill_token_count + decoded_tokens,
                                decode_step: decoded_tokens,
                                tokens: &verify_inputs,
                                checkpoint: true,
                            },
                        )?;
                        let verify = self.execute_embedded_stage_message(
                            &request,
                            downstream,
                            &session_key,
                            &message,
                            &verify_inputs,
                            WireReplyKind::PredictedTokens,
                        )?;
                        speculative_stats.windows += 1;
                        speculative_stats.draft_tokens += draft_tokens.len();
                        speculative_stats.primary_verify_requests += 1;
                        speculative_stats.primary_verify_tokens += verify_inputs.len();
                        speculative_stats.primary_verify_elapsed_ms += verify.elapsed_ms;
                        speculative_stats.primary_verify_stage0_compute_ms +=
                            verify.stats.stage0_compute_ms;
                        speculative_stats.primary_verify_runtime_lock_wait_ms +=
                            verify.stats.runtime_lock_wait_ms;
                        speculative_stats.primary_verify_runtime_lock_hold_ms +=
                            verify.stats.runtime_lock_hold_ms;
                        speculative_stats.primary_verify_activation_encode_ms +=
                            verify.stats.activation_encode_ms;
                        speculative_stats.primary_verify_forward_write_ms +=
                            verify.stats.forward_write_ms;
                        speculative_stats.primary_verify_downstream_wait_ms +=
                            verify.stats.downstream_wait_ms;
                        speculative_stats.primary_verify_output_activation_bytes =
                            speculative_stats
                                .primary_verify_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                        speculative_stats.primary_verify_forward_activation_bytes =
                            speculative_stats
                                .primary_verify_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                        decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                        decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                        decode_runtime_lock_wait_max_ms =
                            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
                        decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                        decode_runtime_lock_hold_max_ms =
                            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
                        decode_runtime_lock_acquires += 1;
                        decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
                        decode_output_activation_bytes = decode_output_activation_bytes
                            .saturating_add(verify.stats.output_activation_bytes);
                        decode_forward_activation_bytes = decode_forward_activation_bytes
                            .saturating_add(verify.stats.forward_activation_bytes);
                        decode_forward_write_ms += verify.stats.forward_write_ms;
                        decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                        speculative_stats.checkpoint_ms +=
                            us_to_ms(verify.reply.stats.checkpoint_total_us);
                        let decision = classify_verify_span(
                            &draft_tokens,
                            &verify.reply.predicted_tokens,
                            decoded_tokens,
                            request.max_tokens as usize,
                            |token| token_is_eog_with_runtime(&self.runtime, token),
                        )?;
                        speculative_stats.observe_verify_decision(
                            decision,
                            &mut adaptive_window,
                            request.adaptive_speculative_window,
                            max_speculative_window,
                        );
                        let mut commit_tokens =
                            verify.reply.predicted_tokens[..decision.commit_count].to_vec();
                        if decision.requires_repair() {
                            speculative_stats.recovery_restores += 1;
                            let restore = self.restore_embedded_stage_session(
                                &request,
                                downstream,
                                &session_key,
                                request_id,
                                session_id,
                            )?;
                            speculative_stats.recovery_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_local_ms += restore.local_ms;
                            speculative_stats.recovery_restore_downstream_write_ms +=
                                restore.downstream_write_ms;
                            speculative_stats.recovery_restore_downstream_wait_ms +=
                                restore.downstream_wait_ms;
                            let repair_input_count = decision
                                .repair_input_count
                                .ok_or_else(|| OpenAiError::backend("missing repair count"))?;
                            if repair_input_count == 1 {
                                let repair_message = embedded_decode_message(
                                    request.wire_dtype,
                                    DecodeMessageArgs {
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        current,
                                        sampling: wire_sampling.clone(),
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    &[current],
                                    WireReplyKind::PredictedToken,
                                )?;
                                commit_tokens = vec![repair.reply.predicted];
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_decode_repairs += 1;
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_decode_elapsed_ms += repair.elapsed_ms;
                            } else {
                                let repair_inputs = &verify_inputs[..repair_input_count];
                                let repair_message = embedded_verify_message(
                                    request.wire_dtype,
                                    VerifySpanMessageArgs {
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        tokens: repair_inputs,
                                        checkpoint: false,
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    repair_inputs,
                                    WireReplyKind::PredictedTokens,
                                )?;
                                commit_tokens = repaired_commit_tokens(
                                    &draft_tokens,
                                    decision.accepted_before_reject,
                                    repair_input_count,
                                    &repair.reply.predicted_tokens,
                                )?;
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_reverify_tokens += repair_inputs.len();
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_reverify_elapsed_ms += repair.elapsed_ms;
                            }
                        }

                        let mut reached_stop = false;
                        for token in commit_tokens {
                            current = token;
                            decoded_tokens += 1;
                            context_tokens.push(current);
                            if on_token(current)? == TokenControl::Stop {
                                reached_stop = true;
                            }
                            if reached_stop || decoded_tokens >= request.max_tokens as usize {
                                break;
                            }
                        }
                        speculative_stats.adaptive_window_final = adaptive_window;
                        if proposal_source == "draft-model" && (decision.rejected() || reached_stop)
                        {
                            let draft_reset_timer = PhaseTimer::start();
                            if let Some(draft) = draft_guard.as_deref_mut() {
                                draft
                                    .reset_to_context(&context_tokens)
                                    .map_err(openai_backend_error)?;
                                speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                            }
                        }
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs
                            .insert("llama_stage.decode_step".to_string(), json!(decode_step));
                        token_attrs
                            .insert("llama_stage.message_kind".to_string(), json!("VerifySpan"));
                        token_attrs.insert(
                            "llama_stage.spec.windows".to_string(),
                            json!(speculative_stats.windows),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposed".to_string(),
                            json!(draft_tokens.len()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.accepted".to_string(),
                            json!(decision.accepted_before_reject),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.rejected".to_string(),
                            json!(decision.rejected()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.draft_propose_ms".to_string(),
                            json!(draft_propose_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_source".to_string(),
                            json!(proposal_source),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_limit".to_string(),
                            json!(proposal_limit),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(verify.stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(verify.stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(verify.stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(verify.stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(verify.stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(verify.stats.downstream_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(verify.stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(verify.stats.forward_activation_bytes),
                        );
                        self.emit_openai_phase(
                            "stage.openai_decode_verify_window",
                            token_timer,
                            token_attrs,
                        );
                        if reached_stop {
                            break;
                        }
                        continue;
                    }
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
                let stage0_timer = PhaseTimer::start();
                let token_runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms;
                let output = {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    let lock_wait_ms = lock_timer.elapsed_ms();
                    token_runtime_lock_wait_ms = lock_wait_ms;
                    decode_runtime_lock_wait_ms += lock_wait_ms;
                    decode_runtime_lock_wait_max_ms =
                        decode_runtime_lock_wait_max_ms.max(lock_wait_ms);
                    decode_runtime_lock_acquires += 1;
                    let lock_hold_timer = PhaseTimer::start();
                    decode_runtime_sessions_before.get_or_insert_with(|| runtime.session_stats());
                    let output = run_binary_stage_message(
                        &mut runtime,
                        &session_key,
                        &message,
                        &[current],
                        None,
                        false,
                    )
                    .map_err(openai_backend_error)?
                    .2;
                    decode_runtime_sessions_after = Some(runtime.session_stats());
                    token_runtime_lock_hold_ms = lock_hold_timer.elapsed_ms();
                    decode_runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                    decode_runtime_lock_hold_max_ms =
                        decode_runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                    output
                };
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                let forwarded = forwarded_stage_message_timed(
                    request.config,
                    &message,
                    &output,
                    request.wire_dtype,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                decode_forward_activation_encode_ms += forwarded.activation_encode_ms;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                if reply.kind != WireReplyKind::PredictedToken {
                    return Err(OpenAiError::backend(format!(
                        "expected predicted-token reply from downstream, got {:?}",
                        reply.kind
                    )));
                }
                current = reply.predicted;
                decoded_tokens += 1;
                context_tokens.push(current);
                let mut token_attrs = self.openai_attrs(request.ids);
                token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                token_attrs.insert(
                    "llama_stage.decode_token_phase".to_string(),
                    json!(decode_token_phase(decode_step)),
                );
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(stage0_compute_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(token_runtime_lock_wait_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(token_runtime_lock_hold_ms),
                );
                token_attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(output.payload.len()),
                );
                token_attrs.insert(
                    "llama_stage.forward_activation_bytes".to_string(),
                    json!(forwarded.message.activation.len()),
                );
                token_attrs.insert(
                    "llama_stage.activation_encode_ms".to_string(),
                    json!(forwarded.activation_encode_ms),
                );
                token_attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(forward_write_ms),
                );
                token_attrs.insert(
                    "llama_stage.downstream_wait_ms".to_string(),
                    json!(downstream_wait_ms),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeEmbd"));
                self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            let mut decode_attrs = self.openai_attrs(request.ids);
            decode_attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            decode_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(decode_stage0_compute_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(decode_runtime_lock_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(decode_runtime_lock_wait_max_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(decode_runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(decode_runtime_lock_hold_max_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(decode_runtime_lock_acquires),
            );
            if let Some(stats) = decode_runtime_sessions_before.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = decode_runtime_sessions_after.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            decode_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(decode_forward_write_ms),
            );
            decode_attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(decode_forward_activation_encode_ms),
            );
            decode_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(decode_output_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(decode_forward_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(decode_downstream_wait_ms),
            );
            speculative_stats.insert_attrs(&mut decode_attrs);
            self.emit_openai_phase("stage.openai_decode", decode_timer, decode_attrs);
            Ok(())
        })();

        let stop_result = write_stage_message(
            &mut lane.stream,
            &StageWireMessage::stop_with_identity(request.wire_dtype, request_id, session_id),
            request.wire_dtype,
        )
        .and_then(|_| recv_reply(&mut lane.stream).map(|reply| reply.kind))
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
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_key) {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset_ms".to_string(),
                    json!(drop_stats.reset_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset".to_string(),
                    json!(drop_stats.reset_session),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    &drop_stats.stats_after,
                );
                self.telemetry
                    .emit_debug("stage.openai_session_stop", attrs);
            }
        }
        let lane_id = lane.id;
        let stop_result = stop_result.map_err(openai_io_error);
        match (&result, &stop_result) {
            (Ok(_), Ok(_)) => lane_pool.return_lane(lane),
            _ => lane_pool.replace_lane(lane_id),
        }
        if result.is_ok() {
            stop_result?;
        }
        result?;
        Ok(cache_stats)
    }
}
