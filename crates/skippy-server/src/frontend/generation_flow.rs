use super::*;

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_text(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<&openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        hook_runtime: Option<tokio::runtime::Handle>,
        cancellation: Option<&openai_frontend::CancellationToken>,
        ids: OpenAiGenerationIds,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        let generation_timer = PhaseTimer::start();
        if prompt.text.is_empty() {
            return Err(OpenAiError::invalid_request(
                "request prompt/messages produced no text",
            ));
        }
        if prompt.has_media() {
            return self.generate_multimodal_text(
                prompt,
                max_tokens,
                stop,
                sampling,
                hook_request,
                hook_runtime,
                cancellation,
                ids,
                on_text_chunk,
            );
        }
        let stop_values = stop.map(|stop| stop.values()).unwrap_or_default();
        let tokenize_timer = PhaseTimer::start();
        let prompt_token_ids = self.tokenize(&prompt.text)?;
        let mut tokenize_attrs = self.openai_attrs(&ids);
        tokenize_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        tokenize_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(prompt_token_ids.len()),
        );
        self.emit_openai_phase("stage.openai_tokenize", tokenize_timer, tokenize_attrs);
        if prompt_token_ids.is_empty() {
            return Err(OpenAiError::invalid_request("prompt produced no tokens"));
        }
        let max_tokens = max_tokens.resolve(prompt_token_ids.len(), self.ctx_size)?;
        let chat_sampling_metadata = prompt.chat_parse_metadata.as_deref();

        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk);
        let cache_stats = match self.mode.clone() {
            OpenAiBackendMode::LocalRuntime => self.generate_local_tokens(
                LocalGeneration {
                    prompt_token_ids: &prompt_token_ids,
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    hook_request: hook_request.clone(),
                    hook_runtime: hook_runtime.clone(),
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
            OpenAiBackendMode::BinaryChain {
                first_stage_addr,
                wire_dtype,
                prefill_chunk_policy,
                startup_timeout_secs,
            } => self.generate_binary_chain_tokens(
                BinaryChainGeneration {
                    first_stage_addr: &first_stage_addr,
                    wire_dtype,
                    prefill_chunk_policy: &prefill_chunk_policy,
                    startup_timeout_secs,
                    prompt_token_ids: &prompt_token_ids,
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
            OpenAiBackendMode::EmbeddedStageZero {
                config,
                wire_dtype,
                prefill_chunk_policy,
                activation_width,
                downstream_wire_condition,
                prefill_reply_credit_limit,
                lane_pool,
            } => self.generate_embedded_stage_zero_tokens(
                EmbeddedStageZeroGeneration {
                    config: &config,
                    wire_dtype,
                    prefill_chunk_policy: &prefill_chunk_policy,
                    activation_width,
                    downstream_wire_condition,
                    prefill_reply_credit_limit,
                    lane_pool,
                    draft: self.draft.clone(),
                    speculative_window: self.speculative_window,
                    adaptive_speculative_window: self.adaptive_speculative_window,
                    prompt_token_ids: &prompt_token_ids,
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    hook_request,
                    hook_runtime,
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
        };

        let output = collector.finish(prompt_token_ids.len(), cache_stats)?;
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        summary_attrs.insert(
            "skippy.kv.cached_prompt_tokens".to_string(),
            json!(output.cached_prompt_tokens),
        );
        summary_attrs.insert(
            "skippy.kv.matched_prefix_tokens".to_string(),
            json!(output.matched_prefix_tokens),
        );
        summary_attrs.insert(
            "skippy.kv.suffix_prefill_tokens".to_string(),
            json!(output.suffix_prefill_tokens),
        );
        if let Some(hit_kind) = output.cache_hit_kind {
            summary_attrs.insert("skippy.kv.hit_kind".to_string(), json!(hit_kind));
        }
        summary_attrs.insert(
            "llama_stage.detokenize_ms".to_string(),
            json!(output.detokenize_ms),
        );
        summary_attrs.insert(
            "llama_stage.text_emit_ms".to_string(),
            json!(output.text_emit_ms),
        );
        summary_attrs.insert(
            "llama_stage.eog_check_ms".to_string(),
            json!(output.eog_check_ms),
        );
        self.emit_openai_summary(
            "stage.openai_generation_summary",
            generation_timer,
            summary_attrs,
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_multimodal_text(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<&openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        hook_runtime: Option<tokio::runtime::Handle>,
        cancellation: Option<&openai_frontend::CancellationToken>,
        ids: OpenAiGenerationIds,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        if let OpenAiBackendMode::EmbeddedStageZero {
            config,
            wire_dtype,
            activation_width,
            downstream_wire_condition,
            lane_pool,
            ..
        } = self.mode.clone()
        {
            if config.downstream.is_some() {
                let lane_pool = lane_pool.ok_or_else(|| {
                    OpenAiError::backend("embedded stage 0 has no downstream lane pool")
                })?;
                return self.generate_split_multimodal_text(
                    SplitMultimodalGeneration {
                        prompt,
                        max_tokens,
                        stop,
                        sampling,
                        cancellation,
                        ids,
                        config,
                        wire_dtype,
                        activation_width,
                        downstream_wire_condition,
                        lane_pool,
                    },
                    on_text_chunk,
                );
            }
        }

        match &self.mode {
            OpenAiBackendMode::LocalRuntime => {}
            OpenAiBackendMode::EmbeddedStageZero { config, .. } if config.downstream.is_none() => {}
            OpenAiBackendMode::EmbeddedStageZero { .. } | OpenAiBackendMode::BinaryChain { .. } => {
                return Err(OpenAiError::unsupported(
                    "multimodal requests require an embedded stage-0 runtime",
                ));
            }
        }

        let stop_values = stop.map(|stop| stop.values()).unwrap_or_default();
        let session_id = ids.session_label.clone();
        let prefill_timer = PhaseTimer::start();
        let (prefill, mut token_signal, mut signal_window) = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            if !runtime.has_media_projector() {
                return Err(OpenAiError::invalid_request(
                    "multimodal request requires a configured projector",
                ));
            }
            let runtime_sessions_before = runtime.session_stats();
            let lock_hold_timer = PhaseTimer::start();
            let prefill = runtime
                .prefill_media(
                    &session_id,
                    &prompt.text,
                    &prompt.media,
                    sampling.enabled.then_some(&sampling),
                )
                .map_err(openai_backend_error)?;
            let token_signal = runtime.last_token_signal(&session_id).ok();
            let signal_window = runtime.signal_window(&session_id, 16).ok();
            let runtime_sessions_after = runtime.session_stats();
            let runtime_lock_hold_ms = lock_hold_timer.elapsed_ms();
            let mut attrs = self.openai_attrs(&ids);
            attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill.token_count),
            );
            attrs.insert(
                "llama_stage.prefill_position".to_string(),
                json!(prefill.position),
            );
            attrs.insert(
                "llama_stage.media_item_count".to_string(),
                json!(prompt.media.len()),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_before",
                &runtime_sessions_before,
            );
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_after",
                &runtime_sessions_after,
            );
            self.emit_openai_phase("stage.openai_media_prefill", prefill_timer, attrs);
            (prefill, token_signal, signal_window)
        };
        let max_tokens = max_tokens.resolve(prefill.position as usize, self.ctx_size)?;

        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk);
        let result = (|| {
            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut current = prefill.first_token;
            let mut runtime_lock_wait_ms = 0.0;
            let mut runtime_lock_wait_max_ms = 0.0_f64;
            let mut runtime_lock_hold_ms = 0.0;
            let mut runtime_lock_hold_max_ms = 0.0_f64;
            let mut runtime_lock_acquires = 0usize;
            let mut runtime_sessions_before = None;
            let mut runtime_sessions_after = None;
            let mut hook_request = hook_request;
            let hook_runtime = hook_runtime;
            let mut post_prefill_hook_checked = false;
            let mut last_mid_generation_hook_at = None;

            while decoded_tokens < max_tokens as usize {
                if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
                    break;
                }
                if let Some(injected_current) = self.maybe_run_generation_hooks(
                    &session_id,
                    &mut hook_request,
                    hook_runtime.as_ref(),
                    decoded_tokens,
                    &mut post_prefill_hook_checked,
                    &mut last_mid_generation_hook_at,
                    token_signal.take(),
                    signal_window.take(),
                )? {
                    current = injected_current;
                    continue;
                }
                if collector.push_token(current)? == TokenControl::Stop {
                    decoded_tokens += 1;
                    break;
                }
                decoded_tokens += 1;
                if decoded_tokens >= max_tokens as usize {
                    break;
                }

                let token_timer = PhaseTimer::start();
                let token_runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms;
                let token_signal_next;
                let signal_window_next;
                let decode_step = decoded_tokens;
                current = {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    let lock_wait_ms = lock_timer.elapsed_ms();
                    token_runtime_lock_wait_ms = lock_wait_ms;
                    runtime_lock_wait_ms += lock_wait_ms;
                    runtime_lock_wait_max_ms = runtime_lock_wait_max_ms.max(lock_wait_ms);
                    runtime_lock_acquires += 1;
                    let hold_timer = PhaseTimer::start();
                    runtime_sessions_before.get_or_insert_with(|| runtime.session_stats());
                    let predicted = runtime
                        .decode_sampled(&session_id, current, sampling.enabled.then_some(&sampling))
                        .map_err(openai_backend_error)?;
                    token_signal_next = runtime.last_token_signal(&session_id).ok();
                    signal_window_next = runtime.signal_window(&session_id, 16).ok();
                    runtime_sessions_after = Some(runtime.session_stats());
                    token_runtime_lock_hold_ms = hold_timer.elapsed_ms();
                    runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                    runtime_lock_hold_max_ms =
                        runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                    predicted
                };
                token_signal = token_signal_next;
                signal_window = signal_window_next;
                let mut token_attrs = self.openai_attrs(&ids);
                token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(token_timer.elapsed_ms()),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(token_runtime_lock_wait_ms),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(token_runtime_lock_hold_ms),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeToken"));
                self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
            }
            let mut attrs = self.openai_attrs(&ids);
            attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(runtime_lock_wait_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(runtime_lock_hold_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            if let Some(stats) = runtime_sessions_before.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = runtime_sessions_after.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            self.emit_openai_phase("stage.openai_decode", decode_timer, attrs);
            Ok(())
        })();
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_id) {
                let mut attrs = self.openai_attrs(&ids);
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
        result?;
        collector.finish(prefill.token_count, GenerationCacheStats::default())
    }

    pub(super) fn generate_split_multimodal_text(
        &self,
        request: SplitMultimodalGeneration<'_>,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        let stop_values = request.stop.map(|stop| stop.values()).unwrap_or_default();
        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk);
        let wire_sampling = wire_sampling_config(&request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let mut lane = request.lane_pool.checkout(&request.ids)?;

        let mut prompt_tokens = 0usize;
        let result = (|| {
            let prefill_timer = PhaseTimer::start();
            let prefill = {
                let lock_timer = PhaseTimer::start();
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                let lock_wait_ms = lock_timer.elapsed_ms();
                if !runtime.has_media_projector() {
                    return Err(OpenAiError::invalid_request(
                        "multimodal request requires a configured projector",
                    ));
                }
                let runtime_sessions_before = runtime.session_stats();
                let lock_hold_timer = PhaseTimer::start();
                let prefill = runtime
                    .prefill_media_frame(&session_key, &request.prompt.text, &request.prompt.media)
                    .map_err(openai_backend_error)?;
                let runtime_sessions_after = runtime.session_stats();
                let runtime_lock_hold_ms = lock_hold_timer.elapsed_ms();
                let mut attrs = self.openai_attrs(&request.ids);
                attrs.insert(
                    "llama_stage.prefill_token_count".to_string(),
                    json!(prefill.token_count),
                );
                attrs.insert(
                    "llama_stage.prefill_position".to_string(),
                    json!(prefill.position),
                );
                attrs.insert(
                    "llama_stage.media_item_count".to_string(),
                    json!(request.prompt.media.len()),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(runtime_lock_hold_ms),
                );
                attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
                attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(prefill.output.payload.len()),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    &runtime_sessions_before,
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    &runtime_sessions_after,
                );
                self.emit_openai_phase("stage.openai_media_prefill", prefill_timer, attrs);
                prefill
            };
            prompt_tokens = prefill.token_count;
            let max_tokens = request
                .max_tokens
                .resolve(prefill.position as usize, self.ctx_size)?;

            if let Some(message) = generation_config_message(
                request.wire_dtype,
                request_id,
                session_id,
                prefill.token_count,
                wire_sampling.clone(),
                request.prompt.chat_parse_metadata.as_deref(),
            )? {
                write_stage_message_conditioned(
                    &mut lane.stream,
                    &message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
                if reply.kind != WireReplyKind::Ack {
                    return Err(OpenAiError::backend(format!(
                        "expected multimodal generation config ACK from downstream, got {:?}",
                        reply.kind
                    )));
                }
            }

            let media_chunks = if prefill.chunks.is_empty() {
                return Err(OpenAiError::backend(
                    "multimodal prefill produced no activation chunks",
                ));
            } else {
                &prefill.chunks
            };
            let prefill_forward_timer = PhaseTimer::start();
            let mut final_reply = None;
            let mut prefill_pos_start = 0usize;
            let mut forward_activation_bytes = 0usize;
            let mut activation_encode_ms = 0.0;
            let mut forward_write_ms = 0.0;
            let mut downstream_wait_ms = 0.0;
            for (chunk_index, chunk) in media_chunks.iter().enumerate() {
                let is_final_chunk = chunk_index + 1 == media_chunks.len();
                let message = multimodal_prefill_message(
                    request.wire_dtype,
                    MultimodalPrefillArgs {
                        request_id,
                        session_id,
                        prompt_token_count: prefill.token_count,
                        pos_start: prefill_pos_start,
                        token_count: chunk.token_count,
                        positions: chunk.positions.clone(),
                        sampling: is_final_chunk.then_some(wire_sampling.clone()).flatten(),
                        final_chunk: is_final_chunk,
                    },
                )?;
                let forwarded = forwarded_stage_message_timed(
                    &request.config,
                    &message,
                    &chunk.output,
                    request.wire_dtype,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut lane.stream,
                    &forwarded.message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                forward_write_ms += write_timer.elapsed_ms();
                let wait_timer = PhaseTimer::start();
                let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
                downstream_wait_ms += wait_timer.elapsed_ms();
                let expected = if is_final_chunk {
                    WireReplyKind::PredictedToken
                } else {
                    WireReplyKind::Ack
                };
                if reply.kind != expected {
                    return Err(OpenAiError::backend(format!(
                        "expected multimodal prefill {expected:?} reply from downstream chunk {chunk_index}, got {:?}",
                        reply.kind
                    )));
                }
                forward_activation_bytes += forwarded.message.activation.len();
                activation_encode_ms += forwarded.activation_encode_ms;
                if is_final_chunk {
                    final_reply = Some(reply);
                }
                prefill_pos_start = prefill_pos_start
                    .checked_add(chunk.token_count)
                    .ok_or_else(|| {
                        OpenAiError::backend("multimodal prefill token offset overflow")
                    })?;
            }
            let reply = final_reply.ok_or_else(|| {
                OpenAiError::backend("multimodal prefill produced no predicted token")
            })?;
            let mut attrs = self.openai_attrs(&request.ids);
            attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(forward_activation_bytes),
            );
            attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(activation_encode_ms),
            );
            attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(forward_write_ms),
            );
            attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(downstream_wait_ms),
            );
            self.emit_openai_phase(
                "stage.openai_media_prefill_forward",
                prefill_forward_timer,
                attrs,
            );

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut current = reply.predicted;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_downstream_wait_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;

            while decoded_tokens < max_tokens as usize {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                if collector.push_token(current)? == TokenControl::Stop {
                    decoded_tokens += 1;
                    break;
                }
                decoded_tokens += 1;
                if decoded_tokens >= max_tokens as usize {
                    break;
                }

                let decode_input_index = decoded_tokens - 1;
                let message = embedded_decode_message(
                    request.wire_dtype,
                    DecodeMessageArgs {
                        request_id,
                        session_id,
                        prompt_token_count: prefill.token_count,
                        pos_start: prefill.token_count + decode_input_index,
                        decode_step: decode_input_index,
                        current,
                        sampling: wire_sampling.clone(),
                    },
                )?;
                let token_timer = PhaseTimer::start();
                let stage0_timer = PhaseTimer::start();
                let output = {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    let lock_wait_ms = lock_timer.elapsed_ms();
                    decode_runtime_lock_wait_ms += lock_wait_ms;
                    decode_runtime_lock_acquires += 1;
                    let hold_timer = PhaseTimer::start();
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
                    decode_runtime_lock_hold_ms += hold_timer.elapsed_ms();
                    output
                };
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                let forwarded = forwarded_stage_message_timed(
                    &request.config,
                    &message,
                    &output,
                    request.wire_dtype,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut lane.stream,
                    &forwarded.message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                if reply.kind != WireReplyKind::PredictedToken {
                    return Err(OpenAiError::backend(format!(
                        "expected multimodal decode predicted-token reply from downstream, got {:?}",
                        reply.kind
                    )));
                }
                current = reply.predicted;
                let mut token_attrs = self.openai_attrs(&request.ids);
                token_attrs.insert(
                    "llama_stage.decode_step".to_string(),
                    json!(decode_input_index),
                );
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(stage0_compute_ms),
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
            }

            let mut decode_attrs = self.openai_attrs(&request.ids);
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
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(decode_runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(decode_runtime_lock_acquires),
            );
            decode_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(decode_forward_write_ms),
            );
            decode_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(decode_downstream_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(decode_output_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(decode_forward_activation_bytes),
            );
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
                let mut attrs = self.openai_attrs(&request.ids);
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
            (Ok(_), Ok(_)) => request.lane_pool.return_lane(lane),
            _ => request.lane_pool.replace_lane(lane_id),
        }
        if result.is_ok() {
            stop_result?;
        }
        result?;
        collector.finish(prompt_tokens, GenerationCacheStats::default())
    }
}
