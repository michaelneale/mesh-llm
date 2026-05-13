use super::*;

#[async_trait]
impl OpenAiBackend for StageOpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let ids = OpenAiGenerationIds::new(OpenAiCacheHints::from_chat_request(&request));
        let request_timer = PhaseTimer::start();
        self.apply_before_chat_hooks(&mut request).await?;
        self.ensure_model(&request.model)?;
        ensure_chat_runtime_features_supported(&request)?;
        let sampling = chat_sampling_config(&request)?;
        let template_options = chat_template_options(&request)?;
        let template_timer = PhaseTimer::start();
        let prompt = self.prepare_chat_prompt(&request, template_options)?;
        let mut template_attrs = self.openai_attrs(&ids);
        template_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        template_attrs.insert(
            "llama_stage.chat_message_count".to_string(),
            json!(request.messages.len()),
        );
        template_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        template_attrs.insert(
            "llama_stage.media_item_count".to_string(),
            json!(prompt.media.len()),
        );
        self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
        let max_tokens = GenerationTokenLimit::from_request(
            request.effective_max_tokens(),
            self.default_max_tokens,
        );
        let chat_parse_metadata = prompt.chat_parse_metadata.clone();
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                Some(request.clone()),
                ids.clone(),
            )
            .await?;
        let response_timer = PhaseTimer::start();
        let parsed_tool_calls =
            self.parse_tool_call_output(&output.text, &request, chat_parse_metadata.as_deref())?;
        let response =
            chat_response_from_generated_text(request.model.clone(), &output, parsed_tool_calls);
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let ids = OpenAiGenerationIds::new(OpenAiCacheHints::from_chat_request(&request));
        self.apply_before_chat_hooks(&mut request).await?;
        self.ensure_model(&request.model)?;
        ensure_chat_runtime_features_supported(&request)?;
        let sampling = chat_sampling_config(&request)?;
        let include_usage = request.include_usage();
        let template_options = chat_template_options(&request)?;
        let template_timer = PhaseTimer::start();
        let prompt = self.prepare_chat_prompt(&request, template_options)?;
        let mut template_attrs = self.openai_attrs(&ids);
        template_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion_stream"),
        );
        template_attrs.insert(
            "llama_stage.chat_message_count".to_string(),
            json!(request.messages.len()),
        );
        template_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        template_attrs.insert(
            "llama_stage.media_item_count".to_string(),
            json!(prompt.media.len()),
        );
        self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
        let max_tokens = GenerationTokenLimit::from_request(
            request.effective_max_tokens(),
            self.default_max_tokens,
        );
        let model = request.model.clone();
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                Some(request.clone()),
                context,
                ids,
            )
            .await?;
        Ok(Box::pin(stream.map(move |event| {
            generation_event_to_chat_chunk(event, &model)
        })))
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        let ids = OpenAiGenerationIds::new(OpenAiCacheHints::from_completion_request(&request));
        let request_timer = PhaseTimer::start();
        self.ensure_model(&request.model)?;
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                None,
                ids.clone(),
            )
            .await?;
        let response_timer = PhaseTimer::start();
        let response = CompletionResponse::new_with_reason(
            request.model,
            output.text.clone(),
            output.usage(),
            output.finish_reason,
        );
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        let ids = OpenAiGenerationIds::new(OpenAiCacheHints::from_completion_request(&request));
        self.ensure_model(&request.model)?;
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let include_usage = request.include_usage();
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let model = request.model.clone();
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion_stream"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                None,
                context,
                ids,
            )
            .await?;
        Ok(Box::pin(stream.map(move |event| {
            generation_event_to_completion_chunk(event, &model)
        })))
    }
}

impl StageOpenAiBackend {
    async fn acquire_generation_permit(&self) -> OpenAiResult<OwnedSemaphorePermit> {
        acquire_generation_permit_with_queue(
            self.generation_limit.clone(),
            self.generation_queue_depth.clone(),
            self.generation_queue_limit,
            GENERATION_ADMISSION_TIMEOUT,
        )
        .await
    }

    pub(super) fn openai_attrs(&self, ids: &OpenAiGenerationIds) -> BTreeMap<String, Value> {
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            attr_key::SESSION_ID.to_string(),
            json!(ids.session_id_string()),
        );
        attrs.insert(
            attr_key::REQUEST_ID.to_string(),
            json!(ids.request_id_string()),
        );
        attrs.insert(
            "llama_stage.openai_backend".to_string(),
            json!(self.mode.label()),
        );
        if let Some(cache_key) = ids.cache.prompt_cache_key.as_deref() {
            attrs.insert("openai.prompt_cache_key".to_string(), json!(cache_key));
        }
        if let Some(retention) = ids.cache.prompt_cache_retention.as_deref() {
            attrs.insert(
                "openai.prompt_cache_retention".to_string(),
                json!(retention),
            );
        }
        attrs
    }

    pub(super) fn insert_runtime_session_stats(
        attrs: &mut BTreeMap<String, Value>,
        prefix: &str,
        stats: &RuntimeSessionStats,
    ) {
        attrs.insert(
            format!("{prefix}.active_sessions"),
            json!(stats.active_sessions),
        );
        attrs.insert(
            format!("{prefix}.idle_sessions"),
            json!(stats.idle_sessions),
        );
        attrs.insert(
            format!("{prefix}.idle_resident_prefixes"),
            json!(stats.idle_resident_prefixes),
        );
        attrs.insert(
            format!("{prefix}.tracked_token_counts"),
            json!(stats.tracked_token_counts),
        );
        attrs.insert(format!("{prefix}.checkpoints"), json!(stats.checkpoints));
    }

    pub(super) fn emit_openai_phase(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_debug_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    pub(super) fn emit_openai_summary(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    pub(super) fn ensure_model(&self, requested: &str) -> OpenAiResult<()> {
        ensure_requested_model(&self.model_id, requested)
    }

    async fn apply_before_chat_hooks(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<()> {
        let Some(hooks) = self.hook_policy.as_ref() else {
            return Ok(());
        };
        if !chat_mesh_hooks_enabled(request) {
            return Ok(());
        }
        let outcome = hooks.before_chat_completion(request).await?;
        apply_chat_hook_outcome(request, &outcome);
        Ok(())
    }

    async fn run_generation(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GeneratedText> {
        let admit_timer = PhaseTimer::start();
        let permit = self.acquire_generation_permit().await?;
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let backend = self.clone();
        let hook_runtime = Some(tokio::runtime::Handle::current());
        task::spawn_blocking(move || {
            let _permit = permit;
            backend.generate_text(
                prompt,
                max_tokens,
                stop.as_ref(),
                sampling,
                hook_request,
                hook_runtime,
                None,
                ids,
                |_| Ok(()),
            )
        })
        .await
        .map_err(|error| OpenAiError::backend(format!("generation task failed: {error}")))?
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generation_stream(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        include_usage: bool,
        hook_request: Option<ChatCompletionRequest>,
        context: OpenAiRequestContext,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GenerationStream> {
        let admit_timer = PhaseTimer::start();
        let permit = self.acquire_generation_permit().await?;
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let backend = self.clone();
        let tool_call_stream = hook_request.as_ref().is_some_and(tool_calls_requested)
            && prompt.chat_parse_metadata.is_some();
        let chat_parse_metadata = prompt.chat_parse_metadata.clone();
        let (tx, rx) = mpsc::channel(16);
        let hook_runtime = Some(tokio::runtime::Handle::current());
        let stream_tool_request = hook_request.clone();
        task::spawn_blocking(move || {
            let _permit = permit;
            let result = backend.generate_text(
                prompt,
                max_tokens,
                stop.as_ref(),
                sampling,
                hook_request,
                hook_runtime,
                Some(&context.cancellation_token()),
                ids,
                |chunk| {
                    if tool_call_stream {
                        return Ok(());
                    }
                    if context.is_cancelled() {
                        return Err(OpenAiError::backend("stream receiver cancelled"));
                    }
                    tx.blocking_send(Ok(GenerationStreamEvent::Delta(chunk.to_string())))
                        .map_err(|_| {
                            context.cancel();
                            OpenAiError::backend("stream receiver dropped")
                        })
                },
            );
            if context.is_cancelled() {
                return;
            }
            match result {
                Ok(output) => {
                    if tool_call_stream {
                        if let (Some(request), Some(metadata)) =
                            (stream_tool_request.as_ref(), chat_parse_metadata.as_deref())
                        {
                            match backend.parse_tool_call_output(
                                &output.text,
                                request,
                                Some(metadata),
                            ) {
                                Ok(Some(tool_output)) => {
                                    if tx
                                        .blocking_send(Ok(GenerationStreamEvent::ToolCalls(
                                            tool_output.tool_calls,
                                        )))
                                        .is_err()
                                    {
                                        context.cancel();
                                        return;
                                    }
                                    if include_usage
                                        && tx
                                            .blocking_send(Ok(GenerationStreamEvent::Usage(
                                                output.usage(),
                                            )))
                                            .is_err()
                                    {
                                        context.cancel();
                                        return;
                                    }
                                    let _ = tx.blocking_send(Ok(GenerationStreamEvent::Done(
                                        FinishReason::ToolCalls,
                                    )));
                                    return;
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    let _ = tx.blocking_send(Err(error));
                                    return;
                                }
                            }
                        }
                        if !output.text.is_empty()
                            && tx
                                .blocking_send(Ok(GenerationStreamEvent::Delta(
                                    output.text.clone(),
                                )))
                                .is_err()
                        {
                            context.cancel();
                            return;
                        }
                    }
                    if include_usage
                        && tx
                            .blocking_send(Ok(GenerationStreamEvent::Usage(output.usage())))
                            .is_err()
                    {
                        context.cancel();
                        return;
                    }
                    let _ = tx.blocking_send(Ok(GenerationStreamEvent::Done(output.finish_reason)));
                }
                Err(error) => {
                    let _ = tx.blocking_send(Err(error));
                }
            }
        });
        Ok(Box::pin(stream::unfold(rx, |mut rx| async {
            rx.recv().await.map(|item| (item, rx))
        })))
    }
}
