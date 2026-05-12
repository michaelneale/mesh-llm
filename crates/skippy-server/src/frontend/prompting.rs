use super::*;

impl StageOpenAiBackend {
    pub(super) fn prepare_chat_prompt(
        &self,
        request: &ChatCompletionRequest,
        options: ChatTemplateOptions,
    ) -> OpenAiResult<PreparedGenerationPrompt> {
        let marker = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime.media_marker()
        };
        let mut media = Vec::new();
        let template_messages = request
            .messages
            .iter()
            .map(|message| chat_message_generation_value(message, &marker, &mut media))
            .collect::<OpenAiResult<Vec<_>>>()?;
        let messages_json = serde_json::to_string(&template_messages).map_err(|error| {
            OpenAiError::invalid_request(format!("serialize messages: {error}"))
        })?;
        let tools_json = request
            .tools
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| OpenAiError::invalid_request(format!("serialize tools: {error}")))?;
        let tool_choice_json = request
            .tool_choice
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| {
                OpenAiError::invalid_request(format!("serialize tool_choice: {error}"))
            })?;
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let result = runtime
            .model
            .apply_chat_template_json(
                &messages_json,
                ChatTemplateJsonOptions {
                    add_assistant: options.add_assistant,
                    enable_thinking: options.enable_thinking,
                    tools_json,
                    tool_choice_json,
                    parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
                },
            )
            .map_err(openai_backend_error)?;
        Ok(PreparedGenerationPrompt {
            text: result.prompt,
            media,
            chat_parse_metadata: Some(result.metadata_json),
        })
    }

    pub(super) fn parse_tool_call_output(
        &self,
        text: &str,
        request: &ChatCompletionRequest,
        metadata: Option<&str>,
    ) -> OpenAiResult<Option<ParsedToolCalls>> {
        if !tool_calls_requested(request) {
            return Ok(None);
        }
        let Some(metadata) = metadata else {
            return Ok(None);
        };
        let parsed_json = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .model
                .parse_chat_response_json(text, metadata, false)
                .map_err(openai_backend_error)?
        };
        Ok(parsed_tool_calls_from_message_json(&parsed_json, request))
    }

    pub(super) fn tokenize(&self, prompt: &str) -> OpenAiResult<Vec<i32>> {
        self.tokenize_with_options(prompt, true)
    }

    pub(super) fn tokenize_continuation(&self, text: &str) -> OpenAiResult<Vec<i32>> {
        self.tokenize_with_options(text, false)
    }

    pub(super) fn tokenize_with_options(
        &self,
        text: &str,
        add_special: bool,
    ) -> OpenAiResult<Vec<i32>> {
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        runtime
            .model
            .tokenize(text, add_special)
            .map_err(openai_backend_error)
    }

    pub(super) fn inject_hook_text_into_session(
        &self,
        session_id: &str,
        text: &str,
    ) -> OpenAiResult<Option<i32>> {
        let token_ids = self.tokenize_continuation(text)?;
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() > 1 {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .prefill(session_id, &token_ids[..token_ids.len() - 1])
                .map_err(openai_backend_error)?;
        }
        Ok(token_ids.last().copied())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn maybe_run_generation_hooks(
        &self,
        session_id: &str,
        hook_request: &mut Option<ChatCompletionRequest>,
        hook_runtime: Option<&tokio::runtime::Handle>,
        decoded_tokens: usize,
        post_prefill_hook_checked: &mut bool,
        last_mid_generation_hook_at: &mut Option<usize>,
        token_signal: Option<TokenSignal>,
        signal_window: Option<GenerationSignalWindow>,
    ) -> OpenAiResult<Option<i32>> {
        let Some(hooks) = self.hook_policy.as_ref() else {
            return Ok(None);
        };
        let Some(handle) = hook_runtime else {
            return Ok(None);
        };
        let Some(request) = hook_request.as_mut() else {
            return Ok(None);
        };
        if !chat_mesh_hooks_enabled(request) {
            return Ok(None);
        }

        if !*post_prefill_hook_checked {
            *post_prefill_hook_checked = true;
            if let Some(signal) = token_signal {
                let signals = PrefillHookSignals {
                    first_token_entropy: f64::from(signal.entropy),
                    first_token_margin: f64::from(signal.margin),
                };
                let outcome = handle.block_on(hooks.after_prefill(request, signals))?;
                apply_chat_hook_outcome(request, &outcome);
                if let Some(text) = hook_injected_text(&outcome) {
                    return self.inject_hook_text_into_session(session_id, &text);
                }
            }
        }

        let Some(window) = signal_window else {
            return Ok(None);
        };
        if !mid_generation_window_should_fire(decoded_tokens, last_mid_generation_hook_at, &window)
        {
            return Ok(None);
        }

        let signals = GenerationHookSignals {
            n_decoded: i64::try_from(decoded_tokens).unwrap_or(i64::MAX),
            window_tokens: window.token_count,
            mean_entropy: f64::from(window.mean_entropy),
            max_entropy: f64::from(window.max_entropy),
            mean_margin: f64::from(window.mean_margin),
            min_margin: f64::from(window.min_margin),
            high_entropy_count: window.high_entropy_count,
            repetition_count: window.repetition_count,
        };
        let outcome = handle.block_on(hooks.mid_generation(request, signals))?;
        *last_mid_generation_hook_at = Some(decoded_tokens);
        apply_chat_hook_outcome(request, &outcome);
        if let Some(text) = hook_injected_text(&outcome) {
            return self.inject_hook_text_into_session(session_id, &text);
        }
        Ok(None)
    }

    pub(super) fn generation_hooks_active(
        &self,
        hook_request: &Option<ChatCompletionRequest>,
        hook_runtime: Option<&tokio::runtime::Handle>,
    ) -> bool {
        self.hook_policy.is_some()
            && hook_runtime.is_some()
            && hook_request.as_ref().is_some_and(chat_mesh_hooks_enabled)
    }
}
