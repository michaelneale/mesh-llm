fn format_prompt_for_model(
    tokenizer: &StageModel,
    chat_template_model: Option<&StageModel>,
    prompt: &str,
    args: &BinaryReplArgs,
) -> Result<String> {
    if args.raw_prompt {
        return Ok(prompt.to_string());
    }

    format_messages_for_model(
        tokenizer,
        chat_template_model,
        &[ChatTemplateMessage::new("user", prompt)],
        args,
    )
}

fn format_messages_for_model(
    tokenizer: &StageModel,
    chat_template_model: Option<&StageModel>,
    messages: &[ChatTemplateMessage],
    args: &BinaryReplArgs,
) -> Result<String> {
    format_messages_for_model_with_options(tokenizer, chat_template_model, messages, args, true)
}

fn format_messages_for_model_with_options(
    tokenizer: &StageModel,
    chat_template_model: Option<&StageModel>,
    messages: &[ChatTemplateMessage],
    args: &BinaryReplArgs,
    add_assistant: bool,
) -> Result<String> {
    if args.raw_prompt {
        bail!("raw prompt mode does not support chat message formatting");
    }
    let enable_thinking = prompt_thinking_override(args)?;

    chat_template_model
        .unwrap_or(tokenizer)
        .apply_chat_template_with_options(
            messages,
            ChatTemplateOptions {
                add_assistant,
                enable_thinking,
            },
        )
        .with_context(|| {
            let mode = match enable_thinking {
                Some(true) => "enabled",
                Some(false) => "disabled",
                None => "default",
            };
            format!("apply chat template with thinking {mode}")
        })
}

fn prompt_thinking_override(args: &BinaryReplArgs) -> Result<Option<bool>> {
    let reasoning = prompt_openai_reasoning_config(args.no_think, args.thinking_token_budget)?;
    let extra = BTreeMap::new();
    normalize_reasoning_template_options(reasoning.as_ref(), None, &extra)
        .map(|options| options.enable_thinking)
        .map_err(|error| {
            anyhow!(
                "normalize OpenAI reasoning controls: {}",
                error.body().error.message
            )
        })
}

fn prompt_openai_reasoning_config(
    no_think: bool,
    thinking_token_budget: Option<usize>,
) -> Result<Option<ReasoningConfig>> {
    if no_think || thinking_token_budget == Some(0) {
        return Ok(Some(ReasoningConfig {
            enabled: Some(false),
            max_tokens: Some(0),
            ..ReasoningConfig::default()
        }));
    }

    let Some(budget) = thinking_token_budget else {
        return Ok(None);
    };
    Ok(Some(ReasoningConfig {
        enabled: Some(true),
        max_tokens: Some(
            budget
                .try_into()
                .context("--thinking-token-budget exceeds u32 range")?,
        ),
        ..ReasoningConfig::default()
    }))
}

fn effective_prompt_max_new_tokens(
    configured: usize,
    ctx_size: u32,
    prompt_token_count: usize,
) -> Result<usize> {
    if configured > 0 {
        return Ok(configured);
    }
    let ctx_size = usize::try_from(ctx_size).context("ctx_size exceeds usize")?;
    if prompt_token_count >= ctx_size {
        bail!("prompt tokens exceed context window (n_ctx={ctx_size})");
    }
    Ok(ctx_size - prompt_token_count)
}

fn format_prompt_max_new_tokens(value: usize) -> String {
    if value == DEFAULT_MESH_PROMPT_MAX_NEW_TOKENS {
        "context-budget".to_string()
    } else {
        value.to_string()
    }
}
