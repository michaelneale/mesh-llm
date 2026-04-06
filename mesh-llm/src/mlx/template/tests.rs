use super::*;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct HfTemplateFixture {
    repo: String,
    source_file: String,
    expect_hf_render: bool,
    family: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    pad_token: Option<String>,
    unk_token: Option<String>,
    template: String,
}

fn hf_template_corpus() -> Vec<HfTemplateFixture> {
    serde_json::from_str(include_str!("../testdata/hf_template_corpus.json"))
        .expect("valid HF template corpus")
}

fn fixture_config(family: &str) -> Value {
    match family {
        "llama" => json!({"model_type":"llama","architectures":["LlamaForCausalLM"]}),
        "qwen" | "qwen3" | "qwen3_coder_next" | "deepseek_qwen3" => {
            json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]})
        }
        "qwen3_coder_30b" => json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}),
        "gemma3" => {
            json!({"model_type":"gemma3","architectures":["Gemma3ForConditionalGeneration"]})
        }
        "mistral" => json!({"model_type":"mistral","architectures":["MistralForCausalLM"]}),
        "lfm2" => json!({"model_type":"lfm2","architectures":["LlamaForCausalLM"]}),
        "devstral" => json!({"model_type":"mistral","architectures":["MistralForCausalLM"]}),
        "glm4v" => json!({"model_type":"glm","architectures":["GlmForCausalLM"]}),
        "kimi" => json!({"model_type":"kimi","architectures":["KimiForCausalLM"]}),
        "gpt_oss" => json!({"model_type":"gpt_oss","architectures":["GptOssForCausalLM"]}),
        other => panic!("unknown fixture family: {other}"),
    }
}

fn fixture_request(family: &str) -> Value {
    match family {
        "llama" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ],
            "add_generation_prompt": true
        }),
        "qwen" => json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}],
            "add_generation_prompt": true
        }),
        "gemma3" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": [
                    {"type": "text", "text": "look "},
                    {"type": "image"},
                    {"type": "text", "text": "here"}
                ]}
            ],
            "add_generation_prompt": true
        }),
        "mistral" => json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"}
            ],
            "add_generation_prompt": true
        }),
        "lfm2" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think>\ninternal\n</think>\nhi"},
                {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image"}]}
            ],
            "keep_past_thinking": false,
            "add_generation_prompt": true
        }),
        "deepseek_qwen3" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ],
            "add_generation_prompt": true
        }),
        "qwen3" | "qwen3_coder_next" | "qwen3_coder_30b" => json!({
            "messages": [{"role": "user", "content": "hello"}],
            "add_generation_prompt": true
        }),
        "devstral" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ],
            "add_generation_prompt": true
        }),
        "glm4v" => json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}],
            "add_generation_prompt": true
        }),
        "kimi" => json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url"}]}],
            "add_generation_prompt": true
        }),
        "gpt_oss" => json!({
            "messages": [{"role": "user", "content": "hello"}],
            "builtin_tools": ["browser", "python"],
            "reasoning_effort": "medium",
            "add_generation_prompt": true
        }),
        other => panic!("unknown fixture family: {other}"),
    }
}

fn write_hf_fixture_dir(fixture: &HfTemplateFixture) -> std::path::PathBuf {
    let slug = fixture
        .repo
        .replace('/', "-")
        .replace('.', "-")
        .replace('_', "-");
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-hf-template-corpus-{}-{}",
        slug,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    match fixture.source_file.as_str() {
        "chat_template.jinja" => {
            std::fs::write(root.join("chat_template.jinja"), &fixture.template).unwrap();
        }
        "chat_template.json" => {
            std::fs::write(
                root.join("chat_template.json"),
                serde_json::to_string(&fixture.template).unwrap(),
            )
            .unwrap();
        }
        "tokenizer_config.json" => {
            std::fs::write(
                root.join("tokenizer_config.json"),
                serde_json::json!({
                    "chat_template": fixture.template,
                    "bos_token": fixture.bos_token,
                    "eos_token": fixture.eos_token,
                    "pad_token": fixture.pad_token,
                    "unk_token": fixture.unk_token,
                })
                .to_string(),
            )
            .unwrap();
            return root;
        }
        other => panic!("unknown template source: {other}"),
    }

    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": fixture.bos_token,
            "eos_token": fixture.eos_token,
            "pad_token": fixture.pad_token,
            "unk_token": fixture.unk_token,
        })
        .to_string(),
    )
    .unwrap();

    root
}

#[test]
fn normalizes_python_get_calls() {
    let template = "{{ msg.get('content') }} {{ msg.get(\"role\", \"user\") }}";
    let normalized = normalize_hf_template(template);
    assert_eq!(
        normalized,
        "{{ msg[\"content\"]|default(none) }} {{ msg[\"role\"]|default(\"user\") }}"
    );
}

#[test]
fn normalizes_tojson_keyword_arguments() {
    let template = "{{ tools | tojson(separators=(',', ':'), ensure_ascii=False) }}";
    let normalized = normalize_hf_template(template);
    assert_eq!(normalized, "{{ tools | tojson }}");
}

#[test]
fn detects_old_qwen_reasoning_tags_from_split_template() {
    let template = "{{ content | split('</think>') | last }}";
    let reasoning = detect_reasoning_template(template);
    assert!(reasoning.supports_explicit_reasoning);
    assert_eq!(
        reasoning.tagged_reasoning,
        vec![TaggedReasoningBlock {
            start: "<think>".to_string(),
            end: "</think>".to_string(),
        }]
    );
}

#[test]
fn detects_gemma4_reasoning_channel_markers() {
    let template = "{% if add_generation_prompt %}<|channel>thought{{ reasoning_content }}<channel|>{% endif %}";
    let reasoning = detect_reasoning_template(template);
    assert!(reasoning.supports_explicit_reasoning);
    assert_eq!(
        reasoning.tagged_reasoning,
        vec![TaggedReasoningBlock {
            start: "<|channel>thought".to_string(),
            end: "<channel|>".to_string(),
        }]
    );
    assert!(reasoning.default_stop_sequences.is_empty());
}

#[test]
fn detects_gemma4_turn_stop_marker() {
    let template = "{% if add_generation_prompt %}<|turn>model\n{% endif %}{% for message in messages %}<turn|>\n{% endfor %}";
    let reasoning = detect_reasoning_template(template);
    assert!(reasoning
        .default_stop_sequences
        .contains(&"<turn|>".to_string()));
}

#[test]
fn prefers_chat_template_jinja_over_tokenizer_config() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-jinja-precedence-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("chat_template.jinja"), "{{ '<jinja-template>' }}").unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{{ '<json-template>' }}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert_eq!(prompt, "<jinja-template>");
}

#[test]
fn falls_back_when_template_uses_unsupported_python_method() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-unsupported-method-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{% if messages[0].content.removeprefix('h') %}<bad>{% endif %}",
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("hello"));
}

#[test]
fn kimi_template_compiles_after_normalization() {
    let fixture = hf_template_corpus()
        .into_iter()
        .find(|fixture| fixture.repo == "mlx-community/Kimi-K2.5")
        .expect("kimi fixture exists");
    let normalized = normalize_hf_template(&fixture.template);
    validate_hf_template(&normalized).expect("normalized Kimi template should compile");
}

#[test]
fn detects_chatml_from_tokenizer_config() {
    let root =
        std::env::temp_dir().join(format!("mesh-llm-template-chatml-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    match template {
        PromptTemplate::HuggingFace {
            fallback,
            reasoning_defaults,
            ..
        } => {
            assert_eq!(
                *fallback,
                PromptTemplate::ChatMl {
                    default_system_prompt: Some("You are a helpful assistant.".to_string())
                }
            );
            assert_eq!(reasoning_defaults, ReasoningDefaults::default());
        }
        other => panic!("expected huggingface template, got {other:?}"),
    }
}

#[test]
fn qwen3_templates_default_enable_thinking_to_false() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-qwen3-thinking-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(
        &root,
        &serde_json::json!({"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}),
    );
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

    assert_eq!(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

#[test]
fn qwen3_templates_honor_explicit_enable_thinking_true() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-qwen3-thinking-true-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(
        &root,
        &serde_json::json!({"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}),
    );
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": true
        }))
        .unwrap();

    assert_eq!(prompt, "<|im_start|>assistant\n");
}

fn corpus_fixture(repo: &str) -> HfTemplateFixture {
    hf_template_corpus()
        .into_iter()
        .find(|fixture| fixture.repo == repo)
        .unwrap_or_else(|| panic!("missing fixture for {repo}"))
}

#[test]
fn glm_templates_default_enable_thinking_to_false() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-glm-thinking-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{%- if add_generation_prompt %}<|assistant|>{{ '/nothink' if (enable_thinking is defined and not enable_thinking) else '' }}{%- endif %}",
    )
    .unwrap();

    let template = PromptTemplate::detect(
        &root,
        &serde_json::json!({"model_type":"glm","architectures":["GlmForCausalLM"]}),
    );
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

    assert_eq!(prompt, "<|assistant|>/nothink");
}

#[test]
fn kimi_templates_map_enable_thinking_to_thinking() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-kimi-thinking-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{%- if add_generation_prompt %}{%- if thinking is defined and thinking is false -%}<think></think>{%- else -%}<think>{%- endif -%}{%- endif %}",
    )
    .unwrap();

    let template = PromptTemplate::detect(
        &root,
        &serde_json::json!({"model_type":"kimi","architectures":["KimiForCausalLM"]}),
    );
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert_eq!(prompt, "<think></think>");

    let explicit_prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": true}
        }))
        .unwrap();
    assert_eq!(explicit_prompt, "<think>");
}

#[test]
fn gpt_oss_templates_map_enable_thinking_to_reasoning_effort() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-gpt-oss-thinking-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{{ reasoning_effort | default('missing') }}",
    )
    .unwrap();

    let template = PromptTemplate::detect(
        &root,
        &serde_json::json!({"model_type":"gpt_oss","architectures":["GptOssForCausalLM"]}),
    );
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert_eq!(prompt, "low");

    let explicit_prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": true
        }))
        .unwrap();
    assert_eq!(explicit_prompt, "medium");
}

#[test]
fn lfm2_templates_map_enable_thinking_to_keep_past_thinking() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-lfm2-thinking-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{{ 'keep' if keep_past_thinking | default(false) else 'strip' }}",
    )
    .unwrap();

    let template = PromptTemplate::detect(
        &root,
        &serde_json::json!({"model_type":"lfm2","architectures":["LlamaForCausalLM"]}),
    );
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert_eq!(prompt, "strip");

    let explicit_prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": true
        }))
        .unwrap();
    assert_eq!(explicit_prompt, "keep");
}

#[test]
fn glm_fixture_defaults_to_nothink() {
    let fixture = corpus_fixture("lmstudio-community/GLM-4.6V-Flash-MLX-4bit");
    let root = write_hf_fixture_dir(&fixture);
    let template = PromptTemplate::detect(&root, &fixture_config(&fixture.family));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}],
            "add_generation_prompt": true
        }))
        .unwrap();

    assert!(prompt.contains("/nothink"));
}

#[test]
fn kimi_fixture_defaults_to_no_thinking() {
    let fixture = corpus_fixture("mlx-community/Kimi-K2.5");
    let root = write_hf_fixture_dir(&fixture);
    let template = PromptTemplate::detect(&root, &fixture_config(&fixture.family));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "add_generation_prompt": true
        }))
        .unwrap();

    assert!(prompt.contains("<think></think>"));
    assert!(!prompt.contains("<|im_assistant|>assistant<|im_middle|>\n  <think>\n"));
}

#[test]
fn gpt_oss_fixture_defaults_to_low_reasoning_effort() {
    let fixture = corpus_fixture("mlx-community/gpt-oss-20b-MXFP4-Q8");
    let root = write_hf_fixture_dir(&fixture);
    let template = PromptTemplate::detect(&root, &fixture_config(&fixture.family));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "builtin_tools": ["browser", "python"],
            "add_generation_prompt": true
        }))
        .unwrap();

    assert!(prompt.contains("Reasoning: low"));
}

#[test]
fn lfm2_fixture_defaults_to_stripping_past_thinking() {
    let fixture = corpus_fixture("lmstudio-community/LFM2-24B-A2B-MLX-4bit");
    let root = write_hf_fixture_dir(&fixture);
    let template = PromptTemplate::detect(&root, &fixture_config(&fixture.family));
    let prompt = template
        .render_request(&json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think>\ninternal\n</think>\nhi"},
                {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image"}]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "again"}
            ],
            "add_generation_prompt": true
        }))
        .unwrap();

    assert!(!prompt.contains("internal"));
    assert!(prompt.contains("hi<|im_end|>"));
}

#[test]
fn renders_llama3_prompt_from_hf_template() {
    let root =
        std::env::temp_dir().join(format!("mesh-llm-template-llama3-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<|begin_of_text|>",
            "chat_template": "{{- bos_token }}{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"llama"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert!(prompt.starts_with("<|begin_of_text|>"));
    assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"));
    assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
}

#[test]
fn llama_hf_template_does_not_enter_tool_mode_when_tools_are_absent() {
    let fixture = corpus_fixture("mlx-community/Llama-3.2-1B-Instruct-4bit");
    let root = write_hf_fixture_dir(&fixture);
    let template = PromptTemplate::detect(&root, &fixture_config(&fixture.family));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "add_generation_prompt": true
        }))
        .unwrap();

    assert!(!prompt.contains("Environment: ipython"));
    assert!(!prompt.contains("Given the following functions"));
}

#[test]
fn renders_qwen_tools_template_with_minijinja() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-qwen-tools-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "chat_template": "{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{{- messages[0]['content'] }}{%- else %}{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}{%- endif %}{{- '\\n\\n# Tools\\n\\n<tools>' }}{%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}{{- '\\n</tools><|im_end|>\\n' }}{%- endif %}{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}]
        }))
        .unwrap();

    assert!(prompt.contains("# Tools"));
    assert!(prompt.contains("\"name\":\"run\""));
    assert!(prompt.contains("<|im_start|>assistant\n"));
}

#[test]
fn qwen_prompt_parity_fixture_matches_expected_output() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-qwen-fixture-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{%- if messages[0]['role'] != 'system' -%}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{%- endif -%}{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

    assert_eq!(
        prompt,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nhello<|im_end|><|im_start|>assistant"
    );
}

#[test]
fn normalizes_missing_optional_message_fields_for_dot_access_templates() {
    let messages = json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"}
    ]);
    let normalized = normalize_hf_messages(
        "{% if message.tool_calls %}{{ message.tool_calls }}{% endif %} {{ message.name }} {{ message.tool_call_id }} {{ message.tool_responses }}",
        messages,
    );

    let array = normalized.as_array().unwrap();
    for message in array {
        let object = message.as_object().unwrap();
        assert!(object.contains_key("tool_calls"));
        assert!(object.contains_key("name"));
        assert!(object.contains_key("tool_call_id"));
        assert!(object.contains_key("tool_responses"));
    }
}

#[test]
fn does_not_inject_tool_calls_when_template_checks_key_presence() {
    let messages = json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"}
    ]);
    let normalized = normalize_hf_messages(
        "{% if 'tool_calls' in message %}{{ message.tool_calls|length }}{% endif %}",
        messages,
    );

    let array = normalized.as_array().unwrap();
    for message in array {
        let object = message.as_object().unwrap();
        assert!(!object.contains_key("tool_calls"));
    }
}

#[test]
fn qwen_hf_template_with_dot_access_does_not_fall_back_to_chatml() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-qwen-dot-access-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}{%- else %}{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}{%- endif %}{%- for message in messages %}{%- if (message.role == 'user') or (message.role == 'assistant' and not message.tool_calls) %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"}
            ],
            "add_generation_prompt": true
        }))
        .unwrap();

    assert!(prompt.starts_with("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"));
    assert!(prompt.contains("<|im_start|>assistant\nsecond<|im_end|>\n"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn llama3_prompt_parity_fixture_matches_expected_output() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-llama3-fixture-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<|begin_of_text|>",
            "chat_template": "{{- bos_token }}{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"llama"}));
    let prompt = template
        .render_request(&json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ]
        }))
        .unwrap();

    assert_eq!(
        prompt,
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    );
}

#[test]
fn gemma3_prompt_parity_fixture_matches_expected_output() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-gemma3-fixture-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<bos>",
            "chat_template": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- endif -%}\n    {{ '<end_of_turn>\\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\\n'}}\n{%- endif -%}\n"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"gemma3"}));
    let prompt = template
        .render_request(&json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": [
                    {"type": "text", "text": "look "},
                    {"type": "image"},
                    {"type": "text", "text": "here"}
                ]}
            ]
        }))
        .unwrap();

    assert_eq!(
        prompt,
        "<bos><start_of_turn>user\nBe concise.\n\nhello<end_of_turn>\n<start_of_turn>model\nhi<end_of_turn>\n<start_of_turn>user\nlook<start_of_image>here<end_of_turn>\n<start_of_turn>model\n"
    );
}

#[test]
fn heuristic_fallback_uses_gemma3_for_gemma_models() {
    let template = PromptTemplate::detect(
        Path::new("/tmp/does-not-need-to-exist"),
        &serde_json::json!({"model_type":"gemma3","architectures":["Gemma3ForConditionalGeneration"]}),
    );
    assert_eq!(template, PromptTemplate::Gemma3);
}

#[test]
fn renders_when_template_uses_strftime_now() {
    let root =
        std::env::temp_dir().join(format!("mesh-llm-template-fallback-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{{ strftime_now('%Y-%m-%d') }}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello world"}]
        }))
        .unwrap();

    assert_eq!(prompt.len(), 10);
    assert_eq!(prompt.chars().filter(|c| *c == '-').count(), 2);
}

#[test]
fn real_hf_template_corpus_behaves_as_expected() {
    for fixture in hf_template_corpus() {
        let root = write_hf_fixture_dir(&fixture);
        let config = fixture_config(&fixture.family);
        let req = fixture_request(&fixture.family);
        let special_tokens = read_special_tokens(&root);
        let normalized = normalize_hf_template(&fixture.template);

        if fixture.expect_hf_render {
            validate_hf_template(&normalized)
                .unwrap_or_else(|err| panic!("{} should compile: {err}", fixture.repo));
            let prompt = render_hf_template(
                &normalized,
                &special_tokens,
                &ReasoningDefaults::default(),
                &req,
            )
            .unwrap_or_else(|err| panic!("{} should render via HF path: {err}", fixture.repo));
            assert!(
                !prompt.trim().is_empty(),
                "{} rendered an empty prompt",
                fixture.repo
            );
        } else {
            if validate_hf_template(&normalized).is_ok() {
                render_hf_template(
                    &normalized,
                    &special_tokens,
                    &ReasoningDefaults::default(),
                    &req,
                )
                .expect_err("fixture should still require fallback");
            }

            let prompt = PromptTemplate::detect(&root, &config)
                .render_request(&req)
                .unwrap_or_else(|render_err| {
                    panic!("{} should render via fallback: {render_err}", fixture.repo)
                });
            assert!(
                !prompt.trim().is_empty(),
                "{} fallback rendered an empty prompt",
                fixture.repo
            );
        }
    }
}
