use anyhow::Result;
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata, PluginRuntime, PluginStartupPolicy,
};

const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";

fn base_url() -> String {
    std::env::var("MESH_LLM_OPENAI_ENDPOINT_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
}

fn build_plugin(name: String) -> mesh_llm_plugin::SimplePlugin {
    let base_url = base_url();
    let health_url = base_url.clone();

    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name,
            crate::VERSION,
            plugin_server_info(
                "mesh-openai-endpoint",
                crate::VERSION,
                "OpenAI-Compatible Endpoint Plugin",
                "Routes inference to an external OpenAI-compatible server (vLLM, TGI, Ollama, etc.).",
                Some(
                    "Set MESH_LLM_OPENAI_ENDPOINT_URL to point at any server \
                     that speaks the OpenAI /v1/chat/completions API.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
        ],
        inference: [
            mesh_llm_plugin::inference::openai_http("openai-endpoint", base_url.clone())
                .managed_by_plugin(false),
        ],
        health: move |_context| {
            let health_url = health_url.clone();
            Box::pin(async move { Ok(format!("base_url={health_url}")) })
        },
    }
}

pub(crate) async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(build_plugin(name)).await
}
