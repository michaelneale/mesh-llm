use anyhow::Result;
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata, PluginRuntime, PluginStartupPolicy,
};

const DEFAULT_LEMONADE_BASE_URL: &str = "http://localhost:8000/api/v1";

fn lemonade_base_url() -> String {
    std::env::var("MESH_LLM_LEMONADE_BASE_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_LEMONADE_BASE_URL.to_string())
}

fn build_lemonade_plugin(name: String) -> mesh_llm_plugin::SimplePlugin {
    let base_url = lemonade_base_url();
    let health_url = base_url.clone();

    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name,
            crate::VERSION,
            plugin_server_info(
                "mesh-lemonade",
                crate::VERSION,
                "Lemonade Endpoint Plugin",
                "Registers a local Lemonade OpenAI-compatible inference endpoint.",
                Some(
                    "Exposes a local OpenAI-compatible inference endpoint to mesh-llm when enabled.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
        ],
        inference: [
            mesh_llm_plugin::inference::openai_http("lemonade", base_url.clone())
                .managed_by_plugin(false),
        ],
        health: move |_context| {
            let health_url = health_url.clone();
            Box::pin(async move { Ok(format!("base_url={health_url}")) })
        },
    }
}

pub(crate) async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(build_lemonade_plugin(name)).await
}
