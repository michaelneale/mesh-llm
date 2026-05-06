use anyhow::Result;
use mesh_llm_plugin::{
    capability, plugin_server_info, PluginMetadata, PluginRuntime, PluginStartupPolicy,
};

fn build_plugin(name: String) -> mesh_llm_plugin::SimplePlugin {
    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name,
            crate::VERSION,
            plugin_server_info(
                "mesh-survey",
                crate::VERSION,
                "Survey Metrics Plugin",
                "Enables host-owned OTLP metrics export for local model lifecycle telemetry.",
                Some(
                    "Configure [telemetry] and enable [[plugin]] name = \"survey\" \
                     to export metrics-only OTLP survey telemetry.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability(crate::plugin::SURVEY_CAPABILITY),
        ],
        health: |_context| {
            Box::pin(async move { Ok("metrics=host-owned".to_string()) })
        },
    }
}

pub(crate) async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(build_plugin(name)).await
}
