use super::collector::RuntimeDataCollector;
use super::plugins::{
    plugin_manifest_key, plugin_providers_key, plugin_summary_key, PluginDataValue,
};
use super::processes::RuntimeProcessSnapshot;
#[cfg(test)]
use super::snapshots::RuntimeDataSnapshots;
use super::snapshots::{PluginDataKey, PluginEndpointKey, RuntimeStatusSnapshot};
use super::subscriptions::{RuntimeDataDirty, RuntimeDataSubscriptionState};
use super::{RuntimeLlamaMetricsSnapshot, RuntimeLlamaSlotsSnapshot};
use crate::network::metrics::RoutingCollectorSnapshot;
use crate::plugin::{
    PluginCapabilityProvider, PluginEndpointSummary, PluginManifestOverview, PluginSummary,
};
use crate::runtime::instance::LocalInstanceSnapshot;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeDataSource {
    pub scope: &'static str,
    pub plugin_data_key: Option<PluginDataKey>,
    pub plugin_endpoint_key: Option<PluginEndpointKey>,
}

#[derive(Clone)]
pub(crate) struct RuntimeDataProducer {
    collector: RuntimeDataCollector,
    source: RuntimeDataSource,
}

impl RuntimeDataProducer {
    pub(crate) fn new(collector: RuntimeDataCollector, source: RuntimeDataSource) -> Self {
        Self { collector, source }
    }

    pub(crate) fn scope(&self) -> &'static str {
        self.source.scope
    }

    pub(crate) fn has_plugin_data_key(&self) -> bool {
        self.source.plugin_data_key.is_some()
    }

    pub(crate) fn has_plugin_endpoint_key(&self) -> bool {
        self.source.plugin_endpoint_key.is_some()
    }

    pub(crate) fn initial_process_count(&self) -> usize {
        self.collector
            .runtime_status_snapshot()
            .local_processes
            .len()
    }

    pub(crate) fn mark_status_dirty(&self) -> RuntimeDataSubscriptionState {
        self.collector.mark_dirty(RuntimeDataDirty::STATUS)
    }

    #[cfg(test)]
    pub(crate) fn mark_models_dirty(&self) -> RuntimeDataSubscriptionState {
        self.collector.mark_dirty(RuntimeDataDirty::MODELS)
    }

    #[cfg(test)]
    pub(crate) fn mark_routing_dirty(&self) -> RuntimeDataSubscriptionState {
        self.collector.mark_dirty(RuntimeDataDirty::ROUTING)
    }

    #[cfg(test)]
    pub(crate) fn mark_processes_dirty(&self) -> RuntimeDataSubscriptionState {
        self.collector.mark_dirty(RuntimeDataDirty::PROCESSES)
    }

    #[cfg(test)]
    pub(crate) fn mark_inventory_dirty(&self) -> RuntimeDataSubscriptionState {
        self.collector.mark_dirty(RuntimeDataDirty::INVENTORY)
    }

    #[cfg(test)]
    pub(crate) fn mark_plugins_dirty(&self) -> RuntimeDataSubscriptionState {
        self.collector.mark_dirty(RuntimeDataDirty::PLUGINS)
    }

    pub(crate) fn publish_runtime_status<F>(&self, update: F) -> bool
    where
        F: FnOnce(&mut RuntimeStatusSnapshot) -> bool,
    {
        self.collector
            .update_runtime_status(RuntimeDataDirty::STATUS, update)
    }

    pub(crate) fn publish_local_processes<F>(&self, update: F) -> bool
    where
        F: FnOnce(&mut Vec<RuntimeProcessSnapshot>) -> bool,
    {
        self.collector
            .update_runtime_status(RuntimeDataDirty::PROCESSES, |runtime_status| {
                update(&mut runtime_status.local_processes)
            })
    }

    pub(crate) fn replace_local_instances_snapshot(
        &self,
        instances: Vec<LocalInstanceSnapshot>,
    ) -> bool {
        self.collector.replace_local_instances_snapshot(instances)
    }

    pub(crate) fn publish_routing_snapshot(&self, snapshot: RoutingCollectorSnapshot) -> bool {
        self.collector.replace_routing_snapshot(snapshot)
    }

    pub(crate) fn publish_llama_metrics_snapshot(
        &self,
        snapshot: RuntimeLlamaMetricsSnapshot,
    ) -> bool {
        self.collector.replace_llama_metrics_snapshot(snapshot)
    }

    pub(crate) fn publish_llama_slots_snapshot(&self, snapshot: RuntimeLlamaSlotsSnapshot) -> bool {
        self.collector.replace_llama_slots_snapshot(snapshot)
    }

    pub(crate) fn publish_plugin_summary(&self, summary: PluginSummary) -> bool {
        let Some(plugin_name) = self.plugin_name() else {
            return false;
        };
        self.collector.publish_plugin_data(
            plugin_summary_key(plugin_name),
            PluginDataValue::Summary(Box::new(summary)),
        )
    }

    pub(crate) fn publish_plugin_manifest(&self, manifest: PluginManifestOverview) -> bool {
        let Some(plugin_name) = self.plugin_name() else {
            return false;
        };
        self.collector.publish_plugin_data(
            plugin_manifest_key(plugin_name),
            PluginDataValue::Manifest(manifest),
        )
    }

    pub(crate) fn publish_plugin_providers(
        &self,
        providers: Vec<PluginCapabilityProvider>,
    ) -> bool {
        let Some(plugin_name) = self.plugin_name() else {
            return false;
        };
        self.collector.publish_plugin_data(
            plugin_providers_key(plugin_name),
            PluginDataValue::Providers(providers),
        )
    }

    #[cfg(test)]
    pub(crate) fn publish_plugin_payload(
        &self,
        data_key: impl Into<String>,
        payload: serde_json::Value,
    ) -> bool {
        let Some(plugin_name) = self.plugin_name() else {
            return false;
        };
        self.collector.publish_plugin_data(
            PluginDataKey {
                plugin_name,
                data_key: data_key.into(),
            },
            PluginDataValue::Payload(payload),
        )
    }

    pub(crate) fn publish_plugin_endpoint(&self, summary: PluginEndpointSummary) -> bool {
        let Some(key) = self.source.plugin_endpoint_key.clone() else {
            return false;
        };
        self.collector.publish_plugin_endpoint(key, summary)
    }

    pub(crate) fn clear_plugin_reports(&self, plugin_name: &str) -> bool {
        self.collector.clear_plugin_reports(plugin_name)
    }

    fn plugin_name(&self) -> Option<String> {
        self.source
            .plugin_data_key
            .as_ref()
            .map(|key| key.plugin_name.clone())
            .or_else(|| {
                self.source
                    .plugin_endpoint_key
                    .as_ref()
                    .map(|key| key.plugin_name.clone())
            })
    }

    pub(crate) fn collector(&self) -> RuntimeDataCollector {
        self.collector.clone()
    }

    #[cfg(test)]
    pub(crate) fn source(&self) -> &RuntimeDataSource {
        &self.source
    }

    #[cfg(test)]
    pub(crate) fn snapshots(&self) -> RuntimeDataSnapshots {
        self.collector.snapshots()
    }
}
