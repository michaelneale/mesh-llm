use super::snapshots::{
    PluginDataKey, PluginDataSnapshot, PluginEndpointKey, PluginEndpointsSnapshot,
};
use crate::plugin::{
    PluginCapabilityProvider, PluginEndpointSummary, PluginManifestOverview, PluginSummary,
};
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) const PLUGIN_SUMMARY_DATA_KEY: &str = "summary";
pub(crate) const PLUGIN_MANIFEST_DATA_KEY: &str = "manifest";
pub(crate) const PLUGIN_PROVIDERS_DATA_KEY: &str = "providers";

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PluginDataValue {
    Summary(Box<PluginSummary>),
    Manifest(PluginManifestOverview),
    Providers(Vec<PluginCapabilityProvider>),
    #[cfg(test)]
    Payload(Value),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PluginsSnapshotView {
    pub plugins: Vec<PluginSummary>,
    pub manifests: BTreeMap<String, PluginManifestOverview>,
    pub providers: Vec<PluginCapabilityProvider>,
    pub payloads: BTreeMap<PluginDataKey, Value>,
    pub endpoints: Vec<PluginEndpointSummary>,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PluginScopedSnapshot {
    pub plugin_name: String,
    pub summary: Option<PluginSummary>,
    pub manifest: Option<PluginManifestOverview>,
    pub providers: Vec<PluginCapabilityProvider>,
    pub payloads: BTreeMap<String, Value>,
    pub endpoints: Vec<PluginEndpointSummary>,
}

pub(crate) fn plugin_summary_key(plugin_name: impl Into<String>) -> PluginDataKey {
    PluginDataKey {
        plugin_name: plugin_name.into(),
        data_key: PLUGIN_SUMMARY_DATA_KEY.into(),
    }
}

pub(crate) fn plugin_manifest_key(plugin_name: impl Into<String>) -> PluginDataKey {
    PluginDataKey {
        plugin_name: plugin_name.into(),
        data_key: PLUGIN_MANIFEST_DATA_KEY.into(),
    }
}

pub(crate) fn plugin_providers_key(plugin_name: impl Into<String>) -> PluginDataKey {
    PluginDataKey {
        plugin_name: plugin_name.into(),
        data_key: PLUGIN_PROVIDERS_DATA_KEY.into(),
    }
}

pub(crate) fn upsert_plugin_data(
    snapshot: &mut PluginDataSnapshot,
    key: PluginDataKey,
    value: PluginDataValue,
) -> bool {
    match snapshot.entries.get(&key) {
        Some(existing) if existing == &value => false,
        _ => {
            snapshot.entries.insert(key, value);
            true
        }
    }
}

pub(crate) fn clear_plugin_data(snapshot: &mut PluginDataSnapshot, plugin_name: &str) -> bool {
    let before = snapshot.entries.len();
    snapshot
        .entries
        .retain(|key, _| key.plugin_name != plugin_name);
    snapshot.entries.len() != before
}

pub(crate) fn upsert_plugin_endpoint(
    snapshot: &mut PluginEndpointsSnapshot,
    key: PluginEndpointKey,
    value: PluginEndpointSummary,
) -> bool {
    match snapshot.entries.get(&key) {
        Some(existing) if existing == &value => false,
        _ => {
            snapshot.entries.insert(key, value);
            true
        }
    }
}

pub(crate) fn clear_plugin_endpoints(
    snapshot: &mut PluginEndpointsSnapshot,
    plugin_name: &str,
) -> bool {
    let before = snapshot.entries.len();
    snapshot
        .entries
        .retain(|key, _| key.plugin_name != plugin_name);
    snapshot.entries.len() != before
}

pub(crate) fn plugins_snapshot(
    plugin_data: &PluginDataSnapshot,
    plugin_endpoints: &PluginEndpointsSnapshot,
) -> PluginsSnapshotView {
    let mut snapshot = PluginsSnapshotView::default();
    for (key, value) in &plugin_data.entries {
        match value {
            PluginDataValue::Summary(summary) => snapshot.plugins.push((**summary).clone()),
            PluginDataValue::Manifest(manifest) => {
                snapshot
                    .manifests
                    .insert(key.plugin_name.clone(), manifest.clone());
            }
            PluginDataValue::Providers(providers) => {
                snapshot.providers.extend(providers.iter().cloned());
            }
            #[cfg(test)]
            PluginDataValue::Payload(payload) => {
                snapshot.payloads.insert(key.clone(), payload.clone());
            }
        }
    }
    snapshot.endpoints = plugin_endpoints.entries.values().cloned().collect();
    snapshot.plugins.sort_by(|a, b| a.name.cmp(&b.name));
    snapshot.providers.sort_by(|a, b| {
        a.capability
            .cmp(&b.capability)
            .then_with(|| a.plugin_name.cmp(&b.plugin_name))
            .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
    });
    snapshot.endpoints.sort_by(|a, b| {
        a.plugin_name
            .cmp(&b.plugin_name)
            .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
    });
    snapshot
}

#[cfg(test)]
pub(crate) fn plugin_snapshot(
    plugin_data: &PluginDataSnapshot,
    plugin_endpoints: &PluginEndpointsSnapshot,
    plugin_name: &str,
) -> PluginScopedSnapshot {
    let mut snapshot = PluginScopedSnapshot {
        plugin_name: plugin_name.to_string(),
        ..PluginScopedSnapshot::default()
    };
    for (key, value) in plugin_data
        .entries
        .iter()
        .filter(|(key, _)| key.plugin_name == plugin_name)
    {
        match value {
            PluginDataValue::Summary(summary) => snapshot.summary = Some((**summary).clone()),
            PluginDataValue::Manifest(manifest) => snapshot.manifest = Some(manifest.clone()),
            PluginDataValue::Providers(providers) => {
                snapshot.providers.extend(providers.iter().cloned())
            }
            PluginDataValue::Payload(payload) => {
                snapshot
                    .payloads
                    .insert(key.data_key.clone(), payload.clone());
            }
        }
    }
    snapshot.endpoints = plugin_endpoints
        .entries
        .iter()
        .filter(|(key, _)| key.plugin_name == plugin_name)
        .map(|(_, value)| value.clone())
        .collect();
    snapshot.providers.sort_by(|a, b| {
        a.capability
            .cmp(&b.capability)
            .then_with(|| a.plugin_name.cmp(&b.plugin_name))
            .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
    });
    snapshot
        .endpoints
        .sort_by(|a, b| a.endpoint_id.cmp(&b.endpoint_id));
    snapshot
}

#[cfg(test)]
pub(crate) fn plugin_endpoint_snapshot(
    plugin_endpoints: &PluginEndpointsSnapshot,
    plugin_name: &str,
    endpoint_id: &str,
) -> Option<PluginEndpointSummary> {
    plugin_endpoints
        .entries
        .get(&PluginEndpointKey {
            plugin_name: plugin_name.to_string(),
            endpoint_id: endpoint_id.to_string(),
        })
        .cloned()
}
