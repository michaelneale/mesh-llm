//! Runtime-data snapshot ownership and compatibility guardrails.
//!
//! Broad runtime reads should go through the collector so API payloads stay
//! stable while subsystem publishers mutate their own snapshots.

mod api_views;
mod collector;
mod inventory;
mod metrics;
mod plugins;
mod processes;
mod producers;
mod snapshots;
mod subscriptions;

pub(crate) use self::api_views::{collect_views, mesh_models, status_payload};
pub(crate) use self::collector::RuntimeDataCollector;
pub(crate) use self::metrics::{
    RuntimeLlamaEndpointStatus, RuntimeLlamaMetricItem, RuntimeLlamaMetricSample,
    RuntimeLlamaMetricsSnapshot, RuntimeLlamaRuntimeItems, RuntimeLlamaRuntimeSnapshot,
    RuntimeLlamaSlotItem, RuntimeLlamaSlotSnapshot, RuntimeLlamaSlotsSnapshot,
};
pub(crate) use self::processes::{
    remove_runtime_process_snapshot, runtime_process_payloads, upsert_runtime_process_snapshot,
    RuntimeProcessSnapshot,
};
pub(crate) use self::producers::{RuntimeDataProducer, RuntimeDataSource};
pub(crate) use self::snapshots::{
    HardwareViewInput, ModelViewInput, PluginDataKey, PluginEndpointKey, StatusViewInput,
};
pub(crate) use self::subscriptions::RuntimeDataDirty;

#[cfg(test)]
mod tests {
    use super::api_views::{collect_views, mesh_models, status_payload};
    use super::processes::{runtime_process_payloads, RuntimeProcessSnapshot};
    use super::snapshots::{
        HardwareViewInput, ModelViewInput, PluginDataKey, PluginEndpointKey, StatusViewInput,
    };
    use super::subscriptions::{RuntimeDataDirty, RuntimeDataVersion};
    use super::{RuntimeDataCollector, RuntimeDataSource};
    use super::{
        RuntimeLlamaEndpointStatus, RuntimeLlamaMetricSample, RuntimeLlamaMetricsSnapshot,
        RuntimeLlamaSlotSnapshot, RuntimeLlamaSlotsSnapshot,
    };
    use crate::api::status::{
        build_gpus, build_ownership_payload, LocalInstance, NodeState, StatusPayload,
    };
    use crate::api::RuntimeProcessPayload;
    use crate::inference::election;
    use crate::mesh::MeshCatalogEntry;
    use crate::models::LocalModelInventorySnapshot;
    use crate::network::openai::transport::{self, ResponseAdapter};
    use crate::plugin::{
        PluginCapabilityProvider, PluginEndpointSummary, PluginManifestOverview, PluginSummary,
    };
    use crate::runtime::instance::LocalInstanceSnapshot;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::{collections::HashMap, collections::HashSet};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    #[test]
    fn runtime_data_collector_shell_constructs_and_clones() {
        let collector = RuntimeDataCollector::new();
        let clone = collector.clone();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
        let plugin_data_key = PluginDataKey {
            plugin_name: "plugin-a".into(),
            data_key: "status".into(),
        };
        let plugin_endpoint_key = PluginEndpointKey {
            plugin_name: "plugin-b".into(),
            endpoint_id: "chat".into(),
        };
        let plugin_data_producer = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(plugin_data_key.clone()),
            plugin_endpoint_key: None,
        });
        let plugin_endpoint_producer = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(plugin_endpoint_key.clone()),
        });

        assert_eq!(producer.source().scope, "runtime");
        assert!(producer.source().plugin_data_key.is_none());
        assert!(producer.source().plugin_endpoint_key.is_none());
        assert_eq!(
            plugin_data_producer.source().plugin_data_key.as_ref(),
            Some(&plugin_data_key)
        );
        assert_eq!(
            plugin_endpoint_producer
                .source()
                .plugin_endpoint_key
                .as_ref(),
            Some(&plugin_endpoint_key)
        );
        assert!(producer
            .snapshots()
            .runtime_status
            .local_processes
            .is_empty());
        assert!(clone.snapshots().local_instances.instances.is_empty());
        assert!(producer
            .collector()
            .plugin_data_snapshot()
            .entries
            .is_empty());
    }

    #[test]
    fn runtime_data_collector_exposes_initial_snapshots() {
        let collector = RuntimeDataCollector::new();
        let views = collect_views(&collector);

        assert!(collector
            .runtime_status_snapshot()
            .local_processes
            .is_empty());
        assert!(collector.local_instances_snapshot().instances.is_empty());
        assert!(collector.plugin_data_snapshot().entries.is_empty());
        assert!(collector.plugin_endpoints_snapshot().entries.is_empty());
        assert!(views.runtime_status.primary_model.is_none());
        assert!(views.runtime_status.primary_backend.is_none());
        assert!(!views.runtime_status.is_host);
        assert!(!views.runtime_status.is_client);
        assert!(!views.runtime_status.llama_ready);
        assert!(views.runtime_status.llama_port.is_none());
        assert!(views.runtime_status.local_processes.is_empty());
        assert!(views.local_instances.instances.is_empty());
        assert!(views.plugin_data.entries.is_empty());
        assert!(views.plugin_endpoints.entries.is_empty());
    }

    #[test]
    fn runtime_data_version_advances_and_marks_dirty_bits() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        let initial = collector.subscription_state();
        assert_eq!(initial.version, RuntimeDataVersion::default());
        assert!(initial.dirty.is_empty());

        let status_state = producer.mark_status_dirty();
        assert_eq!(status_state.version.get(), 1);
        assert_eq!(status_state.dirty, RuntimeDataDirty::STATUS);

        let processes_changed = producer.publish_local_processes(|local_processes| {
            local_processes.push(RuntimeProcessSnapshot {
                model: "Qwen3-8B".into(),
                backend: "metal".into(),
                pid: 4242,
                port: 9337,
                slots: 4,
                command: None,
                state: "ready".into(),
                start: None,
                health: Some("ready".into()),
            });
            true
        });
        assert!(processes_changed);

        let processes_state = collector.subscription_state();
        assert_eq!(processes_state.version.get(), 2);
        assert!(processes_state.dirty.contains(RuntimeDataDirty::STATUS));
        assert!(processes_state.dirty.contains(RuntimeDataDirty::PROCESSES));

        let no_change = producer.publish_local_processes(|_| false);
        assert!(!no_change);
        assert_eq!(collector.subscription_state(), processes_state);

        let models_state = producer.mark_models_dirty();
        assert_eq!(models_state.version.get(), 3);
        assert!(models_state.dirty.contains(RuntimeDataDirty::STATUS));
        assert!(models_state.dirty.contains(RuntimeDataDirty::PROCESSES));
        assert!(models_state.dirty.contains(RuntimeDataDirty::MODELS));

        let routing_state = producer.mark_routing_dirty();
        assert_eq!(routing_state.version.get(), 4);
        assert!(routing_state.dirty.contains(RuntimeDataDirty::ROUTING));

        let processes_state = producer.mark_processes_dirty();
        assert_eq!(processes_state.version.get(), 5);
        assert!(processes_state.dirty.contains(RuntimeDataDirty::PROCESSES));

        let inventory_state = producer.mark_inventory_dirty();
        assert_eq!(inventory_state.version.get(), 6);
        assert!(inventory_state.dirty.contains(RuntimeDataDirty::INVENTORY));

        let plugins_state = producer.mark_plugins_dirty();
        assert_eq!(plugins_state.version.get(), 7);
        assert!(plugins_state.dirty.contains(RuntimeDataDirty::PLUGINS));

        let runtime_status_changed = producer.publish_runtime_status(|runtime_status| {
            runtime_status.primary_backend = Some("metal".into());
            true
        });
        assert!(runtime_status_changed);

        let final_state = collector.subscription_state();
        assert_eq!(final_state.version.get(), 8);
        assert!(final_state.dirty.contains(RuntimeDataDirty::STATUS));
    }

    #[tokio::test]
    async fn runtime_data_subscribe_notifies_once_per_update() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
        let mut subscription = collector.subscribe();

        assert!(!subscription.has_changed().expect("watch channel open"));

        producer.mark_status_dirty();
        subscription
            .changed()
            .await
            .expect("status update delivered");
        let first = *subscription.borrow_and_update();

        assert_eq!(first.version.get(), 1);
        assert!(first.dirty.contains(RuntimeDataDirty::STATUS));
        assert!(!subscription.has_changed().expect("watch channel open"));

        producer.mark_models_dirty();
        producer.mark_routing_dirty();
        subscription
            .changed()
            .await
            .expect("coalesced updates delivered");
        let second = *subscription.borrow_and_update();

        assert_eq!(second.version.get(), 3);
        assert!(second.dirty.contains(RuntimeDataDirty::STATUS));
        assert!(second.dirty.contains(RuntimeDataDirty::MODELS));
        assert!(second.dirty.contains(RuntimeDataDirty::ROUTING));
        assert!(!subscription.has_changed().expect("watch channel open"));
    }

    #[test]
    fn runtime_data_process_snapshot_matches_existing_runtime_views() {
        let legacy_processes = vec![
            RuntimeProcessPayload {
                name: "Zulu".into(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 11,
                slots: 4,
            },
            RuntimeProcessPayload {
                name: "Alpha".into(),
                backend: "llama".into(),
                status: "starting".into(),
                port: 9337,
                pid: 10,
                slots: 4,
            },
        ];
        let collector_rows = legacy_processes
            .iter()
            .map(RuntimeProcessSnapshot::from_payload)
            .collect::<Vec<_>>();

        assert_eq!(collector_rows[0].model, "Zulu");
        assert_eq!(collector_rows[0].backend, "llama");
        assert_eq!(collector_rows[0].pid, 11);
        assert_eq!(collector_rows[0].port, 9444);
        assert_eq!(collector_rows[0].command, None);
        assert_eq!(collector_rows[0].state, "ready");
        assert_eq!(collector_rows[0].start, None);
        assert_eq!(collector_rows[0].health.as_deref(), Some("ready"));

        let round_trip = runtime_process_payloads(&collector_rows);
        assert_eq!(round_trip, legacy_processes);
    }

    #[test]
    fn runtime_data_status_snapshot_matches_api_payloads() {
        let collector = RuntimeDataCollector::new();
        collector.replace_local_instances_snapshot(vec![LocalInstanceSnapshot {
            pid: 111,
            api_port: Some(3131),
            version: Some("0.65.0-test".into()),
            started_at_unix: 456,
            runtime_dir: PathBuf::from("/tmp/runtime-1"),
            is_self: true,
        }]);
        let hardware = collector.build_hardware_view(HardwareViewInput {
            gpu_name: Some("RTX 4090".into()),
            gpu_vram: Some("25769803776".into()),
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            my_vram_gb: 25.769803776,
            model_size_gb: 12.5,
            first_joined_mesh_ts: Some(123),
        });
        let snapshot = collector.build_status_view(StatusViewInput {
            version: "0.65.0-test".into(),
            latest_version: Some("0.66.0".into()),
            node_id: "node-1".into(),
            owner: crate::crypto::OwnershipSummary::default(),
            token: "invite-token".into(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: "Qwen-Test".into(),
            models: vec!["Qwen-Test".into()],
            available_models: vec!["Qwen-Test".into()],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            inflight_requests: 2,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            nostr_discovery: true,
            publication_state: "public".into(),
            local_processes: vec![],
            peers: vec![],
            wakeable_nodes: vec![],
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            hardware,
        });

        let payload = status_payload(snapshot);
        let expected = StatusPayload {
            version: "0.65.0-test".into(),
            latest_version: Some("0.66.0".into()),
            node_id: "node-1".into(),
            owner: build_ownership_payload(&crate::crypto::OwnershipSummary::default()),
            token: "invite-token".into(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().into(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: "Qwen-Test".into(),
            models: vec!["Qwen-Test".into()],
            available_models: vec!["Qwen-Test".into()],
            requested_models: vec![],
            wanted_model_refs: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 25.769803776,
            model_size_gb: 12.5,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![LocalInstance {
                pid: 111,
                api_port: Some(3131),
                version: Some("0.65.0-test".into()),
                started_at_unix: 456,
                runtime_dir: "/tmp/runtime-1".into(),
                is_self: true,
            }],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 2,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            nostr_discovery: true,
            publication_state: "public".into(),
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            gpus: build_gpus(
                Some("RTX 4090"),
                Some("25769803776"),
                None,
                None,
                None,
                None,
            ),
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            routing_metrics: crate::network::metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: Some(123),
        };

        assert_eq!(
            serde_json::to_value(&payload).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }

    #[test]
    fn runtime_data_model_snapshot_matches_api_payloads() {
        let collector = RuntimeDataCollector::new();
        let local_inventory = LocalModelInventorySnapshot {
            model_names: HashSet::from(["Example-Model".to_string()]),
            size_by_name: HashMap::from([("Example-Model".to_string(), 8_000_000_000)]),
            metadata_by_name: HashMap::new(),
        };
        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![MeshCatalogEntry {
                model_name: "Example-Model".into(),
                descriptor: None,
            }],
            served_models: vec![],
            active_demand: HashMap::new(),
            my_serving_models: vec![],
            my_hosted_models: vec![],
            local_inventory,
            node_hostname: Some("node.local".into()),
            my_vram_gb: 24.0,
            model_name: "Another-Model".into(),
            model_size_bytes: 0,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        assert_eq!(payload.len(), 1);
        assert_eq!(payload[0].name, "Example-Model");
        assert_eq!(payload[0].status, "cold");
        assert_eq!(payload[0].size_gb, 8.0);
        assert_eq!(
            payload[0].download_command,
            "mesh-llm models download Example-Model"
        );
        assert_eq!(
            payload[0].run_command,
            "mesh-llm serve --model Example-Model"
        );
        assert_eq!(
            payload[0].auto_command,
            "mesh-llm serve --auto --model Example-Model"
        );
        assert_eq!(payload[0].fit_label, "Likely comfortable");
    }

    #[tokio::test]
    async fn runtime_data_inventory_single_flight_scan_coalesces() {
        let collector = RuntimeDataCollector::new();
        let scan_count = Arc::new(AtomicUsize::new(0));

        let first = {
            let collector = collector.clone();
            let scan_count = scan_count.clone();
            tokio::spawn(async move {
                collector
                    .coalesce_local_inventory_scan(move || {
                        scan_count.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        let mut snapshot = LocalModelInventorySnapshot::default();
                        snapshot.model_names.insert("Qwen3-8B".into());
                        snapshot
                            .size_by_name
                            .insert("Qwen3-8B".into(), 8_000_000_000);
                        snapshot
                    })
                    .await
            })
        };

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let second = {
            let collector = collector.clone();
            tokio::spawn(async move {
                collector
                    .coalesce_local_inventory_scan(LocalModelInventorySnapshot::default)
                    .await
            })
        };

        let first_snapshot = first.await.expect("first inventory scan task should join");
        let second_snapshot = second
            .await
            .expect("second inventory scan task should join");

        assert_eq!(scan_count.load(Ordering::SeqCst), 1);
        assert_eq!(first_snapshot, second_snapshot);
        assert_eq!(collector.local_inventory_snapshot(), first_snapshot);
        assert!(collector
            .local_inventory_snapshot()
            .model_names
            .contains("Qwen3-8B"));
    }

    #[tokio::test]
    async fn runtime_data_llama_metrics_polling_records_success_and_samples() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test metrics server");
        let port = listener.local_addr().expect("local addr").port();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept test request");
            let mut buf = vec![0u8; 2048];
            let n = stream.read(&mut buf).await.expect("read test request");
            let request = String::from_utf8_lossy(&buf[..n]);
            assert!(request.starts_with("GET /metrics "));
            let body = concat!(
                "# HELP llama_requests_processing Current requests\n",
                "llama_requests_processing 2\n",
                "llama_slot_tokens{slot=\"0\",state=\"idle\",unbounded=\"drop-me\"} 7\n",
                "llama_unpermitted_high_cardinality{request=\"abc\"} 99\n"
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write metrics response");
        });

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("build reqwest client");
        crate::inference::launch::poll_llama_metrics_once(&client, port, &producer).await;
        server.await.expect("join test metrics server");

        let snapshot = collector.runtime_llama_snapshot();
        assert_eq!(snapshot.metrics.status, RuntimeLlamaEndpointStatus::Ready);
        assert!(snapshot.metrics.last_attempt_unix_ms.is_some());
        assert!(snapshot.metrics.last_success_unix_ms.is_some());
        assert!(snapshot.metrics.error.is_none());
        assert!(snapshot
            .metrics
            .raw_text
            .as_deref()
            .unwrap_or_default()
            .contains("llama_requests_processing 2"));
        assert_eq!(snapshot.metrics.samples.len(), 2);
        assert_eq!(
            snapshot.metrics.samples[0].name,
            "llama_requests_processing"
        );
        assert_eq!(snapshot.metrics.samples[0].value, 2.0);
        assert_eq!(
            snapshot.metrics.samples[1]
                .labels
                .get("slot")
                .map(String::as_str),
            Some("0")
        );
        assert_eq!(
            snapshot.metrics.samples[1]
                .labels
                .get("state")
                .map(String::as_str),
            Some("idle")
        );
        assert!(!snapshot.metrics.samples[1].labels.contains_key("unbounded"));
        assert_eq!(snapshot.items.metrics.len(), 2);
    }

    #[tokio::test]
    async fn runtime_data_llama_metrics_polling_marks_unavailable_nonfatally() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
        producer.publish_local_processes(|local_processes| {
            local_processes.push(RuntimeProcessSnapshot {
                model: "Qwen3-8B".into(),
                backend: "llama".into(),
                pid: 42,
                port: 9337,
                slots: 4,
                command: Some("llama-server".into()),
                state: "ready".into(),
                start: None,
                health: Some("ready".into()),
            });
            true
        });

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind free port probe");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1))
            .build()
            .expect("build reqwest client");
        crate::inference::launch::poll_llama_metrics_once(&client, port, &producer).await;

        let snapshot = collector.runtime_llama_snapshot();
        assert_eq!(
            snapshot.metrics.status,
            RuntimeLlamaEndpointStatus::Unavailable
        );
        assert!(snapshot.metrics.last_attempt_unix_ms.is_some());
        assert!(snapshot.metrics.last_success_unix_ms.is_none());
        assert!(snapshot.metrics.raw_text.is_none());
        assert!(snapshot.metrics.samples.is_empty());
        assert!(snapshot.metrics.error.is_some());
        assert_eq!(collector.runtime_processes_snapshot().len(), 1);
        assert_eq!(collector.runtime_processes_snapshot()[0].model, "Qwen3-8B");
    }

    #[tokio::test]
    async fn runtime_data_llama_failed_refresh_preserves_previous_payloads() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
        producer.publish_llama_metrics_snapshot(RuntimeLlamaMetricsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            last_attempt_unix_ms: Some(10),
            last_success_unix_ms: Some(10),
            error: None,
            raw_text: Some("llama_requests_processing 2\n".into()),
            samples: vec![RuntimeLlamaMetricSample {
                name: "llama_requests_processing".into(),
                value: 2.0,
                ..RuntimeLlamaMetricSample::default()
            }],
        });
        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            last_attempt_unix_ms: Some(20),
            last_success_unix_ms: Some(20),
            error: None,
            slots: vec![RuntimeLlamaSlotSnapshot {
                id: Some(7),
                id_task: Some(99),
                is_processing: Some(true),
                ..RuntimeLlamaSlotSnapshot::default()
            }],
        });

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind free port probe");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1))
            .build()
            .expect("build reqwest client");
        crate::inference::launch::poll_llama_metrics_once(&client, port, &producer).await;

        let snapshot = collector.runtime_llama_snapshot();
        assert_eq!(
            snapshot.metrics.status,
            RuntimeLlamaEndpointStatus::Unavailable
        );
        assert!(snapshot.metrics.last_attempt_unix_ms > snapshot.metrics.last_success_unix_ms);
        assert_eq!(snapshot.metrics.last_success_unix_ms, Some(10));
        assert_eq!(
            snapshot.metrics.raw_text.as_deref(),
            Some("llama_requests_processing 2\n")
        );
        assert_eq!(snapshot.metrics.samples.len(), 1);
        assert_eq!(snapshot.items.metrics.len(), 1);
        assert_eq!(snapshot.slots.status, RuntimeLlamaEndpointStatus::Ready);
        assert_eq!(snapshot.slots.last_success_unix_ms, Some(20));
        assert_eq!(snapshot.slots.slots.len(), 1);
        assert_eq!(snapshot.items.slots_total, 1);

        crate::inference::launch::poll_llama_slots_once(&client, port, &producer).await;

        let snapshot = collector.runtime_llama_snapshot();
        assert_eq!(
            snapshot.metrics.status,
            RuntimeLlamaEndpointStatus::Unavailable
        );
        assert_eq!(snapshot.metrics.samples.len(), 1);
        assert_eq!(
            snapshot.slots.status,
            RuntimeLlamaEndpointStatus::Unavailable
        );
        assert!(snapshot.slots.last_attempt_unix_ms > snapshot.slots.last_success_unix_ms);
        assert_eq!(snapshot.slots.last_success_unix_ms, Some(20));
        assert_eq!(snapshot.slots.slots.len(), 1);
        assert_eq!(snapshot.items.metrics.len(), 1);
        assert_eq!(snapshot.items.slots_total, 1);
    }

    #[test]
    fn runtime_data_llama_slots_json_parses_permissively() {
        let slots = crate::inference::launch::parse_llama_slots(&json!([
            {
                "id": 3,
                "id_task": "44",
                "n_ctx": 8192,
                "speculative": 1,
                "is_processing": true,
                "next_token": {"id": 99, "piece": "hello"},
                "params": {"temperature": 0.1},
                "state": "busy",
                "kv": {"used": 17}
            }
        ]))
        .expect("parse permissive slots json");

        assert_eq!(slots.len(), 1);
        let slot = &slots[0];
        assert_eq!(slot.id, Some(3));
        assert_eq!(slot.id_task, Some(44));
        assert_eq!(slot.n_ctx, Some(8192));
        assert_eq!(slot.speculative, Some(true));
        assert_eq!(slot.is_processing, Some(true));
        assert_eq!(
            slot.params
                .as_ref()
                .and_then(|value| value.get("temperature")),
            Some(&json!(0.1))
        );
        assert_eq!(
            slot.next_token
                .as_ref()
                .and_then(|value| value.get("piece")),
            Some(&json!("hello"))
        );
        assert_eq!(slot.extra.get("state"), Some(&json!("busy")));
        assert_eq!(slot.extra.get("kv"), Some(&json!({"used": 17})));
    }

    #[test]
    fn runtime_data_llama_slots_parsing_bounds_entries_and_large_json() {
        let mut slot_entries = Vec::new();
        for id in 0..70_u64 {
            slot_entries.push(json!({
                "id": id,
                "is_processing": id == 69,
                "params": {"prompt": "x".repeat(5000)},
                "extra_payload": "y".repeat(5000)
            }));
        }

        let slots = crate::inference::launch::parse_llama_slots(&json!(slot_entries))
            .expect("parse bounded slots json");

        assert_eq!(slots.len(), 64);
        assert_eq!(slots[0].id, Some(0));
        assert_eq!(slots[63].id, Some(63));
        assert_eq!(slots[0].params, Some(json!("[truncated]")));
        assert_eq!(slots[0].extra, json!("[truncated]"));
        assert!(!slots.iter().any(|slot| slot.id == Some(69)));
    }

    #[test]
    fn runtime_data_llama_items_preserve_slot_index_and_busy_state() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            last_attempt_unix_ms: Some(1),
            last_success_unix_ms: Some(1),
            error: None,
            slots: vec![
                RuntimeLlamaSlotSnapshot {
                    id: Some(10),
                    is_processing: Some(false),
                    ..RuntimeLlamaSlotSnapshot::default()
                },
                RuntimeLlamaSlotSnapshot {
                    id: Some(20),
                    id_task: Some(42),
                    n_ctx: Some(8192),
                    is_processing: Some(true),
                    ..RuntimeLlamaSlotSnapshot::default()
                },
            ],
        });

        let snapshot = collector.runtime_llama_snapshot();
        assert_eq!(snapshot.items.slots_total, 2);
        assert_eq!(snapshot.items.slots_busy, 1);
        assert_eq!(snapshot.items.slots[0].index, 0);
        assert_eq!(snapshot.items.slots[0].id, Some(10));
        assert!(!snapshot.items.slots[0].is_processing);
        assert_eq!(snapshot.items.slots[1].index, 1);
        assert_eq!(snapshot.items.slots[1].id, Some(20));
        assert!(snapshot.items.slots[1].is_processing);
    }

    #[test]
    fn runtime_data_local_instance_snapshot_replaces_existing_scan_results() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        let original = LocalInstanceSnapshot {
            pid: 100,
            api_port: Some(3131),
            version: Some("0.1.0".into()),
            started_at_unix: 1,
            runtime_dir: PathBuf::from("/tmp/runtime-a"),
            is_self: false,
        };
        let replacement = LocalInstanceSnapshot {
            pid: 200,
            api_port: Some(4141),
            version: Some("0.2.0".into()),
            started_at_unix: 2,
            runtime_dir: PathBuf::from("/tmp/runtime-b"),
            is_self: true,
        };

        assert!(
            crate::runtime::instance::publish_local_instance_scan_results(
                &producer,
                vec![original.clone()],
            )
        );
        assert_eq!(
            collector.local_instances_snapshot().instances,
            vec![original]
        );

        assert!(
            crate::runtime::instance::publish_local_instance_scan_results(
                &producer,
                vec![replacement.clone()],
            )
        );
        assert_eq!(
            collector.local_instances_snapshot().instances,
            vec![replacement]
        );
    }

    #[test]
    fn runtime_data_plugin_reports_are_scoped_by_name_and_endpoint() {
        let collector = RuntimeDataCollector::new();
        let alpha = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name: "alpha".into(),
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        });
        let alpha_endpoint = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(PluginEndpointKey {
                plugin_name: "alpha".into(),
                endpoint_id: "chat".into(),
            }),
        });
        let beta = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name: "beta".into(),
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        });
        let beta_endpoint = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(PluginEndpointKey {
                plugin_name: "beta".into(),
                endpoint_id: "embed".into(),
            }),
        });

        alpha.publish_plugin_summary(PluginSummary {
            name: "alpha".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: Some(1001),
            version: Some("1.0.0".into()),
            capabilities: vec!["chat".into()],
            command: Some("alpha-plugin".into()),
            args: vec!["--serve".into()],
            tools: Vec::new(),
            manifest: Some(PluginManifestOverview {
                operations: 1,
                resources: 0,
                resource_templates: 0,
                prompts: 0,
                completions: 0,
                http_bindings: 0,
                endpoints: 1,
                mesh_channels: 0,
                mesh_event_subscriptions: 0,
                capabilities: vec!["chat".into()],
            }),
            error: None,
        });
        alpha.publish_plugin_manifest(PluginManifestOverview {
            operations: 1,
            resources: 0,
            resource_templates: 0,
            prompts: 0,
            completions: 0,
            http_bindings: 0,
            endpoints: 1,
            mesh_channels: 0,
            mesh_event_subscriptions: 0,
            capabilities: vec!["chat".into()],
        });
        alpha.publish_plugin_providers(vec![PluginCapabilityProvider {
            capability: "chat".into(),
            plugin_name: "alpha".into(),
            plugin_status: "running".into(),
            endpoint_id: Some("chat".into()),
            available: true,
            detail: None,
        }]);
        alpha.publish_plugin_payload("metrics", json!({"requests": 2}));
        alpha_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
            plugin_name: "alpha".into(),
            plugin_status: "running".into(),
            endpoint_id: "chat".into(),
            state: "healthy".into(),
            available: true,
            kind: "mcp".into(),
            transport_kind: "http".into(),
            protocol: Some("http".into()),
            address: Some("http://127.0.0.1:9000/mcp".into()),
            args: Vec::new(),
            namespace: Some("alpha.chat".into()),
            supports_streaming: true,
            managed_by_plugin: true,
            detail: None,
            models: vec!["alpha-model".into()],
        });

        beta.publish_plugin_summary(PluginSummary {
            name: "beta".into(),
            kind: "external".into(),
            enabled: true,
            status: "disabled".into(),
            pid: None,
            version: None,
            capabilities: vec!["embed".into()],
            command: Some("beta-plugin".into()),
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            error: Some("disabled".into()),
        });
        beta.publish_plugin_payload("metrics", json!({"requests": 5}));
        beta_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
            plugin_name: "beta".into(),
            plugin_status: "disabled".into(),
            endpoint_id: "embed".into(),
            state: "unavailable".into(),
            available: false,
            kind: "inference".into(),
            transport_kind: "tcp".into(),
            protocol: None,
            address: Some("127.0.0.1:9444".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
            detail: Some("disabled".into()),
            models: vec!["beta-model".into()],
        });

        let all = collector.plugins_snapshot();
        assert_eq!(
            all.plugins
                .iter()
                .map(|plugin| plugin.name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
        assert_eq!(
            all.endpoints
                .iter()
                .map(|endpoint| (endpoint.plugin_name.as_str(), endpoint.endpoint_id.as_str()))
                .collect::<Vec<_>>(),
            vec![("alpha", "chat"), ("beta", "embed")]
        );

        let alpha_snapshot = collector.plugin_snapshot("alpha");
        assert_eq!(alpha_snapshot.plugin_name, "alpha");
        assert_eq!(
            alpha_snapshot
                .summary
                .as_ref()
                .map(|summary| summary.name.as_str()),
            Some("alpha")
        );
        assert_eq!(
            alpha_snapshot
                .manifest
                .as_ref()
                .map(|manifest| manifest.endpoints),
            Some(1)
        );
        assert_eq!(alpha_snapshot.providers.len(), 1);
        assert_eq!(
            alpha_snapshot.payloads.get("metrics"),
            Some(&json!({"requests": 2}))
        );
        assert_eq!(alpha_snapshot.endpoints.len(), 1);
        assert_eq!(alpha_snapshot.endpoints[0].endpoint_id, "chat");

        assert!(collector.plugin_snapshot("gamma").summary.is_none());
        assert!(collector.plugin_snapshot("gamma").endpoints.is_empty());
        assert_eq!(
            collector
                .plugin_endpoint_snapshot("alpha", "chat")
                .as_ref()
                .map(|endpoint| endpoint.address.as_deref()),
            Some(Some("http://127.0.0.1:9000/mcp"))
        );
        assert!(collector
            .plugin_endpoint_snapshot("alpha", "embed")
            .is_none());
        assert!(collector.plugin_endpoint_snapshot("beta", "chat").is_none());
    }

    #[test]
    fn runtime_data_plugin_clear_removes_only_target_plugin_reports() {
        let collector = RuntimeDataCollector::new();
        let alpha = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name: "alpha".into(),
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        });
        let alpha_endpoint = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(PluginEndpointKey {
                plugin_name: "alpha".into(),
                endpoint_id: "chat".into(),
            }),
        });
        let beta = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name: "beta".into(),
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        });
        let beta_endpoint = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(PluginEndpointKey {
                plugin_name: "beta".into(),
                endpoint_id: "embed".into(),
            }),
        });

        alpha.publish_plugin_summary(PluginSummary {
            name: "alpha".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: Some(1001),
            version: Some("1.0.0".into()),
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            error: None,
        });
        alpha.publish_plugin_payload("metrics", json!({"requests": 1}));
        alpha_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
            plugin_name: "alpha".into(),
            plugin_status: "running".into(),
            endpoint_id: "chat".into(),
            state: "healthy".into(),
            available: true,
            kind: "mcp".into(),
            transport_kind: "http".into(),
            protocol: Some("http".into()),
            address: Some("http://127.0.0.1:9000/mcp".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: true,
            detail: None,
            models: Vec::new(),
        });
        beta.publish_plugin_summary(PluginSummary {
            name: "beta".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: Some(1002),
            version: Some("2.0.0".into()),
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            error: None,
        });
        beta.publish_plugin_payload("metrics", json!({"requests": 7}));
        beta_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
            plugin_name: "beta".into(),
            plugin_status: "running".into(),
            endpoint_id: "embed".into(),
            state: "healthy".into(),
            available: true,
            kind: "inference".into(),
            transport_kind: "tcp".into(),
            protocol: None,
            address: Some("127.0.0.1:9444".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
            detail: None,
            models: vec!["beta-model".into()],
        });

        assert!(alpha.clear_plugin_reports("alpha"));

        let alpha_snapshot = collector.plugin_snapshot("alpha");
        assert!(alpha_snapshot.summary.is_none());
        assert!(alpha_snapshot.providers.is_empty());
        assert!(alpha_snapshot.payloads.is_empty());
        assert!(alpha_snapshot.endpoints.is_empty());
        assert!(collector
            .plugin_endpoint_snapshot("alpha", "chat")
            .is_none());

        let beta_snapshot = collector.plugin_snapshot("beta");
        assert_eq!(
            beta_snapshot
                .summary
                .as_ref()
                .map(|summary| summary.name.as_str()),
            Some("beta")
        );
        assert_eq!(
            beta_snapshot.payloads.get("metrics"),
            Some(&json!({"requests": 7}))
        );
        assert_eq!(beta_snapshot.endpoints.len(), 1);
        assert_eq!(beta_snapshot.endpoints[0].endpoint_id, "embed");
        assert!(collector
            .plugin_endpoint_snapshot("beta", "embed")
            .is_some());

        let all = collector.plugins_snapshot();
        assert_eq!(
            all.plugins
                .iter()
                .map(|plugin| plugin.name.as_str())
                .collect::<Vec<_>>(),
            vec!["beta"]
        );
        assert_eq!(
            all.endpoints
                .iter()
                .map(|endpoint| (endpoint.plugin_name.as_str(), endpoint.endpoint_id.as_str()))
                .collect::<Vec<_>>(),
            vec![("beta", "embed")]
        );
    }

    async fn start_local_http_server(response: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((mut conn, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = conn.read(&mut buf).await;
                let _ = conn.write_all(response.as_bytes()).await;
                let _ = conn.shutdown().await;
            }
        });
        port
    }

    async fn connected_proxy_stream() -> (TcpStream, tokio::task::JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let client_reader = tokio::spawn(async move {
            let mut client = client;
            let mut buf = Vec::new();
            client.read_to_end(&mut buf).await.unwrap();
            buf
        });
        (server, client_reader)
    }

    #[tokio::test]
    async fn runtime_data_routing_snapshot_reflects_proxy_attempts_and_inflight() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let collector = node.runtime_data_collector();
        let upstream_port = start_local_http_server(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 33\r\n\r\n{\"usage\":{\"completion_tokens\":7}}",
        )
        .await;
        let (proxy_stream, client_reader) = connected_proxy_stream().await;

        let routed = transport::route_to_target(
            node.clone(),
            proxy_stream,
            Some("glm"),
            election::InferenceTarget::Local(upstream_port),
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
            ResponseAdapter::None,
        )
        .await;

        assert!(routed);
        let response = String::from_utf8(client_reader.await.unwrap()).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));

        let snapshot = collector.routing_snapshot();
        assert_eq!(snapshot.status.request_count, 1);
        assert_eq!(snapshot.status.successful_requests, 1);
        assert_eq!(snapshot.status.local_node.current_inflight_requests, 0);
        assert_eq!(snapshot.status.local_node.peak_inflight_requests, 1);
        assert_eq!(snapshot.status.local_node.local_attempt_count, 1);
        assert_eq!(snapshot.status.completion_tokens_observed, 7);
        assert_eq!(snapshot.status.pressure.fronted_request_count, 1);
        assert_eq!(snapshot.status.pressure.locally_served_request_count, 1);

        let model = snapshot
            .models
            .get("glm")
            .expect("glm model snapshot present");
        assert_eq!(model.request_count, 1);
        assert_eq!(model.successful_requests, 1);
        assert_eq!(model.completion_tokens_observed, 7);
        assert_eq!(model.targets.len(), 1);
        assert_eq!(model.targets[0].kind, "local");
        assert_eq!(model.targets[0].attempt_count, 1);
        assert_eq!(model.targets[0].success_count, 1);
    }

    #[tokio::test]
    async fn runtime_data_request_updates_stay_non_blocking() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let collector = node.runtime_data_collector();
        let mut subscription = collector.subscribe();

        let guard = node.begin_inflight_request();
        assert!(subscription.has_changed().expect("watch channel open"));
        let opened = *subscription.borrow_and_update();
        assert_eq!(opened.version.get(), 1);
        assert!(opened.dirty.contains(RuntimeDataDirty::ROUTING));
        assert_eq!(
            collector
                .routing_snapshot()
                .status
                .local_node
                .current_inflight_requests,
            1
        );

        node.record_inference_attempt(
            Some("glm"),
            &election::InferenceTarget::Local(9337),
            std::time::Duration::from_millis(3),
            std::time::Duration::from_millis(12),
            crate::network::metrics::AttemptOutcome::Success,
            Some(5),
        );
        assert!(subscription.has_changed().expect("watch channel open"));
        let attempted = *subscription.borrow_and_update();
        assert_eq!(attempted.version.get(), 2);
        assert!(attempted.dirty.contains(RuntimeDataDirty::ROUTING));

        node.record_routed_request(
            Some("glm"),
            1,
            crate::network::metrics::RequestOutcome::Success(
                crate::network::metrics::RequestService::Local,
            ),
        );
        assert!(subscription.has_changed().expect("watch channel open"));
        let requested = *subscription.borrow_and_update();
        assert_eq!(requested.version.get(), 3);
        assert!(requested.dirty.contains(RuntimeDataDirty::ROUTING));

        drop(guard);
        assert!(subscription.has_changed().expect("watch channel open"));
        let completed = *subscription.borrow_and_update();
        assert_eq!(completed.version.get(), 4);
        assert!(completed.dirty.contains(RuntimeDataDirty::ROUTING));

        let snapshot = collector.routing_snapshot();
        assert_eq!(snapshot.status.request_count, 1);
        assert_eq!(snapshot.status.successful_requests, 1);
        assert_eq!(snapshot.status.local_node.current_inflight_requests, 0);
        assert_eq!(snapshot.status.local_node.peak_inflight_requests, 1);
        assert_eq!(snapshot.status.local_node.local_attempt_count, 1);
        assert_eq!(snapshot.models["glm"].request_count, 1);
        assert_eq!(snapshot.models["glm"].targets[0].success_count, 1);
    }
}
