use anyhow::Result;
use mesh_llm_plugin::{
    accept_bulk_transfer_message, bulk_transfer_sequence, cancel_task_result, complete_result,
    empty_object_schema, get_prompt_result, get_task_payload_result, get_task_result,
    json_reply_channel_message, json_schema_tool, json_string, list_tasks, parse_optional_json,
    plugin_server_info_full, prompt, prompt_argument, proto, read_resource_result, task,
    tool_with_schema, PluginRuntime, PromptRouter, ResourceRouter, SubscriptionSet, TaskStore,
    ToolRouter,
};
use rmcp::model::{
    AnnotateAble, LoggingLevel, PromptMessage, PromptMessageRole, ServerInfo, TaskStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::Mutex;

const PLUGIN_ID: &str = "example";
const EXAMPLE_CHANNEL: &str = "example.v1";
const DEFAULT_BULK_CHUNK_SIZE: usize = 16 * 1024;
const DEFAULT_SNAPSHOT_LIMIT: usize = 20;
const MAX_RECENT_ITEMS: usize = 128;

#[derive(Debug, Deserialize, Default)]
struct SnapshotParams {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SendMessageArguments {
    text: String,
    #[serde(default)]
    target_peer_id: Option<String>,
    #[serde(default)]
    request_ack: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SendBulkArguments {
    text: String,
    #[serde(default)]
    target_peer_id: Option<String>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    request_ack: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct ClearParams {}

#[derive(Clone, Debug, Serialize)]
struct PeerSummary {
    peer_id: String,
    role: String,
    version: String,
    capabilities: Vec<String>,
    models: Vec<String>,
    serving_models: Vec<String>,
    rtt_ms: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedMeshEvent {
    kind: String,
    peer_id: Option<String>,
    local_peer_id: String,
    mesh_id: String,
    detail_json: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedChannelMessage {
    direction: String,
    channel: String,
    source_peer_id: String,
    target_peer_id: String,
    message_kind: String,
    correlation_id: String,
    content_type: String,
    metadata_json: Option<serde_json::Value>,
    text_preview: String,
    body_len: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedBulkEvent {
    direction: String,
    kind: String,
    transfer_id: String,
    source_peer_id: String,
    target_peer_id: String,
    correlation_id: String,
    content_type: String,
    metadata_json: Option<serde_json::Value>,
    total_bytes: u64,
    offset: u64,
    body_len: usize,
    final_chunk: bool,
}

#[derive(Clone, Debug, Serialize)]
struct CompletedTransfer {
    transfer_id: String,
    source_peer_id: String,
    target_peer_id: String,
    content_type: String,
    total_bytes: u64,
    received_bytes: u64,
    preview: String,
}

#[derive(Default)]
struct TransferAccumulator {
    source_peer_id: String,
    target_peer_id: String,
    content_type: String,
    total_bytes: u64,
    bytes: Vec<u8>,
}

struct ExampleState {
    local_peer_id: String,
    mesh_id: String,
    known_peers: BTreeMap<String, PeerSummary>,
    mesh_events: Vec<RecordedMeshEvent>,
    channel_messages: Vec<RecordedChannelMessage>,
    bulk_events: Vec<RecordedBulkEvent>,
    completed_transfers: BTreeMap<String, CompletedTransfer>,
    transfer_state: HashMap<String, TransferAccumulator>,
    sent_channel_messages: usize,
    sent_bulk_transfers: usize,
    next_id: u64,
    subscriptions: SubscriptionSet,
    log_level: LoggingLevel,
    tasks: TaskStore<serde_json::Value>,
}

impl ExampleState {
    fn next_token(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("{prefix}-{}-{}", now_millis(), self.next_id)
    }

    fn snapshot(&self, limit: usize) -> serde_json::Value {
        let limit = limit.max(1);
        let completed_transfers = self
            .completed_transfers
            .values()
            .cloned()
            .collect::<Vec<_>>();
        json!({
            "plugin": PLUGIN_ID,
            "channel": EXAMPLE_CHANNEL,
            "local_peer_id": self.local_peer_id,
            "mesh_id": self.mesh_id,
            "known_peers": self.known_peers.values().cloned().collect::<Vec<_>>(),
            "stats": {
                "known_peer_count": self.known_peers.len(),
                "mesh_event_count": self.mesh_events.len(),
                "channel_message_count": self.channel_messages.len(),
                "bulk_event_count": self.bulk_events.len(),
                "completed_transfer_count": self.completed_transfers.len(),
                "sent_channel_messages": self.sent_channel_messages,
                "sent_bulk_transfers": self.sent_bulk_transfers,
            },
            "recent_mesh_events": recent_items(&self.mesh_events, limit),
            "recent_channel_messages": recent_items(&self.channel_messages, limit),
            "recent_bulk_events": recent_items(&self.bulk_events, limit),
            "completed_transfers": recent_items(&completed_transfers, limit),
            "subscriptions": self.subscriptions.list(),
            "log_level": format!("{:?}", self.log_level).to_lowercase(),
            "tasks": self.tasks.values().map(|task| json!({
                "task_id": task.task.task_id,
                "status": task_status_name(&task.task.status),
                "status_message": task.task.status_message,
            })).collect::<Vec<_>>(),
        })
    }

    fn clear_history(&mut self) {
        self.mesh_events.clear();
        self.channel_messages.clear();
        self.bulk_events.clear();
        self.completed_transfers.clear();
        self.transfer_state.clear();
        self.sent_channel_messages = 0;
        self.sent_bulk_transfers = 0;
    }

    fn record_mesh_event(&mut self, event: &proto::MeshEvent) {
        if !event.local_peer_id.is_empty() {
            self.local_peer_id = event.local_peer_id.clone();
        }
        if !event.mesh_id.is_empty() {
            self.mesh_id = event.mesh_id.clone();
        }
        if let Some(peer) = &event.peer {
            let peer_id = peer.peer_id.clone();
            match proto::mesh_event::Kind::try_from(event.kind).ok() {
                Some(proto::mesh_event::Kind::PeerDown) => {
                    self.known_peers.remove(&peer_id);
                }
                _ => {
                    self.known_peers.insert(peer_id, peer_summary(peer));
                }
            }
        }

        push_bounded(
            &mut self.mesh_events,
            RecordedMeshEvent {
                kind: mesh_event_kind_name(event.kind).into(),
                peer_id: event.peer.as_ref().map(|peer| peer.peer_id.clone()),
                local_peer_id: event.local_peer_id.clone(),
                mesh_id: event.mesh_id.clone(),
                detail_json: parse_optional_json(&event.detail_json),
            },
        );
    }

    fn record_channel_message(&mut self, direction: &str, message: &proto::ChannelMessage) {
        if direction == "outbound" {
            self.sent_channel_messages += 1;
        }
        push_bounded(
            &mut self.channel_messages,
            RecordedChannelMessage {
                direction: direction.to_string(),
                channel: message.channel.clone(),
                source_peer_id: message.source_peer_id.clone(),
                target_peer_id: message.target_peer_id.clone(),
                message_kind: message.message_kind.clone(),
                correlation_id: message.correlation_id.clone(),
                content_type: message.content_type.clone(),
                metadata_json: parse_optional_json(&message.metadata_json),
                text_preview: preview_bytes(&message.body),
                body_len: message.body.len(),
            },
        );
    }

    fn record_bulk_message(&mut self, direction: &str, message: &proto::BulkTransferMessage) {
        if direction == "outbound"
            && matches!(
                proto::bulk_transfer_message::Kind::try_from(message.kind).ok(),
                Some(proto::bulk_transfer_message::Kind::Offer)
            )
        {
            self.sent_bulk_transfers += 1;
        }
        push_bounded(
            &mut self.bulk_events,
            RecordedBulkEvent {
                direction: direction.to_string(),
                kind: bulk_kind_name(message.kind).into(),
                transfer_id: message.transfer_id.clone(),
                source_peer_id: message.source_peer_id.clone(),
                target_peer_id: message.target_peer_id.clone(),
                correlation_id: message.correlation_id.clone(),
                content_type: message.content_type.clone(),
                metadata_json: parse_optional_json(&message.metadata_json),
                total_bytes: message.total_bytes,
                offset: message.offset,
                body_len: message.body.len(),
                final_chunk: message.final_chunk,
            },
        );
    }

    fn note_bulk_receive(&mut self, message: &proto::BulkTransferMessage) {
        match proto::bulk_transfer_message::Kind::try_from(message.kind).ok() {
            Some(proto::bulk_transfer_message::Kind::Offer) => {
                self.transfer_state.insert(
                    message.transfer_id.clone(),
                    TransferAccumulator {
                        source_peer_id: message.source_peer_id.clone(),
                        target_peer_id: message.target_peer_id.clone(),
                        content_type: message.content_type.clone(),
                        total_bytes: message.total_bytes,
                        bytes: Vec::new(),
                    },
                );
            }
            Some(proto::bulk_transfer_message::Kind::Chunk) => {
                let entry = self
                    .transfer_state
                    .entry(message.transfer_id.clone())
                    .or_default();
                if entry.source_peer_id.is_empty() {
                    entry.source_peer_id = message.source_peer_id.clone();
                }
                if entry.target_peer_id.is_empty() {
                    entry.target_peer_id = message.target_peer_id.clone();
                }
                if entry.content_type.is_empty() {
                    entry.content_type = message.content_type.clone();
                }
                if entry.total_bytes == 0 {
                    entry.total_bytes = message.total_bytes;
                }
                entry.bytes.extend_from_slice(&message.body);
            }
            Some(proto::bulk_transfer_message::Kind::Complete) => {
                if let Some(entry) = self.transfer_state.remove(&message.transfer_id) {
                    self.completed_transfers.insert(
                        message.transfer_id.clone(),
                        CompletedTransfer {
                            transfer_id: message.transfer_id.clone(),
                            source_peer_id: entry.source_peer_id,
                            target_peer_id: entry.target_peer_id,
                            content_type: entry.content_type,
                            total_bytes: entry.total_bytes,
                            received_bytes: entry.bytes.len() as u64,
                            preview: preview_bytes(&entry.bytes),
                        },
                    );
                }
            }
            _ => {}
        }
    }

    fn resource_snapshot(&self) -> serde_json::Value {
        json!({
            "plugin": PLUGIN_ID,
            "channel": EXAMPLE_CHANNEL,
            "local_peer_id": self.local_peer_id,
            "mesh_id": self.mesh_id,
            "log_level": format!("{:?}", self.log_level).to_lowercase(),
            "subscriptions": self.subscriptions.list(),
        })
    }
}

impl Default for ExampleState {
    fn default() -> Self {
        let (bootstrap_task, bootstrap_payload) = example_task(
            "example-bootstrap",
            TaskStatus::Completed,
            "Bootstrap complete",
            json!({
                "ok": true,
                "task": "bootstrap",
                "plugin": PLUGIN_ID,
            }),
        );
        let (long_running_task, long_running_payload) = example_task(
            "example-watch",
            TaskStatus::Working,
            "Watching mesh events",
            json!({
                "ok": true,
                "task": "watch",
                "state": "working",
            }),
        );

        let mut tasks = TaskStore::default();
        tasks.insert(bootstrap_task, bootstrap_payload);
        tasks.insert(long_running_task, long_running_payload);

        Self {
            local_peer_id: String::new(),
            mesh_id: String::new(),
            known_peers: BTreeMap::new(),
            mesh_events: Vec::new(),
            channel_messages: Vec::new(),
            bulk_events: Vec::new(),
            completed_transfers: BTreeMap::new(),
            transfer_state: HashMap::new(),
            sent_channel_messages: 0,
            sent_bulk_transfers: 0,
            next_id: 0,
            subscriptions: SubscriptionSet::default(),
            log_level: LoggingLevel::Info,
            tasks,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let state = Arc::new(Mutex::new(ExampleState::default()));
    PluginRuntime::run(build_example_plugin(state)).await
}

fn build_example_plugin(state: Arc<Mutex<ExampleState>>) -> mesh_llm_plugin::SimplePlugin {
    let health_state = state.clone();
    let subscribe_state = state.clone();
    let unsubscribe_state = state.clone();
    let set_log_level_state = state.clone();
    let task_state = state.clone();
    let task_info_state = state.clone();
    let task_result_state = state.clone();
    let cancel_task_state = state.clone();
    let channel_state = state.clone();
    let bulk_state = state.clone();
    let mesh_event_state = state.clone();

    mesh_llm_plugin::SimplePlugin::new(
        mesh_llm_plugin::PluginMetadata::new(PLUGIN_ID, env!("CARGO_PKG_VERSION"), server_info())
            .with_capabilities(vec![
                format!("channel:{EXAMPLE_CHANNEL}"),
                "bulk:example".into(),
                "mesh-events".into(),
            ]),
    )
    .with_tool_router(tool_router(state.clone()))
    .with_prompt_router(prompt_router(state.clone()))
    .with_resource_router(resource_router(state.clone()))
    .with_completion_router(completion_router())
    .with_task_router(
        mesh_llm_plugin::TaskRouter::new()
            .with_list(move |_request, _context| {
                let state = task_state.clone();
                Box::pin(async move {
                    let state = state.lock().await;
                    Ok(list_tasks(state.tasks.list()))
                })
            })
            .with_get_info(move |request, _context| {
                let state = task_info_state.clone();
                Box::pin(async move {
                    let state = state.lock().await;
                    let task = state.tasks.get(&request.task_id)?;
                    Ok(get_task_result(task.task.clone()))
                })
            })
            .with_get_result(move |request, _context| {
                let state = task_result_state.clone();
                Box::pin(async move {
                    let state = state.lock().await;
                    let task = state.tasks.get(&request.task_id)?;
                    get_task_payload_result(task.payload.clone())
                })
            })
            .with_cancel(move |request, _context| {
                let state = cancel_task_state.clone();
                Box::pin(async move {
                    let mut state = state.lock().await;
                    let task = state.tasks.get_mut(&request.task_id)?;
                    task.task.status = TaskStatus::Cancelled;
                    task.task.status_message = Some("Cancelled by MCP client".into());
                    task.payload = json!({
                        "ok": false,
                        "cancelled": true,
                        "task_id": task.task.task_id,
                    });
                    Ok(cancel_task_result(task.task.clone()))
                })
            }),
    )
    .with_health(move |_context| {
        let state = health_state.clone();
        Box::pin(async move {
            let state = state.lock().await;
            Ok(format!(
                "peers={} messages={} bulk={} mesh_events={}",
                state.known_peers.len(),
                state.channel_messages.len(),
                state.bulk_events.len(),
                state.mesh_events.len()
            ))
        })
    })
    .with_subscribe_resource(move |request, _context| {
        let state = subscribe_state.clone();
        Box::pin(async move {
            state.lock().await.subscriptions.subscribe(request.uri);
            Ok(())
        })
    })
    .with_unsubscribe_resource(move |request, _context| {
        let state = unsubscribe_state.clone();
        Box::pin(async move {
            state.lock().await.subscriptions.unsubscribe(&request.uri);
            Ok(())
        })
    })
    .with_set_log_level(move |request, _context| {
        let state = set_log_level_state.clone();
        Box::pin(async move {
            state.lock().await.log_level = request.level;
            Ok(())
        })
    })
    .on_channel_message(move |message, context| {
        let state = channel_state.clone();
        Box::pin(async move {
            let mut state = state.lock().await;
            state.record_channel_message("inbound", &message);
            if should_ack_channel(&message) {
                let mut ack = json_reply_channel_message(
                    &message,
                    "example.ack",
                    &json!({
                        "acknowledged_kind": message.message_kind,
                        "received_bytes": message.body.len(),
                        "received_by": state.local_peer_id,
                    }),
                )?;
                if ack.correlation_id.is_empty() {
                    ack.correlation_id = state.next_token("ack");
                }
                ack.metadata_json = json_string(&json!({
                    "reply_to": message.correlation_id,
                }))?;
                state.record_channel_message("outbound", &ack);
                context.send_channel(ack).await?;
            }
            Ok(())
        })
    })
    .on_bulk_transfer_message(move |message, context| {
        let state = bulk_state.clone();
        Box::pin(async move {
            let mut state = state.lock().await;
            state.record_bulk_message("inbound", &message);
            state.note_bulk_receive(&message);
            if should_ack_bulk_offer(&message) {
                let mut ack = accept_bulk_transfer_message(&message);
                ack.metadata_json = json_string(&json!({
                    "accepted_by": state.local_peer_id,
                }))?;
                state.record_bulk_message("outbound", &ack);
                context.send_bulk(ack).await?;
            }
            Ok(())
        })
    })
    .on_mesh_event(move |event, _context| {
        let state = mesh_event_state.clone();
        Box::pin(async move {
            state.lock().await.record_mesh_event(&event);
            Ok(())
        })
    })
}

fn tool_router(state: Arc<Mutex<ExampleState>>) -> ToolRouter {
    let mut router = ToolRouter::new();

    let snapshot_state = state.clone();
    router.add_json_default::<SnapshotParams, serde_json::Value, _>(
        tool_with_schema(
            "snapshot",
            "Inspect the example plugin state: known peers, mesh events, recent channel messages, recent bulk transfers, and counters.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1 }
                }
            })
            .as_object()
            .cloned()
            .unwrap(),
        ),
        move |params, _context| {
            let state = snapshot_state.clone();
            Box::pin(async move {
                Ok(state
                    .lock()
                    .await
                    .snapshot(params.limit.unwrap_or(DEFAULT_SNAPSHOT_LIMIT)))
            })
        },
    );

    let send_message_state = state.clone();
    router.add_json::<SendMessageArguments, serde_json::Value, _>(
        json_schema_tool::<SendMessageArguments>(
            "send_message",
            "Send a plugin channel message to one peer or broadcast to all peers. Leave target_peer_id empty or set it to 'all' to broadcast.",
        ),
        move |params, context| {
            let state = send_message_state.clone();
            Box::pin(async move {
                let mut state = state.lock().await;
                let target_peer_id = normalize_target_peer_id(params.target_peer_id);
                let correlation_id = state.next_token("msg");
                let mut message = mesh_llm_plugin::channel_message(
                    EXAMPLE_CHANNEL,
                    target_peer_id.clone(),
                    "text/plain",
                    params.text.into_bytes(),
                    "example.message",
                );
                message.correlation_id = correlation_id.clone();
                message.metadata_json = json_string(&json!({
                    "request_ack": params.request_ack.unwrap_or(true),
                }))?;
                state.record_channel_message("outbound", &message);
                context.send_channel(message).await?;
                Ok(json!({
                    "ok": true,
                    "channel": EXAMPLE_CHANNEL,
                    "target_peer_id": render_target(&target_peer_id),
                    "correlation_id": correlation_id,
                }))
            })
        },
    );

    let send_bulk_state = state.clone();
    router.add_json::<SendBulkArguments, serde_json::Value, _>(
        json_schema_tool::<SendBulkArguments>(
            "send_bulk",
            "Send a bulk transfer to one peer or broadcast to all peers. This emits OFFER, CHUNK, and COMPLETE frames so the full bulk transport path is exercised.",
        ),
        move |params, context| {
            let state = send_bulk_state.clone();
            Box::pin(async move {
                let mut state = state.lock().await;
                let target_peer_id = normalize_target_peer_id(params.target_peer_id);
                let correlation_id = state.next_token("bulk-corr");
                let transfer_id = state.next_token("bulk");
                let bytes = params.text.into_bytes();
                let chunk_size = params.chunk_size.unwrap_or(DEFAULT_BULK_CHUNK_SIZE).max(1);
                let metadata_json = json_string(&json!({
                    "request_ack": params.request_ack.unwrap_or(true),
                }))?;

                let sequence = bulk_transfer_sequence(
                    EXAMPLE_CHANNEL,
                    target_peer_id.clone(),
                    "text/plain",
                    bytes,
                    chunk_size,
                    correlation_id.clone(),
                    transfer_id.clone(),
                    metadata_json,
                );
                for message in sequence.messages {
                    state.record_bulk_message("outbound", &message);
                    context.send_bulk(message).await?;
                }

                Ok(json!({
                    "ok": true,
                    "transfer_id": sequence.transfer_id,
                    "correlation_id": sequence.correlation_id,
                    "target_peer_id": render_target(&target_peer_id),
                    "chunk_size": chunk_size,
                }))
            })
        },
    );

    let clear_state = state.clone();
    router.add_json_default::<ClearParams, serde_json::Value, _>(
        tool_with_schema(
            "clear",
            "Clear recorded example-plugin history while keeping the current peer snapshot.",
            empty_object_schema(),
        ),
        move |_params, _context| {
            let state = clear_state.clone();
            Box::pin(async move {
                state.lock().await.clear_history();
                Ok(json!({
                    "ok": true,
                    "cleared": ["mesh_events", "channel_messages", "bulk_events", "completed_transfers"],
                }))
            })
        },
    );

    router
}

fn server_info() -> ServerInfo {
    plugin_server_info_full(
        PLUGIN_ID,
        env!("CARGO_PKG_VERSION"),
        "Plugin Surface Example",
        "Standalone example plugin that exercises tools, prompts, resources, completion, logging, tasks, channel messages, bulk transfers, and mesh events.",
        None::<String>,
    )
}

fn prompt_router(state: Arc<Mutex<ExampleState>>) -> PromptRouter {
    let mut router = PromptRouter::new();

    let status_state = state.clone();
    router.add(
        prompt(
            "status_brief",
            "Create a short status brief summarizing the current example plugin state.",
            Some(vec![
                prompt_argument("topic", "Topic to emphasize in the brief.", false),
                prompt_argument("audience", "Target audience for the brief.", false),
            ]),
        ),
        move |request, _context| {
            let state = status_state.clone();
            Box::pin(async move {
                let args = request.arguments.unwrap_or_default();
                let state = state.lock().await;
                let topic = args
                    .get("topic")
                    .and_then(|v| v.as_str())
                    .unwrap_or("mesh health");
                let audience = args
                    .get("audience")
                    .and_then(|v| v.as_str())
                    .unwrap_or("operators");
                Ok(get_prompt_result(vec![
                    PromptMessage::new_text(
                        PromptMessageRole::User,
                        format!("Write a concise status brief for {audience}. Focus on {topic}."),
                    ),
                    PromptMessage::new_text(PromptMessageRole::User, state.snapshot(5).to_string()),
                ]))
            })
        },
    );

    let peer_state = state.clone();
    router.add(
        prompt(
            "peer_focus",
            "Summarize a specific peer from the current example state.",
            Some(vec![prompt_argument(
                "peer_id",
                "Peer ID to focus on.",
                true,
            )]),
        ),
        move |request, _context| {
            let state = peer_state.clone();
            Box::pin(async move {
                let args = request.arguments.unwrap_or_default();
                let peer_id = args
                    .get("peer_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let state = state.lock().await;
                let peer = state
                    .known_peers
                    .get(peer_id)
                    .map(|peer| serde_json::to_string(peer).unwrap_or_else(|_| "{}".into()))
                    .unwrap_or_else(|| "{\"missing\":true}".into());
                Ok(get_prompt_result(vec![
                    PromptMessage::new_text(
                        PromptMessageRole::User,
                        format!("Summarize peer {peer_id} and its current role in the mesh."),
                    ),
                    PromptMessage::new_text(PromptMessageRole::User, peer),
                ]))
            })
        },
    );

    router
}

fn resource_router(state: Arc<Mutex<ExampleState>>) -> ResourceRouter {
    let mut router = ResourceRouter::new();

    let snapshot_state = state.clone();
    router.add_exact(
        rmcp::model::RawResource::new("example://snapshot", "Example Snapshot")
            .with_description("Current high-level example plugin snapshot.")
            .with_mime_type("application/json")
            .no_annotation(),
        move |request, _context| {
            let state = snapshot_state.clone();
            Box::pin(async move {
                let state = state.lock().await;
                Ok(read_resource_result(vec![
                    rmcp::model::ResourceContents::text(
                        state.resource_snapshot().to_string(),
                        request.uri,
                    )
                    .with_mime_type("application/json"),
                ]))
            })
        },
    );

    let peers_state = state.clone();
    router.add_exact(
        rmcp::model::RawResource::new("example://peers", "Known Peers")
            .with_description("Current peer inventory seen by the example plugin.")
            .with_mime_type("application/json")
            .no_annotation(),
        move |request, _context| {
            let state = peers_state.clone();
            Box::pin(async move {
                let state = state.lock().await;
                let payload =
                    serde_json::to_string(&state.known_peers.values().cloned().collect::<Vec<_>>())
                        .map_err(|err| mesh_llm_plugin::PluginError::internal(err.to_string()))?;
                Ok(read_resource_result(vec![
                    rmcp::model::ResourceContents::text(payload, request.uri)
                        .with_mime_type("application/json"),
                ]))
            })
        },
    );

    let peer_state = state.clone();
    router.add_prefix_template(
        rmcp::model::RawResourceTemplate::new("example://peer/{peer_id}", "Peer Detail")
            .with_description("Dynamic resource for a specific peer.")
            .with_mime_type("application/json")
            .no_annotation(),
        "example://peer/",
        move |request, _context| {
            let state = peer_state.clone();
            Box::pin(async move {
                let peer_id = request.uri.trim_start_matches("example://peer/");
                let state = state.lock().await;
                let payload = state
                    .known_peers
                    .get(peer_id)
                    .map(|peer| serde_json::to_string(peer).unwrap_or_else(|_| "{}".into()))
                    .unwrap_or_else(|| "{\"missing\":true}".into());
                Ok(read_resource_result(vec![
                    rmcp::model::ResourceContents::text(payload, request.uri)
                        .with_mime_type("application/json"),
                ]))
            })
        },
    );

    router
}

fn completion_router() -> mesh_llm_plugin::CompletionRouter {
    let mut router = mesh_llm_plugin::CompletionRouter::new();
    router.add_prompt_argument_values(
        "status_brief",
        "topic",
        vec![
            "mesh health".into(),
            "plugin runtime".into(),
            "bulk transfers".into(),
        ],
    );
    router.add_prompt_argument_values(
        "status_brief",
        "audience",
        vec!["operators".into(), "developers".into(), "testers".into()],
    );
    router.add_prompt_argument_values(
        "peer_focus",
        "peer_id",
        vec!["peer-alpha".into(), "peer-beta".into()],
    );
    router.add_resource("example://peer/{peer_id}", move |_request, _context| {
        Box::pin(async move { complete_result(vec!["peer-alpha".into(), "peer-beta".into()]) })
    });
    router
}

fn example_task(
    id: &str,
    status: TaskStatus,
    status_message: &str,
    payload: serde_json::Value,
) -> (rmcp::model::Task, serde_json::Value) {
    let task = task(id, status, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
        .with_status_message(status_message)
        .with_poll_interval(1000);
    (task, payload)
}

fn task_status_name(status: &TaskStatus) -> &'static str {
    match status {
        TaskStatus::Working => "working",
        TaskStatus::InputRequired => "input_required",
        TaskStatus::Completed => "completed",
        TaskStatus::Failed => "failed",
        TaskStatus::Cancelled => "cancelled",
    }
}

fn should_ack_channel(message: &proto::ChannelMessage) -> bool {
    !message.source_peer_id.is_empty()
        && message.message_kind != "example.ack"
        && parse_optional_json(&message.metadata_json)
            .and_then(|value| value.get("request_ack").and_then(|v| v.as_bool()))
            .unwrap_or(false)
}

fn should_ack_bulk_offer(message: &proto::BulkTransferMessage) -> bool {
    matches!(
        proto::bulk_transfer_message::Kind::try_from(message.kind).ok(),
        Some(proto::bulk_transfer_message::Kind::Offer)
    ) && !message.source_peer_id.is_empty()
        && parse_optional_json(&message.metadata_json)
            .and_then(|value| value.get("request_ack").and_then(|v| v.as_bool()))
            .unwrap_or(false)
}

fn preview_bytes(bytes: &[u8]) -> String {
    let mut preview = String::from_utf8_lossy(bytes).to_string();
    if preview.len() > 160 {
        preview.truncate(160);
        preview.push_str("...");
    }
    preview
}

fn normalize_target_peer_id(target_peer_id: Option<String>) -> String {
    let Some(target_peer_id) = target_peer_id else {
        return String::new();
    };
    let trimmed = target_peer_id.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("all")
        || trimmed.eq_ignore_ascii_case("broadcast")
        || trimmed == "*"
    {
        String::new()
    } else {
        trimmed.to_string()
    }
}

fn render_target(target_peer_id: &str) -> String {
    if target_peer_id.is_empty() {
        "all".into()
    } else {
        target_peer_id.to_string()
    }
}

fn push_bounded<T>(items: &mut Vec<T>, item: T) {
    items.push(item);
    if items.len() > MAX_RECENT_ITEMS {
        let overflow = items.len() - MAX_RECENT_ITEMS;
        items.drain(0..overflow);
    }
}

fn recent_items<T: Clone>(items: &[T], limit: usize) -> Vec<T> {
    let len = items.len();
    let start = len.saturating_sub(limit);
    items[start..].to_vec()
}

fn peer_summary(peer: &proto::MeshPeer) -> PeerSummary {
    PeerSummary {
        peer_id: peer.peer_id.clone(),
        role: peer.role.clone(),
        version: peer.version.clone(),
        capabilities: peer.capabilities.clone(),
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        rtt_ms: peer.rtt_ms,
    }
}

fn mesh_event_kind_name(kind: i32) -> &'static str {
    match proto::mesh_event::Kind::try_from(kind).ok() {
        Some(proto::mesh_event::Kind::PeerUp) => "peer_up",
        Some(proto::mesh_event::Kind::PeerDown) => "peer_down",
        Some(proto::mesh_event::Kind::PeerUpdated) => "peer_updated",
        Some(proto::mesh_event::Kind::LocalAccepting) => "local_accepting",
        Some(proto::mesh_event::Kind::LocalStandby) => "local_standby",
        Some(proto::mesh_event::Kind::MeshIdUpdated) => "mesh_id_updated",
        _ => "unknown",
    }
}

fn bulk_kind_name(kind: i32) -> &'static str {
    match proto::bulk_transfer_message::Kind::try_from(kind).ok() {
        Some(proto::bulk_transfer_message::Kind::Offer) => "offer",
        Some(proto::bulk_transfer_message::Kind::Accept) => "accept",
        Some(proto::bulk_transfer_message::Kind::Reject) => "reject",
        Some(proto::bulk_transfer_message::Kind::Chunk) => "chunk",
        Some(proto::bulk_transfer_message::Kind::Complete) => "complete",
        Some(proto::bulk_transfer_message::Kind::Cancel) => "cancel",
        Some(proto::bulk_transfer_message::Kind::Error) => "error",
        _ => "unknown",
    }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
