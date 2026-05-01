use crate::api::RuntimeProcessPayload;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RuntimeProcessSnapshot {
    pub model: String,
    pub backend: String,
    pub pid: u32,
    pub port: u16,
    pub slots: usize,
    pub context_length: Option<u32>,
    pub command: Option<String>,
    pub state: String,
    pub start: Option<i64>,
    pub health: Option<String>,
}

impl RuntimeProcessSnapshot {
    pub(crate) fn from_payload(payload: &RuntimeProcessPayload) -> Self {
        Self {
            model: payload.name.clone(),
            backend: payload.backend.clone(),
            pid: payload.pid,
            port: payload.port,
            slots: payload.slots,
            context_length: payload.context_length,
            command: None,
            state: payload.status.clone(),
            start: None,
            health: Some(payload.status.clone()),
        }
    }

    pub(crate) fn to_payload(&self) -> RuntimeProcessPayload {
        RuntimeProcessPayload {
            name: self.model.clone(),
            backend: self.backend.clone(),
            status: self.state.clone(),
            port: self.port,
            pid: self.pid,
            slots: self.slots,
            context_length: self.context_length,
        }
    }
}

pub(crate) fn runtime_process_payloads(
    rows: &[RuntimeProcessSnapshot],
) -> Vec<RuntimeProcessPayload> {
    rows.iter()
        .map(RuntimeProcessSnapshot::to_payload)
        .collect()
}

pub(crate) fn upsert_runtime_process_snapshot(
    rows: &mut Vec<RuntimeProcessSnapshot>,
    snapshot: RuntimeProcessSnapshot,
) -> bool {
    if let Some(existing) = rows
        .iter_mut()
        .find(|existing| existing.model == snapshot.model)
    {
        if *existing == snapshot {
            return false;
        }
        *existing = snapshot;
        return true;
    }

    rows.push(snapshot);
    true
}

pub(crate) fn remove_runtime_process_snapshot(
    rows: &mut Vec<RuntimeProcessSnapshot>,
    model_name: &str,
) -> bool {
    let before = rows.len();
    rows.retain(|existing| existing.model != model_name);
    rows.len() != before
}
