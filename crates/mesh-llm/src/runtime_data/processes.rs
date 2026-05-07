use crate::api::RuntimeProcessPayload;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RuntimeProcessSnapshot {
    pub model: String,
    pub instance_id: Option<String>,
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
            instance_id: payload.instance_id.clone(),
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
            instance_id: self.instance_id.clone(),
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
    if let Some(existing) = rows.iter_mut().find(|existing| {
        runtime_process_snapshot_identity(existing) == runtime_process_snapshot_identity(&snapshot)
    }) {
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
    target: &str,
) -> bool {
    let before = rows.len();
    let has_instance_match = rows
        .iter()
        .any(|existing| existing.instance_id.as_deref() == Some(target));
    rows.retain(|existing| {
        if has_instance_match {
            existing.instance_id.as_deref() != Some(target)
        } else {
            existing.model != target
        }
    });
    rows.len() != before
}

fn runtime_process_snapshot_identity(snapshot: &RuntimeProcessSnapshot) -> &str {
    snapshot.instance_id.as_deref().unwrap_or(&snapshot.model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(model: &str, instance_id: Option<&str>, port: u16) -> RuntimeProcessSnapshot {
        RuntimeProcessSnapshot {
            model: model.to_string(),
            instance_id: instance_id.map(str::to_string),
            backend: "skippy".to_string(),
            pid: 100,
            port,
            slots: 4,
            context_length: Some(8192),
            command: None,
            state: "ready".to_string(),
            start: None,
            health: Some("ready".to_string()),
        }
    }

    #[test]
    fn process_snapshots_keep_distinct_same_model_instances() {
        let mut rows = Vec::new();

        assert!(upsert_runtime_process_snapshot(
            &mut rows,
            snapshot("Qwen", Some("runtime-1"), 41001)
        ));
        assert!(upsert_runtime_process_snapshot(
            &mut rows,
            snapshot("Qwen", Some("runtime-2"), 41002)
        ));
        assert_eq!(rows.len(), 2);

        assert!(upsert_runtime_process_snapshot(
            &mut rows,
            snapshot("Qwen", Some("runtime-2"), 41003)
        ));
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows.iter()
                .find(|row| row.instance_id.as_deref() == Some("runtime-2"))
                .map(|row| row.port),
            Some(41003)
        );
    }

    #[test]
    fn remove_process_snapshot_accepts_instance_id_without_dropping_siblings() {
        let mut rows = vec![
            snapshot("Qwen", Some("runtime-1"), 41001),
            snapshot("Qwen", Some("runtime-2"), 41002),
        ];

        assert!(remove_runtime_process_snapshot(&mut rows, "runtime-1"));

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].instance_id.as_deref(), Some("runtime-2"));
    }
}
