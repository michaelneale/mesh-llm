use crate::client::builder::Model;

#[derive(Debug, Clone)]
pub enum Event {
    Connecting,
    Joined { node_id: String },
    ModelsUpdated { models: Vec<Model> },
    TokenDelta { request_id: String, delta: String },
    Completed { request_id: String },
    Failed { request_id: String, error: String },
    Disconnected { reason: String },
}

pub trait EventListener: Send + Sync + 'static {
    fn on_event(&self, event: Event);
}
