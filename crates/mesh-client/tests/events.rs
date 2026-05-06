use mesh_client::events::{Event, EventListener};
use std::sync::{Arc, Mutex};

struct MockListener {
    events: Arc<Mutex<Vec<Event>>>,
}

impl EventListener for MockListener {
    fn on_event(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }
}

#[test]
fn all_7_event_variants_can_be_emitted() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let listener = MockListener {
        events: events.clone(),
    };

    listener.on_event(Event::Connecting);
    listener.on_event(Event::Joined {
        node_id: "abc".to_string(),
    });
    listener.on_event(Event::ModelsUpdated { models: vec![] });
    listener.on_event(Event::TokenDelta {
        request_id: "r1".to_string(),
        delta: "hello".to_string(),
    });
    listener.on_event(Event::Completed {
        request_id: "r1".to_string(),
    });
    listener.on_event(Event::Failed {
        request_id: "r2".to_string(),
        error: "oops".to_string(),
    });
    listener.on_event(Event::Disconnected {
        reason: "test".to_string(),
    });

    let received = events.lock().unwrap();
    assert_eq!(received.len(), 7);
}
