use std::convert::Infallible;

use axum::response::sse::Event;
use serde::Serialize;

pub fn json_event(value: &impl Serialize) -> Result<Event, Infallible> {
    Ok(Event::default().json_data(value).unwrap_or_else(|_| {
        Event::default().data(r#"{"error":{"message":"failed to serialize SSE event"}}"#)
    }))
}

pub fn done_event() -> Result<Event, Infallible> {
    Ok(Event::default().data("[DONE]"))
}
