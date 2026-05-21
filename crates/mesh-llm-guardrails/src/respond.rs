//! Synthetic `respond` tool. Ported from forge
//! `src/forge/tools/respond.py` (v0.6.0).
//!
//! When `respond` is injected, the model is steered toward calling
//! `respond(message="...")` instead of producing bare text. This keeps
//! it in tool-calling mode where the full guardrail stack applies.
//! Small local models (~8B) cannot be trusted to choose correctly
//! between text and tool calls — guiding them to a tool is a must.
//!
//! On the way back to the caller, a `respond` tool-call is converted
//! to a plain assistant `content` string.

use serde_json::{json, Value};

pub const RESPOND_TOOL_NAME: &str = "respond";

pub const RESPOND_DESCRIPTION: &str = "Respond to the user with a message. \
Use this when the user is chatting, asking a question, when you need to \
ask a clarifying question before proceeding, or when no other tool \
action is needed. Also use this after completing the user's request to \
report the result.";

/// JSON Schema for the synthetic `respond` tool — drop-in for an
/// OpenAI `tools` array.
pub fn respond_tool_spec() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": RESPOND_TOOL_NAME,
            "description": RESPOND_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user.",
                    },
                },
                "required": ["message"],
            },
        },
    })
}

/// Inject the `respond` tool into the request's `tools` array if it
/// isn't already there. Returns true if the tool was added.
pub fn inject_respond_tool(tools: &mut Value) -> bool {
    let Some(arr) = tools.as_array_mut() else {
        return false;
    };
    let already_present = arr.iter().any(|t| {
        t.get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            == Some(RESPOND_TOOL_NAME)
    });
    if already_present {
        return false;
    }
    arr.push(respond_tool_spec());
    true
}

/// Extract the message from a `respond` tool call's args. Used when
/// converting `respond` calls back into a plain `content` string for
/// the upstream caller.
pub fn extract_respond_message(args: &Value) -> Option<String> {
    args.get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_when_absent() {
        let mut tools = json!([
            {"type": "function", "function": {"name": "read_file"}},
        ]);
        assert!(inject_respond_tool(&mut tools));
        let arr = tools.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(arr
            .iter()
            .any(|t| t["function"]["name"] == RESPOND_TOOL_NAME));
    }

    #[test]
    fn idempotent_when_present() {
        let mut tools = json!([respond_tool_spec()]);
        assert!(!inject_respond_tool(&mut tools));
        assert_eq!(tools.as_array().unwrap().len(), 1);
    }

    #[test]
    fn does_not_inject_into_non_array() {
        let mut tools = json!({"oops": true});
        assert!(!inject_respond_tool(&mut tools));
    }

    #[test]
    fn extracts_message() {
        let args = json!({"message": "hello"});
        assert_eq!(extract_respond_message(&args).as_deref(), Some("hello"));
        assert_eq!(extract_respond_message(&json!({})), None);
    }
}
