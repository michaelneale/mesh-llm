//! String templates for guardrail nudges. Ported from forge
//! `src/forge/prompts/nudges.py` (v0.6.0).
//!
//! These strings are tuned on Ministral / Qwen3 / Mistral; keep them
//! configurable so we can A/B them on whatever small models we host.

pub fn retry_nudge(_raw_response: &str) -> String {
    // Forge keeps the raw response in the signature for compat but
    // doesn't actually use it; mirror that.
    "Your previous response was not a valid tool call. \
     You must respond with a tool call, not free text. \
     Please try again with a valid tool call."
        .to_string()
}

pub fn unknown_tool_nudge(tool_name: &str, available_tools: &[String]) -> String {
    let tools_list = available_tools.join(", ");
    format!(
        "Tool '{tool_name}' does not exist. \
         Available tools: {tools_list}. \
         Call one of them."
    )
}

pub fn step_nudge(terminal_tool: &str, pending_steps: &[String], tier: u8) -> String {
    let tier = tier.clamp(1, 3);
    let steps = pending_steps.join(", ");
    match tier {
        1 => format!(
            "You cannot call {terminal_tool} yet. \
             You must first complete these required steps: {steps}. \
             Call one of them now."
        ),
        2 => format!(
            "You must call one of these tools now: {steps}. \
             Pick one."
        ),
        _ => format!(
            "STOP. You MUST call one of: {steps}. \
             Do NOT call {terminal_tool}. \
             Your next response MUST be a tool call to one of: {steps}."
        ),
    }
}

pub fn prerequisite_nudge(tool_name: &str, missing_prereqs: &[String]) -> String {
    let prereqs = missing_prereqs.join(", ");
    format!(
        "You cannot call {tool_name} yet. \
         You must first call: {prereqs}. \
         Call the prerequisite tool now."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_nudge_clamps_tier() {
        let pending = vec!["read".to_string(), "plan".to_string()];
        let t0 = step_nudge("submit", &pending, 0);
        let t1 = step_nudge("submit", &pending, 1);
        let t9 = step_nudge("submit", &pending, 9);
        let t3 = step_nudge("submit", &pending, 3);
        assert_eq!(t0, t1, "tier 0 must clamp up to tier 1");
        assert_eq!(t9, t3, "tier 9 must clamp down to tier 3");
        assert!(t3.contains("STOP"));
    }

    #[test]
    fn unknown_tool_nudge_lists_tools() {
        let tools = vec!["a".to_string(), "b".to_string()];
        let msg = unknown_tool_nudge("c", &tools);
        assert!(msg.contains("'c'"));
        assert!(msg.contains("a, b"));
    }
}
