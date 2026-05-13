//! Worker role assignment and text extraction helpers.

use crate::ModelEntry;

/// Worker role determines the context shape and depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// Fast small model — classify, quick proposal.
    Fast,
    /// Specialist — code, domain knowledge.
    Specialist,
    /// Strong reasoner — deeper analysis.
    Strong,
    /// General-purpose worker.
    Generalist,
    /// Reducer/finalizer — only invoked for arbitration.
    Reducer,
}

impl WorkerRole {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Specialist => "specialist",
            Self::Strong => "strong",
            Self::Generalist => "generalist",
            Self::Reducer => "reducer",
        }
    }
}

/// A worker assignment: which model plays which role.
pub struct Assignment {
    pub model_name: String,
    pub backend_index: usize,
    pub role: WorkerRole,
}

/// Assign roles to models.
///
/// Heuristic: more models = more specialization.
/// With 2: fast + strong.  With 3+: fast + specialist(s) + strong.
pub fn assign_roles(models: &[ModelEntry]) -> Vec<Assignment> {
    if models.is_empty() {
        return vec![];
    }
    if models.len() == 1 {
        return vec![Assignment {
            model_name: models[0].name.clone(),
            backend_index: models[0].backend_index,
            role: WorkerRole::Generalist,
        }];
    }

    let mut assignments = Vec::new();

    // First = fast
    assignments.push(Assignment {
        model_name: models[0].name.clone(),
        backend_index: models[0].backend_index,
        role: WorkerRole::Fast,
    });

    // Middle = specialist(s)
    for m in &models[1..models.len() - 1] {
        assignments.push(Assignment {
            model_name: m.name.clone(),
            backend_index: m.backend_index,
            role: WorkerRole::Specialist,
        });
    }

    // Last = strong
    let last = models.last().unwrap();
    assignments.push(Assignment {
        model_name: last.name.clone(),
        backend_index: last.backend_index,
        role: WorkerRole::Strong,
    });

    assignments
}

/// Strip `<think>...</think>` tags, return the remaining content.
pub fn strip_thinking(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</think>".len()..]
            );
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    result = result.replace("</think>", "");
    result.trim().to_string()
}

/// Extract content inside `<think>` tags.
pub fn extract_thinking(text: &str) -> String {
    if let Some(start) = text.find("<think>") {
        let after = &text[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            return after[..end].trim().to_string();
        }
        return after.trim().to_string();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_two_models() {
        let models = vec![
            ModelEntry {
                name: "small".into(),
                backend_index: 0,
            },
            ModelEntry {
                name: "big".into(),
                backend_index: 1,
            },
        ];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Strong);
    }

    #[test]
    fn assign_three_models() {
        let models = vec![
            ModelEntry {
                name: "small".into(),
                backend_index: 0,
            },
            ModelEntry {
                name: "mid".into(),
                backend_index: 1,
            },
            ModelEntry {
                name: "big".into(),
                backend_index: 2,
            },
        ];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Specialist);
        assert_eq!(assignments[2].role, WorkerRole::Strong);
    }

    #[test]
    fn strip_thinking_tags() {
        assert_eq!(strip_thinking("<think>foo</think>bar"), "bar");
        assert_eq!(
            strip_thinking("before<think>mid</think>after"),
            "beforeafter"
        );
        assert_eq!(strip_thinking("<think>only thinking"), "");
        assert_eq!(strip_thinking("no tags here"), "no tags here");
    }
}
