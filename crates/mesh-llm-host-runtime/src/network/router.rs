/// Smart model router — classifies requests and picks the best model.
use serde_json::Value;

// ── Request categories ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Code,
    Reasoning,
    Chat,
    ToolCall,
    Creative,
    /// Factual lookup, summarization, knowledge retrieval
    Info,
    /// Image generation or analysis (future: multimodal models)
    Image,
}

/// How complex/heavy the request appears to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    Quick,    // simple fact, short answer, casual
    Moderate, // normal conversation, standard code
    Deep,     // long reasoning, complex analysis, architecture
}

/// Full classification result.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    pub category: Category,
    pub complexity: Complexity,
    pub needs_tools: bool,
    pub has_media_inputs: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaRequirements {
    pub has_media: bool,
    pub needs_vision: bool,
    pub needs_audio: bool,
}

impl MediaRequirements {
    pub fn requires_runtime_modality(self) -> bool {
        self.needs_vision || self.needs_audio
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RouterSignalScores {
    code: usize,
    reasoning: usize,
    creative: usize,
    info: usize,
    image: usize,
    deep: usize,
}

// ── Model capabilities for routing ──────────────────────────────────

/// Strip split GGUF suffix like "-00001-of-00004" from a model name.
pub fn strip_split_suffix(name: &str) -> &str {
    // Pattern: -NNNNN-of-NNNNN at the end
    if let Some(idx) = name.rfind("-of-") {
        // Check that what follows is digits and what precedes is -digits
        let after = &name[idx + 4..];
        if after.chars().all(|c| c.is_ascii_digit()) && !after.is_empty() {
            // Find the preceding -NNNNN
            if let Some(dash) = name[..idx].rfind('-') {
                let between = &name[dash + 1..idx];
                if between.chars().all(|c| c.is_ascii_digit()) && !between.is_empty() {
                    return &name[..dash];
                }
            }
        }
    }
    name
}

/// Owned version of strip_split_suffix for contexts that need a String.
pub fn strip_split_suffix_owned(name: &str) -> String {
    strip_split_suffix(name).to_string()
}

// ── Request classification ──────────────────────────────────────────

/// Classify a chat completion request body using heuristics.
/// No LLM call, just pattern matching on the request structure.
/// Classify a request body into category + complexity + needs_tools.
/// Tools presence is an attribute, not a category override — a code request
/// with tools is still Code (with needs_tools=true), not ToolCall.
pub fn classify(body: &Value) -> Classification {
    let text = collect_message_text(body);
    let lower = text.to_lowercase();
    let media = media_requirements(body);
    let needs_tools = detect_tool_requirement(body);
    let last_user_len = last_user_message_len(body);
    let scores = router_signal_scores(&lower);
    let category = classify_category(scores, detect_system_code_hint(body), media, needs_tools);
    let complexity = classify_complexity(scores, last_user_len, message_count(body));

    Classification {
        category,
        complexity,
        needs_tools,
        has_media_inputs: media.has_media,
    }
}

fn detect_tool_requirement(body: &Value) -> bool {
    has_tools_schema(body) || has_tool_blocks(body)
}

fn has_tools_schema(body: &Value) -> bool {
    body.get("tools")
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false)
}

fn has_tool_blocks(body: &Value) -> bool {
    body.get("messages")
        .and_then(|m| m.as_array())
        .map(|msgs| {
            msgs.iter().any(|msg| {
                msg.get("content")
                    .and_then(|c| c.as_array())
                    .map(|blocks| {
                        blocks.iter().any(|b| {
                            matches!(
                                b.get("type").and_then(|t| t.as_str()),
                                Some("tool_use") | Some("tool_result")
                            )
                        })
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn count_signals(lower: &str, signals: &[&str]) -> usize {
    signals
        .iter()
        .filter(|signal| lower.contains(*signal))
        .count()
}

fn router_signal_scores(lower: &str) -> RouterSignalScores {
    RouterSignalScores {
        code: count_signals(
            lower,
            &[
                "```",
                "def ",
                "fn ",
                "func ",
                "class ",
                "import ",
                "function",
                "const ",
                "let ",
                "var ",
                "return ",
                "write a program",
                "write code",
                "implement",
                "refactor",
                "debug",
                "fix the bug",
                "write a script",
                "code review",
                "pull request",
                "git ",
                "compile",
                "syntax",
                "python",
                "javascript",
                "typescript",
                " rust ",
                "golang",
                "java ",
                "c++",
                " ruby ",
                " swift ",
                "kotlin",
                "algorithm",
                "binary search",
                " sort ",
                "regex",
                " api ",
                " http ",
                " sql ",
                "database",
                " query ",
            ],
        ),
        reasoning: count_signals(
            lower,
            &[
                "prove",
                "explain why",
                "step by step",
                "calculate",
                "solve",
                "derive",
                "what is the probability",
                "how many",
                "analyze",
                "compare and contrast",
                "evaluate",
                "mathematical",
                "theorem",
                "equation",
                "logic",
                "think carefully",
                "reason about",
            ],
        ),
        creative: count_signals(
            lower,
            &[
                "write a story",
                "write a poem",
                "creative",
                "imagine",
                "fiction",
                "narrative",
                "compose",
                "brainstorm",
                "write a song",
                "screenplay",
                "dialogue",
            ],
        ),
        info: count_signals(
            lower,
            &[
                "what is",
                "who is",
                "when did",
                "where is",
                "how does",
                "define ",
                "explain ",
                "summarize",
                "summary",
                "overview",
                "tell me about",
                "describe ",
                "what are the",
                "list the",
                "difference between",
                "compare ",
                "history of",
            ],
        ),
        image: count_signals(
            lower,
            &[
                "image",
                "picture",
                "photo",
                "draw",
                "generate an image",
                "visualize",
                "diagram",
                "screenshot",
                "describe this image",
            ],
        ),
        deep: count_signals(
            lower,
            &[
                "architect",
                "design a system",
                "trade-off",
                "tradeoff",
                "in depth",
                "comprehensive",
                "thorough",
                "detailed analysis",
                "long-term",
                "strategy",
                "plan for",
                "review this codebase",
                "rewrite",
                "from scratch",
            ],
        ),
    }
}

fn detect_system_code_hint(body: &Value) -> bool {
    body.get("messages")
        .and_then(|m| m.as_array())
        .map(|messages| {
            messages.iter().any(|msg| {
                msg.get("role").and_then(|r| r.as_str()) == Some("system")
                    && msg
                        .get("content")
                        .and_then(|c| c.as_str())
                        .map(|content| {
                            let sys = content.to_lowercase();
                            sys.contains("developer")
                                || sys.contains("coding")
                                || sys.contains("programmer")
                        })
                        .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn classify_category(
    scores: RouterSignalScores,
    system_code: bool,
    media: MediaRequirements,
    needs_tools: bool,
) -> Category {
    if system_code
        || scores.code >= 2
        || (scores.code >= 1 && scores.reasoning == 0 && scores.creative == 0)
    {
        Category::Code
    } else if scores.reasoning >= 2 {
        Category::Reasoning
    } else if scores.creative >= 1 {
        Category::Creative
    } else if media.needs_vision || scores.image >= 1 {
        Category::Image
    } else if needs_tools && scores.code == 0 && scores.reasoning == 0 && scores.creative == 0 {
        Category::ToolCall
    } else if scores.info >= 2 && scores.code == 0 {
        Category::Info
    } else {
        Category::Chat
    }
}

fn message_count(body: &Value) -> usize {
    body.get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

fn classify_complexity(
    scores: RouterSignalScores,
    last_user_len: usize,
    total_messages: usize,
) -> Complexity {
    if scores.deep >= 1 || last_user_len > 500 || total_messages > 10 {
        Complexity::Deep
    } else if last_user_len < 60 && total_messages <= 2 && scores.reasoning == 0 && scores.deep == 0
    {
        Complexity::Quick
    } else {
        Complexity::Moderate
    }
}

pub fn media_requirements(body: &Value) -> MediaRequirements {
    let mut requirements = MediaRequirements::default();
    let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
        return requirements;
    };

    for msg in messages {
        let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for block in blocks {
            let block_type = block
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            match block_type {
                "image_url" | "input_image" | "image" => {
                    requirements.has_media = true;
                    requirements.needs_vision = true;
                }
                "audio_url" | "input_audio" | "audio" => {
                    requirements.has_media = true;
                    requirements.needs_audio = true;
                }
                "file" | "input_file" => {
                    requirements.has_media = true;
                }
                _ => {
                    if block.get("image_url").is_some() || block.get("image").is_some() {
                        requirements.has_media = true;
                        requirements.needs_vision = true;
                    }
                    if block.get("audio_url").is_some() || block.get("audio").is_some() {
                        requirements.has_media = true;
                        requirements.needs_audio = true;
                    }
                }
            }
        }
    }

    requirements
}

pub(crate) fn model_satisfies_media_requirements(
    caps: &crate::models::ModelCapabilities,
    media: &MediaRequirements,
) -> bool {
    (!media.needs_vision || caps.supports_vision_runtime())
        && (!media.needs_audio || caps.supports_audio_runtime())
}

pub(crate) fn filter_media_compatible_candidates<'a>(
    candidates: &[RoutingCandidate<'a>],
    media: &MediaRequirements,
) -> Option<Vec<RoutingCandidate<'a>>> {
    let media_available: Vec<_> = candidates
        .iter()
        .filter(|c| model_satisfies_media_requirements(&c.caps, media))
        .cloned()
        .collect();
    if media_available.is_empty() && media.requires_runtime_modality() {
        None
    } else if media_available.is_empty() {
        Some(candidates.to_vec())
    } else {
        Some(media_available)
    }
}

/// Length of last user message in characters (rough complexity proxy).
fn last_user_message_len(body: &Value) -> usize {
    body.get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| {
            msgs.iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        })
        .map(message_text)
        .map(|s| s.len())
        .unwrap_or(0)
}

fn collect_message_text(body: &Value) -> String {
    let mut text = String::new();
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let content = message_text(msg);
            if !content.is_empty() {
                text.push_str(&content);
                text.push('\n');
            }
        }
    }
    text
}

/// Extract message text for both OpenAI-style and Anthropic-style payloads.
fn message_text(msg: &Value) -> String {
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return s.to_string();
    }

    // Anthropic content blocks: [{"type":"text","text":"..."}, ...]
    if let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) {
        let mut out = String::new();
        for b in blocks {
            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                out.push_str(t);
                out.push('\n');
            }
        }
        return out;
    }

    String::new()
}

// ── Model selection ─────────────────────────────────────────────────

/// A candidate model in the auto-routing pool.
///
/// `tps_hint` and `throughput_samples` come from the node's locally
/// observed `RoutingMetrics`. They are `None` / `0` when we've never
/// successfully completed a token-bearing request against this model
/// (cold start, brand-new peer) — such candidates get a neutral weight
/// so they still participate in routing while accumulating data.
#[derive(Clone, Debug)]
pub struct RoutingCandidate<'a> {
    pub name: &'a str,
    pub caps: crate::models::ModelCapabilities,
    /// Locally observed throughput in tokens/sec. `None` if no
    /// throughput-bearing attempts have completed for this model yet.
    pub tps_hint: Option<f64>,
    /// How many throughput samples back `tps_hint`. Used to decide
    /// whether we trust the hint or treat the model as "cold".
    pub throughput_samples: u64,
}

impl<'a> RoutingCandidate<'a> {
    /// Build a candidate without any throughput hint. Useful for
    /// pre-startup paths or test fixtures.
    pub fn unscored(name: &'a str, caps: crate::models::ModelCapabilities) -> Self {
        Self {
            name,
            caps,
            tps_hint: None,
            throughput_samples: 0,
        }
    }
}

/// Minimum number of throughput samples before `tps_hint` is allowed
/// to influence weighting. Below this, the candidate is treated as
/// cold (neutral weight).
const TPS_MIN_SAMPLES: u64 = 3;

/// Lower clamp on tps used as a weight. Prevents catastrophic peers
/// (~1 tok/s) from being completely starved — they still get the
/// occasional request so they can re-prove themselves.
const TPS_WEIGHT_MIN: f64 = 5.0;

/// Upper clamp on tps used as a weight. Prevents a single very fast
/// outlier from monopolizing routing.
const TPS_WEIGHT_MAX: f64 = 100.0;

/// Neutral weight assigned to cold / unscored candidates so they get
/// a fair shot at picking up data. Set to the midpoint of the clamp
/// range so a cold model is treated as "average" against scored peers.
const TPS_NEUTRAL_WEIGHT: f64 = 25.0;

/// Probability of ignoring weights and picking uniformly. Keeps the
/// system from locking onto stale rankings and gives cold models a
/// guaranteed share of traffic so they accumulate data.
const EXPLORATION_PROBABILITY: f64 = 0.15;

/// Pick the best model for a classified request using gossiped capabilities
/// and locally observed throughput.
///
/// Filtering:
///   - `needs_tools` → prefer models with `tool_use != None`
///   - `Reasoning`   → prefer models with `reasoning != None`
///   - `Image`       → prefer models with `vision != None`
///   - anything else → no capability filter
///
/// Falls back to all models if the filter matches nothing. Then biases
/// toward larger models by partitioning single-digit-B names to the
/// bottom tier. Within the chosen tier, picks weighted by observed
/// tok/s (with cold models treated as average), with a configurable
/// exploration probability that ignores weights and picks uniformly.
pub fn pick_model_classified<'a>(
    classification: &Classification,
    available_models: &[RoutingCandidate<'a>],
) -> Option<&'a str> {
    use crate::models::CapabilityLevel;

    if available_models.is_empty() {
        return None;
    }
    if available_models.len() == 1 {
        return Some(available_models[0].name);
    }

    // Capability filter based on what the request needs.
    let filtered: Vec<&RoutingCandidate<'a>> = match classification.category {
        _ if classification.needs_tools => available_models
            .iter()
            .filter(|c| c.caps.tool_use != CapabilityLevel::None)
            .collect(),
        Category::Reasoning => available_models
            .iter()
            .filter(|c| c.caps.reasoning != CapabilityLevel::None)
            .collect(),
        Category::Image => available_models
            .iter()
            .filter(|c| c.caps.vision != CapabilityLevel::None)
            .collect(),
        _ => Vec::new(),
    };

    // Fall back to all models if the filter matched nothing.
    let candidates: Vec<&RoutingCandidate<'a>> = if filtered.is_empty() {
        available_models.iter().collect()
    } else {
        filtered
    };

    // Bias toward larger models: names that advertise a single-digit
    // parameter count (e.g. "2B", "9B") go to the bottom. Everything
    // else — multi-digit billions (31B, 70B) or names that don't encode
    // a size at all (MiniMax, Coder-Next, fine-tune tags) — stays on
    // top. The big tier is sampled by tok/s-weighted draw; the small
    // tier acts only as a fallback when the big tier is empty.
    let (big, small): (Vec<_>, Vec<_>) = candidates
        .into_iter()
        .partition(|c| !is_single_digit_b_name(c.name));

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;

    if !big.is_empty() {
        return Some(pick_weighted(&big, nanos));
    }
    if !small.is_empty() {
        return Some(pick_weighted(
            &small,
            nanos.wrapping_add(0x9E37_79B9_7F4A_7C15),
        ));
    }
    None
}

/// Compute the routing weight for a single candidate. See the module-level
/// `TPS_*` constants for the rationale on each clamp.
fn candidate_weight(candidate: &RoutingCandidate<'_>) -> f64 {
    if candidate.throughput_samples >= TPS_MIN_SAMPLES {
        candidate
            .tps_hint
            .unwrap_or(TPS_NEUTRAL_WEIGHT)
            .clamp(TPS_WEIGHT_MIN, TPS_WEIGHT_MAX)
    } else {
        TPS_NEUTRAL_WEIGHT
    }
}

/// Pick one candidate from a non-empty slice using tok/s-weighted draw,
/// with `EXPLORATION_PROBABILITY` chance of a uniform pick.
fn pick_weighted<'a>(candidates: &[&RoutingCandidate<'a>], seed: u64) -> &'a str {
    debug_assert!(!candidates.is_empty(), "pick_weighted requires non-empty");

    let mut rng = SplitMix64::new(seed);

    // Exploration branch: ignore weights, pick uniformly. Keeps the
    // system from locking onto stale rankings.
    if rng.next_f64() < EXPLORATION_PROBABILITY {
        let idx = (rng.next_u64() as usize) % candidates.len();
        return candidates[idx].name;
    }

    let total_weight: f64 = candidates.iter().map(|c| candidate_weight(c)).sum();
    // Defensive: if all weights are somehow zero (shouldn't happen given
    // TPS_WEIGHT_MIN > 0), fall back to a uniform pick.
    if total_weight <= 0.0 {
        let idx = (rng.next_u64() as usize) % candidates.len();
        return candidates[idx].name;
    }

    let pick = rng.next_f64() * total_weight;
    let mut acc = 0.0;
    for c in candidates {
        acc += candidate_weight(c);
        if pick < acc {
            return c.name;
        }
    }
    // Numerical tail: pick the last candidate.
    candidates[candidates.len() - 1].name
}

/// Small deterministic PRNG so a single seed drives both the
/// exploration coin flip and the weighted draw. Avoids pulling in a
/// rand dependency just for routing.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        // Avoid the zero state which gives a degenerate sequence.
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits for a uniform float in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Return true if `name` advertises a single-digit billion-parameter
/// count, e.g. "Qwen3.5-2B-Q4_K_M" or "llama-3-7b-instruct".
///
/// Accepts: a standalone digit 1-9 immediately followed by `b` or `B`,
/// with the digit *not* preceded by another digit or `.` (so "12B" and
/// "2.5B" don't count) and the `B` *not* followed by another digit (so
/// "BF16" isn't a match).
///
/// Names without any digit-B pattern return false — they are treated as
/// "probably strong" because small open-weight models almost always
/// advertise their size in the filename.
fn is_single_digit_b_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    for i in 0..bytes.len() {
        let c = bytes[i];
        if !c.is_ascii_digit() {
            continue;
        }
        // Must be a single digit run at a word boundary: previous char
        // must not be another digit, a '.', or an ASCII letter. That
        // last part rules out active-params tags like "A3B" where
        // the 3B is a subset of a larger total count advertised
        // elsewhere in the name (e.g. "Qwen3.6-35B-A3B").
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_digit() || prev == b'.' || prev.is_ascii_alphabetic() {
                continue;
            }
        }
        // Digit must be 1-9 (0B would be nonsense, ignore)
        if c == b'0' {
            continue;
        }
        // Next byte must be b or B
        let Some(&next) = bytes.get(i + 1) else {
            continue;
        };
        if next != b'b' && next != b'B' {
            continue;
        }
        // And the byte after that must not be another digit (avoid BF16-like continuations)
        if let Some(&after) = bytes.get(i + 2) {
            if after.is_ascii_digit() {
                continue;
            }
        }
        return true;
    }
    false
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_classify_tool_call() {
        // Content that implies tool use + tools schema = ToolCall
        let body = json!({
            "messages": [{"role": "user", "content": "Run the tests and check the output"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        assert_eq!(classify(&body).category, Category::ToolCall);
    }

    #[test]
    fn test_classify_code() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Write a Python function to implement binary search and debug any issues"}
            ]
        });
        assert_eq!(classify(&body).category, Category::Code);
    }

    #[test]
    fn test_classify_reasoning() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Prove that the square root of 2 is irrational. Explain step by step."}
            ]
        });
        assert_eq!(classify(&body).category, Category::Reasoning);
    }

    #[test]
    fn test_classify_creative() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Write a story about a robot who learns to paint"}
            ]
        });
        assert_eq!(classify(&body).category, Category::Creative);
    }

    #[test]
    fn test_classify_chat_default() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "What's the capital of France?"}
            ]
        });
        let cl = classify(&body);
        assert_eq!(cl.category, Category::Chat);
        assert_eq!(cl.complexity, Complexity::Quick); // short simple question
        assert!(!cl.needs_tools);
        assert!(!cl.has_media_inputs);
    }

    #[test]
    fn test_classify_deep_analysis() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Design a system architecture for a distributed database with strong consistency guarantees. Provide a detailed analysis of the trade-offs between CAP theorem constraints and explain how to handle network partitions in depth."}
            ]
        });
        let cl = classify(&body);
        assert_eq!(cl.complexity, Complexity::Deep);
    }

    #[test]
    fn test_classify_code_with_tools() {
        // Code request that happens to have tools — should be Code, not ToolCall
        let body = json!({
            "messages": [{"role": "user", "content": "Write a Python function to sort a list and debug it"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        let cl = classify(&body);
        assert_eq!(cl.category, Category::Code);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_tools_schema_always_needs_tools() {
        // Tools schema present = agentic session, always needs_tools
        // even if the message content is plain chat
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_tools_schema_with_tool_content() {
        // Tools in schema AND content implies tool use — needs tools
        let body = json!({
            "messages": [{"role": "user", "content": "Read the file and fix the bug"}],
            "tools": [{"type": "function", "function": {"name": "read"}}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_anthropic_text_blocks_with_tools() {
        // Anthropic-style content blocks should still be parsed as text
        // and trigger needs_tools when tool-intent is present.
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List files in this directory and read README.md"}
                    ]
                }
            ],
            "tools": [{"name": "shell"}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
        assert!(matches!(cl.category, Category::Code | Category::ToolCall));
    }

    #[test]
    fn test_classify_anthropic_tool_use_block_sets_needs_tools() {
        // If an explicit tool_use/tool_result block is present, mark as needs_tools.
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_123", "name": "shell", "input": {"command": "ls"}}
                    ]
                }
            ]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_anthropic_tool_request_sets_needs_tools() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List files in this directory and read README.md"}
                    ]
                }
            ],
            "tools": [{"name": "shell"}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_system_prompt_code() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are a senior developer and coding assistant."},
                {"role": "user", "content": "Help me with this."}
            ]
        });
        assert_eq!(classify(&body).category, Category::Code);
    }

    #[test]
    fn test_media_requirements_detect_audio_block() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this clip"},
                        {"type": "audio_url", "audio_url": {"url": "mesh://blob/client-1/example"}}
                    ]
                }
            ]
        });
        let media = media_requirements(&body);
        assert!(media.has_media);
        assert!(media.needs_audio);
        assert!(!media.needs_vision);
        assert!(classify(&body).has_media_inputs);
    }

    #[test]
    fn test_media_requirements_detect_image_block() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                    ]
                }
            ]
        });
        let media = media_requirements(&body);
        assert!(media.has_media);
        assert!(media.needs_vision);
        assert!(!media.needs_audio);
        assert!(classify(&body).has_media_inputs);
    }

    #[test]
    fn test_model_satisfies_media_requirements_matches_required_modalities() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let text_caps = ModelCapabilities::default();
        let vision_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            ..Default::default()
        };
        let audio_caps = ModelCapabilities {
            audio: CapabilityLevel::Supported,
            ..Default::default()
        };
        let vision_audio_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            audio: CapabilityLevel::Supported,
            ..Default::default()
        };

        let text_only = MediaRequirements::default();
        let image = MediaRequirements {
            has_media: true,
            needs_vision: true,
            needs_audio: false,
        };
        let audio = MediaRequirements {
            has_media: true,
            needs_vision: false,
            needs_audio: true,
        };
        let image_and_audio = MediaRequirements {
            has_media: true,
            needs_vision: true,
            needs_audio: true,
        };

        assert!(model_satisfies_media_requirements(&text_caps, &text_only));
        assert!(!model_satisfies_media_requirements(&text_caps, &image));
        assert!(model_satisfies_media_requirements(&vision_caps, &image));
        assert!(!model_satisfies_media_requirements(&vision_caps, &audio));
        assert!(model_satisfies_media_requirements(&audio_caps, &audio));
        assert!(!model_satisfies_media_requirements(
            &vision_caps,
            &image_and_audio
        ));
        assert!(model_satisfies_media_requirements(
            &vision_audio_caps,
            &image_and_audio
        ));

        let likely_vision_caps = ModelCapabilities {
            multimodal: true,
            vision: CapabilityLevel::Likely,
            ..Default::default()
        };
        assert!(!model_satisfies_media_requirements(
            &likely_vision_caps,
            &image
        ));
    }

    #[test]
    fn test_filter_media_candidates_blocks_hard_media_miss() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let text_caps = ModelCapabilities::default();
        let vision_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            ..Default::default()
        };
        let image = MediaRequirements {
            has_media: true,
            needs_vision: true,
            needs_audio: false,
        };
        let text_only = MediaRequirements::default();

        let text_candidates = vec![RoutingCandidate::unscored("text", text_caps)];
        assert!(filter_media_compatible_candidates(&text_candidates, &image).is_none());

        let mixed_candidates = vec![
            RoutingCandidate::unscored("text", text_caps),
            RoutingCandidate::unscored("vision", vision_caps),
        ];
        let filtered = filter_media_compatible_candidates(&mixed_candidates, &image)
            .expect("vision candidate should satisfy image media request");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "vision");

        let unfiltered = filter_media_compatible_candidates(&text_candidates, &text_only)
            .expect("text-only requests should keep normal router fallback behavior");
        // Text-only requests with text-only candidates pass through unfiltered.
        assert_eq!(unfiltered.len(), text_candidates.len());
        assert_eq!(unfiltered[0].name, text_candidates[0].name);
    }

    #[test]
    fn test_pick_tools_filters_by_capability() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let tool_caps = ModelCapabilities {
            tool_use: CapabilityLevel::Supported,
            ..Default::default()
        };
        let no_caps = ModelCapabilities::default();

        let available = vec![
            RoutingCandidate::unscored("reasoning-model", no_caps),
            RoutingCandidate::unscored("tool-model", tool_caps),
        ];
        let cl = Classification {
            category: Category::Code,
            complexity: Complexity::Moderate,
            needs_tools: true,
            has_media_inputs: false,
        };
        let result = pick_model_classified(&cl, &available);
        assert_eq!(result, Some("tool-model"));
    }

    #[test]
    fn test_pick_reasoning_filters_by_capability() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let reasoning_caps = ModelCapabilities {
            reasoning: CapabilityLevel::Supported,
            ..Default::default()
        };
        let no_caps = ModelCapabilities::default();

        let available = vec![
            RoutingCandidate::unscored("chat-model", no_caps),
            RoutingCandidate::unscored("reasoning-model", reasoning_caps),
        ];
        let cl = Classification {
            category: Category::Reasoning,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        let result = pick_model_classified(&cl, &available);
        assert_eq!(result, Some("reasoning-model"));
    }

    #[test]
    fn test_pick_vision_filters_by_capability() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let vision_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            ..Default::default()
        };
        let no_caps = ModelCapabilities::default();

        let available = vec![
            RoutingCandidate::unscored("text-model", no_caps),
            RoutingCandidate::unscored("vision-model", vision_caps),
        ];
        let cl = Classification {
            category: Category::Image,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: true,
        };
        let result = pick_model_classified(&cl, &available);
        assert_eq!(result, Some("vision-model"));
    }

    #[test]
    fn test_pick_falls_back_when_no_capability_match() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("model-a", no_caps),
            RoutingCandidate::unscored("model-b", no_caps),
        ];
        let cl = Classification {
            category: Category::Code,
            complexity: Complexity::Moderate,
            needs_tools: true,
            has_media_inputs: false,
        };
        // No tool-capable model — falls back to all
        let result = pick_model_classified(&cl, &available);
        assert!(result == Some("model-a") || result == Some("model-b"));
    }

    #[test]
    fn test_pick_empty_returns_none() {
        let available: Vec<RoutingCandidate<'_>> = vec![];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        assert_eq!(pick_model_classified(&cl, &available), None);
    }

    #[test]
    fn test_pick_single_model() {
        use crate::models::ModelCapabilities;

        let available = vec![RoutingCandidate::unscored(
            "only-model",
            ModelCapabilities::default(),
        )];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        assert_eq!(pick_model_classified(&cl, &available), Some("only-model"));
    }

    #[test]
    fn test_pick_chat_no_filter() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("model-a", no_caps),
            RoutingCandidate::unscored("model-b", no_caps),
        ];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        // Chat with no special needs — any model is valid
        let result = pick_model_classified(&cl, &available);
        assert!(result == Some("model-a") || result == Some("model-b"));
    }

    #[test]
    fn test_strip_split_suffix() {
        assert_eq!(
            strip_split_suffix("MiniMax-M2.5-Q4_K_M-00001-of-00004"),
            "MiniMax-M2.5-Q4_K_M"
        );
        assert_eq!(
            strip_split_suffix("Qwen3-Coder-Next-Q4_K_M-00001-of-00004"),
            "Qwen3-Coder-Next-Q4_K_M"
        );
        assert_eq!(
            strip_split_suffix("Hermes-2-Pro-Mistral-7B-Q4_K_M"),
            "Hermes-2-Pro-Mistral-7B-Q4_K_M"
        );
        assert_eq!(strip_split_suffix(""), "");
    }

    #[test]
    fn test_is_single_digit_b_name() {
        // Single-digit sizes — match
        assert!(is_single_digit_b_name("Qwen3.5-2B-Q4_K_M"));
        assert!(is_single_digit_b_name("Qwen3.5-9B-Q4_K_M"));
        assert!(is_single_digit_b_name("llama-3-7b-instruct"));
        assert!(is_single_digit_b_name("Mistral-7B-Instruct-v0.3"));
        assert!(is_single_digit_b_name("gemma-2-2b-it"));

        // Multi-digit sizes — not small
        assert!(!is_single_digit_b_name("gemma-4-31B-it-Q8_0"));
        assert!(!is_single_digit_b_name("Qwen3.6-35B-A3B-BF16"));
        assert!(!is_single_digit_b_name("llama-3.1-70B-Instruct"));
        assert!(!is_single_digit_b_name("deepseek-v3-671B"));

        // Decimal sizes — not single-digit (treat as unknown/big)
        assert!(!is_single_digit_b_name("phi-3.5-mini-3.8B"));
        assert!(!is_single_digit_b_name("Qwen2.5-1.5B"));

        // Unknown names — no match → treated as big
        assert!(!is_single_digit_b_name("MiniMax-M2.5-Q4_K_M"));
        assert!(!is_single_digit_b_name("Qwen3-Coder-Next-Q4_K_M"));
        assert!(!is_single_digit_b_name(""));

        // BF16 / FP16 substrings must not trigger
        assert!(!is_single_digit_b_name("some-model-BF16"));
        assert!(!is_single_digit_b_name("some-model-fp16"));

        // Digit-B embedded with later digits (versions) must not trigger
        assert!(!is_single_digit_b_name("foo-2b1-bar")); // 2b followed by 1
    }

    #[test]
    fn test_pick_prefers_multi_digit_over_single_digit() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("Qwen3.5-2B-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3.5-9B-Q4_K_M", no_caps),
            RoutingCandidate::unscored("gemma-4-31B-it-Q8_0", no_caps),
            RoutingCandidate::unscored("Qwen3.6-35B-A3B-BF16", no_caps),
            RoutingCandidate::unscored("MiniMax-M2.5-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3-Coder-Next-Q4_K_M", no_caps),
        ];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };

        let smalls = ["Qwen3.5-2B-Q4_K_M", "Qwen3.5-9B-Q4_K_M"];
        // Across many picks, small-tier names must never win when big-tier is non-empty.
        for _ in 0..200 {
            let picked = pick_model_classified(&cl, &available).expect("some pick");
            assert!(
                !smalls.contains(&picked),
                "small-tier model {picked} was picked despite a non-empty big tier"
            );
        }
    }

    #[test]
    fn test_pick_falls_back_to_small_when_no_big_tier() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("Qwen3.5-2B-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3.5-9B-Q4_K_M", no_caps),
        ];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };

        let picked = pick_model_classified(&cl, &available).expect("some pick");
        assert!(picked == "Qwen3.5-2B-Q4_K_M" || picked == "Qwen3.5-9B-Q4_K_M");
    }

    #[test]
    fn test_pick_spreads_across_big_tier() {
        use crate::models::ModelCapabilities;
        use std::collections::HashSet;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("gemma-4-31B-it-Q8_0", no_caps),
            RoutingCandidate::unscored("Qwen3.6-35B-A3B-BF16", no_caps),
            RoutingCandidate::unscored("MiniMax-M2.5-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3-Coder-Next-Q4_K_M", no_caps),
        ];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };

        let mut seen = HashSet::new();
        for _ in 0..500 {
            if let Some(m) = pick_model_classified(&cl, &available) {
                seen.insert(m);
            }
            // Sleep a nanosecond-scale amount so the seed changes between iterations
            std::thread::sleep(std::time::Duration::from_nanos(1));
        }
        // Over 500 picks with nanosecond-seeded shuffles, we should see
        // at least 3 of the 4 big-tier models. (Allowing 1 slop for the
        // rare case where timing quantization biases the seed.)
        assert!(
            seen.len() >= 3,
            "expected spread across big-tier models, only saw {seen:?}"
        );
    }

    // ── tok/s-aware weighting ────────────────────────────────────────

    /// Helper: build a scored candidate.
    fn scored<'a>(
        name: &'a str,
        caps: crate::models::ModelCapabilities,
        tps: f64,
        samples: u64,
    ) -> RoutingCandidate<'a> {
        RoutingCandidate {
            name,
            caps,
            tps_hint: Some(tps),
            throughput_samples: samples,
        }
    }

    fn count_picks(available: &[RoutingCandidate<'_>], iterations: usize) -> HashMapCounts {
        use std::collections::HashMap;
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..iterations {
            if let Some(name) = pick_model_classified(&cl, available) {
                *counts.entry(name.to_string()).or_insert(0) += 1;
            }
            // Bump the nanosecond seed between iterations.
            std::thread::sleep(std::time::Duration::from_nanos(1));
        }
        HashMapCounts(counts)
    }

    struct HashMapCounts(std::collections::HashMap<String, usize>);

    impl HashMapCounts {
        fn get(&self, name: &str) -> usize {
            self.0.get(name).copied().unwrap_or(0)
        }
        fn total(&self) -> usize {
            self.0.values().sum()
        }
    }

    #[test]
    fn weighted_pick_all_cold_is_roughly_uniform() {
        // Backwards-compat: when nothing has tps data, picks should be
        // roughly uniform — same effective shape as the old random shuffle.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("alpha-31B", no_caps),
            RoutingCandidate::unscored("beta-31B", no_caps),
            RoutingCandidate::unscored("gamma-31B", no_caps),
        ];
        let counts = count_picks(&available, 600);
        let expected = counts.total() / 3;
        // Allow ±50% (300 picks across 3 models is loose statistical ground,
        // but we just need to see no model is starved).
        for name in ["alpha-31B", "beta-31B", "gamma-31B"] {
            let got = counts.get(name);
            assert!(
                got > expected / 2,
                "cold model {name} was starved: {got}/{expected} expected"
            );
        }
    }

    #[test]
    fn weighted_pick_fast_wins_majority_but_slow_still_gets_some() {
        // Core design claim: fast tok/s tilts routing without starving the slow.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            scored("fast-31B", no_caps, 80.0, 50),
            scored("slow-31B", no_caps, 6.0, 50),
        ];
        let counts = count_picks(&available, 600);
        let fast = counts.get("fast-31B");
        let slow = counts.get("slow-31B");
        // Fast should win clearly more often.
        assert!(
            fast > slow,
            "fast tok/s model should win majority: fast={fast} slow={slow}"
        );
        // Fast wins by a meaningful margin (≥ 1.5x).
        assert!(
            fast as f64 > 1.5 * slow as f64,
            "fast model should win by at least 1.5x: fast={fast} slow={slow}",
        );
        // Slow model still gets meaningful traffic (exploration + clamp keep it alive).
        assert!(
            slow > 30,
            "slow model must not be starved (exploration keeps it alive): got {slow}",
        );
    }

    #[test]
    fn weighted_pick_cold_model_competes_with_hot_fast() {
        // A brand-new peer (no samples) must still get meaningful traffic
        // against an established fast peer — otherwise it can never
        // accumulate the data it needs to be scored.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            scored("hot-fast-31B", no_caps, 80.0, 50),
            RoutingCandidate::unscored("cold-newcomer-31B", no_caps),
        ];
        let counts = count_picks(&available, 600);
        let cold = counts.get("cold-newcomer-31B");
        // Cold gets NEUTRAL_WEIGHT (25) vs hot's clamped 80 — so cold
        // should still see at least a healthy minority of traffic.
        assert!(
            cold > 100,
            "cold newcomer must get fair traffic to accumulate samples: got {cold}/600"
        );
    }

    #[test]
    fn weighted_pick_low_sample_count_treated_as_cold() {
        // A model with only 1-2 samples shouldn't have those samples
        // dominate routing — we want a few real measurements before tps
        // participates.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            // Both are 31B "big-tier" names — single-digit-B partition
            // doesn't separate them.
            scored("alpha-31B", no_caps, 100.0, 1), // 1 sample of "fast" — should be ignored
            scored("beta-31B", no_caps, 100.0, 1),
            scored("gamma-31B", no_caps, 100.0, 1),
        ];
        let counts = count_picks(&available, 600);
        // All three should land near uniform since none has enough samples.
        let expected = counts.total() / 3;
        for name in ["alpha-31B", "beta-31B", "gamma-31B"] {
            let got = counts.get(name);
            assert!(
                got > expected / 2,
                "low-sample model {name} was treated as scored instead of cold: {got}"
            );
        }
    }

    #[test]
    fn candidate_weight_clamps_extremes() {
        // Sanity: weight stays bounded so no peer can fully starve or monopolize.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();

        let glacial = scored("glacial", no_caps, 0.5, 100);
        let blazing = scored("blazing", no_caps, 500.0, 100);
        let cold = RoutingCandidate::unscored("cold", no_caps);

        let wg = candidate_weight(&glacial);
        let wb = candidate_weight(&blazing);
        let wc = candidate_weight(&cold);

        assert!(wg >= TPS_WEIGHT_MIN, "glacial weight floored: {wg}");
        assert!(wb <= TPS_WEIGHT_MAX, "blazing weight capped: {wb}");
        assert!(
            (wc - TPS_NEUTRAL_WEIGHT).abs() < f64::EPSILON,
            "cold weight should be neutral: {wc}",
        );
    }
}
