//! Small-model tool-calling reliability + small-context compaction
//! for Mesh LLM.
//!
//! This crate is a first-pass port of the rules from
//! [forge](https://github.com/antoinezambelli/forge) v0.6.0 (with PR
//! #72 applied) into the host-runtime backend layer of Mesh LLM. See
//! `docs/design/FORGE_INTEGRATION.md` for the design and validation
//! plan.
//!
//! Two independent decorators on the `OpenAiBackend` trait:
//!
//! - [`GuardrailBackend`]: rescue / validate / retry. Engaged when
//!   the host's model is in the small tier (single-digit B) and the
//!   request has tools.
//! - [`CompactingBackend`]: tiered in-place compaction + context
//!   warnings. Engaged when the host's `n_ctx` is small (≤ the
//!   configured threshold) and the inbound request is approaching it.
//!
//! Both are constructed at host backend wiring; the wrap decision is
//! made once, not per request. Use [`backend::maybe_wrap_backend`]
//! from the host runtime to apply them.

pub mod backend;
pub mod compact;
pub mod error_tracker;
pub mod facade;
pub mod nudges;
pub mod rescue;
pub mod respond;
pub mod step_enforcer;
pub mod types;
pub mod validator;

pub use backend::{
    maybe_wrap_backend, CompactingBackend, GuardrailBackend, GuardrailBackendConfig, WrapConfig,
    MESH_COMPACT_FIELD, MESH_GUARDRAILS_FIELD,
};
pub use compact::{CompactConfig, CompactOutcome};
pub use facade::{GuardrailConfig, Guardrails};
pub use types::{GuardrailAction, LlmResponse, MessageType, Nudge, NudgeKind, ToolCall};
