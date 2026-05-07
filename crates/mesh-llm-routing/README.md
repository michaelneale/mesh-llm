# mesh-llm-routing

`mesh-llm-routing` owns the shared routing primitives used by the host binary
and the embedded Rust client.

It intentionally stays small:

- `InferenceTarget` describes where a model request should go.
- `ModelTargets` stores per-model candidate targets and performs round-robin or
  sticky candidate selection.
- `total_model_bytes()` calculates GGUF model size, including split GGUF
  shard sets.

Higher-level request parsing, OpenAI transport behavior, peer observation, and
runtime orchestration stay in the owning application crates. This crate is the
common vocabulary those layers use when they exchange routing decisions.
