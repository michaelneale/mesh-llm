# mesh-llm-types

Shared protocol-facing data types for Mesh LLM crates.

This crate owns data shapes that need to be understood by multiple crates without
pulling in the host runtime, QUIC control plane, CLI, UI, or protobuf conversion
layers. It is intentionally small and dependency-light.

Current ownership:

- model capability flags and capability inference signal helpers
- model topology metadata advertised through the mesh
- served-model identity and descriptor types
- model demand counters and shared routing constants

Keep runtime state, peer connection state, protobuf frame validation, and host
process orchestration out of this crate. Those belong in the future protocol,
control-plane, routing, and host-runtime crates.
