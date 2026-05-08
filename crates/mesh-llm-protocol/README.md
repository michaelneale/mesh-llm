# mesh-llm-protocol

Wire protocol ownership for Mesh LLM control-plane traffic.

This crate owns the dependency-light protocol surface that is shared by the
host runtime and embedded clients:

- the source node protobuf schema under `proto/node.proto`
- generated protobuf message types under `proto::node`
- QUIC ALPN values and mesh stream type constants
- control-frame validation and encode/decode helpers
- legacy JSON v0 compatibility helpers
- canonical config hashing

Host-specific conversion between protocol messages and runtime-owned structs
still lives in the host crate until the control-plane and runtime boundaries are
separated. Keep runtime orchestration, CLI output, routing policy, and process
lifecycle code out of this crate.
