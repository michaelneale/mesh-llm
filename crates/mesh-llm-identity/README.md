# mesh-llm-identity

Shared owner identity and message-envelope crypto for Mesh LLM crates.

This crate owns dependency-light identity primitives that are needed by both the
host runtime and embedded clients:

- owner keypair generation and owner ID derivation
- signed-and-encrypted control-message envelopes
- key provider traits for client/runtime integration
- shared crypto error types

Host-only persistence stays out of this crate. OS keychain access, encrypted
keystore files, trust-store files, and CLI prompting remain in the host-facing
crates because they depend on local machine state and interactive policy.
