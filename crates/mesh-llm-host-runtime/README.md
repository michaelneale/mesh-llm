# mesh-llm-host-runtime

`mesh-llm-host-runtime` owns the host runtime implementation behind the
`mesh-llm` binary.

This crate includes the remaining host-only subsystems that coordinate a local
node:

- CLI parsing, command handlers, terminal output, and dashboard rendering
- management API routes and OpenAI-compatible ingress wiring
- mesh node orchestration, gossip integration, peer state, and host routing glue
- embedded skippy runtime coordination and stage deployment
- host-side plugin runtime, MCP bridge, and built-in plugins
- runtime-data aggregation and API view shaping
- host-local model catalog, inventory, and download flows that still coordinate
  CLI/runtime behavior

The package named `mesh-llm` is now the app assembly crate: it exposes the
binary entrypoint and re-exports this runtime crate for compatibility. Keep new
pure/shared ownership in narrower crates such as `mesh-llm-types`,
`mesh-llm-protocol`, `mesh-llm-routing`, `mesh-llm-identity`,
`mesh-llm-system`, `mesh-llm-ui`, `model-*`, `openai-frontend`, and
`skippy-*`.
