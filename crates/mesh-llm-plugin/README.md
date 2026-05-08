# mesh-llm-plugin

`mesh-llm-plugin` owns the plugin author API and shared plugin wire protocol
types used by host-side plugin runtimes and external plugins.

This crate includes:

- the plugin protobuf schema at `proto/plugin.proto`
- generated plugin protocol types exposed through `mesh_llm_plugin::proto`
- typed helpers for plugin manifests, operations, resources, prompts, tasks,
  mesh events, HTTP bindings, MCP projections, and side-stream I/O

Host-only orchestration, process lifecycle, plugin config loading, MCP bridge
hosting, and built-in plugin wiring should remain in the host crate for now and
move later to a dedicated plugin-host crate. Keep this crate focused on the
stable API and protocol surface that plugin authors can depend on.
