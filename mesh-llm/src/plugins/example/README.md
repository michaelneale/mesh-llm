# Plugin Surface Example

This is a standalone example plugin executable for `mesh-llm`.

It is intentionally not built into `mesh-llm` itself. Build it separately and
point a plugin config entry at the resulting binary.

## What it demonstrates

- `InitializeRequest` / `InitializeResponse`
- MCP-style `tools/*` RPC handling
- MCP-style `prompts/*` RPC handling
- MCP-style `resources/*` and `resources/templates/list` RPC handling
- MCP-style `completion/complete` RPC handling
- MCP-style `logging/setLevel` RPC handling
- MCP-style `tasks/*` RPC handling
- outbound and inbound `ChannelMessage` traffic
- outbound and inbound `BulkTransferMessage` traffic
- inbound `MeshEvent` delivery

## Build

```bash
cargo build --manifest-path mesh-llm/src/plugins/example/Cargo.toml
```

## Use with mesh-llm

Add a plugin entry that points at the built example binary:

```toml
[[plugin]]
name = "example"
enabled = true
command = "/absolute/path/to/plugin-surface-example"
args = []
```

Once loaded, the example exposes these tools through the normal plugin API and
through the generic plugin MCP bridge:

- `example.snapshot`
- `example.send_message`
- `example.send_bulk`
- `example.clear`
