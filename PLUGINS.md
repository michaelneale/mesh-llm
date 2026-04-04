# Plugins

This document defines the `mesh-llm` plugin architecture.

It describes the target architecture, not just the code as it exists today.

As implementation lands, this document should be updated to match the intended end state and the concrete protocol and runtime decisions that have been made.

The main goals are:

- keep `mesh-llm` decoupled from specific plugins
- let bundled plugins be auto-registered without special-casing product behavior
- make MCP and HTTP first-class host projections
- support large request and response bodies without blocking control traffic
- keep plugin author boilerplate low

## Design Summary

A plugin is a local service process launched by `mesh-llm`.

The system has three core pieces:

- one long-lived control connection per plugin process
- zero or more short-lived negotiated streams for large or streaming data
- one declarative plugin manifest that the host `stapler` projects into MCP, HTTP, and optional promoted product APIs

`mesh-llm` remains the owner of:

- plugin lifecycle
- local IPC
- stapling manifest-declared services onto host-facing protocols
- HTTP serving
- MCP serving
- capability routing
- mesh participation and peer-to-peer transport

A plugin owns:

- its own feature logic
- local state
- operation handlers
- resource handlers
- prompt handlers
- plugin-specific mesh channel semantics

Plugins do not need to implement raw MCP or raw HTTP servers.

The `stapler` is the host projection layer that turns plugin manifests into exposed MCP and HTTP surfaces.

## High-Level Model

The plugin system is service-oriented rather than transport-oriented.

A plugin declares services such as:

- operations
- resources
- resource templates
- prompts
- completions
- external service endpoints
- mesh channels
- named capabilities

Those declarations are projected by the host `stapler` into:

- MCP tools, resources, prompts, and completions
- HTTP routes
- optional top-level product APIs backed by named capabilities

The plugin author writes handlers once. The host exposes them across surfaces.

## Core Principles

### 1. Bundled Plugins Are Allowed

Plugins shipped in this source tree may be auto-registered by the host.

That is acceptable coupling.

What is not acceptable is embedding one plugin's runtime behavior directly into core mesh logic. Core mesh transport and state should stay generic.

### 2. One Control Connection, Many Data Streams

Each plugin process has one long-lived control connection.

Use the control connection for:

- initialize / health / shutdown
- manifest registration
- small RPC-style requests
- mesh event delivery
- stream negotiation
- cancellation

Do not use the control connection for large uploads, downloads, or long-lived streaming responses.

For large or streaming payloads, the host and plugin negotiate a short-lived side stream.

### 3. MCP Is A Host Projection

`mesh-llm` is the MCP server.

Plugins do not need to implement MCP JSON-RPC directly. They declare MCP-facing services in the manifest, and the host `stapler` exposes them over MCP.

### 4. HTTP Is A Host Projection

`mesh-llm` owns the HTTP server.

Plugins may declare HTTP bindings, but they do not need to run an HTTP server themselves. The host `stapler` maps HTTP requests onto plugin operations and resources.

### 5. Capabilities Are Stable Product Contracts

When `mesh-llm` wants a stable product API such as `/api/objects`, core should depend on a named capability like `object-store.v1`, not on a specific plugin ID like `blobstore`.

## Architecture

### Control Session

There is one long-lived control session between host and plugin.

The control session is used for:

- plugin startup and manifest exchange
- health checks
- small request / response calls
- plugin-to-host notifications
- host-to-plugin mesh events
- opening and closing streams
- cancellation and error reporting

The control session should stay responsive even while the plugin is sending or receiving large payloads.

### Streams

Streams are short-lived negotiated channels for a single request, response, or transfer.

They are opened via the control session and then carry data independently.

Streams are used for:

- large HTTP request bodies
- large HTTP responses
- streaming uploads and downloads
- server-sent events or similar long-lived responses
- future bulk data flows between host and plugin

On Unix, streams map to short-lived Unix sockets.

On Windows, streams map to short-lived named pipes.

The protocol concept is `stream`, not `socket`, so the transport binding remains platform-specific.

### Why Streams Exist

The current single-socket framed-envelope design is vulnerable to head-of-line blocking. Even chunked transfer traffic still competes with health checks, tool calls, mesh events, and other control messages on the same queue.

This architecture avoids that by separating:

- control plane traffic
- bulk and streaming data traffic

## Manifest

On startup, a plugin returns a manifest that declares what it provides.

Conceptually, the manifest contains:

- plugin identity and version
- operations
- resources
- resource templates
- prompts
- completions
- external service endpoints
- HTTP bindings
- mesh channels
- provided capabilities

The manifest is the source of truth for host projections.

## Plugin Author Experience

The primary design goal is very low boilerplate.

A plugin should look roughly like this:

```rust
plugin! {
    name: "blackboard",
    version: env!("CARGO_PKG_VERSION"),

    mcp: [
        tool("feed", feed)
            .description("Read recent blackboard messages")
            .input::<FeedArgs>()
            .output::<Vec<FeedItem>>()
            .http_get("/feed"),

        tool("post", post)
            .description("Post a blackboard message")
            .input::<PostArgs>()
            .output::<PostResult>()
            .http_post("/post"),

        resource("blackboard://snapshot", snapshot)
            .description("Current blackboard snapshot"),

        prompt("status_brief", status_brief)
            .description("Create a short status brief from recent blackboard activity"),
    ],

    capabilities: [
        capability("mesh-blackboard.v1"),
    ],
}
```

The plugin author declares services and writes normal typed handlers. The runtime and `stapler` handle:

- schema exposure
- MCP projection
- HTTP projection
- request validation
- stream negotiation
- transport details

Plugin authors should not manually implement:

- MCP `tools/list`
- MCP `tools/call`
- MCP `resources/read`
- HTTP routing
- control-plane socket negotiation

## External Endpoints

Plugins may also register external service endpoints.

This is a control-plane declaration, not a request proxying requirement.

In this design, a plugin may tell `mesh-llm`:

- an inference endpoint exists
- an MCP endpoint exists
- how to reach it
- how to health check it
- optionally how to start and stop it

`mesh-llm` then talks to that endpoint directly when appropriate.

This keeps heavy data-plane traffic out of plugin IPC.

### Why Endpoint Registration Exists

Some services already speak a protocol that `mesh-llm` knows how to use.

Examples:

- a local OpenAI-compatible inference server
- an external MCP server reachable over stdio, Unix socket, named pipe, or TCP

In these cases, the plugin should not need to proxy all traffic through itself. It should be able to register the service with the host and remain the control-plane owner for:

- discovery
- lifecycle
- readiness
- availability

### Endpoint DSL

The intended plugin author experience is:

```rust
plugin! {
    name: "local-services",
    version: env!("CARGO_PKG_VERSION"),

    endpoints: [
        inference_endpoint("mlx")
            .protocol(openai_compatible())
            .transport(http("http://127.0.0.1:8091"))
            .models(list_models)
            .health(check_mlx)
            .lifecycle(start_mlx, stop_mlx),

        mcp_endpoint("filesystem")
            .stdio(command("npx").arg("-y").arg("@modelcontextprotocol/server-filesystem"))
            .namespace("filesystem")
            .health(check_filesystem_mcp),

        mcp_endpoint("notes")
            .transport(unix_socket("/tmp/notes-mcp.sock"))
            .namespace("notes")
            .health(check_notes_mcp),
    ],
}
```

For very common cases, a fully declarative form should also be supported:

```rust
plugin! {
    name: "adapters",
    version: env!("CARGO_PKG_VERSION"),

    endpoints: [
        inference_endpoint("vllm")
            .openai_http("http://127.0.0.1:8000")
            .health(http_get("/health"))
            .discover_models(from_openai_models()),

        mcp_endpoint("fetch")
            .stdio(command("uvx").arg("mcp-server-fetch"))
            .namespace("fetch")
            .health(inherit_connection()),
    ],
}
```

### Inference Endpoints

An inference endpoint is a service that `mesh-llm` can call directly using its inference proxy logic.

The plugin is not the inference data path. The plugin is the adapter that describes or manages the endpoint.

This means:

- plugin IPC handles registration, lifecycle, and health
- `mesh-llm` sends `/v1/*` traffic directly to the registered endpoint
- streaming responses do not pass through the plugin process

This is the preferred model for local or managed OpenAI-compatible inference servers.

### MCP Endpoints

An MCP endpoint is a service that `mesh-llm` can connect to directly and aggregate into its own MCP surface.

Again, the plugin is the control-plane owner, not the transport proxy.

This means:

- the plugin declares where the MCP server is
- `mesh-llm` connects to it directly
- `mesh-llm` may namespace and aggregate its tools, resources, prompts, and completions
- the plugin remains responsible for lifecycle and availability metadata

### Health And Availability

Endpoint health is separate from plugin health.

If an endpoint health check fails:

- the endpoint becomes unavailable
- the endpoint is removed from routing or aggregation
- the plugin remains loaded
- the plugin is not marked disabled
- the host keeps checking health

If health returns:

- the endpoint becomes available again automatically

This is important because a plugin may be healthy while its managed or discovered service is:

- starting
- restarting
- temporarily unhealthy
- reloading a model
- intentionally stopped

The host should treat plugin liveness and endpoint liveness as separate concerns.

### Recommended State Model

Conceptually, the system should track at least:

- plugin state
- endpoint state
- model or route availability

Suggested plugin states:

- `starting`
- `running`
- `degraded`
- `disconnected`
- `failed`

Suggested endpoint states:

- `unknown`
- `starting`
- `healthy`
- `unhealthy`
- `unavailable`

Suggested routed availability states:

- `advertised`
- `routable`
- `draining`
- `unavailable`

Routing decisions should depend on endpoint health, not just plugin process health.

## MCP

MCP is implemented by the host, not by individual plugins.

The plugin author marks which services should appear in MCP:

- `tool(...)`
- `resource(...)`
- `resource_template(...)`
- `prompt(...)`
- `completion(...)`

The host then synthesizes:

- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`
- `prompts/list`
- `prompts/get`
- completions where applicable

External MCP endpoints may also be aggregated into the host's MCP surface via the `endpoints:` declarations described above.

### MCP Naming

By default, tool, resource, and prompt names should be plugin-namespaced.

Examples:

- tool: `blackboard.feed`
- tool: `blackboard.post`
- resource: `blackboard://snapshot`
- prompt: `blackboard.status_brief`

Friendly aliases may be added for bundled plugins, but the canonical identity should remain namespaced to avoid collisions.

### MCP Streaming

MCP-facing operations may be:

- buffered
- streaming input
- streaming output
- streaming input and output

For streaming operations, the host uses negotiated side streams internally rather than pushing large data through the control connection.

## HTTP Bindings

Plugins may declare HTTP bindings as part of the manifest.

These bindings let a plugin feel native over HTTP without requiring custom host route code for each plugin.

### Default Mounting

Plugin-defined HTTP bindings should be mounted under a plugin-owned namespace by default.

Examples:

- `/api/plugins/blackboard/feed`
- `/api/plugins/blackboard/post`
- `/api/plugins/object-store/objects`

This avoids collisions and keeps plugin-specific APIs out of the top-level product namespace unless explicitly promoted.

### Promoted Product Routes

Some routes may become stable product APIs owned by `mesh-llm`, for example:

- `/api/objects`

These routes should be backed by named capabilities, not by hard-coded plugin IDs.

Example:

- top-level route: `/api/objects`
- required capability: `object-store.v1`
- provider plugin: whichever plugin the host resolves for that capability

This keeps product APIs stable while allowing the backing plugin to change.

External endpoints do not automatically become HTTP routes. They are service registrations that the host may use for routing or aggregation according to their endpoint kind.

### Buffered vs Streamed HTTP

HTTP bindings may be declared as:

- buffered request / buffered response
- streamed request / buffered response
- buffered request / streamed response
- streamed request / streamed response

The host decides whether to keep the invocation on the control channel or negotiate a side stream based on the binding mode and payload size.

## Streams And Large Transfers

Large payloads must not ride the main control connection.

Instead, the control session negotiates a short-lived stream for the transfer.

Conceptual flow:

1. host sends `OpenStream`
2. plugin accepts
3. host and plugin establish a short-lived local stream
4. request or response bytes flow on that stream
5. either side may cancel
6. stream is torn down and cleaned up

This design supports:

- 10 GB uploads
- large downloads
- long-lived streaming responses
- future websocket-like or SSE-style responses

without blocking health checks or other control traffic.

## Suggested Control Messages

The exact wire format is still open, but the protocol should support concepts like:

- `Initialize`
- `InitializeResponse { manifest }`
- `Health`
- `Shutdown`
- `Invoke`
- `InvokeResult`
- `Notify`
- `MeshEvent`
- `OpenStream`
- `OpenStreamResult`
- `CancelStream`
- `StreamError`

The stream protocol itself may be raw bytes or lightly framed bytes, depending on the use case.

## Capabilities

Capabilities let core depend on behavior rather than on plugin names.

Examples:

- `object-store.v1`
- `mesh-blackboard.v1`
- `artifact-cache.v1`
- `model-catalog-provider.v1`

Capabilities are used when:

- core needs a stable product contract
- multiple plugins could satisfy the same role
- the host wants to promote a route into the top-level API

Capabilities are not required for every plugin. They are mainly for shared contracts that `mesh-llm` itself depends on.

Endpoint registration is related but distinct:

- capabilities express stable contracts that core may depend on
- endpoints express concrete service instances that the host can talk to directly

An endpoint may satisfy a capability, but the two ideas should remain separate in the design.

## Mesh Channels

Plugins may declare mesh channels for plugin-specific peer-to-peer coordination.

These should use the generic plugin mesh transport rather than dedicated core stream types for individual plugins.

Core should not embed plugin-specific wire protocols in the main mesh transport when the behavior can live behind the generic plugin channel mechanism.

## What The Host Owns

The host is responsible for:

- launching plugins
- registering bundled plugins
- validating plugin identity
- keeping the control session alive
- stream negotiation and cleanup
- request validation
- HTTP mounting
- MCP exposure
- capability resolution
- route collision detection
- permissions and policy enforcement

## What Plugins Own

A plugin is responsible for:

- declaring its manifest
- implementing handlers
- handling its own local state
- reading and writing stream payloads when invoked
- implementing any plugin-specific business logic

## Non-Goals

The plugin system should not require each plugin to:

- run its own HTTP server
- run its own MCP server
- manually negotiate Unix socket paths in application code
- hard-code core route registration in `mesh-llm`

The plugin system should also avoid:

- top-level product APIs that are secretly bound to one plugin ID
- plugin-specific core mesh stream types when generic plugin channels are sufficient

## Open Questions

The following are intentionally left open for implementation design:

- exact manifest schema
- exact control protocol message shapes
- exact stream framing format
- capability provider selection when multiple plugins implement the same capability
- whether promoted product routes are configured statically or negotiated dynamically
- how auth and policy rules are expressed for plugin-defined HTTP bindings

## Architecture Baseline

- bundled plugins may be auto-registered
- core mesh logic remains plugin-agnostic
- MCP and HTTP are first-class host projections
- product APIs depend on capabilities, not plugin IDs
- large data flows use negotiated side streams, not the control socket
