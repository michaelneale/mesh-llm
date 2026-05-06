# Jan Mesh API Integration

This document defines the initial integration contract for using `mesh-api` inside
Jan as a Tauri-native client SDK.

Scope:

- Jan acts as a **mesh client**
- Jan does **not** host the mesh console
- Jan persists a stable owner keypair in Jan storage
- Streaming is delivered to the frontend through Tauri events

## Goals

Jan needs:

- a Rust-native integration path with no separate Node client process
- a strong TypeScript surface for the web app and extensions
- stable identity across app restarts
- incremental streaming for chat/responses

Jan does not need in this phase:

- embedded mesh console assets
- mesh host/server APIs
- full embedded node hosting

## Layering

- `mesh-client`
  - internal implementation crate
- `mesh-api`
  - public Rust client SDK
- `tauri-plugin-mesh-api` in Jan
  - Jan-native bridge
- `@janhq/tauri-plugin-mesh-api`
  - guest JS/TypeScript API consumed by Jan frontend

The Jan plugin must depend on `mesh-api`, not `mesh-client`.

## 1. Jan Plugin Surface

The plugin should own all live client instances in Rust.

Recommended plugin name:

- `mesh-api`

Recommended Tauri command namespace:

- `plugin:mesh-api|...`

### Commands

#### `create_client`

Creates a managed client handle in the plugin and returns a Jan-local client ID.

Arguments:

- `inviteToken: string`
- `ownerKeypairStrategy?: "persisted"`
- `label?: string`

Returns:

- `{ clientId: string, ownerId: string }`

Behavior:

- load persisted owner keypair from Jan storage, or generate and persist one if absent
- construct `mesh_api::ClientBuilder`
- store the resulting client in plugin state

#### `dispose_client`

Arguments:

- `clientId: string`

Returns:

- `void`

Behavior:

- remove the client handle from plugin state
- cancel in-flight request listeners owned by that client

#### `join`

Arguments:

- `clientId: string`

Returns:

- `void`

#### `list_models`

Arguments:

- `clientId: string`

Returns:

- `Model[]`

#### `chat`

Arguments:

- `clientId: string`
- `request: ChatRequest`

Returns:

- `{ requestId: string }`

#### `responses`

Arguments:

- `clientId: string`
- `request: ResponsesRequest`

Returns:

- `{ requestId: string }`

#### `cancel`

Arguments:

- `clientId: string`
- `requestId: string`

Returns:

- `void`

#### `status`

Arguments:

- `clientId: string`

Returns:

- `{ connected: boolean, peerCount: number }`

#### `disconnect`

Arguments:

- `clientId: string`

Returns:

- `void`

#### `reconnect`

Arguments:

- `clientId: string`

Returns:

- `void`

#### `get_identity`

Arguments:

- `clientId: string`

Returns:

- `{ ownerId: string }`

#### `reset_identity`

Arguments:

- `profile?: string`

Returns:

- `{ ownerId: string }`

Behavior:

- delete persisted keypair
- generate a new one
- intended for advanced settings/debug flows only

## 2. Streaming Model

Jan should use:

- Tauri commands for unary operations
- Tauri events for streaming and lifecycle events

This matches Jan’s current native integration pattern better than callback-heavy command APIs.

### Event Channel

Use one event topic:

- `mesh-api://event`

Each event payload should include:

- `clientId: string`
- `kind: MeshEventKind`

Recommended TypeScript shape:

```ts
type MeshEvent =
  | { clientId: string; kind: 'connecting' }
  | { clientId: string; kind: 'joined'; nodeId: string }
  | { clientId: string; kind: 'modelsUpdated'; models: Model[] }
  | { clientId: string; kind: 'tokenDelta'; requestId: string; delta: string }
  | { clientId: string; kind: 'completed'; requestId: string }
  | { clientId: string; kind: 'failed'; requestId: string; error: string }
  | { clientId: string; kind: 'disconnected'; reason: string }
```

### Why one event topic

One topic is preferable to per-request topics because:

- Jan already has frontend-side event routing utilities
- subscription management stays simple
- request events can be filtered in TS by `clientId` and `requestId`
- reconnect and model-update events naturally share the same stream

### Rust-side event bridge

The plugin should:

- create an `EventListener` adapter for each `chat` / `responses` request
- emit every `mesh_api::events::Event` through `app_handle.emit(...)`
- include `clientId` on every payload

### Frontend API expectations

The guest JS package should expose:

- `createClient()`
- `join()`
- `listModels()`
- `chat()`
- `responses()`
- `cancel()`
- `status()`
- `disconnect()`
- `reconnect()`
- `onEvent(handler)`

It should not expose raw Tauri details to the rest of Jan.

## 3. Persisted Owner Keypair In Jan Storage

Jan wants a stable mesh identity across restarts.

### Storage requirement

Persist the owner keypair in Jan storage on the native side.

Recommended storage key:

- `mesh_api.owner_keypair.v1`

Recommended serialized payload:

```json
{
  "version": 1,
  "signingKeyHex": "<64 hex chars>",
  "encryptionKeyHex": "<64 hex chars>"
}
```

### Storage location

Use Jan’s existing Tauri store mechanism, not frontend localStorage.

Reason:

- the keypair is native-side identity material
- Rust needs direct access to it during client creation
- Tauri store already exists in Jan app capabilities and startup flow

### Lifecycle

On `create_client`:

1. read `mesh_api.owner_keypair.v1`
2. if present, reconstruct `mesh_api::OwnerKeypair`
3. if absent, generate a new keypair
4. persist it immediately
5. build the client with the persisted identity

### Profile support

If Jan later supports multiple mesh profiles, extend the key to:

- `mesh_api.owner_keypair.v1.<profile>`

Do not introduce profile selection in the first pass unless the Jan UI already needs it.

### Rotation

Rotation should be explicit only:

- not on app update
- not on invite token change
- not on reconnect

The only identity-changing flows should be:

- manual reset from settings
- future import/restore flow

### Security note

This requirement says “persisted owner keypair in Jan storage”.

That is acceptable for the first implementation, but it should be treated as a local-secret storage decision. If Jan later wants stronger local-secret protection, the persistence backend can move behind the same plugin API without changing the frontend contract.

## Recommended First Implementation Order

1. Add `mesh-api` methods the plugin needs without exposing `mesh-client` internals.
2. Implement persisted keypair helpers in the Jan Tauri plugin.
3. Implement client handle state in plugin Rust.
4. Implement command APIs.
5. Implement event emission for `chat` and `responses`.
6. Add the guest JS package with strong TS typing.

## Non-Goals For This Phase

- embedded mesh node hosting
- mesh console embedding
- Swift/Kotlin bindings for Jan
- per-request event channels
- multi-profile identity UX
