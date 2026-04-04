# Multi-Modal Roadmap

## Status

Implemented for the current multimodal scope.

Phases 1 through 5 in this plan are complete:

- multimodal capability advertisement across status, models, routing, and UI
- request-scoped ingress-local blob/object transport
- console support for image, audio, and file attachments
- multimodal `/v1/chat/completions`
- multimodal `/v1/responses` for both non-streaming and streaming requests

What remains is polish and hardening of the existing path, not new endpoint scope.

## Goals

- Support multimodal inference through mesh-llm using llama.cpp-compatible models.
- Route image and audio requests to compatible hosts automatically.
- Keep media transport bounded and private by default.
- Preserve protocol compatibility unless we explicitly choose a breaking change.
- Keep the crate/plugin split clean: core owns inference routing, plugin owns media object storage.

## Non-Goals

- Permanent distributed file storage
- IPFS/libp2p-first design
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `v1/realtime`
- Audio generation from llama alone
- Native end-to-end video inference on the current llama.cpp path

## API Targets

### Phase 1

- `POST /v1/chat/completions`
- `GET /v1/models`

This is the shortest path because llama.cpp already supports multimodal chat here for supported models.

### Phase 2

- `POST /v1/responses`

Implement as a mesh-llm compatibility shim after chat completions are solid.

## Capability Model

Do not collapse everything into `vision`.

Recommended model capabilities:

- `multimodal: bool`
- `vision: CapabilityLevel`
- `audio: CapabilityLevel`
- `reasoning: CapabilityLevel`
- `tool_use: CapabilityLevel`
- `moe: bool`

Why:

- `multimodal` is a useful umbrella signal for UI and coarse filtering.
- `vision` and `audio` are still required for correct routing.
- A model can be multimodal without supporting both image and audio equally.

## Supported Family Matrix

This is the current mesh-llm capability matrix, based on the inference logic in
`mesh-llm/src/models/capabilities.rs`.

Important:

- `supported` means mesh-llm will treat the family/model as runtime-capable for routing and UI.
- `likely` means mesh-llm will surface a weaker multimodal hint, but not rely on it as a hard runtime capability.
- Catalog/download metadata can upgrade a family from `likely` to `supported`, for example:
  - vision models with an `mmproj`
  - models whose metadata exposes `vision_config`, `audio_config`, or modality token IDs

### Vision

| Family / signal | mesh-llm status | Notes / examples |
|---|---|---|
| `Qwen3-VL`, `Qwen3VL` | `supported` | Example: `Qwen3VL-2B-Instruct-Q4_K_M` |
| `Qwen2-VL`, `Qwen2.5-VL` | `supported` | Any matching `qwen2-vl`, `qwen2.5-vl` family name |
| `LLaVA` | `supported` | Covers `llava` family names |
| `mllama` | `supported` | Llama vision variants exposed as `mllama` |
| `PaliGemma` | `supported` | Family-name detection |
| `Idefics` | `supported` | Family-name detection |
| `Molmo` | `supported` | Family-name detection |
| `InternVL` | `supported` | Family-name detection |
| `GLM-4V` / `GLM4V` | `supported` | Family-name detection |
| `Ovis` | `supported` | Family-name detection |
| `Florence` | `supported` | Family-name detection |
| Any catalog model with `mmproj` | `supported` | This is the strongest repo-local signal for GGUF vision models |
| Any model with `vision_config` or vision token IDs | `supported` | Derived from local/remote metadata JSON |
| Generic `-vl`, `_vl`, `video`, `multimodal`, `image` names | `likely` | Hint only until promoted by stronger metadata |

### Audio

| Family / signal | mesh-llm status | Notes / examples |
|---|---|---|
| `Qwen2-Audio` | `supported` | Covers `qwen2-audio` / `qwen2_audio` |
| `SeaLLM-Audio` | `supported` | Covers `seallm-audio` / `seallm_audio` |
| `Ultravox` | `supported` | Family-name detection |
| `Omni` | `supported` | Example: `Qwen2.5-Omni-3B-Q4_K_M` |
| `Whisper` | `supported` | Family-name detection |
| Generic `audio` / `speech` family names | `supported` | Strong signal in current inference code |
| Any model with `audio_config` or audio token IDs | `supported` | Derived from local/remote metadata JSON |
| Generic `voice` naming only | `likely` | Hint only unless stronger metadata is present |

### Multimodal Umbrella

| Case | mesh-llm status | Notes |
|---|---|---|
| `vision != none` | `multimodal = true` | Vision implies multimodal |
| `audio != none` | `multimodal = true` | Audio implies multimodal |
| Both vision and audio supported | `multimodal = true` | Best fit for mixed image+audio routing |
| Name-only generic `multimodal` signal | `likely` for modality-specific support | Not enough on its own for hard vision/audio routing |

### Current Practical Examples

| Model | Vision | Audio | Notes |
|---|---|---|---|
| `Qwen3VL-2B-Instruct-Q4_K_M` | `supported` | `none` | Good current image test model |
| `Qwen2.5-Omni-3B-Q4_K_M` | `none` or metadata-dependent | `supported` | Good current audio test model |
| Catalog GGUF with `mmproj` | `supported` | depends | Vision promoted by sidecar |
| Generic `multimodal` name with no sidecar/metadata | `likely` | `likely`/`none` | UI hint only; not a strong routing guarantee |

## Protocol Plan

Preferred path: additive change.

- add `multimodal`
- add `audio`
- keep `vision` meaning image/vision support

Repurposing `vision` to mean generic multimodal would be a breaking semantic change:

- old nodes would misinterpret the flag
- routing and UI would become incorrect in mixed-version meshes
- protobuf only protects unknown fields, not changed meaning of existing fields

If we ever want to make that breaking change anyway, it should be an explicit protocol-version decision.

## Request Formats

### Chat Completions

Accept and preserve OpenAI-style content parts:

- text
- image URL / data URL
- audio URL / data URL
- future file-style references

Video is not a Phase 1 request shape target. Treat video separately from image/audio until the serving stack can handle it natively.

mesh-llm should detect multimodal intent from structured content blocks, not just text keywords.

### Responses API

Add a translation layer from `responses` input items into chat-completions-style message content after Phase 1 is working.

## Routing Notes

- `audio` and `multimodal` are part of capability inference and surfaced in `/api/status` and `/v1/models`.
- Multimodal payloads are detected by structured content inspection, not just prompt text.
- `model=auto` prefers:
  - vision-capable hosts for image inputs
  - audio-capable hosts for audio inputs
  - models supporting both when both are present
- Media requests bypass the fragile pre-plan path where needed.

## Video Support

Some model families support video at the model level, but that should not be treated as available in mesh-llm yet.

Current working assumption:

- open multimodal models with video support exist
- the current llama.cpp integration path in mesh-llm should be treated as image/audio only
- native video input should stay out of scope until upstream serving support is real and reliable

Recommended first implementation path for video:

- accept uploaded video into the same request-scoped blob plugin
- decode and sample frames server-side
- send sampled frames as ordered image inputs to a vision-capable model
- optionally include timestamp metadata in the prompt or content structure

Why this path first:

- it reuses the existing blob/object lifecycle
- it works with current image-capable serving paths
- it avoids blocking on native video support in llama.cpp

Follow-up requirements when we do video:

- add `video` capability metadata only when there is a real serving path behind it
- define upload limits, codec/container acceptance, and frame-sampling defaults
- decide whether video should be exposed only through `responses`, through `chat/completions`, or both
- make it explicit in the UI when video is being converted into sampled images rather than handled natively

## Media Transport Plan

Large media should not travel inside request JSON by default.

### First Pass

Use a request-scoped mesh content store implemented as a plugin.

Properties:

- ingress node stores the uploaded object locally
- no replication
- client receives an opaque secret token, not a content hash
- completion request references the token
- serving node fetches the object from the ingress node if needed
- ingress node deletes the object when the request reaches terminal state
- short TTL cleanup handles crashes, disconnects, and abandoned requests

### External vs Internal Identity

- external object ID: secret high-entropy token
- internal storage key: content hash for integrity and dedupe only

The token should be user-visible and fetch-authorizing. The content hash should remain internal.

## Blob Plugin Shape

The blob/content store should be a plugin, not core storage logic.

Core responsibilities:

- parse inference requests
- choose inference targets
- decide request lifecycle
- tell the plugin when a request starts and ends

Plugin responsibilities:

- store request-scoped media on ingress
- mint and validate opaque tokens
- serve media fetches to remote hosts
- enforce TTL and cleanup

Initial plugin operations:

- `put_request_object`
- `get_request_object`
- `complete_request`
- `abort_request`
- `reap_expired_objects`

## Object Lifecycle

### Request-Scoped Media

1. Client uploads image/audio/file to the ingress node.
2. Plugin stores bytes locally and returns a short-lived secret token.
3. Client sends completion request referencing that token.
4. Chosen host fetches from ingress if the object is not local.
5. Request completes, fails definitively, or is canceled.
6. Ingress plugin deletes token and blob after a short grace window.
7. TTL fallback cleans up leaks.

Recommended first-pass behavior:

- no replication
- token bound to one request
- allow a small retry budget for reroute / transport retries
- keep only a short grace period after terminal state

## Console State

### Existing

- image, audio, and file attachment UI
- upload-before-send for request-scoped objects
- token-reference request construction for `chat/completions` and `responses`
- attachment previews / badges before send:
  - image thumbnail
  - audio filename, duration if available, and remove action
  - generic file name, size, mime type, and remove action
- upload state handling:
  - pending
  - uploaded
  - failed
  - retrying
- `model=auto` switching when attachments are present
- capability hints in the model picker:
  - vision
  - audio
  - multimodal
- attachment preservation through retries and reroutes for the same request
- clear fallback UX when no compatible warm model is available
- explicit transcript attachment rendering instead of flattening everything into plain text

### Suggested Console Sequence

1. User attaches image/audio/file.
2. Console creates a pending attachment entry in local UI state.
3. Console uploads the object to the ingress node and receives a request-scoped token.
4. Pending entry becomes uploaded attachment metadata.
5. Completion request is sent with structured message content referring to the token.
6. If the request is retried, reuse the same token while the request is still alive.
7. When the request finishes, normal cleanup happens on the ingress node.

### Nice-to-Have Later

- drag-and-drop attachments
- paste image support
- microphone capture for audio input
- waveform / duration preview for audio
- video upload once there is either frame-sampling support or native serving support
- richer attachment rendering in the transcript
- attachment reuse inside the same conversation only if we later decide to support longer-lived object leases

## Body Size and Transport Limits

Today the proxy body limits are tuned for JSON chat, not large media.

We should:

- keep small inline image support where it is practical
- prefer upload-and-reference for audio and larger files
- avoid raising global HTTP body limits as the main solution

## Implementation Phases

### Phase 1: Capability Plumbing

Complete.

### Phase 2: Routing and Request Parsing

Complete.

### Phase 3: Blob Plugin

Complete.

### Phase 4: Console

Complete.

### Phase 5: Compatibility Shims

Complete for `/v1/responses`.

### Phase 6: Optional Extended Backends

- video ingestion and frame sampling pipeline
- richer persistent asset handling if we ever want reusable attachments

## Follow-Up Work

- polish and harden the existing multimodal `responses` path where needed
- improve capability inference for edge-case model families
- refine attachment UX and transcript rendering
- keep video, if ever pursued, as frame sampling over the existing image path

## Open Questions

- exact token format and request binding rules
- whether tiny inline images should remain supported in addition to blob upload
- whether `responses` should become the primary public API once parity is good enough
- whether multimodal requests should always disable pre-plan, or only for audio
- whether plugin-host fetches should flow through the existing mesh transport or a dedicated plugin RPC path
- when to introduce a real `video` capability instead of treating video as a higher-level image sequence workflow
