# skippy-ffi

Raw Rust declarations for the experimental skippy C ABI.

This crate should stay thin: no topology, protocol, benchmark, or product
behavior belongs here. Prefer adding safe behavior in `skippy-runtime`.

## Responsibilities

- bind to `skippy.h`
- expose C-compatible constants, structs, and function declarations
- keep ABI ownership and error shapes explicit
- link the staged llama.cpp and ggml static archives selected by
  `LLAMA_STAGE_BUILD_DIR`

Consumers should use `skippy-runtime` unless they are extending the ABI
boundary itself.

## Architecture Role

`skippy-ffi` is the lowest Rust layer in the staged stack. It exposes the
patched llama.cpp ABI without owning policy:

```mermaid
flowchart TB
    Mesh["mesh-llm<br/>embedded runtime owner"] --> Runtime
    Server["skippy-server<br/>lifecycle + transport"] --> Runtime["skippy-runtime<br/>safe model/session API"]
    Runtime --> FFI["skippy-ffi<br/>raw ABI declarations"]
    FFI --> Llama["third_party/llama.cpp<br/>skippy.h stage execution"]
```

Keep this crate close to the C ABI. Higher-level behavior such as topology
validation, activation wire conversion, exact-cache policy, package
materialization, telemetry, and benchmark policy belongs in the crates above it.

## Build Integration

By default the build script statically links the staged llama.cpp checkout from
`.deps/llama.cpp/build-stage-abi-static`. Prepare and build it with
`just llama-build`. Set `LLAMA_STAGE_BUILD_DIR` to point at another
prepared build directory, for example a Linux GPU backend build:

| Backend | Expected ggml archive | Runtime libraries linked on Linux |
| --- | --- | --- |
| CUDA | `ggml/src/ggml-cuda/libggml-cuda.a` | `cuda`, `cudart`, `cublas`, `cublasLt` |
| ROCm/HIP | `ggml/src/ggml-hip/libggml-hip.a` | `amdhip64`, `rocblas`, `hipblas` |
| Vulkan | `ggml/src/ggml-vulkan/libggml-vulkan.a` | `vulkan` |

The backend archives are optional. The build script links each one only when it
exists in `LLAMA_STAGE_BUILD_DIR`, so CPU-only and accelerator builds share the
same Rust crate.

## ABI Contract

The staged ABI is versioned as `0.1.18` in both this crate and the patched
llama.cpp header. Version `0` is still experimental, so callers should treat the
ABI as feature-probed rather than permanently stable.

All `skippy_*` functions either return `skippy_status` or, for ABI
discovery/error helpers, return plain values directly. Functions that accept
`out_error` may allocate a `skippy_error`; callers must pass it to
`skippy_error_free` after reading the status/message.

Opaque handles are caller-owned after successful creation:

| Handle | Created by | Released by |
| --- | --- | --- |
| `skippy_model` | `skippy_model_open` | `skippy_model_free` |
| `skippy_session` | `skippy_session_create` | `skippy_session_free` |
| `skippy_model_info` | `skippy_model_info_open` | `skippy_model_info_free` |
| `skippy_slice_plan` | `skippy_slice_plan_create` | `skippy_slice_plan_free` |

Buffer-writing functions use the usual C ABI sizing pattern: the caller passes a
pointer, a capacity, and an output byte/count pointer. If the buffer is too
small, the function returns `BUFFER_TOO_SMALL` and reports the required size in
the output pointer when the implementation can compute it.

## Protocol Shape

The same ABI supports single-stage inference, split runtime stages, state
movement, tokenizer/chat helpers, and model slicing:

```mermaid
sequenceDiagram
    participant R as Rust runtime
    participant F as skippy-ffi
    participant L as patched llama.cpp
    R->>F: skippy_model_open(config)
    F->>L: open model or stage slice
    R->>F: skippy_session_create(model)
    R->>F: prefill/decode/verify
    alt middle or final stage
        R->>F: import activation frame or KV/recurrent state
    end
    F-->>R: logits, predicted token, activation frame, or state bytes
    R->>F: skippy_session_free(session)
    R->>F: skippy_model_free(model)
```

## Feature Flags

`skippy_abi_features` returns a bitmask of the capabilities compiled into
the patched llama.cpp library:

| Feature | Bit | Surface |
| --- | ---: | --- |
| `RUNTIME_SLICE` | `1 << 0` | Runtime layer-range execution via `RuntimeConfig` and execution calls |
| `LAYER_PACKAGE` | `1 << 1` | Layer-package load mode |
| `ARTIFACT_SLICE` | `1 << 2` | Artifact-slice load mode |
| `MODEL_INTROSPECTION` | `1 << 3` | `skippy_model_info_*` tensor metadata calls |
| `GGUF_SLICE_WRITE` | `1 << 4` | `skippy_slice_plan_*`, `skippy_write_*` |
| `TOKENIZE_DETOKENIZE` | `1 << 6` | Tokenization, detokenization, EOG checks |
| `ACTIVATION_FRAME` | `1 << 7` | Descriptor-plus-payload execution calls |
| `SESSION_RESET` | `1 << 9` | `skippy_session_reset` |
| `BATCH_VERIFY` | `1 << 10` | `skippy_verify_tokens` |
| `CHAT_TEMPLATE` | `1 << 11` | `skippy_apply_chat_template` |
| `SAMPLING_CONFIG` | `1 << 12` | Sampled decode calls using `SamplingConfig` |
| `BATCH_VERIFY_FRAME` | `1 << 13` | `skippy_verify_tokens_frame` |
| `LOGIT_BIAS` | `1 << 15` | `SamplingConfig.logit_bias` |
| `SESSION_TRIM` | `1 << 16` | `skippy_trim_session` |
| `SESSION_CHECKPOINT` | `1 << 17` | Native checkpoint/restore calls |
| `PACKAGE_PART_LOAD` | `1 << 18` | Ordered GGUF package part loading |
| `GENERATION_SIGNALS` | `1 << 19` | Generation progress and cancellation signal hooks |
| `EXTERNAL_MEDIA_PREFILL` | `1 << 20` | Multimodal prefill from externally materialized media chunks |
| `CHAT_TEMPLATE_TOOLS` | `1 << 21` | llama.cpp OpenAI-compatible chat templating and tool-call response parsing |
| `CHAT_SAMPLING_GRAMMAR` | `1 << 22` | Session-local llama.cpp grammar-constrained sampling from chat template metadata |

## Function Surface

This section maps every C function in the stage ABI, plus the upstream llama log
hook currently bound by this crate.

### Upstream llama hook

| Function | Purpose |
| --- | --- |
| `llama_log_set` | Installs a llama.cpp log callback and opaque user pointer. This comes from upstream `llama.h`, not `skippy.h`. |

### ABI discovery and errors

| Function | Purpose |
| --- | --- |
| `skippy_abi_version` | Returns the compiled stage ABI version. The C header exports it; this Rust crate currently mirrors the version through constants instead of binding the function. |
| `skippy_abi_features` | Returns the compiled feature bitmask. The C header exports it; this Rust crate does not currently bind it. |
| `skippy_status_string` | Converts a status enum to a static C string. |
| `skippy_error_free` | Frees an allocated `skippy_error`. |

### Model and session lifecycle

| Function | Purpose |
| --- | --- |
| `skippy_model_open` | Opens a GGUF model, runtime slice, layer package, or artifact slice using `RuntimeConfig`, including optional selected backend device placement. |
| `skippy_model_free` | Releases a model handle. |
| `skippy_session_create` | Creates a decode session/context from an opened model. |
| `skippy_session_reset` | Clears session state so the session can be reused. |
| `skippy_checkpoint_session` | Records a native session checkpoint and returns the checkpoint token count. |
| `skippy_restore_session_checkpoint` | Restores the native checkpoint when the requested token count matches. |
| `skippy_session_configure_chat_sampling` | Configures session-local sampling with llama.cpp chat metadata so tool-call grammars constrain generated tokens. |
| `skippy_session_free` | Releases a session handle. |
| `skippy_trim_session` | Trims session state to a token count. |

### Execution

| Function | Purpose |
| --- | --- |
| `skippy_prefill_chunk` | Prefills a token chunk using raw activation buffers for staged input/output. |
| `skippy_decode_step` | Decodes one token using raw activation buffers and optionally returns a predicted token. |
| `skippy_verify_tokens` | Runs batched token verification and returns the model-selected tokens. |
| `skippy_decode_step_sampled` | Decodes one token with `SamplingConfig`, including penalties and logit bias. |
| `skippy_prefill_chunk_frame` | Prefills a token chunk using `ActivationDesc` plus payload buffers. |
| `skippy_decode_step_frame` | Decodes one token using activation-frame descriptors and payloads. |
| `skippy_verify_tokens_frame` | Runs batched verification with activation-frame descriptors and payloads. |
| `skippy_decode_step_frame_sampled` | Decodes one token with activation-frame I/O and `SamplingConfig`. |

### Token and chat helpers

| Function | Purpose |
| --- | --- |
| `skippy_tokenize` | Tokenizes text with optional special-token insertion. |
| `skippy_detokenize` | Converts token IDs back to text bytes. |
| `skippy_token_is_eog` | Reports whether a token is an end-of-generation token. |
| `skippy_apply_chat_template` | Applies the model chat template, with optional assistant prompt and thinking-mode override. |
| `skippy_apply_chat_template_json` | Applies llama.cpp's OpenAI-compatible chat template path from JSON messages, tools, and tool-choice metadata, returning the prompt plus parser metadata. |
| `skippy_parse_chat_response_json` | Parses generated assistant text with llama.cpp's chat parser and returns an OpenAI-compatible assistant message JSON object, including tool calls when emitted by the model. |

### Model introspection and GGUF writing

| Function | Purpose |
| --- | --- |
| `skippy_model_info_open` | Opens model metadata without creating an execution session. |
| `skippy_model_info_free` | Releases model metadata. |
| `skippy_model_info_tensor_count` | Returns the number of tensors visible through model metadata. |
| `skippy_model_info_tensor_at` | Returns tensor metadata for one index. |
| `skippy_slice_plan_create` | Creates a GGUF slicing plan from model metadata. |
| `skippy_slice_plan_free` | Releases a slicing plan. |
| `skippy_slice_plan_add_layer_range` | Adds one stage layer range and embedding/output ownership flags to a plan. |
| `skippy_write_slice_gguf` | Writes one planned stage slice as a GGUF artifact. |
| `skippy_write_gguf_from_parts` | Composes multiple GGUF parts into one materialized package. |
