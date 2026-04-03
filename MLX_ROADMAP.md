# MLX Roadmap

This document tracks the remaining work for the native MLX backend in `mesh-llm`.

It is intentionally practical:
- what already works
- what still needs runtime support
- what remains around downloads, templates, CI, and product behavior
- which GitHub issues already track MLX follow-up work

## Current Status

The MLX backend is now a real serving path on macOS for a bounded set of text models.

Working today:
- MLX-native loading and serving on Apple Silicon
- Hugging Face repo shorthand, exact artifact refs, and catalog entries
- MLX sidecar download support:
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `chat_template.json`
  - `chat_template.jinja`
  - sharded safetensors from `model.safetensors.index.json`
- Hugging Face chat templates rendered through MiniJinja with compatibility normalization
- Family-aware thinking/reasoning controls for supported template families
- macOS MLX smoke coverage in CI

What the existing llama.cpp-backed `mesh-llm` path already supports:
- vision models via `mmproj`

What the existing `mesh-llm` product surface does not currently expose as a first-class llama feature:
- audio runtime support

## Supported Runtime Families

These families are now in the supported native MLX runtime set:
- Llama
- Qwen2
- Qwen3
- Gemma 2 text
- Gemma 3 text
- Gemma 4 text

Notes:
- Gemma 4 support currently targets text-capable MLX repos such as `unsloth/gemma-4-E4B-it-UD-MLX-4bit`
- MLX remains a local-only serving path today
- MLX support is currently text-only, even though the llama-backed path already supports vision models

## Families Still Missing Runtime Support

These families have template-level support or partial investigation, but not full runtime support yet:
- GLM
- Kimi
- gpt-oss
- LFM2

These need explicit loader/runtime work, not just template changes.

## Remaining Family Work

### Gemma 4

Gemma 4 text now works, but it is not fully “done”.

Remaining work:
- verify more Gemma 4 repos beyond the current `unsloth` target
- support broader Gemma 4 variants if they differ from the current text-side structure
- harden around layer-type-specific attention behavior if new repos expose gaps
- add more Gemma 4 catalog coverage once confidence is higher

### GLM / Kimi / gpt-oss / LFM2

Remaining work for each family:
- real loader/config support
- tensor-layout implementation
- one known-good public MLX target repo
- focused runtime smoke tests

## Prompt Template Compatibility

The current HF template path is much stronger than before, but there is still follow-up work.

Remaining work:
- keep extending the real Hugging Face template corpus as new MLX repos appear
- improve compatibility with family-specific template quirks only when real repos require it
- prefer real fixtures over speculative support
- keep fallback behavior explicit when HF templates cannot be rendered safely

Priority additions:
- more real Gemma 4 templates
- future Qwen 4 MLX templates once public repos exist
- more tool-calling and multimodal-adjacent text templates where relevant

## Vision and Audio

### Vision

Vision should be part of the MLX roadmap because the existing llama-backed runtime already supports vision models in `mesh-llm`.

Evidence in the current codebase:
- model catalog entries carry `mmproj`
- capability detection marks vision support from `mmproj` and vision metadata
- llama launch wiring passes `--mmproj`

Remaining MLX vision work:
- add a real MLX-side multimodal model-loading path
- support image token / image placeholder handling beyond template-only rendering
- add at least one live MLX vision smoke once a supported model family exists
- extend catalog rules so MLX vision models are only listed when the runtime truly supports them

### Audio

Audio should not yet be treated as a committed MLX roadmap target in the same way as vision.

Reason:
- I checked the current `mesh-llm` repo surface, and unlike vision there is no first-class audio runtime path exposed through `mesh-llm` today
- that means “match llama feature parity” clearly applies to vision now, but not yet to audio at the product layer

So the current stance should be:
- vision: yes, explicit MLX roadmap target
- audio: future possibility, but not yet a committed parity target until `mesh-llm` itself exposes it on the llama path

## Runtime Behavior and Product Gaps

### Local-only MLX serving

Current behavior:
- supported MLX models on macOS run through the local native MLX path
- they do not participate in the existing rpc/split distributed path

Remaining work:
- design distributed MLX serving behavior
- decide whether MLX split/distribution reuses existing orchestration or needs backend-specific rules
- add tests once a real design exists

Tracked issue:
- [#146](https://github.com/michaelneale/mesh-llm/issues/146) Support distributed or split MLX serving

### Download and resolution UX

Current state is much better, but there is still polish left:
- keep catalog entries expanding for supported families
- maintain backend-aware ambiguity handling for `--model org/repo`
- keep `--mlx`, `--gguf`, `--mlx-file`, and `--gguf-file` behavior sharp and documented

## CI and Smoke Testing

The macOS MLX smoke matrix exists now. It should keep expanding only where runtime support is real.

Remaining work:
- stabilize the sequential smoke experience across the supported matrix
- debug and eliminate startup/load flakes such as the intermittent `JOSIE-IT1-Qwen3-0.6B-4bit` startup hang
- keep prompts family-aware so the smoke tests validate useful behavior without becoming brittle
- avoid adding live smokes for unsupported runtime families

## Catalog Work

Current direction:
- explicit `-MLX` catalog names for MLX entries
- only catalog models that the runtime can actually serve

Remaining work:
- keep adding supported MLX entries for Llama, Qwen, Gemma
- add MLX vision catalog entries only after MLX vision runtime support is real
- do not add unsupported families just because templates render
- revisit broader Gemma-family catalog breadth now that Gemma 2/3/4 text are working

## MLX Runtime Engineering Tasks

These are the main technical tasks still on the table:
- family-specific loader/runtime support for GLM
- family-specific loader/runtime support for Kimi
- family-specific loader/runtime support for gpt-oss
- family-specific loader/runtime support for LFM2
- broader Gemma 4 validation and hardening
- MLX vision runtime support for families where `mesh-llm` already supports vision on the llama path
- distributed MLX serving

## Possible `mlx-rs` Follow-up

We should stay willing to fork `mlx-rs` if the backend needs capabilities that are not practical to layer externally.

Reasons a fork might become necessary:
- missing kernels for quantized paths we need
- attention-mask behavior the current API cannot express cleanly
- backend/device movement limitations that block correct runtime behavior

This should remain a last resort, but it is explicitly on the table.

## Existing MLX Issues

Issues already raised for MLX follow-up:

- [#142](https://github.com/michaelneale/mesh-llm/issues/142) Support Gemma MLX models in native MLX runtime
  - Originally opened for Gemma-family runtime support
  - Now partially addressed by Gemma 2, Gemma 3, and Gemma 4 text support
  - Still relevant for broader Gemma 4 coverage and future Gemma-family expansion

- [#146](https://github.com/michaelneale/mesh-llm/issues/146) Support distributed or split MLX serving
  - Tracks the current local-only limitation

## Suggested Next Steps

Recommended order:

1. Land and stabilize Gemma 2 and Gemma 4 text support in CI.
2. Fix the current MLX smoke flake around JOSIE sequential startup.
3. Narrow issue `#142` to the remaining Gemma-family work now that Gemma 2, Gemma 3, and Gemma 4 text support exist.
4. Start scoping MLX vision support, since vision is already supported on the llama-backed path.
5. Decide whether the next major priority is:
   - broader family coverage, or
   - distributed MLX serving from `#146`

If family coverage is the priority after Gemma 2, the next order should be:
- GLM
- Kimi
- gpt-oss
- LFM2
