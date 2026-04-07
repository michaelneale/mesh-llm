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
- explicit MLX runtime selection via `--mlx` or `--mlx-file`
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

Current product behavior:
- MLX is an explicit opt-in backend when launching with `--model`
- using MLX prints an experimental startup warning
- the warning explicitly points users at the GitHub issues page if they hit problems

What the existing llama.cpp-backed `mesh-llm` path already supports:
- vision models via `mmproj`

What the existing `mesh-llm` product surface does not currently expose as a first-class llama feature:
- audio runtime support

## Supported Runtime Families

These families are now in the supported native MLX runtime set:
- Llama
- GLM 4 dense
- Qwen2
- Qwen3
- Gemma 2 text
- Gemma 3 text
- Gemma 4 text
- GLM4 text
- LFM2 text
- DeepSeekV3 / Kimi-K2 text
- gpt-oss text
- Kimi Linear text

Notes:
- Gemma 4 support currently targets text-capable MLX repos such as `unsloth/gemma-4-E4B-it-UD-MLX-4bit`
- MLX remains a local-only serving path today
- MLX support is currently text-only, even though the llama-backed path already supports vision models
- DeepSeekV3 / Kimi-K2, `gpt-oss`, and `Kimi Linear` are correctness-first runtime additions today; they are compile/test verified but not part of the live macOS smoke matrix yet
- MLX is intentionally not auto-selected from `--model`; callers must opt in with `--mlx`

## Families Still Missing Runtime Support

These families still need more validation, broader target coverage, or a dedicated product pass:
- DeepSeekV3 / Kimi-K2
- Kimi Linear
- gpt-oss

These are no longer template-only gaps, but they are not as battle-tested as the smaller live-smoked families yet.

## Remaining Family Work

### Gemma 4

Gemma 4 text now works, but it is not fully “done”.

Remaining work:
- verify more Gemma 4 repos beyond the current `unsloth` target
- support broader Gemma 4 variants if they differ from the current text-side structure
- harden around layer-type-specific attention behavior if new repos expose gaps
- add more Gemma 4 catalog coverage once confidence is higher

### Kimi / Kimi Linear / gpt-oss / LFM2

These should not be treated as one bucket anymore.

#### DeepSeekV3 / Kimi-K2

Current understanding:
- `Kimi-K2` / `K2.5` ride on a DeepSeekV3-style MLA + MoE runtime base
- that base is now implemented in the MLX runtime
- live smoke coverage is still missing because the public MLX repos are very large

Remaining work:
- add a real known-good K2/K2.5 runtime validation pass once practical hardware/CI coverage exists
- catalog only when we are comfortable with real runtime validation, not just compile-time support

#### Kimi Linear

Current understanding:
- `kimi_linear` is a separate architecture from K2/K2.5
- it uses its own linear-attention stack plus MoE and custom projection structure
- it now has a dedicated cacheless runtime path in the MLX backend

Remaining work:
- real public-model validation against a known-good target repo
- focused runtime smoke tests
- cached generation / recurrent state support beyond the correctness-first cacheless path

#### gpt-oss

Current understanding:
- the realistic public target is `mlx-community/gpt-oss-20b-MXFP4-Q4`
- its runtime is now implemented via a correctness-first cacheless path
- it is still not part of the live smoke matrix, and the current support should be treated as earlier-stage than Llama/Qwen/Gemma/GLM/LFM2

Remaining work:
- real public-model validation against a known-good target repo
- focused runtime smoke tests
- decide whether to keep the current path or later add lower-level MXFP4 support via `mlx-rs` for better performance

#### LFM2

Current understanding:
- the best first target is `mlx-community/LFM2-350M-4bit`
- `LFM2` is more tractable than `gpt-oss` because the public MLX target uses plain affine quantization
- the family alternates standard attention blocks with `ShortConv` blocks

Current status:
- `lfm2` config support is implemented
- `ShortConv` runtime is implemented
- `mlx-community/LFM2-350M-4bit` passes a live local MLX smoke
- MLX generation now uses streaming-safe token decoding, which fixed non-ASCII output for this family (`🔴` instead of replacement characters)

Remaining work:
- add broader LFM2 coverage beyond the 350M target
- add GGUF-side parity where practical if we want matrix symmetry
- revisit cached generation for LFM2 after the correctness-first cacheless path

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
- real-model validation and smoke coverage for DeepSeekV3 / Kimi-K2
- real-model validation and smoke coverage for Kimi Linear
- real-model validation and smoke coverage for gpt-oss
- broader Gemma 4 validation and hardening
- MLX vision runtime support for families where `mesh-llm` already supports vision on the llama path
- distributed MLX serving

## Possible `mlx-rs` Follow-up

We should stay willing to fork `mlx-rs` if the backend needs capabilities that are not practical to layer externally.

Reasons a fork might become necessary:
- missing kernels for quantized paths we need
- attention-mask behavior the current API cannot express cleanly
- backend/device movement limitations that block correct runtime behavior
- quantization modes such as MXFP4 that the current bindings/runtime path do not expose cleanly

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

1. Land and stabilize DeepSeekV3 / Kimi-K2 coverage beyond compile-time validation.
2. Narrow issue `#142` to the remaining Gemma-family work now that Gemma 2, Gemma 3, and Gemma 4 text support exist.
3. Start scoping MLX vision support, since vision is already supported on the llama-backed path.
4. Decide whether the next major priority is:
   - broader family coverage, or
   - distributed MLX serving from `#146`

If family coverage is the priority, the next order should be:
- broader DeepSeekV3 / Kimi-K2 validation
- Kimi Linear validation
- broader gpt-oss validation
