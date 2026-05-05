# Model Family Support Matrix

This is the current customer-facing support contract for stage-split serving.
When we say a family is supported, use the settings in this file unless a new
certification run updates them.

Certification process lives in `docs/FAMILY_CERTIFY.md`. Payload measurements
and topology constraints are summarized here so this file stays the only
customer-facing source of truth.

Last updated: 2026-04-28.

## Customer Support Matrix

| Family | Support Level | Recommended Artifact | Stage Plan | Wire | Speculative | Topology Constraint | Other Required Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 dense | Supported | `Qwen/Qwen3-0.6B:Q8_0` | `layer_end=28`, `splits=9,18`, activation width `1024` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; this is the state-size baseline. |
| Llama | Supported | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `layer_end=16`, `splits=5,10`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| DeepSeek2 | Supported | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M` | `layer_end=27`, `splits=7,14`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| DeepSeek3 | Supported for package-backed stages | `unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL` via `meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers` | `layer_end=61`, activation width `7168`; materialize only the owned stage range | `f16`; q8 untested | `baseline,ngram,ngram-adaptive` | None | Use layer-package materialization; do not require the full 406.8 GB layer set to be resident or merged. |
| GLM-4.7 Flash | Supported | `unsloth/GLM-4.7-Flash-GGUF:Q4_K_M` | `layer_end=47`, `splits=15,31`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | GGUF uses the DeepSeek2/MLA runtime path. |
| GLM4 9B | Supported | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `layer_end=40`, `splits=13,27`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma4 A4B | Supported | `batiai/Gemma-4-26B-A4B-it-GGUF:Q6_K` | `layer_end=30`, `splits=8,15`, activation width `2816` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma4 E4B | Supported | `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` | `layer_end=42`, `split=21`, activation width `2560` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | Use boundary `21`; do not cut at `12`, `14`, `24`, or `28`. | Token-id sideband required. |
| Gemma3 | Supported | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `layer_end=26`, `splits=9,18`, activation width `1152` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma2 | Supported | `bartowski/gemma-2-2b-it-GGUF:Q4_K_M` | `layer_end=26`, `splits=9,18`, activation width `2304` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Falcon-H1 | Supported | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `layer_end=24`, `splits=8,16`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..24` sticky. | Do not transfer recurrent state during normal decode. Exact state mobility rejected. |
| OLMo | Supported | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| MiniMax M2.7 | Supported; neural draft pending | `unsloth/MiniMax-M2.7-GGUF:UD-Q2_K_XL` | `layer_end=62`, `splits=20,41`, activation width `3072` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Sharded GGUF supported. Materialize stage artifacts first; tokenizer uses `stage-0.gguf` CPU-only during staged prompt/spec. |
| Qwen3Next | Supported | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `layer_end=48`, `splits=16,32`, activation width `2048` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..48` sticky until exact recurrent layer metadata is available. | Do not transfer recurrent state during normal decode. Exact state mobility rejected. |

## Exceptions

| Family | What To Watch |
| --- | --- |
| Falcon-H1 | Recurrent state is too large to move. Keep recurrent range `0..24` sticky and transfer activation frames only. |
| Qwen3Next | Same policy as Falcon-H1 for now: keep recurrent range `0..48` sticky until exact recurrent layer metadata exists. |
| DeepSeek3 | Package evidence uses selected stage parts only. The local gate covered `0..1` and expert layer `3..4`; full llama-server baseline requires a full GGUF and is intentionally not part of this package-only gate. |
| Gemma4 E4B | Use split `21`. Avoid `12`, `14`, `24`, and `28`. Downstream slices need token-id sideband. |
| MiniMax M2.7 | Sharded GGUF is supported. Materialize stage artifacts first; prompt tokenizer uses `stage-0.gguf` CPU-only. Do not keep full-model and staged-server residency alive together unless memory has been budgeted. |

## q8 Activation Wire

q8 is currently validated for:

```text
Llama
DeepSeek2
GLM-4.7 Flash
Gemma2
Falcon-H1
```

All other supported families should ship with `f16` activation wire until q8 is
recertified for the exact model artifact and split.

## Evidence Snapshot

Qwen3 dense is the state-size baseline. f16 payloads below are one-token
activation handoff sizes for the recommended split.

| Family | f16 Activation Payload | Exact State Mobility |
| --- | ---: | --- |
| Qwen3 dense | 2,048 | Accepted, 115,388 bytes baseline |
| Llama | 4,096 | Accepted, 0.29x Qwen |
| DeepSeek2 | 4,096 | Accepted, 2.40x Qwen |
| DeepSeek3 | 14,336 | Accepted for package-backed resident KV; full-GGUF baseline not required |
| GLM-4.7 Flash | 4,096 | Accepted, 0.47x Qwen |
| GLM4 9B | 8,192 | Accepted, 0.36x Qwen |
| Gemma4 A4B | 5,632 | Accepted, 1.96x Qwen |
| Gemma4 E4B | 5,120 | Accepted, 0.50x Qwen |
| Gemma3 | 2,304 | Accepted, 0.24x Qwen |
| Gemma2 | 4,608 | Accepted, 0.93x Qwen |
| Falcon-H1 | 4,096 | Rejected, 663.5x Qwen recurrent state |
| OLMo | 8,192 | Accepted, 4.55x Qwen |
| MiniMax M2.7 | 6,144 | Accepted, 2.21x Qwen |
| Qwen3Next | 4,096 | Rejected, 685.2x Qwen recurrent state |

## Neural Draft Status

No neural draft model is currently certified for MiniMax M2.7. N-gram
speculative decoding is certified. Candidate drafts should be added here only
after:

1. The draft artifact fits beside the staged target memory budget.
2. Tokenizer compatibility is proven.
3. `spec-bench` target/draft correctness passes.
4. Staged prompt/spec `draft-fixed` and `draft-adaptive` lanes pass.
