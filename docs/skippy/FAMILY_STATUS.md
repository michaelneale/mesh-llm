# Model Family Support Matrix

This is the current customer-facing support contract for stage-split serving.
When we say a family is supported, use the settings in this file unless a new
certification run updates them.

Certification process lives in `docs/FAMILY_CERTIFY.md`. Payload measurements
and topology constraints are summarized here so this file stays the only
customer-facing source of truth.

Last updated: 2026-05-06.

## Customer Support Matrix

| Family | Support Level | Recommended Artifact | Stage Plan | Wire | Speculative | Topology Constraint | Other Required Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 dense | Supported | `Qwen/Qwen3-0.6B:Q8_0` | `layer_end=28`, `splits=9,18`, activation width `1024` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; this is the state-size baseline. |
| Qwen2 | Supported | `meshllm/qwen2.5-0.5b-instruct-parity-q8_0-gguf:Q8_0` | `layer_end=24`, `splits=8,16`, activation width `896` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. `ResidentKv` cache smoke passed. |
| Llama | Supported | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `layer_end=16`, `splits=5,10`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| DeepSeek2 | Supported | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M` | `layer_end=27`, `splits=7,14`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| DeepSeek LLM | Supported | `Morgen0052/deepseek-llm-7b-chat-Q4_K_M-GGUF:Q4_K_M` | `layer_end=30`, `splits=10,20`, activation width `4096` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. `ResidentKv` cache smoke passed. |
| DeepSeek3 | Supported for package-backed stages | `unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL` via `meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers` | `layer_end=61`, activation width `7168`; materialize only the owned stage range | `f16`; q8 untested | `baseline,ngram,ngram-adaptive` | None | Use layer-package materialization and `ResidentKv`; do not require the full 406.8 GB layer set to be resident or merged. |
| GLM-4.7 Flash | Supported | `unsloth/GLM-4.7-Flash-GGUF:Q4_K_M` | `layer_end=47`, `splits=15,31`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | GGUF uses the DeepSeek2/MLA runtime path. |
| GLM4 9B | Supported | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `layer_end=40`, `splits=13,27`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma4 A4B | Supported | `batiai/Gemma-4-26B-A4B-it-GGUF:Q6_K` | `layer_end=30`, `splits=8,15`, activation width `2816` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma4 E4B | Supported | `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` | `layer_end=42`, `split=21`, activation width `2560` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | Use boundary `21`; do not cut at `12`, `14`, `24`, or `28`. | Token-id sideband required. |
| Gemma3 | Supported | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `layer_end=26`, `splits=9,18`, activation width `1152` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma2 | Supported | `bartowski/gemma-2-2b-it-GGUF:Q4_K_M` | `layer_end=26`, `splits=9,18`, activation width `2304` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Phi2 | Supported | `TheBloke/phi-2-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `2560` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Use `ResidentKv`; full-state mobility is rejected as too large. |
| Falcon-H1 | Supported | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `layer_end=24`, `splits=8,16`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..24` sticky for normal decode. | Use `KvRecurrent` for exact prefix cache restore; native sequence remap cache smoke passed. |
| OLMo | Supported | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| MiniMax M2.7 | Supported; neural draft pending | `unsloth/MiniMax-M2.7-GGUF:UD-Q2_K_XL` | `layer_end=62`, `splits=20,41`, activation width `3072` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Sharded GGUF supported. Materialize stage artifacts first; tokenizer uses `stage-0.gguf` CPU-only during staged prompt/spec. |
| Qwen3Next | Supported | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `layer_end=48`, `splits=16,32`, activation width `2048` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..48` sticky for normal decode until exact recurrent layer metadata is available. | Use `KvRecurrent` for exact prefix cache restore; native sequence remap cache smoke passed. |

## Text-Split Candidates

These families now pass the cheap runtime-slice text lane, but are not promoted
to the customer support matrix until the remaining cache smoke, reviewed
topology records, and family-specific policy notes are updated.

```text
Baichuan, Bloom, Cohere2, Command-R, EXAONE, EXAONE4, Falcon, Gemma text,
GPT-NeoX, GPT2, Granite, InternLM2, Jamba, LFM2, Mamba, Mamba2, Mistral3, MPT,
OLMo2, OLMoE, Phi3, Qwen2-MoE, Qwen3-MoE, Qwen2-VL text, Qwen3-VL text,
RWKV6, RWKV7, StableLM, StarCoder2
```

## Exceptions

| Family | What To Watch |
| --- | --- |
| Falcon-H1 | Recurrent state is too large to move. Keep recurrent range `0..24` sticky and transfer activation frames only. |
| Gemma text | The sampled `gemma` architecture artifact only passed stage parity with `f32` activation wire. `f16` and `q8` changed the next-token argmax. |
| Gemma3n | Local text split is still blocked by llama.cpp reporting `runtime-slice execution is not supported for this model architecture yet`. Do not promote multimodal until Gemma3n graph filtering and media/projector handling have dedicated evidence. |
| Jamba | Hybrid attention/SSM text lane and `KvRecurrent` cache smoke passed. Middle-stage cache records can have zero native KV bytes plus recurrent state; keep ownership sticky for normal decode. |
| LFM2 | Recurrent text lane and `KvRecurrent` cache smoke passed. Keep ownership sticky for normal decode. |
| Mamba | Recurrent text lane and `KvRecurrent` cache smoke passed with zero native KV bytes. Keep ownership sticky for normal decode. |
| Mamba2 | Recurrent text lane and `KvRecurrent` cache smoke passed with zero native KV bytes. Keep ownership sticky for normal decode. |
| OLMoE | Text lane, serving cache smoke, and MoE expert-stage smoke passed for one-stage, split-middle, and split-final ranges. `ResidentKv` is the selected cache policy. |
| Qwen2-MoE | Text lane, serving cache smoke, and MoE expert-stage smoke passed for one-stage, split-middle, and split-final ranges. `ResidentKv` is the selected cache policy. |
| Qwen3-MoE | Text lane, q8 activation wire, serving cache smoke, and MoE expert-stage smoke passed for one-stage, split-middle, and split-final ranges. `ResidentKv` is the selected cache policy. |
| Qwen2-VL | Text split and full-model local multimodal OpenAI smoke passed with real mmproj/image fixtures. Do not promote split multimodal yet: filtered stage-0 media prefill currently SIGSEGVs before activation forwarding. |
| Qwen3-VL | Text split and full-model local multimodal OpenAI smoke passed with real mmproj/image fixtures. Do not promote split multimodal yet: filtered stage-0 media prefill currently SIGSEGVs before activation forwarding. |
| RWKV6 | Recurrent text lane and `KvRecurrent` cache smoke passed with zero native KV bytes. Keep ownership sticky for normal decode. |
| RWKV7 | Text lane passed after adding the layer-0 `v_first` activation sideband, and `KvRecurrent` cache smoke passed with zero native KV bytes. Payloads are hidden state plus `v_first`, so budget RWKV7 activation handoffs at 2x hidden width and keep recurrent ownership sticky for normal decode. |
| Qwen3Next | Same normal-decode policy as Falcon-H1 for now: keep recurrent range `0..48` sticky until exact recurrent layer metadata exists. Exact prefix cache restore uses `KvRecurrent`. |
| DeepSeek3 | Package evidence uses selected stage parts only. The local gate covered real-input `0..1`, real-upstream expert layer `3..4`, and synthetic-upstream late layers `30..31` and `60..61`; full llama-server baseline requires a full GGUF and is intentionally not part of this package-only gate. |
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
Qwen3-MoE
RWKV7 sampled artifact
```

All other supported families should ship with `f16` activation wire until q8 is
recertified for the exact model artifact and split.

## Evidence Snapshot

Qwen3 dense is the state-size baseline. f16 payloads below are one-token
activation handoff sizes for the recommended split.

| Family | f16 Activation Payload | Exact State Mobility |
| --- | ---: | --- |
| Qwen3 dense | 2,048 | Accepted, 115,388 bytes baseline |
| Qwen2 | 1,792 | Accepted, 0.11x Qwen; `ResidentKv` 64-token smoke passed |
| Llama | 4,096 | Accepted, 0.29x Qwen |
| DeepSeek2 | 4,096 | Accepted, 2.40x Qwen |
| DeepSeek LLM | 8,192 | Accepted; `ResidentKv` 64-token smoke passed, 1.58x cache-hit speedup |
| DeepSeek3 | 14,336 | Accepted for package-backed `ResidentKv`; full-GGUF llama-server baseline not required |
| GLM-4.7 Flash | 4,096 | Accepted, 0.47x Qwen |
| GLM4 9B | 8,192 | Accepted, 0.36x Qwen |
| Gemma4 A4B | 5,632 | Accepted, 1.96x Qwen |
| Gemma4 E4B | 5,120 | Accepted, 0.50x Qwen |
| Gemma3 | 2,304 | Accepted, 0.24x Qwen |
| Gemma2 | 4,608 | Accepted, 0.93x Qwen |
| Phi2 | 5,120 | Full-state rejected as too large; `ResidentKv` cache restore accepted |
| Falcon-H1 | 4,096 | Accepted for `KvRecurrent` cache restore, 663.5x Qwen recurrent state |
| Qwen2-MoE | 3,072 | Accepted, 0.25x Qwen |
| Qwen3-MoE | 2,048 | Accepted, 1.00x Qwen |
| RWKV6 | 4,096 | Accepted for `KvRecurrent` cache restore, 112.5x Qwen recurrent state |
| OLMo | 8,192 | Accepted, 4.55x Qwen |
| MiniMax M2.7 | 6,144 | Accepted, 2.21x Qwen |
| Qwen3Next | 4,096 | Accepted for `KvRecurrent` cache restore, 685.2x Qwen recurrent state |

## Neural Draft Status

No neural draft model is currently certified for MiniMax M2.7. N-gram
speculative decoding is certified. Candidate drafts should be added here only
after:

1. The draft artifact fits beside the staged target memory budget.
2. Tokenizer compatibility is proven.
3. `spec-bench` target/draft correctness passes.
4. Staged prompt/spec `draft-fixed` and `draft-adaptive` lanes pass.
