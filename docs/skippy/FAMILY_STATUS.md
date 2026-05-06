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
| DeepSeek LLM | Supported | `Morgen0052/deepseek-llm-7b-chat-Q4_K_M-GGUF:Q4_K_M` | `layer_end=30`, `splits=10,20`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. `ResidentKv` cache smoke passed. |
| DeepSeek3 | Supported for package-backed stages | `unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL` via `meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers` | `layer_end=61`, activation width `7168`; materialize only the owned stage range | `f16`; q8 untested | `baseline,ngram,ngram-adaptive` | None | Use layer-package materialization and `ResidentKv`; do not require the full 406.8 GB layer set to be resident or merged. |
| GLM-4.7 Flash | Supported | `unsloth/GLM-4.7-Flash-GGUF:Q4_K_M` | `layer_end=47`, `splits=15,31`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | GGUF uses the DeepSeek2/MLA runtime path. |
| GLM4-MoE | Supported | `noctrex/GLM-4.7-Flash-REAP-23B-A3B-MXFP4_MOE-GGUF:MXFP4_MOE` | `layer_end=47`, `splits=15,31`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| GLM4 9B | Supported | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `layer_end=40`, `splits=13,27`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Baichuan | Supported | `lucasxx/Baichuan2-7B-Chat-Q4_K_M-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| Bloom | Supported | `QuantFactory/bloomz-560m-GGUF:Q4_K_M` | `layer_end=24`, `splits=8,16`, activation width `1024` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| GPT2 | Supported | `QuantFactory/gpt2-GGUF:Q4_K_M` | `layer_end=12`, `splits=4,8`, activation width `768` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| GPT-NeoX | Supported | `warriorknight3/pythia-70m-Q4_K_M-GGUF:Q4_K_M` | `layer_end=6`, `splits=2,4`, activation width `512` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| Gemma4 A4B | Supported | `batiai/Gemma-4-26B-A4B-it-GGUF:Q6_K` | `layer_end=30`, `splits=8,15`, activation width `2816` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma4 E4B | Supported | `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` | `layer_end=42`, `split=21`, activation width `2560` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | Use boundary `21`; do not cut at `12`, `14`, `24`, or `28`. | Token-id sideband required. |
| Gemma3 | Supported | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `layer_end=26`, `splits=9,18`, activation width `1152` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Gemma2 | Supported | `bartowski/gemma-2-2b-it-GGUF:Q4_K_M` | `layer_end=26`, `splits=9,18`, activation width `2304` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| Phi2 | Supported | `TheBloke/phi-2-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `2560` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Use `ResidentKv`; full-state mobility is rejected as too large. |
| Granite | Supported | `bartowski/ibm-granite_granite-3.2-2b-instruct-GGUF:Q4_K_M` | `layer_end=40`, `splits=13,26`, activation width `2048` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| Granite-Hybrid | Supported | `magiccodingman/Granite-4.0-H-350M-Unsloth-MXFP4-Hybrid-GGUF:MXFP4_MOE-output_q6_K-router_gate_emb_q6_K` | `layer_end=32`, `splits=10,21`, activation width `768` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..32` sticky for normal decode. | Use `KvRecurrent`; full-state mobility is rejected as too large. |
| Granite-MoE | Supported for layout parity | `mradermacher/tiny-random-granite-moe-GGUF:Q4_K_M` | `layer_end=6`, `splits=2,4`, activation width `64` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Tiny random GGUF certifies graph/tensor layout and cache mechanics; replace with a real small artifact when available. |
| Hunyuan-Dense | Supported | `Edge-Quant/Hunyuan-1.8B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` cache smoke passed. |
| Hunyuan-MoE | Supported | `unsloth/Hunyuan-A13B-Instruct-GGUF:UD-IQ2_XXS` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Real A13B MoE GGUF passed runtime-slice and `ResidentKv` native sequence remap cache smoke. |
| LFM2 | Supported | `meshllm/lfm2-350m-parity-q4_k_m-gguf:q4_k_m` | `layer_end=16`, `splits=5,10`, activation width `1024` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..16` sticky for normal decode. | Use `KvRecurrent` for exact prefix cache restore; native sequence remap cache smoke passed. |
| Jamba | Supported | `bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M` | `layer_end=28`, `splits=9,18`, activation width `2560` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..28` sticky for normal decode. | Use `KvRecurrent` for exact prefix cache restore; middle-stage recurrent-only slices are valid. |
| Mamba | Supported | `mradermacher/mamba-130m-hf-GGUF:Q4_K_M` | `layer_end=24`, `splits=8,16`, activation width `768` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..24` sticky for normal decode. | Use `KvRecurrent`; cache restore can have zero native KV bytes. |
| Mamba2 | Supported | `mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M` | `layer_end=64`, `splits=21,42`, activation width `2560` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..64` sticky for normal decode. | Use `KvRecurrent`; full-state mobility is rejected as too large. |
| RWKV6 | Supported | `latestissue/rwkv-6-finch-1b6-gguf:Q4_K` | `layer_end=24`, `splits=8,16`, activation width `2048` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..24` sticky for normal decode. | Use `KvRecurrent`; cache restore can have zero native KV bytes. |
| RWKV7 | Supported | `Mungert/rwkv7-191M-world-GGUF:q4_k` | `layer_end=12`, `splits=4,8`, activation width `768` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..12` sticky for normal decode. | Use `KvRecurrent`; downstream slices require the layer-0 `v_first` activation sideband. |
| Falcon-H1 | Supported | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `layer_end=24`, `splits=8,16`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..24` sticky for normal decode. | Use `KvRecurrent` for exact prefix cache restore; native sequence remap cache smoke passed. |
| Falcon | Supported | `Kondara/falcon-7b-instruct-Q4_K_M-GGUF:q4_k_m` | `layer_end=32`, `splits=10,21`, activation width `4544` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| InternLM2 | Supported | `lmstudio-community/internlm2_5-1_8b-chat-GGUF:Q4_K_M` | `layer_end=24`, `splits=8,16`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| Phi3 | Supported | `bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `3072` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| PhiMoE | Supported | `bartowski/Phi-3.5-MoE-instruct-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Shares the Phi3 graph path; exact state mobility accepted after adding PhiMoE to the runtime-slice ABI allowlist. |
| OLMo | Supported | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted. |
| OLMo2 | Supported | `allenai/OLMo-2-1124-7B-Instruct-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| OLMoE | Supported | `bartowski/OLMoE-1B-7B-0924-Instruct-GGUF:Q4_K_M` | `layer_end=16`, `splits=5,10`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` cache restore and MoE expert-stage smoke passed. |
| Mistral3 | Supported | `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M` | `layer_end=26`, `splits=8,17`, activation width `3072` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| Qwen2-MoE | Supported | `mradermacher/Qwen2-1.5B-2x-MoE-GGUF:Q4_K_S` | `layer_end=28`, `splits=9,18`, activation width `1536` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | `ResidentKv` cache restore and MoE expert-stage smoke passed. |
| Qwen3-MoE | Supported | `mradermacher/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-GGUF:Q4_K_M` | `layer_end=28`, `splits=9,18`, activation width `1024` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | `ResidentKv` cache restore and MoE expert-stage smoke passed. |
| EXAONE | Supported | `lmstudio-community/EXAONE-3.5-2.4B-Instruct-GGUF:Q4_K_M` | `layer_end=30`, `splits=10,20`, activation width `2560` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| EXAONE4 | Supported | `bartowski/LGAI-EXAONE_EXAONE-4.0-1.2B-GGUF:Q4_K_M` | `layer_end=30`, `splits=10,20`, activation width `2048` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| Cohere2 | Supported | `Lumia101/c4ai-command-r7b-12-2024-Q4_K_M-GGUF:q4_k_m` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | GGUF reports `cohere2`; exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| MiniMax M2.7 | Supported; neural draft pending | `unsloth/MiniMax-M2.7-GGUF:UD-Q2_K_XL` | `layer_end=62`, `splits=20,41`, activation width `3072` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Sharded GGUF supported. Materialize stage artifacts first; tokenizer uses `stage-0.gguf` CPU-only during staged prompt/spec. |
| Qwen3Next | Supported | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `layer_end=48`, `splits=16,32`, activation width `2048` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | Keep recurrent range `0..48` sticky for normal decode until exact recurrent layer metadata is available. | Use `KvRecurrent` for exact prefix cache restore; native sequence remap cache smoke passed. |
| StableLM | Supported | `TheBloke/stablelm-zephyr-3b-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `2560` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| StarCoder2 | Supported | `combos/starcoder2-3b-Q4_K_M-GGUF:q4_k_m` | `layer_end=30`, `splits=10,20`, activation width `3072` | `f16`; q8 validated | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |
| MPT | Supported | `mradermacher/mpt-7b-chat-GGUF:Q4_K_M` | `layer_end=32`, `splits=10,21`, activation width `4096` | `f16`; q8 rejected | `baseline,ngram,ngram-adaptive` | None | Exact state mobility accepted; `ResidentKv` native sequence remap cache smoke passed. |

## Text-Split Candidates

These families now pass the cheap runtime-slice text lane, but are not promoted
to the customer support matrix until the remaining cache smoke, reviewed
topology records, and family-specific policy notes are updated.

```text
Command-R, Gemma text,
Granite,
Qwen2-VL text, Qwen3-VL text
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
GLM4-MoE
Baichuan
Bloom
GPT-NeoX
OLMo2
OLMoE
StarCoder2
LFM2
Jamba
Mamba
Mamba2
RWKV7
PhiMoE
Mistral3
Hunyuan-MoE
InternLM2
EXAONE
EXAONE4
Falcon
Gemma2
Falcon-H1
Qwen3-MoE
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
| GLM4-MoE | 4,096 | Accepted; `ResidentKv` 64-token smoke passed |
| GLM4 9B | 8,192 | Accepted, 0.36x Qwen |
| Baichuan | 8,192 | Accepted; `ResidentKv` 64-token smoke passed, 130.11x cache-hit speedup |
| Bloom | 2,048 | Accepted; `ResidentKv` 64-token smoke passed, 328.66x cache-hit speedup |
| GPT2 | 1,536 | Accepted; `ResidentKv` 64-token smoke passed, 1535.17x cache-hit speedup |
| GPT-NeoX | 1,024 | Accepted; `ResidentKv` 64-token smoke passed, 282.70x cache-hit speedup |
| Hunyuan-MoE | 8,192 | Accepted; `ResidentKv` 64-token smoke passed, 465.32x cache-hit speedup |
| InternLM2 | 4,096 | Accepted; `ResidentKv` 64-token smoke passed, 80.78x cache-hit speedup |
| Gemma4 A4B | 5,632 | Accepted, 1.96x Qwen |
| Gemma4 E4B | 5,120 | Accepted, 0.50x Qwen |
| Gemma3 | 2,304 | Accepted, 0.24x Qwen |
| Gemma2 | 4,608 | Accepted, 0.93x Qwen |
| Phi2 | 5,120 | Full-state rejected as too large; `ResidentKv` cache restore accepted |
| Granite | 4,096 | Accepted; `ResidentKv` 64-token smoke passed after staged activation rescaling fix |
| Falcon-H1 | 4,096 | Accepted for `KvRecurrent` cache restore, 663.5x Qwen recurrent state |
| Phi3 | 6,144 | Accepted; `ResidentKv` 64-token smoke passed, 2645.30x cache-hit speedup |
| PhiMoE | 8,192 | Accepted; runtime-slice parity passed after PhiMoE ABI allowlist support |
| Qwen2-MoE | 3,072 | Accepted, 0.25x Qwen |
| Qwen3-MoE | 2,048 | Accepted, 1.00x Qwen |
| EXAONE | 5,120 | Accepted; `ResidentKv` 64-token smoke passed, 868.76x cache-hit speedup |
| EXAONE4 | 4,096 | Accepted; `ResidentKv` 64-token smoke passed, 220.01x cache-hit speedup |
| Cohere2 | 8,192 | Accepted; `ResidentKv` 64-token smoke passed, 63.48x cache-hit speedup |
| Falcon | 9,088 | Accepted; `ResidentKv` 64-token smoke passed, 1753.04x cache-hit speedup |
| RWKV6 | 4,096 | Accepted for `KvRecurrent` cache restore, 112.5x Qwen recurrent state |
| RWKV7 | 3,072 | Full-state mobility rejected by recurrent policy; `KvRecurrent` cache restore accepted, 21.09x Qwen recurrent state; activation handoff carries hidden state plus `v_first` |
| LFM2 | 2,048 | Full-state mobility rejected by recurrent policy; `KvRecurrent` cache restore accepted, 0.82x Qwen recurrent state |
| Jamba | 5,120 | Full-state mobility rejected by recurrent policy; `KvRecurrent` cache restore accepted, 87.69x Qwen recurrent state |
| Mamba | 1,536 | Full-state mobility rejected by recurrent policy; `KvRecurrent` cache restore accepted, 24.29x Qwen recurrent state |
| Mamba2 | 5,120 | Full-state rejected as too large; `KvRecurrent` cache restore accepted, 215.84x Qwen recurrent state |
| OLMo | 8,192 | Accepted, 4.55x Qwen |
| OLMo2 | 8,192 | Accepted; `ResidentKv` 64-token smoke passed, 159.19x cache-hit speedup |
| OLMoE | 4,096 | Accepted; `ResidentKv` 64-token smoke passed, 197.09x cache-hit speedup |
| Mistral3 | 6,144 | Accepted; `ResidentKv` 64-token smoke passed, 88.40x cache-hit speedup |
| MiniMax M2.7 | 6,144 | Accepted, 2.21x Qwen |
| Qwen3Next | 4,096 | Accepted for `KvRecurrent` cache restore, 685.2x Qwen recurrent state |
| StableLM | 5,120 | Accepted; `ResidentKv` 64-token smoke passed, 211.80x cache-hit speedup |
| StarCoder2 | 6,144 | Accepted; `ResidentKv` 64-token smoke passed, 198.10x cache-hit speedup |
| MPT | 8,192 | Accepted; `ResidentKv` 64-token smoke passed, 1657.77x cache-hit speedup |

## Neural Draft Status

No neural draft model is currently certified for MiniMax M2.7. N-gram
speculative decoding is certified. Candidate drafts should be added here only
after:

1. The draft artifact fits beside the staged target memory budget.
2. Tokenizer compatibility is proven.
3. `spec-bench` target/draft correctness passes.
4. Staged prompt/spec `draft-fixed` and `draft-adaptive` lanes pass.
