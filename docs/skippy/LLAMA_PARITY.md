# llama.cpp Family Parity

This is the cheap certification lane for reaching practical parity with the
pinned llama.cpp model-family surface.

The goal is one small GGUF representative per llama.cpp `src/models/*.cpp`
implementation. Passing this lane does not mean every model size and quant for
that family is certified. It means the family has at least one reviewed artifact
that proves the stage ABI, activation handoff, topology policy, and cache policy
work together.

## Workflow

1. Join the pinned llama.cpp inventory with the candidate manifest:

   ```bash
   scripts/skippy-llama-parity.py inventory
   ```

2. See which cheap representatives are not downloaded yet:

   ```bash
   scripts/skippy-llama-parity.py download-commands
   ```

   To use an external disk, set `HF_HOME` or `HF_HUB_CACHE` before downloading
   and before running this script.

3. Run local candidates through the existing certification harness:

   ```bash
   scripts/skippy-llama-parity.py run --limit 1
   ```

4. Promote passing rows into:

   - `docs/skippy/FAMILY_STATUS.md`
   - `crates/skippy-topology/capabilities/reviewed-family-capabilities.json`

Raw run artifacts stay under `target/family-certify/...`.

## Tracking Plan

We track each llama.cpp family through separate gates. A family is not promoted
just because one early gate passed.

| Gate | What It Proves | Required For Promotion |
| --- | --- | --- |
| Candidate selected | We have one cheap representative GGUF for the llama.cpp model implementation. | Yes |
| Downloaded/local | The representative is available in the local Hugging Face cache. | Yes |
| Inspect | GGUF metadata yields architecture, layer count, activation width, and split plan. | Yes |
| Text split | `single-step`, `chain`, and `dtype-matrix` pass with default `f16` wire. | Yes for text serving |
| Exact state | `state-handoff` accepts, or the family has a documented reason to reject exact state mobility. | Yes |
| Cache policy | The family is assigned `ResidentKv`, `KvRecurrent`, package-only `ResidentKv`, or cache disabled. | Yes |
| Cache smoke | Serving-path exact-prefix cache hits and misses behave correctly for that family/policy. | Yes for cache-on-by-default |
| Multimodal | Projector/media sidebands pass, where applicable. | Yes for multimodal serving |
| Promotion | `FAMILY_STATUS.md` and reviewed topology records are updated. | Final |

## Cache Scope

Yes, cache support is part of this parity program, but it is a separate gate
from stage-split correctness.

The rule is:

| Family Shape | Default Cache Target | Promotion Rule |
| --- | --- | --- |
| Dense causal decoder | `ResidentKv` | Promote after text split, exact-state decision, and serving cache smoke pass. |
| Recurrent/hybrid causal decoder | `KvRecurrent` | Promote only after recurrent ranges or sticky ownership are documented and cache smoke proves KV plus recurrent state is exact. |
| Package-backed huge model | package-local `ResidentKv` | Promote against materialized stage artifacts; do not require monolithic full-GGUF llama-server baseline. |
| Multimodal decoder | family-specific, usually `ResidentKv` plus sidebands | Text cache smoke is not enough; projector/media-token path must pass. |
| Encoder/embedding/non-causal | disabled or separate future lane | Do not promote as causal stage-split cache support. |
| Unknown/uncertain state layout | disabled | No implicit cache reuse. |

`FullState` remains a certification/debugging payload. It is not the production
cache target for parity. Production evidence should use `ResidentKv` or
`KvRecurrent`.

## Certification Levels

| Level | Meaning |
| --- | --- |
| `candidate` | Cheap GGUF exists or is proposed; run full family certification before promotion. |
| `candidate_stateful` | Recurrent/hybrid family; default to sticky recurrent ownership. |
| `candidate_multimodal` | Text lane can be checked cheaply; projector/image/audio lanes must pass before multimodal promotion. |
| `certified` | Already in the reviewed support matrix. Re-running is allowed when llama.cpp or the ABI changes. |
| `certified_package_only` | Full source model is too large; package/stage evidence is the support contract. |
| `non_causal_aux` | Encoder, embedding, decoder-only audio, or other auxiliary path; do not certify as causal stage-split serving. |
| `needs_candidate` | No cheap representative has been selected yet. |

## Policy

Default activation wire remains `f16`. `q8` is opt-in per family only when the
dtype matrix proves exactness for that representative split.

For recurrent and hybrid families, the cheap lane should use `--recurrent-all`
until exact recurrent layer ranges are known. Activations may cross topology
boundaries; recurrent state defines sticky topology affinity.

For multimodal families, a text-only pass is not enough. The row can move to
text stage-split support after the text lane passes, but multimodal support
requires projector/token-sideband evidence.

The candidate manifest lives at
`docs/skippy/llama-parity-candidates.json`.

## Family Board

This board is the running checklist. Update it whenever a family moves forward
or a blocker is discovered.

| Family | llama.cpp Model | Candidate | Downloaded | Text Split | Exact State | Cache Policy | Cache Smoke | Promotion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2` | `qwen2` | selected | yes | pass | pass | `ResidentKv` | pass | ready for reviewed promotion |
| `deepseek` | `deepseek` | selected | yes | pass | pass | `ResidentKv` | pass | promoted |
| `mistral` | `mistral3` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `lfm2` | `lfm2` | selected | yes | pass | pass | `KvRecurrent` target | pending cache smoke | text split ready; sticky recurrent ownership |
| `gpt2` | `gpt2` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `mpt` | `mpt` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `olmo2` | `olmo2` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `olmoe` | `olmoe` | selected | yes | pass | pass | `ResidentKv` target; MoE smoke required | pending cache smoke | text split ready |
| `qwen3vl` | `qwen3vl` | selected | yes | pass | pass | multimodal policy pending | pending projector/media lane | text split ready; multimodal pending |
| `phi` | `phi3` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `granite` | `granite` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `bloom` | `bloom` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `gptneox` | `gptneox` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `baichuan` | `baichuan` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `exaone` | `exaone` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `exaone4` | `exaone4` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `command_r` | `command-r` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `cohere2` | `cohere2` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `falcon` | `falcon` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `internlm2` | `internlm2` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `stablelm` | `stablelm` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `starcoder2` | `starcoder2` | selected | yes | pass | pass | `ResidentKv` target | pending cache smoke | text split ready |
| `mamba` | `mamba` | selected | yes | blocked: runtime-slice unsupported | pass | `KvRecurrent` target | pending after split support | blocked |
| `rwkv6` | `rwkv6` | selected | no | pending | pending | `KvRecurrent` target | pending | pending |
| `qwen2vl` | `qwen2vl` | selected | yes | pass | pass | multimodal policy pending | pending projector/media lane | text split ready; multimodal pending |
| `qwen2moe` | `qwen2moe` | needs candidate | no | pending | pending | pending | pending | pending |
| `qwen3moe` | `qwen3moe` | needs candidate | no | pending | pending | pending | pending | pending |
| `llama4` | `llama4` | package/remote only | no | package/remote pending | package/remote pending | package-local `ResidentKv` target | pending | pending |

Broader coverage lives in `docs/skippy/llama-parity-candidates.json`. The board
above tracks the active certification queue rather than every pinned llama.cpp
implementation.

## Next Batch

1. Add runtime-slice execution support for the remaining unsupported sampled
   families: `jamba`, `mamba`, `mamba2`, and `rwkv7`.
2. Debug `gemma`: runtime-slice opens, but split output predicts token `0`
   where the full model predicts token `1106`.
3. Finish the missing downloads and run the same sweep for `bert`, `t5`,
   `qwen2moe`, `qwen3moe`, and `llama4` where applicable.
4. Promote multimodal only after projector/media sideband evidence, even when
   text-lane split support passes.

## Current Local Evidence

These rows were collected on the local Mac Studio against the Metal stage ABI.
They are cheap text-split and cache-smoke evidence, not full promotion by
themselves until the reviewed topology records are updated.

| Family | Artifact | Text Split | q8 Wire | Exact State | Cache |
| --- | --- | --- | --- | --- | --- |
| `qwen2` | `meshllm/qwen2.5-0.5b-instruct-parity-q8_0-gguf` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 10.82x cache-hit speedup |
| `deepseek` | `Morgen0052/deepseek-llm-7b-chat-Q4_K_M-GGUF` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 1.58x cache-hit speedup |
| `mistral3` | `meshllm/mistral-7b-instruct-v0.3-parity-f16-gguf` | invalid for this row: GGUF reports architecture `llama` | invalid | invalid | invalid; replaced by `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF` candidate |
| `baichuan`, `bloom`, `gptneox`, `phi`, `stablelm` | see `target/family-certify/llama-parity-dense-tranche-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke pending |
| `command_r`, `cohere2`, `exaone`, `exaone4`, `falcon`, `internlm2`, `mistral3` | see `target/family-certify/llama-parity-dense-tranche-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke pending |
| `granite` | see `target/family-certify/llama-parity-dense-tranche-2-granite-fix2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke pending; fixed staged activation rescaling |
| `starcoder2` | see `target/family-certify/llama-parity-dense-tranche-2-external` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke pending |
| `gpt2` | see `target/family-certify/llama-parity-decoder-tranche-3e` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke pending; fixed mid-stage position input registration |
| `mpt`, `olmo2`, `olmoe` | see `target/family-certify/llama-parity-decoder-tranche-3c` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke pending |
| `qwen2vl`, `qwen3vl` | see `target/family-certify/llama-parity-decoder-tranche-3c` | text `single-step`, `chain`, and dtype matrix passed | validated | accepted | text lane only; projector/media-token lane still required |
| `lfm2` | see `target/family-certify/llama-parity-lfm2-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke pending; keep recurrent ownership sticky |
| `mamba` | see `target/family-certify/llama-parity-cheap-available` | blocked before split execution | untested | accepted | blocked until recurrent runtime-slice support lands |
| `jamba`, `mamba2`, `rwkv7` | see `target/family-certify/llama-parity-external-available` | blocked before split execution | untested | accepted | blocked until per-family runtime-slice support lands |
| `gemma` | `ggml-org/gemma-3-270m-it-GGUF` | runtime-slice opens but split token mismatches | passed | accepted | blocked until split output parity is fixed |

Raw run directories:

- `target/family-certify/llama-parity-cheap-qwen2-full-4`
- `target/family-certify/llama-parity-cheap-mistral-full`
- `target/family-certify/llama-parity-cheap-lfm2`
- `target/family-certify/llama-parity-cheap-olmoe`
- `target/family-certify/llama-parity-cheap-qwen3vl-text`
- `target/family-certify/llama-parity-cheap-available`
- `target/family-certify/llama-parity-external-available`
- `target/family-certify/llama-parity-cache-deepseek`
- `target/family-certify/llama-parity-dense-tranche-1`
- `target/family-certify/llama-parity-dense-tranche-2`
- `target/family-certify/llama-parity-dense-tranche-2-granite-fix2`
- `target/family-certify/llama-parity-dense-tranche-2-external`
- `target/family-certify/llama-parity-remaining-local-1`
- `target/family-certify/llama-parity-remaining-external-1`
- `target/family-certify/llama-parity-decoder-tranche-3c`
- `target/family-certify/llama-parity-decoder-tranche-3e`
- `target/family-certify/llama-parity-lfm2-runtime-slice-2`
- `target/family-certify/cache-smoke/reports`

## Current Blockers

- Runtime-slice expansion now passes for `baichuan`, `bloom`, `command_r`,
  `cohere2`, `exaone`, `exaone4`, `falcon`, `gpt2`, `gptneox`, `granite`,
  `internlm2`, `lfm2`, `mistral3`, `mpt`, `olmo2`, `olmoe`, `phi3`,
  `qwen2vl` text, `qwen3vl` text, `stablelm`, and `starcoder2`.
  These rows still need serving cache smoke before cache-on-by-default
  promotion.
- Remaining sampled families that still reject runtime-slice execution include
  `jamba`, `mamba`, `mamba2`, and `rwkv7`.
- `gemma` is different: runtime-slice execution opens, but split output parity
  fails. In the cheap run, the full model predicted token `1106` after token
  `2`, while the split path predicted token `0`.
- The old `mistral3` candidate was not a `mistral3` GGUF. It reports
  `general.architecture = llama`, so it cannot be used for llama.cpp family
  parity even though the run itself passed. The replacement candidate is
  `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF`.
- qwen2 full-state handoff was fixed by syncing token-count-aware imports back
  into the native session position before decode. Without that, restored KV was
  present but decode restarted at position zero.
- `qwen2vl` and `qwen3vl` are still only text-lane candidates. Full multimodal
  parity also needs projector/media-token sideband evidence.

## Cache Smoke Commands

The production cache gate should use `ResidentKv` or `KvRecurrent`, not
`FullState`. Dense families use this shape:

```bash
target/debug/skippy-correctness state-handoff \
  --model /path/to/model.gguf \
  --model-id org/repo \
  --layer-end 24 \
  --ctx-size 128 \
  --n-gpu-layers 999 \
  --prompt Hello \
  --stage-server-bin target/debug/skippy-server \
  --activation-width 896 \
  --source-bind-addr 127.0.0.1:19831 \
  --restore-bind-addr 127.0.0.1:19832 \
  --activation-wire-dtype f16 \
  --state-payload-kind resident-kv \
  --borrow-resident-hits \
  --cache-hit-repeats 3 \
  --prefix-token-count 64 \
  --report-out target/family-certify/cache-smoke/reports/family-resident-kv.json
```
