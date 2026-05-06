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

## Cache Correctness Exit Gate

Before this branch is considered done, prove exact-prefix cache correctness by
family/state-layout class. The important invariant is native sequence remap:
state recorded from one runtime session/lane must restore exactly into a
different runtime session/lane with a different backend sequence id.

Required scenarios:

1. Dense `ResidentKv` families:
   - Use representative local GGUFs for Llama, Qwen dense, Gemma, GLM, OLMo,
     or the closest locally available dense reviewed families.
   - Use `ResidentKv` only.
   - Record a prefix from session A, restore or borrow it into session B, and
     verify the next token/logits match normal no-cache prefill.
   - Verify suffix prefill after restore also matches normal prefill.
   - Run one-stage and at least one split-stage topology.
2. Hybrid/recurrent `KvRecurrent` families:
   - Use Falcon-H1 and Qwen3Next/Qwen3.6 recurrent when available.
   - Include MiniMax, RWKV, Mamba-like, Jamba, or LFM2 representatives when
     available.
   - Use `KvRecurrent` only.
   - Verify attention KV plus recurrent/SSM state restores exactly into a
     different target session/lane.
   - Verify immediate decode and suffix-prefill-then-decode.
   - Report payload bytes, imported tokens, hit/miss counters, and whether
     recurrent bytes are non-zero.
3. Negative policy checks:
   - Confirm recurrent/stateful GGUFs do not select `ResidentKv`.
   - Check whether current recurrent guards detect tensor names containing
     `.ssm`, `ssm_`, `time_mix`, `recurrent`, or `rwkv`.
   - If a stateful family uses different tensor naming, flag it and propose the
     detection rule before enabling cache reuse.
4. Split-stage correctness:
   - Verify cache restore in staged serving, not only single-stage.
   - Include stage 0, middle-stage, and final-stage restore where the harness
     can observe the relevant cache/activation boundary.
   - Vary layer ranges enough to catch stage-local KV or recurrent state shape
     issues.

Pass criteria:

- Cached restore produces the same next token/logits as normal prefill.
- Restore target is a different runtime session/lane from the recording source.
- Repeated cache hits remain stable.
- Unknown or failing families remain disabled or recompute-only.

The final evidence table should include one row per family/model ref with:
payload mode, one-stage result, split-stage result, source native seq id,
target native seq id, imported tokens, payload bytes, recurrent bytes, hit/miss
counters, repeated-hit stability, suffix-prefill result, and promotion decision.

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
| `mistral` | `mistral3` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `lfm2` | `lfm2` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent cache restore ready; keep normal decode ownership sticky |
| `gpt2` | `gpt2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `gemma` | `gemma` | selected | yes | pass with `f32` wire | pass | `ResidentKv` | pass | cache restore ready with `f32`; `f16`/`q8` rejected |
| `mpt` | `mpt` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `olmo2` | `olmo2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `olmoe` | `olmoe` | selected | yes | pass | pass | `ResidentKv` | pass + MoE expert smoke | cache restore and MoE expert-stage smoke ready |
| `qwen3vl` | `qwen3vl` | selected | yes | pass | pass | multimodal policy pending | local projector pass; split blocked | text split and local multimodal ready; split media sideband crashes in filtered stage-0 prefill |
| `phi` | `phi3` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `phi2` | `phi2` | selected | yes | pass | rejected-too-large | `ResidentKv` | pass | runtime-slice and resident cache ready; full-state payload is too large for mobility |
| `granite` | `granite` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `granite_hybrid` | `granite-hybrid` | selected | yes | pass | rejected-too-large | `KvRecurrent` | pass | runtime-slice and recurrent cache restore ready; keep normal decode ownership sticky |
| `granite_moe` | `granite-moe` | layout probe selected | yes | pass | pass | `ResidentKv` | pass | tiny random layout probe only; replace with a real small artifact when available |
| `hunyuan_dense` | `hunyuan-dense` | selected | yes | pass | pass | `ResidentKv` | pass | runtime-slice and cache restore ready |
| `hunyuan_moe` | `hunyuan-moe` | selected | yes | pass | pass | `ResidentKv` | pass | real A13B MoE runtime-slice and cache restore ready |
| `bloom` | `bloom` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `gptneox` | `gptneox` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `baichuan` | `baichuan` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `exaone` | `exaone` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `exaone4` | `exaone4` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `command_r` | `command-r` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `cohere2` | `cohere2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `jamba` | `jamba` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent cache restore ready; middle stage can be recurrent-only |
| `falcon` | `falcon` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `internlm2` | `internlm2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `stablelm` | `stablelm` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `starcoder2` | `starcoder2` | selected | yes | pass | pass | `ResidentKv` | pass | cache restore ready |
| `mamba` | `mamba` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent-only cache restore ready; keep normal decode ownership sticky |
| `mamba2` | `mamba2` | selected | yes | pass | pass | `KvRecurrent` | pass | recurrent-only cache restore ready; keep normal decode ownership sticky |
| `rwkv6` | `rwkv6` | replacement selected | yes | pass | pass | `KvRecurrent` | pass | recurrent-only cache restore ready; keep normal decode ownership sticky |
| `qwen2vl` | `qwen2vl` | selected | yes | pass | pass | multimodal policy pending | local projector pass; split blocked | text split and local multimodal ready; split media sideband crashes in filtered stage-0 prefill |
| `qwen2moe` | `qwen2moe` | selected | yes | pass | pass | `ResidentKv` | pass + MoE expert smoke | cache restore and MoE expert-stage smoke ready |
| `qwen3moe` | `qwen3moe` | selected | yes | pass | pass | `ResidentKv` | pass + MoE expert smoke | cache restore and MoE expert-stage smoke ready |
| `llama4` | `llama4` | package/remote only | no | package/remote pending | package/remote pending | package-local `ResidentKv` target | pending | no cheap artifact; local glogwa68 sample reports `llama`, not `llama4` |

Broader coverage lives in `docs/skippy/llama-parity-candidates.json`. The board
above tracks the active certification queue rather than every pinned llama.cpp
implementation.

## Next Batch

1. Finish the missing causal sweep for `llama4` where applicable.
2. Keep `bert` and `t5` in the non-causal aux lane instead of promoting them
   as causal stage-split serving.
3. Promote multimodal only after projector/media sideband evidence, even when
   text-lane split support passes.

## Current Local Evidence

These rows were collected on the local Mac Studio against the Metal stage ABI.
They are cheap text-split and cache-smoke evidence, not full promotion by
themselves until the reviewed topology records are updated.

| Family | Artifact | Text Split | q8 Wire | Exact State | Cache |
| --- | --- | --- | --- | --- | --- |
| `qwen2` | `meshllm/qwen2.5-0.5b-instruct-parity-q8_0-gguf` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 10.82x cache-hit speedup |
| `deepseek` | `Morgen0052/deepseek-llm-7b-chat-Q4_K_M-GGUF` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 1.58x cache-hit speedup |
| `mistral3` | `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 88.40x cache-hit speedup |
| `baichuan` | see `target/family-certify/llama-parity-dense-tranche-1` and `/tmp/skippy-cache-correctness-dense-*` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed |
| `phi` | see `target/family-certify/llama-parity-phi-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 2645.30x cache-hit speedup |
| `bloom` | see `target/family-certify/llama-parity-bloom-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 328.66x cache-hit speedup |
| `gptneox` | see `target/family-certify/llama-parity-gptneox-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 282.70x cache-hit speedup |
| `stablelm` | see `target/family-certify/llama-parity-stablelm-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 211.80x cache-hit speedup |
| `phi2` | see `target/family-certify/llama-parity-phi2-runtime-slice-2`, `target/family-certify/llama-parity-phi2-resident-kv-1`, and `/Volumes/External/tmp/skippy-phi2-resident-kv-stage1-20260506.json` | `single-step`, `chain`, and dtype matrix passed | validated | rejected-too-large for full-state | `ResidentKv` one-stage and split-final cache restore passed; stage0 cache probe remaps resident KV but cannot produce logits because it is a non-output slice |
| `command_r`, `cohere2`, `exaone`, `exaone4`, `falcon`, `internlm2` | see `target/family-certify/llama-parity-dense-tranche-2` and `/tmp/skippy-cache-correctness-dense-*` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed |
| `granite` | see `target/family-certify/llama-parity-dense-tranche-2-granite-fix2` and `/tmp/skippy-cache-correctness-dense-medium` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed; fixed staged activation rescaling |
| `granite_hybrid` | see `target/family-certify/llama-parity-granite-hybrid-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | rejected-too-large for full-state | `KvRecurrent` cache smoke passed; fixed Granite-Hybrid graph stage filtering |
| `granite_moe` | see `target/family-certify/llama-parity-granite-moe-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed on tiny random layout probe; fixed Granite-MoE ABI allowlist |
| `hunyuan_dense` | see `target/family-certify/llama-parity-hunyuan-dense-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed; fixed Hunyuan-Dense graph stage filtering |
| `hunyuan_moe` | see `target/family-certify/llama-parity-hunyuan-moe-runtime-slice-1` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed on real A13B MoE GGUF; fixed Hunyuan-MoE graph stage filtering |
| `starcoder2` | see `target/family-certify/llama-parity-dense-tranche-2-external` and `/tmp/skippy-cache-correctness-dense-medium` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed |
| `gpt2` | see `target/family-certify/llama-parity-gpt2-runtime-slice-1` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` borrowed-hit smoke passed, 64-token prefix, 1535.17x cache-hit speedup |
| `gemma` | see `target/family-certify/llama-parity-gemma-f32-wire-1` and `/tmp/skippy-cache-correctness-dense-gemma` | `single-step`, `chain`, and dtype matrix passed with `f32` only | rejected | accepted | `ResidentKv` cache smoke passed with `f32`; `f16` predicted token `0`, `q8` predicted token `107` |
| `mpt`, `olmo2`, `olmoe` | see `target/family-certify/llama-parity-decoder-tranche-3c`, `/tmp/skippy-cache-correctness-dense-*`, and `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke passed; `olmoe` MoE expert-stage smoke passed for one-stage, split-middle, and split-final |
| `qwen2vl`, `qwen3vl` | see `target/family-certify/llama-parity-decoder-tranche-3c`; local smoke via `cargo test -p skippy-server real_multimodal_local_smoke_when_fixture_is_set` with real mmproj/image fixtures | text `single-step`, `chain`, and dtype matrix passed; full-model local multimodal OpenAI smoke passed | validated for text; local projector validated | accepted for text/local runtime only | split multimodal still blocked: filtered stage-0 media prefill reaches `mtmd_helper_eval_chunks`, prints `embeddings required...`, then SIGSEGVs before activation forwarding |
| `gemma3n` | `lmstudio-community/gemma-3n-E2B-it-GGUF` | blocked: llama.cpp reports `runtime-slice execution is not supported for this model architecture yet` | untested | untested | text and multimodal promotion blocked until Gemma3n graph support and media loader/projector strategy are implemented |
| `qwen2moe` | see `target/family-certify/llama-parity-qwen2moe-runtime-slice-4`, `/tmp/skippy-cache-correctness-dense-medium`, and `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506` | `single-step`, `chain`, and f16 dtype matrix passed | rejected | accepted | `ResidentKv` cache smoke and MoE expert-stage smoke passed |
| `qwen3moe` | see `target/family-certify/llama-parity-qwen3moe-runtime-slice-1`, `/tmp/skippy-cache-correctness-dense-medium`, and `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `ResidentKv` cache smoke and MoE expert-stage smoke passed |
| `lfm2` | see `target/family-certify/llama-parity-lfm2-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed; keep recurrent ownership sticky for normal decode |
| `jamba` | see `target/family-certify/llama-parity-jamba-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed; middle-stage recurrent-only slices are valid |
| `mamba` | see `target/family-certify/llama-parity-mamba-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes |
| `mamba2` | see `target/family-certify/llama-parity-mamba2-runtime-slice-2` | `single-step`, `chain`, and dtype matrix passed | validated | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes |
| `rwkv6` | see `target/family-certify/llama-parity-rwkv6-runtime-slice-3` | `single-step`, `chain`, and dtype matrix passed | rejected | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes; keep recurrent ownership sticky |
| `rwkv7` | `Mungert/rwkv7-191M-world-GGUF` plus `target/family-certify/rwkv7-sideband-*.json` | `single-step`, `chain`, and dtype matrix passed | validated on sampled artifact | accepted | `KvRecurrent` cache smoke passed with zero native KV bytes; activation-frame sideband carries layer-0 `v_first` |

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
- `target/family-certify/llama-parity-gemma-f32-wire-1`
- `target/family-certify/llama-parity-jamba-runtime-slice-2`
- `target/family-certify/llama-parity-lfm2-runtime-slice-2`
- `target/family-certify/llama-parity-mamba-runtime-slice-2`
- `target/family-certify/llama-parity-mamba2-runtime-slice-2`
- `target/family-certify/llama-parity-granite-hybrid-runtime-slice-2`
- `target/family-certify/llama-parity-granite-moe-runtime-slice-2`
- `target/family-certify/llama-parity-hunyuan-dense-runtime-slice-2`
- `target/family-certify/llama-parity-hunyuan-moe-runtime-slice-1`
- `target/family-certify/llama-parity-qwen2moe-runtime-slice-4`
- `target/family-certify/llama-parity-qwen3moe-runtime-slice-1`
- `target/family-certify/rwkv7-sideband-single-step.json`
- `target/family-certify/rwkv7-sideband-chain.json`
- `target/family-certify/rwkv7-sideband-dtype-matrix.json`
- `/tmp/skippy-cache-correctness-dense-small`
- `/tmp/skippy-cache-correctness-dense-medium`
- `/tmp/skippy-cache-correctness-dense-large`
- `/tmp/skippy-cache-correctness-dense-mistral3`
- `/tmp/skippy-cache-correctness-dense-gemma`
- `/Volumes/External/tmp/skippy-moe-expert-smoke-20260506`
- `/Volumes/External/tmp/skippy-smoke-qwen3-coder-480b/state-handoff.json`

## Cache Correctness Evidence

The source/target native-sequence remap gate is reproducible with:

```bash
LLAMA_STAGE_BUILD_DIR=$PWD/.deps/llama-build/build-stage-abi-metal \
  python3 evals/skippy-cache-correctness-gate.py \
    --output-dir /tmp/skippy-cache-correctness-gate \
    --llama-stage-build-dir $PWD/.deps/llama-build/build-stage-abi-metal \
    --topology one-stage \
    --topology split-middle \
    --topology split-final \
    --prefix-tokens 16 \
    --suffix-token-count 3 \
    --runtime-lane-count 4 \
    --cache-hit-repeats 2 \
    --n-gpu-layers 999
```

Latest local result: `102/102` rows passed across tranche runs. Every row
restored into a different native sequence (`0 -> 1`), suffix-prefill-then-decode
matched normal prefill, and repeated hits stayed stable. `KvRecurrent` rows
carried non-zero recurrent payloads. Pure recurrent families and recurrent-only
stage ranges correctly recorded zero native KV bytes plus recurrent state.
Dense and MoE-text `ResidentKv` additions passed in:
`/tmp/skippy-cache-correctness-dense-small`,
`/tmp/skippy-cache-correctness-dense-medium`,
`/tmp/skippy-cache-correctness-dense-large`,
`/tmp/skippy-cache-correctness-dense-mistral3`, and
`/tmp/skippy-cache-correctness-dense-gemma`. An attempted unified all-in-one
rerun exited early during process startup before writing a report; the completed
tranche reports are the current evidence.

Negative policy regression tests now assert that Falcon-H1, Qwen3Next, Jamba,
LFM2, Mamba, Mamba2, RWKV6, and RWKV7 select `KvRecurrent`, never `ResidentKv`,
through mesh family policy and server-side auto-payload inference.

| Family | Model ref | Payload | Topology | Result | Seq remap | Source -> target seq | Suffix prefill | Payload bytes | Recurrent bytes | Repeated hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 dense | `Qwen/Qwen3-0.6B:Q8_0` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Qwen3 dense | `Qwen/Qwen3-0.6B:Q8_0` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Qwen3 dense | `Qwen/Qwen3-0.6B:Q8_0` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Llama | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Llama | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Llama | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| GLM4 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| GLM4 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| GLM4 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Gemma3 | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Gemma3 | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Gemma3 | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Phi2 | `TheBloke/phi-2-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Phi2 | `TheBloke/phi-2-GGUF:Q4_K_M` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Granite-Hybrid | `magiccodingman/Granite-4.0-H-350M-Unsloth-MXFP4-Hybrid-GGUF:MXFP4_MOE-output_q6_K-router_gate_emb_q6_K` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `22885052` | `22622908` | pass |
| Granite-MoE | `mradermacher/tiny-random-granite-moe-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Hunyuan-Dense | `Edge-Quant/Hunyuan-1.8B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Falcon-H1 | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `76923484` | `76530268` | pass |
| Falcon-H1 | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `76661340` | `76530268` | pass |
| Falcon-H1 | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `76661340` | `76530268` | pass |
| OLMo | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `ResidentKv` | one-stage | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| OLMo | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `ResidentKv` | split-middle | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| OLMo | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `ResidentKv` | split-final | pass | yes | `0 -> 1` | pass | `0` | `0` | pass |
| Qwen3Next | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `79430524` | `79037308` | pass |
| Qwen3Next | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `79168380` | `79037308` | pass |
| Qwen3Next | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `79168380` | `79037308` | pass |
| Jamba | `bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `10134156` | `10117772` | pass |
| Jamba | `bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `10117772` | `10117772` | pass |
| Jamba | `bartowski/ai21labs_AI21-Jamba2-3B-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `10125964` | `10117772` | pass |
| LFM2 | `meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `278796` | `82188` | pass |
| LFM2 | `meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `147724` | `82188` | pass |
| LFM2 | `meshllm/lfm2-350m-parity-q4_k_m-gguf:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `180492` | `82188` | pass |
| Mamba | `mradermacher/mamba-130m-hf-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `2802268` | `2802268` | pass |
| Mamba | `mradermacher/mamba-130m-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `2802268` | `2802268` | pass |
| Mamba | `mradermacher/mamba-130m-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `2802268` | `2802268` | pass |
| Mamba2 | `mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `24905244` | `24905244` | pass |
| Mamba2 | `mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `24905244` | `24905244` | pass |
| Mamba2 | `mradermacher/mamba-2.8b-hf-GGUF:Q4_K_M` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `24905244` | `24905244` | pass |
| RWKV6 | `latestissue/rwkv-6-finch-1b6-gguf:Q4_K` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `12976732` | `12976732` | pass |
| RWKV6 | `latestissue/rwkv-6-finch-1b6-gguf:Q4_K` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `12976732` | `12976732` | pass |
| RWKV6 | `latestissue/rwkv-6-finch-1b6-gguf:Q4_K` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `12976732` | `12976732` | pass |
| RWKV7 | `Mungert/rwkv7-191M-world-GGUF:Q4_K` | `KvRecurrent` | one-stage | pass | yes | `0 -> 1` | pass | `2433340` | `2433340` | pass |
| RWKV7 | `Mungert/rwkv7-191M-world-GGUF:Q4_K` | `KvRecurrent` | split-middle | pass | yes | `0 -> 1` | pass | `2433340` | `2433340` | pass |
| RWKV7 | `Mungert/rwkv7-191M-world-GGUF:Q4_K` | `KvRecurrent` | split-final | pass | yes | `0 -> 1` | pass | `2433340` | `2433340` | pass |
- `target/family-certify/llama-parity-rwkv6-runtime-slice-3`
- `target/family-certify/cache-smoke/reports`

## Current Blockers

- Runtime-slice expansion and cache restore now pass for `baichuan`, `bloom`,
  `command_r`, `cohere2`, `exaone`, `exaone4`, `falcon`, `gemma` with `f32`
  wire, `gpt2`, `gptneox`, `granite`, `granite_hybrid`, `granite_moe`,
  `hunyuan_dense`, `hunyuan_moe`, `internlm2`, `mistral3`, `mpt`, `olmo2`,
  `phi2`, `phi3`, `stablelm`, and `starcoder2`.
- `hunyuan_dense` required a llama.cpp stage-filter fix in the shared
  Hunyuan-Dense/VL graph and uses `ResidentKv`.
- `hunyuan_moe` required a llama.cpp stage-filter fix in the Hunyuan-MoE graph
  and uses `ResidentKv`.
- `granite_hybrid` required a llama.cpp stage-filter fix for its hybrid graph
  and uses `KvRecurrent`; the exact full-state payload is too large to move as a
  production cache value.
- `granite_moe` reuses the Granite graph and only needed the skippy ABI
  allowlist. Current evidence uses a tiny random GGUF, so it certifies graph and
  tensor-layout support rather than model quality.
- `phi2` required a llama.cpp stage-filter fix for filtered fused-QKV tensors:
  when a merged QKV weight is skipped for a slice, the matching merged QKV bias
  must also be accounted for instead of falling back to separate Q/K/V tensors.
- `gemma3n` is locally downloaded, but remains blocked before promotion because
  the pinned llama.cpp graph rejects runtime-slice execution for `gemma3n`; the
  sampled LM Studio artifact also does not expose a separate mmproj file for the
  current multimodal smoke harness.
- `olmoe`, `qwen2moe`, and `qwen3moe` now pass MoE expert-stage smoke for
  one-stage, split-middle, and split-final ranges. The smoke confirms nonzero
  expert tensor ownership in each tested range, native sequence remap `0 -> 1`,
  suffix-prefill match, repeated hit stability, and nonzero resident KV bytes.
- `jamba`, `lfm2`, `mamba`, `mamba2`, `rwkv6`, and `rwkv7` now pass
  `KvRecurrent` cache smoke for one-stage, split-middle, and split-final
  restore into a different native sequence. Keep recurrent ownership sticky for
  normal staged decode; the cache proof is for exact prefix restore.
- `rwkv7` uses a wider activation-frame contract. Later RWKV7 layers depend on
  the layer-0 `v_first` tensor, so non-first stages receive hidden state plus a
  `v_first` activation sideband. The sampled 12-layer artifact now passes
  two-stage, three-stage, f32/f16/q8 wire checks, and `KvRecurrent` cache smoke.
- The old local `rwkv6` sample was not a GGUF artifact. Its files carry the
  legacy `fmgg` magic and fail GGUF metadata inspection, so the replacement
  candidate is `latestissue/rwkv-6-finch-1b6-gguf`. The replacement passed the
  text lane; exact state is intentionally rejected as too large.
- `gemma` is stage-correct only with `f32` activation wire for the sampled
  artifact. The earlier default-`f16` cheap run predicted token `0`, and `q8`
  predicted token `107`, while `f32` matched token `1106`.
- `llama4` does not have a cheap local certification artifact yet. The local
  `glogwa68/Llama-4-scout-GGUF` sample reports `general.architecture = llama`,
  not `llama4`; official llama4 Scout GGUF artifacts are package/remote-sized.
- The old `mistral3` candidate was not a `mistral3` GGUF. It reports
  `general.architecture = llama`, so it cannot be used for llama.cpp family
  parity even though the run itself passed. The replacement candidate is
  `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF`.
- qwen2 full-state handoff was fixed by syncing token-count-aware imports back
  into the native session position before decode. Without that, restored KV was
  present but decode restarted at position zero.
- `qwen2vl` and `qwen3vl` now pass full-model local multimodal OpenAI smoke
  with real projectors and an image fixture. Split multimodal is still blocked:
  after enabling `filter_tensors_on_load=true` for split stages, stage-0 media
  prefill reaches `mtmd_helper_eval_chunks`, prints `embeddings required but
  some input tokens were not marked as outputs -> overriding`, and then
  SIGSEGVs before activation forwarding. Do not promote multimodal stage-split
  support until the filtered stage-0 media prefill ABI is fixed.

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
