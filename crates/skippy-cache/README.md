# skippy-cache

`skippy-cache` owns the cache model for staged serving. It does not talk to
llama.cpp, open sockets, route OpenAI requests, or plan topology. Those
responsibilities stay in `skippy-server`, `skippy-runtime`, and mesh.

The crate answers cache questions:

- what exact prefix identity should be used for this stage?
- which prefix lengths should be looked up or recorded?
- which resident cache entry should be evicted?
- how should exact state payload bytes be represented and deduplicated?

The cache is deliberately exact. It does not guess that two prompts are
"similar enough"; it reuses runtime state only when the model, topology, stage,
layer range, runtime layout, position, and token prefix all match.

## Boundaries

```mermaid
flowchart LR
    OpenAI["openai-frontend"] --> Server["skippy-server"]
    Server --> Runtime["skippy-runtime"]
    Runtime --> ABI["llama.cpp stage ABI"]

    Server --> Cache["skippy-cache"]
    Cache --> Identity["prefix identity"]
    Cache --> Resident["resident prefix / activation caches"]
    Cache --> Payload["exact state payloads"]
    Payload --> Blocks["BLAKE3 block store"]

    Cache -. no runtime calls .- Runtime
```

`skippy-server` remains the adapter. It turns protocol messages into cache
lookups, performs the runtime save/restore/import/export calls, and records
telemetry. `skippy-cache` only owns pure data structures and policies.

## Prefix Identity

Prefix identity is exact. A hit is valid only for the same:

- model id
- topology id
- stage id and stage index
- layer range
- runtime ABI schema
- KV layout and dtype
- context/position configuration
- token start and token ids

The identity hash uses BLAKE3 over these fields. The short page id is derived
from the hash and is suitable for logs and resident-cache maps.

```mermaid
flowchart TD
    Tokens["token ids + token_start"] --> Hash["BLAKE3 prefix hash"]
    Stage["model/topology/stage/layers"] --> Hash
    Runtime["ABI + layout + dtype + ctx"] --> Hash
    Hash --> Page["page id"]
    Page --> Lookup["resident/exact-state lookup"]
```

## Resident Cache Flow

The current resident path keeps reusable state inside the live llama.cpp
session. `skippy-cache` chooses candidates and tracks entries; `skippy-server`
does the native sequence copy/drop calls.

```mermaid
sequenceDiagram
    participant S as skippy-server
    participant C as skippy-cache
    participant R as skippy-runtime

    S->>C: candidate prefix identities
    C-->>S: longest-first lookup candidates
    S->>C: lookup resident entry
    alt hit
        C-->>S: cache seq id
        S->>R: restore resident prefix
        S->>S: compute only suffix or forward cached activation
    else miss
        S->>R: prefill normally
        S->>C: allocate record slot
        C-->>S: cache seq id + evictions
        S->>R: drop evicted seq ids
        S->>R: save resident prefix
        S->>C: commit record
    end
```

Activation frames are cached separately by `act:{page_id}:w{activation_width}`.
In a split topology, an upstream stage may reuse both its resident prefix and
the activation frame it would have forwarded downstream.

For dense models this is the production fast path. It is intentionally close to
the llama-server slot contract: keep usable KV resident in the runtime and
continue from it. The difference is that Skippy gives the resident state a
backend-neutral prefix identity, keeps hot lane prefixes borrowed in place when
possible, and can reuse the same prefix across Rust-owned serving, staged
execution, and activation forwarding.

## Exact State Payloads

The payload module models the portable exact-cache path we need for
recurrent/stateful families:

| Payload | Contents | Use |
| --- | --- | --- |
| `FullState` | whole exported sequence state | diagnostic/reference only; not selected by production serving policy |
| `RecurrentOnly` | recurrent/SSM state only | diagnostic; not generally exact for non-final stages |
| `KvRecurrent` | attention KV plus recurrent/SSM state | preferred exact payload for hybrid/recurrent repeated-prefix cache |

Large payloads are split into 1 MiB BLAKE3-addressed blocks. Repeated blocks are
stored once, so capacity is accounted by physical bytes as well as logical
payload bytes.

`KvRecurrent` is the production method for hybrid/recurrent families. It moves
attention KV together with recurrent/SSM state, because KV alone is not the full
continuation state for those models. That is where the largest wins come from:
llama-server warm slots still have to reprocess the recurrent prefix in many
request shapes, while Skippy can restore the exact compact state and decode.

```mermaid
flowchart LR
    Payload["KV + recurrent payload"] --> Split["1 MiB chunks"]
    Split --> H1["BLAKE3 block A"]
    Split --> H2["BLAKE3 block B"]
    Split --> H3["BLAKE3 block A reused"]
    H1 --> Store["block store"]
    H2 --> Store
    H3 -. refcount .-> Store
    Store --> Stats["logical / physical / saved bytes"]
```

## Family Cache Policy

Cache support is family-dependent because different model families have
different continuation state.

| Family shape | Safe cache shape | Notes |
| --- | --- | --- |
| Dense attention models such as Llama, Qwen3 dense, DeepSeek2, DeepSeek3, GLM4, GLM-4.7 Flash, OLMo, Gemma, MiniMax-M2.7 | `ResidentKv` | Attention KV is the continuation state for the certified prompt-continuation path. The cache entry is resident in the live runtime, so serving reuses llama.cpp sequence state without serializing bytes. |
| MoE attention models such as OLMoE, Qwen2-MoE, and Qwen3-MoE | `ResidentKv` | Expert routing is part of the layer compute, while attention KV remains the exact continuation state. Promotion requires MoE expert-stage smoke so every tested stage range owns expert tensors and still restores exactly. |
| Hybrid/recurrent models such as Qwen3Next, Falcon-H1, RWKV/Mamba-like tensors | `KvRecurrent` only | KV-only reuse is disabled; recurrent/SSM state must be restored with KV for exact continuation. |
| Families with uncertain state layout | disabled until certified | Unknown families should not silently reuse state. |
| Diagnostic/correctness runs | `FullState`, `RecurrentOnly`, or `KvRecurrent` | Used to prove exactness and payload economics before enabling serving policy. |

The serving integration currently scans GGUF tensor names and disables the
KV-only resident path when recurrent/stateful tensors are present, including
`.ssm`, `ssm_`, `time_mix`, `recurrent`, and `rwkv`. That guard prevents a
hybrid model from importing attention KV while missing the recurrent state that
changes the next token.

The intended recurrent/stateful serving path is:

1. lookup exact prefix identity
2. import/restore `KvRecurrent`
3. recompute safely on any miss or incompatible payload
4. record a new exact payload after successful prefill

The OpenAI serving path now uses the same cache policy. A cache-enabled stage
selects one payload shape:

- `ResidentKv` for certified dense families where resident llama.cpp sequence
  copy is enough
- `KvRecurrent` for hybrid/recurrent families where attention KV and recurrent
  state must move together

`FullState` is intentionally not selected by the production family policy. It is
kept as a correctness/certification tool so a family can first prove exact
restore behavior before we design and promote a compact serving payload.

## Why Skippy Cache Beats llama-server

llama-server slots are very good at one thing: keeping a prompt prefix warm
inside a server slot. Skippy preserves that behavior for dense single-stage
serving, then extends it in several important ways.

| Method | Where it wins | Why |
| --- | --- | --- |
| Exact prefix identity | all cache modes | Prefixes are keyed by model ref, topology, stage, layer range, ABI/layout, position, and token ids, so state can be reused safely outside a single llama-server slot. |
| `ResidentKv` | dense attention families | KV stays resident in the live runtime. Hot-lane hits borrow the existing prefix instead of paying serialize/restore costs, matching llama-server's slot model while using Skippy's lower-overhead Rust serving path. |
| `KvRecurrent` | Qwen3Next, Falcon-H1, and other hybrid/recurrent families | Attention KV and recurrent/SSM state move together. llama-server has no equivalent production cache for this exact state shape, so repeated prefixes avoid expensive recurrent-prefix reprocessing. |
| Activation-frame cache | split serving | A stage can reuse the activation it would have forwarded downstream, so cache hits can remove work from both the local stage and the next stage boundary. |
| BLAKE3 deduped exact payloads | portable exact-state cache | Large exported states are chunked into content-addressed blocks. Repeated blocks are stored once, making exact payload caching practical under a real capacity cap. |
| Package-backed stage cache | giant staged models | A materialized stage can cache only the state for the layer range it owns, without loading or merging a monolithic full GGUF. |

The headline pattern in the benchmark table follows from those methods:

- Hybrid/recurrent families show the largest wins because `KvRecurrent`
  restores state llama-server otherwise has to regenerate.
- Dense families show parity to moderate wins because `ResidentKv` keeps the
  same warm-slot advantage while avoiding extra serving-path overhead.
- DeepSeek3 is package-only evidence: it proves the selected cache method for a
  materialized stage, not a full-GGUF llama-server comparison.

The cache does not claim fuzzy semantic reuse, approximate prefix matching, or
`FullState` production speedups. `FullState` remains a certification tool only.

```mermaid
flowchart LR
    Request["request tokens"] --> Identity["exact prefix identity"]
    Identity --> Hit{"cache hit?"}
    Hit -- dense --> Resident["ResidentKv: borrow resident runtime KV"]
    Hit -- recurrent --> Recurrent["KvRecurrent: restore KV + recurrent state"]
    Hit -- split --> Activation["activation frame reuse"]
    Hit -- miss --> Prefill["normal prefill"]
    Prefill --> Record["record exact state"]
    Record --> Dedupe["BLAKE3 block dedupe + capacity accounting"]
```

```mermaid
sequenceDiagram
    participant O as openai-frontend
    participant S as skippy-server
    participant C as skippy-cache
    participant R as skippy-runtime

    O->>S: request
    S->>C: longest-prefix identities
    alt exact hit
        C-->>S: KvRecurrent payload
        S->>R: import state into lane
        S->>R: decode next token
    else resident hit
        C-->>S: resident seq id
        S->>R: restore resident prefix
        S->>R: decode next token
    else miss
        S->>R: prefill
        S->>R: export selected payload
        S->>C: dedupe/store and evict by cap
        S->>R: decode next token
    end
```

## Benchmark Evidence

README performance claims must be backed by correctness runs and comparable
llama-server baselines. Rows marked `untested` are intentionally not promoted as
default evidence yet.

Reproduce the family evidence and README tables with:

```bash
evals/skippy-cache-family-bench.sh /tmp/skippy-cache-family-bench
```

For a faster rerun after building locally:

```bash
SKIPPY_CACHE_SKIP_BUILD=1 evals/skippy-cache-family-bench.sh /tmp/skippy-cache-family-bench
```

Reproduce the MoE expert-stage cache smoke with:

```bash
python3 evals/skippy-moe-expert-smoke.py \
  --output-dir /tmp/skippy-moe-expert-smoke \
  --llama-stage-build-dir .deps/llama-build/build-stage-abi-metal \
  --n-gpu-layers 999
```

Local correctness evidence below was collected on the same machine with
`n_predict = 1`, `n_gpu_layers = -1`, Skippy `--runtime-lane-count 1`, and
llama-server `--parallel 1`. The payload column is the production serving
payload, not an experimental fallback. Results use median warm-hit latency from
matched repeated prompts. The full-GGUF table uses the same 128-token requested
prefix and one generated token for every row so each Skippy result is compared
against the matching llama-server workload.

`Cache bytes` is serialized payload size for `KvRecurrent` and measured native
KV-page footprint for `ResidentKv` when the runtime can expose it. Rows marked
`metadata-derived` use the same llama.cpp KV dimensions from GGUF metadata:
active KV layers, KV heads, key/value head lengths, SWA pattern, shared-KV
layers, token count, and f16 KV element size.

### Full-GGUF llama-server vs Skippy

Rows are ordered so related runtime/cache families appear next to each other.

| Family | Representative model ref | Production payload | Correctness | Prefix tokens | Prompt tokens | llama-server warm median ms | Skippy hit median ms | Skippy win | Cache bytes | Size method | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3Next | `bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS` | `KvRecurrent` | pass | 128 | 129 | 318.0 | 26.4 | **12.06x faster** | 78.4 MiB | measured | Recurrent-backed state reuse makes llama-server reprocess work Skippy skips. |
| Falcon-H1 | `tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M` | `KvRecurrent` | pass | 128 | 129 | 105.2 | 12.0 | **8.73x faster** | 76.0 MiB | measured | Same recurrent-state advantage on the Falcon-H1 cache payload. |
| Llama | `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M` | `ResidentKv` | pass | 128 | 129 | 8.8 | 4.5 | **1.95x faster** | 4.0 MiB | measured | Matches the llama-server slot contract, then wins on overhead. |
| Qwen3 dense | `Qwen/Qwen3-0.6B:Q8_0` | `ResidentKv` | pass | 128 | 129 | 7.4 | 6.8 | **1.08x faster** | 14.0 MiB | measured | Correct and faster; this is the narrowest win in the normalized matrix. |
| DeepSeek2 | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M` | `ResidentKv` | pass | 128 | 129 | 12.6 | 9.6 | **1.32x faster** | 33.8 MiB | measured | Correct and faster on the resident prefix path. |
| GLM-4.7 Flash | `unsloth/GLM-4.7-Flash-GGUF:Q4_K_M` | `ResidentKv` | pass | 128 | 129 | 26.0 | 18.9 | **1.38x faster** | 12.5 MiB | metadata-derived | Correct and faster under the normalized 128-token workload. |
| GLM4 | `meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M` | `ResidentKv` | pass | 128 | 129 | 23.2 | 18.9 | **1.23x faster** | 5.0 MiB | measured | Correct and faster. |
| Gemma4 A4B | `batiai/Gemma-4-26B-A4B-it-GGUF:Q6_K` | `ResidentKv` | pass | 128 | 129 | 21.0 | 17.6 | **1.20x faster** | 27.5 MiB | metadata-derived | Correct and faster with SWA KV dimensions accounted for in sizing. |
| Gemma4 E4B | `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` | `ResidentKv` | pass | 128 | 129 | 20.7 | 16.1 | **1.29x faster** | 7.0 MiB | metadata-derived | Correct and faster with shared-KV layers accounted for in sizing. |
| Gemma3 | `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` | `ResidentKv` | pass | 128 | 129 | 10.1 | 7.0 | **1.45x faster** | 3.2 MiB | metadata-derived | Hot-lane resident prefix reuse beats llama-server warm slots. |
| Gemma2 | `bartowski/gemma-2-2b-it-GGUF:Q4_K_M` | `ResidentKv` | pass | 128 | 129 | 12.1 | 8.5 | **1.43x faster** | 13.0 MiB | metadata-derived | Correct and faster. |
| OLMo | `meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16` | `ResidentKv` | pass | 128 | 129 | 28.9 | 24.0 | **1.20x faster** | 64.0 MiB | measured | Correct and faster. |
| MiniMax M2.7 | `unsloth/MiniMax-M2.7-GGUF:UD-Q2_K_XL` | `ResidentKv` | pass | 128 | 129 | 39.5 | 31.9 | **1.24x faster** | 31.0 MiB | measured | Correct and faster on the large sharded GGUF. |

### Use-Case Benchmark Matrix

This matrix uses one Hugging Face-sourced representative prompt per use case,
the same `128` requested prefix tokens, one generated token, Skippy
`--runtime-lane-count 1`, llama-server `--parallel 1`, and the same full-GGUF
family set as the table above. Values are Skippy warm-hit latency speedup over
llama-server warm-cache latency. DeepSeek3 stays in the package-only section
because there is no practical local full-GGUF llama-server baseline for that
artifact.

| Use case | Qwen3Next | Falcon-H1 | Llama | Qwen3 dense | DeepSeek2 | GLM-4.7 Flash | GLM4 | Gemma4 A4B | Gemma4 E4B | Gemma3 | Gemma2 | OLMo | MiniMax M2.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tool calling | 15.23x | 8.53x | 1.33x | 1.32x | 1.60x | 1.30x | 1.27x | 1.04x | 1.33x | 1.58x | 1.40x | 1.21x | 1.19x |
| Text-to-SQL | 15.42x | 8.58x | 1.69x | 1.35x | 1.52x | 1.19x | 1.13x | 1.30x | 1.26x | 1.64x | 1.41x | 1.08x | 1.14x |
| Coding agent loop | 14.62x | 8.24x | 1.56x | 1.68x | 1.47x | 1.24x | 1.28x | 1.32x | 1.25x | 1.54x | 1.52x | 1.24x | 1.26x |
| Issue fixing | 14.45x | 8.84x | 1.74x | 1.68x | 1.56x | 1.24x | 1.30x | 1.28x | 1.29x | 1.65x | 1.49x | 1.12x | 1.28x |
| Code refinement | 14.55x | 8.65x | 1.98x | 1.43x | 1.53x | 1.25x | 1.23x | 1.27x | 1.43x | 1.76x | 1.47x | 1.25x | 1.21x |
| Few-shot reasoning | 14.44x | 8.55x | 1.73x | 1.64x | 1.56x | 1.26x | 1.21x | 1.08x | 1.33x | 1.61x | 1.52x | 1.20x | 1.18x |
| Open chat | 13.74x | 9.15x | 1.62x | 1.35x | 1.60x | 1.30x | 1.23x | 1.36x | 1.31x | 1.58x | 1.28x | 1.06x | 1.27x |
| Summarization/RAG | 13.89x | 8.94x | 1.81x | 1.41x | 1.67x | 1.26x | 1.13x | 1.27x | 1.32x | 1.55x | 1.61x | 1.20x | 1.19x |

Prompt sources are checked in at `evals/skippy-usecase-corpus.json` with source
dataset metadata:

| Use case | Dataset | Config | Split | Row |
| --- | --- | --- | --- | ---: |
| Tool calling | `glaiveai/glaive-function-calling-v2` | `default` | `train` | 1 |
| Text-to-SQL | `gretelai/synthetic_text_to_sql` | `default` | `test` | 0 |
| Coding agent loop | `SWE-bench/SWE-smith-trajectories` | `default` | `tool` | 0 |
| Issue fixing | `SWE-bench/SWE-bench` | `default` | `dev` | 0 |
| Code refinement | `google/code_x_glue_cc_code_refinement` | `small` | `test` | 0 |
| Few-shot reasoning | `openai/gsm8k` | `main` | `test` | 0 |
| Open chat | `HuggingFaceH4/mt_bench_prompts` | `default` | `train` | 0 |
| Summarization/RAG | `nvidia/ChatRAG-Bench` | `doc2dial` | `test` | 0 |

### Package-Only Giant Models

These rows validate cache strategy for models where a full llama-server
baseline is not operationally useful because monolithic residency is too large.

| Family | Representative model ref | Production payload | Correctness | Prefix tokens | Prompt tokens | Baseline | Skippy hit median ms | Skippy win | Cache bytes | Size method | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| DeepSeek3 | `unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL` | `ResidentKv` | pass | 4 | 5 | Skippy stage recompute | 3.4 | **4.40x faster** | 8.5 KiB | metadata-derived | Package-only proof on expert slice `3..4`; selected 7.47 GB stage part plus upstream `0..3`, no full 406.8 GB layer set loaded or merged. |

DeepSeek3 uses `ResidentKv` for package-backed serving. The local gate verifies
the package stage cache without requiring a monolithic full-GGUF baseline:
`0..1` passed with real token input, `3..4` passed with a real upstream `0..3`
activation producer, and late layers `30..31` plus `60..61` passed with
deterministic synthetic upstream activations so only the owned stage range had
to be materialized.

The expanded source/target native-sequence correctness gate now covers both
production cache shapes. The latest local Metal tranche evidence passed
`102/102` rows across Qwen3 dense, Llama, GLM4, Gemma3, OLMo, Falcon-H1, Jamba,
LFM2, Mamba, Mamba2, RWKV6, RWKV7, Qwen3Next, and the expanded dense/MoE-text
families for one-stage, split-middle, and split-final topologies. Every row
restored from native sequence `0` into native sequence `1`,
suffix-prefill-then-decode matched, repeated hits stayed stable, and
recurrent-only ranges correctly recorded zero native KV bytes plus recurrent
state.

The MoE expert-stage smoke additionally passed `9/9` rows for OLMoE,
Qwen2-MoE, and Qwen3-MoE. That run verifies nonzero expert tensor ownership in
each tested stage range, nonzero resident KV cache bytes, native sequence remap
`0 -> 1`, suffix-prefill match, and repeated hit stability. `ResidentKv`
reports zero serialized payload bytes in this mode because the production hit
borrows resident runtime state instead of exporting a portable payload.

`state-handoff` reports distinguish behavioral exactness from byte-stable state
re-export. `status = pass` means the restored cache state produced the same next
token/output and repeat hits matched. `roundtrip_state_matches = false` means a
state exported after import was not byte-for-byte identical; that is a
canonicalization/deduplication diagnostic, not a serving correctness failure.
Raw reports from this run are under
`/tmp/skippy-cache-production-bench-track1-apples-p128`.

### Current Performance Read

With llama-server-compatible thread defaults, hot-lane resident prefix reuse,
and the production borrowed-prefix path, Skippy is at parity or ahead for every
locally available family in the normalized single-stage matrix:

| Scenario | Current read | Why |
| --- | --- | --- |
| Recurrent/hybrid families | Strong wins where compact recurrent state is exported | Falcon-H1 and Qwen3Next avoid llama-server reprocessing by restoring `KvRecurrent` state. |
| Dense attention families | Parity to moderate wins | Hot-lane resident prefix reuse preserves the same lane-local slot behavior that made llama-server fast, while the resident cache still supports later exact-prefix reuse. |
| Full-state diagnostics | Not a production target | `FullState` can prove restore exactness, but it is not selected by serving policy and should not be benchmarked as a cache win/loss. |
| Split serving | Still needs its own benchmark | Single-stage numbers do not prove end-to-end split wins because activation transfer, stage scheduling, and upstream/downstream cache hits change the cost model. |

Before claiming a new family is faster than llama-server, run the reproducible
benchmark wrapper and keep the matched conditions fixed: same GGUF, backend,
context size, prompt tokens, generated tokens, slot/parallel settings, and
warm-cache policy. The README table is a serving-correctness table first;
performance wins are only promoted for rows whose matched speedup is above
1.0x.

## Module Map

| Module | Responsibility |
| --- | --- |
| `config` | candidate prefix lengths, record limits, resident cache sizing |
| `identity` | exact prefix hash and page ids |
| `resident` | resident prefix and activation LRU bookkeeping |
| `payload` | exact state payloads and BLAKE3 block dedupe |

## Testing

The crate is intentionally lightweight. It should test without building or
linking llama.cpp:

```bash
cargo test -p skippy-cache --lib
```

Supported-family cache safety is smoked through the topology capability
registry:

```bash
cargo test -p skippy-topology reviewed_supported_families_smoke --lib
```

That smoke does not execute model files. It verifies that every reviewed
supported family can be inferred, planned with `f16` serving wire, and emits the
expected recurrent/sticky, q8 validation, and sideband policy signals.

Server/runtime integration tests belong in `skippy-server`,
`skippy-runtime`, and `skippy-correctness`.
