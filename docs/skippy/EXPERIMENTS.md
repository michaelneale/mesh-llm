# Experiments

Keep the running `skippy-runtime` experiment log here: what we tried, what
happened, and what we should do next.

## Performance Experiment Card Format

Every performance benchmark result should be recorded as a Markdown experiment
card with this fixed shape. Use normalized units so cards can be compared at a
glance: latency and stage totals in seconds, traffic in MiB, KV as token
positions and token-layer cells, and telemetry as exact dropped/export-error
counts.

```markdown
### YYYY-MM-DD - Short Experiment Name

Run: `run-id`
Commit: `git-sha`
Decision: promote | keep experimental | abandon | repeat

Config:

| Field | Value |
| --- | --- |
| Model | `...` |
| Corpus | `...` |
| Requests | `...` |
| Hosts | `10.0.0.2,10.0.0.1,10.0.0.4,10.0.0.3` |
| Endpoints | `10.0.0.2=10.0.0.2,...` |
| Splits | `10,20,30` |
| GPU layers | `-1` |
| Chunk | `256` |
| Inflight / credit | `2 / 1` |
| Wire dtype | `f32` |
| Telemetry | `summary` |

Summary:

| Run | Wire P50 | Wire P95 | Wire P99 | TTFT P50 | TTFT P95 | TTFT P99 | Decode P50 | Decode P95 | Mean | Total tok/s | Telemetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `run-id` |  |  |  |  |  |  |  |  |  |  |  |

Stage Totals:

| Stage | Requests | Messages | Prefill msgs | Compute | Downstream wait | Forward write | Message elapsed | Max KV | Output MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage-0 |  |  |  |  |  |  |  |  |  |
| stage-1 |  |  |  |  |  |  |  |  |  |
| stage-2 |  |  |  |  |  |  |  |  |  |
| stage-3 |  |  |  |  |  |  |  |  |  |

KV / Activation:

| Metric | Value |
| --- | --- |
| KV P50/P95/P99/max |  |
| Max token-layer cells |  |
| Max activation payload |  |
| Stage output MiB |  |
| `ggml_metal_free` |  |
| GPU offload |  |

Notes:

- ...

Decision:

- **Decision:** ...
- **Reason:** ...
- **Next:** ...
```

Non-negotiable fields for current prefill/TTFT work:

- wire P50/P95/P99
- TTFT P50/P95/P99
- decode P50/P95
- mean elapsed
- total tok/s
- per-stage compute, downstream wait, forward write, and message elapsed
- max KV positions and max token-layer cells
- activation output MiB and max activation payload
- telemetry dropped/export-error counts
- GPU offload confirmation

## Prefill Latency Plan

Date: 2026-04-26

Scope:

- optimize prefill and TTFT only
- do not optimize decode in this pass
- use four separate stage hosts for all lab runs
- current lab runs use the Thunderbolt `10.x` fabric; pass literal `10.0.0.x`
  hosts and an explicit `--endpoint-host-map` so SSH, rsync, binary stage
  traffic, and OTLP stay off the Wi-Fi/LAN route
- use `--n-gpu-layers -1` for performance runs so each stage asks llama.cpp to
  offload all available layers for its slice
- use `--stage-telemetry-level summary` for performance runs; reserve
  `--stage-telemetry-level debug` for per-message firehose diagnosis
- stage layer ranges must be evenly balanced across those hosts; for Qwen3.6
  with 40 layers and four hosts, this means `10,20,30`
- keep model split sync simple: coordinator materializes stable stage GGUFs,
  then uses `rsync -az`

Reference material:

- `/Users/jdumay/code/skippy-runtime/docs/BENCHMARK_CORPUS.md`
- `/Users/jdumay/code/moe-experiments/research/PREFILL_PARALLELISM_LITERATURE.md`
- `/Users/jdumay/code/moe-experiments/experiments/streaming-kv-handoff/HISTORY.md`
- `/Users/jdumay/code/moe-experiments/experiments/streaming-kv-handoff/EXPERIMENT.md`
- prior Qwen3.6 corpus run:
  `/Volumes/External/skippy-runtime-bench/lab-m9-qwen36-corpus9-20260425-225212`

The prior Qwen3.6 corpus run is useful for diagnosis only. It used four stages
but not four distinct hosts, so it is not valid for future performance claims.
It showed 9 prompts, 72 generated tokens, prompt lengths from 20 to 335 tokens,
and elapsed request times from 3.311s to 33.034s. That shape is enough to say
prefill and long-prompt TTFT are the urgent problem, but all go-forward numbers
must be rerun with `shadowfax.local`, `black.local`, `studio54.local`, and
`build.local` as distinct stage hosts.

## Ideas To Try Now

### 1. Chunked Prefill

Hypothesis: splitting prompt prefill into bounded chunks lets upstream stages
start forwarding activation frames earlier and gives downstream stages useful
work sooner. This is most likely to help long prompts; short prompts may lose
because each chunk adds a wire frame and ACK.

Trial support added:

- `skippy-bench run --prefill-chunk-size N`
- `driver-result.json` records `prefill_chunk_size`
- chunking applies only to binary prefill frames; decode remains unchanged

Initial outcome:

- Accepted as a real experiment path.
- Existing `moe-experiments` data says chunking is not automatically good.
- `live-prefill-overlap-frontier-2026-04-21` showed tiny chunks were bad:
  `chunk_tokens=1` was much slower than no-overlap on both edge and internet
  profiles.
- In that same frontier, `chunk_tokens=64` beat no-overlap by about `10.4%` on
  the internet profile, but no-overlap was still best on the edge profile.
- Later `HISTORY.md` notes superseded that with a corrected 9-prompt
  shard-backed read where `chunk_tokens=128` beat `64` and stop-and-forward.
- Recommendation: start this repo's lab sweep with no chunking, `64`, and
  `128`; only try smaller chunks after a split-balanced run proves the long
  prompt is still bottlenecked by downstream idleness.

### 2. Prefill Inflight/Credit Sweep

Hypothesis: the binary protocol already supports deferred prefill ACKs. The
right credit window should reduce downstream wait time without letting memory
or socket buffering run away.

Trial support added:

- `skippy-bench run --stage-max-inflight N`
- `skippy-bench run --stage-reply-credit-limit N`
- these are passed to every `skippy-server serve-binary`
- the credit path only affects prefill ACK deferral; decode remains unchanged

Initial outcome:

- Accepted as the strongest historical protocol lead, but it is not a blank
  check.
- `moe-experiments` history recorded bounded-credit runtime drops from
  `2.397322s` to `1.351927s` local, `2.416006s` to `1.421874s` edge-WAN, and
  `2.857193s` to `1.834416s` internet-WAN on the 9-prompt benchmark.
- That same historical path was only exact `8/9`, so this repo must rerun it
  through the current correctness-safe binary path before adopting it.
- The source-buffer/constrained-credit smoke also showed that `1 / 0` credit
  was about `20.5%` slower than the `2 / 1` control on one prompt.
- Recommendation: use `--stage-max-inflight 2 --stage-reply-credit-limit 1` as
  the first live candidate, plus serialized `--stage-reply-credit-limit 0` as a
  control. Only widen to `4/4` or `8/7` if debug spans show credit starvation.

### 4. Stage Split Sweep

Hypothesis: uneven splits can hide the real bottleneck by placing too much
prefill on one node. Prefill is layer-heavy, so every lab run must use balanced
stage layer ranges before protocol or wire-format results are considered valid.

Trial support already existed:

- `skippy-bench run --splits ...`
- `skippy-bench` now rejects duplicate hosts, so each split maps to one
  stage on one machine

Initial outcome:

- Accepted as mandatory, then tightened into a hard rule: benchmark runs must
  be balanced across nodes. The previous duplicate-host corpus run is invalid
  for this question.
- Historical local split work says even layer counts were not best for Qwen3.6:
  `10/10/10/10` gave `8.81843s` staged prefill on long prompt `68335013`,
  `12/8/8/12` gave `8.40161s`, and `12/7/8/13` gave `8.30184s`.
- A follow-up `13/6/8/13` did not beat `12/7/8/13`.
- The historical uneven candidates are no longer valid for go-forward lab
  results. They can remain research references, but not production benchmark
  baselines.
- In this repo's split syntax the valid Qwen3.6 four-node baseline is
  `10,20,30`, producing `0..10 | 10..20 | 20..30 | 30..40`.
- `skippy-bench` now rejects stage ranges whose lengths differ by more
  than one layer.

### 5. Activation Wire Dtype Sweep

Hypothesis: boundary activation transfer may be significant for prefill. `f16`
or `q8` wire payloads can reduce transfer time, but they are only acceptable if
correctness stays inside the tolerance already used by `skippy-correctness`.

Trial support already existed:

- `skippy-bench run --activation-wire-dtype f32|f16|q8`
- debug spans include activation byte counts and forwarding/wait timing

Initial outcome:

- Accepted as a guarded experiment, not a default.
- Historical `q8` activation compression cut one boundary payload from
  `4096 B` per token for `fp16` to `2052 B`, about `49.9%` smaller.
- The `q8` smoke was correctness-positive once, and one smoke was `23.8%`
  faster than `fp16`, but a second smoke was `20.0%` slower and the full sweep
  was stopped because the scalar codec was too noisy.
- Recommendation: compare `f32` and `f16` in the current Rust runner first.
  Keep `q8` out of the main path until its pack/unpack is vectorized or the
  four-host run proves network transfer dominates codec overhead.

## Later

### 6. Context-Parallel Or Ring-Attention Prefill

Hypothesis: for very long single prompts, sharding sequence length instead of
only layers may be the real TTFT fix.

Outcome:

- Deferred. This likely needs llama-side attention changes and is much more
  invasive than the current C ABI surface.
- Revisit after the prefill-only lab sweep proves whether layer-pipeline
  chunking, credits, split balance, and activation dtype are enough.

## Recommended Course

Run the experiments in this order:

1. Establish a valid four-host baseline with no chunking, `f32`, balanced
   `10,20,30` split, and default credit settings.
2. Keep split balancing as a rule. Do not report uneven split results as lab
   benchmark wins, even if a local experiment looks better.
3. On the balanced split, try historical coarse overlap:
   `--prefill-chunk-size 128`
   with `--stage-max-inflight 2 --stage-reply-credit-limit 1`; compare against
   no chunking and serialized credit.
4. On the best balanced split/chunk/credit result, compare `f32` and `f16`.
   Do not put `q8` in the main matrix yet.

Current four-host results support the balanced split plus coarse chunking and a
small deferred-ACK window. `f16` helps P50 slightly, but did not beat the
chunk/credit run on the long-tail prompt in this corpus. If chunking only helps
the single long prompt and hurts the common case on a larger corpus, keep
balanced split and credit tuning, then move repeated-prefix or KV reuse into
the serving layer instead of forcing every request through chunk overhead.

Telemetry and GPU-offload rule:

- The mixed-corpus runs before `2026-04-26` were diagnostic CPU runs because
  `skippy-bench` defaulted `--n-gpu-layers` to `0`; their logs showed
  `offloading 0/41 layers to GPU`.
- The launcher now defaults `--n-gpu-layers -1`, and all go-forward performance
  commands should still pass it explicitly.
- Stage telemetry now has `off`, `summary`, and `debug` levels. Performance
  runs use `summary`, which preserves request-level stage totals and KV growth
  counters without sending every protocol message as an OTLP span. Debug runs
  keep the full per-message spans.

## Research Takeaways For Prefill

Review date: 2026-04-26

Sources reviewed:

- `/Users/jdumay/code/moe-experiments/research/PREFILL_PARALLELISM_LITERATURE.md`
- `/Users/jdumay/code/moe-experiments/research/STREAMING_KV_HANDOFF.md`
- `/Users/jdumay/code/moe-experiments/research/WAN_MOE_RESEARCH.md`

The most useful research direction for this runtime is to sharpen the current
activation-pipeline prefill path rather than replace it. The binary transport
already supports the key SARATHI-like primitive: split prompt prefill into
chunks, forward activation frames as each stage finishes, and bound outstanding
prefill replies with credit. The current best-known setting remains
`--prefill-chunk-size 128 --stage-max-inflight 2 --stage-reply-credit-limit 1`.

Near-term work:

- Add adaptive chunk sizing. Keep unchunked prefill for short prompts, use
  coarse chunks such as `128` for long prompts, and sweep the prompt-length
  threshold. The current fixed-size driver is useful for experiments, but it
  probably leaves short prompts paying avoidable frame and ACK overhead.
- Improve credit telemetry before widening inflight windows. The next useful
  spans are credit wait time, outstanding prefill count, downstream ACK drain
  time, and per-stage queue depth. Only widen beyond `2 / 1` if those counters
  show credit starvation rather than compute or network transfer dominating.
- Treat repeated-prefix and KV reuse as the biggest serving-layer opportunity.
  Hydragen, Preble-style prompt scheduling, and locality-aware scheduling avoid
  repeated prefill instead of making every request pay chunk overhead. This is
  especially relevant for system prompts, tool prefixes, agent loops, and
  multi-turn traffic.

Later work:

- Real KV handoff is promising but should stay behind explicit measurement.
  The streaming-KV note's core idea is sound: KV is built during prefill, so a
  future split prefill/decode system could transfer it incrementally instead of
  copying one large state blob after prefill. This runtime currently streams
  boundary activations, not raw KV, so the next step is llama-side KV byte and
  state import/export timing, not a topology rewrite.
- Prefill/decode disaggregation should be tested only with handoff accounting:
  `state_export_bytes`, `state_export_seconds`, `state_import_bytes`,
  `state_import_seconds`, `kv_attach_seconds`, TTFT, and TPOT. The WAN/MoE
  notes consistently warn that phase splitting loses if duplicated residency or
  KV movement eats the compute win.
- Context-parallel or ring-attention prefill is the long-prompt moonshot. It is
  relevant for very large single prompts, but it likely requires llama-side
  attention changes rather than Rust-stage protocol work.

## Results

### 2026-04-26 Four-Host Balanced Qwen3.6

Environment:

- model: `unsloth/Qwen3.6-35B-A3B-GGUF`
- corpus: `mt_bench_9bucket.jsonl`, 9 prompts
- generation: `--max-new-tokens 8`
- hosts: `build.local`, `studio54.local` (`192.168.86.38`), `shadowfax.local`,
  `black.local`
- split rule: balanced only, `10,20,30`
- sync: coordinator materialized once, then `rsync -az`

Important runner fix:

- The first four-host attempts exposed macOS Local Network Privacy behavior:
  detached `nohup` stage processes spawned from SSH reported `No route to host`
  for LAN destinations even while `nc`, Python, and Rust probes from the live
  shell succeeded.
- `skippy-bench` now keeps the SSH session alive for each remote stage
  process and starts stages downstream-first, waiting for each stage to log
  `listening` before launching the upstream stage.
- A conservative downstream connect fallback remains in `skippy-server`,
  but the real lab fix was avoiding detached SSH-launched stage servers.

Runs:

| Run | Variant | P50 | P95/P99 | Mean | Total | Tok/s | Spans | Telemetry loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lab-prefill-balanced-f32-20260426-062922` | f32, no chunk, default credit | 3.093s | 10.670s | 4.581s | 41.226s | 17.44 | 1016 | 0 dropped, 0 errors |
| `lab-prefill-balanced-chunk128-credit-20260426-064127` | f32, chunk 128, inflight 2, credit 1 | 3.005s | 9.540s | 4.430s | 39.870s | 18.03 | 1040 | 0 dropped, 0 errors |
| `lab-prefill-balanced-f16-20260426-064333` | f16, no chunk, default credit | 2.987s | 10.364s | 4.453s | 40.075s | 17.94 | 1016 | 0 dropped, 0 errors |
| `lab-prefill-balanced-kv2-chunk128-credit-20260426-071104` | f32, chunk 128, inflight 2, credit 1, KV telemetry | 4.162s | 8.706s | 4.722s | 42.497s | 16.92 | 1040 | 0 dropped, 0 errors |

Prompt elapsed times:

- f32 baseline: `3093, 2732, 5379, 6231, 3006, 2716, 2521, 10670, 4878` ms
- chunk/credit: `2969, 2750, 4994, 6216, 3005, 2740, 2863, 9540, 4793` ms
- f16: `2987, 2814, 4949, 5959, 2917, 2744, 2521, 10364, 4820` ms
- KV telemetry run: `4162, 3124, 5238, 6721, 3104, 3253, 3200, 8706, 4989` ms

KV growth notes from the KV telemetry run:

- `skippy-server` debug spans now include `kv_tokens_after`,
  `kv_layer_count`, and `kv_token_layer_cells`.
- With a balanced 10-layer stage, the long prompt reached 343 KV positions per
  stage after eight decode steps, or 3430 token-layer cells per stage.
- Across all four balanced stages that is 13720 token-layer cells for the
  longest request. The shortest requests reached 29 KV positions per stage, or
  1160 token-layer cells across the chain.
- These are token/layer growth counters, not exact llama KV byte counts. Exact
  KV bytes should come from a future llama ABI metric once exposed.

### 2026-04-26 Four-Host Mixed HF Serving Corpus

Run: `lab-prefill-balanced-mixed192-kv-chunk128-credit-20260426-071734`

Environment:

- corpus:
  `/Users/jdumay/code/moe-experiments/experiments/streaming-kv-handoff/results/hf-serving-corpus-2026-04-22/mixed_192.jsonl`
- prompt count: 192
- generation: `--max-new-tokens 8`
- context: `--ctx-size 4096`
- hosts: `build.local`, `studio54.local` (`192.168.86.38`), `shadowfax.local`,
  `black.local`
- split rule: balanced only, `10,20,30`
- prefill setting: `--prefill-chunk-size 128 --stage-max-inflight 2
  --stage-reply-credit-limit 1`

Overall result:

| Requests | Prompt tokens | Generated tokens | P50 | P95 | P99 | Mean | Total | Total tok/s | Generated tok/s | Spans | Telemetry loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 192 | 45341 | 1536 | 6.567s | 20.052s | 30.308s | 8.049s | 1545.344s | 30.33 | 0.99 | 24703 | 45 dropped/export errors |

Category result:

| Category | Count | Prompt tokens | Mean | P50 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coding | 2 | 43 | 3.196s | 3.219s | 3.219s | 3.219s |
| extraction | 3 | 879 | 8.583s | 8.464s | 9.419s | 9.419s |
| humanities | 3 | 50 | 2.841s | 2.737s | 3.091s | 3.091s |
| math | 3 | 117 | 4.613s | 4.694s | 5.728s | 5.728s |
| qa | 83 | 15789 | 7.351s | 7.147s | 9.186s | 11.079s |
| reasoning | 51 | 3072 | 5.250s | 5.318s | 6.458s | 7.885s |
| roleplay | 6 | 320 | 4.990s | 5.424s | 5.818s | 5.818s |
| stem | 4 | 142 | 4.125s | 5.004s | 5.655s | 5.655s |
| summarization | 33 | 24501 | 16.511s | 14.992s | 30.308s | 34.792s |
| writing | 4 | 236 | 5.421s | 5.555s | 5.897s | 5.897s |

KV growth:

- longest prompt: 1927 prefill tokens, 1935 final KV positions after eight
  decode steps
- per balanced 10-layer stage: 19350 token-layer cells
- across the full 40-layer chain: 77400 token-layer cells
- max activation payload observed per stage boundary remained 1048576 bytes
  because prefill was chunked at 128 tokens with f32 activations

Telemetry caveat:

- `driver-result.json` completed all 192 prompts and is the source of timing
  truth.
- `report.json` recorded 45 dropped/export-error telemetry events, so stage
  timing sums are useful directional diagnostics but not a lossless trace.
- Follow-up fix: stage telemetry now batches OTLP span exports, retries failed
  exports through a bounded in-memory replay buffer, uses a 10s off-path export
  timeout, emits stable span IDs across retries so metrics-server can dedupe
  `INSERT OR REPLACE` span writes, and exposes
  `skippy-bench run --stage-telemetry-queue-capacity` for larger debug
  corpus runs. This keeps telemetry non-blocking while making transient
  collector stalls much less likely to lose spans.
- Remote startup is now also fail-fast: if a stage exits before logging its
  binary listener, `skippy-bench` includes the remote log tail in the error
  and terminates any stages it already launched. This prevents stale stage
  servers from poisoning the next run with occupied ports.

Clean replay after telemetry fix:

- run:
  `lab-prefill-balanced-mixed192-ttft3-chunk128-credit-20260426-083900`
- same corpus, hosts, balanced split, `f32`, `--prefill-chunk-size 128`,
  `--stage-max-inflight 2`, `--stage-reply-credit-limit 1`
- telemetry loss: `0 dropped, 0 export errors`
- spans: `24703`

Overall timing:

| Requests | Prompt tokens | Generated tokens | Wire P50 | Wire P95 | Wire P99 | TTFT P50 | TTFT P95 | Decode P50 | Elapsed mean | Total tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 45341 | 1536 | 6.597s | 19.016s | 28.059s | 5.133s | 17.346s | 5.132s | 7.946s | 30.73 |

Driver timing notes:

- `elapsed_ms` still includes driver overhead such as tokenization and prompt
  setup. `wire_elapsed_ms` starts when the driver connects to the first binary
  stage.
- `prefill_elapsed_ms` is client-side prefill submit time, not full remote
  prefill completion. Use TTFT and stage spans for prefill-path decisions.
- Prompt length still explains nearly all request time:
  `wire_elapsed_ms = 4054.7 + 16.004 * prefill_tokens`, `R^2=0.982`.
- TTFT has the same shape:
  `ttft_ms = 2556.7 + 15.939 * prefill_tokens`, `R^2=0.985`.
- Decode is mostly independent of prompt length in this run:
  `decode_elapsed_ms = 4902.7 + 0.672 * prefill_tokens`, `R^2=0.048`.

Token-bin timing:

| Prefill tokens | Count | Wire mean | Wire P95 | TTFT mean | TTFT P95 | Decode mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-64 | 53 | 4.118s | 5.023s | 2.665s | 3.580s | 4.097s |
| 65-128 | 33 | 5.739s | 6.597s | 4.177s | 4.860s | 5.713s |
| 129-256 | 63 | 7.323s | 8.593s | 5.826s | 6.976s | 5.354s |
| 257-512 | 21 | 10.010s | 12.373s | 8.403s | 10.364s | 5.345s |
| 513-1024 | 15 | 15.881s | 19.770s | 14.345s | 18.365s | 5.356s |
| 1025-2048 | 7 | 26.230s | 34.328s | 24.679s | 32.850s | 5.157s |

Category timing:

| Category | Count | Prompt tokens | Wire mean | TTFT mean | Decode mean | Wire max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coding | 2 | 43 | 2.768s | 1.288s | 2.749s | 2.820s |
| extraction | 3 | 879 | 8.441s | 6.891s | 5.544s | 9.773s |
| humanities | 3 | 50 | 2.722s | 1.269s | 2.701s | 2.857s |
| math | 3 | 117 | 3.916s | 2.354s | 3.895s | 5.321s |
| qa | 83 | 15789 | 7.417s | 5.905s | 5.442s | 11.878s |
| reasoning | 51 | 3072 | 4.811s | 3.309s | 4.745s | 8.020s |
| roleplay | 6 | 320 | 4.594s | 3.154s | 4.572s | 5.900s |
| stem | 4 | 142 | 3.686s | 2.232s | 3.666s | 5.023s |
| summarization | 33 | 24501 | 15.985s | 14.427s | 5.290s | 34.328s |
| writing | 4 | 236 | 4.875s | 3.379s | 4.850s | 5.660s |

Stage timing from clean spans:

| Kind | Stage | Spans | Compute | Downstream wait | Forward write | Message elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | stage-0 | 458 | 648.162s | 2.888s | 60.424s | 711.618s |
| Prefill | stage-1 | 458 | 271.129s | 0.737s | 26.612s | 298.710s |
| Prefill | stage-2 | 458 | 395.626s | 0.155s | 23.604s | 419.615s |
| Prefill | stage-3 | 458 | 333.156s | 0.000s | 0.000s | 333.370s |
| Decode | stage-0 | 1536 | 53.725s | 702.868s | 0.170s | 757.146s |
| Decode | stage-1 | 1536 | 41.134s | 461.811s | 0.202s | 503.738s |
| Decode | stage-2 | 1535 | 42.169s | 216.382s | 0.258s | 259.378s |
| Decode | stage-3 | 1536 | 57.059s | 0.000s | 0.000s | 57.434s |

KV growth from clean spans:

- longest prompt: `1927` prefill tokens, `1935` final KV positions after eight
  decode steps
- per balanced 10-layer stage: `19350` token-layer cells
- across the full 40-layer chain: `77400` token-layer cells
- max f32 prefill boundary activation payload: `1048576` bytes
- max f32 decode boundary activation payload: `8192` bytes

GPU-offload summary telemetry replay:

- run:
  `lab-prefill-balanced-mixed192-gpu-summary-chunk256-credit-20260426-113719`
- same corpus, four hosts, balanced split, `f32`, `--prefill-chunk-size 256`,
  `--stage-max-inflight 2`, `--stage-reply-credit-limit 1`
- explicit performance flags: `--n-gpu-layers -1`,
  `--stage-telemetry-level summary`
- GPU verification: stage logs showed Metal devices and `offloaded 41/41
  layers to GPU`
- telemetry loss: `0 dropped, 0 export errors`
- spans: `772` total, down from roughly `22k-25k` debug spans for the same
  corpus shape

Overall timing:

| Requests | Prompt tokens | Generated tokens | Wire P50 | Wire P95 | Wire P99 | TTFT P50 | TTFT P95 | Decode P50 | Elapsed mean | Total tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 45341 | 1536 | 3.649s | 11.030s | 16.892s | 2.472s | 9.707s | 3.176s | 4.381s | 55.72 |

Stage request summaries:

| Stage | Requests | Messages | Compute | Downstream wait | Forward write | Message elapsed | Max KV positions | Max token-layer cells | Activation out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage-0 | 192 | 1819 | 283.665s | 428.759s | 76.686s | 789.513s | 1935 | 19350 | 364.7 MiB |
| stage-1 | 192 | 1819 | 55.221s | 250.875s | 31.699s | 338.765s | 1935 | 19350 | 364.7 MiB |
| stage-2 | 192 | 1819 | 107.045s | 123.064s | 37.845s | 268.608s | 1935 | 19350 | 364.7 MiB |
| stage-3 | 192 | 1819 | 97.663s | 0.000s | 0.000s | 98.310s | 1935 | 19350 | 0.0 MiB |

KV growth from summary spans:

- longest prompt: `1927` prefill tokens, `1935` final KV positions after eight
  decode steps
- prompt-level KV distribution: P50 `158.5`, P95 `938`, P99 `1591`, max
  `1935` positions
- per balanced 10-layer stage max: `19350` token-layer cells
- across the full 40-layer chain max: `77400` token-layer cells
- summary spans preserve the KV growth data needed for benchmark reporting
  while avoiding the debug OTLP firehose

GPU replay recommendation:

- Treat the older mixed-corpus CPU runs as diagnostics only. The go-forward
  baseline is GPU offload with `--n-gpu-layers -1`.
- Summary telemetry is the right default for performance. It cut collector
  traffic by more than an order of magnitude and still preserved stage timing,
  activation byte totals, credit counters, and KV growth.
- `chunk256` on GPU materially improved the mixed-corpus P50/P95/P99 versus
  the earlier CPU/debug `chunk128`/`chunk256` runs. Next compare GPU
  `chunk128`, `chunk256`, and `chunk512` under summary telemetry before
  changing decode or adding more protocol complexity.

GPU chunk sweep:

| Run | Chunk | Requests | Spans | Wire P50 | Wire P95 | Wire P99 | TTFT P50 | TTFT P95 | TTFT P99 | Decode P50 | Decode P95 | Mean | Total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lab-prefill-balanced-mixed192-gpu-summary-chunk128-credit-20260426-121651` | 128 | 192 | 772 | 3.669s | 13.715s | 21.991s | 2.405s | 12.312s | 20.545s | 2.631s | 3.439s | 4.813s | 50.73 |
| `lab-prefill-balanced-mixed192-gpu-summary-chunk256-credit-20260426-113719` | 256 | 192 | 772 | 3.649s | 11.030s | 16.892s | 2.472s | 9.707s | 15.534s | 3.176s | 4.578s | 4.381s | 55.72 |
| `lab-prefill-balanced-mixed192-gpu-summary-chunk512-credit-20260426-123325` | 512 | 192 | 772 | 3.754s | 9.663s | 16.593s | 2.527s | 8.362s | 15.395s | 3.707s | 6.389s | 4.486s | 54.43 |
| `lab-prefill-balanced-mixed192-gpu-summary-adaptive256-512-credit-20260426-125210` | 256, then 512 at 513+ tokens | 192 | 772 | 4.243s | 9.944s | 16.438s | 2.785s | 8.241s | 14.910s | 3.851s | 5.605s | 5.076s | 48.10 |

Chunk sweep stage totals:

| Chunk | Stage | Messages | Prefill messages | Compute | Downstream wait | Forward write | Message elapsed | Credit waits |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | stage-0 | 1994 | 458 | 433.873s | 371.951s | 63.797s | 870.053s | 266 |
| 128 | stage-1 | 1994 | 458 | 62.080s | 225.709s | 28.087s | 316.931s | 266 |
| 128 | stage-2 | 1994 | 458 | 121.728s | 116.534s | 39.577s | 278.602s | 266 |
| 128 | stage-3 | 1994 | 458 | 96.500s | 0.000s | 0.000s | 97.171s | 0 |
| 256 | stage-0 | 1819 | 283 | 283.665s | 428.759s | 76.686s | 789.513s | 91 |
| 256 | stage-1 | 1819 | 283 | 55.221s | 250.875s | 31.699s | 338.765s | 91 |
| 256 | stage-2 | 1819 | 283 | 107.045s | 123.064s | 37.845s | 268.608s | 91 |
| 256 | stage-3 | 1819 | 283 | 97.663s | 0.000s | 0.000s | 98.310s | 0 |
| 512 | stage-0 | 1759 | 223 | 193.359s | 500.149s | 112.018s | 805.925s | 31 |
| 512 | stage-1 | 1759 | 223 | 56.726s | 270.727s | 34.144s | 362.632s | 31 |
| 512 | stage-2 | 1759 | 223 | 101.668s | 127.384s | 47.803s | 277.533s | 31 |
| 512 | stage-3 | 1759 | 223 | 101.241s | 0.000s | 0.000s | 101.916s | 0 |

Chunk sweep interpretation:

- `chunk128` is no longer attractive on GPU for this corpus. It creates the
  most prefill messages, the most credit waits, and the worst P95/P99.
- `chunk256` is the best throughput/default setting in this sweep. It has the
  best mean elapsed time and total token throughput.
- `chunk512` is the long-tail TTFT candidate. It reduces TTFT P95 from
  `9.707s` to `8.362s` and TTFT P99 from `15.534s` to `15.395s` versus
  `chunk256`, but decode P50/P95 and mean elapsed get worse.
- Adaptive `256 -> 512` chunking with `--prefill-chunk-schedule 513:512` did
  not win on the first run. It improved the longest-prompt TTFT tail slightly
  versus fixed `512`, but regressed the common case enough that mean elapsed
  increased to `5.076s` and total throughput fell to `48.10` tok/s. The run
  showed extra Metal pipeline compilation during the corpus, so repeat before
  drawing a final statistical conclusion; do not promote it yet.
- The stage summaries suggest the trade-off is now between stage-0 chunk
  compute/frame overhead and downstream idleness. Larger chunks reduce stage-0
  compute and credit churn, but increase downstream wait and forward-write
  pressure.
- Recommendation: keep `chunk256` as the default performance setting. Use
  `chunk512` only if the product target prioritizes long-tail TTFT over mean
  throughput. Do not optimize decode yet. The next implementation target should
  be reducing per-request llama/Metal context setup and teardown, because the
  stage logs show repeated context reservation/free cycles across the corpus.

Session reuse ABI:

- run:
  `lab-session-reuse-mixed192-gpu-summary-chunk256-20260426-132829`
- change: added `skippy_session_reset` to the stage ABI and reused idle
  `StageSession` handles inside `skippy-server` instead of freeing the
  llama context at request end
- same corpus, four hosts, balanced split, `f32`, `--prefill-chunk-size 256`,
  `--stage-max-inflight 2`, `--stage-reply-credit-limit 1`,
  `--n-gpu-layers -1`, and `--stage-telemetry-level summary`
- telemetry loss: `0 dropped, 0 export errors`
- stage logs: `ggml_metal_free=0` on all four stages for the run; stage 1 and
  stage 2 each processed `283` prefill chunks, confirming the benchmark path
  exercised the reused sessions

Comparison against the previous GPU summary `chunk256` baseline:

| Run | Wire P50 | Wire P95 | Wire P99 | TTFT P50 | TTFT P95 | TTFT P99 | Decode P50 | Decode P95 | Mean | Total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `chunk256` | 3.649s | 11.030s | 16.892s | 2.472s | 9.707s | 15.534s | 3.176s | 4.578s | 4.381s | 55.72 |
| session reuse `chunk256` | 3.969s | 11.046s | 15.757s | 2.440s | 9.475s | 14.326s | 3.807s | 6.041s | 4.993s | 48.90 |

Session reuse stage request summaries:

| Stage | Requests | Messages | Compute | Downstream wait | Forward write | Message elapsed | Max KV positions | Max token-layer cells | Activation out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage-0 | 192 | 1819 | 241.398s | 527.549s | 103.774s | 873.134s | 1935 | 19350 | 364.7 MiB |
| stage-1 | 192 | 1819 | 43.499s | 305.969s | 36.248s | 386.424s | 1935 | 19350 | 364.7 MiB |
| stage-2 | 192 | 1819 | 97.098s | 161.774s | 59.871s | 319.499s | 1935 | 19350 | 364.7 MiB |
| stage-3 | 192 | 1819 | 80.946s | 0.000s | 0.000s | 81.663s | 1935 | 19350 | 0.0 MiB |

Session reuse interpretation:

- The ABI and server pooling are mechanically successful: contexts are reset
  without Metal teardown during the corpus.
- The benchmark did not produce a throughput win. Mean elapsed regressed from
  `4.381s` to `4.993s`, total tok/s fell from `55.72` to `48.90`, and decode
  P50/P95 got worse. The TTFT tail improved slightly, with TTFT P99 falling
  from `15.534s` to `14.326s`.
- KV growth remained bounded by prompt length, not by request count: the max
  request still reached `1935` KV positions per stage, matching the longest
  `1927` token prompt plus eight decode tokens.
- Recommendation: keep session reset/pooling because it removes avoidable
  context churn and is the right production server shape, but do not count it
  as the next prefill optimization. The next prefill work should target
  stage-0 bottleneck and boundary pressure: activation wire dtype and
  llama-side timing for reset/clear versus fresh session creation.

### 2026-04-26 Async Activation Forwarding

Run: `lab-async-prefill-forward-long600-gpu-summary-chunk256-20260426-145255`
Commit: `1672867` plus local async-forwarding experiment patch
Decision: abandon

Config:

| Field | Value |
| --- | --- |
| Model | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Corpus | synthetic 600-token prefill, 1 decode token |
| Requests | `1` |
| Hosts | `build.local,192.168.86.38,shadowfax.local,black.local` |
| Splits | `10,20,30` |
| GPU layers | `-1` |
| Chunk | `256` |
| Inflight / credit | `2 / 1` |
| Wire dtype | `f32` |
| Telemetry | `summary` |

Summary:

| Run | Wire/TTFT | Prefill submit | Decode | Mean/elapsed | Total tok/s | Telemetry |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| async prefill-forward `long600` | 6.250s | 3.672s | 2.577s | 7.432s | 80.87 | 0 dropped, 0 errors |
| control `long600` | 5.947s | 3.400s | 2.546s | 7.225s | 83.18 | 0 dropped, 0 errors |

Stage Totals:

| Run | Stage | Messages | Prefill msgs | Compute | Downstream wait | Forward write | Message elapsed | Max KV | Output MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| async | stage-0 | 4 | 3 | 0.762s | 3.867s | 2.118s | 6.139s | 600 | 4.688 |
| async | stage-1 | 4 | 3 | 0.429s | 1.478s | 0.420s | 1.970s | 600 | 4.688 |
| async | stage-2 | 4 | 3 | 0.794s | 0.671s | 1.939s | 1.720s | 600 | 4.688 |
| async | stage-3 | 4 | 3 | 0.506s | 0.000s | 0.000s | 0.509s | 600 | 0.000 |
| control | stage-0 | 4 | 3 | 0.763s | 3.419s | 1.550s | 5.733s | 600 | 4.688 |
| control | stage-1 | 4 | 3 | 0.377s | 1.678s | 0.240s | 2.303s | 600 | 4.688 |
| control | stage-2 | 4 | 3 | 0.800s | 0.394s | 1.551s | 2.748s | 600 | 4.688 |
| control | stage-3 | 4 | 3 | 0.320s | 0.000s | 0.000s | 0.322s | 600 | 0.000 |

Notes:

- The tested design cloned the downstream TCP stream and moved eligible
  non-final prefill activation writes onto a bounded writer thread. Final
  prefill and decode replies stayed synchronous.
- The targeted long-prefill A/B is the relevant test because it emits multiple
  prefill chunks and should have exposed transfer masking if this design helped.
- It regressed wire/TTFT from `5.947s` to `6.250s`, elapsed from `7.225s` to
  `7.432s`, and stage-0 forward write from `1.550s` to `2.118s`.
- A 16-prompt smoke also regressed: async prefill-forward mean elapsed
  `17.169s` versus `12.673s` for the same-shape control, with worse P95/P99.

Decision:

- **Decision:** abandon this async writer-thread shape.
- **Reason:** it added queueing/backpressure overhead and did not mask boundary
  transfer under compute.
- **Next:** keep the synchronous baseline and test `f16` activation wire format
  under the GPU summary `chunk256` setup. If overlap is revisited later, it
  should be a protocol-level fragmentation/socket experiment, not a full-frame
  writer thread around a cloned stream.

### 2026-04-26 Thunderbolt Async Prefill Forwarding

Runs:

- `lab-thunderbolt10-baseline2-mixed192-gpu-summary-chunk256-credit-20260426`
- `lab-thunderbolt10-async-forward-mixed192-gpu-summary-chunk256-credit-20260426`
- `lab-thunderbolt10-async-forward-f16-mixed192-gpu-summary-chunk256-credit-20260426`
- `lab-thunderbolt10-async-forward-f32-mixed192-gpu-summary-chunk512-credit-20260426`

Decision: promote as current Thunderbolt candidate

Config:

| Field | Value |
| --- | --- |
| Model | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Corpus | `mixed_192.jsonl` |
| Requests | `192` |
| Hosts | `10.0.0.2,10.0.0.1,10.0.0.4,10.0.0.3` |
| Splits | `10,20,30` |
| GPU layers | `-1` |
| Chunk | `256` |
| Inflight / credit | `2 / 1` |
| Telemetry | `summary` |

Summary:

| Run | Wire P50 | Wire P95 | Wire P99 | TTFT P50 | TTFT P95 | TTFT P99 | Decode P50 | Decode P95 | Mean | Total tok/s | Telemetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sync `f32` baseline | 2.228s | 8.004s | 11.985s | 1.473s | 7.226s | 11.167s | 2.149s | 3.141s | 2.867s | 85.17 | 0 dropped, 0 errors |
| async `f32` | 1.883s | 8.234s | 13.443s | 1.196s | 7.457s | 12.539s | 1.841s | 3.156s | 2.633s | 92.74 | 0 dropped, 0 errors |
| async `f16` | 2.253s | 8.229s | 13.240s | 1.539s | 7.441s | 12.392s | 2.100s | 3.304s | 2.858s | 85.42 | 0 dropped, 0 errors |
| async `f32` chunk512 | 1.977s | 6.700s | 10.602s | 1.308s | 5.976s | 9.784s | 1.971s | 3.682s | 2.483s | 98.31 | 0 dropped, 0 errors |

Stage Totals:

| Run | Stage | Requests | Messages | Prefill msgs | Compute | Downstream wait | Forward write | Message elapsed | Max KV | Output MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sync `f32` | stage-0 | 192 | 1819 | 283 | 378.055s | 167.963s | 1.052s | 547.398s | 1935 | 364.7 |
| sync `f32` | stage-1 | 192 | 1819 | 283 | 41.041s | 144.811s | 2.627s | 189.292s | 1935 | 364.7 |
| sync `f32` | stage-2 | 192 | 1819 | 283 | 93.883s | 76.430s | 1.879s | 172.827s | 1935 | 364.7 |
| sync `f32` | stage-3 | 192 | 1819 | 283 | 76.705s | 0.000s | 0.000s | 77.334s | 1935 | 0.0 |
| async `f32` | stage-0 | 192 | 1819 | 283 | 334.658s | 167.035s | 0.240s | 502.256s | 1935 | 364.7 |
| async `f32` | stage-1 | 192 | 1819 | 283 | 40.493s | 142.798s | 0.761s | 184.833s | 1935 | 364.7 |
| async `f32` | stage-2 | 192 | 1819 | 283 | 93.715s | 74.918s | 0.417s | 169.709s | 1935 | 364.7 |
| async `f32` | stage-3 | 192 | 1819 | 283 | 75.125s | 0.000s | 0.000s | 75.795s | 1935 | 0.0 |
| async `f16` | stage-0 | 192 | 1819 | 283 | 304.723s | 213.740s | 0.277s | 545.527s | 1935 | 364.7 |
| async `f16` | stage-1 | 192 | 1819 | 283 | 37.639s | 173.562s | 0.689s | 241.598s | 1935 | 364.7 |
| async `f16` | stage-2 | 192 | 1819 | 283 | 91.058s | 84.830s | 0.393s | 209.784s | 1935 | 364.7 |
| async `f16` | stage-3 | 192 | 1819 | 283 | 75.153s | 0.000s | 0.000s | 92.498s | 1935 | 0.0 |
| async `f32` chunk512 | stage-0 | 192 | 1759 | 223 | 295.776s | 177.022s | 0.234s | 473.357s | 1935 | 364.7 |
| async `f32` chunk512 | stage-1 | 192 | 1759 | 223 | 37.232s | 150.949s | 0.756s | 189.726s | 1935 | 364.7 |
| async `f32` chunk512 | stage-2 | 192 | 1759 | 223 | 87.000s | 77.190s | 0.412s | 165.277s | 1935 | 364.7 |
| async `f32` chunk512 | stage-3 | 192 | 1759 | 223 | 76.771s | 0.000s | 0.000s | 77.442s | 1935 | 0.0 |

Notes:

- Literal `10.0.0.x` hosts were used because hostnames were observed to choose
  mixed routes on this machine.
- `--async-prefill-forward` is now an opt-in server flag, exposed from
  `skippy-bench` as `--stage-async-prefill-forward`.
- Async `f32` improved mean elapsed by `8.2%`, total token throughput by
  `8.9%`, wire P50 by `15.5%`, and TTFT P50 by `18.8%` versus the sync
  Thunderbolt baseline.
- Async `f32` regressed P95/P99, so it is not yet a production default.
- Async `f16` slightly improved async tail versus async `f32`, but gave back
  the common-case and throughput gains. It is not the next promote candidate.
- Async `f32` with `chunk512` reduced prefill messages from `283` to `223` per
  stage and credit waits from `91` to `31`, which fixed the `chunk256` tail
  regression while further improving mean throughput.
- Versus sync `f32` `chunk256`, async `f32` `chunk512` improved mean elapsed by
  `13.4%`, wire P95 by `16.3%`, wire P99 by `11.5%`, TTFT P95 by `17.3%`, and
  total token throughput by `15.4%`.

Decision:

- **Decision:** use async `f32` `chunk512` as the current Thunderbolt candidate.
- **Reason:** it improves mean, throughput, and tail versus the sync baseline
  and versus async `chunk256`.
- **Next:** repeat once for variance, then test adaptive `256 -> 512` only if
  short-prompt P50 becomes more important than the current tail win.

Recommendation:

- Keep `10,20,30` as the only valid Qwen3.6 four-node split.
- Promote `--prefill-chunk-size 512 --stage-max-inflight 2
  --stage-reply-credit-limit 1 --stage-async-prefill-forward` as the current
  Thunderbolt prefill setting.
- Use `f16` as the default exact activation wire dtype. It did not beat async
  `f32` on this Thunderbolt prefill run, but later correctness work showed q8
  is family-specific while f16 is the conservative payload reduction.
- Keep `--stage-async-prefill-forward` opt-in until repeated, but the current
  Thunderbolt candidate is async `f32` with `chunk512`.
- Do not optimize decode yet.
- Next optimisation work should target prefill/TTFT only:
  1. repeat async `f32` `chunk512` once for variance;
  2. add llama-side timing around session reset/clear and fresh session
     creation so we know whether reset is adding latency or just exposing
     normal run-to-run variance;
  3. add credit-window telemetry before widening beyond `2 / 1`.

## Command Template

```bash
cargo run -p skippy-bench -- run \
  --hosts 10.0.0.2,10.0.0.1,10.0.0.4,10.0.0.3 \
  --endpoint-host-map 10.0.0.2=10.0.0.2,10.0.0.1=10.0.0.1,10.0.0.4=10.0.0.4,10.0.0.3=10.0.0.3 \
  --metrics-otlp-grpc-addr 0.0.0.0:14317 \
  --metrics-otlp-grpc-url http://10.0.0.1:14317 \
  --stage-load-mode layer-package \
  --stage-model /path/to/model-package \
  --model-id unsloth/Qwen3.6-35B-A3B-GGUF \
  --ctx-size 512 \
  --layer-end 40 \
  --splits 10,20,30 \
  --n-gpu-layers -1 \
  --prompt-corpus /Users/jdumay/code/moe-experiments/experiments/streaming-kv-handoff/results/same-topology-summary-benchmark-2026-04-21/mt_bench_9bucket.jsonl \
  --max-new-tokens 8 \
  --prefill-chunk-size 512 \
  --stage-max-inflight 2 \
  --stage-reply-credit-limit 1 \
  --stage-async-prefill-forward \
  --stage-telemetry-queue-capacity 8192 \
  --stage-telemetry-level summary \
  --activation-wire-dtype f16 \
  --rsync-model-artifacts \
  --execute-remote
```
