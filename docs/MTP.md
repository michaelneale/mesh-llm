# MTP Skippy Prototype

This note tracks the prototype for Multi-Token Prediction (MTP) with Skippy
staged execution.

The target model is:

```text
unsloth/Qwen3.6-27B-MTP-GGUF
Qwen3.6-27B-IQ4_XS.gguf
```

`IQ4_XS` is the smallest 4-bit quant in that repository. The Hub reports it at
roughly 15.7 GB, which is a safer local default than `UD-Q4_K_XL` at roughly
17.9 GB on this Mac. `UD-Q4_K_XL` remains the higher-quality 4-bit option, but
it hit Metal out-of-memory in the one-stage smoke with full GPU offload.

## Experiment Matrix

The desired matrix is:

| Mode | Runtime | MTP | Status |
|---|---|---:|---|
| `one-stage-baseline` | Skippy, one local stage | No | Runnable |
| `one-stage-mtp` | Skippy, one local stage | Yes | Runnable |
| `two-stage-baseline` | Skippy, two nodes when `MTP_REMOTE_STAGE1_SSH` is set | No | Runnable |
| `two-stage-mtp` | Skippy, two nodes when `MTP_REMOTE_STAGE1_SSH` is set | Yes | Runnable |

The patch queue carries the llama.cpp Qwen MTP loader/graph changes and the
Skippy ABI in `include/skippy/mtp.h`, exposed to Rust through `skippy-ffi` and
`skippy-runtime`. The OpenAI generation path can enable MTP with
`--openai-mtp`; final-stage serving captures `h_pre_norm`, drafts with the
sibling MTP head, and verifies proposals through Skippy's existing staged
`VerifySpan` rollback path. MTP is not a separate draft GGUF pair; it needs a
target GGUF with trained `nextn`/MTP tensors and runtime access to the target
pre-final-norm hidden state.

## Run

From the repo root:

```bash
scripts/run-mtp-skippy-experiment.sh --run all
```

The script downloads the GGUF into:

```text
experiments/mtp/models/
```

It writes combined run JSONs to:

```text
experiments/mtp/*.json
```

Each completed baseline JSON contains:

- the metrics-server `report.json`
- the `skippy-bench chat-corpus` driver result
- the run mode and run id

The script also keeps per-run logs and the metrics DB under:

```text
experiments/mtp/mtp-skippy-<mode>-<timestamp>/
```

Useful overrides:

```bash
MTP_PROMPT_LIMIT=4 \
MTP_MAX_TOKENS=64 \
MTP_CTX_SIZE=8192 \
MTP_ACTIVATION_WIDTH=2048 \
MTP_WINDOW=3 \
scripts/run-mtp-skippy-experiment.sh --run one-stage-baseline,one-stage-mtp,two-stage-baseline,two-stage-mtp
```

`MTP_WINDOW` maps to the same basic proposal-length knob as llama.cpp's
`--spec-draft-n-max`. The upstream MTP PR reports results for windows 3 and 2,
so those are the first two windows to compare before trying larger speculative
windows.

## Vanilla llama.cpp Control

Before attributing one-stage Skippy results to Skippy's staged runtime, run the
same GGUF through the upstream MTP PR's vanilla `llama-server`. On
`studio54.local`, PR `22673` at `ebe4fca4b` was built with Metal and tested
against the same 8-prompt smoke corpus at concurrency 1:

| Runtime | Max tokens | Spec args | Mean latency | Completion tok/s | Signal |
|---|---:|---|---:|---:|---|
| vanilla `llama-server` | 32 | `--spec-type none` | 2200.13 ms | 14.54 | baseline |
| vanilla `llama-server` | 32 | `--spec-type mtp --spec-draft-n-max 2 --spec-draft-n-min 1` | 2157.77 ms | 14.83 | +2.0% tok/s |
| vanilla `llama-server` | 32 | `--spec-type mtp --spec-draft-n-max 3 --spec-draft-n-min 1` | 2014.65 ms | 15.88 | +9.2% tok/s |
| vanilla `llama-server` | 128 | `--spec-type none` | 7446.32 ms | 17.19 | baseline |
| vanilla `llama-server` | 128 | `--spec-type mtp --spec-draft-n-max 2 --spec-draft-n-min 1` | 7606.02 ms | 16.83 | -2.1% tok/s |
| vanilla `llama-server` | 128 | `--spec-type mtp --spec-draft-n-max 3 --spec-draft-n-min 1` | 7253.26 ms | 17.65 | +2.7% tok/s |

The local control says MTP can win on this model, but the gain is window- and
workload-sensitive. Window 3 is the useful comparison point for Skippy; window
2 is not reliably positive here.

Raw control JSONs are stored under:

```text
experiments/mtp/vanilla/studio54/
```

## Current Skippy Comparison

After updating against main, the first refreshed Skippy MTP run failed with
`MTP decode failed`. The native log showed M-RoPE position validation rejecting
the MTP batch because the proposal head was asked to decode at an already
resident position:

```text
last position stored ... X = 56
input batch starting position ... Y = 56
for M-RoPE, it is required that the position satisfies: X < Y
```

The Skippy OpenAI path now keeps a synchronized sibling MTP context instead of
drafting statelessly from one hidden row. It also delays MTP verification ingest
until after `VerifySpan` classification, so early rejects do not poison the MTP
context before repair. With those fixes, the refreshed 32-token window-3
comparison on `studio54.local` is:

| Runtime | MTP | Mean latency | Completion tok/s | Result |
|---|---:|---:|---:|---|
| vanilla `llama-server` | No | 2200.13 ms | 14.54 | baseline |
| vanilla `llama-server` | Yes, window 3 | 2014.65 ms | 15.88 | +9.2% over vanilla baseline |
| Skippy one-stage | No | 2187.16 ms | 14.63 | matches vanilla baseline |
| Skippy one-stage | Yes, window 3 | 2424.10 ms | 13.20 | -9.8% vs Skippy baseline |

That narrows the issue: vanilla MTP is a real win for this short smoke workload,
and Skippy's non-MTP path is not the bottleneck. The remaining gap is in the
Skippy MTP proposal/verify path, not in the model, quant, prompt corpus, or
single-stage baseline.

The latest Skippy MTP run completes without stream errors and confirms the
metrics-server speculation aggregate is populated:

```text
windows=32, proposed_tokens=77, accepted_tokens=70, rejected_tokens=5,
accept_rate=0.909, draft_propose_ms=1148.3, primary_verify_elapsed_ms=3530.6
```

The acceptance rate is now in a plausible range, but Skippy still loses because
the separate proposal plus verification path costs more than it saves on this
short smoke workload. The next optimization target is not correctness; it is
reducing `mtp_propose_ms` and `primary_verify_elapsed_ms`, or batching/reusing
the MTP/verify work more like vanilla llama-server's in-process speculative
loop.

### Fused Local MTP Fast Path

A fused one-stage native fast path was added to test whether the Rust-side
speculative contract was the dominant overhead. The new ABI call performs the
target decode, MTP sibling ingest, MTP proposal, target verification,
classification, and repair decision inside `skippy.cpp`, returning committed
tokens plus native counters to `openai.rs`.

On the same 8-prompt, 32-token, window-3 smoke, the fused path is stable but
does not improve throughput:

| Skippy path | Errors | Completion tok/s | Accept rate | MTP propose | Target verify |
|---|---:|---:|---:|---:|---:|
| Rust-orchestrated synced MTP | 0 | 13.20 | 0.909 | 1148.3 ms | 3530.6 ms |
| Native fused synced MTP | 0 | 12.89 | 0.870 | 1364.9 ms | 3181.2 ms |

This falsifies the first easy hypothesis: the Rust/OpenAI orchestration and ABI
crossings were not the main bottleneck. The expensive pieces are still the
actual MTP proposal context and the target verification decode. The fused call
is still useful because it gives native counters and is the right place to
experiment with more aggressive speculative contracts, but it is not enough by
itself.

The finer fused counters from the latest instrumented run make the next issue
clear:

```text
completion_tok_s=12.87
windows=30, proposed_tokens=69, accepted_tokens=60, accept_rate=0.870
fused_target_decode_calls=134
fused_target_verify_decode_calls=30, fused_target_verify_tokens=69
fused_target_repair_decode_calls=3, fused_target_repair_tokens=4
fused_mtp_draft_calls=131, fused_mtp_draft_decode_calls=100
fused_mtp_ingest_calls=158, fused_mtp_ingest_tokens=193
fused_checkpoint_calls=30, fused_restore_calls=3
mtp_propose_ms=1368.4, primary_verify_elapsed_ms=3186.8
```

The vanilla control already logs comparable MTP-side counters at the end of the
same w3 smoke:

```text
statistics mtp: #calls(b,g,a) = 8 84 70,
#gen tokens = 163, #acc tokens = 162, dur(b,g,a) = 0.003, 1141.121, 0.020 ms
```

So Skippy was doing many more small control operations for fewer useful draft
tokens: 131 MTP draft calls for 69 proposed tokens, plus 30 separate target
verification decodes. Vanilla generates 163 MTP tokens across 84 draft calls and
lets the server speculative loop sample/accept from target logits directly.

The next local rewrite moved the native path closer to the vanilla contract:

- MTP confidence now uses the same top-k sampler shape as the PR
  (`top_k=10`, `p_min=0.75`) instead of full-vocabulary probability.
- Early rejects repair and resynchronize the MTP context instead of disabling
  MTP for the rest of the request.
- Full-accept windows verify `current + draft`, allowing the target batch's
  final row to contribute the extra sampled token that vanilla keeps.
- Configured window 1 is special-cased back to `current`-only verification
  because the extra-sample shape is not meaningful for a one-token draft
  window.

The current `studio54.local` 32-token one-stage sweep is:

| Window | Completion tok/s | Windows | Proposed | Accepted | Accept rate | Target verify | Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.38 | 116 | 116 | 25 | 0.216 | 6673.6 ms | 5179.3 ms |
| 2 | 13.55 | 59 | 90 | 82 | 0.911 | 6761.8 ms | 492.9 ms |
| 3 | 11.82 | 65 | 125 | 70 | 0.560 | 8047.4 ms | 1487.3 ms |
| 4 | 13.40 | 58 | 96 | 84 | 0.875 | 6753.3 ms | 718.2 ms |
| 5 | 11.82 | 65 | 125 | 70 | 0.560 | 8040.9 ms | 1488.6 ms |

Window 2 is the best Skippy MTP result so far on this branch, but it is still
below the Skippy one-stage baseline (`14.63 tok/s`) and far below vanilla
`llama-server` MTP w3 (`15.88 tok/s`). The remaining problem is not proposal
acceptance alone; target verification and repair are still too expensive for
the short generated spans. The window 1 result is still a useful red flag even
after the special case: single-token MTP windows do not provide enough accepted
work to pay for the proposal, verify, and repair machinery.

### 192-token steady-state result

The 32-token smoke is useful for catching regressions, but it is too short to
represent the MTP PR's steady-state benchmark shape. A 192-token run on
`studio54.local` exposed one important bug in the native span path: when
`on_token()` returned `Stop`, the MTP branches broke only the local token
emission loop and then continued generating hidden tokens up to `max_tokens`.
The OpenAI-visible text was correct, but the runtime did unnecessary target
verify and MTP draft work after stop.

That has been fixed, and the native span path now emits the first visible token
through the normal target decode path before entering MTP spans. This keeps TTFT
bounded while still letting the steady-state decode use fused MTP
proposal/verify.

The clean one-stage 192-token comparison on `studio54.local` uses the same
request shape as vanilla `llama-server`: `temperature: 0`, streaming enabled,
and no `enable_thinking` override. Keeping that field aligned matters for Qwen:
`enable_thinking:false` changes the chat template token count and made the MTP
comparison look artificially bad.

| Runtime | MTP | Window | Completion tok/s | TTFT p50 | Mean latency | Result |
|---|---:|---:|---:|---:|---:|---|
| vanilla `llama-server` | No | n/a | 17.63 | n/a | 10892.0 ms | baseline |
| vanilla `llama-server` | Yes | 2 | 17.35 | n/a | 11064.8 ms | -1.6% vs vanilla baseline |
| Skippy one-stage | No | n/a | 17.81 | 469.3 ms | 10778.7 ms | +1.1% vs vanilla baseline |
| Skippy one-stage | Yes | 2 | 17.36 | 634.5 ms | 11059.2 ms | +0.1% vs vanilla MTP w2, -2.5% vs Skippy baseline |

Raw JSONs:

```text
experiments/mtp/vanilla/studio54/llama-server-baseline-20260512T220839Z.json
experiments/mtp/vanilla/studio54/llama-server-mtp_w2-20260512T221011Z.json
experiments/mtp/studio54/mtp-skippy-one-stage-baseline-20260512T232842Z.json
experiments/mtp/studio54/mtp-skippy-one-stage-mtp-20260512T232610Z.json
```

The Skippy MTP counters for the comparable run, including two warmup decode
spans in the metrics report:

```text
decode_spans=10, windows=404
proposed_tokens=633, accepted_tokens=606, rejected_tokens=21
accept_rate=0.957
mtp_propose_ms=5449.5, mtp_ingest_ms=163.4
primary_verify_elapsed_ms=46797.6, recovery_ms=1534.4
fused_target_decode_calls=654
fused_target_verify_decode_calls=404, fused_target_verify_tokens=1037
fused_target_repair_decode_calls=21, fused_target_repair_tokens=30
```

This changes the current read: Skippy one-stage MTP now matches vanilla
`llama-server` MTP w2 on the comparable request shape. MTP w2 is still not a
win over the non-MTP baseline on this 192-token corpus for either runtime, so
the next useful work is testing other windows and reducing target verify cost,
not moving proposal generation to a separate server yet. MTP ingest remains
small compared with target verification.

By default the two-stage modes use `runtime-slice`, which loads the source GGUF
in both stage processes. If `MTP_REMOTE_STAGE1_SSH` is unset, both stages run
locally. For the intended two-node Skippy experiment, stage 0 runs locally and
stage 1 runs on the remote Skippy node over SSH:

```bash
MTP_REMOTE_STAGE1_SSH=skippy \
MTP_REMOTE_STAGE1_ENDPOINT_HOST=<skippy-lan-ip-or-hostname> \
MTP_REMOTE_STAGE1_ROOT=mesh-llm \
MTP_REMOTE_STAGE1_MODEL_PATH=/path/on/skippy/Qwen3.6-27B-IQ4_XS.gguf \
MTP_REMOTE_METRICS_HOST=<local-lan-ip> \
scripts/run-mtp-skippy-experiment.sh --run two-stage-baseline,two-stage-mtp
```

Remote prerequisites:

- The Skippy node has this repo checked out at `MTP_REMOTE_STAGE1_ROOT`.
- `target/debug/skippy-server` exists on the Skippy node, or
  `MTP_REMOTE_STAGE1_BIN` points to the remote binary.
- The GGUF exists at `MTP_REMOTE_STAGE1_MODEL_PATH` on the Skippy node.
- The local metrics-server OTLP port is reachable from the Skippy node. Set
  `MTP_REMOTE_METRICS_HOST` to the local machine's LAN address; do not use
  `127.0.0.1` for this value. If needed, `MTP_REMOTE_METRICS_OTLP` can override
  the full OTLP URL.
- The local stage 0 process can reach
  `tcp://$MTP_REMOTE_STAGE1_ENDPOINT_HOST:<stage1-port>`.

To materialize local stage GGUF artifacts first:

```bash
MTP_USE_STAGE_ARTIFACTS=1 scripts/run-mtp-skippy-experiment.sh --run two-stage-baseline
```

Stage artifacts are intentionally disabled for remote two-node MTP in this
script because the final node needs runtime access to the full source GGUF's
MTP tensors.

## Dedicated MTP Proposal Node

Another useful topology is to split target verification and MTP proposal work
onto different kinds of servers:

```text
request router
  -> target replica A: layers 0..N, normal decode and verification
  -> target replica B: layers 0..N, normal decode and verification
  -> target replica C: layers 0..N, normal decode and verification
  -> MTP proposal node: MTP/NextN head only
```

In this shape, the first three servers are full target-model replicas. They own
the normal generation state, KV cache, sampling, and verification path. The
fourth server is not a draft model in the classic speculative-decoding sense;
it is a lightweight proposal service for the target model's trained MTP head.

The important constraint is that the MTP head cannot draft from tokens alone.
It needs the target trunk's latest pre-final-norm hidden state, `h_pre_norm`,
plus the last accepted token and position. A dedicated MTP node is therefore
only viable if each target replica can send a compact proposal request:

```text
{
  session_id,
  position,
  last_token,
  h_pre_norm,
  max_proposal_tokens,
  sampling / p_min controls
}
```

The MTP node returns candidate token ids. The target replica still verifies
those candidates with the real target model before committing them:

```text
target replica:
  decode one token
  capture h_pre_norm
  ask MTP node for K proposals
  run VerifySpan locally
  accept prefix, rollback rejected suffix
```

This could be attractive when the MTP head is small enough to keep hot on a
separate device and the target replicas are already saturated with trunk
decode. It also avoids loading the MTP head on every full target replica.

The tradeoff is network sensitivity. `h_pre_norm` is a dense vector per proposal
step. For Qwen3.6-27B the vector is much smaller than an activation frame for a
whole layer span, but it is still latency-sensitive. This topology only helps
if:

- The MTP proposal node is on a low-latency link to the target replicas.
- The MTP head compute plus round trip is cheaper than local MTP execution on
  each target replica.
- Proposal requests are batched or pipelined enough that the MTP node stays
  busy without becoming the global bottleneck.
- Verification remains local to the target replica so rollback does not cross
  servers.

Implementation work for this topology:

1. Add a wire message for `MtpDraftFromHidden` that carries `h_pre_norm`
   directly, instead of assuming the final-stage runtime can copy it from its
   own session.
2. Add a server mode that loads only the MTP head and required tokenizer/output
   metadata from the GGUF.
3. Teach full target replicas to capture `h_pre_norm`, send it to the proposal
   node, then reuse the existing `VerifySpan` path for acceptance.
4. Add telemetry for MTP RPC queue time, network time, draft time, proposal
   count, accepted count, and fallback count.

This is different from the current prototype. The current implementation keeps
MTP local to the final target stage, where `h_pre_norm` already exists in
process. The dedicated-node topology makes MTP independently schedulable, but
it requires serializing `h_pre_norm` over the wire and adding a head-only MTP
server mode.

## Runtime Surface

The native ABI is defined in `include/skippy/mtp.h` and carried durably by:

```text
third_party/llama.cpp/patches/0081-Expose-Qwen-MTP-head-for-Skippy.patch
```

It provides:

- `skippy_mtp_session_set_capture_pre_norm()` and
  `skippy_mtp_session_copy_pre_norm()` for final-stage trunk hidden state.
- `skippy_mtp_head_open()` to load the sibling MTP head from the same GGUF via
  llama.cpp `override_arch`.
- `skippy_mtp_session_reset()` to clear proposal-context memory before an
  independent draft window. MTP proposal state is derived from the caller's
  target `h_pre_norm` row, so sequence/KV state must not carry across requests.
- `skippy_mtp_session_ingest()` and `skippy_mtp_session_draft_synced()` for the
  synchronized sibling-context path, matching the upstream process/draft hook.
- `skippy_mtp_session_draft()` to produce local draft tokens from `h_pre_norm`
  and the last accepted token.
- `skippy_mtp_decode_step()` for the local fused experiment. It keeps target
  decode, MTP ingest, proposal, target verification, accept/reject
  classification, repair, and native counters inside the ABI boundary.
- `skippy_mtp_generate_span()` for native span generation. The OpenAI frontend
  currently calls it in chunks of up to 32 tokens when generation hooks are not
  active, then emits returned tokens through the normal stop/text collector.

Rust wrappers live in `skippy-runtime` as `MtpHead`, `MtpSession`,
`MtpParams`, `StageSession::set_capture_pre_norm()`, and
`StageSession::copy_pre_norm()`. The fused path is exposed as
`StageSession::mtp_decode_step()` and `StageSession::mtp_generate_span()`.

## What Skippy Needs For Real MTP

The current prototype now wires the core runtime path. Remaining follow-up
work before treating numbers as production quality:

1. Tune `MTP_WINDOW` and native `p_min` defaults against real acceptance data.
2. Add a dedicated CLI surface for MTP confidence thresholds if the default
   native `p_min` is too conservative or too aggressive.
3. Broaden tests around staged MTP rejection, repair, and final-stage session
   cleanup.

The most important correctness constraint is rollback. If MTP proposes four
tokens and the target rejects token two, every stage must preserve state for the
accepted prefix and discard state for rejected positions.

## Model Prerequisites

Real MTP requires a specific kind of model:

- The target GGUF must contain trained MTP/NextN tensors.
- The tokenizer and output head must match the target model. Loading the MTP
  head from the same GGUF satisfies this.
- The architecture must be supported by the patched llama.cpp graph. The
  upstream PR currently targets Qwen3.5/Qwen3.6 dense and MoE style MTP.
- The GGUF must include `nextn_predict_layers` metadata.

This is different from normal speculative decoding, where the prerequisite is a
compatible target/draft model pair.
