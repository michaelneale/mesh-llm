# Family Certification Runbook

This document explains how to certify a model family for the stage-split
runtime. It is intentionally a runbook, not the customer-facing support matrix.

Current supported families, recommended artifacts, and shipping settings live in
`docs/FAMILY_STATUS.md`.

## What Certified Means

A family is certified when the current recommended artifact has evidence for:

| Lane | Required Evidence |
| --- | --- |
| `single-step` | A two-stage split produces the same next token as full-model execution. |
| `chain` | The recommended multi-stage split produces the same next token as full-model execution, unless the family has a documented two-stage-only split. |
| `dtype-matrix` | `f32` and `f16` activation transfer are exact; `q8` is marked validated or rejected for that family/split. |
| `state-handoff` | Exact live state mobility is accepted or explicitly rejected by the Qwen3 baseline rule. |
| `llama-spec-bench` | Optional target/draft speculative compatibility checks. |

The default shipping wire dtype is `f16`. `q8` is opt-in only when the
dtype-matrix lane proves exactness for that family and split.

## Certification Command

Use the repo harness. Pass the model id whenever the target comes from
Hugging Face so the run can be promoted into reviewed topology policy.

```bash
just family-certify FAMILY /path/to/model.gguf \
  --model-id org/repo:selector \
  --layer-end N \
  --split-layer K \
  --splits A,B \
  --activation-width H \
  --ctx-size 256 \
  --n-gpu-layers 999 \
  --corpus crates/skippy-bench/corpora/kv_mixed_prompts.jsonl \
  --corpus-limit 24 \
  --max-tokens 8 \
  --run-id family-cert-$(date +%Y%m%d-%H%M%S)
```

For recurrent or hybrid families, pass `--recurrent-all` unless exact recurrent
layer ranges are known. Use `--recurrent-ranges A..B,C..D` when the layer layout
is known.

For a validated two-stage-only topology, pass one split in `--splits`. The
`chain` lane may be skipped because it requires exactly two split indexes;
`single-step` and prompt/spec cover the recommended boundary.

## Speculative Modes

N-gram modes do not require a neural draft model:

```bash
just family-certify FAMILY /path/to/model.gguf \
  ... \
  --draft-model /path/to/draft.gguf
```

Neural draft modes require a separate compatible draft model:

```bash
just family-certify FAMILY /path/to/target.gguf \
  ... \
  --draft-model /path/to/draft.gguf \
  --draft-model /path/to/draft.gguf
```

Before promoting neural draft support, run a target/draft preflight with the
`spec-bench` workflow to prove tokenizer compatibility and useful acceptance.
Do not list neural draft as a shipping feature in `docs/FAMILY_STATUS.md` until
that preflight and staged prompt/spec lane pass.

## Large Model Lifecycle

Large-family certification must keep full-model and staged-server residency
separate unless the combined resident set has been intentionally budgeted.

| Phase | Resident Model Data | Teardown Rule |
| --- | --- | --- |
| Full correctness lanes | Source GGUF opened by `skippy-correctness`. | The full `StageModel` is dropped before staged prompt/spec starts. |
| Stage materialization | Source GGUF is read by `skippy-model-package` to write stage artifacts. | The package tool exits before prompt stages launch. |
| Staged prompt/spec | Local stage artifacts plus optional draft model. The prompt tokenizer uses the first local stage artifact with CPU-only loading. | `skippy-prompt prompt` owns and kills stage/KV/metrics children before returning; corpus modes run one at a time. |

Using the target as its own draft is allowed only when it is useful and the
combined staged target plus draft memory fits. It is not a default certification
shortcut.

## Promotion Checklist

After a meaningful run:

1. Update `docs/FAMILY_STATUS.md` first. This is the source of truth for what a
   customer can run and with which settings.
2. Promote the reviewed capability record in
   `crates/skippy-topology/capabilities/reviewed-family-capabilities.json`
   when the run should drive planner policy.
3. Keep raw run directories under `target/family-certify/...`; do not paste
   running logs into the docs.

## Output Files

Certification runs are written under:

```text
target/family-certify/<run-id>/<family>/<model-slug>/
```

Important files:

| File | Meaning |
| --- | --- |
| `summary.md` | Human-readable command result table. |
| `manifest.json` | Machine-readable family, model, knobs, and lane metadata. |
| `capability-draft.json` | Planner capability evidence extracted from reports. |
| `commands.jsonl` | One JSON record per lane. |
| `reports/*.json` | Correctness reports from `skippy-correctness`. |
| `speculative/summary.tsv` | Prompt/spec corpus result table. |

## Acceptance Rules

Compare against the Qwen3 dense baseline:

```text
exact full-state baseline = 115,388 bytes
reject exact state mobility by default when > 100x baseline
```

Accept small activation handoff even when exact full-state mobility is rejected.
For Falcon-H1/Qwen3Next-style models, the rule is:

```text
activation crosses topology boundaries
recurrent state defines topology affinity
```

Do not recommend transferring recurrent state during normal decode. Keep
recurrent owners sticky and route future tokens for the same sequence back to
those owners.

## Verification

Run focused checks after changing certification docs, planner records, or the
prompt/spec lifecycle:

```bash
bash -n scripts/family-certify.sh
cargo fmt --check -p skippy-topology -p skippy-model-package -p skippy-prompt
cargo test -p skippy-topology
LLAMA_STAGE_BUILD_DIR=.deps/llama-stage.cpp/build-stage-abi-static \
  cargo build -p skippy-prompt -p skippy-model-package -p skippy-topology
git diff --check
```
