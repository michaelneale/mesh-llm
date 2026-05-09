---
name: hf-family-certification-jobs
description: Use when offloading skippy family certification to Hugging Face Jobs, submitting certification jobs, reviewing certification artifacts, or chaining successful certification into layer-package jobs.
metadata:
  short-description: Drive HF Jobs family certification safely
---

# HF Family Certification Jobs

Use this skill when an agent needs cloud hardware to certify a GGUF model family
for skippy staged serving.

## Rules

1. Certification jobs are spend-bearing. Always run a JSON dry run first.
2. Do not submit unless the user or supervising workflow has accepted the dry-run
   `jobPlan.max_cost_usd`.
3. Confirmed submissions use `--confirm`. `models certify --hf-job --timeout` is the
   certification job's hard max-cost timeout;
   do not rely on package-splitting size floors for certification jobs.
4. Submit jobs against a pushed branch or exact commit SHA via
   `--mesh-llm-ref`. Prefer exact SHAs once a branch has been pushed.
5. If the job fails, inspect logs/artifacts, patch the branch, push, and
   resubmit the new SHA. Keep retries bounded and explain repeated failures.
6. Do not promote topology policy or docs as certified until a passing artifact
   set exists.
7. Keep family certification separate from layer packaging. Queue packaging only
   after certification passes and slicing support does not require further code
   changes.
8. Certify the model family with the smallest same-family text GGUF that still
   exercises the same llama.cpp architecture/topology. Do not default to the
   largest 4-bit quant for certification; reserve the largest useful 4-bit quant
   for layer packaging after certification passes.

## Representative Quant Selection

Before spending money on certification, list the repo's GGUF variants and rank
them by `total_bytes`:

```bash
mesh-llm models package ORG/REPO --hf-job --json
```

Pick the smallest candidate that is representative of the target family:

- Must be a text GGUF for the same family and architecture.
- Must not be a VL/multimodal repo when certifying a text family.
- Must not be BF16 or another oversized precision unless no smaller quant exists.
- Prefer tiny/low-bit quants such as `TQ1_0`, `UD-IQ1_S`, or `UD-IQ1_M` when
  they are present and resolve to the same family.
- Keep the certification source and the later packaging source separate in notes
  and PRs so a small-quant certification is not confused with packaging the
  largest production quant.

## Commands

Dry run:

```bash
mesh-llm models certify ORG/REPO:QUANT \
  --hf-job \
  --family FAMILY \
  --mesh-llm-ref REF \
  --artifact-repo meshllm/family-certification-runs \
  --json
```

Submit after reviewing `jobPlan.max_cost_usd`:

```bash
mesh-llm models certify ORG/REPO:QUANT \
  --hf-job \
  --family FAMILY \
  --mesh-llm-ref REF \
  --artifact-repo meshllm/family-certification-runs \
  --confirm \
  --follow \
  --json
```

Inspect:

```bash
mesh-llm models certify --status JOB_ID --json
mesh-llm models certify --logs JOB_ID
mesh-llm models certify --cancel JOB_ID
mesh-llm models certify --list --json
```

## Branch Loop

1. Create a branch with the standard `jd/` prefix.
2. Make the llama pin, patch queue, topology, certification, or docs change.
3. Run local focused validation before spending HF money.
4. Push the branch and record `git rev-parse HEAD`.
5. Dry-run certification with `--mesh-llm-ref <sha> --json`.
6. Submit with `--confirm` after accepting the dry-run max cost.
7. If the job fails, classify the failing phase, patch locally, push, and
   resubmit the new SHA.
8. Open a PR only after the certified SHA has passing artifacts.

## MiMo Policy

For MiMo V2.5, use true Unsloth MiMo text GGUFs as evidence:

```bash
mesh-llm models certify unsloth/MiMo-V2-Flash-GGUF:TQ1_0 \
  --hf-job \
  --family mimo2 \
  --mesh-llm-ref REF \
  --artifact-repo meshllm/family-certification-runs \
  --json
```

Do not use `unsloth/MiMo-VL-7B-RL-GGUF` as MiMo2 certification evidence; that
GGUF reports the `qwen2vl` architecture. As of May 9, 2026,
`unsloth/MiMo-V2-Flash-GGUF` includes smaller representative quants:

- `TQ1_0`: about 72.36 GiB, single GGUF file; preferred first certification
  candidate.
- `UD-IQ1_S`: about 81.23 GiB across 2 shards; backup certification candidate.
- `UD-IQ1_M`: about 87.36 GiB across 2 shards; backup certification candidate.

If certification passes and no slicing code change is needed, package the
largest true Unsloth MiMo 4-bit quant the workflow has selected. For MiMo Flash,
the larger 4-bit candidates include `IQ4_XS`, `IQ4_NL`, `Q4_0`, `Q4_K_S`,
`UD-Q4_K_XL`, `Q4_K_M`, and `Q4_1`; choose the packaging target based on the
current Unsloth candidate policy and a fresh dry run.
