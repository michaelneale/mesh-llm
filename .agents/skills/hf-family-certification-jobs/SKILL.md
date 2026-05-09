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
3. Confirmed submissions use `--confirm`. `certify-family --timeout` is the
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

## Commands

Dry run:

```bash
mesh-llm models certify-family ORG/REPO:QUANT \
  --family FAMILY \
  --mesh-llm-ref REF \
  --artifact-repo meshllm/family-certification-runs \
  --json
```

Submit after reviewing `jobPlan.max_cost_usd`:

```bash
mesh-llm models certify-family ORG/REPO:QUANT \
  --family FAMILY \
  --mesh-llm-ref REF \
  --artifact-repo meshllm/family-certification-runs \
  --confirm \
  --follow \
  --json
```

Inspect:

```bash
mesh-llm models certify-family --status JOB_ID --json
mesh-llm models certify-family --logs JOB_ID
mesh-llm models certify-family --cancel JOB_ID
mesh-llm models certify-family --list --json
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
mesh-llm models certify-family unsloth/MiMo-V2-Flash-GGUF:IQ4_XS \
  --family mimo2 \
  --mesh-llm-ref REF \
  --artifact-repo meshllm/family-certification-runs \
  --json
```

Do not use `unsloth/MiMo-VL-7B-RL-GGUF` as MiMo2 certification evidence; that
GGUF reports the `qwen2vl` architecture. If certification passes and no slicing
code change is needed, package the largest true Unsloth MiMo 4-bit quant the
workflow has selected.
