---
name: skippy-family-certification
description: Use this skill when certifying a GGUF model family for skippy stage-split serving, reviewing capability data, promoting family evidence into topology policy, or updating staged split certification docs.
metadata:
  short-description: Certify model families for staged splits
---

# skippy-family-certification

Use this skill for end-to-end family certification, not a one-off correctness
smoke. Certification means collecting evidence for full-model parity, staged
activation handoff, recurrent/hybrid state behavior, topology constraints,
selected-device behavior, and package materialization.

## Workflow

1. Inspect the model with `skippy-runtime::ModelInfo` or the model-package
   helpers before choosing split points. Keep topology policy in
   `crates/skippy-topology`.

2. Prefer reviewed capability data in
   `crates/skippy-topology/capabilities/reviewed-family-capabilities.json`.
   Do not enable default staged splits for a family without reviewed evidence.

3. For dense models, validate at least one representative two-stage boundary
   and one multi-stage boundary. For recurrent or hybrid families, validate
   recurrent ranges explicitly and treat recurrent owners as topology-affinity
   constraints.

4. Compare staged output against full-model execution with the correctness
   harness when it is present. In this mesh repo, some standalone skippy harness
   crates may still be migration candidates; do not invent replacement commands
   without checking `cargo metadata`.

## Decision Rules

Default activation wire dtype is `f16`. Treat `q8` as per-family and per-split
opt-in only after exactness evidence exists.

Do not recommend transferring recurrent state during normal decode unless the
family has explicit reviewed evidence for it. Prefer sticky recurrent ownership
and route future tokens for the same sequence back to those owners.

Keep lifecycle phases separate for large models: inspect/materialize, drop any
full source model, then launch staged serving. Avoid holding a full source GGUF
resident while testing staged servers.
