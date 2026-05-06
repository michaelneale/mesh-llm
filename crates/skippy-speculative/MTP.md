# Gemma 4 Multi-Token Prediction Notes

Google's Gemma 4 multi-token prediction (MTP) release is relevant to Skippy,
but it splits into two different implementation levels.

## What Applies Now

The general technique is speculative decoding:

1. A cheaper proposer predicts multiple future tokens.
2. The target model verifies the proposed span in one target pass.
3. The runtime commits the accepted prefix and repairs or falls back after a
   rejection.

This matches the shape of Skippy's existing speculative path. The
`skippy-speculative` crate owns the reusable policy and verification logic:

- n-gram proposal history and adaptive policy state;
- verify-span decision classification;
- repair strategy selection;
- proposal and verification accounting helpers.

Gemma 4 assistant models can potentially fit as another neural draft source,
similar to the current draft-model path, if the target and assistant GGUFs load
through the patched llama.cpp runtime and have compatible tokenization.

## What Does Not Apply For Free

Google's full Gemma 4 MTP path is deeper than a normal external draft model.
The described assistant can share target-model structures such as embeddings,
last-layer activations, KV/cache data, and compact embedder components.

Skippy's current `DraftRunner` is independent:

- it loads a separate draft model;
- it proposes tokens before target verification;
- it does not consume target final-layer activations;
- it does not share KV/cache state with the target model.

Using the full Gemma 4 MTP design would require runtime support rather than
just policy-code changes in this crate.

## Architecture Fit

The current path looks like this:

```text
stage0/session context
        |
        v
proposal source: n-gram or independent draft model
        |
        v
target staged VerifySpan request
        |
        v
commit accepted prefix, repair rejection, adapt policy
```

A native Gemma 4 MTP path would likely look more like this:

```text
target final-stage activation / logits context
        |
        v
MTP assistant head or assistant model
        |
        v
draft token span
        |
        v
target staged VerifySpan request
```

For staged serving, that points toward final-stage or target-runtime ownership
of MTP proposal generation. Stage 0 can still orchestrate the request, but the
best data for native MTP lives near the final target layers.

## Correctness Caveat

Token-equality verification is sufficient for greedy or deterministic-style
decode paths. For exact stochastic speculative decoding under temperature,
top-p, top-k, or similar sampling controls, the verifier needs target and
draft probabilities and the standard acceptance/correction rule.

Before claiming distribution-preserving sampled decoding, Skippy needs a
probability-aware verification path through llama-stage and the wire protocol.

## Proposed Plan

1. **Compatibility probe**
   - Check whether current patched llama.cpp can load Gemma 4 target and
     `*-assistant` models in the formats we use.
   - Confirm tokenizer agreement and GGUF metadata expectations.

2. **Classical assistant benchmark**
   - Treat a Gemma 4 assistant model as an independent draft model.
   - Benchmark target-only, assistant-draft, n-gram, and adaptive hybrid modes
     through the same OpenAI/staged telemetry path.
   - Track proposal cost, verify cost, accepted tokens, repair cost, and
     end-to-end completion tokens per second.

3. **Probability-aware verification design**
   - Define the target/draft probability data needed for exact sampled
     speculative decoding.
   - Decide whether that belongs in llama-stage ABI, `StageReply`, or a
     separate verifier message.

4. **Native MTP runtime investigation**
   - Inspect upstream llama.cpp support for Gemma 4 MTP assistant models.
   - Identify whether assistant proposal generation can reuse target
     activations or KV/cache in the embedded runtime.
   - If viable, add a runtime-owned proposal source rather than forcing MTP
     through the current independent `DraftRunner`.

## Working Decision

Use Gemma 4 MTP first as evidence for a better neural proposal source, not as a
promise that Skippy can immediately use Google's full optimized MTP stack.

The near-term product path is:

```text
n-gram proposer + independent assistant draft proposer + adaptive policy
```

The deeper research path is:

```text
native target-runtime MTP proposer + probability-aware verification
```

