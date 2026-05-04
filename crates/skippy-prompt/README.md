# skippy-prompt

Prompt REPL and diagnostics client for staged runtimes.

`skippy-prompt` is the operator-facing CLI for driving interactive text
generation against a running first stage. In mesh-llm, topology launch,
materialization, metrics wiring, and lifecycle are owned by mesh, so the
imported `prompt` launcher is retained only as a diagnostic planning surface.
Use the `binary` subcommand against a mesh-managed first stage for live checks.

## Architecture Role

The mesh-managed path builds a topology from model/package metadata and peer
inventory, starts stage servers, waits for readiness, publishes the stage-0
route, and then lets diagnostic clients attach to the first stage.

```mermaid
flowchart TB
    Mesh["mesh-llm coordinator"] --> Plan["skippy-topology<br/>validate splits"]
    Mesh --> Slice["skippy-model-package<br/>materialize derived stage cache"]
    Mesh --> Metrics["metrics-server<br/>optional debug sink"]
    Mesh --> S0["stage-0<br/>skippy-server"]
    S0 --> S1["stage-1"]
    S1 --> SF["final stage"]
    CLI["skippy-prompt binary<br/>diagnostic client"] --> REPL["interactive REPL<br/>history, interrupts"]
    REPL --> S0
    SF --> S0
```

The `binary` subcommand skips launching servers and connects to an already
running first-stage `serve-binary` endpoint.

```mermaid
sequenceDiagram
    participant U as user
    participant P as skippy-prompt binary
    participant T as local tokenizer/runtime
    participant S as first stage
    participant F as final stage

    U->>P: prompt text
    P->>T: apply chat template and tokenize
    P->>S: PrefillEmbd / DecodeEmbd
    S->>F: activation frames
    F-->>S: predicted tokens
    S-->>P: predicted tokens
    P->>T: detokenize stream
    P-->>U: generated text
```

## Commands

```bash
skippy-prompt binary --model-path model.gguf --first-stage-addr 127.0.0.1:19031
```

Useful REPL commands include `:history`, `:logs [name] [lines]`, and `:quit`.

## Notes

- Default local state lives under `/tmp/skippy-prompt`.
- Remote runs stage inputs under `/tmp/skippy-remote-prompt` by default.
- `--activation-wire-dtype q8` is accepted only when topology policy has
  validation for the requested family/split.
- `--draft-model-path` enables draft-model speculative proposals.
- Standalone cache and n-gram sidecars are not imported into mesh-llm; topology
  launch exits before starting sidecars.
- Thinking controls are forwarded through the shared `openai-frontend`
  reasoning/template normalization helpers.

Keep server transport behavior in `skippy-server`, model/session ABI
wrapping in `skippy-runtime`, and reusable OpenAI request shapes in
`openai-frontend`.
