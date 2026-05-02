# llama-benchy Against serve-openai

[`llama-benchy`](https://github.com/eugr/llama-benchy) is an OpenAI-compatible
benchmark client. Its upstream README currently says it evaluates
`/v1/chat/completions` only, auto-discovers a model from `/v1/models` when
`--model` is omitted, and expects `--base-url` to include the `/v1` prefix.

## Contract

`skippy-server serve-openai` should expose enough of the OpenAI API for
benchy to run without the stage protocol leaking through:

| Endpoint / field | Why benchy needs it | Our owner |
| --- | --- | --- |
| `GET /v1/models` | Optional model auto-discovery | `openai-frontend` route, backend model list |
| `POST /v1/chat/completions` | Main benchmark request path | `openai-frontend` route, stage backend generation |
| `stream: true` SSE | Token timing and throughput | `openai-frontend` SSE framing |
| `[DONE]` SSE marker | Stream termination | `openai-frontend` |
| streaming and non-stream `usage` | Prompt/completion accounting | stage backend usage counts |
| `max_tokens` | Generation length control | stage backend enforcement |

`/v1/completions` is not required by benchy today, but the frontend keeps it in
the same contract tests because other OpenAI-compatible load tools still use
the legacy completions API.

## Start the OpenAI Frontend

Single-stage or final-stage local runtime:

```bash
skippy-server serve-openai \
  --config stage.json \
  --bind-addr 127.0.0.1:9337 \
  --model-id org/repo:Q4_K_M
```

Existing staged binary chain:

```bash
skippy-server serve-openai \
  --config final-stage.json \
  --bind-addr 127.0.0.1:9337 \
  --first-stage-addr 127.0.0.1:19031 \
  --model-id org/repo:Q4_K_M
```

The `--model-id` value is the exact OpenAI `model` id that `/v1/models`
advertises and requests must use. Mesh-style ids such as `org/repo:Q4_K_M` are
treated opaquely by `serve-openai`; the suffix selects a model artifact, not a
stage-server topology.

## Run Benchy

## Local Smoke

Run the repeatable local smoke before changing the OpenAI frontend or
`serve-openai` adapter:

```bash
just build
RUN_BENCHY=1 scripts/openai-smoke.sh
```

The smoke downloads a small SmolLM2 GGUF unless `MODEL_PATH` is already set,
infers `layer_end`, writes a temporary single-stage config, starts
`skippy-server serve-openai`, probes the OpenAI routes, optionally runs the
tiny benchy case, and then stops the server.

Useful overrides:

```bash
MODEL_PATH=/path/to/model.gguf \
MODEL_ID=org/repo:Q4_K_M \
TOKENIZER=HuggingFaceTB/SmolLM2-135M-Instruct \
RUN_BENCHY=1 \
scripts/openai-smoke.sh
```

Artifacts default to `/tmp/skippy-openai-smoke`.

## Run Benchy

The helper script defaults to the local `serve-openai` address:

```bash
MODEL=meta-llama/Llama-3.2-1B-Instruct:Q4_K_M \
SERVED_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct:Q4_K_M \
TOKENIZER=meta-llama/Llama-3.2-1B-Instruct \
PP="128 512" \
TG="16 32" \
DEPTH="0" \
RUNS=3 \
SAVE_RESULT=/tmp/skippy-benchy.md \
scripts/run-llama-benchy-openai.sh
```

Equivalent direct command:

```bash
uvx --from git+https://github.com/eugr/llama-benchy llama-benchy \
  --base-url http://127.0.0.1:9337/v1 \
  --model meta-llama/Llama-3.2-1B-Instruct:Q4_K_M \
  --served-model-name meta-llama/Llama-3.2-1B-Instruct:Q4_K_M \
  --tokenizer meta-llama/Llama-3.2-1B-Instruct \
  --pp 128 512 \
  --tg 16 32 \
  --depth 0 \
  --runs 3 \
  --latency-mode generation \
  --skip-coherence
```

Use `BASE_URL` if the OpenAI frontend is hosted elsewhere. The base URL should
include `/v1`.

## Caveats

- Benchy reports both TTFR and end-to-end TTFT. Our chat stream emits an
  assistant-role chunk before content, so TTFR can include that non-content
  chunk while TTFT tracks the first content token.
- Prefix-cache benchmarking should be treated carefully until the OpenAI
  frontend is wired to stage/KV cache controls intentionally.
- Temperature, top-p, top-k, seed, common penalties, and token-id logit bias are
  passed through the stage ABI; unsupported controls are still rejected instead
  of silently ignoring benchmark inputs.
- The binary stage protocol carries sampling as an optional extension, so
  default greedy requests do not include sampling fields or logit-bias entries.
- Token-array prompts for `/v1/completions` parse but are rejected until a
  backend can honor token IDs directly.
