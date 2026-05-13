# Skippy WAN Docker Lab

This lab runs a four-stage CPU-backed Skippy chain in Docker with Linux
`tc netem` shaping between stages and `metrics-server` enabled.

The lab is intentionally host-cache first: `up.sh` ensures the Hugging Face
layer package is present and complete in the host HF cache before Docker
containers are brought up. Containers then mount that cache read-only.

## Layout

- `metrics` runs `metrics-server` on HTTP `:18080` and OTLP/gRPC `:14317`.
- `stage0` owns the first layer range, exposes OpenAI on host port `9337`, and
  forwards binary activation traffic to `stage1`.
- `stage1`, `stage2`, and `stage3` run the remaining layer ranges.
- Every stage container has `NET_ADMIN` and applies `tc netem` to the Docker
  interface used for stage-to-stage traffic.

The shaping is Linux-level traffic control, not Skippy's artificial
`--downstream-wire-delay-ms` or `--downstream-wire-mbps` flags.

## Configure

Create the local env file if you want to override defaults:

```bash
cp docker/skippy-wan-lab/.env.example docker/skippy-wan-lab/.env
```

The default package is:

```text
hf://meshllm/gemma-4-26B-A4B-it-UD-Q4_K_M-layers
```

`HF_HOME` controls which host Hugging Face cache is used. If unset, the launcher
uses `${HOME}/.cache/huggingface`. Authenticate with the host `hf` CLI when
needed:

```bash
hf auth login
```

To calibrate WAN latency from a target such as `100.90.121.70`:

```bash
scripts/skippy-wan-calibrate.sh 100.90.121.70 docker/skippy-wan-lab/.env.link
```

The script writes `WAN_RTT_MS` and `WAN_DELAY_MS`. If the target has an `iperf3`
server, it also records `WAN_RATE_MBIT`; otherwise fill bandwidth manually if
you want rate limiting as well as latency.

## Run

Use the launcher, not raw `docker compose`, so the model invariant is enforced:

```bash
docker/skippy-wan-lab/up.sh
```

Before compose starts, the launcher runs the equivalent of:

```bash
hf download meshllm/gemma-4-26B-A4B-it-UD-Q4_K_M-layers \
  --include model-package.json \
  --include 'shared/*' \
  --include 'layers/*' \
  --include 'projectors/*'
```

It then verifies the package manifest, layer count, activation width, artifact
presence, and artifact byte sizes from the host cache. If anything is missing,
the lab exits before any containers start.

## Interactive Prompt

With the lab running in one terminal, attach a `skippy-prompt binary` REPL to
stage0 from another terminal:

```bash
docker/skippy-wan-lab/prompt.sh
```

Extra REPL flags are passed through to `skippy-prompt binary`, for example:

```bash
docker/skippy-wan-lab/prompt.sh --max-new-tokens 64 --no-think
```

The prompt helper does not start containers or download the model. It attaches
to the existing `stage0` container and uses the same env files and read-only
host HF cache mount as `up.sh`.

The OpenAI-compatible endpoint is:

```bash
curl -s http://127.0.0.1:9337/v1/models | jq
```

Example request:

```bash
curl -s http://127.0.0.1:9337/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M",
    "messages": [{"role": "user", "content": "Write one short sentence."}],
    "max_tokens": 16
  }' | jq
```

## Metrics

Stage telemetry is emitted to `metrics-server` using OTLP/gRPC:

```bash
curl -s http://127.0.0.1:18080/v1/runs/skippy-docker-wan/status | jq
curl -s -X POST http://127.0.0.1:18080/v1/runs/skippy-docker-wan/finalize | jq
curl -s http://127.0.0.1:18080/v1/runs/skippy-docker-wan/report.json | jq
```

The DuckDB file is stored in the `metrics_data` Docker volume.

## Inspect Traffic Control

```bash
docker compose \
  --env-file docker/skippy-wan-lab/.env \
  --env-file docker/skippy-wan-lab/.env.link \
  -f docker/skippy-wan-lab/docker-compose.yml \
  exec stage0 tc -s qdisc
```

## Notes

- The host HF cache is mounted read-only at `/hf-cache`.
- The default load mode is `layer-package` with tensor filtering enabled.
- CPU execution is forced with `n_gpu_layers: 0`.
- Use `WAN_ENABLE=0` to disable shaping without changing the compose topology.
- Docker Desktop runs Linux containers inside a Linux VM; `tc netem` is applied
  inside that VM.
