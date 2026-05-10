# Public Mesh

The public mesh is available at:

[public.meshllm.cloud](https://public.meshllm.cloud)

Join as a serving node if this machine can contribute compute:

```sh
mesh-llm serve --auto
```

Join as an API-only client if this machine should only send requests:

```sh
mesh-llm client --auto
```

Point OpenAI-compatible tools at the local Mesh API:

```sh
export OPENAI_BASE_URL=http://localhost:3131/v1
```
