# OpenAI-Compatible API

Mesh exposes an OpenAI-compatible endpoint so existing tools can use Mesh without a custom client.

## Base URL

```text
http://localhost:3131/v1
```

Set it for tools that read OpenAI-style environment variables:

```sh
export OPENAI_BASE_URL=http://localhost:3131/v1
```

## Chat completions

Use OpenAI-compatible chat clients against the local Mesh endpoint. Mesh handles model routing and placement behind the API.

## Streaming

Clients that support streamed OpenAI-compatible responses can use the same base URL.

## Tool calling

Tool-calling support depends on the selected model family and the agent client. Use the catalog to choose models with the right capabilities.

## Structured outputs

Structured output support depends on the model and client behavior. Treat schema enforcement as model- and tool-specific until the catalog marks stronger guarantees.

