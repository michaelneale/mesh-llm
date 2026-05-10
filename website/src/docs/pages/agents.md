# Using Agents

Mesh exposes an OpenAI-compatible endpoint, so most agent tools only need a base URL change.

Start Mesh:

```sh
mesh-llm serve --auto
```

Set the local endpoint:

```sh
export OPENAI_BASE_URL=http://localhost:3131/v1
```

## Goose

Configure Goose to use an OpenAI-compatible provider and point it at:

```text
http://localhost:3131/v1
```

## Pi

Use the OpenAI-compatible endpoint setting and set the base URL to:

```text
http://localhost:3131/v1
```

## opencode

Set the OpenAI base URL before launching opencode:

```sh
export OPENAI_BASE_URL=http://localhost:3131/v1
opencode
```
