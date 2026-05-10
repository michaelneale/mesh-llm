# Installing Mesh

Install Mesh with the hosted installer:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | sh
```

After install, start this machine as a model-serving node:

```sh
mesh-llm serve --auto
```

Or join as an API-only client:

```sh
mesh-llm client --auto
```

Mesh exposes an OpenAI-compatible API on the local machine, so existing tools can point at:

```sh
export OPENAI_BASE_URL=http://localhost:3131/v1
```

## macOS

Use the default installer on Apple Silicon. Mesh uses the local Metal-capable runtime when available.

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | sh
mesh-llm serve --auto
```

## Linux

Install on the machines that should contribute GPU or CPU capacity.

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | sh
mesh-llm serve --auto
```

Linux nodes can use CUDA, ROCm, Vulkan, or CPU builds depending on the release flavor and local hardware.

## Windows

Windows support is planned. For now, use WSL2 on Windows machines and follow the Linux install path.

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | sh
mesh-llm client --auto
```
