# CLI User Guide

This is a practical user guide to the `mesh-llm` CLI.
It explains what to run for common tasks, then documents each command and switch.

Catalog id definition: a catalog id is the model id shown in `mesh-llm models recommended` (for example `Qwen3-0.6B-Q4_K_M`).

## Get help

```bash
mesh-llm --help
mesh-llm <command> --help
mesh-llm models --help
mesh-llm models <subcommand> --help
```

## Start here (common tasks)

If you want to:

1. Start serving right away:

```bash
mesh-llm serve --auto
```

2. Find a model you can run:

```bash
mesh-llm models search gemma --gguf
mesh-llm models search smoll --mlx
```

3. Inspect a model before downloading:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

4. Download a model:

```bash
mesh-llm models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

5. Check what is already installed:

```bash
mesh-llm models installed
```

## Runtime entrypoints (`serve` / `client`)

If you want to start serving, join a mesh, or run as an API-only client, start here.

Examples:

```bash
mesh-llm serve
mesh-llm serve --model Qwen3-0.6B-Q4_K_M
mesh-llm client --auto
```

Runtime switches:

- `--join <TOKEN>`: join a specific mesh using an invite token (repeatable).
- `--discover [QUERY]`: discover a mesh via Nostr and join.
- `--auto`: auto-join the best discovered mesh.
- `--model <MODEL>`: model to serve (catalog id from `models recommended`, HF ref/URL, or path).
- `--gguf <GGUF>`: serve a specific local GGUF file directly (repeatable).
- `--port <PORT>`: API port (default `9337`).
- `--client`: API-only mode (no GPU/model serving).
- `--console <CONSOLE>`: console/API management port (default `3131`).
- `--publish`: publish your mesh for discovery.
- `--mesh-name <MESH_NAME>`: friendly mesh name in discovery.
- `--region <REGION>`: region hint for discovery.
- `--blackboard`: enable blackboard on public meshes.
- `--name <NAME>`: your blackboard display name.
- `--max-vram <MAX_VRAM>`: cap VRAM used for planning and fit decisions.
- `--llama-flavor <LLAMA_FLAVOR>`: force backend binary flavor (`cpu|cuda|rocm|vulkan|metal`).
- `--config <CONFIG>`: explicit config file path.
- `--owner-key <OWNER_KEY>`: keystore used to attest this runtime node.
- `--owner-required`: fail startup if owner attestation cannot be loaded.
- `--node-label <NODE_LABEL>`: attach a human label to this runtime node certificate.
- `--trust-policy <TRUST_POLICY>`: override peer ownership trust policy.
- `--trust-owner <TRUST_OWNER>`: add trusted owner IDs on top of the local trust store.

## Commands

### `models`

Start with `models` when you’re working with models: finding them, checking details, downloading them, or checking update state.

Subcommands:

- `recommended`
- `installed`
- `search`
- `show`
- `download`
- `updates`

### `models recommended`

Run this when you want the official built-in model IDs (catalog IDs) and sizes.

Switches:

- `--json`: machine-readable output.

### `models installed`

Run this when you want to see what’s already on your machine.

Switches:

- `--json`: machine-readable output.

### `models search`

Use this to find something you can actually download and run (GGUF or MLX).

Usage:

```bash
mesh-llm models search gemma --gguf
mesh-llm models search smoll --mlx --limit 5
mesh-llm models search qwen --catalog
```

Switches:

- `--gguf`: GGUF-only search (default).
- `--mlx`: MLX-only search.
- `--catalog`: search only built-in catalog.
- `--limit <LIMIT>`: max results (default `20`).
- `--json`: machine-readable output.

### `models show`

Use this when you want to sanity-check one exact model ref before you download or serve it.

Usage:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
mesh-llm models show mlx-community/SmolLM-135M-8bit
```

Switches:

- `--json`: machine-readable output.

### `models download`

Use this when you’re ready to download one specific resolved model.

Usage:

```bash
mesh-llm models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
mesh-llm models download mlx-community/SmolLM-135M-8bit
```

Switches:

- `--draft`: also download the recommended draft model (if available).
- `--json`: machine-readable output.

### `models updates`

Use this when you want to check for new upstream revisions or refresh cached repo metadata.

Usage:

```bash
mesh-llm models updates --check
mesh-llm models updates --all
mesh-llm models updates unsloth/gemma-4-31B-it-GGUF
```

Switches:

- `--all`: operate on all cached HF repos.
- `--check`: check only; do not refresh cache.
- `--json`: machine-readable output.

### `download`

Use this to quickly download by built-in catalog ID or shorthand.

Usage:

```bash
mesh-llm download
mesh-llm download 32b
mesh-llm download Qwen3-0.6B-Q4_K_M --draft
```

Switches:

- `--draft`: download recommended draft model too.

### `update`

Use this to update mesh-llm and exit.

Switches:
- `--auto-update`: available on most commands; when set, mesh-llm checks for a newer bundled release before proceeding.


### `gpus`

Use this to inspect local GPU identity/capacity.


### `load`

Use this to load a model into an already-running local mesh-llm runtime.

Usage:

```bash
mesh-llm load Qwen3-0.6B-Q4_K_M
```

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `unload`

Use this to unload a model from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `status`

Use this to inspect model status from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `discover`

Use this to discover meshes via Nostr and optionally select one automatically.

Switches:

- `--model <MODEL>`: filter discovered meshes by model name substring.
- `--min-vram <MIN_VRAM>`: filter by minimum VRAM (GB).
- `--region <REGION>`: filter by region.
- `--auto`: print best invite token (useful for piping).
- `--relay <RELAY>`: custom relay URL(s).

### `goose`

Use this to launch Goose already wired to mesh-llm’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: mesh-llm API port (default `9337`).

### `claude`

Use this to launch Claude Code already wired to mesh-llm’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: mesh-llm API port (default `9337`).

### `stop`

Use this to stop local `mesh-llm`, `llama-server`, and `rpc-server` processes.


### `blackboard`

Use this to post/search/read shared mesh notes, or to run blackboard as MCP over stdio.

Usage:

```bash
mesh-llm blackboard
mesh-llm blackboard "STATUS: testing gguf resolution"
mesh-llm blackboard --search "gemma"
mesh-llm blackboard --mcp
```

Switches:

- `--search <SEARCH>`: search blackboard entries.
- `--from <FROM>`: filter by author.
- `--since <SINCE>`: last N hours.
- `--limit <LIMIT>`: max rows (default `20`).
- `--port <PORT>`: target management/API port (default `3131`).
- `--mcp`: run as MCP server over stdio.

### `plugin`

Use this to inspect plugin status or run plugin compatibility shims.

Subcommands:

- `plugin list`: list auto-registered/configured plugins.
- `plugin install <NAME>`: old install workflow shim.


### `auth`

Use this to manage owner identity and keystore files.

Subcommands:

- `auth init`: generate/save owner keypair.
- `auth status`: show identity/keystore status.

`auth init` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--force`: overwrite existing keystore.
- `--no-passphrase`: leave keys unencrypted.
- `--keychain`: store random unlock passphrase in OS keychain.

`auth status` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

`auth sign-node` / `auth renew-node` / `auth rotate-node` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--node-label <NODE_LABEL>`: attach a human label to the signed node certificate.

`auth rotate-owner` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

## Model reference formats

Supported for `models show`, `models download`, and `serve --model`:

1. Catalog id (an id from `mesh-llm models recommended`):

```bash
mesh-llm models show Qwen3-0.6B-Q4_K_M
```

2. HF repo or GGUF selector:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

3. HF URL:

```bash
mesh-llm models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
```

4. Revision pin:

```bash
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@main
mesh-llm models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@<commit-sha>
mesh-llm models show mlx-community/SmolLM-135M-8bit@<commit-sha>
mesh-llm models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/main
```

For MLX, use repo shorthand (not `/model`):

```bash
mesh-llm models show mlx-community/SmolLM-135M-8bit
mesh-llm models download mlx-community/SmolLM-135M-8bit
```

## Model resolution behavior

Resolution order:

1. exact catalog id
2. exact HF ref
3. HF URL
4. bare-name discovery

GGUF behavior:

1. GGUF search uses Hub `gguf` pre-filter.
2. Excludes sidecars like `mmproj*.gguf`.
3. Split GGUF uses first shard (`-00001-of-...`) for selection/display.
4. `repo` with no selector uses fit-aware ranking against local VRAM.
5. `repo:SELECTOR` resolves exact quant/variant.

MLX behavior:

1. MLX search uses Hub `mlx` pre-filter.
2. Model must include weight files (`model.safetensors` or split first shard).
3. `model.safetensors.index.json` by itself is not treated as a model artifact.
4. Display reference stays repo shorthand.

## Machine-readable output (`--json`)

All `models` subcommands support `--json`.

Examples:

```bash
mesh-llm models search smoll --mlx --limit 1 --json | jq .
mesh-llm models show mlx-community/SmolLM-135M-8bit --json | jq .
mesh-llm models download Qwen3-0.6B-Q4_K_M --json | jq .
mesh-llm models installed --json | jq .
mesh-llm models recommended --json | jq .
mesh-llm models updates --check --json | jq .
```

Shape summary:

- `search --json`: `{ filter, query, machine, results[] }`
- `show --json`: resolved model + `variants[]`
- `download --json`: requested/resolved refs + local `path`
- `installed --json`: `{ cache_dir, results[] }`
- `recommended --json`: `{ source, results[] }`
- `updates --json`: check/update results

Automation tips:

1. Prefer explicit refs in scripts.
2. Pin `@<commit-sha>` when reproducibility matters.
3. Parse stable keys such as `type`, `ref`, `fit`, `path`, and `results`.
