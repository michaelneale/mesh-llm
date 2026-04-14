# llama.cpp Fork

mesh-llm uses a lightly patched fork of llama.cpp:

**[github.com/Mesh-LLM/llama.cpp](https://github.com/Mesh-LLM/llama.cpp)**

All custom patches live on `master`. The pinned commit SHA is in `LLAMA_CPP_SHA` at the repo root.

---

## What's on the fork

8 commits on top of upstream `ggml-org/llama.cpp`:

| Commits | Area | What |
|---|---|---|
| 3 | RPC | Zero-transfer GGUF loading, alloc cache, B2B direct transfers |
| 4 | MoE | Expert mask routing, analysis tool, split tool, shared-expert fix |
| 1 | Mesh hooks | Virtual LLM engine — inter-model collaboration during inference |

All patches are additive. When `--mesh-port` is not passed, hook code is completely inert.

---

## Updating the fork from upstream

When you want to pick up new llama.cpp features:

```bash
cd /path/to/Mesh-LLM/llama.cpp    # the fork checkout

# Add upstream if not already set
git remote add upstream https://github.com/ggml-org/llama.cpp.git

# Rebase our commits onto latest upstream
git fetch upstream
git rebase upstream/master

# Resolve any conflicts in source files, then:
#   git add <fixed files>
#   git rebase --continue

# Force-push (this is our fork, linear history, force-push is correct here)
git push origin master --force-with-lease
```

Then in the mesh-llm repo:

```bash
# Update the pinned SHA
cd /path/to/mesh-llm
echo "$(cd /path/to/Mesh-LLM/llama.cpp && git rev-parse HEAD)" > LLAMA_CPP_SHA

# Update local checkout
cd llama.cpp
git fetch origin
git checkout master
git reset --hard origin/master

# Build and test
cd ..
just build
cargo test -p mesh-llm

# Commit the SHA bump
git add LLAMA_CPP_SHA
git commit -m "bump llama.cpp to $(cat LLAMA_CPP_SHA | head -c 12)"
```

## Telling an agent to sync the fork

Give an agent these instructions:

> The llama.cpp fork is at `github.com/Mesh-LLM/llama.cpp`. It has our custom
> patches (RPC, MoE, mesh hooks) rebased on top of upstream `ggml-org/llama.cpp`.
>
> To sync with upstream:
> 1. Clone or cd into the fork: `git clone git@github.com:Mesh-LLM/llama.cpp.git`
> 2. `git remote add upstream https://github.com/ggml-org/llama.cpp.git` (if needed)
> 3. `git fetch upstream && git rebase upstream/master`
> 4. Fix any conflicts — our patches touch: `ggml-rpc.cpp`, `rpc-server.cpp`,
>    `llama-model.cpp`, `llama-graph.cpp`, `server-context.cpp`, `server-common.cpp`,
>    `server-task.h/cpp`, `common.h`, `arg.cpp`, `server-mesh-hook.h` (new file)
> 5. `git push origin master --force-with-lease`
> 6. In mesh-llm repo: update `LLAMA_CPP_SHA`, rebuild, test.
>
> The fork's `master` should always be: upstream HEAD + our 8 commits on top.
> Never merge — always rebase to keep linear history.

## Adding new patches to the fork

```bash
cd /path/to/Mesh-LLM/llama.cpp

# Edit files, build, test
# ...

# Commit
git add -p && git commit -m "description of change"

# Push
git push origin master

# In mesh-llm repo: update LLAMA_CPP_SHA
```

## Files we touch

**RPC (3 commits):**
- `ggml/include/ggml-rpc.h`
- `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `src/llama-model.cpp`
- `tools/rpc/rpc-server.cpp`

**MoE (4 commits):**
- `common/common.h` (hparams)
- `include/llama.h`
- `src/llama-graph.cpp`, `src/llama-graph.h`, `src/llama-hparams.h`, `src/llama-model.cpp`
- `tools/moe-analyze/` (new)
- `tools/moe-split/` (new)
- `tools/CMakeLists.txt`, `tests/CMakeLists.txt`

**Mesh hooks (1 commit):**
- `tools/server/server-mesh-hook.h` (new)
- `tools/server/server-context.cpp`
- `tools/server/server-common.cpp`, `tools/server/server-common.h`
- `tools/server/server-task.h`, `tools/server/server-task.cpp`
- `common/common.h`, `common/arg.cpp`
