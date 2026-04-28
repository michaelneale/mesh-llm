# RPC 切分 GGUF 只索引首个分片问题

## 背景

- 目标：在 Mac host + 两台 PGX worker 上启动 `Hermes-3-Llama-3.1-405B-Q8_0`。
- 切分：PGX A `0.14`，PGX B `0.14`，Mac `0.72`。
- 验证版本：`0.65.0-rc2+splitgguf.1`。

## 碰到的问题

- 多机加载时 RPC 失败：`Remote RPC server crashed or returned malformed response`。
- 直接错误：`send failed (bytes_sent=0, size_to_send=285212976)`，随后 `SIGABRT`。
- 表面现象：像是 QUIC/RTT/大 payload 超时问题。
- 实际现象：worker 本地明明有 GGUF shards，但仍在通过网络传大 tensor。

## 根因

- `SET_TENSOR_GGUF` 快路径只索引了 `--gguf` 传入的第一个 shard。
- 对 split GGUF，例如 `00001-of-00011.gguf`，后续 shard 里的 tensor 会被误判为本地不存在。
- 本地查找失败后，client 回退到原始 `SET_TENSOR`，把几百 MiB 到近 1 GiB 的 tensor 通过 RPC tunnel 发送。
- 所以问题不是“0.14 这一整块直接通过 QUIC 发走”，而是 split GGUF 本地查找失败导致 fallback。

## 解决方案

- 放宽 mesh RPC tunnel 上限：
  - `MAX_RPC_PAYLOAD_BYTES`: 256 MiB -> 1 GiB
  - `RPC_PAYLOAD_MAX_SECS`: 120 秒 -> 600 秒
- 新增 llama.cpp RPC patch：
  - `third_party/llama.cpp/patches/0007-rpc-index-split-GGUF-shards-for-local-tensor-loads.patch`
- patch 内容：
  - 根据首个 shard 文件名推导同组 split 文件。
  - 读取 `split.count` 并索引全部 GGUF shards。
  - 查找 tensor 时遍历所有 shard index。
  - 找到 tensor 后由 worker 从本地 GGUF shard 读取，不再走网络传大 tensor。

## 验证结果

- 三台机器版本一致：`mesh-llm 0.65.0-rc2+splitgguf.1`。
- Mac host 状态：`node_state=serving`，`llama_ready=true`。
- 两台 PGX worker 均在线并参与切分。
- `/v1/models` 返回 `Hermes-3-Llama-3.1-405B-Q8_0`。
- chat completion 测试返回 `OK`。
- PGX RPC 日志确认从本地 shard 加载 tensor：

```text
[set_tensor_gguf] loaded 'blk.29.attn_q.weight' (285212672 bytes) from local GGUF: ...00003-of-00011.gguf
[set_tensor_gguf] loaded 'blk.35.ffn_down.weight' (926941184 bytes) from local GGUF: ...00004-of-00011.gguf
```

## 提交前检查

- `cargo test -p mesh-llm rewrite`
- `cargo fmt --all -- --check`
- `cargo check -p mesh-llm`
- `just build`
- 两台 PGX 分别执行 `just build cuda 121`
- `scripts/prepare-llama.sh pinned`

## 注意事项

- 该修复是针对 split GGUF + llama.cpp RPC 本地加载快路径的通用修复，不是 Hermes 专用。
- 每个 worker 仍需要有完整 GGUF shards，否则仍可能 fallback 到原始 tensor 传输。
- `3131` console 当前只监听 `127.0.0.1`，远程访问需要 SSH tunnel。
- 不要把 IP、密码、token、SSH 私密信息写入 git 跟踪文件。

