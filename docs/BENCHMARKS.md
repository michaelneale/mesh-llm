# Benchmarks

These numbers are a quick reality check, not a universal promise.

## Example results

GLM-4.7-Flash-Q4_K_M (17GB), tested on an M4 Max and a Mac mini M4 over Wi-Fi:

| Configuration | tok/s |
|---|---|
| Solo (no mesh) | 68 |
| 2-node split (85/15) | 21 |
| 3-node split (62/31/8) | 12-13 |

Cross-network from Sydney to Queensland at roughly 20ms RTT measured 10-25 tok/s. In those runs, the overhead was dominated by per-token RPC latency.

## Notable implementation win

Stock llama.cpp RPC transfers about 16.88GB on connect.

This fork uses local GGUF loading on peers, which cuts that to:

- 0 bytes transferred
- about 9 seconds to connect

For deeper design and performance notes, see [design/DESIGN.md](design/DESIGN.md).
