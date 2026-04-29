# Server Warmup Readiness Gate

## Context

The CUDA server could bind HTTP before scheduler warmup finished. On the L4
c=16 bench shape, `/v1/models` could therefore become reachable while CUDA
graph capture, FlashInfer decode planning, and cublasLt autotune were still
running inside the scheduler thread. Bench runs worked around this by sending a
manual 8-token warmup request before guidellm.

## What Worked

ARLE now keeps the HTTP listener closed until `Scheduler::warmup_cuda_graphs()`
finishes and the scheduler thread sends its readiness signal. The smoke run on
Qwen3-4B, `--num-slots 16 --max-seq-len 4352 --kv-cache-dtype fp8`, observed:

| event | UTC time |
|---|---:|
| wait for scheduler warmup | 13:36:58.312 |
| CUDA graph / GEMM warmup done | 13:37:05.856 |
| scheduler run loop started | 13:37:05.856 |
| HTTP listener opened | 13:37:05.913 |

`curl /v1/models` returned connection failures before the listener opened and
HTTP 200 immediately after warmup. This means guidellm can start from normal
readiness without an out-of-band synthetic warmup request.

## Verification

```bash
CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig CUDA_HOME=/usr/local/cuda \
  cargo check -p infer --features cuda

CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig CUDA_HOME=/usr/local/cuda \
  cargo test -p infer --features cuda scheduler_runtime_guard --lib

CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig CUDA_HOME=/usr/local/cuda \
  cargo build --release -p infer --features cuda
```

## Rule

External readiness must be downstream of runtime warmup. If startup has to
capture graphs, plan kernels, or autotune GPU kernels, the server should not
bind its public listener until that work is done.
