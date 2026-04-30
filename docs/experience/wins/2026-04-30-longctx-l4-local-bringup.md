# Longctx L4 Local Bring-Up

## Context

Phase 1 S5 for `docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
needs a real L4 CUDA host, local Qwen3-4B weights, GuideLLM, TileLang, Zig,
and an ARLE CUDA server launched with the longctx FP8 KV envelope.

This workspace has an NVIDIA L4 and enough disk, but the first build attempts
hit environment blockers before the server could run.

## What Worked

- Installed bench dependencies with `pip install -e '.[bench]'`.
- Downloaded `Qwen/Qwen3-4B` into the HuggingFace cache and linked it at
  `infer/models/Qwen3-4B` so both ARLE and SGLang scripts use the same path.
- Avoided the Drive/FUSE Cargo registry stall by setting:

```bash
CARGO_HOME=/tmp/arle-cargo-home
CARGO_TARGET_DIR=/tmp/arle-target
```

- Installed TileLang with `pip install -e '.[tilelang]'`.
- Used the repo Zig toolchain:

```bash
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig
```

- Built the CUDA server:

```bash
CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TRITON_PYTHON=/usr/bin/python3 \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
CARGO_HOME=/tmp/arle-cargo-home \
CARGO_TARGET_DIR=/tmp/arle-target \
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
cargo build -p infer --release --no-default-features --features cuda --bin infer
```

Result: pass, `release` profile finished in 7m40s.

Server launch:

```bash
/tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs
```

Startup evidence:

| metric | value |
|---|---:|
| GPU | NVIDIA L4 |
| CUDA driver | 580.82.07 / CUDA 13.0 runtime host |
| nvcc | `/usr/local/cuda/bin/nvcc` |
| model | Qwen3-4B |
| weight shards | 3 |
| paged KV format | FP8E4M3 |
| slots | 16 |
| max seq len | 131072 |
| TokenKVPool max tokens | 136976 |
| TokenKVPool budget | 11.0 GB |

GuideLLM smoke:

```bash
scripts/bench_guidellm.sh longctx-s4-arle-smoke \
  --workload longctx-32k \
  --smoke \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Result: pass.

| metric | value |
|---|---:|
| shape | 512 input / 16 output / c=1 / 5s |
| output tok/s | 26.59 |
| total tok/s | 879.29 |
| input tok/s | 941.6 |
| TTFT p50 | 100.9 ms |
| ITL p50 | 34.06 ms |
| E2E mean | 0.61 s |
| peak waiting | 0 |
| peak active | 1 |
| peak kv_util | 3.4% |

Artefacts:

- Raw: `bench-output/2026-04-30-longctx-s4-arle-smoke/`
- Service trace:
  `bench-output/2026-04-30-longctx-s4-arle-smoke/service_stats_trace_summary.md`

## Rule

On this workspace, do CUDA builds with local `/tmp` Cargo state, not the
Drive/FUSE Cargo registry. Keep `ZIG`, `INFER_TILELANG_PYTHON`, and
`TORCH_CUDA_ARCH_LIST=8.9` explicit for L4 longctx work.
