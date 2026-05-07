# M6 - World Rank Snapshot (CUDA RTX 4070 Ti SUPER)

## Goal

- Establish the first reproducible M6 CUDA ranking snapshot for ARLE-CUDA vs
  vLLM on the local RTX 4070 Ti SUPER host.

## Hypothesis

- M4.5 should keep ARLE from sticking under GuideLLM pressure, and ARLE should
  be competitive enough to win at least 4 of 8 CUDA score cells
  (TTFT p50 and output tok/s across four workloads).

## Environment

- **Host:** Linux, NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM, driver
  595.71.05
- **CUDA:** 13.2.78 (`Build cuda_13.2.r13.2/compiler.37668154_0`)
- **ARLE commit:** `48d31ace09f8` (`docs(bench): note .venv/bin PATH precondition for guidellm wrapper`)
- **ARLE feature set:** `cargo build --release -p infer --no-default-features --features cuda`
- **Model:** `/home/ckl/projects/arle/infer/models/Qwen3-4B`
- **GuideLLM:** project `.venv`, wrapper `scripts/bench_guidellm.sh`
- **vLLM:** `vllm 0.20.1`, `torch 2.11.0+cu130`, `transformers 5.8.0`
- **vLLM attention backend:** `TRITON_ATTN`
- **KV dtype:** ARLE auto-FP8 paged KV, vLLM `--kv-cache-dtype fp8`
- **Metal half:** pending-remote, needs Apple Silicon runner.

## Commands

ARLE build:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo build --release -p infer --no-default-features --features cuda
```

ARLE server for prefill-heavy, decode-heavy, and high-conc:

```bash
RUST_LOG=info RUST_BACKTRACE=full \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/infer \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 5120
```

ARLE server for longctx-32k:

```bash
RUST_LOG=info RUST_BACKTRACE=full \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/infer \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 4 \
  --max-seq-len 33792 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --mem-fraction-static 0.92
```

vLLM server for prefill-heavy and decode-heavy:

```bash
PATH=/tmp/arle-vllm-venv/bin:$PATH \
NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-14' \
CC=/usr/bin/gcc-14 \
CXX=/usr/bin/g++-14 \
CUDAHOSTCXX=/usr/bin/g++-14 \
CUDA_VISIBLE_DEVICES=0 \
/tmp/arle-vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 5120 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --no-enable-log-requests \
  --uvicorn-log-level warning
```

vLLM server for high-conc:

```bash
PATH=/tmp/arle-vllm-venv/bin:$PATH \
NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-14' \
CC=/usr/bin/gcc-14 \
CXX=/usr/bin/g++-14 \
CUDAHOSTCXX=/usr/bin/g++-14 \
CUDA_VISIBLE_DEVICES=0 \
/tmp/arle-vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --no-enable-log-requests \
  --uvicorn-log-level warning
```

vLLM server for longctx-32k:

```bash
PATH=/tmp/arle-vllm-venv/bin:$PATH \
NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-14' \
CC=/usr/bin/gcc-14 \
CXX=/usr/bin/g++-14 \
CUDAHOSTCXX=/usr/bin/g++-14 \
CUDA_VISIBLE_DEVICES=0 \
/tmp/arle-vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 33792 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --no-enable-log-requests \
  --uvicorn-log-level warning
```

GuideLLM invocation shape:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh <label> \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --concurrencies <c> \
  --max-seconds 120 \
  --warmup 5 \
  --data '<prompt/output token shape>'
```

Workload shapes:

| workload | GuideLLM data | c |
|---|---|---:|
| prefill-heavy | `prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=16,output_tokens_stdev=1,output_tokens_min=16,output_tokens_max=16` | 1 |
| decode-heavy | `prompt_tokens=128,prompt_tokens_stdev=1,prompt_tokens_min=128,prompt_tokens_max=128,output_tokens=2048,output_tokens_stdev=1,output_tokens_min=2048,output_tokens_max=2048` | 1 |
| longctx-32k | `prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256` | 4 |
| high-conc | `prompt_tokens=1024,prompt_tokens_stdev=1,prompt_tokens_min=1024,prompt_tokens_max=1024,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256` | 64 |

## Server Envelopes

| backend | workload group | capacity notes |
|---|---|---|
| ARLE | prefill/decode/high | `max_num_batched_tokens=16384`, `chunked_prefill_size=2048`, `max_prefill_tokens=16384`, `mem_fraction_static=0.85`, `max_slots=14`; FP8 paged pool 51,520 tokens / 4.1 GB |
| ARLE | longctx-32k | `num_slots=4`, `max_seq_len=33792`, `max_prefill_tokens=2048`, `mem_fraction_static=0.92`; FP8 paged pool 69,728 tokens / 5.6 GB |
| vLLM | prefill/decode | GPU KV cache 69,872 tokens, max concurrency 13.65x at 5120 tokens |
| vLLM | high-conc | GPU KV cache 67,792 tokens, max concurrency 33.10x at 2048 tokens |
| vLLM | longctx-32k | GPU KV cache 81,216 tokens, max concurrency 2.40x at 33,792 tokens |

## Results

Median of three runs. Error is max relative deviation across the tracked
columns in that row; throughput itself was stable except for ARLE longctx-32k.

| workload | c | backend | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | out tok/s | req/s | run maxdev |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| prefill-heavy | 1 | ARLE | 518.0 | 571.8 | 15.70 | 15.74 | 20.98 | 1.304 | 0.6% |
| prefill-heavy | 1 | vLLM | 516.4 | 517.5 | 13.91 | 14.00 | 22.18 | 1.383 | 1.1% |
| decode-heavy | 1 | ARLE | 27.2 | 331.7 | 14.45 | 14.46 | 69.17 | 0.035 | 28.3% |
| decode-heavy | 1 | vLLM | 23.1 | 310.8 | 13.75 | 13.75 | 72.71 | 0.035 | 97.4% |
| longctx-32k | 4 | ARLE | 60337.7 | 77712.8 | 37.02 | 98.59 | 13.18 | 0.046 | 65.6% |
| longctx-32k | 4 | vLLM | 35942.0 | 45852.8 | 56.72 | 59.97 | 20.25 | 0.070 | 0.2% |
| high-conc | 64 | ARLE | 1059.0 | 8652.4 | 30.81 | 31.80 | 413.59 | 1.678 | 1.6% |
| high-conc | 64 | vLLM | 1606.2 | 6688.1 | 44.63 | 68.85 | 1114.71 | 4.530 | 92.1% |

## Scorecard

The P0 first cut is complete, but the M6 CUDA acceptance target is not met on
this local 16 GiB card. ARLE wins 1 of 8 score cells.

| workload | TTFT p50 winner | out tok/s winner | ARLE out tok/s vs vLLM |
|---|---|---|---:|
| prefill-heavy | vLLM | vLLM | -5.4% |
| decode-heavy | vLLM | vLLM | -4.9% |
| longctx-32k | vLLM | vLLM | -34.9% |
| high-conc | ARLE | vLLM | -62.9% |

## ARLE EngineTelemetry

The ARLE runs used `/v1/stats` through the GuideLLM wrapper trace. vLLM has no
compatible `/v1/stats`, so its service-side rows are `n/a`.

| workload | peak active | peak waiting | peak running_batch | peak prefill_queue | peak kv_util | final drain |
|---|---:|---:|---:|---:|---:|---|
| prefill-heavy | 1 | 0 | 1 | 0 | 81.8% | `active=0 waiting=0` |
| decode-heavy | 1 | 0 | 1 | 0 | 78.1% | `active=0 waiting=0` |
| longctx-32k | 2 | 2 | 2 | 1 | 94.7% | `active=0 waiting=0` after wait |
| high-conc | 14 | 50 | 14 | 9 | 75.8% | `active=0 waiting=0` |

## Problems

- ARLE did not satisfy the roadmap "win at least 4/8 workload cells" acceptance
  on this host. Follow-up plan: [m6-cuda-vllm-gap-followups.md](../../plans/m6-cuda-vllm-gap-followups.md).
- ARLE longctx-32k is memory-capacity limited on this 16 GiB card. The T0 pool
  holds 69,728 tokens, while four full 32k requests need roughly 132k tokens; it
  therefore peaked at `active=2 waiting=2`.
- ARLE longctx run-to-run variance exceeded the 5% target. The median uses
  `r1`, `r2`, and clean replacement `r3b`; `r2` had a delayed post-run drain and
  `r3b` was waited to `active=0 waiting=0`.
- vLLM decode-heavy and high-conc TTFT tails varied heavily across the first
  and later runs, although output tok/s was stable.
- vLLM default FlashInfer JIT failed on this CUDA 13.2 / system GCC 16 host.
  The reproducible baseline required `--attention-backend TRITON_ATTN` plus
  `NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-14'`.
- vLLM longctx startup failed at `--gpu-memory-utilization 0.92` because free
  memory was 14.09 GiB and the request wanted 14.32 GiB. Retrying with 0.90
  started cleanly.

## Learnings

- The M4.5 drain fix held under M6 pressure: high-conc reached
  `active=14 waiting=50`, longctx reached `active=2 waiting=2`, and both drained.
- On 16 GiB, 32k-concurrency results mostly measure KV residency policy and
  preemption behavior, not peak model math throughput.
- vLLM's strongest local advantage is high-conc throughput; ARLE's strongest
  local result is lower high-conc TTFT p50.
- The M1 telemetry surface is useful enough for bench triage: ARLE can report
  active slots, waiting queues, prefill queues, KV utilization, and final drain
  while vLLM cannot be compared through the same `/v1/stats` contract.

## Artefacts

| workload | ARLE raw labels | vLLM raw labels |
|---|---|---|
| prefill-heavy | `2026-05-07-m6-arle-prefill-heavy-r1`, `r2`, `r3` | `2026-05-07-m6-vllm-prefill-heavy-r1`, `r2`, `r3` |
| decode-heavy | `2026-05-07-m6-arle-decode-heavy-r1`, `r2`, `r3` | `2026-05-07-m6-vllm-decode-heavy-r1`, `r2`, `r3` |
| longctx-32k | `2026-05-07-m6-arle-longctx-32k-r1`, `r2`, `r3b` | `2026-05-07-m6-vllm-longctx-32k-r1`, `r2`, `r3` |
| high-conc | `2026-05-07-m6-arle-high-conc-r1`, `r2`, `r3` | `2026-05-07-m6-vllm-high-conc-r1`, `r2`, `r3` |

Each artifact directory contains `benchmarks.json`, `benchmarks.csv`,
`benchmarks.html`, `guidellm.log`, and service trace files. vLLM service trace
summaries contain `n/a` for ARLE-specific `/v1/stats` fields.

## Status

- **M6 P0 CUDA + vLLM:** complete, but acceptance red.
- **M6 P1 CUDA + SGLang:** pending.
- **M6 P2 CUDA + TRT-LLM:** pending / optional.
- **M6 P3 Metal:** pending-remote, needs Apple Silicon runner.
