# Unified L3 direct storage — CUDA DSv4, 2026-07-16

> Status: pending-remote

## Goal

Add one L3 backend for inference and optional checkpoint spill without reducing
DeepSeek-V4-Flash-FP8 TP=4 serving throughput or increasing payload write
amplification above 1.01×.

## Hypothesis

Batching KV pages through aligned `O_DIRECT` + `io_uring` removes page-cache
copies on local NVMe; mmap remains the default until the target mount wins.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8123 --model default \
  --prompts-jsonl /host/dsv4-kv-readme-20.jsonl \
  --concurrency-grid 1,4,8,16 --requests-per-concurrency 20 \
  --max-tokens 96 --temperature 0 --timeout-seconds 300 \
  --output /host/dsv4-mmap-new/bench
```

- Binary: `sha256:a64b5024c413af9f46452b53c0a0d43109870fb582492017831ab4100b766353`
- Prompt: one 1,649-token README prompt, repeated 20 times
- Completion: 96 tokens
- KV: 512 MiB host, 4 GiB disk, mmap control

## Environment

- 4× NVIDIA H20, driver 535.161.08, CUDA 12.9
- DeepSeek-V4-Flash-FP8, TP=4, EP=4, GPUs 0-3
- Automatic NUMA: ranks 0-3 pinned to node 0 cores 0-21, 22-43, 44-65,
  66-87
- L3 mount: `/host`, ext4 on virtio `/dev/vda2`; no mounted local NVMe

## Results

| concurrency | completed | TTFT p99 ms | ITL p99 ms |
| ---: | ---: | ---: | ---: |
| 1 | 20/20 | 472.0 | 42.0 |
| 4 | 20/20 | 1,130.2 | 99.3 |
| 8 | 20/20 | 1,651.0 | 95.6 |
| 16 | 20/20 | 3,307.7 | 122.6 |

The latest historical DSv4 mmap L2+L3 run was 40.00/72.86/104.60/119.96
tok/s at c=1/4/8/16. The directional deltas are +0.9%/+0.3%/+5.0%/+1.6%; this
is cross-binary evidence, not a licensed matched A/B.

- Rank-0 L3 counters: 11.40 GB useful reads, 12.63 GB useful writes, zero mmap
  I/O failures. The multiprocess stats endpoint queries rank 0 only.
- 2 GiB substrate: mmap 1.14 GiB/s write and 2.40 GiB/s warm read; direct QD32
  0.19 GiB/s write and 0.18 GiB/s read.
- Real DSv4 direct probe: 142,749,912 useful read bytes versus 142,786,560
  submitted; 285,499,824 useful write bytes versus 285,573,120 submitted.
  Read and write amplification are both 1.00026× versus the historical 1.0×
  payload-only baseline.
- Qwen3-4B matched warm process-to-ready median: L3 off 2.446 s, direct enabled
  2.444 s (-0.08%). DSv4 warm process-to-ready: 30.506 s; historical cold local
  NVMe: 80.95 s.

Raw artifacts: `/host/dsv4-mmap-new/bench-ignore-eos.{json,csv}`,
`/host/dsv4-direct-new/{prime.json,stats-after-prime.json,server3.log}`.

## Problems

The canonical runner marks a first-token EOS as incomplete because no text SSE
event exists. The throughput-only A/B used the same temporary runner with
`ignore_eos=true`; an unmodified request separately produced coherent output.

Direct QD32 on virtio returned transient `EAGAIN` under the full grid and one
`EFAULT`; that grid is invalid. The engine now defaults to QD8 and retries three
`EAGAIN` completions. A zero-failure local-NVMe rerun remains required.

TP=8 was not run: GPUs 4, 6 and 7 were occupied by unrelated training jobs; no
process was preempted. The remote tunnel dropped before the QD8 rerun.

GDS correctly stayed off: cuFile compatibility mode is enabled, `nvidia_fs` is
absent, P2PDMA is disabled, and the container has no mounted NVMe filesystem.

## Learnings

SHIP the unified backend with mmap default and explicit direct opt-in. KILL
direct on this virtio mount. The next gate is a same-binary TP=4/8 rerun on a
mounted local NVMe with zero tier failures and a positive wall-clock delta.
