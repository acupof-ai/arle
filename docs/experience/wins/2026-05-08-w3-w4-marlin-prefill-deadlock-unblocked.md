# W3/W4 Marlin prefill deadlock unblocked

## Context

W3 c=16 and W4 c=8 both showed the same production-shape deadlock
fingerprint before this fix:

| Workload | Before |
|---|---|
| W3 c=16 | `active=16`, `prefill_queue=15`, `prefill_rows=0`, `tokens_out=0` |
| W4 c=8 | `active=8`, `prefill_queue=7`, `prefill_rows=0`, `tokens_out=0` |

The shared signature ruled out a harness-only issue. A rebuilt W4A16 Marlin
server exposed the missing failure edge in the scheduler log:

```text
thread '<unnamed>' panicked at infer/src/ops/linear.rs:724:68:
alloc y_fp16: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

The HTTP server stayed alive after the scheduler thread died, so external
stats looked like an admission deadlock rather than an explicit model error.

## What Worked

Two fixes were needed together:

1. Prefill page budgeting now reserves decode-growth headroom only for active
   decoding slots, including temporarily emit-gated rows. Queued/prefill slots
   no longer consume current-step decode-growth budget.
2. Marlin prefill GEMM allocation/kernel failures now propagate as `Result`
   instead of panicking the scheduler thread. Qwen3 Marlin models also cap
   concurrent prefill requests at four rows per step, matching the real
   temporary FP16 scratch footprint on 16 GiB Ada.

This keeps dense BF16 Qwen3 behavior unchanged while preventing W4A16/W4A8
Marlin prefill from admitting a token-budget-valid but scratch-invalid burst.

## Results

Server:

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 16 --max-seq-len 9216
```

W3 c=16:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH timeout 900 \
python scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --num-concurrent 16 \
  --label arle-w3-c16-admission-fix \
  --server http://localhost:8000 \
  --model Qwen3-4B-W4A16-sym-g128-marlin
```

W4 c=8:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH timeout 1200 \
python scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --num-concurrent 8 \
  --label arle-w4-c8-admission-fix \
  --server http://localhost:8000 \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --out bench-output/2026-05-08-arle-w4-c8-admission-fix.json
```

| Workload | Before | After | TTFT p50 / p99 | ITL p50 / p99 |
|---|---|---|---:|---:|
| W3 c=16 | 0-1 / 384 turns OK, frozen with `tokens_out=0` | 376 / 384 turns OK, `active=0 waiting=0 prefill_queue=0` after drain | 721.8 / 2604.8 ms | 13.1 / 14.7 ms |
| W4 c=8 | 0 / 256 turns OK, frozen with `tokens_out=0` | 256 / 256 turns OK, `active=0 waiting=0 prefill_queue=0` after drain | 7919.1 / 118402.6 ms | 16.8 / 17.1 ms |

W4 artifact:
`bench-output/2026-05-08-arle-w4-c8-admission-fix.json`

## Problems

- W3 c=16 still produced eight zero-token turns. The production deadlock is
  gone, but the harness/runtime should still explain those early terminations.
- W4 c=8 is unblocked but not performant: TTFT p99 remains extreme. The next
  issue is tail latency under 8K resume pressure, not scheduler liveness.
- Startup telemetry still prints `prefill_max_requests=none` because the cap is
  model-side. A follow-up should expose the effective model cap in `/v1/stats`
  or startup logs.

## Rule

CUDA hot-path allocations that can fail must return errors into the scheduler
control plane. Never `expect()` an allocation inside a scheduler-owned worker
thread: a panic leaves HTTP alive but the runtime unable to make progress,
which looks like an admission deadlock from the outside.
