# Axis 3 Chunked Prefill Size KILL

## Context

`docs/plans/2026-05-25-cuda-perf-codex-collab.md` Axis 3 proposed raising
the L4 chunked-prefill size from the resolved default `2048` to `4096` or
`8192`, while keeping `max_prefill_tokens=16384`. Axis 2 failed first and was
reverted, so this test ran as an independent Split-policy experiment against
the same-day baseline.

No production default was changed. Both variants were tested with explicit
CLI overrides on the L4:

- `--chunked-prefill-size 4096`
- `--chunked-prefill-size 8192`

The existing admission path already allows multiple prefill chunks per tick for
Qwen3 dense: `prefill_max_requests=none`, `max_prefill_tokens=16384`, and
Qwen3 dense does not cap `max_concurrent_prefill_requests`.

## Command

```bash
# sync only the Axis 2 revert files back to the L4 to avoid unrelated local
# DeepSeek/MoE worktree changes
rsync -av --dry-run --relative infer/src/main.rs infer/src/scheduler/types.rs \
  outdoors-arrow-guide-participate.trycloudflare.com:/content/workspace/agent-infer/
rsync -av --relative infer/src/main.rs infer/src/scheduler/types.rs \
  outdoors-arrow-guide-participate.trycloudflare.com:/content/workspace/agent-infer/

ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   CUDA_HOME=/usr/local/cuda PATH=/root/.cargo/bin:$CUDA_HOME/bin:$PATH \
   LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=8.9 \
   cargo build -p infer --release --features cuda'

# 4096 variant
ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   nohup setsid -f env CUDA_HOME=/usr/local/cuda \
     PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH \
     LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=8.9 \
     ./target/release/infer --model-path infer/models/Qwen3-4B \
       --port 8000 --num-slots 16 --max-seq-len 5120 \
       --chunked-prefill-size 4096 \
       > /tmp/arle_server_axis3_4096.log 2>&1 < /dev/null'
ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   scripts/bench_guidellm.sh axis3-chunk4096 \
     --target http://localhost:8000 --model Qwen3-4B \
     --processor infer/models/Qwen3-4B \
     --concurrencies 1,4,8,16 --max-seconds 60 --warmup 5'

# 8192 variant
ssh outdoors-arrow-guide-participate.trycloudflare.com 'pkill -x infer || true'
ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   nohup setsid -f env CUDA_HOME=/usr/local/cuda \
     PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH \
     LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=8.9 \
     ./target/release/infer --model-path infer/models/Qwen3-4B \
       --port 8000 --num-slots 16 --max-seq-len 5120 \
       --chunked-prefill-size 8192 \
       > /tmp/arle_server_axis3_8192.log 2>&1 < /dev/null'
ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   scripts/bench_guidellm.sh axis3-chunk8192 \
     --target http://localhost:8000 --model Qwen3-4B \
     --processor infer/models/Qwen3-4B \
     --concurrencies 1,4,8,16 --max-seconds 60 --warmup 5'
```

## Environment

- L4 23034 MiB, driver 580.82.07, CUDA 12.8 build env
- Model: `infer/models/Qwen3-4B`
- Runtime flags shared by both variants: `--num-slots 16 --max-seq-len 5120`
- Graph warmup: server logs showed CUDA Graph capture succeeded for batched
  decode `B=1` through `B=16`.
- 4096 envelope: `chunked_prefill_size=4096`, `max_prefill_tokens=16384`,
  `prefill_max_requests=none`
- 8192 envelope: `chunked_prefill_size=8192`, `max_prefill_tokens=16384`,
  `prefill_max_requests=none`

## Results

Raw artefacts:

- `/content/workspace/agent-infer/bench-output/2026-05-25-axis3-chunk4096`
- `/content/workspace/agent-infer/bench-output/2026-05-25-axis3-chunk8192`

### 4096

Plan labels:

```text
idle=20755, decode=4597, prefill=56, split=0, mixed=0
```

| c | TTFT p50 ms | ITL p50 ms | out tok/s |
|---:|---:|---:|---:|
| 1 | 714.2 | 35.94 | 26.21 |
| 4 | 3027.0 | 43.75 | 75.09 |
| 8 | 6716.0 | 52.12 | 115.12 |
| 16 | 14770.3 | 70.88 | 160.61 |

Delta vs same-day baseline
[`2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](../wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md):

| c | TTFT p50 | ITL p50 | out tok/s | Verdict |
|---:|---:|---:|---:|---|
| 1 | -4.2% | -0.6% | +0.7% | noise |
| 4 | +0.5% | -1.0% | -0.5% | noise |
| 8 | +13.7% | -1.4% | -1.9% | fail |
| 16 | +14.4% | -1.3% | -2.1% | fail |

4096 fails because TTFT regressed by more than 5% at c=8 and c=16.

### 8192

Plan labels:

```text
idle=19273, decode=4342, prefill=29, split=32, mixed=0
```

| c | TTFT p50 ms | ITL p50 ms | out tok/s |
|---:|---:|---:|---:|
| 1 | 702.2 | 36.00 | 26.19 |
| 4 | 2394.7 | 46.59 | 61.95 |
| 8 | 5580.9 | 59.84 | 72.50 |
| 16 | 11577.5 | 97.50 | 68.32 |

Delta vs baseline:

| c | TTFT p50 | ITL p50 | out tok/s | Verdict |
|---:|---:|---:|---:|---|
| 1 | -5.8% | -0.4% | +0.7% | TTFT win only |
| 4 | -20.5% | +5.5% | -17.9% | fail |
| 8 | -5.5% | +13.3% | -38.2% | fail |
| 16 | -10.3% | +35.7% | -58.3% | fail |

8192 improves c=16 TTFT p50 by more than 10%, but ITL and output throughput
regress far beyond the acceptance gate.

## Root Cause

This axis is a tradeoff, not a free win:

- 4096 preserves decode throughput but increases c=16 TTFT. Larger chunks
  reduce per-request chunk count, but each prefill step becomes heavier and
  the FIFO queue still serializes enough work to hurt TTFT at high c.
- 8192 reduces TTFT by front-loading larger prefill work, but it starves decode:
  c=16 ITL p50 regressed 71.85 ms -> 97.50 ms and out tok/s dropped
  164.00 -> 68.32.

The planner can already admit multiple prefill requests per step under the
token budget; changing chunk size alone does not fix the admission starvation
seen in the plan.

## Fix

Do not change the L4 default chunked-prefill size. Keep the resolved default at
2048 for this workload. No code change landed.

## Problems

- `cargo build --release --features cuda` from the remote root is polluted by
  stale root-crate files in `/content/workspace/agent-infer`; `cargo build -p
  infer --release --features cuda` is the correct server build command for the
  synced source used here.
- The local worktree contained unrelated DeepSeek/MoE modifications, so Axis 3
  used path-specific rsync for `infer/src/main.rs` and
  `infer/src/scheduler/types.rs` instead of syncing all of `infer/src`.

## Rule

Do not default larger chunked-prefill sizes from TTFT alone. The acceptance
gate must include ITL and output throughput because larger chunks can move the
loss from prefill queueing into decode starvation.
