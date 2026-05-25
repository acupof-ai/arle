# Axis 2 Mixed Default KILL

## Context

`docs/plans/2026-05-25-cuda-perf-codex-collab.md` Axis 2 proposed changing
the CUDA serving default from `SchedulerMixedPolicy::Split` to `Mixed`.
The target was Qwen3-4B dense BF16 weights on the L4 box, not the Qwen3.5 W4
hybrid path killed by
[`2026-05-09-mixed-policy-ab-killed-w4hybrid-restriction.md`](2026-05-09-mixed-policy-ab-killed-w4hybrid-restriction.md).

The local patch made the default Mixed for CUDA while preserving the existing
`model.supports_mixed_batch(kv_pool_format)` runtime gate. It was reverted
after the L4 c-sweep below failed the acceptance gate.

## Command

```bash
# local structural checks
cargo fmt --check
cargo test --release
cargo check -p infer --no-default-features --features metal,no-cuda

# local CUDA typecheck attempted, but this Mac has no nvcc:
cargo check -p infer --no-default-features --features cuda,no-cuda

# sync to L4
rsync -av --dry-run --delete --exclude target --exclude .git --exclude infer/models \
  ./infer/src/ outdoors-arrow-guide-participate.trycloudflare.com:/content/workspace/agent-infer/infer/src/
rsync -av --delete --exclude target --exclude .git --exclude infer/models \
  ./infer/src/ outdoors-arrow-guide-participate.trycloudflare.com:/content/workspace/agent-infer/infer/src/

# build actual server package on L4
ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   CUDA_HOME=/usr/local/cuda PATH=/root/.cargo/bin:$CUDA_HOME/bin:$PATH \
   LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=8.9 \
   cargo build -p infer --release --features cuda'

# server
ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   CUDA_HOME=/usr/local/cuda PATH=/root/.cargo/bin:$CUDA_HOME/bin:$PATH \
   LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=8.9 \
   nohup setsid ./target/release/infer --model-path infer/models/Qwen3-4B \
     --port 8000 --num-slots 16 --max-seq-len 5120 \
     > /tmp/arle_server.log 2>&1 < /dev/null &'

# bench
ssh outdoors-arrow-guide-participate.trycloudflare.com \
  'cd /content/workspace/agent-infer && \
   scripts/bench_guidellm.sh axis2-mixed-default \
     --target http://localhost:8000 --model Qwen3-4B \
     --processor infer/models/Qwen3-4B \
     --concurrencies 1,4,8,16 --max-seconds 60 --warmup 5'
```

## Environment

- L4 23034 MiB, driver 580.82.07, CUDA 12.8 build env
- ARLE source synced to `/content/workspace/agent-infer`, server package built
  with `cargo build -p infer --release --features cuda`
- Model: `infer/models/Qwen3-4B`
- Runtime flags: `--num-slots 16 --max-seq-len 5120`
- Scheduling envelope from server log: `chunked_prefill_size=2048`,
  `max_num_batched_tokens=16384`, `max_prefill_tokens=16384`,
  `prefill_max_requests=none`
- Graph warmup: server log showed CUDA Graph capture succeeded for batched
  decode `B=1` through `B=16`.

## Results

Raw artefacts:
`/content/workspace/agent-infer/bench-output/2026-05-25-axis2-mixed-default`.

Plan labels from service trace:

```text
idle=22690, decode=3587, prefill=46, split=0, mixed=12
```

So Axis 2 did activate the Mixed path; this was not a silent Split fallback.

| c | TTFT p50 ms | ITL p50 ms | out tok/s |
|---:|---:|---:|---:|
| 1 | 726.8 | 35.91 | 26.21 |
| 4 | 2938.2 | 51.03 | 35.98 |
| 8 | 5791.5 | 53.86 | 41.13 |
| 16 | 12355.4 | 71.24 | 131.71 |

Delta vs same-day baseline
[`2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](../wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md):

| c | TTFT p50 | ITL p50 | out tok/s | Verdict |
|---:|---:|---:|---:|---|
| 1 | -2.5% | -0.6% | +0.7% | noise |
| 4 | -2.5% | +15.5% | -52.3% | fail |
| 8 | -2.0% | +1.9% | -65.0% | fail |
| 16 | -4.3% | -0.8% | -19.7% | fail |

The explicit acceptance baseline in the plan was c=16: TTFT p50 12913 ms,
ITL p50 71.85 ms, out tok/s 164. Mixed improved c=16 TTFT by only 4.3% and
regressed output throughput by 19.7%, exceeding the 5% revert threshold.

## Root Cause

The measured failure is policy-level, not a silent gating bug:

- `mixed=12` and `split=0` prove the default change reached real mixed ticks.
- The p50 TTFT gain at c=16 was below the 10% win threshold.
- Completed output throughput regressed at c=4, c=8, and c=16.
- c=4 and c=8 also showed ITL tail instability (`ITL p99/p50` 2.07 and 2.88
  in the bench warning).

Exact kernel-level attribution is deferred. The safe conclusion is narrower:
Qwen3-4B dense mixed default is not licensed by this c-sweep and must stay
opt-in.

## Fix

Reverted the default change. `SchedulerMixedPolicy` remains `Split` by default;
operators can still opt into `--scheduler-mixed-policy mixed` for targeted
experiments guarded by `model.supports_mixed_batch(...)`.

## Problems

- Local `cargo check -p infer --no-default-features --features cuda,no-cuda`
  cannot run on this Mac because `cudarc` requires `nvcc --version` and no local
  `nvcc` exists. The CUDA build was therefore performed on the L4.
- The remote root crate had stale root-level files that made
  `cargo build --release --features cuda` compile the wrong root surface.
  Building the actual server package with `cargo build -p infer --release
  --features cuda` succeeded and produced `target/release/infer`.

## Rule

Mixed default must not land from source audit alone. Even when
`supports_mixed_batch` is true and plan labels show real Mixed ticks, the
decision is still governed by c-sweep wall-clock evidence. A small TTFT
improvement does not compensate for a larger output-throughput regression.
