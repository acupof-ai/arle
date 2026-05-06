# M4 GuideLLM Canonical Sweep Stuck Under KV Pressure

## Context

Track B M4 unified the op dispatch surface for Qwen3 CUDA and added a Metal
`OpsBackend` implementor. After the code slices and CUDA verification passed,
the required GuideLLM regression check was run on the local RTX 4070 Ti SUPER.

Runtime code included the M4 commits through `cf78986`:

- `8ce20e0 feat(ops): scaffold unified backend trait`
- `961ac13 refactor(qwen3): route norm ops through backend trait`
- `477b761 refactor(qwen3): route linear ops through backend trait`
- `c70ad34 refactor(qwen3): route elementwise and embedding ops through backend trait`
- `5f209d2 refactor(qwen3): route sampling ops through backend trait`
- `84caef0 feat(metal): implement unified ops backend`
- `5e5784f fix(cuda): guard batched rmsnorm vector alignment`
- `cf78986 test(scheduler): align admission budget expectations`

Server:

```bash
RUST_LOG=info \
RUST_BACKTRACE=full \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run --release -p infer --no-default-features --features cuda -- \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 5120
```

The server auto-selected `max_slots=14`, `max_num_batched_tokens=16384`,
`chunked_prefill_size=2048`, and FP8E4M3 paged KV.

Bench:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh cuda-m4-unified-ops \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Artifacts:

- `bench-output/2026-05-07-cuda-m4-unified-ops/`
- `bench-output/2026-05-07-cuda-m4-unified-ops/service_stats_trace_summary.md`

## Symptoms

This is not the earlier 4097-token rejection from the M3 first attempt. With
`--max-seq-len 5120`, canonical GuideLLM requests were accepted and generated
tokens before the scheduler stopped making forward progress.

Final trace summary:

```text
Samples: 403
Peak waiting: 560
Peak active: 12
Peak running_batch: 12
Peak prefill_queue: 10
Peak kv_util: 99.8%
Plan labels: idle=242807,decode=4085,prefill=49,split=0,mixed=39
```

Final `/v1/stats` snapshot:

```text
requests=39 active=12 waiting=560 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=12 prefill_queue=0 batch_width=0 decode_tokens=0 prefill_tokens=0 tokens_out=9607 ... kv_util=95.5% ... engine_queue_depth=560 engine_active_requests=12 engine_batch_occupancy=0.9551
```

After terminating the GuideLLM client process group, the server still reported:

```text
requests=39 active=12 waiting=560 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=12 prefill_queue=0 ... engine_queue_depth=560 engine_active_requests=12
```

The server was then terminated manually so the GPU was not left occupied.

## Current Read

M4 code verification passed, but the M4 performance gate did not. This run did
not produce a trustworthy `benchmarks.json`, and it reproduced the red-light
shape from `2026-05-07-m3-guidellm-bench-stuck.md` on the canonical workload:
active slots remain resident, queued work grows, and the scheduler emits idle
plans while no decode or prefill rows are runnable.

The evidence points at a scheduler/runtime cleanup or memory-pressure recovery
bug rather than GuideLLM infrastructure:

- accepted requests generated `9607` tokens before the stall;
- `kv_util` peaked at `99.8%`;
- active slots stayed resident after clients were killed;
- `kv_fetch_q`, `kv_fetch_waiters`, and `kv_store_q` stayed at zero, so the
  visible stall is not waiting on KV-tier IO.

This blocks claiming M4 bench acceptance. It also means M5 should not start on
this machine until the canonical GuideLLM stuck-active path is fixed or
explicitly waived.

## Fix

Pending. The next concrete checks should instrument the CUDA scheduler loop at
the point where `running_batch > 0`, `waiting > 0`, `scheduled == 0`, and both
`decode_rows` and `prefill_rows` are zero:

- per-slot phase, generated-token count, target-token count, and client
  receiver state;
- KV page ownership and remaining allocatable pages;
- whether slot cleanup observes disconnected streams after the HTTP client is
  killed;
- whether admission/preemption has a path to free slots when KV utilization is
  near full.

## Rule

Do not publish M4 GuideLLM numbers unless the run produces benchmark output and
the service drains to `active=0 waiting=0` after the client exits. A
stuck-active trace is an error artifact, not a performance result.
