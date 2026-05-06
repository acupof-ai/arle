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

## 2026-05-07 follow-up — likely root cause: missing preemption path

A grep over `infer/src/scheduler/cuda/` for `preempt|evict_slot|abort_request`
returns zero hits. The eviction path that does exist is
`Scheduler::evict_prefix_cache_for_allocation` (called from
`Scheduler::alloc_pool_tokens_with_retry`, `core.rs:1647-1656`), and it
only frees **un-referenced prefix-cache** blocks via
`RadixCache::evict_with_policy`. It does NOT touch pages owned by active
running requests.

That fits the observed deadlock shape:

- 12 active running requests each hold their KV pages.
- Each active request needs **1 more page** for the next decode step.
- `kv_util` ≈ 99.8% — no free pages.
- No prefix-cache blocks are evictable (all KV is held by active slots).
- `evict_prefix_cache_for_allocation` returns 0 freed pages.
- Allocation of the next decode page fails → the slot can't advance.
- Scheduler emits `idle` plans because `decode_rows == 0` and
  `prefill_rows == 0` and there's no path to make either non-zero.
- Waiting queue accumulates indefinitely (admission can't take new
  requests because there's no free KV either).

This is a **flow-control hole**: when active slots collectively saturate
KV and each needs further KV, ONE slot must be **preempted** (request
returned to waiting queue, its KV pages released). vLLM, SGLang, and
TRT-LLM all implement this — vLLM via abort + reschedule, SGLang via
`--preempt-mode=swap-out` (host swap) or `--preempt-mode=recompute`.

Resolution direction (next sub-plan, NOT this fix):

1. Add `Scheduler::preempt_one_active_slot` that picks a victim
   (least-progressed slot, tie-break to longest prompt — already the
   policy alluded to in the 2026-04-13 errors entry's historical note).
2. Victim's KV pages are released back to the pool; victim's request
   moves back to the waiting queue with its progress checkpointed
   (either via radix prefix re-publish, or via host-side recompute
   marker).
3. `alloc_pool_tokens_with_retry` calls `preempt_one_active_slot` as
   the third tier after prefix-cache eviction fails (and possibly
   T1/T2 demotion of cached blocks).

This is genuinely separate from M4's op-trait work and should not gate
M4 acceptance. Recommend opening a follow-up
(`docs/plans/M4.5-kv-preemption.md`) and scoping the fix there; M4 wins
entry can cite this errors entry as "performance gate not yet runnable
due to scheduler preemption gap" rather than counting M4 itself as
failed.

## Rule

Do not publish M4 GuideLLM numbers unless the run produces benchmark output and
the service drains to `active=0 waiting=0` after the client exits. A
stuck-active trace is an error artifact, not a performance result.
