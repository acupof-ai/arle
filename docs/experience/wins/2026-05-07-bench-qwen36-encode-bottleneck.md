# Bench — Qwen3.6 MoE: localized the 95% bottleneck to MLX encoder — 2026-05-07

## Goal

Per
[`2026-05-07-bench-qwen36-mle-perf.md`](2026-05-07-bench-qwen36-mle-perf.md)
the dominant ITL phase on Qwen3.6 35B-A3B is `async_kick` at 95% of
step time (23 ms at c=4, 65 ms at c=16). The Rust-side phase timing
was bracketing the call boundary; this entry adds C++-side timing to
confirm whether the cost is (a) graph build inside `m->forward(...)`
or (b) the subsequent `mx::async_eval(...)` call.

## Hypothesis

Per the MLX async_eval research subagent (this date), `mx::async_eval`
does graph traversal + Metal command-buffer encoding **synchronously**
on the calling thread. Only GPU *execution* runs async. For Qwen3.6
MoE (~600-1000 primitives per step), encoding alone could be the 23 ms
we see. Graph build (`m->forward`) should be cheap (just lazy
construction).

## What we added

`INFER_CPP_PHASE_TIMING=1` env-gated stderr probes in `crates/mlx-sys/src/`:
1. `mlx_qwen35_model.cpp:2541` — `forward_build_us` around `m->forward(inputs)`
2. `mlx_bridge.cpp:2072` — `async_eval_call_us` around the actual `mx::async_eval(arrs)`

Cached env probe (one atomic load after first call). Zero prod cost
when env unset.

## Params

- Binary: `target/release/metal_serve` rebuilt at this commit
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` (Metal canonical;
  auto-wired-limit = 20 GiB on this commit's default)
- Workload: c=4 + c=8 smoke (`/tmp/cN_smoke.sh`)
- Env: `INFER_PHASE_TIMING=1 INFER_CPP_PHASE_TIMING=1`

## Results

```
forward_build_us (m->forward inside step_batch_packed):
  c=4 n=62 avg=1548 p50=1509      ← lazy graph build is FAST
  c=8 n=56 avg=1839 p50=1793

async_eval_call_us (mx::async_eval call) by eval_refs count:
  count=1   n=13  avg=455883  p50=76102   ← warmup outliers (first-step pipeline-state init)
  count=81  n=12  avg=21029   p50=19526
  count=82  n=124 avg=32147   p50=24992   ← steady-state: ~25ms encoding
```

→ **Hypothesis confirmed.** The graph build (`m->forward`) is 1.5-1.8 ms at
c=4-8 — basically free. The 23-25 ms steady-state cost lives entirely
inside `mx::async_eval`'s synchronous Metal encoder work. Encoding ~80
output arrays (logits + new_sampled + ~80 packed_kv_flat slabs) at the
end of a 40-layer MoE forward = 600-1000 primitives = 25 ms of CPU
work submitting kernels and committing command buffers.

## Implications

**The "async" name is misleading.** `mx::async_eval` only skips the
final wait for GPU completion; the encoder work runs on the caller
thread synchronously. From `mlx/transforms.cpp` `eval_impl(... ,
async=true)`:

1. DFS traversal (in-degree, fence detection)
2. BFS tape build (topological order, width-bound)
3. Reverse-tape dispatch loop — for each primitive: `gpu::eval(arr)`
   sets MTL pipeline state, binds buffers, dispatches threadgroups
4. Per-cmdbuf `needs_commit()` check: 50 ops or 50 MB on M-series Max
   (`mlx/backend/metal/device.cpp`) → multiple synchronous commits per
   MoE step

Only the **wait at end** is skipped; the encoder runs on the caller
thread.

For Qwen3.6 35B-A3B at c=4: ~600 primitives encoded sync per step;
c=8: ~800; c=16: ~1100. This explains why p50 scales nearly linearly
with c (more primitives, more encode time) while p99 was helped
massively by wired_limit (eliminating the orthogonal page-fault tail).

## What CAN'T fix this here

- **`mx::compile` of the whole MoE forward** is blocked on
  ml-explore/mlx PR #3485 (`GatherQMM` doesn't yet implement
  `output_shapes` for shapeless compile). Even with that, the routing
  index permutation from `mx::sort` is value-dependent — would re-trace
  per step. (b)/(c) — out of scope this week.
- **Multi-thread the encoder** — blocked by ml-explore/mlx #3078: each
  Metal `DeviceStream` has a per-thread encoder; can't share across
  threads without "command encoder is already encoding" assertion.
  Worker-thread encoding requires MLX-side change. Out of scope.

## What CAN fix this here

| # | Lever | Effort | Status |
|---|---|---|---|
| 1 | Pre-encode router on a dedicated MLX **stream** so encode of step N+1 overlaps with GPU exec of step N | M | candidate next tick |
| 2 | Reduce primitive count in `switch_glu_forward` — fuse `expand_dims` + collapse the `_gather_sort`/`_scatter_unsort` round-trip when `do_sort=false` | M | candidate next tick |
| 3 | Tune `MLX_MAX_OPS_PER_BUFFER` deliberately — bench 50 vs 200 vs 500 with matched A/B since prior cmdbuf=200 try on Qwen3.5 doesn't transfer | S | candidate next tick |
| 4 | Drop redundant entries from `eval_refs` (count=82 → fewer) by collapsing per-layer KV slabs into per-step views | S–M | requires KV layout change; M_e.1 oMLX-B territory |

## Learnings

1. **Phase timing has to span the FFI.** The Rust-side phase boundaries
   (`async_kick = t_async_eval - t_step_built`) merged "graph build +
   FFI overhead + actual MLX work" into one bucket. Adding C++-side
   timing inside the FFI splits these in one bench cycle.
2. **`std::fprintf(stderr, …)` env-gated probes are sufficient
   diagnostic infrastructure.** Don't reach for tracing crates — a
   single-line stderr printf with cached env probe is zero-cost when
   off and immediately greppable when on.
3. **Read the upstream source before tuning.** The "async" in
   `mx::async_eval` is asynchrony of *GPU completion*, not asynchrony
   of *encoder work*. This was knowable from `mlx/transforms.cpp`
   without running anything. The research subagent saved a tick of
   chasing the wrong lead.

## What worked / Rule

- C++ env probe pattern (cached `std::getenv` → static int) shipped
  cleanly in two TUs (`mlx_qwen35_model.cpp` + `mlx_bridge.cpp`) by
  duplicating the helper file-locally — file-static keeps each TU
  independent.
- `count=82` in the eval_refs is logits(1) + new_sampled(1) + per-layer
  KV slabs (40×2 = 80) at c=4. The breakdown matches the per-layer
  primitive count reported by the research subagent.

## Rule

When a Rust phase timing call shows >50% of step time in a single FFI
call, **add C++-side timing inside that call before optimizing**. The
FFI boundary often hides "graph build vs MLX evaluation vs sync wait"
attribution that determines which optimization is correct. Without
the C++ probe, this tick would have spent its budget on the wrong
direction (wrapping `m->forward` in `mx::compile`) — but the data
showed `m->forward` was already 1.5 ms.

## Next

- **Implement lever #2 (primitive-count reduction in
  `switch_glu_forward`)** — fuse `expand_dims(..., {-2, -3})` and
  collapse the `_gather_sort`/`_scatter_unsort` round-trip on the
  unsorted path.
- **Bench MLX_MAX_OPS_PER_BUFFER properly** with matched A/B at
  c=4/c=8/c=16 to determine whether Qwen3.6 wants 50 (default) or 200.

## References

- Research source (this date): MoE async_eval research subagent —
  citations in this entry's body.
- Predecessors:
  [`2026-05-07-bench-qwen36-baseline.md`](2026-05-07-bench-qwen36-baseline.md)
  +
  [`2026-05-07-bench-qwen36-mle-perf.md`](2026-05-07-bench-qwen36-mle-perf.md)
- Upstream code refs:
  - mlx-explore/mlx `mlx/transforms.cpp` (`async_eval` /
    `eval_impl`)
  - mlx-explore/mlx `mlx/backend/metal/device.cpp`
    (`needs_commit`)
  - mlx-explore/mlx `mlx/backend/metal/eval.cpp` (`gpu::eval` per
    primitive)
- Blocking upstream PR: ml-explore/mlx#3485
