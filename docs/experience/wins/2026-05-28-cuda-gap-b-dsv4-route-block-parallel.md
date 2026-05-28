# GAP-B · `dsv4_route_kernel` block-parallel rewrite (pending-remote bench)

## SLO-shape probed? — N (kernel-only edit; Mac can't run nvcc / bench)

## TL;DR

Implements GAP-B from
[`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../../research/2026-05-28-arle-kernel-vs-sota-audit.md):
replace the `if (threadIdx.x == 0)` serial body of `dsv4_route_kernel` with a
full-block-parallel path. The kernel still launches with
`<<<num_tokens, DSV4_ROUTE_BLOCK=256, …>>>` but now actually uses all 256
threads instead of leaving 255 idle.

Diff in this commit (entire scope):

| Path | Change |
|---|---|
| `crates/cuda-kernels/csrc/moe/dsv4_route.cu` | New `__device__` helpers `dsv4_route_block_reduce_max`, `dsv4_route_block_reduce_sum`, `dsv4_route_block_argmax`; `dsv4_route_kernel` rewritten as three phases (parallel score + parallel top-k via masked block-argmax loop + serial renorm) |

ABI / FFI / call-site invariants:
- `dsv4_route_kernel` signature unchanged.
- `extern "C" CUresult dsv4_route_cuda(...)` unchanged.
- No Rust-side edits (`crates/cuda-kernels/src/*`, `infer/src/**`).
- Sister kernels in the same file (lines 484, 594, 665, 1013) **not touched** —
  see survey below; their `threadIdx.x == 0` pattern is a *correct* slot-allocator
  followed by block-parallel hidden-dim copy, not the audit's anti-pattern.

## Survey: which kernels in `dsv4_route.cu` match the audit pattern?

Searched all 5 `threadIdx.x == 0` sites in
[`crates/cuda-kernels/csrc/moe/dsv4_route.cu`](../../../crates/cuda-kernels/csrc/moe/dsv4_route.cu):

| Line | Kernel | Pattern | Audit target? |
|---|---|---|---|
| 225 | `dsv4_route_kernel` | **Entire body** serial on thread 0; 255 threads idle | **YES — fixed in this commit** |
| 484 | `dsv4_pack_local_experts_kernel` | `atomicAdd` slot allocator + scalar header writes; then **block-parallel `hidden_dim` copy** | No (already block-parallel) |
| 594 | `dsv4_pack_expert_ranks_kernel` | Same slot-allocator + block-parallel copy | No |
| 665 | `dsv4_pack_local_experts_with_slots_kernel` | Same | No |
| 1013 | `dsv4_pack_received_experts_kernel` | Same | No |

Bonus finding: `dsv4_prepare_packed_local_experts_small_kernel`
(line 903) has a `tid == 0` final exclusive scan over `experts_per_rank` (≤
256 by the cuda-side guard). Small constant-bounded loop; not in the audit's
top-leverage list. Deferred.

The audit's "other 28 supporting kernels" line refers to **launch churn**
(28 small kernels × per-token launches in a decode wave), not to the same
single-thread anti-pattern. None of the sister kernels surveyed actually wastes
255 threads.

## What changed in `dsv4_route_kernel`

Three phases, all block-parallel except the topk≤16 renorm tail:

1. **Per-expert score** — block-stride loop over `n_experts` (≤ 512). For
   `scoring_kind == 0`, parallel softmax via two block-reductions
   (`block_reduce_max`, `block_reduce_sum`). For `scoring_kind != 0`, the
   per-expert score function is independent and fully data-parallel.
2. **Top-k selection**:
   - `routing_kind == 0`: serial table read on thread 0 (topk ≤ 16; the
     read is trivially small).
   - `routing_kind == 1`: build `combined[e] = scores[e] + bias[e]` in
     parallel, then run `topk` rounds of masked block-argmax. Each round picks
     one expert via warp tournament + cross-warp final reduction, then masks
     it with `-INFINITY` for subsequent rounds. Tie-break (lower expert wins
     on equal score) matches the original serial selection-sort.
3. **Renorm + writeout** — kept on thread 0 (topk ≤ 16, no benefit from
   parallelizing).

Tie-break carefully replicates the original — both inside each thread's
local merge and across the warp tournament: `s > best` strictly, or
`s == best` AND **both are real candidates** (`> -INFINITY`) AND
`e < best_expert`. Mask sentinel (`-INFINITY`) never participates in tie-break,
so masked experts in a multi-round selection cannot pollute later rounds.

## Numerical-parity stance

**Not bitwise** — parallel softmax sums in a different tree order than the
original serial loop, and FP32 addition is non-associative. Top-k indices
should match in the typical bf16-logit regime (relative score differences
dominate parallel-summation jitter by orders of magnitude). Weights should
match to within bf16 ulp.

If the remote bench reveals divergence:
- Top-k index mismatches on tie-break: there's a subtle ordering issue in
  the argmax. Need to gate on the same `n_experts`/`bias` shape and root-cause.
- Weight drift > bf16 ulp: parallel sum jitter exceeded budget for some
  small-denominator scoring kind; may need Kahan or pairwise sum.

## Pending-remote items

- **CUDA build & smoke**: cannot run nvcc on Mac. `cargo check -p cuda-kernels
  --features cuda` fails at nvcc invocation as expected (no CUDA toolchain on
  Darwin). Need remote pod nvcc compile.
- **Functional parity test**: would extend `infer/tests/` with a smoke test
  that runs `dsv4_route_cuda` against a known-fixed bf16 logits tensor for
  multiple (n_experts, topk, routing_kind, scoring_kind) shapes and compares
  indices + weights to a saved CPU reference. Deferred to remote since CUDA
  test execution is gated on `tn` (currently broken — see
  `docs/experience/errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md`).
- **`scripts/bench_guidellm.sh dsv4-route-gap-b` vs latest DSv4 decode
  baseline**: pending — `tn` restore needed.

## Mac typecheck status

`cargo check -p infer --no-default-features --features cuda,no-cuda` is **red
on HEAD** before my edit (errors in `infer/src/main.rs` line 507-508
referencing `SchedulerHandle::ep_nccl` and `DistributedRequestCoordination::new_nccl`
that don't exist in the current `infer/src/scheduler/types.rs`). Confirmed
pre-existing by stashing this commit's `.cu` change and re-running — same
errors. The `.cu` file is the entire diff in this commit; no Rust-side change,
so the Rust crate-graph delta is zero.

`cargo check -p cuda-kernels --no-default-features` clean (cuda feature
inactive on Mac).

## Rule

When parallelizing a single-thread CUDA kernel that drives selection / top-k:
- Keep two arrays — original scores (read by renorm) and selection key
  (mutated by masking). Don't overwrite the score buffer.
- Bake the tie-break into both per-thread local merge **and** every
  warp-tournament round, with explicit guards against the mask sentinel.
- Block-reduction helpers should `__syncthreads()` after reading the
  broadcast slot, so subsequent uses of the same smem buffer in the same
  block don't race.

## Refs

- Plan: [`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../../research/2026-05-28-arle-kernel-vs-sota-audit.md) §GAP-B
- Trace anchor: `dsv4_route_kernel 4.4% GPU time` (2026-05-14 trace cited in
  the audit) — wall-clock decode ceiling for this change is ~3–4 %.
- SOTA reference: SGLang `sgl-kernel/csrc/moe/topk_softmax.cu`
  (one-warp-per-token bitonic top-k), FlashInfer
  `flashinfer/python/csrc/sampling.cu::top_k_renorm_probs`.
