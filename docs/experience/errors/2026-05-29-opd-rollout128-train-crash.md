# OPD rollout≥128 TRAIN-step crash is a VRAM OOM, not a kernel index overflow or a b0a62c4c regression

## Context

Long-CoT OPD (reasoning experiment needs rollout ≥256) crashed on the
first full TRAIN step with the generic, masked error:

```
Error: InvalidInput("OPD student chunk KL Qwen3.5 forward autograd error:
  cuda synchronize failed. ...")
```

rollout-64 trained fine; rollout-256 step-0 EVAL passed but the first
TRAIN step (with backward) crashed; reproduced with the W4 and BF16
teacher; across `kl_chunk_size {16,32,128}`. Repro:
`runs/2026-05-29-rollout256-w4teacher-200steps/probe10.log`.

Hardware: RTX 4070 Ti SUPER, **16 GB** (sm_89). Build env per brief
(`CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14 TORCH_CUDA_ARCH_LIST=8.9
ARLE_CUDA_DISABLE_FLASHMLA=1`, `--release`). Example
`opd_step_cuda_infer_teacher_train`, `mem_fraction_static=0.05`.

## Root Cause

**Out-of-memory (VRAM capacity), not a kernel/grid/index overflow.**

The brief's hypothesis (a u16/int grid dim, shared-mem-vs-seq_len, or
indexing overflow in a forward/backward autograd kernel) was **refuted**.
Running the smallest crashing shape with `CUDA_LAUNCH_BLOCKING=1` to make
the async failure synchronous resolved the masked "cuda synchronize
failed" to the REAL error:

```
Error: Autograd(TapeInvariant("cuda alloc_zeros failed (add_into_device)"))
```

- `add_into_device` is the device-resident gradient-accumulation fusion
  in `crates/autograd/src/tape.rs:516` (called during `tape.backward()`).
- The failing call is `stream.alloc_zeros::<f32>(size)` at
  `crates/autograd/src/backend_cuda.rs:3659` — a plain allocation failure.
  `size` passes the subsequent `i32::try_from` guard, so it is **not** an
  i32/index overflow; it is genuine `cudaMalloc`/pool exhaustion.
- The "...forward autograd error" wording in the user-facing message is
  the `map_qwen35_forward_error` wrapper; the async OOM merely surfaces at
  the next synchronize, which the non-blocking run attributed to the
  student forward. With blocking on, the true site is the backward.

The transient is dominated by the **per-layer activation tape over the
full sequence** (~30 MiB/token, attributed in commit `f8ef6aca` via
`cuMemGetInfo` per phase). It scales linearly with `seq_len`:

| rollout | seq_len (prompt+rollout) | free before step | step-1 outcome |
|---|---|---|---|
| 64  | ~80  | ample      | trains |
| 128 | ~144 | ~6.6 GB    | step-1 OK, **step-2 OOM** (knife-edge) |
| 256 | ~270 | ~6.6 GB    | **step-1 OOM** (genuine capacity shortfall) |

Resident VRAM at step start ≈ 9.3 GB / 16 GB (CUDA ctx+kernels 1.3 GB,
train-student base 1.4 GB, infer-student rollout engine 1.6 GB,
infer-teacher 3.1 GB W4-resident, optimizer/eval scratch ~1.7 GB),
leaving only ~6.6 GB free. The rollout-256 backward needs ~8 GB.

**Not a `b0a62c4c`/`446cf5b8` regression.** The O(n²) backward rewrite
(one full-prefix forward + one backward instead of per-chunk) was
suspected of newly hitting a limit. But the OLD per-chunk loop's LAST
chunk forwarded `rollout[..rollout_len]` (the full prefix) and ran its
backward over that — i.e. the OLD code had the **identical peak
activation footprint**. The rewrite changed compute (O(n²)→O(n)) but not
the backward peak memory. So this is a **pre-existing 16 GB capacity
limit**, only newly exposed because the reasoning experiment pushes
rollout past where the full-sequence tape fits.

## Fix

No correct, scoped fix lands rollout-256 on a 16 GB card while keeping
the fast infer-engine rollout. The genuinely-correct fix is **gradient
(activation) checkpointing** in the autograd engine — recompute each
transformer layer's activations during backward instead of retaining the
full-sequence tape — which is an architectural change to the tape engine
(`crates/autograd/src/tape.rs` + `crates/train/src/qwen35.rs`
`forward_batch_hidden_indices`) and exceeds the >5-file / architectural
guard; deferred with this attribution rather than landed half-done.

Cheap levers tested and **refuted** (single-variable, matched):

1. **Memory-pool defrag.** cudarc 0.19.7 allocs via `cuMemAllocAsync`
   (release-threshold 0, freed blocks cached in the pool). Hypothesis:
   rollout-128 step-2 fails on a fragmented pool. Added a best-effort
   `cuMemPoolTrimTo(0)` (`crates/autograd` + a trim call before the
   backward) and re-ran rollout-128×3: **still OOMs at step 2**, free
   unchanged (6602 vs 6666 MiB). The pool was not holding cached free
   blocks at step start — the VRAM is genuinely committed. Reverted (a
   no-op runtime change must not ship). Fragmentation hypothesis killed.
2. **Free the 1.6 GB infer-student engine** (`ARLE_OPD_INFER_ROLLOUT=0`).
   Frees the headroom (free 6.6→8.2 GB) so the backward would fit, BUT
   routes rollout through the in-process KV-cache decode, which at
   rollout-256 does not finish even one step in 590 s (vs ~5 s with the
   infer engine). Not viable: trades the OOM for an unusable wall-clock.

### What works today

- **rollout ≤128, single-pass / short runs** train (step-1 fits).
- **rollout-128 multi-step** is on the cliff edge (step-2 OOM); not
  reliable without more headroom.
- For rollout ≥256 the experiment needs one of: gradient checkpointing
  (the real fix), a 24 GB card, or a materially smaller resident
  footprint (e.g. a quantized-resident teacher under ~1.5 GB +
  freeing the infer-student engine while keeping a fast rollout path).

## Rule

- **Get the REAL error before theorizing about kernels.** A masked "cuda
  synchronize failed" is almost never the actual failure. One
  `CUDA_LAUNCH_BLOCKING=1` run turned a hypothesized index/grid overflow
  into a confirmed `alloc_zeros` OOM — and the `i32::try_from` guard right
  after the alloc proves it is capacity, not indexing. Always do this
  first; it is the cheapest SOLID experiment.
- **A "regression suspect" commit is not the cause until the pre-change
  peak is computed.** `b0a62c4c` looked like the trigger but the OLD
  per-chunk loop's last chunk had the identical full-prefix backward
  peak — the limit pre-existed. Compute the resource peak of BOTH the old
  and new path before attributing.
- **Refuted cheap levers are real eliminations, not failures — and a
  no-op runtime change must not ship.** The pool-trim experiment cleanly
  killed the fragmentation hypothesis; it was reverted rather than landed
  for a bench entry, because shipping a change that demonstrably does
  nothing burns review and bench cycles (kernel-opt anti-pattern #13 +
  §Benchmarks "no entry → not shipped, and no-op → no entry").
