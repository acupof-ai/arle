# autograd: GPU fused chunked-CE replaces the host scalar loop in OPD writeback — ~3944× per-target

## Context

`fused_linear_ce_loss_indexed` (the agent-OPD masked-CE writeback core,
`crates/autograd/src/ops/fused_linear_distill.rs`) ran a single-threaded HOST
scalar loop: per masked target, a nested `vocab=248320 × hidden=5120` dot-product
to build logits + CE + grad_hidden. Measured **25.6 s/target** — seq-16K's ~15744
targets = a multi-hour host compute, the remaining wall to USABLE long-seq
training after the writeback OOM was closed
([2026-06-27 writeback OOM win](2026-06-27-agent-opd-writeback-oom-closed-frozen-hidden-and-sdpa-chunk-leaks.md)).

Commissioned by `docs/plans/2026-06-20-opd-35b-infra-design.md` (long-seq
agent-OPD writeback path).

## What Worked

Backend dispatch in `fused_linear_ce_loss_indexed`: CPU backend keeps the host
scalar loop (the numerical reference); CUDA/Metal take a new
`fused_linear_ce_loss_indexed_device` built **entirely from existing
device-resident autograd ops** — no hand-rolled GEMM or softmax kernel
(adopt-existing-first). Per position-chunk (preserving the host's chunking + the
per-chunk-free discipline commit `189155c0` established):

- `embedding(hidden_2d, chunk_positions)` — row-gather the chunk's hidden rows
  (non-contiguous masked positions) → reshape `[1,chunk,hidden]`→`[chunk,hidden]`
  (embedding emits an implicit batch dim; matmul_bt's backward is rank-2-only).
- `matmul_bt(rows, lm_head)` → `[chunk, vocab]` logits — the SAME device GEMM the
  materialize-then-`cross_entropy` reference uses.
- `log_softmax` → `gather_last_dim(targets)` → `sum`, scaled `-1/N`.
- per-chunk **sub-tape** `backward_collect` yields the chunk's `grad_hidden`
  (and `grad_weight` only if the head is trainable — frozen under
  `--share-frozen-base`, so usually skipped); accumulated into a running
  full-shape device grad while each chunk's `[chunk, vocab]` tile is freed before
  the next. Accumulated grad is saved on the existing `FusedLinearDistill` tape
  entry, so outer backward scales it by upstream exactly like the host path.

**Reused GPU kernels** (all pre-existing device-resident paths, dispatched via
`store.backend()`): `embedding`, `reshape`, `matmul_bt` (cuBLAS), `log_softmax`,
`gather_last_dim`, `sum`, `mul_scalar`, `add`, `clone_tensor`.

### Numerical correctness gate (the license)

`crates/autograd/tests/test_fused_linear_ce.rs` — added two CUDA-gated tests
(`fused_linear_ce_gpu_matches_reference_*`) asserting the GPU fused path vs the
CPU materialize-then-CE reference, ≤1e-3 on **loss and `d_hidden`**, dense +
sparse positions. On the 8×H20 box (sm_90, GPU 5):

```
running 4 tests
test fused_linear_ce_gpu_matches_reference_dense ... ok
test fused_linear_ce_gpu_matches_reference_sparse_positions ... ok
test fused_linear_ce_matches_reference_dense ... ok       (CPU reference, unchanged)
test fused_linear_ce_matches_reference_sparse_positions ... ok
test result: ok. 4 passed; 0 failed
```

### Micro-bench (per-target wall, production shape)

`examples/bench_fused_ce.rs`, vocab=248320, hidden=5120, chunk=256, GPU 5
(physical, `CUDA_VISIBLE_DEVICES=5`→logical 0; GPU 4 foreign-busy, untouched):

| path | targets | wall | per-target | Δ vs host |
|------|--------:|-----:|-----------:|----------:|
| HOST scalar loop (CPU) | 4 | 102.39 s | **25.598 s** | baseline |
| GPU fused chunked-CE (CUDA) | 512 | 3.32 s | **0.0065 s** | **−99.97% (3944×)** |

(Host run at few targets — the loop is per-target linear; per-target is
representative. Loss values are not comparable across different N / random rows;
the ≤1e-3 finite-diff gate above is the correctness license, not the bench loss.)

At seq-16K (15744 targets) this turns a ~112-hour host compute into ~102 s of GPU
writeback — the long-seq agent-OPD writeback wall is removed.

## Rule

Port a host scalar loop to GPU by **composing the existing device-resident
autograd ops** (the same ones the reference uses) inside a per-chunk sub-tape +
`backward_collect`, accumulating into the existing saved-grad tape entry — never
hand-roll a GEMM/softmax. The license is the finite-diff gate (loss + grad ≤1e-3
vs the materialize-then-CE reference, dense AND sparse), run on the GPU backend,
not byte-identity. `embedding` row-gather emits an implicit `[1,...]` batch dim —
reshape to rank-2 before any rank-2-only backward (matmul_bt).
