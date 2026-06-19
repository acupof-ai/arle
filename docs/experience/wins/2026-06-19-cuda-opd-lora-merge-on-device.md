# OPD per-step LoRA sync: on-device merge — 78s → 0.15s (510×)

## Context

OPD on-policy rollout re-syncs the student LoRA into the inference engine every
training step. A trace showed `sync_lora` = **~84 s/step (62 % of the step)**.

Root cause (code-confirmed, `crates/infer-cuda/src/qwen35.rs`):
`merge_lora_proj` computed the merged weight
`W[r,c] = base[r,c] + scale·Σ_k B[r,k]·A[k,c]` as a **host bf16 triple-loop**,
and `remerge_student_lora` called it for the full all-linear set
(~7 projections × 40 layers = 280 projections) per step. `B·A` is a
`rows×rank · rank×cols` GEMM; doing it single-core on the host is
`O(rows·cols·rank)` per projection (~0.28 s each) → ~78 s aggregate, then each
merged matrix was uploaded host→device.

## What Worked

Moved the dense-BF16 merge **entirely on-device**:

- Cache the pristine base on **device** (`lora_base_dev`, a D2D clone of the
  resident matrix on first touch) instead of (only) the host snapshot.
- Per step: transpose the tiny `A` on host (`[rank,cols]`→`[cols,rank]`, ~32×in),
  upload `A^T` + `B` (both small), run `B·A` on the GPU via the existing
  `gemm_cuda` (`Y[M,N] col-major = W[M,K]·X[K,N] col-major`; mapping
  `M=cols, N=rows, K=rank` makes `Y` byte-identical to the row-major
  `[rows,cols]` `DeviceMatrix.data`), then fold
  `W = base_dev + scale·(B·A)` straight into the resident matrix with a
  full-buffer scaled-add (`add_scaled_row_cuda`, whole matrix as one row of
  `hidden_dim = rows·cols`). No host triple-loop, no full-W H2D upload.
- Reused a single device delta scratch (grown to the largest matrix) — one
  alloc, not per-projection.
- FP8 / grouped-expert targets keep the host snapshot + re-quant path
  unchanged (re-quant needs host per-block scaling; smaller fraction).
- Zero-adapter restore + `lora_dirty` tracking + TP single-GPU guard + all
  shape `ensure!` checks preserved byte-for-byte.

New ops (`crates/infer-cuda/src/ops.rs`): `lora_device_gemm`,
`lora_scaled_add_into` (both generic over `DevicePtr`/`DevicePtrMut` so a reused
over-sized scratch view works without a copy).

### Correctness gate (GPU7, H20, CUDA 12.x — `device_lora_merge_matches_host_reference`)

Device merge vs the host triple-loop reference, rows=512 cols=384 rank=32:

```
cosine = 1.00000000   max_abs_err = 0e0
```

Bit-exact (both reduce in f32, round to bf16 once; the rank-32 cublasLt GEMM
agrees with the host f32 accumulation under bf16 rounding).

### Perf (GPU7, microbench `bench_lora_remerge_host_vs_device`, 4B shapes)

280 projections (7 per-layer × 40 layers), rank 32, alpha 16:

| Path | per-step LoRA merge | Δ |
|---|---:|---|
| HOST triple-loop (old) | 78 133 ms | baseline (≈ the traced 84 s) |
| DEVICE merge (new) | **153 ms** | **510× faster** |

Drops per-step LoRA-sync from ~84 s → ~0.15 s — well under the <1 s target;
the `sync_lora` phase ceases to be the OPD step bottleneck.

## Rule

For a per-step weight re-merge `W = base + scale·(B·A)`, a host triple-loop is
`O(rows·cols·rank)` single-core and dominates the step; the merge is two device
ops (rank-`r` GEMM + full-buffer scaled-add) reusing the resident GEMM kernel.
Cache the pristine base **on device** so the per-step path is upload-tiny-A/B →
GEMM → scaled-add-in-place, never host-compute + full-W upload. The
GEMM-layout trap: `gemm_cuda` is `[M,K] row-major · [K,N] col-major →
[M,N] col-major`; to land `B·A` row-major into `data`, pass `A^T` as `W` and
`B` as-is, with `M=cols, N=rows, K=rank`.

Verified on the node-62 container (`/data01/arle-build`), GPU7 only.
