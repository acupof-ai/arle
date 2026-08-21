# Three Metal decode optimization directions — all three ruled out — Metal, 2026-08-21

> Status: Investigated, no change shipped

## Context

Three directions were proposed to close the Metal decode gap on
Qwen3.8-27B-MLX-2bit (M4 Pro, 273 GB/s peak, 10.09 GB weight bytes per
decode step, theoretical ceiling 27.1 tok/s):

1. Fuse GDR recurrent + postprocessing (norm + gate into the scan kernel,
   state in registers).
2. Prefill-only fused preprocessing (gate to S>1).
3. Faster 2-bit matmul (matmul is ~70% of decode).

## Investigation

### Direction 1 — fused GDR postprocessing: no measurable win

Two variants:

- **Weak variant (implemented, on branch `metal-fused-gdr-postprocess`):**
  fuse rms_norm + silu_mul into one postprocessing kernel. Measured
  21.5 → 21.6 tok/s (within noise). Cause: MLX async CPU→GPU dispatch
  already hides kernel launch overhead for small (4 KB/layer) kernels.
- **Strong variant (fuse norm+gate into the scan kernel, y/z in registers):**
  ruled out by first principles. Per GDR layer, y and z are each
  hv×dv×2 = 48×128×2 = 12 KB (bf16). Fusing into the scan kernel saves
  4 transits (y write, z write, y read, z read) = 24 KB/layer. Across 48
  GDR layers: 1.125 MB/step. At 273 GB/s that is 4.1 µs — 0.009% of the
  44.6 ms step. The weight read (10.09 GB) dominates by 4 orders of
  magnitude.

### Direction 2 — prefill-only fused preprocessing: not worth it

Prefill is matmul-dominated: the 10.09 GB weights are read once and the
compute is O(S) per weight. The preprocessing kernels (SiLU + QK norm +
g/beta) operate on B×S×hv×dv tensors; at S=4096 that is ~100 MB/layer or
~18 ms total across 48 layers — under 1% of a prefill step that runs to
seconds. Fusing them saves a fraction of that 1%.

### Direction 3 — faster 2-bit matmul: headroom is <17% and mostly non-kernel

Re-verified decode speed on a clean release binary (main @ec839b0fa,
Metal, `--max-running-requests 1`, 60-token streaming decode, 3-run
median): **22.4 tok/s = 44.6 ms/step**.

Effective bandwidth = 10.09 GB ÷ 44.6 ms = **226 GB/s = 83% of the 273
GB/s peak**. This figure includes all non-matmul overhead (attention,
GDR scan, norms, launches). The pure qmv_fast kernel efficiency is
higher — non-matmul time dilutes the effective figure downward.

One kernel-level experiment was run: the vendored MLX qmv tile config
for 2-bit (bn=8→16, num_simdgroups=2→4) showed 21.4 vs 21.5 tok/s — no
improvement. Reverted.

The remaining 17% gap is split between qmv_fast bandwidth inefficiency
and non-matmul step time. NAX (Apple's matrix engine) is the only
hardware lever that could change the matmul ceiling; it requires
macOS 26.2+, unavailable on this box.

## Rule

- Before fusing kernels to save memory traffic, compute the traffic:
  y/z intermediate tensors are 1.125 MB/step against a 10.09 GB weight
  read. Traffic under 0.1% of the step's dominant transfer cannot
  produce a measurable speedup, fused or not.
- Effective bandwidth (weight bytes ÷ step time) is the decode ceiling
  proxy. At 83% of peak including all overhead, the 2-bit matmul is not
  the lever it appears to be from its 70% time share — the time share
  is high because the kernel is already near-bandwidth-bound.
- Kernel launch overhead is hidden by MLX async dispatch on Metal;
  launch-count reductions do not move decode speed.

## Environment

- Host: M4 Pro 48GB, macOS
- Model: majentik/Qwen3.8-27B-MLX-2bit (64 layers, 48 GDR + 16 full attn,
  hv=48, dv=128)
- Binary: `arle` release, `--no-default-features --features metal,no-cuda`,
  main @ec839b0fa
- Files: `crates/mlx-sys/src/mlx_qwen35_model.cpp` (fused postprocessing
  variant on branch `metal-fused-gdr-postprocess`),
  `crates/mlx-sys/vendor/mlx/mlx/backend/metal/quantized.cpp` (bn=16
  experiment, reverted)
