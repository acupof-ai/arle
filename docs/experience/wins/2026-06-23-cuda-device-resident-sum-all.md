# CUDA device-resident `sum_all` (correct op-fix) + the real 27B-CE win was the fused→dense default flip

## Context

OPD-training (27B Qwen3.6-FP8 student, vocab=248320) reported ~4.5 min/step,
CPU 99.9%, GPU idle. The first hypothesis (this entry's original draft) pinned it
on `sum_all`: the CUDA `Backend::sum_all` (used by `mean`, via
`ops::reduce::mean_device_lazy`) did `memcpy_dtoh(full x) → host iter().sum() →
clone_htod(scalar) → synchronize()` per chunk — a `[32, 248320]` f32 (~31.8 MB)
DtoH + single-thread CPU reduce + a blocking `synchronize()` that stalls the GPU.
That round-trip is real, and a device-resident reduce is the right fix for it.

**But H20 measurement corrected the root-cause attribution (§0 case-as-fact).**
The 27B step stayed ~205 s even with `sum_all` device-resident. Decoding the op
trace (`ARLE_OPD_BACKWARD_PROFILE=1` + `ARLE_OPD_STEP_TRACE=1` + a GPU-util
sampler) showed the dominant cost was a **different op**: `fused_linear_distill`
(the default windowed-KL path) running the lm_head on the HOST — 201.7 s, GPU at
0%. The full backward, *including* the now-device-resident `sum_all`, is only
2.1 s. So `sum_all` lives in the fast 2.1 s backward; it was never the 4.5-min
bottleneck. The wall-clock fix was flipping the default to the dense device path
(fused 205 s → dense 3.8 s, 53×) — see
[`errors/2026-06-23-opd-fused-distill-default-host-bound.md`](../errors/2026-06-23-opd-fused-distill-default-host-bound.md).

This entry now documents `sum_all` honestly as a **correct micro-optimization**
(eliminates a real but minor per-`mean` DtoH+sync), not the wall-clock win.

## What Worked (Mac-verified + H20-confirmed)

- **Device-resident multi-pass reduce.** Added `sum_partial_f32` NVRTC kernel
  (`backend_cuda/kernels/reduce.cu`, registered in `kernels.rs`), a block reduce
  summing raw values into one f32/block (sibling of `sum_squares_partial_f32`).
  New `cuda_sum_all_device` (`backend_cuda.rs:~6000`): pass 1 reduces `size` →
  `ceil(size/256)` partials; recurses on the partials until 1 element. Returns a
  1-element device handle — **no DtoH, no `synchronize()`** (caller's terminal
  eval owns it, so it composes into the existing device-resident chain; the only
  host transfer stays the final 4-byte loss scalar in `tape.backward`).
- **Buffers enumerated**: `current` (`alloc_zeros::<f32>(blocks)`, owned, reused
  pass→pass via `current = next`), `next` (per-pass owned partial), `in_slice`
  (borrowed input, pass 1 only), `d_out` (size==0 empty-sum path). No buffer
  outlives the function; no host staging.
- **Scope**: ONLY `sum_all`'s CUDA `Dirty::Device` impl changed. CPU/host
  `sum`/`mean` eager paths and all op-layer dispatch are untouched. Forward math
  identical: device block-reduce f32 sum vs host f32 `iter().sum()` —
  reduction-order noise only.
- **H20 verified (this session):** `cargo build --release --features cuda` green
  (typechecks the CUDA launch loop — not validatable on Mac). OPD smoke
  (`--smoke --backend cuda`) ran 3 steps, loss sane → `reduce.cu`'s
  `sum_partial_f32` NVRTC-compiles + `sum_all` correct on small tensors. 27B
  dense-path step: backward 2.1 s, GPU 98% (device-bound), loss 5.1510 == fused
  5.1535 to ~5e-4.
- **Mac gates green**: `cargo check`/`clippy` `-p autograd` (cuda,no-cuda +
  no-cuda) `-D warnings`; `cargo test -p autograd --features no-cuda` (CPU
  sum/mean/KL reference unchanged).

## Rule

A correct op-locality fix is not automatically the wall-clock fix. The CE forward
*was* mostly device-resident, and `sum_all`'s DtoH+sync was a genuine leak — but
decoding the op trace + GPU util showed the dominant 200 s was a *different* op
(`fused_linear_distill` on the host). **Decode the op trace before crediting the
op you just optimized**; the bottleneck and the thing you fixed can be two
different ops. The 27B-CE wall-clock win came from the default flip, not this.
