# CUDA training-step nsys profile — attribution that retires "port one more op" thinking

**Date**: 2026-05-17 · **Hardware**: RTX 4070 Ti SUPER (sm_89, 16 GB)
· **Workload**: `arle train pretrain --preset small-25m --model-family qwen35
--batch 2 --seq 512 --grad-accum-steps 16` post M5.3b (commit M5.3b on `main`)
· **Headline tok/s at this profile**: 92.08 (baseline 78.6 → M5.3b +17.2 %)
· **Raw artefact**: `/home/ckl/arle-data/benches/profile-m53b/m53b_step1.nsys-rep`

## What the profile says

Profile of **1 full training step** (16 micro-batches, 178 s wall):

### Top CUDA API time

| Time % | Total / step | Calls | API |
|---:|---:|---:|---|
| **77.6 %** | **36.94 s** | **4 760** | `cuMemcpyDtoHAsync_v2` (Device→Host readback) |
| **17.8 %** | 8.46 s | 4 272 | `cuMemcpyHtoDAsync_v2` (Host→Device upload) |
| 3.9 % | 1.87 s | 4 817 | `cuStreamSynchronize` |
| 0.4 % | 0.20 s | 6 168 | `cuMemAllocAsync` |
| < 0.1 % | rest | — | kernel launches, event API, etc. |

**95.4 % of CUDA-API time is `memcpy`**, and DtoH is **4.4×** larger
than HtoD. Total bytes per step:

| Direction | GB / step | calls | avg / call | max single |
|---|---:|---:|---:|---:|
| Device→Host | 93.2 | 4 760 | 19.6 MB | **1 015.4 MB** |
| Host→Device | 45.4 | 4 272 | 10.6 MB | 1 016.1 MB |
| Device→Device | 0.48 | 72 | 6.7 MB | 158.8 MB |

The **`1 015 MB` single DtoH** ≈ `[B=2, S=512, V=248 070] × 4 B = 1 015.6 MB`.
This is the **`log_softmax` gradient** being read back to host so the
CPU `log_softmax_backward` can run. M5.3b shipped the *forward* device
override but the backward path still goes through the host.

### Top GPU kernels by time

| GPU % | Time / step | Calls | Kernel |
|---:|---:|---:|---|
| 46.4 % | 412 ms | 80 | `ampere_sgemm_128x128_nn` (matmul fwd) |
| **31.4 %** | **278 ms** | 16 | **`log_softmax_last_axis_f32`** (M5.3b) |
| 8.9 % | 79 ms | 16 | `ampere_sgemm_64x32_sliced1x4_tn` |
| 6.0 % | 53 ms | 144 | `cutlass_80_simt_sgemm_128x64_8x5_nt_align1` |
| **3.1 %** | **28 ms** | 24 | **`adamw_step_f32`** (G3) |
| 0.8 % | 7 ms | 144 | `mul_scalar_f32` |
| 0.5 % | 5 ms | 32 | `add_broadcast_f32` |
| 0.5 % | 5 ms | 144 | `rms_norm_f32` |
| 0.4 % | 3 ms | 64 | `ampere_sgemm_128x128_nt` |
| 0.3 % | 3 ms | 32 | `softmax_last_axis_f32` (M5.3b) |
| 0.1 % | 1 ms | 128 | `rope_f32` |
| 0.0 % | 0.3 ms | 16 | `embedding_f32` |
| ... | ... | ... | `silu_f32`, `scatter_add_rows_f32`, `neg_f32`, `gather_last_dim_f32`, ... |

**Total GPU kernel time per step ≈ 888 ms** (0.5 % of the 178 s wall).
GPU is **doing nothing 99.5 % of every step.**

### `linear_attention` is *not* in the kernel list

The `small-25m` Qwen3.5-family scratch preset defaults to dense
full-attention for both layers under this preset/config. The
`linear_attention` SSM scan path is not exercised, so any future
optimization should not assume Mamba2-style SSM is in the hot loop
for this shape. (It WILL appear when we use `--layers ≥ 4` with the
hybrid pattern, but that's a separate workload.)

## Attribution

For each 178 s step:
- **~45.4 s** spent in `memcpy` (PCIe Gen4 16x ≈ 32 GB/s peak; 138 GB
  per step ≈ 4.3 s ideal bandwidth bound; we pay 10× because every
  copy is small + serialized).
- **~133 s** the CUDA API is "idle" from `nsys` perspective but the
  **CPU is busy** running the autograd graph for the unported ops
  (`rms_norm_forward` / `rope_forward` / etc on host `Vec<f32>` for
  the now-uploaded inputs, plus the host-side backward path the
  ported forwards have to read back from).
- **~888 ms** the GPU is actually computing.

The "host-readback chain" thesis is **fully confirmed**: it's not the
size of any one op's roundtrip that hurts, it's the **number** of
roundtrips. 9 032 memcpys per step × ~5 ms p50 average launch+sync
cost = ~45 s, matching the API-side measurement.

## Why M5.3b only delivered +17 %

We ported 3 forwards (softmax / log_softmax / gather). Profile shows
those 3 kernels combined take 281 ms / step (0.16 % of wall), and the
memcpy savings from skipping their host-roundtrip in forward is
visible but small. **What we did NOT do**:

- `*_backward` for those same 3 ops (which is where the 1 GB
  `[B,S,V]` log_softmax-grad DtoH actually happens — note the
  forward log_softmax kernel ran on-device, but the gradient that
  needed it was still computed host-side because backward is unported)
- ~12 other host-only ops (`rms_norm`, `rope`, `silu`, `mul`,
  `mul_scalar`, `add_broadcast`, `embedding`, `mean`, `exp`, `neg`,
  `scatter_add_rows`, plus their backwards) — each contributes its
  own ensure_host pair per call.

## Evidence-based ranking for next move

This retires the speculative "M5.3b → 500-1200 tok/s" prediction in
the baseline doc and replaces it with the profile-grounded plan:

1. **Wave 1 — kill the single 1 GB DtoH**: port `log_softmax_backward`
   + `gather_last_dim_backward` to device-lazy on CUDA. Saves the
   single largest readback (~1 GB / step) → ~5 s / step wallclock
   immediate (~3 % of step). **Compounds** because the saved host
   compute time also disappears.
2. **Wave 2 — batch-port the remaining 12 ops** (both forward AND
   backward where missing): `rms_norm`, `rope`, `silu`, `mul`,
   `mul_scalar`, `add_broadcast`, `embedding`, `mean`, `exp`, `neg`,
   `scatter_add_rows`. All of their forward kernels are already
   present in `crates/autograd/src/backend_cuda/kernels/` — wiring
   the lazy-device path is mechanical (mirror the M5.3b pattern). The
   memcpy count goes from 9 032 → likely ≪ 1 000.
3. **Wave 3 — FusedLinearCE (Liger-style)**: avoid materializing
   `[B,S,V]` entirely. Unblocks `batch ≥ 8` (currently OOM at vocab
   248 070 × fp32 logits = 4 GB). **Only worth doing once Wave 1 + 2
   are committed**, because it requires `log_softmax_backward` to be
   on-device first to fuse with the LM-head matmul.
4. **Wave 4 — bf16 throughout activations + grads**: master fp32
   weights kept. Halves memory and gives ~1.5× compute on sm_89.
   Sequenced last because going to bf16 *before* device-resident ops
   means every `ensure_host` also pays a bf16→fp32 conversion.

The 92 tok/s vs ≥ 15 000 tok/s industry-target gap (163×) breaks down
roughly as:

- Wave 1 (1 GB DtoH kill): expected ~3-5 % wall → ~95-97 tok/s
- Wave 2 (memcpy count to ≪ 1 000): expected ~3-5× speedup → ~280-460 tok/s
- Wave 3 (FusedLinearCE + batch ≥ 8): expected ~3-5× → ~840-2300 tok/s
- Wave 4 (bf16): expected ~1.5-2× → ~1 260-4 600 tok/s

Closing the 163× target with just these four would require **everything
on the optimistic end**. The remaining headroom past Wave 4 lives in:
- **Kernel fusion across consecutive elementwise ops** (TileLang or
  hand-written): the per-call launch + sync cost dominates for tiny
  pointwise kernels. Fusing `mul → add → silu → ...` into a single
  launch cuts overhead proportional to op count.
- **CUDA-graph capture for the entire training step**: only worth
  doing once shapes are fixed (Wave 3 packs sequences) and the per-op
  overhead has been reduced enough that launch overhead is the
  residual cost. Probably another ~1.3-1.5×.
- **A re-evaluation of the per-op-eager autograd contract** itself.
  ARLE's autograd today is tape-eager: each op records a tape entry
  and runs immediately. A lazy graph that defers all ops until a
  terminal `backward()` call would fuse aggressively and skip
  intermediate materializations. This is the big-bet structural
  rewrite; reserve for if Waves 1-4 don't close the gap.

## Acceptance gate for the next milestone

Next wins entry must report:
- `tok_per_sec` median across ≥ 5 steps
- `nsys cuda_api_sum`: number of memcpys per step (gate: < 4 000 from
  current 9 032 after Wave 1 alone; < 1 000 after Wave 2)
- `nsys cuda_gpu_kern_sum`: confirm `*_backward` kernels appear on the
  GPU side (today they're absent because backward runs on CPU)

If a milestone bumps `tok/s` without lowering the memcpy count, it's
optimizing the wrong thing — same SOLID gate as G3 and M5.3b.

## Rule

**When optimizing a host-bound pipeline, profile the API surface first,
not the kernel timeline.** The kernel timeline shows the GPU is idle
99 % of the time, but that's the *symptom*. The *cause* is the
4 760-call DtoH + 4 272-call HtoD count. Future training-perf wins
entries must include a `memcpy count per step` row from `nsys
cuda_api_sum`, and the acceptance gate must be on that count
dropping, not on tok/s alone.
