# FlashQLA takes over the chunkwise GDN route, forward and backward

**Date:** 2026-08-05 · **Status:** pending-remote (#81) — no nvcc on this Mac

## Context

The GDN backward was ARLE's own chunked-scan chain
(`linear_attention_chunked_scan_backward_f32` + chunk transfer / carry / grad),
measured at 25.8 s of a 50 s training step (26%, nsys seq=8192 cp=2). FlashQLA
(QwenLM, MIT, TileLang sm_90+) reports ~2x backward and 2-3x forward over the
FLA Triton reference, and its forward was already ported and AOT-built here —
but only reachable from `infer-cuda`. Training had a second, separate chunkwise
implementation: the six `gdr_{cumsum,a,solve,recompute,state,o}` stages in
`backend_cuda.rs`, themselves unreachable until the stale `seq<=32` clamp came
off (`e084cc8b0`).

## What landed

`fq_prepare_h` and `fq_bwd` ported from upstream `prepare_h.py` (675 lines) and
`fused_bwd.py` (1205 lines), following the forward port's convention: no torch
wrappers, varlen and intra-card-CP variants deleted, `(H, Hg)` as AOT
instantiation parameters, `DK=DV=128`, `chunk=64`. AOT rows for both head
geometries (H=32, H=48), ABI blocks, FFI externs.

The chunkwise route is now FlashQLA end to end, and **the six-stage forward it
replaced is deleted** along with the `a_tril` / `w` / `u` / `v_new` /
`initial_state` scratch that went dead with it. The recurrent forward and the
chunked-scan backward stay — they are the default path and the `use_mono` VRAM
fallback.

Everything sits behind `gdr_chunkwise_prefill` (default false), so the shipped
default path is unchanged.

**Tape shape change on the FlashQLA route.** `chunk_state` holds chunk 0 (the
carry) instead of every chunk, and `raw_output` is saved; `q/k/v/g_cumsum/a_inv`
are recomputed in the backward from the three cheap stages
(`gdr_fq_prep` + `fq_cumsum` + `fq_kkt`) rather than kept resident. At 8K tokens
/ 48 heads that trades ~400 MB/layer of resident tape for ~200 MB transient.
Which route ran is recorded in `SavedContext::LinearAttentionCtx::flashqla` from
the forward result — never re-read off the runtime flag, which can flip between
calls.

Two bridge kernels were unavoidable: `linear_attention_chunk_grad_f32` fuses the
RMS-gate backward, the GDN core and the prep backward in one per-token loop, so
inserting `fq_bwd` required splitting off
`linear_attention_rms_gated_backward_f32_to_bf16` and
`linear_attention_gdr_prepare_backward_f32`.

## Deviations from upstream, and from the brief

- `h` is bf16, not f32. Upstream allocates it as `k.dtype` and feeds it straight
  into a `T.gemm` against bf16 operands; f32 does not type-check there.
- `T.tma_copy(..., barrier=B)` became `T.copy`. TMA lowering must stay off (the
  AOT C wrapper cannot construct tilelang 0.1.11's host-built `*_desc` params).
  The ordering the barrier form buys survives — the issuing warp reaches its next
  `barrier_arrive(B)` only after a synchronous copy returns. Cost is producer-warp
  overlap, not correctness. Same pattern `fq_fwd` already ships.
- `fq_bwd` keeps upstream's `TL_DISABLE_DATA_RACE_CHECK`; the warp-specialized
  shared-buffer reuse is exactly what the checker rejects.
- `fq_prepare_h`'s `mt` output and `calc_mt` deleted — statically dead at
  `is_cp=False` (only ever written as zeros). `bar_3` went with it; arrive counts
  on `bar_0/1/2` re-derived and unchanged.
- `dg` is w.r.t. `g_cumsum`; upstream follows `fq_bwd` with a reverse chunk-local
  cumsum, folded here into the head of the prepare-backward kernel rather than
  instantiating a fourth TileLang kernel.
- `use_dht=True` but `dht` is fed zeros and `dh0` discarded — nothing in the tape
  chains a final-state gradient yet. Wired kernel-side so a future CP carry needs
  only the two pointers swapped.

## Verification — what is and is not covered

Local: `cargo check -p cuda-kernels`, `cargo check`/`clippy -D warnings -p
autograd --features cuda,no-cuda --lib`, `cargo test -p autograd` (60 passed),
`ast.parse` on both Python modules. The new CUDA C was preprocessed through
`clang++ -std=c++17 -fsyntax-only` with CUDA stubs.

**Not covered locally.** Every new line in `backend_cuda.rs` sits behind
`#[cfg(not(feature = "no-cuda"))]`, which the Mac lane cfg's out; compiling it
needs real nvcc. The first pod build is the compile gate, and pod clippy is the
only `undocumented_unsafe_blocks` gate (all five new unsafe blocks carry
`// SAFETY:`).

Pod gates, in order:
1. Build with `cuda,nccl` — the real compile check.
2. Liveness at a training length with `--gdr-chunkwise-prefill`. The 2026-06-19
   WGMMA deadlock the old clamp hid either reproduces on these kernels or does not.
3. f32-anchored parity. FlashQLA carries q/k/v/a in bf16 with g/beta/h/state in
   f32; the current path is f32 throughout. Precision change, so agreement between
   the two arms is not a gate — it needs the CPU f32 anchor.
4. Matched A/B on step wall, same seq, flag on vs off — and the tape trade above
   priced in VRAM, not just time.

## Rule

When one kernel replaces another, the guards added to avoid the old one outlive
it. Here a `seq<=32` clamp kept 48 of 64 layers on the slow path for six weeks
and was found only because different work needed the fast path's outputs as
inputs.
