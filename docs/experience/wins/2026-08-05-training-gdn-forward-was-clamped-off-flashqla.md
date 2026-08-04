# The training GDN forward never reached FlashQLA — a stale seq<=32 clamp guarded a kernel that no longer exists

**Date:** 2026-08-05 · **Commit:** this one · **Status:** pending-remote (#81)

## Context

nsys put the GDN chunk backward at 25.8 s of a 50 s training step (26%, seq=8192
cp=2), second only to ring attention. The plan was to adopt FlashQLA's backward
(QwenLM, 2026-04, MIT, TileLang sm_90+; ~2x backward and 2-3x forward over the
FLA Triton reference). Its forward is already ported and AOT-built here —
`fq_cumsum` / `fq_kkt` / `fq_fwd` in `kernels.toml`, default-on, licensed for
inference at 33K prefill -27%.

FlashQLA's backward reads `h` (per-chunk states, from upstream `prepare_h.py`)
and `a` (the `kkt_solve` output). Neither exists unless the forward ran through
the same chunked path. So the first question is whether the training forward
does — and it does not.

## What was found

`linear_attention_forward_device` routed on
`p.seq_len <= 32 && gdr_chunkwise_prefill_enabled()` (backend_cuda.rs:6028).
At any training sequence length that is false, so the training forward always
took the recurrent branch and the backward always took the hand-rolled
`linear_attention_chunked_scan_backward_f32` / `chunk_transfer` / `chunk_carry`
/ `chunk_grad` chain — the 26%.

The clamp came from `abb1ed995` (2026-06-19), "route OPD gated-delta forward to
device recurrent (sm_90 chunk-WGMMA deadlock)". The kernel it was working
around was replaced six weeks later by the FlashQLA port (`778fef873`,
2026-08-02), whose warp-specialized kernels are a different implementation. The
clamp was never revisited.

## The change

Delete the length clause; the `gdr_chunkwise_prefill` flag alone gates the path.
The flag still defaults false in the autograd runtime flags, so this is inert
until `--gdr-chunkwise-prefill` is passed — the default is unchanged and no
measured path moves without an explicit opt-in.

## Verification — pending-remote

Mac has no nvcc; typecheck only
(`cargo check -p autograd --features cuda,no-cuda --lib`, clean). The gates that
matter run on the 8xH20 pod under #81:

1. **Liveness / deadlock.** A long-seq (>=32768) training step with
   `--gdr-chunkwise-prefill`. The 2026-06-19 deadlock either reproduces on the
   new kernels or it does not; that is the whole question the clamp was hiding.
2. **f32-anchored parity.** FlashQLA carries q/k/v/a in bf16 with g/beta/h/state
   in f32; the current training path is f32 throughout. This is a precision
   change on the GDN forward, so mutual agreement between the two arms is not a
   gate — it needs the CPU f32 anchor.
3. **Matched A/B** on step wall at the same seq, flag on vs off.

## Rule

A workaround clamp outlives the thing it worked around. When a kernel is
replaced, grep for the guards that were added to avoid its predecessor — the
clamp here silently kept 48 of 64 layers on the slow path for six weeks, and
was found only because a *different* piece of work needed the fast path's
outputs as inputs.
