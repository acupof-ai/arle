# Healed reshape_backward, but the cost was reshape FORWARD under recompute

**Date:** 2026-08-06 · **Commit:** `7da312d0d` (kept — correct, parity-clean, not
reverted) · **Verdict:** WASH — the heal is a no-op for the profiled cost ·
Ref: [[reference_opd_backward_is_72pct_host_idle_launch_bound]]

## Context

Probe C flamegraph of the 80K OPD backward put **21% of on-CPU self-time in a
`reshape`/`rmsnorm` → libc `memcpy`**. An Explore agent confirmed
`reshape_backward` (layout.rs:498) and `rmsnorm_backward` (norm.rs:190) gate
device-vs-host on operand residency but — unlike `matmul_backward` — never
`ensure_device` first, so a `Dirty::Host` upstream grad drops them to a host
fallback (`tensor_host` full-`Vec` clone + DtoH + downstream HtoD). I read that
as the cause of the 21% and shipped the heal (mirror matmul's `ensure_device`).

## Result

Pod-measured (devops, `7da312d0d`):
- **Parity: PASS** — loss 4.537510 exact, grad_norm 7.965–7.985 in-envelope. The
  heal is numerically equivalent.
- **Step wall: WASH** — backward 315.6 s → 315.7 s (Δ≈0). (A 348.1 s first run was
  shared-CPU contention, not the heal; re-measured idle = 315.7 s.)
- **Re-profile: NULL** — reshape self-time 16.4% → 16.2%, rmsnorm 4.4% → 4.4%,
  upload_slice H2D 36.2% → 35.0%. Unchanged.

## Root cause of the null

Probe D resolved the caller frame: **50,958 of 50,958 reshape samples come from
`train::lora::LinearWithLora::forward`** — the forward `reshape` op *replayed
during checkpoint-recompute inside the backward*, NOT `reshape_backward`. The heal
targets the gradient op's operand residency; the actual `memcpy` is the forward
op's own host path, because the activation it reshapes is host-resident (offloaded
by the checkpoint machinery, not yet reloaded). Different code path — the fix
cannot touch it.

The 16.2% reshape and the 35% `upload_slice` HtoD are two symptoms of ONE thing:
recompute-forward running on host-resident (offloaded) activations. This is the
same checkpoint offload/reload bottleneck Probe A/C already named, so **Lever 2
(pinned + async offload of `offload_checkpoint_to_host`/`upload_slice`) is the
real lever, and it likely subsumes the 16.2%** — reshape falls to host only
because its input is on host; reload it to device and the forward reshape takes
its device path for free. No separate forward-reshape heal needed.

## Fix

`7da312d0d` stays: `reshape_backward`/`rmsnorm_backward` *were* missing the heal
(a latent host-fallback trap under a host upstream grad), parity is clean, and it
does not regress the wall — reverting a correct fix to restore a bug is wrong
(cf. the FP8 stream-guard fix precedent). It is simply not on the profiled hot
path. Lever 2 proceeds against the offload/reload round-trip on the same
315.7 s baseline (the heal left the baseline uncontaminated because it was a
no-op).

## Rule

A flamegraph names a *symbol*; it does not name the *call path* unless you read
the resolved caller frame. "reshape → memcpy, 16%" was `reshape` the **forward op
under recompute**, not `reshape_backward`. Before fixing the op a sampler labels,
confirm which caller frame the samples resolve to — a structural host-fallback in
the backward op can be real *and* irrelevant to the profiled cost at the same
time. This is the fourth mis-attribution of this backward (after 61k-dispatch,
graph-capture, pinned-halving); the first three were caught before code by
"measure before you act", this one reached a commit because I equated the Explore
agent's "backward op has a host fallback" with "the profiled cost is that op."
Read the caller frame, not just the symbol.
