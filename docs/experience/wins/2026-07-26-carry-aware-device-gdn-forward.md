# Carry-aware device GDN forward — ABI + seed (tranche 1)

> Status: Shipped (device path dead until tranche 2 routes it — see Rule)

## Goal

Let the OPD frozen-prompt carry (recurrent state + conv window) seed the Gated
DeltaNet **device** forward, so the existing seq-independent device chunked
backward can replace the host full-sequence recompute (`state_history` ≈ 86 GB
at seq=40960 → the 97 GB OOM wall). Tranche 1 is additive: build the capability,
keep the default path byte-identical, route nothing yet.

## What Worked

Seeding `initial_state` into `final_state` before the recurrent loop makes
`chunk_state[0]` snapshot the carry automatically (`backend_cuda.rs:4305` copies
final_state → chunk_state[0] before the first chunk). The three chunked backward
kernels read `chunk_state[c]` with uniform indexing and no `chunk_idx==0`
special-case, so **seeding a nonzero carry needs zero backward-kernel changes** —
verified by reading chunk_transfer/chunk_carry/chunk_grad. `chunk_grad` discards
chunk-0's grad_state (never written back), which is correct: carry is a frozen
prompt state (`requires_grad=false`), gradient must not flow back into it.

Conv1d carry: forward reads the carried left-history taps
(`conv_tail[(src_t+tail_len)*channels+c]`, nullptr → byte-identical zero-pad);
backward adds the boundary-tap `grad_weight` (the carried history did contribute
to the forward, so its weight-grad is real) but writes no `grad_input` (carry
frozen). Both `.cu` kernels gained trailing `const float* conv_tail, int tail_len`.

Change set: `LinearAttentionDeviceForwardArgs` +2 nullable carry fields
(`backend.rs`); forward seed + fan-out threading + two conv launch sites
(`backend_cuda.rs`); conv fwd/bwd boundary (`linear_attention.cu`); carry Option
threaded through `try_linear_attention_forward_device` (`ops/linear_attention.rs`).

## Measured (H20 pod, sm_90, `-p autograd` cuda lane)

- `cargo build --release --features cuda`: clean (first nvcc compile of both
  edited kernels + all cuda-gated bodies linked).
- `cargo clippy --release --features cuda -- -D warnings`: clean.
- `linear_attention_carry_grad_matches_numeric`: 1 passed — **host path** (tranche 1
  does not route carry to device; this confirms the host oracle intact + cuda
  build linked, NOT device-carry coverage).

Default (no-carry) path byte-identical: every new arg nullable/None,
`conv_tail==nullptr` reproduces the prior zero-pad, seed gated on
`carry_state.is_some()`, backward chunked kernels untouched.

## Rule

The device carry path (forward seed + both `.cu` edits) is **dead code in tranche 1**
— nothing passes `Some`, so it is validated only by the pod compile gate, never
by a live run. This is deliberate staging (CUDA-on-Mac is pending-remote), but
additive dead code rots: land tranche 2 (route `linear_attention_core_with_carry_taped`
host→device, `linear_attention.rs:550`) promptly so the seed gets its live
device-carry gradcheck + the VRAM-wall proof. Backward tape records carry ctx as
`None` on the device path by design — carry lives in `chunk_state[0]`, so
`has_carry` stays false and backward auto-takes the device chunked path; the
host recompute survives as the CPU/unsupported fallback. See
`docs/research/2026-07-26-carry-aware-chunked-gdn-backward.md`.
