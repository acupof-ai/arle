# Carry GDN device reroute + live gradcheck (tranche 2)

> Status: Shipped. Closes the tranche-1 "dead device path" Rule — the seed now
> gets its live device-carry gradcheck and the VRAM-wall proof.

## Goal

Route the OPD carry path (`linear_attention_core_with_carry_taped`) from the host
full-sequence recompute onto the tranche-1 device carry-seeded forward + existing
chunked backward. Tranche 1 built the capability dead; this makes it live.

## What Worked

Flip the forward gate: try `try_linear_attention_forward_device` first (seeds
carry → `chunk_state[0]`), host recompute demoted to CPU/unsupported fallback.
Backward gate changed from `has_carry` to `needs_host_recompute =
initial_state.is_some()` — the device path records carry ctx as `None`
(carry lives in `chunk_state[0]`), so `needs_host_recompute` is false and backward
auto-takes the device chunked path. `initial_conv_window` threaded into the
backward launch (tranche 1 added the kernel ABI but hardcoded the launch to null)
so the conv-boundary `grad_weight` is real, not zero.

## Measured (H20 pod, sm_90)

**Live device-carry gradcheck — PASSED** (`5fbf38e4e`, GPU 1): build 0, clippy 0,
`cuda_linear_attention_carry_grad_matches_cpu` exit 0, non-carry regression exit 0.
This is the coverage tranche 1 could not have (dead path). dq 1.74e-3, dconv
6.29e-3 — both **bf16-rounding artifacts**, not logic bugs (see below).

**bf16-artifact A/B** (controlled, one build, runtime switch on the 5 backward
read sites, same carry fixture): device backward reads saved bf16 `qkv_conv`
(bit-matches the production forward — `backend_cuda.rs:4164` feeds bf16 to
`chunk_prepare_cuda`); the CPU oracle recomputes conv in f32.

| grad | bf16 qkv_conv (production, shipped) | f32 silu(preact) |
|------|-------------------------------------|------------------|
| dq max_abs   | 1.738e-3 | 5.08e-4 |
| dconv max_abs | 6.292e-3 | 3.23e-4 |

Flipping ONLY the read precision drops both under floor → pure bf16 rounding.
Concentrated on carry-fed boundary tokens (dq worst tok 0-1) and conv boundary
taps (dconv worst tap 0-1), decaying to ~0 mid-sequence — the boundary signature.
The bf16 gradient is the **correct adjoint of the bf16 forward**; f32 would be
finer-but-wrong. Fix was test-side: carry variant `abs_tol` 1e-3→1e-2 (`c4709d348`),
kernel untouched.

**VRAM wall (masked writeback, `--writeback-offload true`, 27B-FP8):**

| seq | rc | smi peak | result |
|---|---|---|---|
| 24576 | 0 | 70603 MiB | completes |
| **40960** | **0** | **93163 MiB** | **was OOM (97GB / 409MiB free) → completes** |
| 65536 | 1 | — | forward alloc_zeros fail (new forward wall) |

Trainable seq (single H20): **24576 → 40960** (1.67×). NOT the 64× arithmetic —
that counted only backward `state_history`, ignoring forward's retained checkpoint
activations (75.9 GB at 40960). 256K still needs LA-chunk / sequence parallel;
this reroute lifts the wall one notch and gives the carry path a chunked backward.

## Pod A/B — loss parity + perf license

Cross-commit A/B (the tranche-2 flip is in the forward — the device path records
`initial_state` as `None`, so a runtime host-vs-device toggle can't exist). Arm A =
HEAD `5fbf38e4e` (device chunked backward); arm B = `a03bf04f2` (= `d6ae52dc1^`,
clean host recompute). Both arms: same records, same seed, GPU 1, two full builds
in isolated trees. 27B-FP8, `--writeback-offload true`.

**#2 loss parity — PASS.** Deterministic `--replay-records` lane (student-only, no
MoE rollout), 1 record (seq 4111, 717 masked tok — carry fires: >writeback-window
512) × 12 epochs × 3 runs/arm. The carry-boundary logic (chunk-0 seed, dq tok 0-1 /
dconv tap 0-1) is structurally identical at 4111 or 24576, so 4K exercises the exact
device carry path. Median loss curve:

| epoch | A (device) | B (host) | rel Δ |
|---|---|---|---|
| 0 | 0.1083 | 0.1083 | 0 (fwd byte-identical) |
| 1 | 0.0876 | 0.0874 | 2.3e-3 |
| 2 | 0.0667 | 0.0668 | 1.5e-3 |
| 4 | 0.0355 | 0.0354 | 2.8e-3 |
| 6 | 0.0180 | 0.0180 | 0 |
| 8 | 0.0092 | 0.0092 | 0 |
| 11 | 0.0039 | 0.0039 | 0 |

Epoch 0-1 bit-identical (forward is byte-identical; divergence only enters epoch 2+
via backward-driven weight updates — the expected signature). Every-epoch cross-arm
mean Δ ≤ 2e-4 abs, **smaller than arm A's own run-to-run jitter (≤5e-4)** — device
is statistically indistinguishable from host. The single-run tail rel spikes (epoch
9 ~2.9%) are `%.4f` print-quantization on ~0.007 losses, not real divergence; against
medians-with-jitter they vanish. Well inside the <1e-2 bf16-grad-noise bar.

**#3 perf license — device is +2.6% slower (VRAM-for-time trade), not faster.**
seq=24576, backward wall-clock/step, 3 runs/arm, medians:

| arm | backward median | forward median |
|---|---|---|
| A device chunked | **565.16 s** (565.16/564.26/566.64) | 137.9 s |
| B host recompute | **550.61 s** (549.79/550.61/550.74) | 133.8 s |

Device backward is **+14.55 s (+2.6%)** vs host at seq=24576. The reroute is not a
speed win — it buys the VRAM headroom that lifts the trainable-seq wall 24576→40960
(host recompute OOMs at 40960; device completes at 93.2 GB peak). Trade: pay ~2.6%
per-step wall to unlock 1.67× trainable sequence. Both arms same forward (~2.9%
forward gap is bf16/MoE run jitter, not a code difference — forward is byte-identical).



Mixed-precision gradcheck's judge must match the forward's ACTUAL precision. A
bf16-reading backward that matches a bf16 forward is correct even when it fails an
f32 oracle — decide artifact-vs-bug with an A/B flipping ONLY the read precision,
then fix the test tolerance, not the kernel. Retract theory-derived VRAM ratios
until measured: the 64× was arithmetic on one term, the real win is 1.67× trainable
seq. See `feedback_mixed_precision_backward_adjoint_of_forward_precision`.
