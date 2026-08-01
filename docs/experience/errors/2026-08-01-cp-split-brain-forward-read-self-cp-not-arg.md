# CP parity FAIL was a real bug (cp split-brain), not gate miscalibration — 2026-08-01

## Context

The 256K context-parallel (CP) ring landed and `nd_parallel_parity` "FAILed":
seq=16 `cp_vs_f32=5.5e-2` against a single-card bf16 floor of `single_vs_f32=1.5e-5`
— 3700× the floor. The 131072 gate had also diverged. I spent time arguing this
was a *gate miscalibration* (two independent bf16 kernels compared at
near-identity tolerance), recalibrated the gate to anchor on CPU-f32, ran a
3-stage device bisection that all PASSED, and concluded "no bug." **That
conclusion was wrong.** A real bug existed; the f32-anchored gate was right to
fail.

## Root Cause

**A cp split-brain: two sources of truth for the CP axis.**
`masked_writeback_step` shards the sequence using its `cp` **argument**
(`opd.rs`: `cp.shard`, `cp.is_enabled`, inv_n). But the forward decided whether
to run the ring by reading a **different** source — the model field `self.cp`
(`qwen35.rs` `forward_batch_hidden_indices`). `self.cp` was set only by
`set_cp`, whose **sole non-test caller in the whole tree was the
`cp_hidden_parity` diagnostic**. No cli/production path called it, so `self.cp`
was always `single()`. Net: the sequence got sharded to rows
`[0,1,2,3,12,13,14,15]`, but `self.cp=single()` told the forward "these 8 rows
are one contiguous block → plain attention." Each rank attended only its own KV,
never the ring. Wrong hidden → loss 3.2425 vs f32 3.0729.

Why it hid so long: `tp` (cp's peer axis) is a **constructor arg** (it shards
weights at load, so it can't be forgotten). `cp` was bolted on as a **setter**
that fights the "weights are `&self`, pool-shared" invariant, so nobody threaded
it — only the diagnostic did.

## The two-step error that let it stand

1. First I called the 5.5% "bf16 noise" — before an f32 anchor existed. That was
   the right instinct to *distrust bf16-vs-bf16*, wrong to stop there.
2. After building the f32 anchor (correct move), the gate still failed 3700× the
   floor — a genuine bug signal — but I dismissed it as "miscalibration" on the
   strength of a **diagnostic probe** that passed at 8e-8. The probe re-ran a
   clean device-CE on a **correctly-assembled** `cp_full` hidden. It proved the
   CE aggregation was fine; it said nothing about the forward that feeds it,
   because the diagnostic **called `set_cp`** and so ran the ring — the exact
   step the shipping path skipped. A reconstruction of a path is not the path.

The tell I ignored: `cp_vs_f32 / single_vs_f32 = 3700`. Two same-precision
siblings cannot differ by 3700× the measured rounding of one of them. That ratio
alone was proof of a defect.

## Fix

Thread `cp` as a **forward argument** (beside `positions`) and delete the
`self.cp` field + `set_cp` setter (`3d9bc3717`): `forward_hidden_states` and
`forward_batch_hidden_indices` take `cp`; the non-CP internal callers pass
`single()`; the CP writeback branch passes the real `cp`. First-principles
placement — `tp` is model state (shards weights at load), `cp` is a forward-time
routing choice (weights are replicated across the cp group), so it belongs with
the call, not in the struct. `layer.forward` already took `cp` as a param; the
field was the anomaly. Production reaches the fix for free — it already supplies
`cp` as the step argument. Pod-verified (HEAD `3d9bc3717`, GPUs 1,3): seq=16
`cp_vs_f32` 5.5e-2 → **2.4e-4** (~bf16 floor, 83× under the 2e-2 margin); and the
256K rung (`ARLE_ND_SEQ=131072`, cp=2, local shard 65536 = the >65535 ring path)
completes a full forward+backward+optimizer step with `loss_single=3.232068`,
`loss_cp_sum=3.232163` (two ranks 1.629638 + 1.602526), `rel_err=2.958e-5` —
~3400× under the `bf16_tol=1e-1` gross-error bound. Both RUN_EXIT=0. The ring now
actually fires at 65536.

## Rule

When a value flows both as a call ARGUMENT and a struct FIELD for the same
concept, grep every setter of the field: a setter whose only caller is a
test/example means the shipping path silently uses the default, and the arg and
field have quietly diverged. And when an f32-anchored parity gate fails by many×
the same-precision floor, that is a bug — compute the floor ratio before ever
theorizing "miscalibration," and never accept a "no bug" verdict from a
diagnostic that reconstructs the suspect step instead of running it. Prefer
threading a forward-time concern (CP routing) as an argument over a mutable
setter that fights the immutable-weights invariant. See
`wins/2026-07-31-zigzag-ring-device-kernel-per-row-positions.md`.
