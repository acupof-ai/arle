# qwen4_exp decode 899 → 85 ms/token: seven levers, one loop, every step parity-gated

## Context / Goal

After first token, the bring-up path ran 899 ms/token (later re-measured 961 in
a clean sitting). Goal, set as a completion criterion rather than a wish:
**≤ 100 ms/token with the residual fully attributed** (profile residual < 5%),
top-1 unchanged, parity error profile not degraded, every lever either landed
or measured-and-declined.

## The loop

Every lever followed the same cycle: profile (the env-gated per-stage table
whose rows PARTITION the wall) → land the top line → parity harness must stay
digit-identical → full forward must keep ` Paris` top-1 → commit with the
same-sitting before/after. No lever shipped on an expected number.

## The seven levers, measured

| # | lever | ms/token | mechanism |
| --- | --- | --- | --- |
| 0 | (S6 round) dense tier onto device | 961 → 266 | 75.8% of the token was the host bf16 matvec at 9.76 GB/s — ~338 ms of it spawning 4816 OS threads/token |
| 1 | resident linear attention | 266 → 170 | GDN state + conv ring live ON device (36 × 3.1 MB); projections from the tier; HF↔GGUF head map applied to ~24 KB of ACTIVATIONS (a 20-line block-perm kernel), not 7.8 GiB of weights |
| 2 | device PLE | 170 → 140 | the F32-only assert fell; the conv RING ≠ the host SHIFT buffer — one rotation reconciles them, and the harness's new `ple.ring` stage read rel 1.6e4 before the rotation and 2.5e-4 after |
| 3 | MoE tail in one submit | 140 → ~137 | three scale0 slots written before recording kill the per-projection fences; shared expert joins the batch (its F32-only gate had it running as 96 host-driven submits) |
| 4 | device full attention | ~137 → 114 | TWO gates kept 12 layers on host: an F32 format check, and hybrid constructing Qwen4Dev with an EMPTY full-layer list — no KV planes, ready never true |
| 5 | staged token loop | 114 → 90 | the baton (h/x/y) never leaves the device; record-only variants of every stage; fences 339 → **50** (48 MoE ids + PLE ring + logits) |
| 6 | BF16 verbatim dense tier | 90 → 88 | the checkpoint's own bytes (memcpy, no re-encode): load −45 s AND 221,186 weights back to true value — the logit moved 15.75 → 15.80, TOWARD the checkpoint |
| 7 | BF16 hyper-connection tier | 88 → **84.9** | 2.5 GB/token of F32 mix weights halved; 8-line WEIGHTS_BF16 shader variants; exactness of bf16→f32 makes the end-to-end logit a sharp validator (moved 2e-4) |

Final: **84.9 ms/token = 11.8 tok/s**, 50 submits, residual 0.5%, all in one
sitting per comparison. GPU busy ~67 ms of it — the remaining wall IS the
bytes, as it should be.

## What the harness caught (the reasons the loop has a gate)

- **Resident-state drift** (lever 1): with device state persisting, device and
  host trajectories legitimately diverge (the bf16 conv quantizer flips
  near-zero channels differently and the state compounds it) — `linear.gated`
  read 2.4e-2 against a 4.7e-3 record. `seed_state()` per token restored
  single-token isolation and the table returned DIGIT-FOR-DIGIT, worst-element
  positions included. Proof by restoration, not by argument.
- **Ring vs shift** (lever 2): the first read-back of the device-advanced PLE
  ring was rotated by one slot — corrupted history that would have surfaced
  three tokens later as subtle degradation. The comparison caught it at 1.6e4
  BEFORE it had a threshold; it has one now.
- **The reserve guard fired on its own author** (lever 7): halving the HC tier
  let spill_to_fit keep an extra expert stack on device, leaving 0.98 GiB
  against a 1 GiB reserve that predated the KV planes and resident state this
  same arc added. The loader refused, loudly, with the arithmetic in the
  message. Reserve is now 1.5 GiB with the breakdown documented.

## Learnings

**The profile's job is to make the next lever boring.** Every step was "the
top line of the table, again" — no debates, no speculative tuning. The two
one-line fixes with the biggest yields (the empty full-layer list, the
F32-only asserts) were invisible to reasoning and obvious to the table.

**Record/execute split beats flag-parameterized stages.** Every stage became a
record-only function plus a thin write/record/flush/read wrapper, so the
staged loop and the fallback path run THE SAME recorded code — numerical
identity between them is structural. The staged loop's logit matched the
per-stage path digit-for-digit on the first run.

**Format gates must name what they actually require.** Three separate
"F32-resident" asserts turned out to mean "written before the tier existed."
The fix each time was the same: route through the one format-dispatching
record path. A format enum only helps if there is exactly one place that
matches on it.

**On UMA, the last 20% is fences.** After the bytes are on the right side,
90 µs × submits is the tax; the only structural fix is keeping the baton on
device and fencing only where the host genuinely must look (ids for the
scale gather, the logits).

## Rule

Set the finish line as a measured criterion before optimizing ("≤ X ms,
residual attributed, output unchanged"), then run profile → top lever →
parity gate → commit until the criterion holds. Never land two levers in one
measurement window, and never re-derive a number the table can give you.
