# DSv4-Flash adaptive MTP gate — make speculative decode good-by-default (pending-remote)

**Status:** pending-remote. Code + unit test landed; pod calibration A/B owed.

## Goal

Make MTP speculative decode "默认好用" (good by default) for B=1 without
regressing typical prompts. The gate is opt-in (`ARLE_DSV4_MTP_ADAPTIVE=1`,
default-off) until the threshold is pod-calibrated.

## Hypothesis (break-even physics, not A/B fishing)

DSv4-Flash B=1 8×H20 TP4: MTP(dt=3) ≈ 68 ms/step, no-spec ≈ 26 ms/step. MTP
beats no-spec only when it emits > 68/26 = **2.6 tok/step**, i.e. accept rate
**> ~55%**. Measured: typical ~38% accept → ~2.3 tok/step → **34 t/s vs no-spec
39 (−13%)**; predictable text (counting) ~52% → 41 t/s. So a naive MTP-default-on
ships a typical-prompt regression. A gate that runs MTP only while a running
accept-rate EMA clears break-even — and falls back to a *warm* no-spec step
(same cost, keeps the draft head staged) otherwise — should recover no-spec
speed on typical prompts while keeping the MTP win on predictable text.

## What landed (code)

- `executor.rs`: `forward_mtp_warm_step` (the no-spec-but-stage-draft step,
  REUSED by both final-prefill and the gate fallback); gate block in
  `forward_decode_tokens` (B=1) before `spec_step`; EMA fields.
- `executor/spec_decode.rs`: pure `mtp_should_speculate(accept_ema, skip_streak,
  min_accept, probe_interval)` (unit-tested); `mtp_note_accept` EMA update in
  `spec_step`; OnceLock-cached env config (`ARLE_DSV4_MTP_ADAPTIVE` /
  `_MIN_ACCEPT` default 0.55 / `_PROBE` default 8); periodic probe to refresh the
  EMA after a run of skips.
- B=1 only (the batched B>1 path is already a win, never gated, never perturbs
  the EMA). Default-off → decode path byte-identical to baseline.

## Params / Env

8×H20 **TP4** (GPUs 0-3 only), DSv4-Flash, B=1, greedy. Gate flags above.
Verified locally: Mac CUDA-Rust typecheck (`cuda,no-cuda`) clean; `cargo fmt`
clean; unit test compiles (link needs CUDA → runs on pod/CI).

## Results

**PENDING-REMOTE.** Pod A/B owed (same-binary, same-prompt, two env flips):
1. Calibrate `MIN_ACCEPT` so the gate's emit/step ≥ no-spec on a *typical* SLO
   prompt (not a counting smoke shape) — sweep 0.45–0.65.
2. License gate-on as default only if: typical prompt ≈ no-spec (regression
   gone) AND predictable prompt keeps the MTP win AND needle gate passes. Then
   flip `--spec-type` default → MTP **together with** the gate default-on (the
   held `cli/serve.rs` Auto→MTP flip stays uncommitted until this passes).

## Problems / Learnings

- MTP-default-on is NOT licensed by a counting-shape sweep (44 t/s peak) — that's
  a smoke shape; the SLO workload (typical chat, ~38% accept) is where it
  regresses. SLO verdict from the SLO workload.
- The gate is the responsible answer to the acceptance cap; the *other* path to
  raw B=1 speed (55 t/s) needs a 2-head MTP draft head (training, out of
  inference scope) — see `errors/2026-06-19-dsv4-b1-foundation-lever-search-exhausted.md`.
