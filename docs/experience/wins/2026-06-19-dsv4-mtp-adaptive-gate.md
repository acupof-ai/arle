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

## Results — MEASURED on pod (2026-06-20, TP4 GPUs 0-3, 6 prompts ×2, best-of)

| prompt | no-spec tok/s | MTP tok/s | accept | Δ |
|--------|---------------|-----------|--------|---|
| prose | 39.00 | 38.27 | 0.581 | −1.9% |
| explain | 38.62 | 36.79 | 0.549 | −4.7% |
| code | 38.30 | 39.29 | 0.638 | +2.6% |
| reason | 38.64 | 39.06 | 0.618 | +1.1% |
| list | 38.57 | 38.43 | 0.599 | −0.4% |
| counting | 38.22 | 34.99 | 0.504 | −8.4% |

Gate@0.55 (separate 3-config run): pulled both a prose essay (raw MTP 42.26) and
counting (raw MTP 35.11) to ~37 — i.e. it gave up MTP's win and only partly
recovered the loss. **Net-negative / misfires.**

## Verdict — DROP the gate, do NOT flip MTP default-on (1-head MTP is a wash)

The load-bearing fact: **1-head MTP's break-even is ~0.59 acceptance, which sits
dead in the middle of the natural-text acceptance distribution (0.55–0.64).** So
MTP is structurally a wash (±5%) on natural text and a loss on low-acceptance
content (counting 0.50). The pre-compaction "−13% typical" and the single-essay
"+10.7%" were both prompt-specific outliers; the controlled 6-prompt run shows
near-break-even. A gate cannot turn a wash into a win — and ours misfires.

Decisions: (1) the held `spec_type Auto→MTP` default flip is DROPPED (keep MTP
opt-in via explicit `--spec-type mtp` / `ARLE_DSV4_SPEC_DECODE=1`). (2) The gate
code stays as a default-off opt-in (harmless, revisit only with a higher-
acceptance draft head). (3) "投机解码默认好用" with the current head is NOT
achievable by tuning — it needs a **2-head MTP draft head** to push acceptance
clear of ~0.59 (training, out of inference scope) — see
`errors/2026-06-19-dsv4-b1-foundation-lever-search-exhausted.md`.

## Learnings

- A decoding-default verdict needs the acceptance DISTRIBUTION, not one prompt:
  MTP tok/s swung −8% to +11% across prompts; single-prompt A/Bs (both the −13%
  and the +10.7%) were outliers that would have mis-set the default either way.
- The break-even ACCEPTANCE (not tok/s) is the invariant to reason from: with
  break-even ~0.59 inside the 0.55–0.64 natural-text band, no threshold gate
  helps materially. Effects are <5pp → a firm ship/no-ship still wants multi-seed,
  but the acceptance-physics verdict (wash) is already solid.
