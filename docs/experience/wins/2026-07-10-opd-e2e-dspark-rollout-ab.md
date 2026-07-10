# OPD e2e round: DSpark on the rollout serve — matched-task wall −25%, 4.11 tok/step, pass-rate in noise

## Context

Task #11 close: first controlled A/B of DSpark on the real agent-OPD rollout
lane (CC-as-harness, `scripts/cc_swe_baseline.py`), 1×H20, Qwen3.6-27B-FP8 +
z-lab DFlash, same binary/tasks/seeds, 16 swe_smith-easy tasks from
`staged-run1` (gold-gate verified on-pod), temp>0, cc-timeout 600 s.
Binary carries the 07-10 licenses (P2.5 partial-ctx + device sampled path).

**Structural finding:** `--spec-type dspark` exists only on `arle serve`; the
in-process `train agent-opd` rollout engine builds `EngineLoadConfig` without
it (`crates/cli/src/train_cli.rs:2434`) — wiring that is the follow-up that
makes DSpark reach the trainer's own rollouts.

## What Worked

| metric | plain | dspark | Δ |
|---|---|---|---|
| arm wall (16 tasks) | 3112 s | 2671 s | **−14.2%** |
| matched-task wall (12 non-timeout) | 1558 s | 1167 s | **−25.1%** |
| tokens/step | 1.00 | **4.11** | 4.1× |
| median per-task tok/s (incl. tool time) | 9.3 | 13.7 | +47% |
| prefix-cache hit | 75.8% | 84.3% | — |
| timeouts / OOM / serve errors | 2/16, none | 2/16, none | = |

- dspark generated +41% more tokens on matched tasks in 25% less wall; peak
  sustained 35.6 tok/s end-to-end including tool exec.
- Accept vs ctx (separate instrumented pass, 2680 chains): mean 3.32;
  **base>0 share 0.900** — partial-ctx engages on 90% of chains at CC's ~20K
  system-prompt regime, and deep-ctx accept (3.46 at 16–32K) beats cold
  (2.08) — the restore-blind-span worry inverted at this shape.
- Pass rates: plain 9/16 vs dspark 6/16 (≈1.1σ, one loss a timeout artifact,
  one re-passed cleanly on rerun). No decoded-case evidence of spec damage,
  but n=16 cannot license "no regression" below ~18pp — default flip needs a
  multi-sample gate.
- Backward anchor (synthetic-writeback 4096, qv-LoRA r16): 59.3 s/step =
  fwd 12.0 + CE 0.15 + bwd 46.8 + opt 0.45. Rollout remains the dominant
  wall-clock phase at this shape; distill attach needs
  `serve --dump-messages-dir` (not run this round, declared).

## Rule

- Spec-decode A/B on rollouts must report tokens/step and matched-task wall,
  not just arm wall — timeout tasks saturate the arm number.
- `[dspark-phase]` is sync-polluting; accept counters
  (`Qwen35DsparkExec.accepts/rejects`, executor.rs:4663) are write-only —
  export them to `/v1/stats` so future A/Bs are one-pass.
- A pass-rate "no regression" claim at n=16 is noise; gate default flips on
  multi-sample (≥3×) before claiming quality neutrality.
