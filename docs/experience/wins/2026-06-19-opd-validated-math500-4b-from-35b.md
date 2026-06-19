# OPD validated on MATH-500: Qwen3.5-4B student 0.60→0.78 from a 35B teacher (82% of the gap, 25 steps)

## Context
First end-to-end validation that ARLE's On-Policy Distillation actually lifts a
student's capability. Qwen3.5-4B LoRA student (rank 32, all-linear), distilled
on its own greedy rollouts against the Qwen3.6-35B-A3B-FP8 teacher running on
ARLE's own serving runtime (`--teacher-runtime infer`). MATH-500 greedy
exact-match, max_tokens=4096, single-seed bring-up.

## What Worked
Capability curve (n=100/point, retry-clean, **0 request_error**):

| point | MATH-500 acc | CI95 |
|---|---|---|
| base 4B (step 0) | 0.60 | — |
| fwd-KL @ step25 | 0.78 | [0.689, 0.850] |
| fwd-KL @ step50 | 0.80 | [0.711, 0.867] |
| fwd-KL @ step75 | 0.77 | [0.678, 0.842] |
| reverse-KL @ step25 | 0.78 | [0.689, 0.850] |
| reverse-KL @ step50 | 0.75 | [0.657, 0.825] |
| teacher 35B (ceiling) | 0.82 | [0.69, 0.90] |

**0.60 → 0.78 = ~82% of the base→teacher gap closed in 25 OPD steps** — STRONG /
SOTA-magnitude (the ≥0.71 bar). Curve plotted on the README
(`docs/assets/opd-capability-curve.png`).

Three enabling fixes (each its own wins/commit):
- **gated-delta forward → device recurrent** (`4d9c77ae`): the chunk-WGMMA path
  deadlocks on sm_90; route to the recurrent kernel like the inference path.
- **device LoRA-merge** (`011ec48f`): per-step LoRA re-merge into the rollout
  engine was a host triple-loop (~84 s/step, 62% of the step) → on-device GEMM,
  **510× (78 s → 0.15 s)**, bit-exact.
- **multi-arm concurrency** (per-arm unique session/port/JIT) — see
  [[2026-06-19-opd-multiarm-shared-resource-collision]] (errors).

## Caveats (honest)
- **Single-seed, n=100** (CI ±~0.08; step25 lower bound 0.689 ≈ the 0.71 bar) →
  the SOTA-magnitude claim is point-estimate-strong but **multi-seed (≥5) is
  required to lock it** — in flight (`opd-multiseed-20260619-150340`, 3 arms × 5
  seeds).
- **Arms not yet differentiated**: fwd-KL ≈ reverse-KL at step25 (both 0.78);
  step-to-step variation (0.78/0.80/0.77) is within noise. Multi-seed needed to
  rank the recipe (w71uqbosx predicted reverse-KL/stochastic > fwd-KL-greedy).
- **The anchors were a harness trap**: the original 0.10 base / 0.70 teacher were
  artifacts (max_tokens=1024 truncating CoT + Qwen contamination + a broken
  proxy); corrected at 4096 to 0.60 / 0.82. Always re-measure anchors on the
  exact student harness before any gap math.

## Rule
OPD lifts a small student materially **only when the teacher genuinely beats the
base on the target benchmark, measured on the same harness** — verify that
gate-zero first (the 0.10→0.70 artifact nearly produced a false "no room"
verdict; corrected 0.60→0.82 showed real room). This gate now governs the
agentic extension (BFCL V4, base 0.503 → teacher 0.673).
