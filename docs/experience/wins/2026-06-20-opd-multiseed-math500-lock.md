# OPD multi-seed LOCK: Qwen3.5-4B 0.518 → 0.792 MATH-500 (+27pp, CI-separated, reverse-KL wins)

## Context
Multi-seed confirmation of the single-seed OPD bring-up win
([[2026-06-19-opd-validated-math500-4b-from-35b]]). That entry showed base 0.60
→ student 0.78 at single seed, n=100, with two open caveats: **(1) single-seed**
(±~0.08, step25 lower bound grazed the 0.71 bar), **(2) arms not differentiated**
(fwd-KL ≈ reverse-KL ≈ 0.78). This run closes both: **3 recipe arms × 5 seeds ×
step50**, each evaluated on the **full MATH-500 (n=500)**, greedy exact-match
@4096 tokens, retry-clean (0 request errors across all 16 jobs). Teacher
Qwen3.6-35B-A3B-FP8 (`--teacher-runtime infer`), student Qwen3.5-4B LoRA
(r32/a64 all-linear, `--kl-mask completion`, `--no-fused-distill`).

## What Worked

**Locked capability table (step50, n=500/seed, 0 request_error):**

| arm | seeds | mean | sd | mean 95% CI | per-seed acc |
|---|---|---|---|---|---|
| **base 4B** (step 0) | — | **0.518** | — | [0.474, 0.561] (Wilson, n=500) | 259/500 |
| **reverse-KL** (greedy) | 5 | **0.792** | 0.0045 | [0.788, 0.796] | .786 .790 .792 .794 .798 |
| forward-KL (greedy) | 5 | 0.781 | 0.0064 | [0.775, 0.786] | .772 .778 .780 .786 .788 |
| stochastic (temp 0.9) | 5 | 0.777 | 0.013 | [0.766, 0.788] | .758 .770 .782 .786 .788 |
| teacher 35B-A3B (ceiling) | — | ~0.82 | — | — | (prior anchor) |

**Step25 trajectory** (intermediate ckpt, 5 seeds each): reverse-KL **0.783 ± 0.011**
(already near-converged — it barely moves 25→50), forward-KL **0.748 ± 0.042** (noisy;
seed3=0.678 is an outlier that recovers to 0.786 by step50), stochastic **0.749 ± 0.023**.
Every arm rises 25→50; reverse-KL is the highest *and* lowest-variance at **both**
checkpoints (see `opd-multiseed-curve.png`).

**Headline: base 0.518 → reverse-KL 0.792 = +27.4pp, fully CI-separated** — the
base Wilson upper bound (0.561) sits **far below** the student per-point lower
bound (~0.752); zero overlap. SOTA-magnitude (≥0.71 bar) is now **locked**, not
point-estimate.

**Arm differentiation (the open question from the single-seed run):**
**reverse-KL leads** — highest mean (0.792) AND tightest spread (sd 0.0045) at
both step25 and step50. Its mean-CI lower bound (0.788) sits *just above*
forward-KL's upper bound (0.786) — so reverse-KL is **leading, but only
borderline CI-separated from forward-KL at n=5** (high-confidence on
mean+variance, not decisively separated). **stochastic is worst and noisiest**
(0.777, sd 0.013 — 3× reverse-KL's), i.e. temperature-0.9 rollout *adds variance
without lifting the mean*. Recipe verdict: **reverse-KL greedy** is the default
to carry forward (best mean, lowest variance, earliest convergence). The
base-vs-all-students separation (+26-27pp) is overwhelming and unambiguous; only
the inter-arm ranking is n=5-limited.

**Anchor correction (intellectual honesty):** the single-seed entry's base
**0.60 (n=40)** was a small-n overestimate; at **n=500 the base is 0.518**. The
lower, properly-measured base makes the lift *larger* (+27pp vs the claimed
+18pp), not smaller — the win strengthens under scrutiny.

## Rule
A small-n anchor (n≤50) on the *base* can mis-size a distillation gap by ~8pp in
either direction — re-measure the base at full n before quoting "X% of the gap
closed". Here it cut the other way (base lower → bigger lift), but the discipline
is the same: the load-bearing number in a lift claim is the baseline, and it
deserves the same n as the treatment. Arm-ranking with magnitude <2pp
(reverse 0.792 vs fwd 0.781) genuinely needed the ≥5-seed mean±σ to call —
single-seed they were indistinguishable (both 0.78).
