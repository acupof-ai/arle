# Rubric-based On-policy Distillation (ROPD) — adoption analysis for ARLE / SOPD

**Date**: 2026-06-14 · **Driver**: ckl · **Status**: research → recommended OPD-faithful cut → issue [#98](https://github.com/cklxx/arle/issues/98)

> Companion to the [SOPD survey](2026-06-14-self-training-lora-options-survey.md) (Axis A gains **A5**)
> and [SOPD plan](../plans/2026-06-14-self-training-lora-opd-sopd.md) (bridges #93 → #97).

## TL;DR (verdict first)

- **What it is.** ROPD ([arXiv 2605.07396](https://arxiv.org/abs/2605.07396), May 2026) scores on-policy
  student rollouts against **auto-induced, prompt-specific rubrics** and trains on that score. It extends
  distillation to **non-verifiable / open-ended** tasks (agentic, writing, HealthBench, IFEval) where
  exact-match has no purchase. Reported strong: AIME24 24→65%, closes 74% of the student–teacher gap
  **text-only** (logits ignored), 9.6× fewer samples than logit-OPD.
- **The catch (load-bearing).** ROPD-as-published optimizes with **GRPO** — the rubric score becomes a
  scalar reward `s_i = (Σ w_k v_{i,k}) / (Σ w_k + ε)` fed to policy gradient. That collides head-on with
  ARLE's hard OPD-only / **never-GRPO** line ([2026-05-18 pivot](../projects/2026-05-18-opd-only-pivot.md)).
- **Recommendation (Path A, default).** Adopt the **rubric machinery** (Rubricator + Verifier + rubric
  scoring) but drive it with a **rejection / best-of-N distillation** update (RFT-style: select the
  rubric-best τ*, GKD `λ·CE(student‖τ*) + (1−λ)·KL(student‖EMA)`), **not** GRPO. This is the rubric
  generalization of [#93](https://github.com/cklxx/arle/issues/93)'s verifier and the **bridge that makes
  [#97](https://github.com/cklxx/arle/issues/97) (skills / agentic self-evolving) actually work** —
  agent tasks are not exact-match-verifiable. Stays inside the pivot.
- **Strategic fork for ckl (Path B).** The literal GRPO-ROPD results require **overturning** the
  2026-05-18 GRPO KILL (the re-do-a-killed-item rule: overturn its evidence first). Proceeding with Path A
  by default; say the word to reopen GRPO.

## ROPD mechanism (grounded — from the paper, not memory; it post-dates the Jan-2026 cutoff)

1. **Update rule = GRPO.** Verified rubric score `s_i = (Σ_k w_k v_{i,k}) / (Σ_k w_k + ε)`, `v_{i,k}∈{0,1}`
   per criterion, used as the **reward for on-policy (GRPO) optimization** (lr 1e-6, batch 32). The
   distillation signal is the rubric, but the optimizer is policy-gradient RL.
2. **Rubricator** — induces prompt-specific rubrics by **contrasting teacher answers vs student rollouts**:
   `C_x = Rubricator(x, Y^T_x, Y^S_x) = {(ρ_k, w_k)}`, each a textual criterion ρ_k + importance weight
   w_k. **m=4 teacher answers is critical** (−17.9 pts without it).
3. **Verifier** — binary per-criterion **blind scoring** `v_{i,k} = Verifier(x, y^S_i, c_k; Y^T_x, Y^S_x)`:
   the judge sees teacher and student responses but **not which is which** (prevents identity bias; uses
   teacher answers as difficulty anchors).
4. **Black-box by design** — needs only teacher **textual outputs**, not logits; the paper ignores logits
   even when available (black-box: GPT-5.2 via API; white-box test: Qwen3-30B, logits ignored).
5. **Teacher** — a **separate** model fills Rubricator+Verifier roles; replacing them with an auxiliary LLM
   has "marginal impact," so self-distillation is *possible* but not the paper's primary config.
6. **Reward-hacking** — <2% of rollouts game the rubric (high score, not substantively correct), mostly
   early (<1k steps), self-corrects; **no active regularization** — relies on multi-criterion design +
   explicit correctness checks in the verify prompt.
7. **Eval** — AIME24/25, HMMT25, GPQA-Diamond, HealthBench, IFEval (reasoning + open-ended).

## The GRPO problem, and why Path A is genuinely distillation (not hair-splitting)

| | GRPO (ROPD-as-published) | Rejection / best-of-N distillation (Path A = our A2/#93) |
|---|---|---|
| Per prompt | sample group of N, reward r_i | sample N, score r_i |
| Update | advantage `A_i=(r_i−μ)/σ` · ∇log π(y_i), **clipped ratio** | select rubric-best τ* (argmax / threshold), **CE→τ*** + KL→EMA |
| Losers | **pushed down** (negative gradient) | **dropped** (no negative gradient, no ratio, no clip) |
| Family | policy-gradient RL (verl/TRL territory) | RFT / rejection-sampling-FT — **distillation family** |

Best-of-N rejection is **already** in the SOPD plan as A2 ([#93](https://github.com/cklxx/arle/issues/93)); rubric
grading just replaces the exact-match verifier with a rubric Verifier. The **soft-weighted** variant
(weight CE by `f(s_i)`, i.e. reward-weighted regression) is the boundary case — default to the
**argmax/threshold** form to stay unambiguously inside the pivot; treat soft-weight as a flagged option,
not the bring-up.

## Where it fits the SOPD axes — A5 (rubric-graded signal)

Axis A gains **A5 = rubric-graded selection/weighting** alongside A1 EMA / A2 best-of-N+verifier / A3
self-consistency / A4 peer-teacher. Two teacher configurations, both supported by Path A:

- **Teacher-free (default, extends A1+A2).** EMA self-teacher is the judge; a **self-Rubricator** induces
  the rubric by contrasting EMA-teacher answers vs student rollouts (the m=4 "teacher answers" are EMA
  samples). No second model — keeps the SOPD memory budget.
- **Black-box external teacher (A4).** The one thing rubric-OPD uniquely unlocks: distill from a **frontier
  API teacher we have no logits for** (white-box KL-OPD is impossible there). Needs an external text-only
  teacher; time-share device via the existing `offload_engine_weights` (`infer-core/src/lib.rs`).

## Framework support — code-grounded (what's reuse, what's net-new)

| Component | Status | Anchor |
|---|---|---|
| Distillation update (GKD KL+SFT λ-blend; rubric-best τ* → SFT anchor) | **REUSE** | `opd.rs:1329` `mix_gkd_losses`, `:192` `GkdSftAnchor::StudentRollout` |
| `TrajectoryScorer` trait (exact-match + rubric impls — don't special-case) | **NET-NEW** | none today; only telemetry `reward` scalars (`control.rs:30`) — #93 defines the trait, #98 adds the rubric impl |
| Rubricator (rubric-induction judge-forward + structured parse) | **NET-NEW** | reuses the infer engine forward + `xgrammar-sys` (grammar-constrained decode) to force structured rubric output |
| Verifier grading (binary per-criterion, blind scoring) | **NET-NEW** prompt+parse | reuses the judge forward (student / EMA / external) |
| Black-box text-only teacher interface (no logits) | **NET-NEW** (external only) | `TeacherForward` is logit-based; teacher-free EMA has logits, so n/a there |
| Rubric data / prompt source | **NET-NEW** | with the Rubricator, "data" = prompts + a teacher source → connects to #97 skills |
| Seam / kernel / scheduler changes | **NONE** | grading is forward-only inference + host selection + the existing backward |

**Net**: concentrated in the `train` crate + an `xgrammar-sys` dependency for structured rubric output; the
infer engine is reused for judge forwards; the GKD update is unchanged. No seam, kernel, or scheduler change.

## Reward-hacking — sharper here than exact-match

A rubric judge is **soft** ⇒ more gameable than exact-match (which is hard to game). ROPD's own <2%/self-
correcting figure is for a strong external GPT-5.2 judge; a teacher-free EMA judge is weaker and more
gameable. Mitigations to adopt: (a) ROPD's **blind scoring** (judge doesn't know teacher vs student);
(b) **EMA-as-judge** (lagged weights are harder to game in-step than the live policy); (c) the [#93](https://github.com/cklxx/arle/issues/93)
**tripwire** — held-out rubric + held-out open-ended task + manual trajectory spot-check, revert if in-loop
score rises while held-out falls. Judge-policy separation tensions with teacher-free; the lag + held-out
rubric are the affordable substitute.

## Recommended cut → issue [#98](https://github.com/cklxx/arle/issues/98)

**A5 rubric-graded OPD**, Path-A (distillation, never GRPO): teacher-free EMA-judge + self-Rubricator first;
black-box external teacher (A4) later. Gated on #93's `TrajectoryScorer` trait; the open-ended bridge to #97.
- **PASS** — held-out lift over the base on **≥2 open-ended dims** (e.g. IFEval + one agentic skill task),
  multi-seed ≥5, mean±σ + Wilson 95% CI (2026-05-28 rule); reward-hack tripwire holds.
- **KILL** — no CI-separated lift after a sweep, or rubric-gaming can't be contained at acceptable cost ⇒
  rubric-graded teacher-free distillation doesn't work here; fall back to A1/A2 (verifiable-only).

## Sources

- [Rubric-based On-policy Distillation (ROPD), arXiv 2605.07396](https://arxiv.org/abs/2605.07396) — primary
- [OpenRubrics — scalable synthetic rubric generation, arXiv 2510.07743](https://arxiv.org/abs/2510.07743)
- [Step-wise Rubric Rewards for LLM Reasoning, arXiv 2605.17291](https://arxiv.org/html/2605.17291)
- [Reward and Guidance through Rubrics (RGR-GRPO), arXiv 2511.12344](https://arxiv.org/pdf/2511.12344)
- [Alternating RL for Rubric-Based Reward Modeling (non-verifiable), arXiv 2602.01511](https://arxiv.org/pdf/2602.01511)
- "Rubrics as Rewards (RaR)" — RL beyond verifiable domains (named in the survey above)
- [awesome-on-policy-distillation (curated OPD index)](https://github.com/chrisliu298/awesome-on-policy-distillation)
