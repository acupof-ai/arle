# OPD ceiling + elicitation on a 27B dense student — rubric-OPD mainline

**Goal (ckl, 2026-06-21):** verify the *ceiling* of On-Policy Distillation and
whether it can *elicit* (激发) latent capability in a 27B dense student. Headline:
**DeepSeek-V4-Flash as a rubric JUDGE → Qwen3.6-27B-FP8 dense student (LoRA)** on 8×H20.
Dense 27B chosen to dodge the MoE-router autograd problems of the 35B-A3B student. Hard
constraint: **complete plan + code first, GPU only after.**

**Route decision (ckl delegated the choice, 2026-06-21):** **Rubric-OPD**, not token-KL.
- *Same-vocab token-KL OPD is already validated* — 4B math 0.518→0.792 (5-seed) and
  agentic abstention 0.60→1.00. Re-running it proves nothing new (ckl: "同词表咱们已经验证了").
- *Cross-vocab token-KL (DSv4→Qwen)* is a double risk — the loss hard-asserts vocab
  equality (`loss.rs:451`), and cross-tokenizer KD is "a fundamentally open problem,
  no method consistently wins" (ULD/BLD/MultiLevelOT survey, §1).
- **Rubric-OPD sidesteps both**: DSv4-Flash judges the student's rollouts at the *text*
  level (vocab-agnostic), using Flash's genuine strength (frontier reasoning/judgment),
  on the proven RFT/RaR substrate. ckl: "flash可以验证rubric opd模式…写判断方法然后按照规则选择 回写".
- *Skills-SOPD* (the alternative ckl offered) is **deferred** — less grounded, higher
  infra (skills/tools wired into the rollout), and self-distillation's ceiling is bounded
  without a strong external judge. Revisit after rubric-OPD lands.

---

## 1. Industry practice (surveyed 2026-06-21)

**On-policy distillation (the elicitation mechanism):**
- OPD = dense KL-constrained RL; teacher per-token log-ratio is an implicit reward;
  scaling it >1× can push the student *past* the teacher (Rethinking-OPD 2604.13016).
- **Thinking-pattern alignment dominates teacher strength** — a 75% and a 50% teacher
  had ~identical effect when reasoning patterns mismatched; mismatch weakens distillation
  *regardless of benchmark advantage*. RL-post-trained teachers transfer "new knowledge
  beyond family" (DSv4-Flash qualifies). → a cross-family judge must bridge patterns; the
  rubric does this by selecting the *student's own* on-pattern rollouts.
- OPD is 7–10× fewer steps / 50–100× less compute than RL (Thinking-Machines 2025).

**Rubric / rejection-sampling distillation (the chosen substrate):**
- **RFT/RAFT** (sample N → filter by quality → SFT on accepted): "RFT consistently > SFT,
  improvement ~**log-linear in the number of distinct accepted CoTs** per example." Numbers:
  SFT 46.3% → RFT 66.8% → RL 71.5% — RFT captures most of the RL gain at SFT cost.
- **STaR / AdaSTaR**: iterative train on self-generated CoT; adaptive sampling −58.6% FLOPs.
- **Rubrics as Rewards (RaR, 2507.17746)**: strong-LLM checklist rubrics extend reward
  beyond verifiable domains. **Step-wise rubric rewards (2605.17291)**: step-attributed
  judging avoids penalizing correct steps inside wrong trajectories.
- **Reward-hacking risk** (over-producing surface behaviors): mitigate with **Factual
  criteria** (verify intermediate correctness) + **Process criteria** (valid reasoning
  steps), not a single scalar.

**Implication:** the rubric-OPD curve (accuracy vs # accepted CoTs / RFT rounds) is the
"上限" measurement; a much-stronger judge (DSv4-Flash) raises the acceptance quality bar,
eliciting the student's latent best-of-N behavior — the "激发" axis.

---

## 2. Mechanism — rubric-OPD loop

```
prompt ─► student (Qwen3.6-27B) samples N on-policy rollouts (temp>0)
       ─► DSv4-Flash judges each rollout vs the rubric (Factual + Process criteria)
       ─► select accepted (pass / top-score) rollouts                 [rejection sampling]
       ─► WRITEBACK: student CE/SFT on its OWN accepted rollouts       [回写, on-policy]
       ─► iterate (RAFT/STaR rounds) ─► capability curve vs #accepted-CoTs
```

On-policy because the training targets are the *student's own* generations, filtered by a
strong external judge. Elicits latent capability (the student already produces good
rollouts *sometimes*; the rubric reinforces them) without any cross-vocab logit machinery.

---

## 3. Codebase reuse + infra gaps (Explore map, 2026-06-21)

**Reusable as-is:**
- Student arch: `qwen35.rs` (Qwen3.5/3.6). **Qwen3.6-27B dense loads with zero new code**
  — `qwen35_loader.rs:675-690` keys on `num_experts==0` → no MoE layers.
- On-policy rollout: `infer_student.rs` / `opd.rs:846 forward_rollout_cached`.
- Flash teacher via `infer-api` generic path (`train_cli.rs:1803-1836` → `InferTeacher`);
  for rubric mode we use it as a **text judge**, not a logit source → the vocab-match check
  (`teacher_infer.rs:740-769`) is bypassed (no token-KL).
- CE-on-own-rollout (SOPD `StudentRollout` CE anchor) — the writeback trainer already
  exists (see `reference_sopd_coldstart_loss_near_zero_greedy_rollout`).

**Net-new infra (the "修复所有基础设施问题" scope):**
- I1. **Multi-sample rollout**: N rollouts/prompt at temp>0 (`--rollout-samples N`,
  `--rollout-temperature`); today OPD does 1 rollout/prompt.
- I2. **Rubric-judge step** (`train/src/rubric.rs`): render a judge prompt (Factual +
  Process criteria) → DSv4-Flash via infer-api → parse a structured verdict (pass/score
  per criterion). Robust parse (JSON-ish), timeout-clean (per §0 case-as-fact: never bucket
  a timeout as a pass/fail).
- I3. **Selection**: accept by rubric threshold / top-k; log acceptance rate + #distinct
  accepted CoTs/prompt (the RFT log-linear x-axis).
- I4. **Writeback trainer**: CE/SFT on accepted rollouts (`--kl-mask completion` semantics:
  loss on completion tokens only). Reuse the SOPD CE anchor; LoRA r32/a64, `--lora-layer-start`
  available if the 27B backward is the wall (measured 8.1× at seq-2048).
- I5. **RFT loop driver**: rounds of generate→judge→select→train; checkpoint + eval each round.
- I6. **Rubric library**: math (Factual=answer correctness, Process=valid steps) + agentic
  (BFCL: Factual=correct tool/abstain, Process=reasoning-then-act).

---

## 4. Recipe (research-grounded defaults)

- N samples/prompt: start 4 (RaR/RFT), sweep to 8–16 (log-linear gain).
- Rollout temp: 1.0 (diversity for rejection sampling); judge at temp 0.
- Rubric: Factual + Process criteria, step-attributed; **avoid single-scalar** (reward-hack).
- Writeback: CE on accepted completions; 1 epoch/round; ~3–5 RFT rounds for the curve.
- Eval: clean, timeout-free, multi-seed (§0 case-as-fact); math (MATH-500) + agentic (BFCL).
- Baseline = the 27B base best-of-1; ceiling probe = base best-of-N vs OPD best-of-1
  (does OPD internalize the best-of-N gain → "激发").

---

## 5. DAG, critical path, GPU budget

```
download 27B ─ load smoke ─┐
rubric lib (I6) ───────────┼─ I1 multi-sample ─ I2 judge ─ I3 select ─ I4 writeback ─ I5 loop ─► curve
Flash teacher smoke ───────┘
```
Critical path = I1→I2→I3→I4→I5 (the loop); I6 (rubrics) + downloads parallelize.

**GPU (8×H20, 96 GB):** DSv4-Flash judge TP4 (4 GPUs, ckl-confirmed it fits TP4) +
Qwen3.6-27B-FP8 student rollout/train (1–2 GPUs; 35B-A3B used ~75 GB single-GPU at
seq-2048). Comfortable within 8. Dev loop: `scripts/pod_pipeline.sh` (sync→build→verify).
Build env proven in `pod_pipeline.sh`; profile via `ARLE_OPD_STEP_PROFILE=1` (no `--json`).

---

## 6. Status

Plan + route decided (rubric-OPD). Next: implement I1–I6 (no GPU), then validate per §4–5.
Same-vocab token-KL is already validated and is NOT re-run. Skills-SOPD deferred.
