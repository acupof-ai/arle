# SOP: Industry-Standard RL Post-Training Experiments (RLHF → RLVR → GRPO era)

> Status: Active. A general, industry-grounded SOP for running RL post-training
> experiments the way leading labs and open frameworks actually run them
> (2024–2026). Sections 1–4 are the industry spine; §5 is the 二八 distillation;
> §6 is a thin ARLE adapter. Drives the agent-OPD RL lane (base vs ThinkingCap).

## Verdict

The field has converged on a **critic-free, group-relative, verifiable-reward
loop** (GRPO/DAPO on RLVR), run as **iterative synchronous rollout→score→update
rounds** with `vLLM`/`SGLang` for generation and `FSDP`/`Megatron` for the update,
orchestrated by **verl / TRL / OpenRLHF / open-instruct**
([verl](https://github.com/verl-project/verl),
[Tulu-3](https://arxiv.org/abs/2411.15124)). The methodology consensus from
DeepSeek-R1, Tulu-3, DAPO, and Kimi-k1.5 is narrow and load-bearing: **rule-based/
verifiable rewards over neural RMs (anti-reward-hacking), held-out + decontaminated
eval, KL/entropy monitoring, and multi-seed variance reporting**. The single
biggest methodological correction of 2025 — *"A Sober Look at Progress in LM
Reasoning"* — is that **single-seed pass@1 on small evals is statistically fragile
(σ 5–15pp); most reported RL gains fall within seed noise**
([arXiv 2504.07086](https://arxiv.org/abs/2504.07086)). So the non-negotiables are:
verifiable reward, held-out+decontaminated eval, ≥ multi-seed mean±σ before any
capability claim, and one-variable ablation.

---

## 1. The industry-standard RL post-training experiment lifecycle

How real teams sequence and gate it (each stage gates the next — you don't proceed
on a broken upstream):

1. **Hypothesis** — a falsifiable claim tied to a *specific* benchmark delta
   ("RLVR on math lifts held-out AIME/MATH pass@1 by ≥X, no regression on general
   evals"). No hypothesis → no run.
2. **Dataset / environment design** — assemble a **verifiable** prompt set
   (answer-checkable math, executable code, constraint-checkable IF). Split
   train / **held-out dev** / **unseen test** up front. Curate for *difficulty
   spread* — prompts the base model solves 0% or 100% of the time produce
   zero-variance groups and no gradient (DAPO's dynamic sampling and Tulu-3's
   curation both target this).
3. **Baseline selection** — the SFT/base checkpoint the RL run starts from,
   evaluated on the *same* harness (same decoding params, same prompt template).
   Every later number is a delta against this. Leading recipes are staged:
   **SFT → DPO → RLVR** (Tulu-3), or **SFT cold-start → RL** (R1); "RL-Zero"
   (R1-Zero, RL directly on base) is the ablation, not the default.
4. **Rollout infrastructure** — a fast inference engine (`vLLM`/`SGLang`) generates
   `G` samples/prompt at temp>0 (typ. 0.6–1.0, top-p 0.95–1.0); the trainer
   (`FSDP`/`Megatron`) does the update. **Weight sync** cadence between them defines
   on-policy-ness. Kimi-k1.5's **partial rollouts** (reuse prior trajectories, cap
   regeneration) is the standard cost lever for long-CoT
   ([arXiv 2501.12599](https://arxiv.org/abs/2501.12599)).
5. **Reward design & verification** — **rule-based/verifiable** wherever possible.
   R1 uses accuracy (answer-match / code-exec) + format reward, and **deliberately
   avoids neural RMs to prevent reward hacking at scale**
   ([R1 review](https://arxiv.org/pdf/2503.11486)). Kimi adds a **length penalty**
   to curb length-hacking. This is the highest-leverage design decision — a
   gameable reward invalidates everything downstream.
6. **Training loop** — group-relative advantage `A = (r − mean)/std`, clipped
   policy gradient, optional KL-to-ref penalty. GRPO drops the critic; DAPO drops
   the KL penalty and adds asymmetric ("clip-higher") clipping + token-level loss +
   dynamic sampling
   ([DAPO/RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/reference/algorithms/dapo.html)).
7. **Evaluation** — held-out **dev** during training (checkpoint selection),
   **unseen test** once at the end. Fixed decoding params, fixed template,
   **decontaminated** against train (Tulu-3 open-sources its decontamination
   tooling). Report pass@1 as **mean over seeds**, not peak.
8. **Analysis / attribution** — when a number moves, decode actual generations;
   separate capability gain from harness/format/eval artifacts. When it regresses,
   treat it as a case to debug, not a structural conclusion.
9. **Ablation** — one variable at a time (reward component, algorithm, KL coeff,
   data mix, sampling temp), each vs the same baseline.
10. **Scaling / ship decision** — only a config that cleared held-out with
    multi-seed CI and no reward-hacking signature scales up (more data / larger
    model / more steps) or ships.

---

## 2. How leading open frameworks operationalize it

- **verl (ByteDance, EuroSys'25)** — models RL as a multi-stage dataflow graph; a
  GRPO/PPO/DAPO run is a YAML `ppo_trainer` config: `actor_rollout_ref`
  (policy+rollout+ref), `reward_model`, `algorithm.adv_estimator={gae,grpo,...}`,
  `algorithm.kl_ctrl`. Rollout = `vLLM`/`SGLang`; train = `FSDP`/`Megatron`. Knobs
  teams sweep: `rollout.n` (samples/prompt), `rollout.temperature`,
  `data.train_batch_size`, `actor.ppo_mini_batch_size`, `actor.clip_ratio` (or
  `clip_ratio_low/high` for DAPO), `kl_ctrl.kl_coef`. Supports PPO, GRPO, GSPO,
  DAPO, Dr.GRPO, RLOO, REINFORCE++, PRIME
  ([verl](https://github.com/verl-project/verl),
  [Qwen×verl](https://qwen.readthedocs.io/en/latest/training/verl.html)).
- **TRL (HuggingFace)** — `GRPOTrainer` + `GRPOConfig`: `num_generations` (G),
  `max_completion_length`, `beta` (KL coeff), `num_iterations`,
  `report_to="wandb"`. Logs `reward`, `reward_std`, `kl`, `completion_length`,
  `clip_ratio`, `loss`, and optional `log_completions`
  ([TRL logging](https://huggingface.co/docs/trl/en/logging),
  [grpo_config.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)).
- **OpenRLHF (Ray-based)** — Ray-orchestrated, async RL, PPO/DAPO/REINFORCE++/vLLM;
  the "easy scalable" reference for distributed rollout↔train
  ([OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)).
- **NeMo-Aligner (NVIDIA)** — Megatron-scale PPO/DPO/SteerLM; the path for very
  large models.
- **AllenAI open-instruct (Tulu-3)** — the fully-open **SFT → DPO → RLVR**
  reference recipe, *with* its eval suite (dev/unseen split) and **decontamination
  tools** open-sourced — the reference for eval hygiene, not just training
  ([open-instruct](https://github.com/allenai/open-instruct/blob/main/README.md),
  [Tulu-3 blog](https://allenai.org/blog/tulu-3-technical)).

Common structure across all: a **config file** (algorithm + data + rollout +
reward + KL), an **experiment tracker** (wandb/tensorboard/trackio), and a
**standard metric panel** (reward mean/std, KL, completion length, clip fraction,
entropy).

---

## 3. What the major papers report as methodology

- **DeepSeek-R1 / R1-Zero (GRPO)** — critic-free GRPO; **rule-based rewards only**
  (accuracy via answer-match/code-exec + `<think>` format reward); neural RMs
  avoided *specifically to prevent reward hacking*; KL term inside the loss (not
  the reward). R1-Zero (RL on base, no SFT) is the ablation showing emergent
  reasoning ([review](https://arxiv.org/pdf/2503.11486),
  [phil schmid](https://www.philschmid.de/deepseek-r1)).
- **Tulu-3 RLVR (AllenAI)** — reward = scoring function returning positive reward
  iff answer verifiably correct (math, precise IF); staged SFT→DPO→RLVR;
  **multi-task eval with explicit dev (seen) vs unseen split + substantial
  decontamination**, all open-sourced
  ([arXiv 2411.15124](https://arxiv.org/abs/2411.15124)).
- **DAPO (ByteDance/Tsinghua)** — four deltas over GRPO: **clip-higher**
  (asymmetric ε_low/ε_high, fights entropy collapse), **dynamic sampling** (drop
  all-correct/all-wrong groups → guarantees non-zero gradient), **token-level
  loss** (long-CoT stability), **remove KL penalty**. Fully open recipe+data
  ([DAPO](https://rlinf.readthedocs.io/en/latest/rst_source/reference/algorithms/dapo.html)).
- **Kimi-k1.5** — iterative synchronous rollout↔train; 128K context; **partial
  rollouts** (trajectory reuse for cost); **length penalty** for length-hacking;
  deliberately *no* MCTS / value function / PRM
  ([arXiv 2501.12599](https://arxiv.org/abs/2501.12599)).
- **Reward-hacking / KL / contamination literature** — R1's "avoid neural RMs" is
  the design-time defense; runtime detectors reach ~78% precision / ~82% recall on
  RLVR via statistical+behavioral ensembles; entropy/policy collapse is *stronger
  on explicitly-rewarded samples*
  ([reward-hacking in RLVR](https://arxiv.org/pdf/2606.04923),
  [emergentmind](https://www.emergentmind.com/topics/reward-hacking-in-rlvr)). RL
  post-training can itself **contaminate** eval
  ([arXiv 2510.09259](https://arxiv.org/pdf/2510.09259)). The
  **statistical-fragility** finding: pass@1 σ 5–15pp across 20 seeds on AIME'24
  (n=30) / AMC'23 (n=40); many RL gains are within noise; report **stable, not
  peak**, performance ([arXiv 2504.07086](https://arxiv.org/abs/2504.07086),
  [MarkTechPost](https://www.marktechpost.com/2025/04/15/llm-reasoning-benchmarks-are-statistically-fragile-new-study-shows-reinforcement-learning-rl-gains-often-fall-within-random-variance/)).

---

## 4. Experiment-management best practice — non-negotiable vs optional

**Non-negotiable (the field treats these as table stakes):**
- **Verifiable / rule-based reward** where the task allows; if using a neural RM,
  expect and monitor hacking (R1's explicit rationale).
- **Held-out dev + unseen test**, both **decontaminated** against train and against
  pretraining benchmarks (Tulu-3).
- **Multi-seed variance reporting** — mean±σ (and CI) over ≥ several seeds; report
  *stable*, not best-of-N-checkpoints (the checkpoint-selection bias). Single-seed
  small-eval deltas are not evidence
  ([2504.07086](https://arxiv.org/abs/2504.07086)).
- **Fixed, disclosed eval config** — decoding params, prompt template,
  framework/hardware. Subtle changes here explain most spurious gains.
- **KL-to-ref + entropy monitoring** — KL bounds drift; entropy collapse = mode/
  diversity collapse (DAPO clip-higher exists precisely for this). Watch KL
  climbing past ~0.1; early-stop on runaway.
- **Reward-hacking detection** — reward↑ while held-out flat/down; response length
  blow-up; format/verifier exploits. Decode generations to confirm.
- **One-variable ablation** — the only way a delta is attributable.
- **An experiment tracker** — wandb/tensorboard/trackio with the standard panel
  (reward mean/std, KL, completion length, clip frac, entropy, grad norm, LR,
  pass@k train+held-out).

**Optional / context-dependent:** critic (GRPO/DAPO drop it), KL penalty at all
(DAPO removes it, R1 keeps it), partial rollouts (cost optimization for long-CoT),
advanced hacking detectors (Mahalanobis/MOP, gradient fingerprints — nice, not
required), process reward models / MCTS (Kimi & R1 both reject them as unnecessary
complexity).

---

## 5. 二八 (80/20) distillation

**The ~20% that carries ~80% of the decision value:**

1. **One number decides accept/kill: held-out (decontaminated) pass@1 delta vs
   baseline, as a multi-seed mean±σ.** Everything else is diagnostic. A gain inside
   seed-σ is *not* a gain.
2. **Two reward-hacking tripwires, both free: response length and the
   reward↔held-out divergence.** If reward climbs while held-out is flat, or length
   explodes — kill, regardless of reward. (R1/Kimi/length-penalty lineage.)
3. **One liveness check: group reward variance (zero-variance fraction).** If most
   groups are all-pass or all-fail, there is no gradient — the run is dead before
   it starts (DAPO dynamic sampling). Fix data difficulty first.

**Metrics that are diagnose-only (don't gate):** grad norm, entropy, clip fraction,
IS-ratio, throughput. Watch them to explain *why*, not to decide.

**Gates you can skip early:** full ablation grid, multi-seed CI (needed to
*publish*, not to *direction-check*), scaling, advanced hacking detectors.

**Gates never skippable:** verifiable reward; held-out set with enforced no-overlap
+ decontamination; a baseline evaluated on the identical harness;
one-variable-at-a-time.

**Leanest experiment that can still KILL a bad idea:** baseline eval → **1 short
training run (a few hundred updates / 1–3 rounds)** → 1 held-out eval. If held-out
drops, length explodes, groups collapse to zero-variance, or reward↔held-out
diverge — kill it there, before spending on seeds/scale. One seed is enough to see
*direction*; it is **not** enough to claim a gain.

**Minimum the field considers trustworthy for a *claim* (not just a direction):**
- **Seeds:** ≥ 3–5 (the sober-look study used up to 20 on tiny sets); always
  mean±σ, ideally CI.
- **Eval size:** avoid n≈30–40 sets as a sole signal — there each 1-question flip
  is ~2.5–3.3pp; prefer larger held-out sets or aggregate multiple.
- **Rounds/steps:** enough that the held-out curve is *stable across ≥2 consecutive
  checkpoints*, and report the pre-committed checkpoint, not the peak.
- **A control arm:** same-everything except the one variable.

---

## 6. Appendix — thin adapter onto ARLE (`arle train agent-opd`)

Our stack instantiates the §1 loop; the industry gates map directly:

| Industry stage | ARLE mechanism |
|---|---|
| Verifiable reward | cc-harness pass/partial on hidden tests (rule-based, execution — the R1-style anti-hacking choice; good) |
| Held-out + no-overlap | `--eval-dataset` with **enforced** overlap-bail (`train_cli.rs:2905`); round-0 baseline before training (`:3260`). *Add decontamination vs pretraining, not just train-overlap.* |
| Rollout engine ↔ trainer | in-process serve (rollout) + autograd student (update); `--sync every-group` = faithful `π_behavior` |
| Group-relative update | `--update-strategy {rejection-ce, grpo, dapo, dr-grpo, gspo, cispo, sao-*}`; `--samples-per-prompt` = G |
| Dynamic sampling (DAPO) | `--task-selection` (zero-variance skip + retire) is our analog |
| Standard metric panel | `metrics.jsonl` already logs `held_out_pass_rate/baseline/delta`, `pass_at_k`, `zero_variance_group_frac`, `reward_mean`, `completion_tokens`, `kl_rollout`, `is_ratio_mean/max`, `clip_frac`, `mean_train_loss` — the panel is there; wire it to a tracker for run-over-run diffing |

**The 二八 three, in ARLE fields:** ① `delta` (held-out − baseline) = accept/kill;
② `completion_tokens` trend = length-hack tripwire; ③ `zero_variance_group_frac` =
liveness.

**Phasing for the current experiment (base vs ThinkingCap-Qwen3.6-27B-FP8,
sweetspot3 29-train / 33-held-out):**

- **Phase 1 — `rejection-ce` greedy** (`--rollout-temperature 0.0`, the running
  config): this is *SFT-on-wins / rejection sampling*, **not** on-policy RL. A
  positive `delta` is a capability lift, not evidence GRPO works. Cheapest kill
  available; run it, one variable = student model.
- **Phase 2 — `grpo` on-policy: gated on #59.** GRPO needs temp>0 for group
  variance. temp>0 sampling on hd256/FP8 was corrupt (#59), root-caused by bisect
  to kernel `b4b293f0c` (OFFSET→STANDARD q/k RMSNorm, which broke both greedy-at-
  length and temp>0). The OFFSET-restore fix is **already in-tree** (current HEAD
  has `(1+w)` at all 4 sites); a confirm probe (decode ~50 completions at temp 0.7:
  no garble/looping, sane entropy, non-empty behavior-logprobs) is the gate. Once
  the probe is green, flip `--update-strategy grpo --rollout-temperature 0.7
  --samples-per-prompt 4 --sync every-group`. Do **not** run GRPO before the probe
  — a garbage `π_behavior` makes the IS ratio meaningless and the whole arm
  unattributable (exactly the §1.5/§4 confounder).
- **Phase 3 — ablations/scale** on a Phase-1/2 survivor: LoRA surface, layers,
  rounds; then **≥5-seed mean±σ + Wilson CI** before any capability claim <5pp
  (matches both the ARLE rule and the sober-look finding), and multi-shape (≥2 task
  sets) before any default flip.

**Project-rule overlay:** one-variable A/B; correct-inference gate
(`needle_gate.py` ×3) on the serve config before trusting rollouts; every runtime
change → a dated `wins/`(or `errors/`) bench entry; held-out before any default
flip; decode actual generations when a metric looks catastrophic.

---

**Sources:** [verl](https://github.com/verl-project/verl) ·
[Qwen×verl](https://qwen.readthedocs.io/en/latest/training/verl.html) ·
[TRL logging](https://huggingface.co/docs/trl/en/logging) /
[grpo_config](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py) ·
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) ·
[open-instruct](https://github.com/allenai/open-instruct/blob/main/README.md) ·
[Tulu-3 arXiv 2411.15124](https://arxiv.org/abs/2411.15124) /
[blog](https://allenai.org/blog/tulu-3-technical) ·
[DeepSeek-R1 review](https://arxiv.org/pdf/2503.11486) /
[phil schmid](https://www.philschmid.de/deepseek-r1) ·
[DAPO](https://rlinf.readthedocs.io/en/latest/rst_source/reference/algorithms/dapo.html) ·
[Kimi k1.5 arXiv 2501.12599](https://arxiv.org/abs/2501.12599) ·
[Sober Look arXiv 2504.07086](https://arxiv.org/abs/2504.07086) ·
[Reward hacking in RLVR arXiv 2606.04923](https://arxiv.org/pdf/2606.04923) ·
[RL post-train contamination arXiv 2510.09259](https://arxiv.org/pdf/2510.09259)
