# RL comfort zone: keeping tasks in the difficulty band where the signal lives

> Status: Active — research brief for agent-OPD / SAO curriculum

## Verdict

The learning signal in verifiable-reward RL is the **within-group reward
variance**, not the reward level. For binary pass/fail it is `p(1−p)` where `p` =
per-task pass probability: **zero at p=0 (all fail) and p=1 (all pass), maximal at
p≈0.5.** The "comfort zone" is the task band that keeps `p` off the two dead ends.
Keep it with three levers, cheapest first: **dynamic sampling** (drop uniform
groups), **banded curriculum** (filter by online pass-rate, and re-band as the
policy improves), **dense reward** (partial credit to manufacture variance where
binary gives none).

## First principle — signal = variance, not level

GRPO/SAO advantage is `A_i = r_i − mean(r)`. If every sample in a prompt's group
gets the same reward, every `A_i = 0` → zero gradient. So a prompt only teaches
when its group has **mixed outcomes**. Binary reward variance = `p(1−p)`:

| per-task p (pass prob) | group outcome | advantage | teaches? |
|---|---|---|---|
| 1.0 (too easy) | all pass | 0 | no |
| 0.0 (too hard) | all fail | 0 | no |
| 0.2–0.8 | mixed | ≠0 | **yes** |

We measured both dead ends this week: easy synthetic → base pass_rate **1.0**
(saturated, no headroom); real swesmith @ 1 sample → pass_rate **~0** (0-accept).
Neither produces gradient. The passive symptom-patch already landed (skip
zero-advantage batches, codex P2); the comfort zone is the cure.

## Sharp correction for our setup — SAO does NOT rescue the p=0 tail

A tempting hope: "SAO learns from failures, so it survives the hard 0-accept
corpus where rejection-sampling starves." **False for binary reward.** At p=0 all
rewards are 0 → centered advantage 0 → SAO gradient 0, same dead end as
rejection-ce. The two methods differ only in the *middle*:
- **rejection-ce** (SFT on passes) needs **p>0** — one pass to imitate.
- **SAO/GRPO** (centered advantage) needs **0<p<1** — variance for a baseline.

What actually extends the low end is **dense reward** (fraction of tests passing →
non-zero variance even when no run fully passes) or a **value critic** (SAO
Phase 2: per-token advantage from V(s), signal without group variance). Binary-
reward SAO alone does not.

## The lever toolbox (ranked by ROI for agent-OPD)

We already roll out N samples/prompt and pytest-score each → **the per-task pass
count is in hand for free.** That makes the top two levers nearly zero-cost.

1. **Dynamic sampling (DAPO)** — active version of our P2 skip: when a prompt's
   group is all-pass or all-fail, discard it and resample another prompt until the
   training batch is full of mixed-outcome groups. Keeps every gradient step on
   useful data. `agent_opd.rs` already loops samples per task; add: after scoring,
   drop uniform groups and top up from the pool. (arXiv:2503.14476 DAPO;
   openreview kiXFIESZKv on exploiting even zero-variance prompts — a nuance, not
   a default.)
2. **Banded online curriculum** — track a per-task pass-rate EMA from the rollouts
   we already score; each round keep tasks with EMA in ~[0.2, 0.8], retire
   mastered (>0.8) and shelve too-hard (<0.2) for later. The band is the ZPD
   (Vygotsky); "online difficulty filtering" sets the proxy = current policy
   capability (arXiv:2504.03380, 2505.08364). **The band MOVES**: a p=0.5 task
   becomes p=0.9 after learning, so re-estimate + re-band, don't freeze a static
   tier.
3. **Dense / partial-credit reward** — replace binary pytest exit-0 with **fraction
   of `fail_to_pass` tests now passing** (+ keep `pass_to_pass` as a regression
   guard). Manufactures variance on hard tasks that never fully pass → rescues the
   p→0 tail that levers 1–2 can only shelve. Cheapest way to widen the zone
   downward without a critic.
4. **Sampling temperature / N** — higher temp + more samples raise the chance of
   catching a mixed outcome on borderline tasks (moves p off 0/1); a knob, not a
   curriculum. Costs rollout compute.
5. **Value baseline (SAO Phase 2)** — a learned V(s) gives per-token advantage even
   at constant group reward, the principled extension below the variance floor.
   Largest lift, largest cost (critic + its cold-start). Deferred.

## Recommendation for agent-OPD / SAO

Order matches ROI and unblocks the stalled experiment:
1. **Dense reward first** (partial-credit score_workdir) — turns both dead corpora
   usable: easy tasks still saturate but hard tasks stop being 0-signal.
2. **Dynamic sampling** — drop uniform groups, resample; near-free given we already
   score N samples.
3. **Banded curriculum with online re-estimation** — the corpus becomes a pool;
   each round trains the middle band and re-bands as capability rises.
4. Revisit a **value critic** only after 1–3, as SAO Phase 2.

This directly resolves the easy(1.0)/hard(0.0) dead-zone that stalled the
SAO-vs-baseline run: a banded, dense-reward pool keeps `p` in the teaching range
for both arms, and dynamic sampling spends every step there.

## References

- DAPO — dynamic sampling drops uniform-reward groups (arXiv:2503.14476).
- Online Difficulty Filtering for Reasoning-Oriented RL (arXiv:2504.03380).
- Adaptive Difficulty Curriculum Learning (arXiv:2505.08364).
- Exploiting zero-variance prompts in LLM RL (openreview kiXFIESZKv).
- ZPD: Vygotsky 1978 — tasks in the proximal zone maximize learning progress.
