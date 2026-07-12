# SAO Phase 2 — value critic + Skip-Obs GAE + per-token PG

> Status: pending-remote — code landed + pod-built + 3-arm A/B running (H20).
> Verdict fills in from `/host/opd-sweetspot3/{rceF,saoF,valF}-out/`.

## Context

SAO Phase 1 (`sao-dis`) is the weaker half of the algorithm: one batch-centered
scalar advantage spread over a long agentic trajectory = weak credit assignment,
and **zero** learning signal on all-pass/all-fail batches (advantage centers to
0). Phase 2 adds the missing piece — a learned value critic giving a per-token
Skip-Obs GAE advantage, so failing trajectories gain a *negative* signal and
credit is assigned per token, not per trajectory.

Commits: `4568ec005` (main) · `8e1d2f2b7` (MSE-graph leak fix) · `a8be61f35`
(critic telemetry).

## What landed

- **PG op** `fused_linear_pg_loss_indexed`: `advantage: f32 → &[f32]` (per-token);
  host + device path, reduces-to-CE unit-weight self-check still passes.
- **`ValueCritic`** (`opd.rs`): linear head `V(s)=hidden·wᵀ`, own AdamW + LR,
  zero-init (V₀=0 → round-0 GAE = discounted reward-to-go = graceful cold start,
  no separate value-pretraining). **Frozen-Attention** via a detached masked-hidden
  host round-trip → grad reaches `weight` only, never the base. **One** 27B
  forward/traj feeds both the GAE values (host dot) and the MSE update.
- **`skip_obs_gae`**: values indexed over masked (LLM) positions only → tool/obs
  tokens skipped for free; terminal reward on the last generated token.
- **`UpdateStrategy::SaoValue`**: per-traj GAE → per-token DIS PG (policy) + MSE
  step (critic); no batch-mean centering (the critic is the baseline).
- **CLI**: `--update-strategy sao-value`, `--sao-gamma` (1.0), `--sao-lambda`
  (0.95), `--value-lr` (1e-3). Critic built after the trainable filter → policy
  optimizer / LoRA sync / adapter save never touch it.

Local verify: autograd + train green, `pg_reduces_to_ce` + `skip_obs_gae` tests
pass, clippy clean. Pod: `--features cuda` BUILD_EXIT=0, `sao-value` wired.

## Experiment (running)

Matched full-scale 3-arm A/B on the sweet-spot corpus `/host/opd-sweetspot3`
(29 train / 33 held-out), Qwen3.6-27B-FP8 student, `--rounds 8 --eval-every 2
--samples-per-prompt 4 --max-turns 8 --max-tokens 768`, LoRA r16/α32 attention-qv,
lr 1e-5. Only scale + strategy differ from the trimmed rceS3/saoS3 pair.

| arm | strategy | GPU | out |
|-----|----------|-----|-----|
| rceF | rejection-ce | 0 | rceF-out |
| saoF | sao-dis (Phase 1) | 1 | saoF-out |
| valF | sao-value (Phase 2) | 2 | valF-out |

Base held-out (trimmed pair, same corpus): rejection-ce 0.0303 / sao-dis 0.1212,
mean_dense ≈ 0.49 (clean A/B start). Watch: `bash ~/watch-sao.sh` — valF's
`mean_critic_mse` ↓ and `mean_adv_abs` > 0 are the load-bearing Phase-2 signals.

## Rule

_(pending — fill from decoded per-round eval + critic telemetry: did the critic
learn (MSE ↓), did per-token GAE beat Phase 1's scalar advantage, and did either
SAO arm clear the rejection-CE incumbent on held-out pass_rate AND mean_dense?)_
