# Agent RFT uses generation-time behavior probabilities — 2026-07-26

> Status: accepted — denominator identity, offline replay, and a real online
> stochastic ratio-weighted update are verified on H20.

## Context

CUDA serving already records the probability of each committed sampled token
after temperature and top-k/top-p/min-p filtering. Training nevertheless rebuilt
the denominator from raw train-model logits for fresh online and offline replay
updates. That reconstruction is not the behavior distribution when sampling
transforms are active, even with synchronized weights.

Old estimator input:

```text
r = exp(log pi_current - log pi_recomputed_after_generation)
```

Required estimator input:

```text
r = exp(log pi_current - log pi_behavior_at_generation)
```

## What Worked

- `ScoredTrajectory` has one denominator field: generation-time
  `behavior_logprobs`.
- Fresh online, stale online, experience replay, and offline replay preserve the
  same immutable sidecar. Staleness changes policy distance, not denominator
  provenance.
- One train-owned preflight validates reward, response/mask shape, binary masks,
  sequence/index bounds, target vocabulary, writeback window, and—only for
  ratio-weighted survivors—sidecar existence, alignment, and finiteness before
  model work.
- CE/GKD remain ratio-free and accept legacy records without a sidecar.
- Ratio-weighted online rollout rejects greedy temperature before sandbox,
  model, store, or serving initialization.
- `capture_rollout_logprobs` remains only where GSPO needs the current-policy
  sequence numerator. The default denominator recompute and `ratio_floor_*`
  diagnostic forward were deleted.

## Verification

Local:

- `cargo test -p train --release update_strategy`: 10 passed.
- CUDA/no-CUDA CLI replay/temperature tests: 5 passed.
- CPU CLI and CUDA/no-CUDA `infer-api` checks passed.
- `cargo fmt --check` and `git diff --check` passed.
- Codex review found no defect in the scoped Rust changes; its findings were only
  for an unrelated untracked benchmark script.

H20, isolated source `/host/arle-denom-verify`:

- CUDA build: `denom-verify-build5`, `BUILD_EXIT=0`.
- Greedy ratio negative: `cli-ratio-greedy`, `RUN_EXIT=1` at the early guard.
- Ratio-free greedy control passed the guard and reached the deliberately absent
  model.
- Offline replay positive: `denom-replay-positive`, `RUN_EXIT=0`; both epochs
  trained two trajectories (`trained=2`) with finite policy loss, KL, and clip
  fraction.
- Missing sidecar: `replay-missing-realmodel`, `RUN_EXIT=1` before store/model
  initialization.
- Misaligned sidecar: `replay-misaligned-realmodel`, `RUN_EXIT=1` before
  store/model initialization.
- CE missing-sidecar control passed sidecar validation and reached the deliberate
  LoRA-rank failure.
- Online stochastic positive: `rft-toy08b-g2`, Qwen3.5-0.8B, one existing
  synthetic SWE-shaped task, two Claude-Code samples, GRPO, temperature `0.3`,
  staleness `0`; `RUN_EXIT=0`.
- Real scoring produced rewards `[0.0, 0.5]` (`std=0.25`), so both trajectories
  survived the ratio-weighted update. The update trained 672 tokens with
  `policy_loss=0.0218927`, `is_ratio_mean=0.952895`,
  `is_ratio_max=9.580126`, `clip_frac=0.174107`, and
  `kl_rollout=0.0339127`.
- The online case followed the normal Claude Code → in-process ARLE serve →
  tool/edit → pytest scoring → sidecar conversion → `UpdatePreset::update`
  path. It did not use `--replay-records` or fabricate behavior probabilities.
- A separate `staleness=1` probe recorded a real version-lagged group. The
  combined stale + experience-replay positive is still pending an exclusive
  GPU window: a variance-bearing attempt reached writeback but an unrelated 27B
  campaign co-resident on GPU 1 reduced KV capacity and then caused allocation
  failure. No foreign process was killed, and that run is not counted as gate
  evidence.

Raw online artifacts:

- `/host/arle-runs/rft-toy08b-g2/run.log`
- `/host/arle-runs/rft-toy08b-g2/metrics.jsonl`
- `/host/arle-runs/rft-toy08b-g2/eval/dumps/`

Earlier 27B and long-prompt 0.8B attempts were invalid acceptance cases because
KV capacity and harness timeouts produced no trainable trajectories. The smaller
existing task changed the workload, not the gate, and exercised the complete
online path.

No performance claim is made. Control flow removes one redundant forward, but no
matched wall-clock A/B was run.

## Rule

A ratio denominator is generation evidence, not a value that training may
reconstruct later. Capture it at the sampler boundary, preserve it immutably
through replay, and reject the whole trainable batch before model work when the
evidence is absent or malformed.
