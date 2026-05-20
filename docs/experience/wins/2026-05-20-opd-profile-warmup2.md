# OPD CPU moderate profile uses two warmup runs

## Goal

Make the OPD CPU moderate profile harness stable enough for single-variable
optimization decisions. Recent runs repeatedly had a slow first measured run,
which inflated sigma and made small wall-clock axes impossible to license or
kill cleanly.

## Hypothesis

The first measured run is still paying cache/allocation warmup. Increasing only
`WARMUP_RUNS` from 1 to 2 in `opd_step_cpu_moderate_profile` should remove that
cold sample from the measured set and bring sigma below the OPD license gate
of 5%.

## Params

- Backend: CPU
- Shape: hidden=512, intermediate=1536, layers=12, vocab=32768
- Prompt: `[1, 3, 8]`
- Rollout length: 2
- Optimizer: AdamW, lr=1e-3
- Measured runs: 3
- Steps per measured run: 5
- Single variable: `WARMUP_RUNS = 1` -> `2`

## Results

Baseline, one warmup:

```text
summary mean_steps_per_sec=1.001866 median_steps_per_sec=1.073371 sigma_steps_per_sec=0.146441 sigma_pct=14.617 total_step_seconds=15.332941
```

After, two warmups:

```text
summary mean_steps_per_sec=1.120221 median_steps_per_sec=1.127224 sigma_steps_per_sec=0.011329 sigma_pct=1.011 total_step_seconds=13.391521
```

| Metric | 1 warmup | 2 warmups | Delta |
|---|---:|---:|---:|
| sigma / mean | 14.617% | 1.011% | -13.606 pp |
| median steps/sec | 1.073371 | 1.127224 | +5.02% |

The throughput delta is not a production OPD speedup; it only means the old
measurement set included a cold sample. The licensed claim is measurement
stability for the profile harness.

## Problems

This does not fix the underlying cold-start behavior. It only makes the
profile harness report warmed steady-state measurements so future OPD
optimization axes can use sigma under 5% without changing the production step.

## Learnings

For OPD CPU moderate profiling, one warmup is not enough after the model and
autograd paths grew large enough for allocator/cache state to matter. Keep
cold-start behavior out of warmed steady-state license decisions; measure it
as a separate axis if startup latency becomes the target.

## Artefacts

- Baseline raw: `bench-output/2026-05-20-opd-rollout-last-logits-relicense/baseline.txt`
- After raw: `bench-output/2026-05-20-opd-profile-warmup2.txt`
