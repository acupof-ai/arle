# OPD CPU moderate step baseline after retain prune + matmul_bt

## Goal

Establish a real multi-step OPD wall-clock baseline after the retain-prune and
`matmul_bt` CPU linear changes. Earlier moderate runs on this host only emitted
the config line and did not finish, so they could not be used as ground truth.

## Hypothesis

With `opd_step` pruning temporaries after backward and `linear_forward`
avoiding physical weight transposes, a moderate Qwen3.5-like shape should run
for repeated steps without swap growth or kernel SIGKILL. This is a baseline,
not a new optimization claim.

## Params

- Backend: CPU
- Shape: hidden=512, intermediate=1536, layers=12, vocab=32768
- Attention: num_heads=8, num_kv_heads=4, head_dim=64
- Prompt: `[1, 3, 8]`
- Rollout length: 2
- Optimizer: AdamW, lr=1e-3
- Runs: 1 warmup, 3 measured, 10 OPD steps per measured run
- Command:

```bash
cargo run -p train --example opd_step_cpu_moderate_bench --release \
  | tee bench-output/2026-05-20-opd-step-cpu-moderate-post-matmul-bt/run.txt
```

## Results

```text
run=1 wall_seconds=35.351009 per_step_seconds=3.535101 steps_per_sec=0.282877 first_loss=0.000314202 last_loss=0.000315440
run=2 wall_seconds=35.105748 per_step_seconds=3.510575 steps_per_sec=0.284854 first_loss=0.000314202 last_loss=0.000315440
run=3 wall_seconds=34.918523 per_step_seconds=3.491852 steps_per_sec=0.286381 first_loss=0.000314202 last_loss=0.000315440
summary mean_steps_per_sec=0.284704 median_steps_per_sec=0.284854 sigma_steps_per_sec=0.001434 sigma_pct=0.504 mean_step_seconds=3.512509 median_step_seconds=3.510575
```

Peak observed memory during the run stayed around 9.5 GiB used with ~21 GiB
available and no additional swap growth. That is the load-bearing safety check
for this bench after the earlier cooperative-session SIGKILL.

## Verification

- `cargo run -p train --example opd_step_cpu_moderate_bench --release`

## Problems

This is a moderate-shape baseline, not Qwen3-0.6B. It exists because the 31 GiB
dev box cannot safely run repeated full-shape OPD steps while other agent work
is active. Use it for regression direction and memory safety, not as a direct
production throughput number.

## Learnings

The retain prune fixed the immediate bench blocker: multi-step OPD no longer
accumulates previous-step activations enough to kill the process. With that
bounded, further OPD CPU optimizations can use this moderate shape as a safe
wall-clock guardrail before attempting any larger shape.
