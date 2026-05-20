# OPD CPU moderate phase profile after matmul_bt

## Goal

Measure wall-clock phase attribution for the moderate OPD CPU step after the
retain prune and `matmul_bt` linear path. This is a diagnostic bench: it picks
the next axis from measured phase cost instead of source-level guesses.

## Hypothesis

At the moderate Qwen3.5-like shape, end-to-end time should still be dominated
by model forward/backward and optimizer work. `kl_distill_loss` should remain
too small to justify more loss-kernel work before the larger phases move.

## Params

- Backend: CPU
- Shape: hidden=512, intermediate=1536, layers=12, vocab=32768
- Attention: num_heads=8, num_kv_heads=4, head_dim=64
- Prompt: `[1, 3, 8]`
- Rollout length: 2
- Optimizer: AdamW, lr=1e-3
- Runs: 1 warmup, 3 measured, 5 OPD steps per measured run
- Command:

```bash
timeout 900 bash -lc 'cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release \
  | tee bench-output/2026-05-20-opd-step-cpu-moderate-profile/run.txt'
```

## Env

| Item | Value |
|---|---|
| Backend | CPU `TensorStore::default()` |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Cargo | `cargo 1.95.0 (f2d3ce0bd 2026-03-21)` |
| Runtime substrate commit under test | `b27b390` |
| Feature set | `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release` |
| Non-default flags / env vars | `timeout 900`; cargo `-j 1` to cap build memory |

## Results

```text
run=1 wall_seconds=18.003963 summed_step_seconds=18.003938 steps_per_sec=0.277717 first_loss=0.000314202 last_loss=0.000315835
run=2 wall_seconds=17.655396 summed_step_seconds=17.655371 steps_per_sec=0.283200 first_loss=0.000314202 last_loss=0.000315835
run=3 wall_seconds=17.841531 summed_step_seconds=17.841520 steps_per_sec=0.280245 first_loss=0.000314202 last_loss=0.000315835
summary mean_steps_per_sec=0.280387 median_steps_per_sec=0.280245 sigma_steps_per_sec=0.002241 sigma_pct=0.799 total_step_seconds=53.500829
```

Memory stayed bounded: observed used memory rose from 5.6 GiB to about 9.3 GiB
during the run, then returned to 5.6 GiB after exit. No additional swap growth
or kernel SIGKILL occurred.

## Wall-Clock Phase Attribution

| rank | phase | seconds | pct total |
|---:|---|---:|---:|
| 1 | `rollout_student_forward` | 16.289272 | **30.447%** |
| 2 | `backward` | 11.558201 | **21.604%** |
| 3 | `teacher_forward` | 8.164515 | **15.261%** |
| 4 | `student_forward` | 8.153000 | **15.239%** |
| 5 | `optimizer_step` | 8.091713 | **15.124%** |
| 6 | `grad_clip` | 0.826883 | 1.546% |
| 7 | `optimizer_zero_grad` | 0.253190 | 0.473% |
| 8 | `post_step_cleanup` | 0.133691 | 0.250% |
| 9 | `kl_distill_loss` | 0.028031 | 0.052% |
| 10 | `rollout_argmax_readback` | 0.001799 | 0.003% |
| 11 | `keep_extra_build` | 0.000330 | 0.001% |
| 12 | `loss_readback` | 0.000006 | 0.000% |
| 13 | `rollout_positions` | 0.000003 | 0.000% |
| 14 | `full_positions` | 0.000001 | 0.000% |
| 15 | `rollout_tape_disable` | 0.000000 | 0.000% |
| 16 | `student_tape_enable` | 0.000000 | 0.000% |

## Delta vs Moderate Baseline

| Metric | moderate baseline | moderate profile | Delta |
|---|---:|---:|---:|
| median steps/sec | 0.284854 | 0.280245 | -1.62% |
| sigma / mean | 0.504% | 0.799% | +0.295 pp |

The throughput delta is inside the stability band for a diagnostic harness.
Use the phase percentages as the artifact, not as a speedup claim.

## Problems

This is still a moderate-shape CPU bench, not Qwen3-0.6B. It is intentionally
bounded after the earlier cooperative-session SIGKILL: cargo ran with `-j 1`,
the bench ran under `timeout 900`, and memory was checked during the run.

The phase names are coarse. `rollout_student_forward`, `teacher_forward`,
`student_forward`, `backward`, and `optimizer_step` each contain many tensor
ops; attributing those regions to a specific matmul, norm, softmax, or AdamW
inner loop still needs a narrower single-variable bench.

## Learnings

The next licensed CPU axis should come from the measured top phases, not from
`kl_distill_loss`: KL is only 0.052% of total step time at this shape. Rollout
student forward is the largest single phase because `rollout_len=2` triggers
two student forwards before the full-sequence teacher/student pass. Optimizer
step is now large enough to be a plausible follow-up axis after forward and
backward attribution.

## Artefacts

- Raw: `bench-output/2026-05-20-opd-step-cpu-moderate-profile/run.txt`
- Raw sha256:
  `d8762db2b9e16ab26086834285ca96c74adfcd04794ad5c5d72ce3ae82e074c0`
