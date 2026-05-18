# OPD step CPU tiny phase profile - component bench, AMD Ryzen 7 3700X, 2026-05-19

## Goal

- **(diagnosis)** Measure `train::opd::opd_step` wall-clock phase attribution
  on the CPU tiny OPD substrate so follow-up B3 work starts from measured
  regions instead of source-level guesses.

## Hypothesis

- The tiny CPU step should spend most wall-clock time in model forward and
  backward. `kl_distill_loss` itself should be small at vocab=16; if it
  dominates, A2-style loss work would move earlier in the queue.

## Command

```bash
cargo run -p train --example opd_step_cpu_profile --release \
  | tee bench-output/2026-05-19-opd-step-cpu-tiny-profile/opd_step_cpu_tiny_profile.txt
```

Benchmark constants:

```text
backend=cpu
hidden=16 layers=2 vocab=16
prompt=[1, 3, 8]
rollout_len=2
lr=0.001
steps_per_run=100
warmup_runs=1
measured_runs=5
```

## Environment

| Item | Value |
|---|---|
| Backend | CPU `TensorStore::default()` |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Cargo | `cargo 1.95.0 (f2d3ce0bd 2026-03-21)` |
| Runtime substrate commit under test | `4da07c9` (`train` / `autograd` OPD path unchanged) |
| Feature set | `cargo run -p train --example opd_step_cpu_profile --release` |
| Non-default flags / env vars | none |

## Results

### Steps/sec sanity

| run | wall seconds | summed step seconds | steps/sec | first loss | last loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.089645 | 0.089565 | 1115.513560 | 0.173121601 | 0.173121557 |
| 2 | 0.085113 | 0.085035 | 1174.914587 | 0.173121601 | 0.173121557 |
| 3 | 0.085755 | 0.085676 | 1166.114504 | 0.173121601 | 0.173121557 |
| 4 | 0.085244 | 0.085165 | 1173.104413 | 0.173121601 | 0.173121557 |
| 5 | 0.084739 | 0.084661 | 1180.097319 | 0.173121601 | 0.173121557 |

| metric | value |
|---|---:|
| mean steps/sec | 1161.948876 |
| median steps/sec | **1173.104413** |
| sigma steps/sec | 23.645349 |
| sigma / mean | **2.035%** |
| summed profiled step seconds | 0.430102 |

Median throughput is within the prior tiny baseline noise band
(`1195.334842` steps/sec), so the instrumentation is suitable for diagnosis.

### Wall-clock phase attribution

| rank | phase | seconds | pct total |
|---:|---|---:|---:|
| 1 | `backward` | 0.134678 | **31.313%** |
| 2 | `rollout_student_forward` | 0.128167 | **29.799%** |
| 3 | `teacher_forward` | 0.079005 | **18.369%** |
| 4 | `student_forward` | 0.072612 | 16.882% |
| 5 | `optimizer_step` | 0.010350 | 2.406% |
| 6 | `grad_clip` | 0.002031 | 0.472% |
| 7 | `kl_distill_loss` | 0.001062 | 0.247% |
| 8 | `post_step_tape_clear` | 0.000757 | 0.176% |
| 9 | `optimizer_zero_grad` | 0.000567 | 0.132% |
| 10 | `rollout_argmax_readback` | 0.000171 | 0.040% |
| 11 | `loss_readback` | 0.000041 | 0.009% |
| 12 | `rollout_positions` | 0.000028 | 0.007% |
| 13 | `full_positions` | 0.000013 | 0.003% |
| 14 | `rollout_tape_disable` | 0.000011 | 0.003% |
| 15 | `student_tape_enable` | 0.000011 | 0.003% |

Top measured regions: backward (31.3%), rollout student forwards (29.8%),
and teacher forward (18.4%). All forward regions together account for 65.0%
of step wall-clock at this shape.

## Problems

- `perf` / `cargo flamegraph` are not available on this host (`perf: command
  not found`), so this tranche records coarse `Instant` phase attribution
  rather than symbol-level op timing. Any claim that matmul, RMSNorm, RoPE,
  or softmax dominates inside the forward/backward regions remains a
  hypothesis until the next op-instrumented tranche or a perf-capable host run.
- This is a component bench, not a `guidellm` service benchmark; `/v1/stats`
  counters and request-token accounting do not apply.
- The workspace had an unrelated uncommitted `crates/cli/src/train_cli.rs`
  from-dir OPD diff during the run. The profile command links and executes
  the `train` crate example directly, and the OPD substrate files under
  `crates/train/src` and `crates/autograd/src` were not changed for this
  measurement.

## Learnings

- B3 should start with backward and forward internals, not `kl_distill_loss`
  micro-optimizations: KL is only 0.247% of tiny CPU step wall-clock.
- Rollout cost scales with `rollout_len` because each sampled token runs a
  student forward. At rollout_len=2 it is already the second-largest measured
  region.
- Optimizer and gradient clipping together are below 3% on this tiny CPU
  shape, so they are not the first CPU substrate bottleneck.

## Delta vs baseline

| Metric | B1 tiny baseline | B2 phase profile | Delta |
|---|---:|---:|---:|
| median steps/sec | 1195.334842 | 1173.104413 | -1.86% |
| sigma / mean | 2.254% | 2.035% | -0.219 pp |

The throughput delta is inside the matched-control noise band; treat the phase
percentages as the useful artifact, not as an optimization win.

## Artefacts

- Raw: `bench-output/2026-05-19-opd-step-cpu-tiny-profile/opd_step_cpu_tiny_profile.txt`
- Raw sha256:
  `c54edcbc6384523c9f9f63e17753df446244ec38be0dca56ac9232d668fc9cd2`
