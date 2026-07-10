# OPD step CPU tiny baseline — component bench, AMD Ryzen 7 3700X, 2026-05-19

## Goal

- **(baseline)** Measure `train::opd::opd_step` wall-clock steps/sec on the CPU
  reference path with the embedded tiny Qwen3.5-shaped config before any OPD
  performance work.

## Hypothesis

- The tiny CPU path should be dominated by Rust-side matmul/softmax/backward
  overhead rather than model size. Expected baseline: roughly 500-1,500
  OPD steps/sec with run-to-run sigma below 5% after one warmup run.

## Command

```bash
cargo run -p train --example opd_step_cpu_bench --release \
  | tee bench-output/2026-05-19-opd-step-cpu-tiny/opd_step_cpu_tiny_repo_example.txt
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
| Runtime commit under test | `d84b303` (`train` / `autograd` clean) |
| Feature set | `cargo run -p train --example opd_step_cpu_bench --release` |
| Non-default flags / env vars | none |

## Results

### Steps/sec

| run | seconds | steps/sec | first loss | last loss |
|---:|---:|---:|---:|---:|
| 1 | 0.088543 | 1129.400115 | 0.173121616 | 0.173121527 |
| 2 | 0.083659 | 1195.334842 | 0.173121616 | 0.173121527 |
| 3 | 0.083738 | 1194.200846 | 0.173121616 | 0.173121527 |
| 4 | 0.083490 | 1197.741749 | 0.173121616 | 0.173121527 |
| 5 | 0.083568 | 1196.633081 | 0.173121616 | 0.173121527 |

| metric | value |
|---|---:|
| mean steps/sec | 1182.662127 |
| median steps/sec | **1195.334842** |
| sigma steps/sec | 26.657698 |
| sigma / mean | **2.254%** |

The matched-control variance is below the goal's sigma <5% bar, so this is
usable as the CPU tiny baseline for follow-up B2/B3 profiling and A/B work.

## Problems

- This is a component bench, not a `guidellm` service benchmark; `/v1/stats`
  counters and request-token accounting do not apply.
- The workspace had an unrelated uncommitted `crates/cli/src/train_cli.rs`
  from-dir OPD diff during the run. The benchmark command links and executes
  the `train` crate example directly, and `git diff --name-only -- crates/train
  crates/autograd crates/qwen35-spec` was empty before recording the result.
- Qwen3-0.6B OPD CPU baseline is not included here. The un-gated full-attention
  path has landed, but a full teacher+student CPU backward run is a separate
  B1b measurement because it is expected to be much slower and needs its own
  timeout/artefact handling.

## Learnings

- The current tiny OPD CPU substrate has a stable baseline around
  1.2k steps/sec, with enough repeatability for single-variable CPU A/B tests.
- Loss values are bit-stable across repeated benchmark runs, matching the A3
  determinism test and keeping B2 wall-clock profiling attributable.

## Delta vs baseline

- First OPD `opd_step` CPU tiny baseline; no prior snapshot.

## Artefacts

- Raw: `bench-output/2026-05-19-opd-step-cpu-tiny/opd_step_cpu_tiny_repo_example.txt`
- Raw sha256:
  `d227dc6a820e1f475efda8632d8c33f36b632579c637a360871c99fb3dbb31c2`
