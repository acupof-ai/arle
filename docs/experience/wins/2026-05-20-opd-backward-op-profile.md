# OPD CPU backward op attribution

## Goal

Identify the next OPD CPU optimization axis after the `LinearWithLora`
`matmul_bt` rewrite and host AdamW loop cleanup. The coarse phase profile now
shows `backward` as the largest step phase, but optimizing "backward" as a
single bucket is too broad. This tranche adds measured attribution by
`BackwardOp`.

## Hypothesis

The largest remaining backward cost should be one of:

- `MatmulBT` from layer and `lm_head` projections,
- `merge_grad` from accumulating many parameter gradients, or
- a non-matmul op whose count exploded after the layer rewrite.

The output is diagnostic. It does not claim a speedup.

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
  | tee bench-output/2026-05-20-opd-backward-op-profile/run.txt'
```

## Env

| Item | Value |
|---|---|
| Backend | CPU `TensorStore::default()` |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Cargo | `cargo 1.95.0 (f2d3ce0bd 2026-03-21)` |
| Feature set | `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release` |
| Non-default flags / env vars | `timeout 900`; cargo `-j 1` |

## Results

```text
run=1 wall_seconds=6.273664 summed_step_seconds=6.273646 steps_per_sec=0.796982 first_loss=0.000314203 last_loss=0.000315835
run=2 wall_seconds=4.295157 summed_step_seconds=4.295130 steps_per_sec=1.164102 first_loss=0.000314203 last_loss=0.000315835
run=3 wall_seconds=4.078714 summed_step_seconds=4.078695 steps_per_sec=1.225877 first_loss=0.000314203 last_loss=0.000315835
summary mean_steps_per_sec=1.062320 median_steps_per_sec=1.164102 sigma_steps_per_sec=0.189310 sigma_pct=17.820 total_step_seconds=14.647471
```

The first measured run is again slower, so throughput is not a licensed
performance number here. The backward attribution is still useful because it
aggregates 15 profiled backward passes and the op ranking is stable across the
two local runs made during development.

## Phase Summary

```text
phase_summary rank=1 phase=backward seconds=4.231776 pct_total=28.891
phase_summary rank=2 phase=rollout_student_forward seconds=3.267531 pct_total=22.308
phase_summary rank=3 phase=optimizer_step seconds=2.949581 pct_total=20.137
phase_summary rank=4 phase=teacher_forward seconds=1.531127 pct_total=10.453
phase_summary rank=5 phase=student_forward seconds=1.449392 pct_total=9.895
phase_summary rank=6 phase=grad_clip seconds=0.821821 pct_total=5.611
```

## Backward Attribution

```text
backward_profile_summary total_seconds=4.231762 op_seconds=2.572210 merge_grad_seconds=1.653195 prelude_seconds=0.002379 unattributed_seconds=0.003979
backward_op_summary rank=1 op=MatmulBT count=1275 seconds=2.374967 pct_backward=56.122
backward_op_summary rank=2 op=AddBroadcast count=540 seconds=0.040517 pct_backward=0.957
backward_op_summary rank=3 op=Transpose count=1080 seconds=0.038116 pct_backward=0.901
backward_op_summary rank=4 op=Embedding count=15 seconds=0.029221 pct_backward=0.691
backward_op_summary rank=5 op=Mul count=375 seconds=0.017840 pct_backward=0.422
backward_op_summary rank=6 op=Slice count=360 seconds=0.016575 pct_backward=0.392
backward_op_summary rank=7 op=Softmax count=195 seconds=0.011070 pct_backward=0.262
backward_op_summary rank=8 op=LogSoftmax count=15 seconds=0.010368 pct_backward=0.245
backward_op_summary rank=9 op=RMSNorm count=735 seconds=0.008201 pct_backward=0.194
backward_op_summary rank=10 op=Silu count=180 seconds=0.007380 pct_backward=0.174
backward_op_summary rank=11 op=Matmul count=360 seconds=0.006766 pct_backward=0.160
backward_op_summary rank=12 op=Reshape count=4725 seconds=0.006557 pct_backward=0.155
```

The next licensed axis is not generic softmax/norm work. It is either:

- `MatmulBT` backward, which accounts for 56.1% of backward wall-clock, or
- `merge_grad`, which accounts for 39.1% of backward wall-clock.

## Implementation

- Added additive `Tape::backward_profiled`, returning the same gradients as
  `Tape::backward` plus a `BackwardProfile`.
- Regular `Tape::backward` still calls the shared implementation with no
  profile object, so normal training does not run per-op timers.
- Exported `BackwardProfile` and `BackwardOpProfile` from `autograd`.
- Extended `opd_step_cpu_moderate_profile` to print
  `backward_profile_summary` and `backward_op_summary` rows.
- Added a unit test that `backward_profiled` matches plain `backward` and
  counts the expected ops for a tiny `sum(x*x)` graph.

## Verification

- `cargo fmt --check -p autograd -p train`
- `cargo test -j 1 -p autograd --release`
- `cargo clippy -j 1 -p autograd --all-targets --release -- -D warnings`
- `cargo test -j 1 -p autograd --lib --release`
- `cargo clippy -j 1 -p train --all-targets --release -- -D warnings`
- `cargo check -j 1 --workspace`
- `cargo test -j 1 -p train --release`
- `cargo build -j 1 --workspace --release`
- `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release`
- `git diff -- <this tranche> | codex review -`

## Problems

This is an attribution tool, not a speedup. The profile still has cold-run
noise in the first measured run, so only the coarse ranking should drive the
next tranche. Any change to `MatmulBT` backward or `merge_grad` must still
ship its own single-variable A/B.

## Learnings

The long tail of non-matmul backward ops is small. `MatmulBT` plus gradient
merge explain almost all remaining backward cost. The earlier small-M
outer-product hypothesis for `matmul_at_b_into` was tested and killed before
this commit: it regressed the existing backward microbench total from
0.969334 s to 1.156189 s, so the next `MatmulBT` axis needs a different
approach.

## Artefacts

- Raw: `bench-output/2026-05-20-opd-backward-op-profile/run.txt`
- Raw sha256:
  `17ab09d74454b7014415ab1a9c6f8a2c82773f7a26be79bba1ba2fd2ba8f46fe`
