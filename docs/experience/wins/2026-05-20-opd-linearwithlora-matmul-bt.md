# OPD CPU LinearWithLora uses matmul_bt

## Goal

Remove the remaining physical weight transposes from train-side Qwen3.5 layer
projections. The prior `matmul_bt` tranche switched only the private
`qwen35.rs::linear_forward` path used by `lm_head`; the layer projections still
went through `LinearWithLora::forward`, which materialized `weight^T` before
every q/k/v/o/gate/up/down projection.

## Hypothesis

Replacing `transpose(weight) + matmul` with direct `matmul_bt(flat_x, weight)`
inside `LinearWithLora` should preserve math for both base weights and LoRA
adapters while cutting OPD forward and backward wall-clock. The licensing gate
is the same moderate OPD CPU phase harness used in the previous commit:
matched controls, `n=3`, and sigma under 5%.

## Params

- Backend: CPU
- Shape: hidden=512, intermediate=1536, layers=12, vocab=32768
- Attention: num_heads=8, num_kv_heads=4, head_dim=64
- Prompt: `[1, 3, 8]`
- Rollout length: 2
- Optimizer: AdamW, lr=1e-3
- Profile runs: 1 warmup, 3 measured, 5 OPD steps per measured run
- Command:

```bash
timeout 900 bash -lc 'cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release \
  | tee bench-output/2026-05-20-opd-linearwithlora-matmul-bt/profile_after.txt'
```

## Env

| Item | Value |
|---|---|
| Backend | CPU `TensorStore::default()` |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Cargo | `cargo 1.95.0 (f2d3ce0bd 2026-03-21)` |
| Baseline | `67a4d63` profile entry `2026-05-20-opd-step-cpu-moderate-profile.md` |
| Feature set | `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release` |
| Non-default flags / env vars | `timeout 900`; cargo `-j 1` to cap build memory |

## Results

Baseline from `67a4d63`:

```text
summary mean_steps_per_sec=0.280387 median_steps_per_sec=0.280245 sigma_steps_per_sec=0.002241 sigma_pct=0.799 total_step_seconds=53.500829
```

After `LinearWithLora::forward -> matmul_bt`:

```text
run=1 wall_seconds=6.341896 summed_step_seconds=6.341886 steps_per_sec=0.788408 first_loss=0.000314203 last_loss=0.000315835
run=2 wall_seconds=5.801416 summed_step_seconds=5.801397 steps_per_sec=0.861859 first_loss=0.000314203 last_loss=0.000315835
run=3 wall_seconds=5.838115 summed_step_seconds=5.838105 steps_per_sec=0.856441 first_loss=0.000314203 last_loss=0.000315835
summary mean_steps_per_sec=0.835569 median_steps_per_sec=0.856441 sigma_steps_per_sec=0.033421 sigma_pct=4.000 total_step_seconds=17.981388
```

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| median steps/sec | 0.280245 | 0.856441 | **3.06x** |
| summed profiled step seconds | 53.500829 | 17.981388 | **2.98x faster** |
| sigma / mean | 0.799% | 4.000% | pass (<5%) |

Memory stayed bounded: observed used memory rose to about 8.1 GiB during the
profile run, then returned to about 5.5 GiB after exit. No additional swap
growth or kernel SIGKILL occurred.

## Phase Delta

| phase | Before seconds | After seconds | Delta |
|---|---:|---:|---:|
| `rollout_student_forward` | 16.289272 | 2.558369 | **6.37x faster** |
| `teacher_forward` | 8.164515 | 1.186765 | **6.88x faster** |
| `student_forward` | 8.153000 | 1.123409 | **7.26x faster** |
| `backward` | 11.558201 | 3.764599 | **3.07x faster** |
| `optimizer_step` | 8.091713 | 8.180662 | 0.99x |

The forward regions moved exactly where expected: layer projections no longer
copy and transpose weights every call. `optimizer_step` is now the largest
coarse phase because the projection path shrank around it.

## Secondary Bench

The 10-step moderate bench showed similar median throughput but did not meet
the sigma gate because the first measured run was repeatably slower:

```text
run=1 wall_seconds=15.656855 per_step_seconds=1.565686 steps_per_sec=0.638698 first_loss=0.000314202 last_loss=0.000315440
run=2 wall_seconds=11.775446 per_step_seconds=1.177545 steps_per_sec=0.849225 first_loss=0.000314202 last_loss=0.000315440
run=3 wall_seconds=11.517334 per_step_seconds=1.151733 steps_per_sec=0.868257 first_loss=0.000314202 last_loss=0.000315440
summary mean_steps_per_sec=0.785393 median_steps_per_sec=0.849225 sigma_steps_per_sec=0.104020 sigma_pct=13.244 mean_step_seconds=1.298321 median_step_seconds=1.177545
```

This is supportive only, not the licensing number.

## Implementation

- Switched `LinearWithLora::forward` base projection from
  `transpose(self.weight) + matmul` to `matmul_bt(flat_x, self.weight)`.
- Switched LoRA adapter projections from transposed `lora_a` / `lora_b` to
  direct `matmul_bt(flat_x, lora_a)` and `matmul_bt(low_rank, lora_b)`.
- Added a LoRA zero-B forward test: with deterministic base weights and
  zero-initialized `lora_b`, a LoRA model must match the frozen base logits.

## Verification

- `cargo fmt --check -p train`
- `cargo test -j 1 -p train --test test_qwen35_forward --release`
- `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release`
- `cargo clippy -j 1 -p train --all-targets --release -- -D warnings`
- `cargo check -j 1 --workspace`
- `cargo test -j 1 -p train --release`
- `cargo build -j 1 --workspace --release`

## Problems

The phase profile is moderate-shape CPU, not Qwen3-0.6B. It is the safe
wall-clock guardrail for this 31 GiB host. The 10-step bench's first measured
run remains noisy, so the commit uses the phase harness as the license signal.

`optimizer_step` now dominates the coarse profile. The previous borrow-grad
AdamW experiment regressed by about 2x, so the next optimizer axis needs a
different license experiment rather than removing the existing grad clone.

## Learnings

The earlier `matmul_bt` commit left a second train-side linear path behind.
OPD forward was still paying physical transpose copies for every
`LinearWithLora` layer projection. Keeping one canonical `A @ weight^T`
primitive across base and LoRA linears is both simpler and much faster.

## Artefacts

- Raw profile: `bench-output/2026-05-20-opd-linearwithlora-matmul-bt/profile_after.txt`
- Raw profile sha256:
  `ddb80b56aa6965e85aa9af3f4a48a694418d2c1d42c440810a7c2b1ae8343e77`
- Raw secondary bench:
  `bench-output/2026-05-20-opd-linearwithlora-matmul-bt/moderate_bench_after_rerun.txt`
- Raw secondary bench sha256:
  `76d33089d0d42ffdcb268b17b97a03db741e04a8126e8ac8a419c180ba10d60f`
