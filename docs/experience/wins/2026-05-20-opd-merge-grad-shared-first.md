# OPD CPU merge_grad shared first gradient

## Goal

Reduce the OPD CPU `backward` phase after
`2026-05-20-opd-backward-op-profile.md` attributed 39.1% of backward
wall-clock to `merge_grad`.

## Hypothesis

`merge_grad` was doing duplicate work for tensors whose first gradient arrived
in a backward pass: it cloned `new_grad_id` into the returned `grads` map, then
`TensorStore::accumulate_grad` cloned the same tensor again into `.grad`.
Sharing the first cloned tensor between both surfaces should preserve values
while removing one clone/accumulate path for fresh activations.

## Params

- Backend: CPU
- Shape: hidden=512, intermediate=1536, layers=12, vocab=32768
- Attention: num_heads=8, num_kv_heads=4, head_dim=64
- Prompt: `[1, 3, 8]`
- Rollout length: 2
- Optimizer: AdamW, lr=1e-3
- Runs per sample: 1 warmup, 3 measured, 5 OPD steps per measured run
- Command:

```bash
timeout 900 bash -lc 'cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release \
  | tee bench-output/2026-05-20-merge-grad-shared-first-ab/<sample>.txt'
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

| sample | total_step_s | backward_s | merge_grad_s |
|---|---:|---:|---:|
| baseline | 14.687293 | 4.233578 | 1.656864 |
| baseline-r2 | 14.597029 | 4.201522 | 1.644249 |
| shared-first-r4-fixed | 13.334064 | 4.100393 | 1.491302 |
| shared-first-r5-fixed | 13.337599 | 4.112950 | 1.497887 |

| metric | baseline mean | shared-first mean | delta |
|---|---:|---:|---:|
| `merge_grad_seconds` | 1.650557 | 1.494595 | -9.45% |
| `backward total_seconds` | 4.217550 | 4.106672 | -2.63% |
| summed `total_step_seconds` | 14.642161 | 13.335832 | -8.92% |

Variance was low on the direct target:

| metric | baseline sigma_pct | shared-first sigma_pct |
|---|---:|---:|
| `merge_grad_seconds` | 0.382% | 0.220% |
| summed `total_step_seconds` | 0.308% | 0.013% |

The headline licensed claim is the direct target:
`merge_grad_seconds` improved by 9.45% with matched controls. The full step
also improved in this harness, but some non-backward phases moved as well, so
the conservative attribution is the measured `merge_grad` reduction.

## Implementation

- `merge_grad` now records the tensor's existing `.grad` id before merging.
- If this is the first gradient for a tensor and `.grad` is empty, the cloned
  gradient inserted into the returned `grads` map is also installed as
  `tensor.grad`.
- If a later merge updates the same id already stored in `.grad`, the separate
  `accumulate_grad` call is skipped to avoid double-adding.
- The fast path keeps the original shape check and uses `TensorStore::set_grad`
  so device-resident parameters are not demoted through `get_mut`.
- `AdamW::zero_grad` semantics are unchanged: existing persistent param grads
  are still zero-filled in place, so this does not change optimizer behavior.

## Verification

- `cargo fmt --check -p autograd -p train`
- `cargo test -j 1 -p autograd --lib --release`
- `cargo test -j 1 -p autograd --release`
- `cargo clippy -j 1 -p autograd --all-targets --release -- -D warnings`
- `cargo clippy -j 1 -p train --all-targets --release -- -D warnings`
- `cargo check -j 1 --workspace`
- `cargo test -j 1 -p train --release`
- `cargo build -j 1 --workspace --release`
- `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release`

## Problems

Per-run `steps_per_sec` still includes a slower first measured run in the
moderate profile harness. For this axis, the licensed signal comes from the
aggregated per-phase wall-clock counters over matched baseline/after samples.

A separate `cpu_matmul_bt_backward` direct-`grad_a` experiment was killed before
this commit: estimated OPD student backward cost regressed from 0.073240 s to
0.073727 s, so no code from that axis was kept.

## Learnings

The `merge_grad` cost was not just hash-map overhead. A meaningful slice was
duplicate gradient materialization between the return map and tensor `.grad`
storage. Sharing the first gradient keeps correctness stable and shrinks the
largest non-matmul part of OPD backward.

## Artefacts

- Baseline raw: `bench-output/2026-05-20-merge-grad-shared-first-ab/baseline.txt`
  - sha256: `4c1a4c51c556ae652a80f24a28a028341da336fc44bb44fbbbf2a356e35c2363`
- Baseline-r2 raw: `bench-output/2026-05-20-merge-grad-shared-first-ab/baseline-r2.txt`
  - sha256: `866118e5c0e0fbec058f51b62f24ae87c4de4f0ff6e38cad76308643c089e26a`
- Shared-first-r4-fixed raw: `bench-output/2026-05-20-merge-grad-shared-first-ab/shared-first-r4-fixed.txt`
  - sha256: `f770c60e2cea22d4a9f7739ada6487bb69e2ea087b3bed7fc7a8024c7661a056`
- Shared-first-r5-fixed raw: `bench-output/2026-05-20-merge-grad-shared-first-ab/shared-first-r5-fixed.txt`
  - sha256: `cc9e1a8c652c5714ee0d0cba40b9e8813c0edda46d393bf5db54762f40060323`
