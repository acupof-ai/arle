# Autograd Host AdamW Invariant Precompute

## Goal

Type: optimization. Improve the default host AdamW training operator without
changing optimizer state, checkpoint schema, backend selection, or gradient
semantics.

## Hypothesis

The AdamW host loop performs two bias-correction divisions per element. Since
`bc1` and `bc2` are step-level invariants, precomputing `lr / bc1` and
`1 / bc2` should reduce scalar arithmetic in the per-element update loop while
preserving the same AdamW formula within the existing numerical tolerance.

## Command

Baseline was measured by temporarily restoring the original per-element
division formula in `crates/autograd/src/optim.rs` and running:

```bash
cargo run --release -p autograd --example bench_adamw_host -- \
  --params 256 --len 4096 --iters 100 --wd 0.01
```

Treatment used the final patch and the same command.

Correctness gates:

```bash
cargo test -p autograd --release adamw -- --nocapture
cargo test -p autograd --release
cargo test -p train --release --test test_trainer_loop
```

## Environment

- Host: AMD Ryzen 7 3700X, 8 cores / 16 threads.
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`, LLVM 22.1.2.
- Feature set: default CPU/host path, no CUDA or Metal features.
- Commit base: `e8b5cb5`; workspace had unrelated dirty docs, but the matched
  A/B touched only `crates/autograd/src/optim.rs` and the new bench example.
- Benchmark shape: 256 params x 4096 f32 values, 100 AdamW steps, `lr=0.0003`,
  `wd=0.01`.

## Results

Matched A/B, serial runs:

| Run | Baseline ns/element | Treatment ns/element | Baseline step ms | Treatment step ms | Checksum |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 4.457 | 3.984 | 4.673362 | 4.177206 | -13.173820 |
| 2 | 4.519 | 4.075 | 4.739017 | 4.272503 | -13.173820 |
| 3 | 4.713 | 4.064 | 4.941510 | 4.261671 | -13.173820 |
| Mean | 4.563 | 4.041 | 4.784630 | 4.237127 | -13.173820 |

Decision: LICENSE for this host microbench. Mean `ns/element` improved by
11.4%; all treatment checksums matched the baseline checksum.

Correctness:

- PASS: `cargo test -p autograd --release adamw -- --nocapture`
- PASS: `cargo test -p autograd --release`
- PASS: `cargo test -p train --release --test test_trainer_loop`

## Problems

The first optimization candidate tried to merge decoupled weight decay into
the AdamW update pass. It preserved the benchmark checksum but regressed:

| Candidate | ns/element range | Decision |
| --- | ---: | --- |
| A1: branch inside update loop | 5.357-5.413 | KILL |
| A2: branch outside update loop | 4.859-5.123 | KILL |

The likely mechanism is weaker vectorization and/or worse instruction
scheduling once the decay multiply is mixed into the sqrt-heavy update loop.
That mechanism is a hypothesis; the KILL decision rests on matched benchmark
regression, not source reading.

## Learnings

- For host AdamW, keep the decoupled weight-decay pass separate; fewer memory
  passes was slower on this CPU.
- Step-level scalar invariants are the safe first lever: precompute `lr / bc1`
  and `1 / bc2`, then keep the per-element loop shape otherwise unchanged.
- A checksum is a useful smoke signal for microbench parity, but it is not a
  replacement for the AdamW reference tests and trainer-loop state tests.

## Delta vs Baseline

First local host AdamW microbench entry for this shape. The treatment improves
mean step time from 4.784630 ms to 4.237127 ms (-11.4%) on the measured host.
