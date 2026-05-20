# AdamW host step uses zip loop locals

## Goal

Reduce the now-visible CPU `optimizer_step` cost in OPD after the
`LinearWithLora` projection fix. The previous borrow-grad experiment was
killed because avoiding the gradient clone regressed; this tranche keeps the
same data movement and only rewrites the hot AdamW element loop to use
iterator zips and local `m_next` / `v_next` values.

## Hypothesis

The old index loop repeatedly indexed `param.data`, `grad`, `m`, and `v` by
the same counter. A zip loop should give LLVM a simpler alias/bounds-check
shape while preserving the exact AdamW update order:

```text
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g*g
p -= step_size*m / (sqrt(v / bc2) + eps)
```

License if the isolated host AdamW wall-clock improves by at least 5% with
`n=3`, sigma under 5%, and identical checksum.

## Params

- Backend: CPU
- Bench: `crates/autograd/examples/bench_adamw_host.rs`
- Params: 64 tensors
- Elements per tensor: 262144
- Iterations per run: 10 optimizer steps
- Weight decay: 0.0, matching current OPD smoke/moderate benches
- Command:

```bash
timeout 600 bash -lc 'for run in 1 2 3; do \
  cargo run -q -j 1 -p autograd --example bench_adamw_host --release -- \
    --params 64 --len 262144 --iters 10 --wd 0.0; \
done'
```

## Env

| Item | Value |
|---|---|
| Backend | CPU `TensorStore::default()` |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Cargo | `cargo 1.95.0 (f2d3ce0bd 2026-03-21)` |
| Baseline | same-session pre-change host AdamW baseline |
| Non-default flags / env vars | `timeout 600`; cargo `-j 1` |

## Results

Baseline:

```text
params=64 len=262144 iters=10 lr=0.0003 wd=0 wall=0.656091s step_ms=65.609065 steps/s=15.242 ns_per_element=3.911 checksum=-40.144982
params=64 len=262144 iters=10 lr=0.0003 wd=0 wall=0.654245s step_ms=65.424497 steps/s=15.285 ns_per_element=3.900 checksum=-40.144982
params=64 len=262144 iters=10 lr=0.0003 wd=0 wall=0.644315s step_ms=64.431520 steps/s=15.520 ns_per_element=3.840 checksum=-40.144982
```

After zip-loop locals:

```text
params=64 len=262144 iters=10 lr=0.0003 wd=0 wall=0.216935s step_ms=21.693500 steps/s=46.097 ns_per_element=1.293 checksum=-40.144982
params=64 len=262144 iters=10 lr=0.0003 wd=0 wall=0.218727s step_ms=21.872654 steps/s=45.719 ns_per_element=1.304 checksum=-40.144982
params=64 len=262144 iters=10 lr=0.0003 wd=0 wall=0.217468s step_ms=21.746773 steps/s=45.984 ns_per_element=1.296 checksum=-40.144982
```

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| median step ms | 65.424497 | 21.746773 | **3.01x faster** |
| median steps/sec | 15.285 | 45.984 | **3.01x** |
| checksum | -40.144982 | -40.144982 | identical |
| after sigma / mean | 0.34% |  | pass (<5%) |

## OPD Profile Cross-Check

The moderate OPD phase profile shows the optimizer phase moving, but this
secondary end-to-end run is slightly above the sigma gate because the first
measured run remains slower:

```text
run=1 wall_seconds=4.675599 summed_step_seconds=4.675588 steps_per_sec=1.069382 first_loss=0.000314203 last_loss=0.000315835
run=2 wall_seconds=4.137672 summed_step_seconds=4.137653 steps_per_sec=1.208409 first_loss=0.000314203 last_loss=0.000315835
run=3 wall_seconds=4.161628 summed_step_seconds=4.161617 steps_per_sec=1.201453 first_loss=0.000314203 last_loss=0.000315835
summary mean_steps_per_sec=1.159748 median_steps_per_sec=1.201453 sigma_steps_per_sec=0.063962 sigma_pct=5.515 total_step_seconds=12.974858
```

Compared with the `LinearWithLora` post-change profile, median OPD profile
throughput moved from 0.856441 to 1.201453 steps/sec (1.40x), and
`optimizer_step` moved from 8.180662 s to 3.024048 s over the same 15 profiled
steps (2.71x). Treat this as supportive only; the licensed signal is the
isolated AdamW wall-clock A/B above.

## Implementation

- Added a grad length assertion before the host update loop.
- Hoisted `1-beta1` and `1-beta2` out of the element loop.
- Replaced repeated index reads/writes with
  `param.data.iter_mut().zip(&grad).zip(m.iter_mut().zip(v.iter_mut()))`.
- Writes `m_next` and `v_next` once, then reuses those local values for the
  parameter update.

## Verification

- `cargo run -q -j 1 -p autograd --example bench_adamw_host --release -- --params 64 --len 262144 --iters 10 --wd 0.0`
- `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release`
- `cargo fmt --check -p autograd -p train`
- `cargo test -j 1 -p autograd --release`
- `cargo clippy -j 1 -p autograd --all-targets --release -- -D warnings`
- `cargo test -j 1 -p train --test test_opd_determinism --release`
- `cargo clippy -j 1 -p train --all-targets --release -- -D warnings`
- `cargo check -j 1 --workspace`
- `cargo test -j 1 -p train --release`
- `cargo build -j 1 --workspace --release`

## Problems

The OPD end-to-end profile is just above the sigma gate (5.515%), so this
commit does not claim the full 1.40x OPD speedup as a licensed end-to-end
number. The isolated optimizer-step wall-clock A/B is stable and directly
targets the measured top phase.

## Learnings

After projection transposes were removed, host AdamW became large enough that
plain Rust loop shape mattered. The winning change did not alter data movement
or math; it made the compiler's hot loop simpler.

## Artefacts

- Baseline raw: `bench-output/2026-05-20-adamw-host-borrow-grad-ab/baseline.txt`
- Baseline raw sha256:
  `5905b1ab6616632d8f6932fba3c7cc85184d485e92240f9d120b23a06667d062`
- AdamW after raw: `bench-output/2026-05-20-adamw-host-zip-loop-ab/zip.txt`
- AdamW after raw sha256:
  `19f8fe18a9a8d2bb3fc2af5e28ad3886efad1c071a42fbcc8806e71de6ad166e`
- OPD profile raw: `bench-output/2026-05-20-adamw-host-zip-loop-ab/opd_profile_after.txt`
- OPD profile raw sha256:
  `0640eccd28dd07ccbc0415adecb79c0ec4c042ed303249d5a8ae78d4fca303d9`
