# DSv4 CUDA SW Decode Forward Phase 0

status=pending-remote

## Goal

Add the smallest falsifiable CUDA-side DSv4 one-token decode surface for the
V4-only 1B init checkpoint: load the model shell, run a single decode token,
and prove the returned logits vector has vocab length `129280` with finite
values.

## Hypothesis

If the CUDA loader and decode dispatch can stand up a small-window-only smoke
path without touching CSA/HCA, MoE routing, MTP, or prefill, then Phase 2A can
start from a shape/finite contract before numerical parity work.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
cargo test -p infer --features cuda dsv4_cuda_sw_one_token -- --ignored --nocapture
```

Full serving bench is pending remote/e2e wiring:

```bash
scripts/bench_guidellm.sh dsv4-cuda-sw-forward-phase0
```

## Environment

- Backend: CUDA
- Model: `infer/models/dsv4-mini-1B-init`
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, driver `595.71.05`
- CUDA: `nvcc 13.2.78`
- Commits under test:
  - `9374c653524737d8aaeeae9db47255358561abea`
  - `432fbb697399816ae9802e558e2085c7666a33fc`
- Feature set: `cargo test -p infer --features cuda`
- Non-default flags/env vars:
  - `NVCC_CCBIN=/usr/bin/g++-14`
  - `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`
- Server launch: not run in this tranche; unit smoke only.

## Results

| Check | Result |
| --- | --- |
| CUDA DSv4 one-token ignored test | PASS |
| Logits shape | PASS, `129280` |
| Logits finite | PASS |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | PASS |
| `cargo check -p infer --features cuda` | PASS with `NVCC_CCBIN` and `INFER_TILELANG_PYTHON`; unpinned host compiler path fails on GCC 16 / nvcc C++ frontend compatibility |
| `cargo clippy -p infer --features cuda -- -D warnings` | PASS with `NVCC_CCBIN` and `INFER_TILELANG_PYTHON` |
| guidellm serving sweep | pending-remote |

## Problems

- This is not a throughput or latency result. It is a shape/finite smoke for
  the CUDA DSv4 decode surface.
- The path intentionally does not implement numerical parity, real SW attention
  math, MoE routing, CSA/HCA, MTP, or prefill.
- The local default C++ compiler is GCC 16. CUDA builds need
  `NVCC_CCBIN=/usr/bin/g++-14` on this host.

## Learnings

- The V4 config contains sliding-window layers, so Phase 2A can start without
  reordering into CSA/HCA first.
- The Phase 0 acceptance contract did not require new CUDA kernels: zero new
  kernels were added, staying below the license-or-kill threshold of five.

## Delta vs baseline

- Baseline: no CUDA DSv4 decode smoke existed.
- Delta: one-token CUDA decode now returns finite logits of length `129280`.

## Artefacts

- Unit smoke: terminal output from
  `cargo test -p infer --features cuda dsv4_cuda_sw_one_token -- --ignored --nocapture`
- Full guidellm artefacts: pending.

## Notes

- This entry is a bench stub required by the runtime-change process. The
  serving benchmark remains deferred until the DSv4 CUDA server path is wired
  beyond the shape/finite unit smoke.
