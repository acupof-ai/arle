# Qwen3.5 packed conv1d prefill surface — guidellm sweep, cuda, 2026-04-22

**Status:** `pending-remote`  
**Plan anchor:** [`docs/plans/2026-04-22-sglang-gap-closure-execution.md`](../../plans/2026-04-22-sglang-gap-closure-execution.md)  
**Change scope:** `crates/cuda-kernels/src/ffi/recurrent.rs`, `crates/cuda-kernels/csrc/misc/conv1d_prefill_batch.cu`, `infer/src/ops/recurrent.rs`

## Goal

- Regression / runway: record the required CUDA benchmark stub for adding the
  packed conv1d prefill launch surface that Qwen3.5 needs before model-side
  packed multi-request prefill can call into a canonical conv1d operator.

## Hypothesis

- This tranche alone should have no serving delta because the new packed conv1d
  surface is not wired into `Qwen3.5` yet.
- Once model-side integration lands, the packed conv1d path should let Qwen3.5
  share the same packed-prefill orchestration shape as the already-landed
  packed GDR surface, removing one remaining sequential linear-attention
  boundary from the current batch override.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen35-packed-conv1d-prefill \
  --target http://<remote-host>:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B \
  --trace-interval-ms 1000
```

Invoked via: `scripts/bench_guidellm.sh cuda-qwen35-packed-conv1d-prefill [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]`

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3.5-4B`
- **Hardware:** `pending-remote`
- **Commit:** `pending-remote`
- **Feature set:** `cargo build -p infer --release --bin infer`
- **Non-default flags / env vars:** `pending-remote`
- **Server launch:** `pending-remote`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-qwen35-packed-conv1d-prefill`

## Results — sweep headline table

Pending remote CUDA run after the Qwen3.5 model-side packed conv1d call site
lands.

## Problems

- This workspace is not a CUDA bench host.
- The current change is intentionally surface-only: until Qwen3.5 switches from
  per-request conv1d prefill replay to the new packed op, a remote sweep would
  only measure the unchanged baseline behavior.

## Learnings

- Qwen3.5 packed prefill unification needs both linear-attention pieces:
  packed GDR and packed conv1d.
- Recording the bench stub at surface-landing time keeps the runtime paper
  trail honest and makes the later model-side integration bench explicit.

## Δ vs baseline

- **Baseline:** [2026-04-22-bench-guidellm-qwen3-qwen35-model-side-unification.md](./2026-04-22-bench-guidellm-qwen3-qwen35-model-side-unification.md)
- Delta table: `pending-remote`

## Artefacts

- Raw: `pending-remote`
- CSV: `pending-remote`
- HTML: `pending-remote`
- Service trace (before): `pending-remote`
- Service trace (during): `pending-remote`
- Service trace (after): `pending-remote`
- Service trace (summary): `pending-remote`

## Notes

- What changed in the code since baseline:
  - added packed conv1d prefill FFI and CUDA launcher
  - added a Rust ops launch struct and wrapper mirroring the packed GDR
    surface
  - kept the numerical implementation canonical by reusing the existing
    single-request conv1d prefill kernel per packed segment
- Suspected cause of any regression: n/a until model-side integration uses the
  surface
- Follow-ups:
  - wire Qwen3.5 packed linear-attention prefill to call the new conv1d
    surface
  - run the canonical remote CUDA sweep on that integration commit
