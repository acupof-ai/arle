# Infer observability v1 — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the observability v1 rollout: always-on scheduler metrics,
  sampled request tracing hooks, and bench-anchored `nsys`/`ncu` wrappers
  should land without a material serving regression when tracing sinks stay off
  by default.

## Hypothesis

- With tracing exporters disabled, the new request-span plumbing should be
  effectively inert and the only steady-state cost should come from the
  already-landed low-overhead metrics counters.

## Command

```bash
scripts/bench_guidellm.sh cuda \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

Status: `pending-remote` (CUDA bench host required).

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** pending remote CUDA host
- **Commit:** `b862c2c`
- **Feature set:** `cargo build --release --no-default-features --features cuda,no-cuda`
- **Non-default flags / env vars:** none for the regression bench; tracing sinks
  remain disabled by default
- **Server launch:** pending remote validation

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda`

## Results — sweep headline table

Pending remote run.

## Problems

- This workspace is not a CUDA bench host, so the canonical `guidellm` sweep
  and the paired `nsys` / `ncu` captures could not be executed locally.
- A local CUDA `cargo build --release --features cuda,no-cuda` still fails at
  link time on macOS because the machine is missing the CUDA stub/link surface
  (`cublas_init`, Triton AOT symbols). Per repo policy, the local CUDA gate on
  this host remains `cargo check -p infer --no-default-features --features cuda,no-cuda`.

## Learnings

- For observability work in this repo, the right default is still
  `metrics always on, tracing sampled/off by default, profilers bench-anchored`
  rather than always-on file/OTLP export.
- The smooth operator path is `bench_guidellm -> profile_nsys_guidellm.sh /
  profile_ncu_guidellm.sh`, not hand-written profiler invocations.

## Δ vs baseline

- **Baseline:** first run for this observability tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: completed observability v1 across
  `L1 metrics`, `L2/L3` profiling wrappers, and `L4` sampled tracing/export
  plumbing
- Suspected cause of any regression: if one appears, first suspect request-span
  creation on sampled traces or any accidentally enabled file/OTLP sink
- Follow-ups: run the remote CUDA sweep, then pair it with
  `scripts/profile_nsys_guidellm.sh cuda-observability-v1` and one targeted
  `scripts/profile_ncu_guidellm.sh ... --family attention`
