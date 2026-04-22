# Bench Stub — Qwen3.6 DFlash Restore B16 Default

## Context

The local Metal lane had temporarily auto-tuned
`z-lab/Qwen3.6-35B-A3B-DFlash` / `z-lab/Qwen3.5-35B-A3B-DFlash` down from the
published draft default `block_size=16` to `2` because local Apple Silicon
benchmarks showed better throughput at `2`.

This tranche removes that local A3B-specific override and restores the shipped
draft default so Metal matches the public DFlash configuration and predicts 16
tokens per speculative round unless the operator passes `--speculative-tokens`
explicitly.

## What Worked

- The Metal DFlash loader now follows the draft config default again for the
  A3B drafts; no pair-specific auto-tune branch remains in `dflash.rs`.
- The user-facing Metal DFlash doc no longer claims an A3B-only auto-tuned
  default of `2`.
- Comparison table, local Apple Silicon serial runs:

| workload | mode | block_size | gen tok/s | repo e2e tok/s | TTFT ms | avg accepted inputs | acceptance rate |
|---|---|---:|---:|---:|---:|---:|---:|
| step-driver, prompt=20, gen=128 | baseline | n/a | 88.49 | 84.57 | 66.96 | n/a | n/a |
| step-driver, prompt=20, gen=128 | DFlash autotuned | 2 | 77.40 | 74.50 | 64.43 | 2.00 | 50.0% |
| step-driver, prompt=20, gen=128 | DFlash restored default | 16 | 53.15 | 51.49 | 77.68 | 4.41 | 77.34% |
| step-driver, prompt=20, gen=1024 | baseline | n/a | 80.69 | 80.25 | 68.28 | n/a | n/a |
| step-driver, prompt=20, gen=1024 | DFlash autotuned | 2 | 69.05 | 68.72 | 69.35 | 2.00 | 50.0% |
| metal_request, math reasoning prompt, gen=512 | baseline | n/a | 83.0 | n/a | 83.0 | n/a | n/a |
| metal_request, math reasoning prompt, gen=512 | DFlash autotuned | 2 | 58.5 | n/a | 394.5 | n/a | n/a |

- Local smoke:
  - `metal_request` logged
    `Metal DFlash enabled: ... block_size=16 ...`
  - request completed with `exit 0`
- Local service-path serial benchmark (`prompt_tokens=20`,
  `generation_tokens=128`, `runs=1`):
  - DFlash step-driver: `generation_tps = 53.15 tok/s`
  - `repo_e2e_tps = 51.49 tok/s`
  - `ttft_ms = 77.68`
  - `acceptance_rate = 77.34%`
  - `avg_accepted_inputs = 4.41` with `block_size = 16`
- The existing local serial data that motivated the temporary override remains
  recorded in
  [2026-04-22-bench-guidellm-qwen36-dflash-blocksize-autotune.md](2026-04-22-bench-guidellm-qwen36-dflash-blocksize-autotune.md)
  and should be used as the local regression reference for this restored-B16
  behavior.

## Rule

Status: `pending-remote`

Restoring the published B16 default is a configuration-alignment change, not a
proof of local speedup. Keep using explicit local regression checks on Apple
Silicon until a remote benchmark sweep confirms that matching the public DFlash
configuration also improves the Metal lane.
