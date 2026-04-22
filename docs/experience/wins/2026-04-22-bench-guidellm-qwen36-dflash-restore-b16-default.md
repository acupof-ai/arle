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
