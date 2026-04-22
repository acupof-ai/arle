# Bench Stub — Qwen3.6 DFlash Block-Size Autotune

## Context

Local Metal regression checks on `mlx-community/Qwen3.6-35B-A3B-4bit` showed two
separate issues:

- `metal_bench --use-step-driver --dflash-draft-model ...` was not threading the
  loaded DFlash runtime into `MetalRequestState`, so the "service path" benchmark
  silently measured the non-DFlash path.
- After fixing that wiring, the real step-driver DFlash path was still slower
  than baseline because the draft checkpoint defaulted to `block_size=16`, while
  the observed acceptance for this pair was only about `4.4` accepted inputs per
  block.

This tranche fixed the bench wiring and auto-tuned the default block size for
the published `Qwen3.6-35B-A3B-DFlash` / `Qwen3.5-35B-A3B-DFlash` pair down to `2`
unless the operator passes `--speculative-tokens` explicitly.

## What Worked

- Service-path serial benchmark, baseline step-driver (latest rerun after the
  request-state / bench API cleanup):
  - `generation_tps.mean = 88.49 tok/s`
  - `repo_e2e_tps.mean = 84.57 tok/s`
- Service-path serial benchmark, DFlash step-driver before autotune (`block_size=16`):
  - `generation_tps.mean = 53.95 tok/s`
  - `repo_e2e_tps.mean = 52.39 tok/s`
  - `acceptance_rate = 77.34%`
  - `avg_accepted_inputs = 4.41` with `block_size = 16`
- Service-path serial benchmark, DFlash step-driver with explicit `--speculative-tokens 2`:
  - `generation_tps.mean = 75.14 tok/s`
  - `repo_e2e_tps.mean = 72.35 tok/s`
  - `acceptance_rate = 50.0%`
  - `avg_accepted_inputs = 2.0` with `block_size = 2`
- Service-path serial benchmark, DFlash step-driver with explicit `--speculative-tokens 3`:
  - `generation_tps.mean = 74.95 tok/s`
  - `repo_e2e_tps.mean = 72.18 tok/s`
  - `acceptance_rate = 57.03%`
  - `avg_accepted_inputs = 2.33` with `block_size = 3`
- Service-path serial benchmark, DFlash step-driver after the default autotune
  landed (latest rerun, no explicit override):
  - `generation_tps.mean = 77.40 tok/s`
  - `repo_e2e_tps.mean = 74.50 tok/s`
  - `acceptance_rate = 50.0%`
  - `avg_accepted_inputs = 2.0` with `block_size = 2`
- Single-request serial benchmark, DFlash direct path after the default
  autotune landed (no explicit override):
  - `generation_tps.mean = 75.26 tok/s`
  - `repo_e2e_tps.mean = 65.52 tok/s`
- One-token local smoke after the autotune:
  - log printed `Metal DFlash auto-tuning block_size ... 16 -> 2`
  - log printed `Metal DFlash enabled ... block_size=2`
  - request completed with `exit 0`
- Long-output synthetic service-path rerun (`prompt_tokens=20`,
  `generation_tokens=1024`, `runs=1`):
  - baseline step-driver: `generation_tps = 80.69 tok/s`,
    `repo_e2e_tps = 80.25 tok/s`, `ttft_ms = 68.28`
  - DFlash step-driver: `generation_tps = 69.05 tok/s`,
    `repo_e2e_tps = 68.72 tok/s`, `ttft_ms = 69.35`,
    `acceptance_rate = 50.0%`, `avg_accepted_inputs = 2.0`
- Long-output local reasoning prompt rerun via `metal_request`
  (`"How many positive whole-number divisors does 196 have? Solve it step by step."`,
  ChatML prompt, `max_new_tokens=512`, `ignore_eos=true`):
  - baseline: `Prompt tokens = 28`, `Gen TPS = 83.0 tok/s`,
    `TTFT = 83.0 ms`, `Total time = 6253.5 ms`
  - DFlash: `Prompt tokens = 28`, `Gen TPS = 58.5 tok/s`,
    `TTFT = 394.5 ms`, `Total time = 9148.4 ms`
  - The model emitted `<think>...</think>` in both runs, so the local gap is
    not explained by a totally non-reasoning output distribution alone.

## Rule

Status: `pending-remote`

Local serial data says the A3B draft's shipped `block_size=16` is too large for
the current Metal runtime. Keep the pair-specific default at `2` until a remote
guidellm sweep proves a better default or a runtime/kernel change shifts the
acceptance curve materially. Matching the public `2.4x-2.9x` Qwen3.6 numbers
will require more than longer outputs: the local Metal lane still sits at about
`2.0` accepted inputs per speculative round, far below the public B16
acceptance range, and the single-request Metal DFlash path still carries much
higher prompt/TTFT overhead than the baseline path.
