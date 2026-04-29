# Trajectory token-ids (Phase 2) regression check — guidellm sweep, pending-remote, 2026-04-29

> Status: **`pending-remote`**. This entry is a stub opened per CLAUDE.md
> §Benchmarks — every runtime change requires a wins entry, and the
> Phase 2 trajectory token-IDs work touches `infer/src/` hot paths (Metal
> scheduler runtime emits per-delta `token_ids`, the
> `RequestHandleInferenceEngine::complete` collator now tokenizes the
> prompt up-front). The implementation surface has no GPU-resident
> hot loop changes — but it does add per-request `tokenize()` calls and
> per-token Vec pushes inside the Metal runtime — so a regression check
> against the most recent baseline is required before declaring "done".
> The runner is **not this Mac** (cannot drive guidellm + a real Qwen3-4B
> server in this autonomy session); the next remote-host runner who
> picks this up should fill in the result tables and clear the
> `pending-remote` flag.

## Goal

- Confirm the Phase 2 trajectory token-IDs change does not regress
  TTFT, ITL, or saturation throughput on the canonical guidellm sweep.

## Hypothesis

- Neutral. The hot-path additions are: one extra `tokenize()` per
  `complete()` (already paid by the scheduler before submission, just
  in a different place), and `Vec<u32>` pushes per sampled token in
  `ActiveMetalRequest::process_token`. Both should sit in the noise
  floor of a 60-second sweep — i.e. Δ% within ±2 % on TTFT p50 / ITL
  p50 / out tok/s @ saturation.

## Command

```bash
scripts/bench_guidellm.sh trajectory-token-ids-after \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

## Environment

- **Backend:** metal (primary regression-check target — Mac is the
  development host); CUDA mirror to follow on a CUDA host.
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `<fill on remote host — M-series SoC + VRAM, or H100 / A100>`
- **Commit:** Phase 2 = `<short SHA when shipped>` (Phase 1 baseline is
  `6e9c5e3` — see `docs/experience/wins/` for the most recent
  guidellm-metal entry close to that commit).
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda` (Metal) /
  `cargo build --release --features cuda` (CUDA).
- **Non-default flags / env vars:** none.
- **Server launch:** `scripts/start_infer.sh Qwen/Qwen3-4B 8000`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh trajectory-token-ids-after`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| synchronous | _pending-remote_ | _pending-remote_ | _pending-remote_ | _pending-remote_ | _pending-remote_ | _pending-remote_ |
| saturation | _pending-remote_ | _pending-remote_ | _pending-remote_ | _pending-remote_ | _pending-remote_ | _pending-remote_ |

## Results — service-side KV / scheduler metrics

_pending-remote_

## Results — request accounting

_pending-remote_

## Problems

- _pending-remote_ — none expected; the change is observability-side and
  doesn't touch attention / prefill / decode kernels.

## Learnings

- _pending-remote_

## Δ vs baseline

- **Baseline:** the most recent `*-bench-guidellm-metal-*.md` entry
  before Phase 2 ships (the runner picks the closest pre-Phase-2
  Metal sweep — Phase 1 ships at `6e9c5e3` 2026-04-28; the runner
  should locate the corresponding before snapshot under
  `docs/experience/wins/`).
- **Delta table:** _pending-remote_ — runner fills both columns and
  the Δ% column. ±2 % is acceptable; outside that, open an entry
  under `docs/experience/errors/` and surface in the next bench.

## Artefacts

- _pending-remote_ — paths follow the canonical
  `bench-output/2026-MM-DD-trajectory-token-ids-after/...` layout.

## Notes

- **Trigger:** Phase 2 trajectory token-ids work
  (`docs/projects/agent-trajectory-export.md` Phase 2). Touched files:
  `infer/src/server_engine/{types,backend_engine,request_handle_engine,loaded}.rs`,
  `infer/src/backend.rs`, `infer/src/backend/{metal.rs,cpu.rs}`,
  `infer/src/backend/metal/runtime.rs`,
  `infer/src/backend/runtime.rs`,
  `infer/src/scheduler/cuda/{request.rs,core/emit_worker.rs,runtime/helpers.rs}`,
  `infer/src/http_server/types.rs`, `crates/agent/src/lib.rs`,
  `crates/cli/src/trace.rs`. The diff is observability-side (per-delta
  `token_ids: Vec<u32>` + cumulative `prompt_token_ids` /
  `response_token_ids` on `CompletionOutput`); no kernel changes.
- **Next runner:** a CUDA host operator (or Metal host operator
  if Mac M2/M3 is the SUT). When you pick this up, run the canonical
  `scripts/bench_guidellm.sh trajectory-token-ids-after` against the
  current `main`, fill in every `_pending-remote_` cell, drop the
  status flag from the title, and cross-link this entry from
  `docs/projects/agent-trajectory-export.md` (Phase 2 status section).
- **Why pending-remote:** this autonomy session was on a Mac without
  the model assets staged + without a guidellm server harness running.
  Per CLAUDE.md §Benchmarks: "If the bench can't run locally … the
  commit body MUST cite the remote-machine ticket or plan entry that
  will execute it, and the entry is opened as a stub under `wins/`
  with status `pending-remote`. No silent skips." This file is that
  stub.
