# M3 Decode-Slot Recovery Fix

## Context

`codex review --base 788b15a` found one M3 CUDA lowering regression: a
recoverable malformed decode slot (`Phase::Decoding` with empty
`generated_tokens`) could be skipped by logical IR lowering instead of flowing
through the legacy cleanup path.

## What Worked

CUDA logical launch now falls back to the legacy step launcher when a
decode-bearing `StepPlan` lowers to zero logical decode rows. That preserves the
existing `collect_decode_batch_inputs()` behavior: log the malformed request and
call `finish_slot()` so the scheduler does not keep an unlaunchable slot in
`running_batch`.

## Verification

| command | result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo check -p infer --no-default-features --features no-cuda` | pass |
| `NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 cargo check -p infer --features cuda` | pass |
| `bun run /home/ckl/.bun/bin/codex review --base 788b15a -c sandbox.timeouts.exec_seconds=900` | found this issue; no additional M3 code finding was reported before the fix |

## Bench Status

GuideLLM perf remains blocked by the pre-existing c=8 stuck-active reproducer
documented in
`docs/experience/errors/2026-05-07-m3-guidellm-bench-stuck.md`. The same stuck
shape reproduces at `788b15a`, before M3 logical IR was connected, so this fix
does not have a trustworthy local tok/s delta yet.

## Rule

When a logical plan is a lowering of an existing scheduler plan, empty lowered
rows must preserve the legacy recovery semantics for the skipped work. Falling
through to a no-op is only correct for a true idle plan.
