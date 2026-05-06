# M3 Unified Scheduler IR

## Context

Track B M3 landed the shared logical scheduler IR migration described in
`docs/plans/M3-unified-schedule-ir.md` S1-S6.

Implemented commits:

- `e289520 feat(scheduler): add unified logical plan IR`
- `c463dd2 refactor(metal): reuse unified scheduler plan IR`
- `dd48e5f feat(cuda): shadow logical scheduler plan`
- `99fae49 feat(cuda): lower happy path through logical scheduler plan`
- `620eddb refactor(metal): lower schedule step from logical plan`
- `a9f0327 refactor(scheduler): retire legacy batch decision enum`

## What Worked

- Added backend-neutral `LogicalServePlan` IR in `infer/src/scheduler/plan.rs`.
- Replaced Metal's duplicate local plan schema with aliases to the shared IR.
- Added CUDA shadow emission from the existing `StepPlan` path.
- Put CUDA happy-path lowering behind the default-on `unified_scheduler` Cargo feature.
- Kept CUDA spec-decode on the legacy path.
- Retired the old `scheduler::batch::ScheduleDecision` enum and moved CPU scheduler tests to the shared IR.

## Verification

Environment:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB
- Driver: 595.71.05
- CUDA: 13.2.78
- Model: `infer/models/Qwen3-4B`
- CUDA env: `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python`

Passing gates:

| command | result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo test --release -p infer --no-default-features --features no-cuda scheduler::` | pass, 91 tests |
| `cargo check -p infer --features cuda` | pass |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | pass |
| `cargo check -p infer --no-default-features --features metal,no-cuda` | pass |
| `cargo clippy -p infer --features cuda -- -D warnings` | pass |
| `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings` | pass |
| `cargo test --release -p infer --features cuda --test greedy_consistency` | pass |
| `cargo test --release -p infer --features cuda --test e2e` | pass |
| `cargo test --release -p infer --features cuda --test spec_decode_correctness` | pass, 4 tests |

Blocked gates:

| command | result |
|---|---|
| `cargo test --release -p infer --features cuda --test e2e_qwen35` | fail: Qwen3.5 greedy baseline mismatch |
| `cargo test --release -p infer --no-default-features --features cuda --test e2e_qwen35` | same fail; not caused by `unified_scheduler` feature |
| `scripts/bench_guidellm.sh cuda-m3-unified ...` | blocked; see `docs/experience/errors/2026-05-07-m3-guidellm-bench-stuck.md` |

## Line Delta

Measured against `788b15a`:

- M3 code paths: `827 insertions(+), 439 deletions(-)`, net `+388`.
- `infer/src/scheduler/cuda/core.rs`: `1742 -> 1742`, delta `0`.
- `infer/src/backend/metal/scheduler.rs`: `1015 -> 1017`, delta `+2`.
- Combined `core.rs + metal/scheduler.rs`: `2757 -> 2759`, delta `+2`.

The original rough acceptance target of `>=800` deleted lines in those two files was not met by the accepted S1-S7 design path. The actual landed milestone unifies the IR and removes duplicate decision schemas; it does not yet collapse the two production scheduler loops into one shared CPU policy call.

## Bench Status

Canonical guidellm did not produce a trustworthy result table on this machine:

- First run failed preflight because `guidellm` was missing from PATH.
- After installing `.[bench]`, the next run failed because the environment configured a SOCKS proxy and `httpx` needed `socksio`.
- After installing `socksio`, the canonical 4096-in/256-out run sent prompts tokenized as 4097 tokens, which the server rejected with `scheduler max_input=4090 max_request=4095`.
- A fallback `--quick` run progressed to c=8 but ended stuck at `active=4 waiting=8 scheduled=0 decode_rows=0 running_batch=4`.

Raw artifacts:

- `bench-output/2026-05-07-cuda-m3-unified/`
- `bench-output/2026-05-07-cuda-m3-unified-run2/`
- `bench-output/2026-05-07-cuda-m3-unified-run3/`

Metal bench is `pending-remote`; this Linux CUDA box has no Metal runner.

## Rule

Treat M3 as an IR convergence tranche, not a completed scheduler-loop deletion. Before claiming perf parity, rerun fixed-concurrency guidellm after resolving the c=8 stuck-active reproducer and the canonical 4096-token tokenizer/server limit mismatch.
