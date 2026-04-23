# CUDA mixed retract page budget pending remote verification

## Goal

Validate the 2026-04-24 CUDA scheduler fix that makes mixed-batch decode
retract reserve the prefill row's immediate page growth instead of treating the
chunk token count as page count.

## Hypothesis

Mixed decode+prefill ticks should stay aligned with SGLang's allocator-facing
memory accounting and stop over-retracting decode rows whenever a prefill chunk
is larger than one page.

## Params

- Label: `cuda-mixed-retract-page-budget`
- Planned commands:
  - `scripts/bench_guidellm.sh cuda-mixed-retract-page-budget-qwen3`
- Planned models:
  - `Qwen/Qwen3-4B`
- Feature set: `cargo build --release`

## Env

- Local change date: `2026-04-24`
- Local machine: macOS workspace without a CUDA bench host
- Follow-up to:
  - [2026-04-23-bench-guidellm-cuda-scheduler-emit-admission-cleanup-pending-remote.md](2026-04-23-bench-guidellm-cuda-scheduler-emit-admission-cleanup-pending-remote.md)

## Results

- Status: `pending-remote`
- Local verification completed:
  - `cargo fmt --check`
  - `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- Local verification attempted but blocked by host limitations:
  - CUDA-linked `cargo test` remains unavailable on this macOS host because the
    workspace cannot link the required CUDA symbols locally.

## Problems

- No remote `guidellm` sweep or CUDA trace has run yet for this mixed retract
  budget fix, so the exact admission / throughput delta is still pending.

## Learnings

- Mixed-batch retract must budget in allocator units, not scheduler chunk
  units; otherwise even correct high-level planning can become much more
  conservative than the underlying paged pool requires.
- Keeping the page-growth conversion in one helper is simpler than relying on
  call sites to remember whether a variable is in tokens or pages.
