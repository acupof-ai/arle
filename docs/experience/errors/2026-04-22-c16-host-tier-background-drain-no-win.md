# C16 host-tier background drain does not beat current head

## Context

- Goal: test whether decoupling T1 host-pinned drain from the existing GPU reclaim trigger could recover more host-tier headroom and improve canonical `Qwen3-4B` L4 `c16` throughput.
- Canonical envelope stayed fixed:
  - `--num-slots 16`
  - `--max-seq-len 4608`
  - `--mem-fraction-static 0.94`
  - `--chunked-prefill-size 4096`
  - `--max-prefill-tokens 16384`
- Local verification before each run:
  - `cargo fmt --all`
  - `cargo test --release -p infer --lib scheduler::cuda:: -- --nocapture`
  - `cargo build --release -p infer --bin infer`
- Relevant c16 artefacts from this investigation:
  - pure current-head control: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-f969507-head-control`
  - per-tick background-drain attempt: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-f969507-host-tier-background-drain`
  - naive idle-store-wait attempt: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-f969507-idle-store-wait`
  - fixed idle-wait + per-tick drain attempt: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-f969507-host-drain-idlewait-fix`

## Root Cause

- This optimization line did **not** produce a win on the current tree/host.
- The first important control is that pure `HEAD` on `f969507` now measures:
  - `c16 out tok/s = 118.24`
  - `TTFT p50 = 7360.8 ms`
  - `ITL p50 = 59.93 ms`
- So the older `123.38 tok/s` anchor from `18c116d` is **not** the right local control for this turn.

Two attempted changes were evaluated:

1. **Naive idle store wait**
   - Change: let `store_waiting` use the same coordinator-wait path as fetch-waited requests.
   - Result: `103.84 tok/s`.
   - Root cause: the first implementation waited on coordinator events whenever any store ticket was outstanding, which blocked active work as well, not just idle drain.

2. **Per-tick background drain after cleanup**
   - Change: call host-tier drain once per scheduler tick after `cleanup()`, with the idle-wait bug fixed.
   - Result: `118.01 tok/s`.
   - Root cause: once the wait bug was fixed, the background drain no longer catastrophically stalled the loop, but it still did not beat the pure-head control; on this tree it was effectively neutral-to-slightly-worse.

So the host-tier issue here is not solved by “run drain more often”. The current tree still spends time in large reclaim waves, but forcing an extra host-tier scan/submission pass every tick does not improve end-to-end throughput on this workload.

## Fix

- Do **not** ship the per-tick background-drain path.
- Keep the worktree on pure `HEAD` rather than landing a runtime change with no measured benefit.
- Treat this line as closed until there is stronger observability on actual store submission/completion volume or a different reclaim policy hypothesis.

## Rule

- When a scheduler optimization is intended to improve throughput, always re-establish a **same-tree, same-host control** before comparing against an older historical win.
- If a new host-tier policy does not beat the current pure-head control, do not ship it just because it looks architecturally cleaner.
- Be careful with “wait on coordinator” changes: any store/fetch wait path must prove it only blocks when the scheduler is otherwise idle or explicitly fetch-bound.
