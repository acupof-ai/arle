# kv_tier review-driven refactor — guidellm sweep, cuda, 2026-04-26

> Status: **pending-remote** — refactor is CUDA-lane code (`kv_tier::` +
> `scheduler::cuda::`); local Mac runs Metal, can't exercise this path.
> Bench will be filled in by the next remote-CUDA run.

## Goal

- Regression-check that the kv_tier code-review refactor (Wave 1: backend
  enum dispatch collapse, policy.rs relocation, host_pool zero-copy slice
  API + runtime promote-path migration) has **zero** measurable impact on
  serving throughput. Goal type: regression-check minimum (per
  `bench-and-trace-spec.md` §7).

## Hypothesis

- All three Wave-1 changes are pure refactors with no algorithmic delta:
  - backend.rs: macro-collapsed dispatch is identical to manual match
    (compiler should produce the same code; only difference is the
    eliminated unreachable runtime error arm)
  - scheduler/cuda/policy.rs: file move only, no type/method changes
  - host_pool.rs `with_region_slice`: removes one `Vec<u8>` alloc + memcpy
    per readmitted block on the H2D promote path. **Should be neutral or
    very slightly faster** at high readmission rates; not measurable on a
    cold sweep without prior staged blocks.

  Expected Δ on canonical Qwen3 sweep: **0% TTFT / 0% ITL / 0% out-tok-s
  within run-to-run noise (≤1.5%)**.

## Command

```bash
scripts/bench_guidellm.sh cuda-kv-tier-review-refactor
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** TBD (whichever remote CUDA box runs the sweep)
- **Commit:** TBD (range covering the three Wave-1 commits + any subsequent
  Wave-2/Wave-3 batches that land in the same review-driven refactor)
- **Feature set:** `cargo build --release` (default cuda)
- **Non-default flags / env vars:** none
- **Server launch:** `scripts/start_infer.sh Qwen/Qwen3-4B 8000`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — pending

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | — | — | — | — | — | — |

## Δ vs baseline — pending

Baseline: most recent CUDA Qwen3-4B guidellm sweep before this batch
landed. Expect ≤1.5% Δ on every column (regression-check bar).

## Problems / Learnings

- Refactor was driven by a code-quality review (this conversation) — not a
  perf optimization. The bench is a *regression check*, not an A/B.
- If any column moves >1.5%, the relevant Wave-1 commit should be bisected
  immediately; none of these changes should plausibly move numbers.
- Wave 2 (coordinator.rs: builder, error class, RAII guard) and Wave 3
  (file split) will land on top before this stub gets filled. Cite the
  full commit range when running.

## Rule

- The kv_tier surface is touched by every readmission and store on the
  CUDA scheduler. Any refactor here is in-scope for the
  `feedback_bench_every_change.md` rule, even if the diff is structural.
