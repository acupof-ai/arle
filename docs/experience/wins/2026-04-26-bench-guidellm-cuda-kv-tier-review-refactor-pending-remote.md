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

- All Wave-1/2/3 changes are pure refactors with no algorithmic delta:
  - **Wave 1**:
    - backend.rs: macro-collapsed dispatch is identical to manual match
      (compiler should produce the same code; only difference is the
      eliminated unreachable runtime error arm)
    - scheduler/cuda/policy.rs: file move only, no type/method changes
    - host_pool.rs `with_region_slice`: removes one `Vec<u8>` alloc + memcpy
      per readmitted block on the H2D promote path. **Should be neutral or
      very slightly faster** at high readmission rates; not measurable on a
      cold sweep without prior staged blocks.
  - **Wave 2**: CoordinatorBuilder (telescoping ctor → builder), typed
    FailureClass, AllocatedRegions RAII guard, three `*_failed` methods
    collapsed to one `report_failure`. All structural; zero behavioral
    delta. The new `with_region_slice` slice-API shaves one alloc per
    readmission promote — same direction as Wave 1 §host_pool.
  - **Wave 3**: pure file split (coordinator.rs 2132 → 623 lines, 5 new
    coordinator/* files). Zero code change, only relocation + visibility
    narrowing.

  Expected Δ on canonical Qwen3 sweep: **0% TTFT / 0% ITL / 0% out-tok-s
  within run-to-run noise (≤1.5%)**.

## Local Metal sanity-check (regression-only, 2026-04-26)

Per the user request to validate locally as well, ran the Metal lane
end-to-end after the full refactor (commits 64e350c..a94682a). Note
that **Apple Silicon skips T1** (per `kv_tier/AGENTS.md`) — the Metal
serve path doesn't exercise the refactored coordinator/host_pool hot
paths, only their always-on type compilation. So the Metal evidence
proves the refactor doesn't break Metal **compile + serve**, not that
the CUDA hot paths are perf-equivalent.

### Codex review

```bash
codex review --base cae7e05    # Wave 1
codex review --base 99b7bcb    # Wave 2 + 3
```

Wave-1 verdict: *"No actionable correctness issues were found in the
changes relative to the specified base. The touched feature
combinations type-check in the available local no-CUDA/Metal lanes."*

### Lib test sweep (Metal)

```bash
cargo test -p infer --release --no-default-features --features metal --lib
```

Result: **475 passed; 0 failed; 19 ignored** (all kv_tier 57 + coordinator
17 included; zero new test failures across the wave-1/2/3 commit range).

### metal_bench Qwen3-0.6B bf16, 1024 prompt + 128 gen, 2 warmup + 3 timed runs

```bash
./target/release/metal_bench --model models/Qwen3-0.6B \
  --prompt-tokens 1024 --generation-tokens 128 --warmup 2 --runs 3
```

| metric | mean | p50 | p99 |
|---|---:|---:|---:|
| Prompt speed (tok/s) | 6435.4 | 6435.1 | 6440.1 |
| Generation (tok/s) | 140.2 | 140.2 | 140.4 |
| TTFT (ms) | 159 | 159 | 159 |
| Total wall (ms) | 1072 | 1072 | 1074 |
| Repo E2E (tok/s) | 119.4 | 119.4 | 119.6 |
| Peak RSS | 1598 MB | — | — |

Per-run variance < 0.1% — Metal lane is stable post-refactor. This is
not a Δ baseline (no pre-refactor Metal snapshot at these exact params),
just a "Metal still works" regression check.

### What was NOT validated locally

- The actual CUDA `kv_tier::coordinator` hot paths (Apple Silicon skips
  T1; CUDA bench remains pending-remote).
- The user's in-progress Qwen3.5 work (`metal/qwen35.rs`,
  `mlx-sys/mlx_qwen35_model.cpp`) — `metal_bench --use-step-driver
  --model models/Qwen3.5-0.8B` triggered a GPU Hang in the user's WIP
  paths; those changes are uncommitted and unrelated to this refactor.

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
