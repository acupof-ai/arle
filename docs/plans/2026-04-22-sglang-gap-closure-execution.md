# SGLang Gap Closure — Execution Plan

**Status:** active execution plan (2026-04-22)  
**Commissioned by:** benchmark gap sweep (`c1/c2/c4/c8/c16`) + repo-grounded `nlm` review  
**Complements:** [`p99-unified-mixed-batch.md`](p99-unified-mixed-batch.md), [`qwen35-single-graph-prefill.md`](qwen35-single-graph-prefill.md), [`scheduler-gpu-cpu-overlap.md`](scheduler-gpu-cpu-overlap.md)

## Goal

Close the current high-concurrency CUDA gap against SGLang by finishing the
five load-bearing runtime items end to end:

1. Qwen3.5 paged prefill becomes the shipped default path
2. MixedBatch becomes the only mixed prefill+decode path
3. scheduler overlap reaches the full intended launch/prepare/readback shape
4. Qwen3.5 prefill full-forward graph becomes real, not just planned
5. admission becomes deterministic via two-pass budget + full-ISL reservation

This plan is execution-first. Documentation sync is required, but it does not
count as completion unless the runtime behavior is landed and validated.

## Why this is the right cut

Current benchmark shape:

- `c1`: decode hot path is already near parity
- `c2`: small deficit, consistent with remaining scheduling/prefill overhead
- `c4/c8/c16`: throughput collapses while TTFT explodes

That points at one bottleneck cluster:

- Qwen3.5 still falls back to contiguous prefill
- mixed prefill+decode still carries legacy branches
- scheduler overlap is only partially landed
- prefill graph capture is not on the canonical path
- admission is still less deterministic than the SGLang/vLLM-style token+KV budget

## Non-goals

- new remote KV transports
- speculative decode changes
- INT8/FP8 paged-prefill kernels
- Metal parity for the CUDA-only reshaping here

## Current truth before execution

- `Qwen3` already uses paged prefill on the canonical path
- `Qwen3.5` has paged-prefill code but returns `prefill_uses_paged_pool() = false`
- mixed prefill+decode still uses `step_decode_launch_mixed` plus the old
  dual-write helper
- decode overlap has a first tranche landed, but not the full zero-overhead
  scheduler shape
- decode graphing exists; prefill full-forward graph is still separate future work

## Execution tracks

### Track A — Qwen3.5 paged prefill + prefill graph

**Outcome**

- `Qwen3.5` writes full-attention KV directly into the paged pool on the
  canonical prefill path
- the contiguous prefill fallback is no longer the default path
- prefill full-forward graph capture/replay is wired onto the canonical prefill
  path for the supported shape bucket policy

**Primary files**

- `infer/src/model/qwen35/forward.rs`
- `infer/src/model/qwen35/prefill.rs`
- `infer/src/model/qwen35/batch_decode.rs`
- `infer/src/model/cuda_graph.rs`
- `infer/src/scheduler/cuda/prefill.rs`

**Delete-first rules**

- do not keep a second "temporary" scheduler route after paged prefill is re-enabled
- if a guard remains, it must be a correctness guard with a narrow condition,
  not a silent global fallback

### Track B — MixedBatch unification

**Outcome**

- one canonical mixed prefill+decode contract
- `step_decode_launch_mixed` deleted
- Qwen3 mixed path no longer calls the dual-write prep helper
- scheduler step chooses one mixed/pre/decode path through the same batch-shape contract

**Primary files**

- `infer/src/scheduler/cuda/execution.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/model.rs`
- `infer/src/model/qwen3/batch_decode.rs`
- `infer/src/model/qwen35/batch_decode.rs`
- `infer/src/model/glm4/batch_decode.rs`

**Delete-first rules**

- remove old mixed launch branches rather than layering adapters
- remove the legacy dual-write helper from the shipped path

### Track C — Scheduler overlap + deterministic admission

**Outcome**

- launch/prepare/readback is the only decode scheduling shape
- scheduler no longer busy-spins when work is fetch-wait bound
- admission uses an explicit two-pass token budget
- full-ISL reservation is enforced on admission

**Primary files**

- `infer/src/scheduler/cuda/runtime.rs`
- `infer/src/scheduler/cuda/execution.rs`
- `infer/src/scheduler/cuda/core.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/scheduler/types.rs`
- `infer/src/scheduler/metrics.rs`

**Delete-first rules**

- remove duplicate admission logic rather than adding a second planner
- keep `assign_slots()` as the single waiting-queue normalization/admission entry

## Parallelization shape

These tracks can execute in parallel with one integration pass:

- Track A owns Qwen3.5 prefill + graph files
- Track B owns the mixed-batch contract and model batch decode call sites
- Track C owns scheduler control flow and admission budgeting

Integration pass resolves the shared seam between:

- `execution.rs` consumer side
- model trait surface in `model.rs`
- Qwen3.5 prefill enablement from Track A

## Acceptance

### Correctness

- `Qwen3.5` shipped path reports `prefill_uses_paged_pool() == true`
- no shipped path uses `step_decode_launch_mixed`
- no shipped path uses the legacy dual-write helper for mixed prefill+decode
- scheduler keeps one canonical launch/prepare/readback flow
- two-pass admission and full-ISL reservation are both live on the CUDA scheduler

### Validation

Minimum local validation:

- `cargo check -p infer --release --no-default-features --features no-cuda`
- `cargo check -p infer --release --no-default-features --features metal,no-cuda`
- `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- `cargo test -p infer --release --no-default-features --features no-cuda scheduler -- --nocapture`
- `cargo test -p infer --release --no-default-features --features no-cuda qwen35 -- --nocapture`
- `cargo clippy -p infer --release --no-default-features --features no-cuda -- -D warnings`

CUDA remote follow-up is mandatory for any performance claim:

- `scripts/bench_guidellm.sh` before/after snapshot
- win entry under `docs/experience/wins/`
- if remote machine is required, commit a `pending-remote` stub immediately

## Bench expectations

Expected order of impact:

1. Track A removes the Qwen3.5 contiguous prefill cap and should move TTFT first
2. Track B removes mixed-path duplication and should move `c4+` throughput
3. Track C reduces scheduler-side bubbles and admission thrash, improving both TTFT and tail throughput

## Exit condition

This execution line is not done until:

- code is landed
- a non-trivial diff review pass is run
- review findings are fixed
- local validation is green
- bench entry is committed

