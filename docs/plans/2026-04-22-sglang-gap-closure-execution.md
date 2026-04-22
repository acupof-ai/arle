# SGLang Gap Closure — Execution Plan

**Status:** code landed locally, remote CUDA bench pending (2026-04-22)  
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

Benchmark shape that commissioned this plan:

- `c1`: decode hot path is already near parity
- `c2`: small deficit, consistent with remaining scheduling/prefill overhead
- `c4/c8/c16`: throughput collapses while TTFT explodes

That pointed at one bottleneck cluster:

- Qwen3.5 still fell back to contiguous prefill
- mixed prefill+decode still carried legacy branches
- scheduler overlap was only partially landed
- prefill graph capture was not on the canonical path
- admission was still less deterministic than the SGLang/vLLM-style token+KV budget

## Non-goals

- new remote KV transports
- speculative decode changes
- INT8/FP8 paged-prefill kernels
- Metal parity for the CUDA-only reshaping here

## Current truth after landing

- `Qwen3` still uses paged prefill on the canonical path.
- `Qwen3.5` now ships through paged prefill on the canonical path:
  `prefill_uses_paged_pool() == true`.
- The old mixed prefill+decode legacy surface is gone:
  - no shipped `step_decode_launch_mixed`
  - no shipped dual-write helper path
  - no mixed-only scheduler flag
- CUDA scheduler overlap now has:
  - cross-iteration `pending_decode` launch/readback split
  - event-driven fetch-wait sleep instead of `recv_timeout(2ms)` polling
- Deterministic admission now has:
  - full-ISL reservation in `assign_slots()`
  - explicit prefill planner `score -> fit` structure
  - one canonical mutable prefill token budget
- Qwen3.5 paged-prefill full-forward graph is on the canonical path, with one
  remaining correctness guard: replay is invalidated when the captured
  `start_pos` changes because that scalar is still baked into the prep-kernel
  launch parameters.

## Landed tracks

### Track A — Qwen3.5 paged prefill + prefill graph

**Outcome**

- `Qwen3.5` writes full-attention KV directly into the paged pool on the
  canonical prefill path
- the contiguous prefill fallback is no longer the default path
- paged-prefill full-forward graph capture/replay is wired onto the canonical
  prefill path
- graph replay is invalidated when the page-index buffer reallocates or the
  captured `start_pos` changes

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

### Track B — MixedBatch legacy deletion

**Outcome**

- one canonical mixed/pre/decode scheduler surface
- `step_decode_launch_mixed` deleted
- the old dual-write helper is gone from the shipped path
- scheduler step chooses one `decode / prefill / decode+prefill` plan shape

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
- prefill planning uses an explicit `score -> fit` structure
- full-ISL reservation is enforced on admission
- the mutable prefill token budget is canonicalized to one value:
  `min(max_num_batched_tokens - running_decode_rows, max_prefill_tokens)`

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

## Delivered commits

- `796de61` `refactor(qwen35): delete paged-prefill hd256 shadows`
- `e4554cd` `refactor(scheduler): delete mixed-batch legacy surface`
- `5c8aa81` `feat(qwen35): reuse paged-prefill graph after first chunk`
- `e99be66` `feat(scheduler): make fetch waits event-driven`
- `50ab021` `refactor(scheduler): canonicalize prefill planner budget`
- `5dfde31` `fix(qwen35): guard paged-prefill graph start-pos reuse`

## Parallelization shape used

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
- explicit prefill `score -> fit` and full-ISL reservation are both live on the CUDA scheduler

### Validation

Minimum local validation executed across the landed tranches:

- `cargo check -p infer --release --no-default-features --features no-cuda`
- `cargo check -p infer --release --no-default-features --features metal,no-cuda`
- `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- `cargo test -p infer --release --no-default-features --features no-cuda scheduler -- --nocapture`
- `cargo test -p infer --release --no-default-features --features no-cuda qwen35 -- --nocapture`
- `cargo clippy -p infer --release --no-default-features --features no-cuda -- -D warnings`

CUDA remote follow-up is still mandatory for any performance claim:

- `scripts/bench_guidellm.sh` before/after snapshot
- win entry under `docs/experience/wins/`
- if remote machine is required, commit a `pending-remote` stub immediately

## Bench expectations

Expected order of impact after local landing:

1. Track A removes the Qwen3.5 contiguous prefill cap and should move TTFT first
2. Track B removes mixed-path duplication and should move `c4+` throughput
3. Track C reduces scheduler-side bubbles and admission thrash, improving both TTFT and tail throughput

## Remaining external-only work

All in-repo code work for these five items is landed. Remaining work is
external-only:

- non-local CUDA before/after `scripts/bench_guidellm.sh` parity sweep
- delta table vs SGLang for `c1/c2/c4/c8/c16`
- fix any review findings or remote-regression findings that the CUDA run
  surfaces
