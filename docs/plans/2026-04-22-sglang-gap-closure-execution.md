# SGLang Gap Closure — Execution Plan

> Status (2026-04-23): **Historical execution ledger.** This file records real landed cleanup work, but it should not be read as proof that the current CUDA decode path is already aligned with SGLang `main`. Use [`2026-04-23-cuda-decode-sglang-alignment.md`](2026-04-23-cuda-decode-sglang-alignment.md) for the current decode-focused plan.

**Status:** partially landed; in-repo runtime follow-ons still active, remote CUDA bench pending (2026-04-22)  
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
- `Qwen3.5` now also ships through paged prefill on the canonical path:
  `prefill_uses_paged_pool() == true`.
- Scheduler/admission/page sizing is mostly unified across `Qwen3` and
  `Qwen3.5`:
  - both go through the same CUDA scheduler and paged-KV admission flow
  - shipped paged/block size is `16` on the main KV formats used here
  - MixedBatch scheduling no longer forks by model family
- Model-side prefill execution is **not fully unified** yet:
  - `Qwen3` has a real packed batched paged-prefill path
- `Qwen3.5` now also has a true packed multi-request paged-prefill path:
  the scheduler-visible batch override builds one packed token/page-table
  layout, runs one packed layer loop, and uses packed recurrent-state
  launches for the hybrid linear-attention layers
- model-side batching is still not perfectly identical between `Qwen3` and
  `Qwen3.5` because `Qwen3.5` remains hybrid and therefore retains its real
  `supports_partial_prefix() == false` capability difference
- `Qwen3.5` therefore does **not** currently take the cross-slot paged-prefix
  attach / staged-readmission reuse path. Shared paged KV pages alone are not
  sufficient to reconstruct its hybrid recurrent state on a fresh slot, so the
  scheduler now routes it through same-slot stateful reuse or cold prefill
  instead of pretending the shared pages are enough.
- `Qwen3.5` also remains a real capability outlier because it is hybrid:
  `supports_partial_prefix() == false` is a model constraint, not just an
  implementation gap.
- The old mixed prefill+decode legacy surface is gone:
  - no shipped `step_decode_launch_mixed`
  - no shipped dual-write helper path
  - no mixed-only scheduler flag
- CUDA scheduler overlap now has:
  - cross-iteration `pending_decode` launch/readback split
  - event-driven fetch-wait sleep instead of `recv_timeout(2ms)` polling
  - incrementally priority-ordered waiting queue, so `assign_slots()` no
    longer sorts the whole queue every tick
  - HTTP-originated CUDA requests now arrive with cached `prompt_tokens`, so
    scheduler-side tokenization is a fallback path rather than the normal hot path
  - a dedicated emit worker that owns streaming text decode, delta emission,
    and stop-sequence scanning; the scheduler now only waits on gate results
    before the next decode launch
- Deterministic admission now has:
  - full-ISL reservation in `assign_slots()`
  - explicit prefill planner `score -> fit` structure
  - one canonical mutable prefill token budget
  - active-slot future page headroom reserved in admission, so later waiting
    requests can no longer over-admit against pages already promised to
    prefilling/decoding slots
- Qwen3.5 paged-prefill full-forward graph now reuses safely across chunk
  offsets because `start_pos` is uploaded through stable device-backed metadata
  instead of being baked into the prep-kernel launch parameters.

## Landed tracks

### Track A — Qwen3.5 paged prefill + prefill graph

**Outcome**

- `Qwen3.5` writes full-attention KV directly into the paged pool on the
  canonical prefill path
- the contiguous prefill fallback is no longer the default path
- paged-prefill full-forward graph capture/replay is wired onto the canonical
  prefill path
- graph replay is invalidated only when a pointer-stable prerequisite such as
  the page-index buffer storage changes
- `Qwen3.5` batch prefill now uses one packed multi-request paged-prefill path
  instead of sequentially replaying `prefill_forward_paged(...)` per request
- the remaining difference versus `Qwen3` is model capability shape
  (hybrid recurrent layers / no partial-prefix restore), not an extra
  scheduler-visible batch prefill fallback

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
- `0c49fca` `fix(qwen35): make paged-prefill graph start-pos device-backed`
- `b76c4bf` `feat(qwen35): override paged prefill batch path`
- `fd3b27b` `feat(qwen35): add packed conv1d prefill surface`
- `94c7df6` `refactor(scheduler): keep waiting queue incrementally ordered`
- `a01a124` `refactor(scheduler): pretokenize cuda http admissions`
- `14b4db6` `feat(scheduler): offload stopless streaming emit`
- `d2e29bd` `fix(scheduler): align admission with active headroom`
- latest local tranche: unify all streaming emit behind one worker and consume
  stop-sensitive gate results on the scheduler side before the next decode launch
- latest local tranche: finish `Qwen3.5` packed multi-request paged-prefill on
  the canonical model path and unify paged-prefill logits postconditions

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
- `Qwen3.5` no longer falls back to the trait-default paged-prefill batch path
- `Qwen3.5` batch prefill no longer sequentially replays per-request paged
  prefill over a batched page allocation; it runs one packed multi-request
  model-side paged-prefill path
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

## Remaining work

The high-priority in-repo closure for the five commissioned items is landed,
but one model-shape boundary remains important to state accurately:

- `Qwen3.5` is now on paged prefill and has its own batch override
- `Qwen3.5` still does **not** use the same packed varlen batched paged-prefill
  implementation shape as `Qwen3`
- the remaining gap is therefore no longer "contiguous fallback vs paged
  prefill", but "model-side packed batched prefill unification"

External follow-up remains mandatory:

- non-local CUDA before/after `scripts/bench_guidellm.sh` parity sweep
- delta table vs SGLang for `c1/c2/c4/c8/c16`
- fix any review findings or remote-regression findings that the CUDA run
  surfaces
- latest local rerun on pulled `39152ac`:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c4-c8-c16-39152ac-localfix.md`
- latest `c16` end-to-end bottleneck trace on the same local tree:
  `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-end-to-end-bottleneck-39152ac-localfix.md`
- latest waiting-aware reclaim bench on the same local tree:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c8-c16-39152ac-waiting-reclaim.md`
- unified scheduler budget follow-up plan on the same local tree:
  `docs/plans/2026-04-22-cuda-unified-budget.md`
- unified scheduler budget local regression check:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-4eddda8-unified-budget.md`
- unified scheduler budget bottleneck trace:
  `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-unified-budget-bottleneck.md`
- prefill-completion batching + prefix-aware deferral local regression check:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8.md`
- async prefill overlap local rewrite plan:
  `docs/plans/2026-04-22-cuda-async-prefill-overlap.md`
- async prefill overlap local regression check:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-18c116d-async-prefill-overlap.md`
