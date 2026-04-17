# Codex review — paged-KV prefill plan

**Subject:** `docs/plans/paged-kv-prefill.md`
**Reviewer:** Codex (via codex-rescue)
**Date:** 2026-04-17

## Verdict
ship with changes — the direction is sound, but the plan misses active contiguous-KV dependencies, phases work in the wrong order, and overstates the expected throughput gain.

---

## Section-by-section findings

### 1. File-level changes

The file list is incomplete for the stated end state. The common migration and capacity hooks live in `infer/src/model.rs:96-120` and `infer/src/model/generation_state.rs:55-167`, not just the per-model `forward.rs` files. If prefill no longer uses contiguous KV or `set_max_seq_len`, those shared contracts need explicit plan coverage.

`KVCache` still does more than "INT8 working-buffer only." `GenerationStateBase::migrate_kv_range_to_paged` dispatches BF16, FP8, INT8, and TurboQuant via `k_caches()/v_caches()` plus `max_seq_len()` (`infer/src/model/generation_state.rs:121-165`). Single-token decode for Qwen3 and GLM4 also still writes/reads contiguous KV via `fused_attention_decode_into` (`infer/src/model/qwen3/decode.rs:11-38`, `infer/src/model/glm4/decode.rs:11-38`).

The plan omits `infer/src/model/qwen3/batch_decode.rs`. That mixed decode+prefill path explicitly requires contiguous caches: it checks `kv_cache.k_caches().is_empty()` and `max_seq_len()` before launching `prefill_attention_prep_dual_write_cuda` (`batch_decode.rs:360-364`, `batch_decode.rs:543-566`).

---

### 2. Phasing

Phase 3 is blocked by Phase 4. The scheduler still caps chunk size with `.min(CONTIGUOUS_KV_TOKENS)` in `infer/src/scheduler/cuda/core.rs:871-887`, and still migrates contiguous KV only after the final chunk in `infer/src/scheduler/cuda/prefill.rs:253-277`. Wiring models first cannot produce the claimed "one forward per prompt" behavior without the scheduler also being changed first.

The Phase 2 acceptance command does not map to the repo. `ops/tests.rs` is an internal unit-test module declared from `infer/src/ops.rs:20-22`, not a runnable integration target named `ops`.

The Phase 3 acceptance command pattern is also off. The repo has `infer/tests/e2e.rs` and `infer/tests/e2e_qwen35.rs`, but no `e2e_qwen3.rs` or GLM4-specific e2e harness. Until Phase 1 kernels are called by any production path, they are untested dead code — there is no way to exercise them between Phase 1 and Phase 3.

---

### 3. Bitwise parity claim

The plan's "attention arithmetic is unchanged" claim is too strong. Current prefill uses `SinglePrefillWithKVCacheDispatched` with `SinglePrefillParams` (`csrc/attention/flashinfer_prefill.cu:92-99`, `flashinfer_prefill_hd256.cu:66-73`). The proposed path uses `BatchPrefillWithPagedKVCacheDispatched` plus `PrefillPlanInfo` and `BatchPrefillPagedParams` (`flashinfer_tc_decode.cu:53-76`, `flashinfer_tc_decode.cu:117-182`). Same attention family, yes; same kernel/template instantiation, no. The parity claim is optimistic — our CI does not prove this today.

The existing prefill unit tests already use tolerances, not bitwise checks: `infer/src/ops/tests.rs:596-606` and `infer/src/ops/tests.rs:816-818`. So even "parity" in the current test suite only means within tolerance.

---

### 4. Risk list

The biggest missing risk is same-slot prefix reuse. Admission only treats a radix hit as reusable when a free slot still "materializes" that prefix via `block_owner_slots` and `slot_materialized_prompt_lens` (`infer/src/scheduler/cuda/runtime.rs:343-356`). `step_new()` then truncates/restores local state and migrates the prefix range from contiguous KV into the pool (`infer/src/scheduler/cuda/prefill.rs:99-156`, `prefill.rs:174-194`). The plan's statement that "prefix-cache already operates on the paged pool" (`paged-kv-prefill.md:145-153`) is not accurate today — that path still depends on contiguous state as the source.

Pool fragmentation is under-described. The allocator is a LIFO `free_pages` stack (`crates/infer-cuda-kernels/src/paged_kv.rs:60-61`) and `alloc_tokens()` appends whatever physical pages are popped (`paged_kv.rs:405-438`). Long prefill allocations will change page reuse patterns even if total bytes stay flat.

The Metal backend is correctly out of scope — `infer/src/backend/metal/` has its own prefill path and the plan's CUDA-only gating is correct. No issue there.

CUDA Graph capture interaction: no active capture code was found via grep (`enable_cuda_graph`, `graph_cache`, etc. — no results). The Qwen3.5 decode graph described in context is not yet in main, so this risk is forward-looking only. No action needed now, but worth noting in the risk list.

---

### 5. Acceptance-criteria realism

The Qwen3.5 sync baseline in the plan is wrong. The cited benchmark file reports TTFT p99 `982.6 ms`, not `820 ms`, and throughput `91.43 tok/s` (`docs/experience/wins/2026-04-17-bench-guidellm-qwen35-4b-infer-l4-p99.md:44-57`).

The +26% Qwen3.5 throughput target is not supported by the timing breakdown. That note shows 92.9% of chunk time is in the 32-layer forward loop and chunk elimination saves only about 50-80 ms over an 820 ms prefill (`docs/experience/wins/2026-04-17-qwen35-prefill-timing-breakdown.md:47-50`, `47-66`). The same doc says graph capture remains the bigger throughput lever (`timing-breakdown.md:88-90`). A pure layout change cannot credibly be the sole acceptance gate for 91.43 → 115 tok/s. A defensible claim would be +5-10% TTFT reduction with throughput improvement requiring P1 graph capture on top.

---

### 6. Structural alternatives

The root-cause win doc explicitly recommends a lower-risk sequence: try Path A (`CONTIGUOUS_KV_TOKENS = 2048`) first, keep P1 in parallel, and defer Path B until measured (`docs/experience/wins/2026-04-17-prefill-ttft-root-cause-contiguous-kv-cap.md:81-137`). The plan skips this de-risking step without explanation.

There is also already an incremental dual-write hook in the shared model contract: `forward_prefill_with_pool()` is documented as the future direct-pool path in `infer/src/model.rs:249-272`, and Qwen3/GLM4 already implement it (`infer/src/model/qwen3/forward.rs:163-188`, `infer/src/model/glm4/forward.rs:164-180`). That is a better stepping stone than jumping straight from contiguous prefill to full paged prefill plus scheduler cleanup in the same phase.

---

## Recommended plan edits

- `paged-kv-prefill.md` file list — add `infer/src/model.rs`, `infer/src/model/generation_state.rs`, and `infer/src/model/qwen3/batch_decode.rs`; otherwise the stated end state is under-scoped.
- Phase ordering — reorder so scheduler/core/prefix-reuse work lands before or with the first model migration; Phase 3 currently depends on Phase 4 changes to be meaningful.
- Bitwise parity claim — replace with greedy-token parity plus per-op tensor tolerances; fix the Qwen3.5 TTFT baseline from 820 ms to 982.6 ms and throughput target to match what the timing breakdown actually supports.
- Prefix-cache risk section — rewrite: today same-slot resurrection depends on contiguous state (`prefill.rs:99-156`), not just paged pages; the claim that prefix cache already operates on the paged pool is premature.
- Test commands — replace with repo-real targets (`infer/tests/e2e.rs`, `infer/tests/e2e_qwen35.rs`); spell out what covers GLM4 decode parity.
- Add an explicit "Path A first" decision gate (raise cap to 2048, measure TTFT delta, then decide whether full Path B is worth the scope) or justify why the root-cause doc's recommended sequence is being bypassed.

---

## Reviewer note on the "Path A first" recommendation

The user has explicitly stated
(feedback memory: `feedback_architecture_ideal.md`,
"架构永远按照理想态做事") that for architectural problems, the
structural fix is preferred over tactical shortcuts regardless of
diff size. So Codex's final bullet ("Add an explicit Path A first
decision gate") should be answered by **documenting the rejection of
Path A in the plan**, not by adopting Path A. All other Codex
findings (scope gaps, phase reordering, parity-claim softening,
baseline-number correction) are objectively correct and will be
applied.
