# KV swap-out path deleted entirely — vLLM V1 + SGLang both gave up on this approach

## Context

Phase 2 of [`active-kv-swap-out-unification`](../../projects/active-kv-swap-out-unification.md)
landed end-to-end as a multi-commit chain
(`8635fea1` Unit A → `c1e063b3` Unit B → `221704e0` Unit C+D →
`2ea2a8a9` CLI flag → `a430b1ee` host pool sizing fix), codex-clean
across nine review iterations, and bench-validated against the
2026-04-26 SGLang head-to-head config (Qwen3-4B / L4 / c=16 /
4096-in / 256-out / `--num-slots 16 --max-seq-len 4608
--mem-fraction-static 0.94 --chunked-prefill-size 4096`).

A/B between modes on the same Phase 2 commit:

| metric | Recompute (default) | Swap |
|---|---:|---:|
| TTFT p50 (ms) | 4065 | 12663 |
| ITL p50 (ms) | 48.6 | 1351 |
| out tok/s | 88.0 | 1.49 |
| peak_active | 6 | 14 |
| swap events | n/a | 8 |

Swap admits more concurrent requests (`peak_active` 6→14, +133%) but
the per-tick swap cost regresses ITL by **28×**. Net effect: tok/s
collapses from 88 to 1.49.

## Root Cause

The CUDA scheduler intentionally overlaps the prior tick's GPU readback
with this tick's admission planning (`runtime.rs::run` calls
`assign_slots` BEFORE the readback half of `step()`). Any synchronous
T0↔T1 memcpy issued from `assign_slots` blocks the scheduler thread
while the GPU drains, **destroying that overlap entirely**.

Lifting the `slot_has_pending_gpu_work` filter doesn't help — even with
no explicit `step_decode_readback` drain, `paged_kv_pool.copy_pages_to_host`
runs on the same forward stream and **its return implicitly syncs**
because it produces a `Vec<u8>` of bytes that must be GPU-complete. The
sync just shifts location; it doesn't disappear.

## What the production-grade systems do (primary-source research)

A 2026-04-27 deep dive into vLLM and SGLang trunks confirmed our 28×
regression isn't a bug we missed — it's the same arithmetic that drove
both projects to give up on KV swap:

**vLLM V0** (`vllm/core/scheduler.py @ v0.6.3:1054-1088`):
> "We use recomputation by default since it incurs lower overhead than
> swapping. However, when the sequence group has multiple sequences
> (e.g., beam search), recomputation is not currently supported. In
> such a case, we use swapping instead."

V0 keeps swap **only** for `n>1` (beam search). The memcpy rides
`at::cuda::getCurrentCUDAStream()` (cache_kernels.cu @ main:38-78,
line 72) — i.e., the forward stream, serialized against decode. No
dedicated swap stream. `swap_space=4 GiB` per GPU default.

**vLLM V1** (`vllm/v1/core/sched/scheduler.py @ main`,
`_preempt_request` lines 592-633): swap **deleted entirely**. Just
`kv_cache_manager.free(request) + status=PREEMPTED + num_computed_tokens=0`.
Discussion #11082 ("Why is there no swap queue in V1?") + Issue #18115
("`--preemption-mode is not supported by the V1 Engine. Falling back
to V0`") confirm this is intentional — prefix-cache-on-recompute wins
on every metric they care about.

**SGLang** (`python/sglang/srt/managers/schedule_batch.py @ main`,
`retract_decode` lines 1341-1398): no KV swap path. `release_req`
just frees pages, expects the radix prefix cache to make rerun cheap.
The `copy_stream` field in scheduler.py is for **sampler output D→H
copy**, not KV (per SGLang v0.4 blog + PR #1738).

**Conclusion from research agent**: "our 28× ITL regression isn't a
bug we're missing the fix for; it's the same reason both reference
systems gave up."

## Fix

Delete the entire Phase 2 swap path. Revert all 5 commits in the chain
(`8635fea1`..`a430b1ee`) including the host pool sizing fix that was
specifically motivated by Phase 2.

What's gone:
- `PreemptionMode { Recompute, Swap }` enum + `SchedulerConfig` field
- `--preemption-mode` CLI flag
- `try_swap_out_victim_for_admission` + `try_resume_swapped_victims` +
  `release_swapped_kv_regions` + `is_swap_out_victim` helpers
- `PlanKind { Prefix, WholeKv { last_token_pos } }` + `new_whole_kv`
- `release_slot_pages_only` factor (folded back into
  `requeue_preempted_decode`)
- Host pool sizing formula change (back to prefix-demote-only sizing)
- The earlier errors entry that documented the two layered blockers

What stays:
- `PreemptionMode::Recompute` semantics (= original behavior, unchanged)
- The active-kv-swap-out project doc kept as-is for historical reference;
  this entry is the closure note. Future iterations should NOT re-attempt
  swap without first invalidating both vLLM V1's and SGLang's reasoning.

The `peak_active=6` cap on c=16 / 4096-token-prompt workload remains —
that's a separate problem rooted in budget/scheduling/kernel quality,
not in preemption strategy. SGLang reaches higher concurrency on the
same workload **without** swap, via its prefix-cache + better admission
math + better kernels.

## Rule

**Validate strategy against production precedent before building.**
The Phase 2 design doc cited `host_pool=604.0MB sat idle` as motivation
without checking whether the target tier's capacity was sized for the
new use case OR whether vLLM/SGLang had concluded swap was viable. Both
checks would have shown the approach was structurally a dead end on the
intended workload.

Future preemption work in this repo should:
1. Start by scanning `vllm/v1/` and `sglang/` for the equivalent path.
2. If both projects deleted/never-had it, treat that as strong evidence
   the math doesn't work, not as "opportunity to differentiate."
3. If pursuing anyway, write a simulation of cost/benefit FIRST against
   the canonical workload — bench last, not first.

## Cross-references

- Phase 2 design (kept for historical reference):
  [`projects/active-kv-swap-out-unification.md`](../../projects/active-kv-swap-out-unification.md)
- Phase 1 (Recompute) thrash that triggered Phase 2:
  [`errors/2026-04-27-recompute-admission-thrash-on-long-prompts.md`](2026-04-27-recompute-admission-thrash-on-long-prompts.md)
- Reverted commits: `8635fea1`, `c1e063b3`, `221704e0`, `2ea2a8a9`,
  `a430b1ee`.
- Bench artefacts (Swap mode showing the 28× regression):
  - `bench-output/2026-04-27-cuda-l4-phase2-swap/`
  - `bench-output/2026-04-27-cuda-l4-phase2-swap-clean/`
- vLLM V0 swap site: `vllm/core/scheduler.py @ v0.6.3:1054-1088`
- vLLM V1 (swap deleted): `vllm/v1/core/sched/scheduler.py @ main`
- SGLang retract_decode: `python/sglang/srt/managers/schedule_batch.py @ main:1341-1398`
- vLLM Discussion #11082, Issue #18115 — V1 has no swap, by design.
