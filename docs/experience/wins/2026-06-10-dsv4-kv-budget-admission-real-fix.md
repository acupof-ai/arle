# DSv4 KV-budget admission real fix (#57) — slots×arena capacity, clamp→scheduler flow, NCCL min-reduced

**Date:** 2026-06-10. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commit:** `42e7e039`. **Scope:** `infer-api/loaded.rs`, `infer-cuda/{tp,dsv4,executor,lib}.rs`.
**Replaces** the `39be5f83` band-aid per its own FOLLOW-UP note, on top of
`b63f7987` (uniform `cuda_admission_total_pages`).

## Goal

Issue #57: admission must derive from the real per-model KV budget. Three
holes closed in one tranche:

1. **c>1 under-admission** — the DSv4 capacity arm covered ONE max-context
   request (`max_seq+4096` tokens); DSv4 KV is a pre-sized arena PER SLOT, so
   true capacity is `num_slots × per-slot tokens`. Below the 8192-page floor
   this serialized concurrent long prompts on fictional pages while a real
   slot arena sat free (binds for 2×prompt > 131K tokens).
2. **Over-admission on clamp** — `kv_budget_num_slots` (aa445112) clamps the
   executor's slot count to free HBM, but the engine scheduler kept
   `config.num_slots` → admission could target slots the executor has no
   arena for. `CudaExecutor::effective_num_slots()` now feeds BOTH the
   scheduler and the admission pool.
3. **Cross-rank clamp divergence** — per-rank `mem_get_info()` is not
   guaranteed identical, and the clamp now feeds scheduler-visible capacity;
   divergence ⇒ deterministic-planner divergence ⇒ NCCL deadlock (the
   `b374aef5` desync class). The local affordable count is NCCL min-reduced
   (`TpRuntime::all_reduce_min_scalar_i32`, identity on single rank; a rank
   that cannot query contributes `i32::MAX` instead of skipping the
   collective).

## Params / Env

Serve: `arle serve --backend cuda --model-path /data01/models/DeepSeek-V4-Flash
--port 18189`, allreduce MoE default lane, deepgemm experts, RUST_LOG=info.
Binary: main HEAD post-`42e7e039` build (`DONE ec=0`, symbol
`cross-rank-min affordable` verified in binary). Old binary preserved at
`/data01/build/arle_pre57` for the A/B.

## Results

| Check | Shape | Before (`arle_pre57`) | After | Verdict |
|---|---|---|---|---|
| V1 boot (collective at construction) | 16K max_seq, 8 ranks | boots | boots to 200, all 8 ranks enter lockstep driver | **PASS** — the construction-time NCCL min all-reduce sequences correctly (a mis-joined collective deadlocks boot) |
| Decode smoke (perf neutrality) | c=1, 128 tok greedy | 39.83 tok/s | 39.38 tok/s (−1.1%), first sentence byte-identical | **PASS** (single-run noise; B=1 decode is GPU-bound, admission untouched) |
| Needle gate (correctness neutrality) | 115/300/446/2000/8000 ×3 | 3/0/0 · 1/0/2 · 2/1/0 · 2/1/0 · 3/0/0 (#56) | 3/0/0 DET · 0/2/1 · 2/1/0 · 2/1/0 · 3/0/0 | **PASS** — within the same-config envelope (len-300 flips 1 exact↔partial, inside the MoE floor) |
| V3 dual-long admission | 128K max_seq, 2×82,608-tok prompts (page need exceeds old fictional capacity) | **ENGINE CRASH** — both admitted, pool exhausts mid-prefill (`CudaKvPool out of pages: slot 0 needs 256, free 0`, all 8 worker ranks die at tick #17, engine thread closed, every later request 400s) | **CONCURRENT** — both walls 38.3s (overlapped chunked prefill), serve healthy after | **PASS** — the old fictional page accounting over-admitted then died; slots×arena capacity keeps the page gate from binding before the slot gate |
| V2 256K admission | 262144 max_seq, ~200K prompt | 55.4s (band-aid verify, 06-09) | **BLOCKED** — 256K no longer BOOTS on either binary (`DSv4 official DSA logits alloc OOM`, 8/8 ranks; pre-#57 binary fails identically) | **Deferred to #67** — a 06-09→06-10 boot regression outside #57's diff (4-byte collective scratch); admission capacity itself is exercised at 128K |
| KV-budget clamp | 16K/128K boot | clamp internal to executor only | no clamp triggered at either shape (affordable ≥ 4 slots); clamp→scheduler flow verified by code path + boot health, warn log untriggered | PASS (shape-limited) — a clamping shape requires #67 fixed first |

## Problems

- 256K (`INFER_DSV4_MAX_SEQ_LEN=262144`) no longer boots on EITHER binary —
  `DSv4 official DSA logits alloc` OOM at engine build, a regression between
  the 06-09 band-aid verify and the 06-10 06:46 build. Filed **#67**; V2 of
  this verification transfers there.
- The old binary's dual-82K behavior was WORSE than the predicted
  "serialized": the fictional page accounting admitted both, the pool
  exhausted mid-prefill, and the engine died on all 8 ranks (every later
  request 400s "engine thread closed"). The fix prevents a crash, not just a
  latency artifact.
- The old binary 400s >32K-ish prompts at this serve shape in a first attempt
  (pre-crash) — the under-admission hole was partially masked by serve-level
  prompt caps until the recent cap lifts; it would have bitten as those lifts
  landed.

## Learnings

- For slot-arena (recurrent-KV) models, page-pool capacity must be
  `slots × per-slot tokens` so the page gate NEVER binds before the slot
  gate; any "one big request" sizing eventually over- or under-admits at
  c>1 and the failure mode is a mid-prefill alloc crash, not a queue.
- Any capacity decision that feeds the scheduler on a multi-rank engine must
  be cross-rank reduced (NCCL min) AT THE DECISION POINT; error paths
  contribute the identity (i32::MAX) instead of skipping the collective.
- Verification shapes must check WHERE the old capacity actually binds
  (the 8192-page floor = 131K tokens masked the 16K-max_seq shape; only
  >32K-token concurrent prompts reach the hole).
