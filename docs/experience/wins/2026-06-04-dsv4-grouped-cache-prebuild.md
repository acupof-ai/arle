# DSv4 MoE: prebuild the FP8 grouped-expert cache at load (kill 529ms/token rebuild)

**Date:** 2026-06-04. **Backend:** CUDA (DSv4-Flash, TP=8/EP=8, 8×H20).
**Scope:** `crates/infer-cuda/src/{dsv4.rs,loader.rs,moe.rs}`.
**Status:** **prefill correctness verified on 8×H20** (TP=8/EP=8, DeepGEMM + allreduce);
decode tok/s bench **pending a separate pre-existing DSv4 incremental-decode (start_pos>0)
fix** (the path was never completed — see below). Bundled with the wq_b TP-shard fix
(`load_dsv4_block_scaled_sharded`) that this verification also surfaced + landed.

## Goal

Eliminate the dominant DSv4 decode cost. Per-op profiling (#16) of DSv4 decode found the
hot path was **not** a kernel: `deepgemm_grouped_experts` (`moe.rs:903`) called
`build_grouped_cache` for w13 **and** w2 **on every layer, every token** — each call
re-allocating a contiguous group-major buffer and D2D-copying every local expert's FP8
weight + scales into it. The per-expert caches are **static after load**, so this rebuild
is pure waste, and its cost is token-count-independent (so even single-token decode pays it).

## Evidence (nsys decode trace, per token)

| Op | Before |
|---|---|
| `build_grouped_cache` (w13 + w2, ×43 layers) | **~529 ms/token** |
| DeepGEMM grouped-GEMM kernel pipeline | 13.5 ms/token |

~768 MiB D2D/layer (w13 ~512 MiB + w2 ~256 MiB at TP/EP=8, ~32 local experts/rank) × 43
layers = tens of GiB of redundant D2D copy + alloc per token. The fix is not tuning DeepGEMM.

## What changed

- `Dsv4MoeLayer` (`dsv4.rs`): per-expert `w13: Vec<…>` / `w2: Vec<…>` replaced by the
  **prebuilt** `GroupedCache` (`w13_grouped`/`w2_grouped`) + metadata (`num_groups`,
  `hidden_dim`, `intermediate`). The per-expert Vec is **dropped** after the concat —
  keeping both would double MoE weight memory (~32 GiB/rank).
- `loader.rs`: the group-major D2D concat (was `build_grouped_cache` per call) runs **once
  at load**, with the same 128-aligned shape validation.
- `moe.rs` `deepgemm_grouped_experts`: uses `&layer.w13_grouped`/`&layer.w2_grouped`
  directly; the two per-call `build_grouped_cache(...)` invocations are gone.

The concat is byte-identical to the old per-call build, so the DeepGEMM kernel reads the
same bytes → **correctness must be preserved by construction.**

## Verify (8×H20, on a tree symbol-checked to be current-main + this delta)

- **Base trust:** the pod tree was first found at a stale base (`bc4aa4e`, 66 commits behind
  main, missing the attn_sink TP-offset fix `d5f74c0b` + F32 loader `3d60af93`). Re-synced to
  current main + this delta and symbol-verified (`w13_grouped` present, attn_sink fix present,
  per-call `build_grouped_cache` gone) before trusting any result. (See
  [`errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md`] — pod base must == main.)
- **Correctness:** `scripts/dsv4_multigpu_parity.sh`, `ARLE_DSV4_EXPERT_BACKEND=deepgemm`,
  `ARLE_DSV4_MOE_TRANSPORT=allreduce`, `ARLE_DSV4_INCREMENTAL_KV=1`, prompt `671,6102,294,8760,344`,
  16 new tokens, 8 ranks. Gate: matches the bf16 oracle (must hold — byte-identical concat).
  → **PENDING** (slot verdict).
- **Decode A/B:** same binary/model/shape, single-user decode, before (per-call build) vs
  after (prebuilt). → **PENDING** tok/s before → after + Δ%, and `build_grouped_cache`
  absent from the after-trace.

## Results

**Prefill correctness — PASS (8×H20 TP=8/EP=8, DeepGEMM + allreduce):**
- 5-token prompt `[671,6102,294,8760,344]` → prefill argmax token#1 = **11111** = oracle[0] ✓.
- 6-token prefix `[…,11111]` → prefill argmax = **603** = oracle[1] ✓.
- The grouped concat is byte-identical, so prefill matching the bf16 oracle confirms the
  prebuild is correctness-neutral; the wq_b TP-shard fix cleared the `wo_a cols 4096 !=
  local attention width 32768` load error (wq_b was loaded full 32768 vs wo_a sharded 4096).

**Decode tok/s Δ — PENDING.** Blocked by a **separate, pre-existing** DSv4 bug: incremental
decode (`start_pos>0`) diverges from prefill — the 5-token-prompt + 15-step decode run yields
garbage `[16,11111,0,…]` after the correct first token, while feeding the same prefix as a
*fresh prefill* gives the correct 603. The harness has always flagged this ("only the FIRST
token is a verified gate; the remaining 15 need the incremental-decode follow-up"), so the
prior "3/3 16/16" was prefill-only. The 529 ms/token rebuild is paid in **prefill too**
(per-layer, token-count-independent), so a prefill-TTFT before/after will quantify the win
once a clean pod run is available; the canonical decode tok/s Δ follows the decode fix.

## Rule

For tiny-per-op-but-massive-aggregate costs, profile the **wall-clock decode breakdown**, not
the kernel catalog: the dominant DSv4 decode cost was a host-orchestrated D2D **rebuild of
static weights**, invisible to a kernel-only view. Static per-expert → group-major data must
be built **once at load**, never per forward. And before trusting any pod perf/correctness
result, symbol-verify the pod tree is current-main + the delta — a stale base silently
confounds both attribution and the correctness gate.
