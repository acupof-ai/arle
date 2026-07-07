# DSv4 compact-FP8 MoE decode-tail scratch pooled — correctness verified

> Status: Verified (correctness) | wall-clock A/B deferred to full sweep | 2026-07-07

## Context
Launch-bound Step 1 commit 2 (`docs/plans/2026-07-07-dsv4-decode-launch-bound-plan.md`).
`dsv4_moe_forward_decode_fp8` allocated 8 device buffers + memsets **every MoE
layer every step** on the launch-bound batched decode path (nsys: alloc+memset =
16.8% wall, decode is 66%-wall launch+sync-bound, no CUDA graph). Commit `4f589cfb`.

## What Worked
Added `Dsv4MoeTailScratch` (moe.rs) sized to the decode-band route ceiling
(`DSV4_DECODE_CONTIG_MAX_ROUTES = 128`), held model-wide on the kv_adapter
(`moe_tail_scratch`, allocated whenever the model has a MoE layer — independent of
the decode graph). Threaded via a `tail: Option<&mut Dsv4MoeTailScratch>` param on
`dsv4_moe_forward` → `dsv4_moe_forward_masked_tail` → `dsv4_moe_forward_decode_fp8`;
the batched-stream path passes `Some(kv_adapter.moe_tail_scratch_mut())`, the other
4 call sites pass `None` (throwaway scratch, byte-identical to the old per-call
allocs).

Buffer init discipline (from the pre-implementation kernel audit):
- **6 pure-output, reused as-is**: offsets, scan_total, packed_hidden,
  packed_weight, act, expert_out (writer fully overwrites the rows it later reads).
- **4 re-init per step** (`reinit`): counts→0, cursors→0 (atomicAdd/bump),
  route_out→0 (EP: non-local route positions read by combine must be 0),
  packed_route_slot→-1 (scatter sentinel that keeps packed_weight/expert_out
  pure-output). memsets on the live `[0,rows)` span only.

VRAM budget (`dsv4.rs`) counts the new fixed term so KV-pool sizing doesn't OOM.

- **Compile**: BUILD_EXIT=0 (cuda,nccl,deepep), clippy-clean. First attempt caught
  the 3-layer threading (`masked_tail` was the missing middle hop, not a direct
  `dsv4_moe_forward`→decode_fp8 edge) — fixed.
- **Correctness (§0 case-as-fact)**: TP=4/EP=4 GPU 4-7, DSv4-Flash-FP8, greedy,
  MTP-on. Decoded prompts coherent: "capital of France" → `content:'Paris'`;
  "three primary colors" → coherent reasoning. No garbage/NaN/empty. The 4-buffer
  re-init is correct (a wrong route_out zero or packed_route_slot sentinel would
  garble MoE output — it didn't).
- **Alloc/memset removal (code-certain)**: 8 allocs + the born-zero/-1 memsets per
  MoE layer per step → 4 targeted re-init memsets on the pooled buffers, 0 allocs.

## Honest scope
Same framing as commit 1: alloc COUNT drops by construction; **wall-clock impact
of this commit alone is a fraction of the 16.8% alloc+memset wall** and expected
noise-level in isolation. The lever is the accumulated Step-1 sweep (commit 1
shared-expert + this MoE-tail + commit 3 attn/ffn stream double-buffer + N-ring).
Wall-clock A/B is deferred to the full-sweep vs baseline run; a per-commit A/B
would measure noise. Correctness is gated per commit (done here).

## Rule
Thread pooled scratch through the ACTUAL call chain, not the apparent one — a
`grep` of the leaf-fn callers can hide a middle dispatcher (`masked_tail`) that
also needs the param. Verify the chain compiles before trusting the wiring.
