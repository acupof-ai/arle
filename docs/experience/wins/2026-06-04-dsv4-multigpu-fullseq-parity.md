# DSv4 TP=8/EP=8 full 16-token greedy parity PASS (rewrite == oracle)

## Context

Follow-on to `2026-06-04-dsv4-multigpu-token1-parity.md` (token-1 gate). R6 clean-CUDA
DSv4-Flash forward, 8×H20 sm_90a, TP=8/EP=8, canonical model
`/data01/models/DeepSeek-V4-Flash` (NOT the base-43 workaround view — loaded directly
via the MTP-tolerant `ensure_loadable`, commit `7a7bd70d`). `pending-remote` (no local
CUDA). Verified pod fixes mirrored to the repo in `08b74b35` (per-slot decode state +
MoE shared-after-allreduce contract; typecheck + clippy clean on Mac).

## What Worked

Prompt `671,6102,294,8760,344` ("The capital of France is"), MAX_NEW=16, ranks 5/6/7
(and rank 0 prefill argmax) all produced:

```
clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111, 603, 671, 6102, 294, 8760, 344, 11111, 603]
```

= **the captured oracle, all 16 tokens, exactly** (" Paris.\nThe capital of France is
Paris.\n…"). This closes the full-sequence correctness gate: the per-slot decode state
(SW ring + compressor pending/compressed buffers retained across `start_pos>0` steps,
not reallocated per call) plus the full forward stack (MLA + CSA/HCA + hyperconnections
+ hash routing + FP8 MoE grouped GEMM + native experts + TP all-reduce) reproduce the
reference greedy continuation end-to-end.

Two bugs were fixed to get here, both with direct evidence:
1. **Per-slot decode state** — `forward_tokens` reallocated SW ring caches per call and
   `compressor_forward` required `start_pos==0`; made them executor per-slot state with
   `start_pos` threaded prefill→decode (`dsv4_compressor_update_cuda`'s
   `pending_len`/`compressed_base`/`has_prev_overlap`/`start_pos` params). Result: 16
   tokens, **no incremental-decode bail**.
2. **MoE shared-expert all-reduce contract** — the correct multi-rank contract is
   `dsv4_moe_forward` returns **routed-only** (EP-partial) → caller **all-reduces** →
   then adds the **shared expert once per rank**. The repo had shared added *inside*
   `dsv4_moe_forward` before the all-reduce, and shared weights load **whole** on every
   rank (`loader.rs`, no TP slice) → shared summed `world_size`× (8×). Restoring the
   shared-after-all-reduce contract took canonical token-1 from a regressed 223 back to
   11111 (clean argmax, margin 2.25), then the full 16 to oracle.

## Update (multi-prompt breadth — same day)

The single-prompt pass did **not** generalize. Gate-1 breadth (native+bf16, 3 distinct
prompts vs re-captured legacy oracles): "pancakes" MATCH 16/16; "largest planet" MISMATCH
at decode step 4 (270 vs 260); "hash table" MISMATCH at step 3 (396 vs 778) — **1/3 clean**.
Prime suspect is a **precision confound**, not necessarily a forward bug: the rewrite runs
the **bf16-KV** correctness path while the legacy oracle recipe uses `--kv-cache-dtype fp8`,
so tight-margin decode steps flip under bf16-vs-FP8. Isolation in flight: (1) logit margin
at first divergence, (2) legacy-**bf16** vs rewrite-bf16 same-precision A/B. If legacy-bf16
== rewrite-bf16, the per-slot decode state is correct and the real gate is FP8-KV decode
(`alloc_fp8_arena`) vs the FP8 oracle (ckl: "fp8 验证好就行"); if not, a per-slot
decode-state bug. **This entry's PASS is scoped to one prompt; full DSv4 correctness is
NOT yet established.**

## Rule

Full-sequence (not just token-1) greedy parity on the canonical model is the real DSv4
correctness gate — token-1 alone misses the per-slot decode KV-retention path. **Scope
not yet closed:** single prompt (multi-prompt/multi-shape pending), **native** expert
backend + **bf16** KV correctness path (the FP8-KV decode arena, DeepEP *serving*, and
the DeepGEMM production expert backend remain — they are the goal's perf/completeness
items, decoupled from this correctness gate). For TP-MoE: a replicated dense shared
expert must be added **after** the routed all-reduce (or be TP-sharded), never folded in
before — folding-before silently `world_size`×'s it.
