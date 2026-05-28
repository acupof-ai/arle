# DSv4 A4 multi-stream TP comm/compute overlap — Phase 1 KILL (AllGather Q is not the binding constraint)

## SLO-shape probed? — Y (4K, 16K, 24K all measured end-to-end on 8×H20, TP=8)

## TL;DR

Phase 1 of the A4 axis (hoist FlashMLA prefill AllGather Q onto
`ctx.comm_stream` to overlap with compute-stream `arle_flashmla_csa_pack_kv`
+ `arle_flashmla_csa_build_indices`) implemented correctly, runs without
errors, and produces **byte-identical greedy output to the baseline**.
But wall-clock TTFT is ≈ 0% changed at 16K and 24K (the target case)
and +1.4% at 4K (within noise). **KILL on the wall-clock framing
per CLAUDE.md §0.**

The implementation is correct — the hypothesis that AllGather Q is a
material fraction of per-layer wall-clock is wrong on this HW + shape.

## Measured results — 8×H20, DSv4-Flash, TP=8, fp8 KV cache, num-slots=4

| Workload | overlap=0 (V2.4 baseline) | overlap=1 (Phase 1) | Δ ms | Δ% | Verdict |
|---|---:|---:|---:|---:|---|
| 4K  (4017 tok)   | 19401 ms  | 19678 ms  | +277 | +1.4% | wash / mild regression |
| 16K (16017 tok)  | 105466 ms | 105259 ms | −207 | −0.2% | wash |
| 24K (24016 tok)  | 193277 ms | 193271 ms | −6   | ≈ 0% | wash — **target case** |

Greedy responses at temperature=0 are byte-identical between
overlap={0,1} at each prompt length — verified by `diff` on the
`response: '...'` line of `result.txt`. This is strong evidence that
the fence ordering is correct (no race, no garbage).

Artifacts (on pod): `/sgl-workspace/arle-fresh/docs/trace-artifacts/2026-05-28-a4-overlap{0,1}-{4,16,24}k/`.

## Roofline check — why overlap doesn't help

Back-of-envelope, per layer at TP=8 / 16384 tokens / head_dim=192:

- `padded_send_count` = `16384 × 8 heads × 192 / 8 ranks` = **3.1 MB
  per rank per layer per AllGather**.
- NVLink AllGather on H20 at ~30 GB/s ≈ **100 µs/layer**.
- Across 61 DSv4 layers: **~6 ms total AllGather** (`payload_per_layer ×
  layer_count`, ignoring NCCL launch overhead).
- 16K wall-clock baseline: 105466 ms total → ~1.73 s/layer of which
  ~100 µs is AllGather Q → **AllGather is ~0.006% of per-layer
  wall-clock**.

Even if Phase 1 hides 100% of AllGather Q latency behind
`kv_pack`+`build_indices` on compute, the achievable gain is ≤ 0.1%.
That's at the noise floor and well below the 5% PASS threshold.

The **actual binding constraints** at this shape (per
`docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`):

- **L4 attention** — FlashMLA itself (~12.4% of 16K TTFT was unlocked
  by V2.4 doing FlashMLA at all, but the kernel still dominates the
  remaining per-layer time).
- **MoE expert dispatch / combine** — L6 NCCL combine ≈ 20 ms / rank
  range. Dwarfs AllGather Q.
- **L5 DtoH** — 344 sync metadata transfers per decode token. Not
  applicable to prefill but indicative of where the launch-bound axis
  lives.

A4 multi-stream overlap **only helps when the collective on
`comm_stream` is a measurable fraction of the layer's serial budget**.
That's true for the **EP combine reduce-scatter** at TP=8 / token-route
shapes (the existing `dsv4_combine_overlap_enabled()` lever — 20 ms/rank,
~1-2% of layer wall-clock), but **NOT for AllGather Q** at this prompt
shape.

## Why the 24K wash is NOT an AllGather problem

The original premise of this work was: "24K wash = AllGather serializes
chunk-2 (7632 tokens) per-layer, killing the FlashMLA win". Let me
re-check by chunk:

| Chunk | Tokens | `padded_send_count`/rank | Est AllGather µs/layer |
|---|---:|---:|---:|
| chunk-1 | 16384 | 3.1 MB | ~100 µs |
| chunk-2 |  7632 | 1.5 MB |  ~50 µs |

Across 61 layers per chunk:
- chunk-1: ~6 ms AllGather
- chunk-2: ~3 ms AllGather
- Total: ~9 ms AllGather across both chunks.

24K probe overlap=0 = 193277 ms. Total AllGather ~9 ms / 193277 ms
= **0.005% of wall-clock**. Even Phase 2 TokenWeave-style 100% chunk
pipelining gives sub-noise improvement.

The 24K wash root cause must be **elsewhere** — most likely:

1. **Chunk-2 amortizes the per-chunk fixed cost worse** (chunk-2 has
   7632 tokens vs chunk-1's 16384 → ~46% the useful work but same
   FlashMLA launch / metadata / sched overhead).
2. **Chunked-prefill state-flush between chunks** (CSA selector
   recompute, indices rebuild, etc).
3. **Compressed-KV size grows between chunks** so chunk-2's
   `kv_pack` reads a larger `compressed_count` than chunk-1, raising
   the per-layer compute.

None of those are addressable by AllGather Q overlap.

## SOLID check on the framing trap

Two framings give different conclusions — wall-clock framing is ground
truth (CLAUDE.md §0):

| Framing | Conclusion |
|---|---|
| Per-NVTX-window: "AllGather Q is X% of FlashMLA window" | Would suggest overlap is meaningful. nsys would show the overlap collapsing the window — looks like a "win". |
| Per-wall-clock TTFT: 4K/16K/24K elapsed_ms with byte-identical greedy output | ≤ 0% change. Phase 1 is null. |

This mirrors the `M_pf-graph v2 framing trap` recorded in CLAUDE.md §0:
"nsys 55.7%/window but 0.32%/wall-clock". I did **not** rely on nsys
window % to license; I went directly to wall-clock A/B. PASS on the
methodology — KILL on the optimization.

## What stays in the tree

The plumbing is left in place for two reasons:

1. **Pattern reusability**: `LayerCommunicator::with_tp_overlap_nccl` /
   `tp_overlap_nccl()` mirrors the EP overlap surface and may be used
   by future work (e.g. attention output AllReduce or context-parallel
   forward — A7). Cost of leaving it is one extra `Option` field +
   ~50 LOC.
2. **Default OFF**: `ARLE_DSV4_FLASHMLA_TP_OVERLAP=0` is the default;
   the early-hoist code path is fully dead at runtime by default.
   Setting `=1` is opt-in for future experimentation.

Future revival paths:

- If a **larger TP shape** (TP=16, 32) makes AllGather Q a bigger
  fraction → re-bench. The math here is TP=8-specific.
- If a **TokenWeave-style per-chunk pipelining** also overlaps the
  FlashMLA kernel itself with the next layer's AllGather (cross-layer
  pipeline), the budget changes. That's Phase 2 territory and would
  need a separate plan.
- If the binding constraint moves (e.g. after A2 fuses CSA + hybrid
  attention) and AllGather becomes a larger fraction, re-bench.

## Code-correctness fence pattern (preserve)

The fence pattern itself is correct and reusable. It worked exactly
as designed — output byte-identical between overlap={0,1}:

```
compute stream                              comm stream
  dsv4_prepare_qk_fused      (writes q_prepared)
  ─── comm_waits_for_compute() ────────────→
                                              overlap.all_gather_bf16_device
                                                (reads q_prepared, writes gathered)
  arle_flashmla_csa_pack_kv
  arle_flashmla_csa_build_indices
  arle_flashmla_fill_pad_rows
  ─── compute_waits_for_comm() ←─────────
  dsv4_tp_q_repack_cuda      (reads gathered, writes packed)
  arle_flashmla_sm90_sparse_prefill_fwd
  dsv4_tp_out_slice_cuda
```

The early-hoist position (records fence after `qk_prep`, before
`kv_pack`) is critical — without it, the comm-stream wait orders past
`kv_pack` + `build_indices`, collapsing the overlap window. See plan
doc § Concrete code edits, rule (D).

## Commits

- `7e67a1c8` — `feat(layer-comm): add TP overlap NCCL group surface for A4`
- `91d5c077` — `perf(dsv4): A4 multi-stream TP comm/compute overlap (FlashMLA prefill)`
- (this entry) — KILL recording.

The two prior commits are **not** being reverted: plumbing is left
dormant (default OFF) for the pattern-reuse rationale above.

## Rule

**Before licensing a multi-stream overlap on the TP axis, measure the
collective's wall-clock fraction directly (not just nsys window %).**
Per-layer AllGather at H20 / NVLink is 30 GB/s — at TP=8 the per-rank
payload divides by 8, often dropping the per-layer collective cost to
< 100 µs. Across 61 layers that's < 6 ms total — sub-noise vs a 100 s
TTFT. EP combine reduce-scatter (~20 ms/rank-range) is structurally
different because each rank participates in a O(total_tokens) traffic
pattern, not O(local_q_count).

Corollary: prior to writing an overlap plan, do the
`payload × layer_count / total_wall_clock` envelope. If that ratio is
< 1%, **route attention to a different axis** (in this case, A2
hybrid-attention fusion or A1 Mega-MoE has the leverage). The plan's
A1 ranking ("multi-rank H20 long prompt prefill TTFT ≥ 20% improvement")
is still the correct next target.

## Refs

- Plan: `docs/plans/2026-05-28-dsv4-a4-multi-stream-overlap.md`
- V2.4 prefill state: `docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md`
- Binding constraints: `docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`
- Backlog axis: `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md` (A4)
- Pod artifacts: `/sgl-workspace/arle-fresh/docs/trace-artifacts/2026-05-28-a4-overlap{0,1}-{4,16,24}k/`
