# DSv4 concurrency throughput — lever ranking (SGLang-grounded)

Status: **design** (2026-06-13). Supersedes the lever framing in
[`dsv4-dp-attention.md`](dsv4-dp-attention.md). Grounded in the measured
concurrency baseline
([wins](../experience/wins/2026-06-13-dsv4-concurrency-baseline-serial-capped.md))
+ a study of SGLang's DeepSeek-MLA decode path (`/sgl-workspace/sglang`).

## The problem (measured)

DSv4 concurrent decode barely scales: batched aggregate **1.40× for 16× load**
(c=1→16: 44→62 tok/s); the default serve is flat ~53 (batched-decode opt-in,
off). The ceiling is **not** the GPU.

## Root cause (4 converging lines)

1. **c-sweep arithmetic**: `T_step ≈ 7ms floor + 15.7ms/req`. The floor
   (collectives + launch + skew) amortizes (= the 1.40×, 2.7% of step at B=16);
   the ~15.7ms/req is per-request compute that does **not** amortize.
2. **stage profiler**: step is host-launch-bound (~36ms host issuing kernels:
   attn-half ~16ms + moe-half ~20ms).
3. **our code**: `forward_decode_batch` runs attention **per-row**
   (`dsv4.rs:2498-2544`, `for r in 0..seq_len { mla_attention(...) }`); MoE
   batched. At B=8 → 8× per-row attention kernels + 8× launches.
4. **SGLang**: batched MLA decode = ONE `flash_mla_with_kvcache` for all B
   (`block_table`+`cache_seqlens`), inside a CUDA graph.
5. **MoE ∝ active_experts ∝ c** (ckl `74b721db`, root-caused on Qwen #88; same
   structure on DSv4): at decode each *active* expert needs ≥1 kernel block
   regardless of how few tokens route to it, and #active-experts grows with c
   (E=256, top-6: c=8 → ~45 distinct). So per-step MoE time grows with c → the
   MoE half does not amortize. EP=8 distributes it (~/8 per rank) — which is why
   DSv4 gets 1.4× not dead-flat (ckl's Qwen `fused_moe` was flat). Fundamental to
   sparse-MoE decode below expert saturation (c << E/top_k ≈ 43).

→ The non-amortizing marginal (~15.7ms/req) is **TWO halves**: (a) per-row
attention compute (attn-half, ~16ms host — fixable) + (b) MoE ∝ active_experts
(moe-half, ~20ms host — fundamental). Eager host-launch compounds both. NOT the
collectives. **Gap**: the precise attention-vs-MoE *GPU* `cuda_ms` split is not
yet measured — the stage profiler emitted only host-time this run; host-time says
moe-half (20) > attn-half (16). Close with a clean rank-0 per-stage GPU profile.

## Lever ranking

The cap is two non-amortizing halves. Levers 1-2 attack the **tractable** half
(attention + launch); the MoE half is **fundamental** (lever 3); DP-attn is
orthogonal (lever 4). Which of attn-half vs moe-half is bigger in GPU time is the
open gap above (host-time says moe-half).

### 1. Batched MLA decode — most tractable win (#60; "Phase 5 batched FlashMLA")
- **SGLang**: `flashmla_backend.py` `forward_decode` issues one
  `flash_mla_with_kvcache(q=[bs*heads,…], block_table=[bs,max_pages],
  cache_seqlens=[bs], tile_scheduler_metadata, num_splits)` — all B requests'
  KV addressed by the `block_table` tensor; KV-index build is one batched Triton
  kernel (`create_flashmla_kv_indices_triton[(bs,)]`). No per-request loop.
- **Our gap**: attention is per-row (Step-A loop). MoE/MLP already batch.
- **Port**: a batched MLA-decode kernel taking a `[bs]` cache_seqlens + per-req
  KV page/index table, one launch for all rows. This removes the ~15.7ms/req
  marginal — the actual 1.4×→linear lever.

### 2. Whole-step CUDA graph w/ capture-safe MLA metadata — SECONDARY (#70)
- **SGLang**: captures the decode forward at bs=[1,2,4,8,…]; pads a real batch
  up to the nearest captured bs. Attention metadata is **device-resident,
  pre-allocated, updated in-place** before replay
  (`init_forward_metadata_replay_cuda_graph`: `create_…_kv_indices_triton`
  + `get_mla_metadata` then `.copy_()` into the captured buffers). The graph
  replays as ONE launch → kills host-launch.
- **Our gap**: eager (~36ms host-launch); our whole-step graph IMAs on the
  FlashMLA path (host-computed per-step offsets captured as constants).
- **Port**: the SGLang device-metadata pattern (derive every per-step offset on
  device, update captured buffers in-place) is exactly the fix for our
  capture-IMA blocker — and it must wrap the batched-decode kernel from lever 1.
- **Coupling**: levers 1+2 are ONE mechanism in SGLang (batched kernel inside
  the graph). Build them together; a captured per-row loop is pointless.

### 3. MoE ∝ active_experts — the FUNDAMENTAL half (#88 lessons)
- **Mechanism**: at decode each active expert = ≥1 kernel block regardless of its
  (tiny) token count; #active-experts ∝ c (E=256, top-6) → MoE per-step time ∝ c
  → non-amortizing. The moe-half (~20ms host, likely the bigger half).
- **Already mitigated**: EP=8 distributes active experts (~/8 per rank) → DSv4
  gets 1.4× not dead-flat (ckl's Qwen `fused_moe` without enough EP was flat,
  `74b721db`).
- **Further gains are hard**: batching can't fix it below expert saturation
  (c << ~43). Options: a decode-tuned grouped-MoE kernel (small-M-per-expert
  efficient), higher EP, or accept the sparse-MoE-decode floor. This is the
  hard ceiling on concurrent DSv4 decode throughput — track with the #88 MoE
  kernel-shape work. **Diagnose by the curve SHAPE** (flat ⇒ per-step ∝ c ⇒
  work ∝ active_experts), not Δ% (ckl's rule).

### 4. DP-attention — LOWER / orthogonal (#89)
- **SGLang**: `dp_attention.py` splits TP into `attn_tp × dp`; gather/scatter at
  the attn↔MLP boundary (`dp_gather_partial`/`dp_scatter`, MAX_LEN/SUM_LEN
  padding for rank-uniform collectives); scheduler entry = `attn_tp_rank==0`.
- **Why lower for us**: it removes the attention collectives (the ~7ms floor,
  2.7% at B=16) + lockstep skew, but does **not** reduce per-rank attention
  compute (TP `B×8 heads` == DP `B/8×64 heads`). It pays off when memory-bound /
  at very high concurrency, not for the compute-bound 1.4× gap. Detail +
  phased plan: [`dsv4-dp-attention.md`](dsv4-dp-attention.md).

## Recommended sequence

0. **First close the gap**: a clean rank-0 per-stage GPU `cuda_ms` profile (or
   nsys) at B=1 vs B=8 to rank attn-half vs moe-half. If moe-half dominates GPU
   time (host-time suggests it), the attention levers below cap out at the MoE
   floor and the #88 MoE-kernel work is the higher-ROI track.
1. **Batched MLA decode kernel** (#60) — the tractable win; removes the attn-half.
2. **Couple it into a whole-step CUDA graph** with SGLang's device-metadata
   capture pattern (#70) — kills the residual host-launch and unblocks our IMA.
3. **MoE decode-kernel shape** (#88 lessons) — the fundamental ceiling; the
   higher-ROI track if moe-half dominates. Independent of 1-2.
4. **DP-attention** (#89) — only after the above, re-baselined; orthogonal.

Each step gates on a c-sweep wall-clock A/B (TTFT + ITL + agg), multi-shape,
per the bench spec — no default flip on a single-shape ROI.
