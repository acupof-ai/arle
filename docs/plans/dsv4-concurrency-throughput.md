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

→ The cap is **per-row attention compute + eager host-launch**, not collectives.

## Lever ranking

### 1. Batched MLA decode — PRIMARY (#60; "Phase 5 batched FlashMLA")
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

### 3. DP-attention — LOWER / orthogonal (#89)
- **SGLang**: `dp_attention.py` splits TP into `attn_tp × dp`; gather/scatter at
  the attn↔MLP boundary (`dp_gather_partial`/`dp_scatter`, MAX_LEN/SUM_LEN
  padding for rank-uniform collectives); scheduler entry = `attn_tp_rank==0`.
- **Why lower for us**: it removes the attention collectives (the ~7ms floor,
  2.7% at B=16) + lockstep skew, but does **not** reduce per-rank attention
  compute (TP `B×8 heads` == DP `B/8×64 heads`). It pays off when memory-bound /
  at very high concurrency, not for the compute-bound 1.4× gap. Detail +
  phased plan: [`dsv4-dp-attention.md`](dsv4-dp-attention.md).

## Recommended sequence

1. **Batched MLA decode kernel** (#60) — biggest, direct win on the measured cap.
2. **Couple it into a whole-step CUDA graph** with SGLang's device-metadata
   capture pattern (#70) — kills the residual host-launch and unblocks our IMA.
3. **DP-attention** (#89) — only after 1+2, and only if a re-baselined c-sweep
   shows the residual collective/skew floor (now larger relative to a fast
   batched+graph step) is worth the scheduler re-architecture.

Each step gates on a c-sweep wall-clock A/B (TTFT + ITL + agg), multi-shape,
per the bench spec — no default flip on a single-shape ROI.
