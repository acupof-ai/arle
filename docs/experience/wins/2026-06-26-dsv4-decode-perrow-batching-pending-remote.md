# DSv4 decode per-row batching — grouped O-LoRA done, perf PENDING-REMOTE (GPU-blocked)

Status: grouped O-LoRA batched + committed (`30befcd9`); correctness gate (n=1 bit-identity)
green; **pod perf/needle verify BLOCKED on GPU availability** — honest pending-remote.

## Context
afd6d717 batched the small decode GEMMs (indexer-query/gating/slice_out/single-group
O-LoRA) — VERIFIED correct (needle exact=3) but a **TP=4 perf WASH** (decode-phase
unchanged). Research (workflow wq8xsowti) root-caused it: mainstream (vLLM/SGLang/
DeepSeek-V3.2) batches ALL of decode via slot-indexed scatter (reshape_and_cache /
fused_store_index_k_cache / paged_mqa_logits) — per-request is NOT fundamental. The real
per-row holdouts (the perrow≈15ms/finish≈15ms buckets) are: (a) DSA indexer cache-write,
(b) redundant gathers, (c) flashmla_pack, (d) grouped O-LoRA. afd6d717 missed all four.

## What landed (30befcd9)
**(d) grouped O-LoRA (groups>1, the TP=4 lane)** — was per-row × per-group (n·groups M=1
GEMVs), the dominant un-batched finish cost at TP=4. Made `dsv4_wo_a_grouped_deepgemm_decode`
M-parametric via the existing gather→GEMM(m=n)→scatter (`dsv4_oproj_group_gather/scatter`
already num_tokens-parametric → no .cu change, no FFI churn). Per group: one
`decode_proj_deepgemm_raw(m=n)`. The per-row grouped fallback loop + dead `local_attn_row`
scratch DELETED — single-group/plain-o/grouped converge on ONE batched flow (deletion-style).
n=1 byte-identical (gather idx == old slice; m=1 skips active_counts H2D) + CPU test.

## What's NOT done
- **(a) cache-write + (c) pack** = new batched FP8/Hadamard kernels, byte-identity
  **pod-only-verifiable** — deferred (my pod lane, needs GPUs).
- **(b) gathers** = NOT a localized deletion: prepare consumers take owned `&HiddenStates`
  (no view variant), so the per-row copies are structurally required until the consumer API
  is reworked. Bigger structural change (task 5).

## Measured (TP=4, GPUs 0-3 freed, 30befcd9)
needle exact=3 all lengths (correctness ✓). Finish batched WORKED: decode-phase n=8
`finish=14.7ms` (flat — per-row would scale to ~24ms). BUT throughput UNCHANGED: c=1/8/16/32
= 10.9(cold)/39.5/56.3/65.3 tok/s ≈ baseline 31/43/57/66. The finish was only ~12% of the
88-119ms step; the DOMINANT batchable per-row cost is the PREPARE (~41ms @ n=8: proj 3 +
compidx/indexer-compressor cache-write 18 + pack/gathers ~20) + MoE (54ms, 45%, physics —
needs batch≥32-64). So grouped O-LoRA is CORRECT + a real (flat) batch, but MARGINAL at TP=4.

## The real lever (task 5, NOT done)
The prepare per-row (~41ms) is the throughput lever: (a) DSA indexer cache-write + (c)
flashmla_pack → batched slot-scatter kernels (NEW FP8/Hadamard .cu, pod-verified), (b)
gathers → consumer-API rework (owned &HiddenStates → views). Structural, multi-step.
afd6d717 (small GEMMs) + 30befcd9 (grouped O-LoRA) are both CORRECT but the throughput
needle won't move until the prepare cache-write/pack are batched. TP≥4 required (TP=2 OOMs).

## Rule
DSv4 TP=4 verify needs 4 dedicated ≥91GB GPUs (TP can't go lower — weights don't shard
enough). On a shared box, check `nvidia-smi` per-GPU free + pick a 4-GPU set disjoint from
other users' jobs (never kill them); if <4 big GPUs free, the verify is genuinely blocked —
defer pending-remote, don't fake it on an undersized config (TP=2 OOMs).
