# DSv4 B=1 decode forward profile: where the 26ms goes (the forward-efficiency levers)

## Context

After MTP landed (~15ms) and the spec-ceiling path was bounded (depth-2-top1=33%), the
*other* path to 6ms is forward efficiency: my per-step forward is ~4× SGLang's. I'd never
profiled the current forward post-MTP. Did it with `ARLE_DSV4_STAGE_PROFILE` (per-stage,
synced so inflated but ranks consumers) + `ARLE_DSV4_LINEAR_PROFILE` (per-projection),
8×H20 TP=8, 37-token needle, B=1, non-spec.

## Per-stage (cuda ms/token, synced)

| stage | ms/tok | % wall | note |
|---|---:|---:|---|
| **mla_attn** | **12.05** | 24% | dominant, 3× the next |
| shared_expert | 4.06 | 8% | |
| moe_grouped | 3.32 | 7% | |
| moe_allreduce | 2.70 | 5% | TP |
| attn_hc_params / ffn_hc_params | 2.38 / 2.33 | mHC | launch-bound (wash) |
| moe_route | 1.35 | | |
| attn_allreduce | 1.10 | | TP |

`mla_attn` is NOT weight-read (attention weights ~0.27GB/GPU ≈ 0.08ms) — it's projection
kernel time. Linear-profile of it:

| projection | % linear | avg µs | kernel |
|---|---:|---:|---|
| wqkv_a_fused | 24.8 | 51 | scalar (unfused) |
| indexer_wq_b | 13.1 | 53 | scalar |
| wo_b | 13.1 | 26 | scalar |
| compressor_wkv | 12.3 | 17 | DSA |
| compressor_wgate | 11.9 | 16 | DSA |
| wo_a | 9.2 | 18 | scalar |
| wq_b | 7.9 | **16** | **DeepGEMM** ✓ |
| indexer_weights | 5.7 | 23 | DSA |

## Real levers (evidence-backed, NOT wash like graph/mHC)

1. **Finish projection→DeepGEMM**: wq_b (DeepGEMM) is the *fastest* projection (16µs) vs
   unfused scalar wqkv_a 51 / wo_b 26 / wo_a 18 / indexer_wq_b 53. Same lever that took
   prefill 408→23ms; for M=1 decode the gain is smaller (bandwidth-leaning) but real
   (DeepGEMM 16µs < scalar 26µs at similar size) — est ~1-2ms/token.
2. **DSA compressor+indexer = ~43% of projections, WASTED for seq < sliding_window**
   (compressor 24% + indexer 19%). The 37-token needle (seq < 128) has nothing beyond
   the window to compress/select — skipping the compressor/indexer when
   `start_pos+tokens <= sliding_window` is correct (output unused) and saves ~3-4ms/token
   for short decode. Correctness-gate: verify CSA layers fall back to SW for seq < window.

## Rule

- The decode-6ms gap is a forward-efficiency STACK, now mapped with data: projections
  9ms (43% wasted DSA for short seq + unfused scalar), MoE 11ms (shared/grouped/route/AR),
  mHC 4.7ms (launch-bound wash). No single lever reaches 6ms; the achievable with the real
  levers (DeepGEMM-all + DSA-skip-short + MTP) is ~11-13ms. MoE (vendored DeepGEMM) and
  the spec ceiling (depth-1 head) are the hard walls to literal 6ms.
- Profile the forward per-stage before claiming "near the floor" — `mla_attn`=12ms was
  invisible until measured; it's projection-kernel overhead, not weight read.

## UPDATE (2026-06-08): DSA-skip lever KILLED by A/B — the DSA is necessary, not wasted

Implemented the "skip csa_select when 0 compressed blocks (seq <= window)" lever
(`ARLE_DSV4_DSA_SKIP_EMPTY`) and A/B'd it (needle, B=1, SPEC off): output BYTE-IDENTICAL
(correct) but **−3.7%** (38.1 vs 39.6 tok/s — the per-call env-var read, and the skip
never fires). So `indexer_rows_after > 0` even for the 37-token needle: **DSv4's CSA
compresses ALL tokens into blocks + selects top-k — it is NOT "window + beyond-window
compressed"**. The compressor/indexer/csa_select are *necessary* CSA-attention compute,
not waste. My "wasted for seq < window" hypothesis was WRONG; reverted.

So the only real forward lever left is finishing projection→DeepGEMM (wqkv_a/wo, modest
at M=1). The forward (26ms) is mostly NECESSARY compute (CSA compress+select, MoE,
projections). Combined with the spec ceiling (depth-2-top1=33%), **6ms is bounded for
DSv4-Flash B=1** — achievable ~13-15ms; literal 6ms needs model-level (multi-layer MTP)
or library-level (SGLang MoE/CSA kernels) changes, proven now by A/B not assumption.
