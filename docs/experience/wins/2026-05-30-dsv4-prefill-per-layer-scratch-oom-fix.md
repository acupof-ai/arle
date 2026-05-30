# DSv4 prefill OOM — root-caused + fixed: per-layer scratch accumulated across 43 layers

## SLO-shape probed? — Y (8×H20 TP=8, single 15.5–16K-token prompt, nvidia-smi baseline vs peak)

## Context

While validating FlashMLA prefill >24K, every prompt above ~12K tokens 500'd with
`CUDA_ERROR_OUT_OF_MEMORY` mid-prefill — on a 96 GB GPU with only ~23 GB used at idle.
The user's challenge ("为什么 oom 也不应该的" — the OOM shouldn't happen) was correct:
**74 GB free vs a 16K forward that needs ~3 GB ⇒ a memory BUG, not capacity.**

A decisive anomaly ruled out capacity: V2.4 ran a 16017-token prefill CLEAN with
`--num-slots 4` (MORE KV pool reserved, LESS free memory); my failing run used
`--num-slots 1` (MORE free) yet OOM'd. So the OOM is a specific over-allocation, not a
free-memory shortfall.

## Root cause (measured)

`nvidia-smi` around a single 15.5K-token request:

| | GPU mem used | free |
|---|---:|---:|
| idle (weights + KV pool) | 23.3 GB | 74.2 GB |
| **peak during the 16K forward (before fix)** | **97.5 GB** | **0.02 GB → OOM** |

The DSv4 incremental prefill (`compute_top_level_logits_incremental`, weights.rs:593)
walks all 43 layers, and **each layer's compute scratch lives in the PER-LAYER
`state.incremental.layers[i]` cache**: attention `q_prepared/k_prepared/local_attn/
output_latent`, `attn_pre/normed/post`, `ffn_pre/normed`, attn/ffn MHC, and the MoE
route/expert scratch — every one sized by `tokens.len()`. For a prefill chunk that is
the chunk's token count (16384). Nothing freed them between layers, so **all 43 layers'
16384-token scratch coexisted for the whole forward → ~74 GB**. (This design is correct
for DECODE: `tokens.len()==1` makes each layer's scratch tiny and per-layer retention is
a reuse win — which is exactly why it was never caught.)

`trim_prefill_scratch` already frees this scratch (and keeps the KV caches decode reads:
`window_gpu` / compressed / FP8 pool) — but it was only called ONCE at the end of the
chunk (weights.rs:621/645), long after the mid-forward OOM.

## Fix

Apply that same trim **per layer**, inside the prefill loop, right after each layer's
forward (gated on `tokens.len() > 1` so decode keeps its per-layer scratch):

```rust
put_hidden_scratch(&mut layer_cache.stream_recycle, stream);
if tokens.len() > 1 {
    layer_cache.trim_prefill_scratch();   // free THIS layer's chunk-sized compute
}                                          // scratch now, keep its KV caches
stream = next_stream;
```

(+ made `DeepseekLayerRuntimeCache::trim_prefill_scratch` `pub(crate)`.) Safe by
construction: the end-of-chunk trim already frees exactly these buffers and decode works
after it, so doing it per-layer only lowers the peak — it cannot free anything a later
layer needs (each layer owns its own cache; the KV caches are kept).

## Validation (8×H20, single 15.5–16K-token prompt, after fix)

| | GPU mem peak | OOM | request |
|---|---:|---:|---|
| before fix | 97.5 GB | **OOM** | 500, never completes |
| **after fix** | **34–38 GB** | **0** | **HTTP 200, completes, coherent English** |

Peak dropped **97.5 → 34–38 GB** (scratch above the 23 GB baseline: 74 GB → 11–15 GB),
no OOM, and the 15.5K-token request now **completes with coherent output** (`text="I'm
sorry, I can't"` to a 15 500×"word"+question prompt — coherent, not the garbage a wrong
forward would emit). The 43-layer accumulation is gone; peak is now ~1–2 layers' scratch
+ the growing (compressed) KV caches.

## Still open (separate)

The incremental prefill is also SLOW (the `8f4db3b6` ">2× TTFT" regression): a 15.5K
prefill does not finish in ~100 s. The real fix is the deferred "FlashMLA prefill writes
KV directly" — see [`../errors/2026-05-30-dsv4-flashmla-prefill-gt16k-incremental-oom-slow.md`].
This OOM fix is orthogonal and necessary regardless.

## Rule

- **"It's just big" is a non-answer on a 96 GB GPU.** If a forward that should need a few
  GB OOMs with tens of GB free, find the over-allocation — here, a per-layer cache (right
  for decode's 1 token) silently became 43× a 16384-token buffer in prefill.
- **A per-step/per-token cache reused across a fixed set (layers) is O(N) in the batch
  dim.** Sizing it by `tokens.len()` is invisible at decode (1) and catastrophic at
  prefill (16384). Free it within the loop when the batch dim is large.
- **nvidia-smi baseline-vs-peak is the cheapest OOM oracle.** idle 23 GB → peak 97 GB on
  one request localized the bug to the forward's transient allocations in one probe,
  before any source spelunking.
