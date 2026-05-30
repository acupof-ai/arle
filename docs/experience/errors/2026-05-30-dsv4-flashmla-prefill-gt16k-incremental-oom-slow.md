# DSv4 FlashMLA prefill >24K — cap lifted, but blocked by the incremental-prefill path's memory+speed at >16K chunks

## SLO-shape probed? — Y (8×H20 TP=8, 18K/26K/29K prompts, FlashMLA on AND legacy)

## Context

Goal: finish FlashMLA prefill integration so >24K prompts use FlashMLA instead of
the legacy fallback. Two correct, committed foundations landed:
- **`282e797f` cap-lift**: the stale `FLASHMLA_TOTAL_POSITION_LIMIT=24576` gate →
  bound to `max_position_embeddings`, so >24K CSA/HCA prefill is eligible for FlashMLA.
- **`deab74f8` chunked-prefill regression fix**: `8f4db3b6` ("route prefill through
  the incremental path", to fix the P→D decode-KV handoff) only routed the FINAL
  prefill chunk through `compute_top_level_logits_incremental`; intermediate chunks
  stayed on the stateless path, so any prompt >16384 (scheduler chunks at 16384)
  aborted with `DeepSeek V4 incremental state length 0 does not match scheduler KV
  length 16384` (weights.rs:570) — chunk-1 advanced the scheduler KV but not the
  model's `processed_tokens`. The fix routes EVERY chunk through the incremental
  forward (`emit_logits=false` for non-final), so `processed_tokens` and `kv_cache.len()`
  advance in lockstep. This **eliminated the assertion error** — confirmed on pod.

## The blocker (why >24K still does not complete)

With the assertion gone, FlashMLA *engages* at >24K (the failure mode moved from the
scheduler assert into the attention/forward), but the prompt still never completes —
and the cause is the **incremental-prefill path itself**, not FlashMLA:

| chunk size | prompt | result |
|---|---|---|
| 16384 (default) | 26–29K, FlashMLA on | **OOM** — `FlashMLA TP full-out/packed-Q scratch alloc failed: OUT_OF_MEMORY`, ~84 s in |
| 16384 (default) | 18K, **legacy attn** (FLASHMLA_PREFILL=0) | **OOM** — generic `Alloc failed: OUT_OF_MEMORY`, ~97 s in (so NOT FlashMLA-specific) |
| 16384, max-seq 20480 (tighter) | 18K, legacy | **OOM** still (so NOT the max-seq KV reservation) |
| **4096** | 29K, FlashMLA on | **no OOM**, but **504 timeout** — prefill reached only 16384/29186 tokens in 300 s |

Root cause: `8f4db3b6`'s incremental prefill **seq-batches the whole chunk in one
call** (`forward_transformer_layer_stream_incremental_into`). DSv4 has `hc_mult=4`, so
the per-layer hidden-carrier stream is `tokens × (hidden×4) = 16384 × 16384 × 2 B =
536 MB` for a 16384-token chunk, ×43 layers + the growing-context attention + the
per-layer SW/compressed/FP8 KV it must populate → **OOM at large chunks**. Shrinking
the chunk to 4096 fixes the memory but the incremental forward is **inherently slow**
(attention over the full prefix per chunk; `8f4db3b6`'s own comment admits "incremental
prefill regresses TTFT >2×") → **timeout at small chunks**. So the incremental path is
memory-prohibitive large-chunk and latency-prohibitive small-chunk: there is no chunk
size that makes a >24K prompt both fit and finish.

NB: this is NOT a regression introduced by `deab74f8` — before it, >16384 prefill was
already broken (the assert). `deab74f8` is correct and necessary; it just exposes that
the underlying incremental-prefill path was never viable for >16384. V2.4's "24K
chunked clean" run predates `8f4db3b6` and used the light **stateless** chunk path.

## The real fix (next axis — substantial, deferred)

`8f4db3b6`'s comment already names it: **"FlashMLA prefill writing KV directly"**. The
fast FlashMLA prefill computes attention but does NOT populate the per-layer
incremental SW/compressed/FP8 KV caches that decode reads — that is why prefill is
forced through the slow incremental forward purely to populate KV. The fix: make the
FlashMLA prefill path **write those per-layer KV caches as it runs**, so prefill is
fast (FlashMLA, ~15× over legacy) AND correct (decode reads populated KV), with no
dependence on the seq-batched incremental forward. Then chunk at the default 16384 with
FlashMLA, no OOM, no timeout. This is a multi-step feature (KV-write hooks inside the
FlashMLA CSA/HCA prefill dispatch), not a tuning change.

## What IS validated and shipped this session

- native-deepep MoE fully fixed + pod-validated ("Paris"/"406"/primes coherent) — see
  [`../wins/2026-05-30-dsv4-native-deepep-correctness-fix.md`] and
  [`../wins/2026-05-30-dsv4-native-deepep-combine-ima-crossstream-fix.md`].
- `deab74f8` chunked-prefill assert fix (eliminates the >16384 `incremental state
  length 0` abort) + `282e797f` cap-lift — both correct foundations, kept.

## Rule

- **A "route prefill through the incremental path" correctness fix must cover EVERY
  chunk, not just the final one** — scheduler chunking (16384) silently splits the
  prompt, and a path that only the logits-emitting chunk takes desyncs model state from
  scheduler KV on chunk 2+.
- **"It no longer crashes" ≠ "it works."** Lifting a gate / fixing an assert can just
  move the failure downstream (assert → OOM → timeout). Probe the SLO shape end-to-end
  (finish_reason + decoded tokens), not just the absence of the previous error.
- **A correctness-first stopgap (incremental prefill for KV population) has a real cost
  ceiling.** It was fine for ≤16K single-chunk; it does not scale to >16K. The deferred
  "FlashMLA writes KV directly" optimization is now load-bearing, not optional, for
  long-context prefill.
