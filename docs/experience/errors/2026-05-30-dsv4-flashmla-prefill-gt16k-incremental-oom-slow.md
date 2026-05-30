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

## Update 2026-05-30 — OOM fixed; speed bottleneck PROFILED, and it is the MoE, not the KV-pack

Two findings collapsed the "incremental prefill is the blocker" framing:

**1. The OOM was a separate, fixable bug — NOT capacity.** A single 16K prompt drove a
96 GB GPU from 23 GB idle to 97 GB → OOM with 74 GB nominally free. Root cause:
per-layer compute scratch (sized by `tokens.len()`) accumulated across all 43 layers
because the prefill forward only trimmed it at the chunk's END. Fixed by trimming
per-layer (commit 2c133cc8, peak 97 → 34 GB, validated) — see
[`../wins/2026-05-30-dsv4-prefill-per-layer-scratch-oom-fix.md`]. After this, a 15.5K
prompt completes coherently; ALL >~12 K prefill is unblocked.

**2. The slowness is the MoE all-reduce + expert GEMM, NOT the KV-pack.** A per-phase
trace (`ARLE_DSV4_TRACE_LAYER=1`, summed across layers, relative ratios — the trace
adds sync so absolutes inflate):

| phase | share | note |
|---|---:|---|
| **ffn_total** (MoE half) | **~84%** | dominates |
| └ ffn_all_reduce | 52% of FFN | the NCCL all-reduce after MoE |
| └ ffn_routed_local (expert GEMM, native backend) | 41% of FFN | deepgemm faster but JIT-blocked on the CUDA-12.2 pod |
| attn_total (FlashMLA attention) | ~16% | NOT the bottleneck — FlashMLA already fast |

So `8f4db3b6`'s hypothesis ("the >2× TTFT is the KV-pack; fix = FlashMLA writes KV
directly") is **wrong** — the KV-pack phases (`attn_window_update`, `attn_compressor_update`,
`ffn_local_route_compact_pack`) are all <1% each. The prefill is MoE-bound.

**3. native-deepep is the right lever but does not yet make >24K fit the 300 s window.**
native-deepep (the +46% lever, fixed this session) replaces the all-reduce with EP
dispatch/combine. But a 24–27 K prefill still TIMES OUT at ~290 s with native-deepep
(8 ranks boot, NO IMA/OOM — it runs, just slow). Two reasons: (a) the cross-stream
correctness fix `f30043af` host-syncs before every dispatch+combine — 43 layers × 2
chunks × 2 = 172 host syncs serialize the forward (the deferred event-based
`stream_wait` would remove this); (b) the expert GEMM (native backend) is still heavy at
16384-token chunks, and deepgemm (faster FP8 grouped GEMM) is JIT-blocked on this pod.

## Update 2026-05-30 (round 2) — native-deepep is SLOWER for prefill; event-based combine correct but not the lever

Acted on the profile ("MoE all-reduce dominates → native-deepep removes it") and it did
NOT pan out for prefill:

- **Event-based `stream_wait` for combine (941c7d6c)** — replaced the per-layer host
  `ctx.stream.synchronize()` (f30043af) before/after combine with DeepEP's official
  on-device `stream_wait(comm,compute)` pattern (the combine receives `compute_stream`).
  CORRECT (validated: native-deepep short decode still coherent — "Paris"/"406"/primes,
  8 ranks, 0 IMA) and it removes a real CPU block. But it did NOT make prefill fast.
- **native-deepep is SLOWER than allreduce for PREFILL.** A ~16.5K native-deepep prefill
  still times out at 285 s, where a 15.5K allreduce prefill completed (≤285 s). So the
  +46% native-deepep win — measured at DECODE (1 token) — does NOT transfer to prefill:
  at 16384-token chunks the dispatch/combine all-to-all + the per-layer `num_recv`
  host-poll (the recv count sizes the expert compute, so it is an inherent CPU↔GPU sync
  that event-based streams cannot remove) cost MORE than the single NCCL all-reduce.
- So the earlier "~130 tok/s after native-deepep" projection was WRONG. native-deepep is
  not the prefill speed lever.

**Revised conclusion: >16K prefill is ~50–80 tok/s on BOTH backends, MoE-compute +
serialization bound.** Making it fast (>200 tok/s, TTFT <60 s for 24K) needs a major
effort that is NOT incremental and is partly pod-toolchain-blocked: (a) faster expert
GEMM — deepgemm FP8 grouped GEMM is the lever but JIT-blocked on the CUDA-12.2 pod;
(b) removing the per-layer `num_recv` host-poll (device-side capacity sizing) so prefill
can pipeline; (c) the all-reduce/GEMM kernels themselves. The event-based combine
(941c7d6c) is kept — it is correct and de-serializes the (production) decode combine —
but it is not the prefill fix.

## Status: >24K prefill is FUNCTIONALLY unblocked, PERF-bound on the MoE

cap-lift + chunked-prefill fix + OOM fix make >24K run correctly (no crash, no OOM,
coherent ≤16K). The remaining work is **MoE speed**, not FlashMLA: (a) event-based
`stream_wait` for native-deepep prefill (remove the 172 host syncs), (b) unblock deepgemm
experts on the pod toolchain, (c) the all-reduce/expert-GEMM kernels. This is a separate
optimization axis from the FlashMLA prefill integration, which is itself complete.

## Rule

- **Profile before optimizing a "slow path" — the obvious hypothesis is often wrong.**
  Three weeks of "FlashMLA writes KV directly" was the planned fix; a 10-minute
  per-phase trace showed the prefill is 84% MoE (all-reduce + expert GEMM) and the
  KV-pack is <1%. The attention (FlashMLA) was never the bottleneck.
- **"It no longer crashes/OOMs" ≠ "it's fast enough."** Lifting a gate, fixing an
  assert, and fixing an OOM made >24K RUN; it still times out on MoE compute. Functional
  and performance completion are different milestones — probe TTFT end-to-end.
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
