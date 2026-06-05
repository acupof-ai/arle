# SGLang DSv4 decode optimizations → ARLE systematic adopt plan

**Date:** 2026-06-05. Distilled from a code-level SGLang DeepSeek-V4 reference
(§3.2 Waterfill, §3.3 EPLB/exec-paths, §3.4 DeepEP dispatch/combine, §5.1
multi-stream overlap, §5.2 metadata-in-CUDA-Graph). Principle:
`先用最好的再自己写` ([[feedback_no_closed_door_solutions]]). **§0 filter applied:
not every SGLang lever is a B=1 decode-latency lever** — prioritized against
ARLE's measured stage profile (scalar 23.7 → 33.0 tok/s; remaining slices: MLA
attention ~11.3 ms, HC ~4.9 ms [fuse in-progress], MoE [contig landed]).

## Lever map (prioritized for the B=1 single-token-latency goal)

| SGLang lever | B=1 decode lever? | ARLE current state | Adopt action | Priority |
|---|---|---|---|---|
| **§5.1 Multi-stream overlap** (hide indexer/compressor/KV-write/q-proj behind the big `wq_b` GEMM via alt-streams + fine-grained events, capture + small-batch only) | **YES** — pure decode-latency | **NONE** (decode runs attention-prepare serially — verified: no alt-stream/fork/record_event) | adopt SGLang's `_forward_prepare_multi_stream` structure | **HIGH (next big lever after HC)** |
| **HC fuse** (`mhc_pre_big_fuse_tilelang`: RMSNorm+Sinkhorn+mix one kernel + PDL) | YES | 86 launch/tok + single-CTA Sinkhorn (anti-pattern) | adopt fused TileLang mhc_pre | **in-progress** |
| **§5.2 Metadata-in-CUDA-Graph** (`SGLANG_PREP_IN_CUDA_GRAPH`: raw→full upgrade inside graph; replay copies a few small tensors, not rebuild metadata tree on CPU) | partial — collapses per-step host launch | has decode graph; metadata host-built | **verify host_ms first**, adopt if significant | MEDIUM (verify-gated) |
| **§3.3 exec-paths / DP-attention** (CP / `_use_tp_moe_gather` / `_use_tp_attn_a2a_scatter`) | weak | `attn_dp_size` axis unwired | wire DP-attn | LOW — profile says attn_allreduce only **1.05 ms** |
| **§3.2/3.4 DeepEP Waterfill** (shared expert as routable 9th, dispatched to least-loaded rank) | **NO for B=1** — `MIN_BATCH_FOR_BALANCE=64` skips it; keeps shared expert local | shared expert local | — | LOW (throughput/multi-req lever, not single-token latency) |
| **§3.4 DeepEP low-latency dispatch** (decode small-batch comm primitive) | maybe | #24 left a combine `ctx.sync` | revisit after overlap | LOW (moe_allreduce 2.34 ms) |

## §5.1 multi-stream overlap — the HIGH lever (deep-dive)

**SGLang structure** (`_forward_prepare_multi_stream`, 5 streams CUDA / 2 HIP):
main stream computes `wqkv_a` + `q_a`/`q_norm`, records 2 events; **3 alt-streams
run concurrently** — `stream_indexer` (weight-proj + fused-Q rope+hadamard+fp8 +
`fp8_paged_mqa_logits` + `topk_transform_512`; itself forks again for
weight-proj ∥ fused-Q), `stream_kv` (fused norm+RoPE+direct-write FlashMLA paged
cache, no BF16 spill), `stream_compressor` (c4/c128 KV compression). **Meanwhile
the main stream runs the big `wq_b` GEMM** (`_compute_q_b`), then `wait_stream`
the three. The indexer FP8-score+select-512, compressor, and KV-write are **hidden
behind the `wq_b` projection** — a standard MLA backend runs them serially.
Cross-stream deps use fine-grained `record_event`/`wait_event` (KV waits only
`qkv_a`; indexer only `q_lora`; indexer-Q only `q_lora`+`weights`) to minimize
false deps. Enabled only in **CUDA-graph capture + batch ≤ 64/128** (host
launch/event cost amortized by replay; above the cap the main GEMM saturates SMs
so no headroom — SGLang's own assumption, nsys-unverified).

**ARLE adopt action** (gated, in the existing `INFER_CUDA_DECODE_GRAPH` capture):
ARLE today runs the whole attention-prepare serially. Add a side-stream set;
launch the indexer (CSA select), compressor KV update, FP8-KV pack/write, and the
q-projection prep on alt-streams while the main stream runs the `wq_b`-equivalent
GEMM; join with fine-grained events before the FlashMLA fwd. This directly
shrinks the **~11.3 ms MLA-attention slice** by overlapping its prepare ops with
the largest attention GEMM, without changing any kernel. Reuse the decode-graph
machinery (events are capturable). **Caveat (ARLE memory):** this is CUDA, not
Metal — [[feedback_mlx_async_eval_is_caller_thread]] (MLX encode-on-caller) does
NOT apply; CUDA streams genuinely overlap. Watch [[reference_disabled_event_tracking_premature_buffer_free]]
— cross-stream buffers need keepalive past the join, and a private stream must
`stream_wait` the caller ([[feedback_private_stream_needs_stream_wait]]).

## §5.2 metadata-in-graph — verify-first

SGLang's `init_forward_metadata_decode` returns a thin raw struct
`(req_pool_indices, seq_lens, out_cache_loc)`; full materialization (c4/c128
compress meta, core_attn, indexer meta, FlashMLA split) happens **inside** the
captured graph, so each replay copies a few small tensors instead of rebuilding
the metadata tree on CPU. Ordering: raw→full upgrade on the main stream **before
any alt-stream fork**; `_GraphBucket` (DECODE_OR_IDLE / TARGET_VERIFY /
DRAFT_EXTEND) × bs; warmup builds full, then restores to raw + re-allocs FlashMLA
meta per bucket so capture materializes in-graph.

**ARLE gate:** ARLE already closed the structural host-overhead arc (host-route /
alloc / D2D / launch) and the stage profile shows host_ms is small. **Measure
ARLE's per-decode-step host metadata-construction cost first** (stage-profile the
host side of metadata build); adopt only if it's a material slice. Lower ROI than
§5.1 on current evidence.

## Execution sequence + gates

1. **HC fuse** (adopt `mhc_pre_big_fuse_tilelang`) — *in-progress*.
2. **§5.1 multi-stream overlap** — the next big lever; attacks the 11.3 ms
   attention slice. Gated, resident same-load A/B (stage profile shows the
   overlapped slices shrink + real tok/s rises), oracle16 + 80-tok no-bail.
3. **§5.2 metadata-in-graph** — only if step-2 profiling shows host metadata
   construction is still a material per-step cost.
- **Deprioritized:** Waterfill (throughput, B=1 skips it), DP-attn (1.05 ms),
  DeepEP-LL (2.34 ms) — revisit only after the attention slice is overlapped.

Every lever: license-or-kill on the wall-clock B=1 SLO A/B (not narrow-window
%, not mixed/reachable); gated default-off; full KV-precision-parity is the
precondition for any default flip (still legacy-`infer/`-only, un-re-ported).
