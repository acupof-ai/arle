# DSv4 decode CUDA graph is a WASH at B=1 — the "34% launch gap" is a framing artifact

## Context

The decode-6ms plan assumed the nsys "34% launch gap / 10ms" was closable by a
FlashMLA-decode CUDA graph (the assumed last −10ms lever). Tested the existing DSv4
decode graph (`ARLE_DSV4_DECODE_GRAPH`, `forward_tokens_decode_graph`) on the committed
binary, B=1 needle, SPEC off.

## What the A/B showed

| | steady tok/s | output |
|---|---:|---|
| DECODE_GRAPH=0 (eager) | 38.39 | `[223,30793,…,8308,344]` |
| DECODE_GRAPH=1 (graph) | **36.49 (−5%)** | byte-identical (captures FlashMLA + MoE + allreduce fine) |

The graph is CORRECT (byte-identical, no FlashMLA-capturability error — the transport
defaults to allreduce, and FlashMLA-decode captures cleanly) but **NET SLOWER**.

## Root cause / Rule

- **B=1 decode is GPU-bound; launch-overhead removal is a WASH** (`feedback_b1_decode_gpu_bound_overhead_removal_wash`,
  now 4× confirmed). The per-kernel launches OVERLAP the GPU backlog, so the nsys "34%
  launch gap" is a FRAMING ARTIFACT — the trailing NVTX range absorbs the async backlog
  and reads as idle "gap" that isn't dead time (`reference_nvtx_range_ending_in_sync_phantom_bottleneck`).
  The CUDA graph removes launches that were already hidden, and adds per-step H2D/replay
  overhead → −5%. **There is no −10ms graph lever.** Do NOT default-on `ARLE_DSV4_DECODE_GRAPH`.
- **The decode-6ms path is therefore LESS GPU WORK PER TOKEN, not launch removal.** From
  ~15ms (MTP single-draft, +71%), the lever is **multi-token MTP draft** (draft 2-3
  tokens, verify K+1 via the landed batched verify — amortizes the 149GB weight read
  over more accepted tokens) → ~3 tokens/step → ~5-6ms. mHC fuse (~2ms GPU) is a
  secondary GPU-compute lever (license per-A/B, also possibly wash if launch-bound).
- License decode levers on wall-clock same-binary A/B, never on nsys "% of window".
