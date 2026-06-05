# DSv4 FlashMLA prefill (final-call swap) — KILLED: 6-12%, diverges, wrong lever

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.

## Context

After unblocking prefill (the i32-overflow fix), the assumed perf lever was wiring
**FlashMLA prefill** (`arle_flashmla_sm90_sparse_prefill_fwd`) — the prior session's
22× number. Wired it gated (`ARLE_DSV4_FLASHMLA_PREFILL=1`, scalar default):
unified KV pack, CSA/HCA index builders, TP Q all-gather/repack, `attn_sink_f32`,
max_logits/lse, output slice, inverse-RoPE, bf16 SW-ring update.

## Root Cause (why it was killed)

Two independent failures, both fatal to landing it:

1. **Only 6-12%, not 22×** — 4K: scalar 19.1s → FlashMLA 17.9s (6%); 8K: 41.1s →
   36.2s (12%). Stage profile (4K FlashMLA): **`mla_attn` 58.5%, `moe_route` 18.4%**.
   The wiring **replaces only the final hybrid attention call** — the
   compressor (c4/c128 KV), indexer (CSA top-512 select), metadata build, and TP
   Q-gather/repack/FP8-pack **wrappers still dominate the `mla_attn` block**. So
   swapping the *kernel* barely moves the slice: the cost is the serial *prepare*
   chain, not the attention math.
2. **Correctness diverges** — 8K 16-token needle: scalar `[1162,344,260,3549,...]`
   vs FlashMLA `[1162,344,270,3287,16,...]`, diverging at **token index 2** after
   matching the prefill argmax + first decode token. Not a scratch-lifetime issue
   (sync didn't fix it) — a prefill state-carry / numerical-trajectory mismatch.

## Fix

**Reverted the prototype** (`attention.rs`). Kept only the independent prefill
stage-profile printing in `dsv4_parity.rs` (calls the committed `stage_profile`
exports — useful for the redirect). FlashMLA prefill (final-call swap) is **not**
the prefill lever.

## Rule

**Swapping the headline kernel is not the lever when a serial *prepare* chain
dominates the stage.** The prefill `mla_attn` block is 58.5% but the attention
*math* (the FlashMLA fwd) is a small part — the compressor/indexer/metadata/Q-pack
prepare stages, run serially before it, are the cost. The real prefill levers are
**(a) overlap the prepare path behind the GEMM (multi-stream, like decode §5.1),
(b) optimize the compressor/indexer/route kernels directly** — not a final-call
kernel swap. (And license-or-kill caught it: first-token parity looked fine; the
full-sequence needle is what exposed the divergence — an SLO verdict needs the SLO
workload, not the smoke first token.)
