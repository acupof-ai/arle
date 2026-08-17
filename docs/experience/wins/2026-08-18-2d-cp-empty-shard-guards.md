# 2D CP empty-shard decode + short-prompt prefill guards — CUDA, 2026-08-18

> Status: pending-remote (typecheck green on `cuda,no-cuda`; GPU needle ladder pending pod)

## Context

The 2D KV ownership path (attn_tp × attn_cp, world ≥ 4) shipped T3.2b with two
crash/hang bugs that the needle ladder missed because the verified lengths
(4K / 128K / 224K) are all multiples of cp_size=8:

- **Decode empty shard.** A rank holding no resident pages (prompt shorter
  than `page_size` on the cp shard) called FA3 with `seqlen_k=0`. The FA3 shim
  rejects `seqlen_k <= 0` (`arle_fa3_shim.cu:105`), so the rank returned
  `cudaErrorInvalidValue` while peers hung on the cross-cp merge all-gather.
  The `refresh_sharded_decode` comment claimed "kv_lens=0 bounds the read to
  zero" — false; the shim rejects before launching.

- **Prefill slice underflow.** The ceil-div distribution
  `((p+1)*per).min(len) - p*per` underflowed (negative → huge usize) whenever
  `len % cp_size != 0`, leaving trailing ranks at 0 rows. Those ranks then hit
  the ring kernel's `rows<=0` / `blk_len<=0` rejections
  (`ring_prefill.cu:161,190`). This is the common case, not just short
  prompts — any prompt length not divisible by cp_size triggers it.

## What Worked

- **Decode bypass.** When `two_d && meta.max_kv_len() == 0`, skip the FA3
  launch and write `-inf` lse (host-staged `memcpy_htod_async`) + zero out the
  partial (`memset_d8_async`). The cross-cp merge
  (`cross_cp_merge_bf16_hd256_cuda`) computes `w = exp(lse - max_lse)`, so a
  `-inf` lse ranks weights this rank at zero. `max_kv_len()` returns the local
  shard's token count under 2D decode (`refresh_sharded_decode` pins
  `seqlen_k_capture = None`), so the guard is exact.

- **Balanced prefill distribution.** `base = len / cp` rows each, one extra to
  the first `rem = len % cp` ranks. Every rank holds ≥ 1 row when `len >= cp`
  (the common case), eliminating 0-row ranks by construction. `pad` stays
  `ceil(len/cp)` (the max slice length), so ping-pong buffer sizing is
  unchanged.

- **0-row ring guard.** For `len < cp` (still has 0-row ranks), an `active =
  rows > 0` flag skips prep / merge / scatter / finalize on 0-row ranks while
  they still post the ring send/recv so peers' KV rotates through. Buffer
  allocs use `rows.max(1)` / `acc_rows.max(1)` to avoid 0-length
  `cuMemAllocAsync`. Scatter is additionally guarded on `blk_len > 0` (the
  owner's block), since `cp.k_pos[owner][0]` panics on an empty owner.

- **`for_ring_prefill` positions.** `start_pos + rows - 1` underflowed for
  `rows=0`; switched to `saturating_sub`.

## Rule

A rank that holds no data for a collective step must still join the
collective, but skips the compute: write the algebraic identity ( `-inf` lse,
zero partial) and let the merge fold it in. Slice distributions that can yield
a 0-length shard need a balanced form, not a ceil-div that strands trailing
ranks. Verified lengths that are all multiples of the shard count hide this
class — the needle ladder should include a non-multiple length.
