# DSv4 A2 — s_q=K FlashMLA verify, implementation-level spec (the decode speedup)

## Superseded by later evidence

**The s_q=K verify was KILLED, then the kill was un-killed by the frozen-KV
redesign — but MTP is now parked regardless.** The naive s_q=K verify both diverges
from autoregressive AND fails to amortize on DSv4's stateful compressed attention
([`../experience/errors/2026-06-06-dsv4-eagle-sqk-no-amortize-kill.md`](../experience/errors/2026-06-06-dsv4-eagle-sqk-no-amortize-kill.md));
SGLang's **frozen-KV** approach (freeze the compressor + reuse the selection during
verify) makes an off-boundary K-span exactly autoregressive and amortizing — see
[`2026-06-06-dsv4-frozen-kv-mtp-redesign.md`](2026-06-06-dsv4-frozen-kv-mtp-redesign.md).
Even so, MTP is parked at the **draft-quality wall** (39% accept vs SGLang 68%):
[`../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md`](../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md).
Re-anchor the decode-speedup target on the
[H20 reference baseline](2026-06-06-dsv4-h20-reference-baseline.md). Kept for history
(the kernel-already-supports-s_q=K analysis is still accurate).

---

**Date:** 2026-06-06. **Status:** structural design from CLEAN reference code (the
s_q=1 decode builder + the prefill multi-query builders); the runtime confirmation
of tranche-2's exact break comes AFTER A1 lands (re-run `/tmp/tranche2_sqk_broken.diff`
on the A1-clean base to isolate the pure glue bug — designing on the rollback-buggy
base is confounded, root `AGENTS.md` §0.1). Applies the §0.1 granularity.

## Why it's reachable (the kernel already supports it)

- FlashMLA decode attention FFI (`csrc/attention/arle_flashmla_decode_stubs.cu`)
  already carries `s_q` + every per-`s_q` stride (`stride_q_s_q`,
  `stride_indices_s_q`, `stride_lse_s_q`, `stride_o_s_q`, …). The vendored
  `splitkv_mla` kernel reads `s_q` from the Q shape (`sparse_decode.h:208`). So the
  ATTENTION kernel needs only a `[1,K,h_q,d_qk]` Q + `[1,K,topk]` indices +
  `[1,K,h_q,d_v]` out + `get_meta(h_q, K)`.
- The index device fn `arle_dsv4_flashmla_decode_index_at(...)`
  (`csrc/attention/dsv4_flashmla_decode_build_indices.cu`) is ALREADY parameterized
  by `start_pos`: it bounds the SW ring to ≤ start_pos and masks compressed block
  `c` when `c*ratio + (ratio-1) > start_pos`. It is only *called* for one row.

## The crux — A2b: extend the indices BUILDER to s_q=K (the .cu kernel, not glue)

tranche-2 allocated `indices[max_s_q * topk]` + `query_start_pos[s_q]` (Rust glue)
but did NOT extend the kernel, so only row 0 was filled → rows 1..K garbage → the
draft query attends to zero/garbage KV → structural divergence + α collapse + the
3× slowdown (likely a scalar fallback on the malformed path).

**Fix (exact):**
1. Host wrapper takes `s_q` and `query_start_pos[s_q]` (= base `start_pos + r`).
2. Launch the builder over **K rows** (grid `.y = s_q`, or an inner `for r in 0..s_q`).
3. Row `r` calls the EXISTING `arle_dsv4_flashmla_decode_index_at(..., start_pos =
   query_start_pos[r], ...)` and writes to `indices[r * stride_indices_s_q + ...]`,
   `topk_length[r]`. The device fn is reused verbatim — no new index math.
4. **Per-query causal is automatic**: row `r`'s `start_pos+r` bounds its own SW +
   compressed validity. Intra-batch causality (query 1 = draft at start_pos+1 sees
   pending at start_pos) works because the s_q=K forward packs ALL K tokens' K into
   the SW ring / fp8 pool BEFORE the attention (prefill-style prep), so pending's
   slot is present when draft's row is built. No separate mask.

## The rest of A2 (each a §0.1-detailed sub-task)

- **A2a Q layout** `[1,K,h_q,d_qk]`: lay the K verify tokens as the s_q dim;
  the TP gather/pack scratch sized `max_s_q * h_q_d` (tranche-2 had this).
- **A2d sched_meta**: `get_meta(h_q, s_q=K)` → `num_sm_parts` for K rows; alloc
  `sched_meta[num_sm_parts_max * 8]` (tranche-2 had this). lse/o_accum sized
  `accum_rows * max_s_q * h_q`.
- **A2e s_q=K rollback**: A1 reverts ONE draft slot; the s_q=K verify writes K-1
  speculative slots (positions start_pos+1 .. start_pos+K-1; row 0 = pending is
  committed). On reject, revert those K-1 ring slots (sw_window + fp8) + the
  compressor/indexer running buffers — extend A1's single-slot revert to the K-1
  draft slots (still O(K·elem), not O(ring)). Enumerate per §0.1.

## Gate (correct-inference, not byte-identity)

Per `feedback_correct_inference_not_baseline_identity`: needle (short + long) +
same-config-twice non-determinism floor. The s_q=K verify's per-position argmax
will differ from the s_q=1 baseline on near-ties (different query-tile float order)
— that is correct inference. Cross-check the s_q=K argmax against the per-token
s_q=1 reference (`ARLE_DSV4_MTP_VERIFY_PERTOKEN=1`): a handful of near-tie flips OK,
systematic divergence = a real glue bug. Perf: wall-clock spec-ON vs spec-OFF
(Claude licenses); expected ~(1+α)× at depth-1.

## Sequence

A1 (rollback, in flight) → re-run broken s_q=K on A1-clean to isolate the pure
glue bug (confirm it's the row-1..K builder fill) → implement A2b kernel extension
+ A2a/d/e → needle gate → perf A/B. A2 is the depth-1 ~1.5× lever (26.6→~18ms);
A3 (depth>1 tree, reuse single mtp.0 head) builds on a working A2.
