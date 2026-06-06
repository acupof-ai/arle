# DSv4 EAGLE s_q=K verify KILLED — compressed attention makes the K-token verify both non-equivalent to autoregressive AND non-amortizing

> **⚠️ CORRECTION 2026-06-06 (same day): this "fundamental" conclusion was WRONG.**
> The divergence + slowdown were artifacts of MY implementation (I re-ran
> `dsv4_compressor_update` DURING the verify → re-compressed mid-batch) and a
> WRONG workload (synthetic 64-tok, not GSM8K/ShareGPT). SGLang does native MTP on
> exactly this sparse attention (DSA, `dsa_mtp_fixture.py`, DeepSeek-V3.2,
> accept_length≈2.7) via **frozen-KV**: the draft+verify READ the frozen target KV
> and never re-run the compressor. With the compressor frozen, an off-boundary
> K-draft span is EXACTLY autoregressive and the prepare-chain is paid ~once →
> correct AND amortizes. See the redesign:
> [`docs/plans/2026-06-06-dsv4-frozen-kv-mtp-redesign.md`](../../plans/2026-06-06-dsv4-frozen-kv-mtp-redesign.md).
> Lesson below (axis 2) — I tested a smoke shape AND mutated frozen-should-be state.

## Context

EAGLE/MTP Phase 2 on DSv4: A1 (the per-token greedy verify) landed CORRECT
(`25a92e8a`, needle short+long PASS) but is −32% (25.1 vs 37.9 tok/s) because the
verify runs 2 base forwards per round. A2 was meant to collapse the K verify tokens
into ONE FlashMLA s_q=K forward → the standard EAGLE amortization (~(1+α)× ). It
**fails for DSv4 on two independent axes.**

## Root Cause

**1. The K-token batch forward is NOT equivalent to autoregressive (correctness).**
s_q=K verify per-position argmax systematically diverges from the per-token
reference (`ARLE_DSV4_MTP_VERIFY_PERTOKEN=1`): first mismatch row1 pos68, row0
pos85. **Decisive control: with `ARLE_DSV4_FLASHMLA_DECODE=0` (scalar attention) it
STILL diverges** (row1 pos74, row0 pos84) — so it is NOT a FlashMLA-glue bug; the
*generic multi-token forward* differs from K sequential forwards. The mismatch
positions cluster near compression boundaries (compress_ratio≈16 → 64, 80): DSv4's
**stateful compressed attention** (the compressor accumulates a chunk and compresses
on a boundary; the indexer/SW state evolves per token) makes a 2-token batch that
straddles a boundary compress differently than token-then-token. Standard causal
attention has batch == autoregressive; DSv4's compressor/indexer/SW statefulness
breaks that invariant.

**2. The K-token forward does not amortize (perf).** spec-OFF 37.9 → spec-ON s_q=K
**12.85 tok/s** — 3× SLOWER, slower even than the per-token A1 (25.1). EAGLE
amortizes only when a K-token forward ≈ 1-token cost (weights dominate, attention
cheap). DSv4's per-query **prepare-chain** (`dsv4_csa_select` bitonic top-512 +
compressor update + indexer) is expensive and runs PER query token, so a K-token
verify is ~K× the attention, not ~1×. Same wall that killed FlashMLA-prefill
(`docs/.../2026-06-06-dsv4-decode-6ms-remaining-levers.md`: the prepare-chain
overhead exceeds the attention-math savings).

## Fix

**Killed, not fixed.** The simple s_q=K verify is dead. A viable EAGLE-on-DSv4
would need BOTH (a) a boundary-aware verify that never straddles a compression
boundary (or replays the compressor per draft token) for correctness, AND (b) a
shared/cheap prepare-chain so the K-token forward amortizes — a deep, uncertain
redesign. The s_q=K attempt is saved at `/tmp/a2_sqk_attempt.diff` (not committed);
the tree stays at A1 (`25a92e8a`, correct per-token, default-off).

## Rule

**Before assuming EAGLE/spec-decode gives a speedup on a model, verify the K-token
verify forward AMORTIZES — that batch(K) ≈ cost(1) AND batch(K) == autoregressive.**
Both hold for vanilla causal attention; BOTH FAIL for compressed/sparse attention
(DSv4 CSA/HCA) with a stateful compressor + an expensive per-query selection
prepare-chain. The real DSv4 decode lever is therefore the **prepare-chain itself**
(`csa_select` top-512 / compressor / indexer) — making it cheaper or shared-across-
queries would speed raw decode AND be the prerequisite for any EAGLE amortization.
[[feedback_correct_inference_not_baseline_identity]] (the gate that exposed the
non-equivalence) · the per-token A1 stays the correct-but-slow fallback.
