# DSv4 EAGLE/MTP — reject rollback is incomplete for compressed attention; s_q=K verify glue broken; single-prompt gate hid both

## Context

DSv4 EAGLE/MTP Phase 2 (greedy speculative decode): draft via the MTP head,
verify against the base model, roll back KV on reject. Two tranches this session:

- **Tranche 1** (per-token verify, committed `625a4f06`, default-off): passed
  greedy-identity on the [11111] prompt (6 accepts / 4 rejects, byte-identical) →
  committed as "correct". **Wrong call.**
- **Tranche 2** (amortized s_q=K FlashMLA verify, NOT committed): the intended
  speedup. Structurally broke.

## Root Cause

**1. Reject rollback does not revert the DSv4 compressor running-state.**
On the canonical [344] prompt, pure tranche 1 (clean HEAD `625a4f06`) DIVERGES
from non-spec at output index 21:

```
... (idx 0-20 identical) ...
reject: pending=34788 draft=271 base_next=45750   <- spec emits 45750
spec OFF token[21] = 271                            <- truth is 271 (== the draft)
```

The draft (271) was correct, but the verify's argmax at pending's position
(`base_next`) computed `45750`. Same FlashMLA decode kernel, same position as
non-spec → the verify read a **corrupted attention state**. The divergence is
**cumulative** (clean through idx 20, two rejects at idx ~4-5 did not immediately
diverge), which rules out a per-step logic bug and fingerprints **state that
accumulates and surfaces later**: the DSv4 **compressor** ingests every appended
(incl. speculative draft) token into a running buffer and periodically emits a
compressed KV block. The reject path reverts KV pages (`truncate_slot`) and the
decode-length counter (`truncate_decode_len`), but **not** the compressor's
running ingestion — so a rejected draft's contribution persists and corrupts a
later compressed block, mis-computing attention downstream. (The [11111] prompt's
reject pattern happened not to cross a compression boundary with a poisoned
buffer → passed by luck.)

**2. s_q=K FlashMLA verify glue is numerically wrong.** Tranche 2 wired the verify
as one `s_q=K` FlashMLA forward (the kernel supports it: `params.h` `s_q`,
`q[b,s_q,h_q,d_qk]`, `indices[b,s_q,topk]`; `sparse_decode.h:208` reads `s_q` from
`q.size(1)`). Result: structural divergence (decode collapses into a
`14,818,14,818…` loop), **3× SLOWER** (12.2 vs 38.9 tok/s), acceptance 0.6→0.24.
The per-query top-k indices / causal-mask construction for K>1 query rows is wrong
(the s_q=1 decode indices builder was specialized; the s_q>1 prefill builder it
references wasn't correctly reused for the verify's KV coord space). Diff saved
`/tmp/tranche2_sqk_broken.diff` (681 lines) for a future fix attempt — NOT committed.

**3. The gate hid both.** A single-prompt ([11111]) greedy-identity gate passed,
so tranche 1 shipped "correct". The bug is prompt-dependent (reject pattern must
cross a poisoned compression boundary). The canonical [344] prompt exposes it
immediately.

## Fix

- **FIXED (2026-06-06, same day).** Complete snapshot/restore of the full mutated
  state — see [`wins/2026-06-06-dsv4-eagle-rollback-fix-correct.md`](../wins/2026-06-06-dsv4-eagle-rollback-fix-correct.md).
  The §0.1 mutated-buffer enumeration surfaced what the partial fix missed:
  compressor + indexer running buffers (4 small fields each, NOT `compressed.data`)
  + `sw_window_cache` one ring slot + FlashMLA `fp8_kv_pool` SW one ring slot
  (split token-data/scale) + the `fp8_kv_comp_packed_rows` scalar. Pre-allocated,
  spec-path-only, single-slot (`O(head_dim)`). Mechanism dump confirmed the bug was
  real; correct-inference gate (needle short seq<W AND long seq≥W + same-twice
  floor) PASSES. Spec decode is now CORRECT but still −32% (per-token verify);
  **A2 (s_q=K) is the amortization.**
- **Tranche 2** s_q=K glue needs the per-query index/mask build fixed against the
  decode KV coord space before it can be numerically valid (and only then is it a
  perf candidate).

## Rule

1. **Speculative-decode correctness gates need ≥2 prompts, including the canonical
   one, BEFORE the "correct" claim.** The accept/reject pattern is prompt-dependent;
   one passing prompt proves nothing about the rollback paths a different pattern
   exercises. [[feedback_spec_decode_gate_needs_multi_prompt]]
2. **Compressed-attention models (DSv4 CSA/HCA) make speculative rollback hard:**
   reverting KV pages + the position counter is NOT enough — derived running state
   (compressor buffer, cross-layer indexer cache) must also be reverted or the
   draft must not feed it until accepted. This is the real cost of MTP/EAGLE on a
   compressed-attention model; it is not a wholesale-copy from vanilla-attention
   EAGLE. [[feedback_no_closed_door_solutions]]
3. When a contaminated tree (dirty broken experiment) reports a divergence in an
   *adjacent* committed path, **re-test the committed path on a clean checkout
   before trusting the claim** — here the "tranche 1 diverges" claim was real, but
   it had to be confirmed on pure HEAD, not the dirty s_q=K tree, to be evidence.
