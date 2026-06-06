# DSv4 EAGLE rollback FIXED — verify loop now CORRECT (complete compressor/indexer/sw/fp8 rollback); per-token still −32%, A2 is the speedup

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** **correctness milestone, default-off** (`ARLE_DSV4_SPEC_DECODE=1`).
Resolves [`errors/2026-06-06-dsv4-eagle-rollback-compressor-gap.md`](../errors/2026-06-06-dsv4-eagle-rollback-compressor-gap.md)
(the tranche-1 verify loop diverged on the canonical prompt). The EAGLE verify
loop is now correct on a short AND a long needle; spec decode is still slower than
non-spec because the verify is per-token — **A2 (s_q=K, one forward) is the
amortization** ([`2026-06-06-dsv4-a2-sqk-verify-detail.md`](../../plans/2026-06-06-dsv4-a2-sqk-verify-detail.md)).

## What worked — the complete mutated-buffer enumeration (§0.1)

The reject rollback now reverts EVERY buffer the speculative draft mutates (the
partial tranche-1 fix reverted only the KV pages + `compressed.seq_len`):

| buffer | revert |
|---|---|
| main paged KV | `truncate_slot` (existing) |
| compressor + indexer `pending_kv/pending_score/prev_overlap_kv/prev_overlap_score` | snapshot/restore the 4 small running buffers (NOT `compressed.data` — self-heals via `compressed.seq_len`, which is saved as a scalar) |
| `sw_window_cache` | **one ring slot** at `draft_abs_pos % sliding_window`, `head_dim` bf16 |
| FlashMLA `fp8_kv_pool` SW sub-pool | **one ring slot** (token-data bytes + scale bytes, the real split layout) |
| `fp8_kv_comp_packed_rows` | scalar (so a rejected boundary compression can't skip repacking the next real row) — **found by the enumeration**, beyond the original spec |

- **Speed-safe**: snapshot buffers allocated ONCE per slot (only when
  `ARLE_DSV4_SPEC_DECODE=1`), captured by D2D; single-slot copies are `O(head_dim)`,
  not `O(ring)`. **Spec-off path byte-for-byte untouched** (verified: spec-off
  36.928 tok/s ≈ committed baseline).
- Captured after `pending` is forwarded/committed, before the `draft` forward;
  restored on reject (with `keep_len` threaded for the slot index).

## Verify — mechanism + correct-inference gate

- **Mechanism dump confirmed the bug was REAL** (not MoE non-determinism): before
  the fix, layer-0 SW slot + layer-2 CSA compressor/indexer `pending` changed after
  a reject; compression-boundary rejects also changed `prev`/`compressed` checksums.
- **Correct-inference gate** (needle + same-config-twice, NOT byte-identity —
  [[feedback_correct_inference_not_baseline_identity]]):
  - SHORT (seq_len=27 < sliding_window): spec-off×2 + spec-on all identical
    `" 4271.\nCorrect answer: Yes.\nNow, consider"`, accept/reject 4/3.
  - LONG (seq_len=145 ≥ sliding_window, exercises the sw/fp8 ring revert): needle
    "Paris" retrieved, spec-off & spec-on matched `" Paris.\nThe capital of France
    is Paris"`, accept/reject 2/3.
- Build: pod CUDA PASS (12.92s); local `cuda,no-cuda` PASS.

## Cost (honest — A1 is correctness, not yet perf)

Canonical 64-tok: spec-off **36.928** → spec-on **25.116 tok/s (−32%)**, α≈0.47.
The slowdown is the per-token verify (2 base forwards + 1 MTP draft per round), NOT
the snapshot (single-slot D2D is ~µs). A2 collapses the K verify tokens into ONE
FlashMLA s_q=K forward → the actual ~(1+α)× speedup. A1 unblocks A2: the rollback
is now correct, so a correct *fast* verify is buildable on this base.

## Rule

The complete mutated-buffer enumeration (§0.1) is what surfaced the two buffers a
"obvious" fix misses (`sw_window` + `fp8` ring slots) AND a third nobody listed
(`fp8_kv_comp_packed_rows`). Ring buffers self-heal a speculative write ONLY for
`seq_len < ring_size`; the long-needle (seq ≥ sliding_window) is the test that
proves the ring revert. Gate on correct inference, not baseline byte-identity.
