# DSv4 MTP frozen-KV — long-context P1 correctness fix

**VALIDATED on 8×H20** (`e945a3f2`, `scripts/dsv4_needle_gate.py`, 2026-06-12).
Long-context needle with MTP spec-decode active — `sliding_window=128`, so 3k/6k
fully wraps the SW ring and crosses many compression boundaries:

| length | depth-1 | depth-4 (accept 0/4–1/4, reject-heavy) |
|--------|---------|------------------------------------------|
| 3000 (depth 0.5) | **exact ×3** | **exact ×3** |
| 6000 (depth 0.5) | **exact ×3** | **exact ×3** |

depth-4 rejects on nearly every step (`reject_total=132` vs `accept_total=52`),
so `restore_spec_ring_tail` fires constantly into a wrapped ring — and the secret
code retrieves exactly every run. "NONDET" classifier label = MoE continuation
variance, not corruption (the code is exact ×12). **codex's P1-C (HCA decode
count) does NOT manifest** at these lengths; per §0, the gate supersedes the
source-analysis hypothesis — not chased.

## Context

Frozen-KV MTP verify (`476da9d7`) freezes the compressor during the K-token
speculative verify so it forms no new compressed blocks — fixing the depth-K
draft corruption whose root cause was the compressor mutating on speculative
tokens. Short-context validation passed: depth-4 coherent
(Paris/Berlin/Rome/Madrid), draft0 100%, needle clean, no "human heart" loop.

That commit also deleted the whole rollback unit (capture/restore snapshot)
as a clean deletion-refactor. **`codex review` + the 2026-06-06 rollback
lesson then surfaced two P1 holes the short-context tests never exercised** —
the deletion of the SW/FP8 snapshot was premature for long context.

## Root cause (two P1s)

- **P1-1 — compression-boundary phantom block.** The frozen verify skipped the
  compressor/indexer CUDA update but `compressor_forward` still ran
  `state.compressed.seq_len = compressed_rows`. So CSA/FlashMLA in the *same*
  verify attended to a compressed/indexer row whose data was never produced,
  and `csa_select` advanced DSA `packed_rows` off `indexer_rows_after`. Only
  bites when the K verify tokens straddle a compression boundary → invisible
  to short prompts.
- **P1-2 — sliding-window ring wrap.** The SW ring (`sw_window_cache`) and FP8
  ring are written by the verify for the K+1 positions and are *not* frozen
  (the draft chain needs its own KV). When `start_pos >= sliding_window` a
  rejected draft's ring write **aliases a still-active window slot**; the
  commit truncate only resets lengths, so the next decode reads corruption.
  Traced concretely (W=4, L=10, K=2, reject n=1): rejected draft at pos 12
  overwrites slot 0, which still holds pos 8 — needed by the next window
  [8,9,10,11]. The deleted snapshot was **single-slot** (depth-1-correct only —
  the historical depth-K bug); truncate's `seq_len`-reset self-heal works
  *only* when `seq_len < sliding_window`.

## What worked

§0.1 complete buffer enumeration — every buffer the frozen verify mutates,
proven:

| buffer | disposition |
|--------|-------------|
| compressor data | frozen (gate in `compressor_forward`) |
| compressor + indexer `seq_len` | **P1-1**: gate the `seq_len` assignment too. Both route through `compressor_forward`; DSA packs off `indexer_rows_after`, so freezing the length keeps `after == before` → indexer + DSA `packed_rows` + `dsa_key_cache` all frozen by one gate |
| SW ring + FP8 ring + `fp8_kv_comp_packed_rows` | **P1-2**: pre-allocated K+1-slot `Dsv4SpecRingSnapshot`, captured BEFORE any speculative write, `restore_spec_ring_tail` restores the rejected tail `[n+1..=depth]` on partial accept |
| dense latent KV | self-heals via truncate + accepted-prefix re-forward |

- **One gate freezes the whole compressed+sparse+DSA path** (P1-1) — because
  compressor and indexer share `compressor_forward` and DSA derives from
  `indexer_rows_after`.
- **Capture must precede the draft loop, not just the verify** (caught in
  review): `mtp_forward → run_mtp_transformer_layer → mla_attention` also
  writes the *frozen target layer's* SW/FP8 ring at `start_pos+i`. Capturing
  after the draft loop would snapshot draft-polluted slots for that layer and
  restore garbage. Capturing before any speculative write is committed-state
  for all layers.
- K+1 slots (not single-slot) → depth-K correct.

## Perf state

Option A always-re-forwards the accepted prefix (correctness-first), so it is
currently **net-negative**: pod 8×H20 depth-1 20.5 / depth-4 16.1 tok/s
(deeper-is-slower; baseline no-spec ~44). The wall-clock win is the
follow-up — **compressor-only commit** (drop the re-forward; the SW/FP8 KV is
already written by the verify, only the compressor needs committing for the
accepted prefix) + **batched s_q=K verify** (amortize the per-token attention)
— both unblocked by the freeze. Tracked as the MTP perf axis.

## Rule

- **A "clean deletion" of rollback machinery must pass the §0.1 buffer
  enumeration, not just the short-context test.** Freezing the compressor
  retired the *compressor* snapshot, but the *SW/FP8 ring* snapshot was still
  load-bearing for `seq_len >= sliding_window` rejects. Short-context tests
  never wrap the ring or cross a compression boundary, so they cannot license
  deleting wrap/boundary recovery.
- **Speculative ring snapshots capture before the FIRST speculative writer.**
  The MTP draft path writes the frozen target layer's ring too — capture
  before the draft loop, not before the verify.
- **The correctness gate for this is the long-context + SW-wrap + spec-decode
  needle** (depth > sliding_window), not the smoke prompt. See
  [[feedback_spec_decode_gate_needs_multi_prompt]],
  [[reference_dsv4_deepep_ll_and_lockstep_state]].
