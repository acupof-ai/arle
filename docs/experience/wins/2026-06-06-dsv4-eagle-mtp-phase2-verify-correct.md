# DSv4 EAGLE/MTP Phase 2 tranche 1 — verify loop + scheduler accounting landed; correctness INCOMPLETE (diverges on canonical prompt)

> **⚠️ SUPERSEDED 2026-06-06 (same day): the "correct" claim below was FALSIFIED.**
> Tranche 1 passed greedy-identity on ONE prompt ([11111]) but **DIVERGES on the
> canonical [344] prompt** (pure HEAD `625a4f06`, clean tree). Root cause: the
> reject rollback is incomplete for DSv4 compressed attention — see
> [`errors/2026-06-06-dsv4-eagle-rollback-compressor-gap.md`](../errors/2026-06-06-dsv4-eagle-rollback-compressor-gap.md).
> The committed code (`625a4f06`) is **default-off** so nothing regresses, but it
> is **NOT correct** and must not be enabled until the rollback is fixed. The
> scheduler multi-token accounting + state-machine *shape* are reusable; the
> per-token verify + the rollback are the broken/incomplete parts.

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** **default-off, correctness INCOMPLETE.** Commit `625a4f06`.

## What actually landed (and is reusable)

- **Scheduler multi-token accounting** (`infer-core/lib.rs` `apply_output`): drains
  a `VecDeque<SlotToken>` per slot, advancing host `kv_seq_len` by the real
  materialized count (1 on reject, 2 on accept), EOS/max per token. Backed by a
  **CPU regression test** (`decode_kv_seq_len_advances_by_speculative_output_count`).
  This part is genuinely correct and GPU-independent.
- **Depth-1 verify state-machine shape** (`executor.rs` `forward_decode_tokens`):
  per-slot `Dsv4SpecSlotState{pending, hidden}`; accept → emit `[draft, bonus]`,
  reject → `truncate_slot` + emit `[base_next]`. The control-flow is right; the
  **KV/attention rollback it relies on is incomplete** (the bug).
- **`forward_tokens_verify`** (`dsv4.rs`): per-token greedy argmax + per-row hidden
  (the ~31%-slower correctness baseline; the amortized s_q=K version is tranche 2,
  also failed — see the errors entry).

## What broke it

Greedy-identity held on the [11111] prompt (6 accepts / 4 rejects → byte-identical)
but **fails on the canonical [344] prompt**: first divergence at output index 21,
on a `reject (pending=34788 draft=271 base_next=45750)` where the **draft 271 was
the correct token** (spec-OFF `token[21]=271`) but the verify's base-position
argmax computed `45750`. Same kernel + same position as non-spec → the verify read
a **corrupted attention state**. The divergence is cumulative (clean through idx
20, then diverges), fingerprinting a DSv4 **compressor running-state** that the
reject rollback (`truncate_slot` + `truncate_decode_len`) does not revert. Full
trace + root cause in the errors entry.

## Lesson (the §0 miss)

A single-prompt greedy-identity gate is **insufficient** to call speculative
decode "correct" — the accept/reject *pattern* is prompt-dependent, and a prompt
whose rejects never trigger the buggy state path passes by luck. The canonical
[344] prompt (the one every other DSv4 A/B uses) must be in the gate. I committed
"correct" on [11111] alone; that was the error. See
[[feedback_spec_decode_gate_needs_multi_prompt]].
