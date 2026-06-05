# DSv4 EAGLE/MTP Phase 2 tranche 1 — greedy verify loop CORRECT (identity PASS, α=0.6); per-token verify, s_q=K amortization is tranche 2

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** **correctness-complete, DEFAULT-OFF** (opt-in `ARLE_DSV4_SPEC_DECODE=1`,
which already loads the MTP head). This tranche delivers a **proven-correct**
depth-1 greedy verify loop + KV rollback. It is **NOT yet a perf win** — the
verify forward is per-token (no amortization, ~31% slower by forward-count); the
amortized **s_q=K FlashMLA** verify is tranche 2. Committing the correct,
strict-identity-proven baseline first deliberately isolates "state machine
correct" from "kernel numerics" for tranche 2. Plan:
[`2026-06-06-dsv4-eagle-mtp-phase2-verify-loop.md`](../../plans/2026-06-06-dsv4-eagle-mtp-phase2-verify-loop.md).

## What worked (correctness — the hard part)

- **Depth-1 greedy verify state machine** (`executor.rs` `forward_decode_tokens`):
  per-slot `Dsv4SpecSlotState{pending, hidden}` persisted across submit calls.
  Each decode step: `draft = mtp_forward(hidden, pending)`; `(argmax, hiddens) =
  forward_tokens_verify([pending, draft], start_pos)`. **Accept** (`argmax[0]==draft`):
  emit `[draft, bonus=argmax[1]]`, `pending=bonus`, `hidden=hiddens[1]`. **Reject:**
  `truncate_slot(keep_len=start_pos+1)` + `truncate_decode_len` (rolls back the
  compressor/indexer decode length too — not just KV pages), emit `[base_next=argmax[0]]`,
  `pending=base_next`, `hidden=hiddens[0]`.
- **Scheduler multi-token accounting** (`infer-core/lib.rs` `apply_output`): drains
  a `VecDeque<SlotToken>` per slot, advancing host `kv_seq_len` by the **real
  materialized count** (1 on reject, 2 on accept), with EOS/max checked per token.
  Backed by a **CPU regression test** (`decode_kv_seq_len_advances_by_speculative_output_count`,
  a depth-1 "spec mirror" backend) that catches host `kv_seq_len` drift without a GPU.
- **`forward_tokens_verify`** (`dsv4.rs`): all-position greedy argmax + per-row
  hidden. For strict identity it scores each verify position through the **same
  B=1 FlashMLA decode path** as baseline greedy (per-token loop) — this is the
  correctness baseline, replaced by one s_q=K forward in tranche 2.

## Verify (TP=8/EP=8, 8×H20, greedy)

- **Step A isolated self-test PASS:** `verify_two=[14,455]`; the verify forward's
  one-token argmax equals the normal single-token forward (`forward_tokens_verify([a])
  == forward_tokens([a])`).
- **Greedy-identity gate PASS — byte-identical:**
  `ARLE_DSV4_SPEC_DECODE=1` and unset both emit
  `[11111,14,305,270,6102,294,8760,344,11111,16,455,6102,294,8760,344,11111]`.
  This exercised 6 accepts + 4 rejects (the accept/reject/truncate/multi-emit
  paths) and STILL matched the baseline → the state machine + scheduler
  accounting + attention-state rollback are correct.
- **Acceptance α = 6/10 = 0.6** (raw `accept_total=6 reject_total=4`).

## Why no perf number yet (honest §0)

The per-token verify runs a full B=1 decode forward **per** verify token, so a
depth-1 round costs `1 MTP + 2 base` forwards: `E[cost]≈2.1 base/round`,
`E[tokens]=1+α=1.6` → **1.31 base/token = ~31% slower** than non-spec. It is
default-off; nobody should enable it for speed yet. The whole EAGLE win lives in
tranche 2: the K-token verify as **one** FlashMLA forward. The vendored kernel
**already supports it** — `api/sparse_decode.h get_meta(int h_q, int s_q)` is
parameterized by `s_q`; DSv4 only hard-codes `DSV4_FLASHMLA_S_Q=1` (`attention.rs:25`).
Threading `s_q=2` (=K) gives `E[cost]≈1.2 base/round` → `0.75 base/token = ~25%
faster` at α=0.6, scaling with α. The comm-overlap (`1b0222e7`) also compounds:
the s_q=K verify makes the overlapped shared expert 2× larger.

## Rule

Land the **strict-identity** speculative baseline before the amortized kernel.
The per-token s_q=1 verify proves the accept/reject/truncate/multi-emit logic is
correct against the exact production decode numerics; when tranche 2's s_q=K
FlashMLA breaks strict byte-identity (different query-tile float order — expected,
still a valid greedy generation), you **know** it's the kernel, not the state
machine, and gate tranche 2 on valid-generation (needle) instead. Adopt the
vendored kernel's existing `s_q` parameter — do not hand-roll a multi-token verify
([[feedback_no_closed_door_solutions]]). [[reference_dsv4_moe_nondeterminism_confounds_4096_parity]]
