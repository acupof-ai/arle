# Qwen3.6 radix prefix reuse is unsound for the hybrid recurrent state

## Context

Phase 3 (`893a93fd`, "radix prefix reuse for Qwen3.6, uniform mechanism") flipped
`Qwen35CudaExecutor::reusable_prefix_blocks` from a hardcoded `0` to delegate to
`pages_only_reusable_prefix_blocks`, so the scheduler began reusing a matched
radix prefix for Qwen3.6 (setting `row.start_pos = matched_len > 0`). On a
repeated-prefix workload the serve then crashed:

```
PagedKVPool seq_len 6033 != materialized total_len 6016 for slot 0
  at crates/infer-cuda/src/decode_graph.rs:124
```

## Root Cause

Two layers, both fatal:

1. **Accounting divergence.** The dense Qwen3 prefill is host-mirror-driven:
   `submit` (`executor.rs:1025`) calls `mirror_slot(slot, host_pages, start_pos +
   tokens.len())`, which SETS the device pool `seq_len` from the engine's
   host page table (already carrying the attached prefix pages). The Qwen3.6
   default-paged path is *self-allocating* — `prefill_row_paged_default`
   (`executor.rs:3351`) calls `pool.alloc_tokens(slot, row.tokens.len())`, which
   INCREMENTS `seq_len` by the tail length only, and the device slot is freed
   only at `start_pos == 0` (`submit_prefill_row`, `executor.rs:3969`). On a
   reuse (`start_pos > 0`) the slot is never freed, so the device pool carries
   the prior occupant's `seq_len` (6016 prompt + 17 decode = 6033) while the
   engine credits `start_pos + tail = 6016` — they diverge, surfacing many steps
   later at the `decode_graph` ensure.

2. **The reuse is architecturally unsound regardless (the deeper kill).**
   Qwen3.6 is a HYBRID: ~48 gated-delta *linear* layers carry per-slot recurrent
   + conv state that is content-based, position-free, and NOT page-addressable
   (`Qwen35SlotState`, `qwen35.rs:472` — "do NOT self-heal under a seq_len
   rewind"). The radix cache keys on full-attn KV *pages* only; it has no way to
   restore the recurrent state a reused prefix would have produced. So even with
   the accounting fixed, the full-attn layers would attend correct reused KV
   while the linear layers see a slot whose recurrent state never processed the
   prefix → silently wrong output. Dense Qwen3 is pure full-attention, so reuse
   is sound there — the "uniform mechanism" premise of `893a93fd` does not carry
   to the hybrid.

## Fix

`crates/infer-cuda/src/executor.rs` only:

- **`reusable_prefix_blocks` soundness gate.** Reuse is licensed ONLY for a pure
  full-attention checkpoint: `num_full_attention_layers() == num_hidden_layers`.
  Otherwise return `0`, so the scheduler never sets `start_pos > 0` and every
  Qwen3.6 prefill starts fresh at position 0 — byte-identical to the no-reuse
  path, recurrent state always correct.
- **Defense-in-depth `ensure!`s.** `prefill_row_paged_default` now asserts
  `pool.seq_len(slot) == row.start_pos` before `alloc_tokens` (matches the
  engine's chunked-prefill contract at `infer-core/lib.rs:1817`), and
  `decode_row_paged_default` asserts `pool.seq_len(slot) == row.kv_seq_len`. A
  stray reuse fails loudly at the append instead of as a confusing `decode_graph`
  mismatch many steps later. Legitimate fresh (`start_pos==0`) and chunked
  (`start_pos>0`, no reuse) prefills both satisfy the invariant.

Verified: clean-tree typecheck `cargo check -p infer-cuda --release
--no-default-features --features cuda,no-cuda --lib` (CUDARC_CUDA_VERSION=12080)
compiles with zero errors (the dirty working tree's only error is unrelated
diffusion/multimodal WIP in `infer-seam`). pending-remote: repeated-prefix GPU
A/B on the 8×H20 pod — request B reuses request A's prefix → with the gate,
Qwen3.6 reports radix hit = 0 and re-prefills cleanly (no crash, correct output);
a follow-up wins entry confirms.

## Rule

**A page-keyed prefix cache is only sound for models whose entire attended state
is page-addressable.** Hybrid (linear/recurrent + full) and SSM/Mamba-class
models keep per-slot recurrent state that no page cache can restore for a reused
prefix — gate prefix reuse on "pure full-attention", never assume a dense-tested
mechanism transfers to a hybrid. Re-enabling it needs a prefix-keyed
recurrent-state snapshot/restore, not just page attach. And: when two parallel
length trackers must agree (engine `kv_seq_len` vs device pool `seq_len`), assert
the invariant at the *mutation* (the append), not only at the far-downstream
consumer — the crash site is then the bug site.
