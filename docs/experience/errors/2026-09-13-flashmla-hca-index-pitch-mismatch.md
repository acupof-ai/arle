# FlashMLA HCA parity failed: harness built index rows at 256 pitch, consumer read at 640

Date: 2026-09-13. Found from reading the P1 positive failure of
`flashmla_prefill_parity`; card confirmation pending.

## Context

`HCA s_q=128 start=63 ... indices=false out=0.96677 lse=inf maxlogit=inf`
on the first HCA case, after nine CSA cases passed at output error ~6e-4.
The sparse prefill gate compares the vendor FlashMLA SM90 kernel against
independent f32/f64 host math (`oracle_indices`, `reference_row`).

## Root cause

The harness gave the HCA index builder and the index consumer different
per-token row pitches on one buffer:

- Producer: `crates/cuda-kernels/csrc/attention/arle_flashmla_csa_prep.cu:115`
  writes `indices + token * topk_unified`, and HCA sets
  `topk_unified = sw_window + max_compressed_keys` (`:317`). The example
  passed `max_keys = compressed_count.div_ceil(128)*128 = 128`
  (flashmla_prefill_parity.rs old :346), so the write pitch was
  128+128 = **256**.
- Consumer: vendor `phase1.cuh:470` reads
  `params.indices + s_q_idx * params.stride_indices_s_q`; the example
  passed stride `TOPK = sw_window + index_topk = 640`, and allocated
  `s_q * TOPK` at example:322. Read pitch **640**.

Token 0 sits at offset 0 under both pitches and is also the all-SW row
(start=63 ⇒ 0 compressed keys), so it cannot expose the mismatch. Token 1
was written at 256 but read at 640, landing on zeroed memory / the prior
row: a genuinely different (shorter, invalid) index set, so attention
gathered an unselected pool slot and the running max `rM` went to +inf —
hence `lse=inf` and `maxlogit=inf` (`max_logit = rM`,
reference_row:825) together with `out=0.97`. The inf is a consequence of
the same fault, not a second bug.

CSA was immune because its builder pitch is
`sw_window + index_topk = 640` (arle_flashmla_csa_prep.cu:290), equal to
the consumer pitch, and the harness replicates one identical selected row
per token (example:242-247). `pack=true` and `topk=true` held because
those checks read the correctly-strided KV pool and the 1-int-per-token
length array; only the strided index buffer disagreed.

The builder obeys its FFI contract: `ffi/misc.rs:746-748` defines the row
pitch as `sw_window + max_compressed_keys` on a caller-allocated
`s_q * (sw_window + max_compressed_keys)` buffer. Production
(`crates/infer-cuda/src/attention.rs`) is self-consistent — it allocates
`token_count * topk_unified` (:1363) and passes `topk_unified` as both
the fwd `topk` (:1673) and `stride_indices_s_q` (:1679). Only the test
harness mixed a 640 buffer/stride with a 256-capacity builder argument.

## The hypothesis that looked right and was not

First reading blamed `stride_indices_s_q=0`. That was a positional-argument
miscount: numbering the 24 parameters against the call (signature
attention.rs:1184, call example:379-404) puts `stride_indices_s_q` at
param 21 = `TOPK` (640); the `0` at param 20 is `stride_kv_h_kv`. With
`h_kv == 1` (shim:66) the sparse prefill kernel never references
`stride_kv_h_kv` (grep: zero uses under
vendor/flashmla/csrc/sm90/prefill/sparse), so that zero is dead, not a
live multiply. Lesson: when a conclusion rests on argument position, print
the signature and call side by side and number them — counting by eye
against an unenumerated signature produced the wrong mechanism.

## Fix

Derive the per-mode pitch once and propagate it to the buffer, the builder,
and the consumer — the shape production already uses:

- `max_keys` stays `compressed_count.div_ceil(128)*128` for HCA
  (`INDEX_TOPK` for CSA); `let topk_unified = SW + max_keys;`
- allocate indices as `s_q * topk_unified` (not `s_q * TOPK`);
- pass `topk_unified` as both the fwd `topk` and `stride_indices_s_q`.

CSA then carries `SW + INDEX_TOPK = 640` and HCA its variable
`SW + ceil128(compressed_count)`, exactly the per-mode pitches production
derives. Two `ensure!` guard the invariant: `topk_unified % 128 == 0`
(mirrors the builder guard at `csa_prep.cu:318`) and the allocated vector
length equals `s_q * topk_unified`, so a future pitch drift fails loudly
instead of producing inf.

Two wrong fixes were considered before this one:

1. *`stride_indices_s_q=0`* (the first hypothesis). A positional-argument
   miscount: numbering the 24 parameters against the call (signature
   attention.rs:1184, call example:379-404) puts `stride_indices_s_q` at
   param 21 = `TOPK` (640); the `0` at param 20 is `stride_kv_h_kv`. With
   `h_kv == 1` (shim:66) the sparse prefill kernel never references
   `stride_kv_h_kv` (zero uses under
   vendor/flashmla/csrc/sm90/prefill/sparse), so that zero is dead, not a
   live multiply.
2. *Freeze HCA `max_keys = INDEX_TOPK` (512)* (the first proposed patch).
   It makes alloc/builder/consumer numerically agree at 640, but by
   freezing the pitch it stops the gate exercising the variable
   `compressed_count`-derived pitch production actually runs — the exact
   bug class would never be caught again. A gate that agrees by no longer
   testing the thing is not a fix.

Lesson: when a conclusion rests on argument position, print the signature
and call side by side and number them; and when making a test's numbers
agree, prefer mirroring how production derives them over pinning a
constant that hides the varying path.

Card confirmation (prediction): with the derived pitch `indices=true`,
`lse` and `maxlogit` are finite, and HCA output error lands near the CSA
~6e-4. If the inf survives there is a second fault and investigation stops
rather than widening the tolerance. Tolerance constants at lines 94-96 are
unchanged.

## Rule

A producer and a consumer of a row-major buffer must agree on the row
pitch; the capacity argument to the producer and the stride/alloc given to
the consumer are one invariant. When a gate fails on only one of two
shapes that share a consumer, diff the producer pitch before the consumer
math.
