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

3. *Flat-prefix compare of two differently-pitched index buffers* (the
   verification gate's own fault, found on the card after fix 1/2 removed the
   inf). The check compared
   `indices[..s_q*topk_unified] == ref_indices[..s_q*topk_unified]`:
   the device vector at HCA pitch 256 and the oracle at fixed pitch 640. The
   first 32768 ints of the device buffer hold 128 complete 256-wide rows; the
   oracle's first 32768 hold only ~51 rows and then cross a 640-row boundary,
   so the slices differ by construction and the gate reports `indices=false`
   even though every per-token row agrees — which the per-token diagnostic
   loop (correctly) printed no mismatch from. The compare itself was the
   false arm, not the kernel. The fix compares per token:
   `indices[t*topk_unified+k] == ref_indices[t*TOPK+k]` over
   `k < topk_unified`. Lesson: two buffers with different row pitches cannot
   be compared as one flattened prefix; stride each slice by its own pitch or
   the check measures the stride difference, not the content.

Lesson: when a conclusion rests on argument position, print the signature
and call side by side and number them; when making a test's numbers agree,
prefer mirroring how production derives them over pinning a constant that
hides the varying path; and when a gate reports a mismatch its own
per-element diagnostic cannot find, suspect the gate's aggregation
(here: flattening across two pitches) before the producer.

4. *The pre-existing index-builder negative tooth cannot fire its output
   checks* (main `9a19d0f01`, first GPU execution 2026-09-13 — the arm had
   never run). The corruption swaps one CSA compressed selection entry at the
   last token, but the harness builds compressed rows at magnitude ±0.02
   (`comp` latent, example:213-218) while Q is aligned to the chunk keys, and
   the swapped key is one of ~640 in the softmax. Measured clean-vs-corrupted
   delta: output rel 0.00542 vs threshold 0.05, lse 1.1e-5 vs 0.10, maxlogit
   0.0 vs 0.05 — `indices=true` (the kernel honors the bad indices, and the
   kernel-built indices match the corrupted oracle) but out/lse/maxlogit all
   read `false` (did not fire) and the run exits 1 with
   "output negative control did not fire". The tooth is too weak by
   construction, not a kernel fault. Fix direction is an index corruption that
   swaps in a high-weight key (e.g. a SW/chunk-row slot the token is causally
   allowed to read) so the output must move; deferred to the owner of
   `9a19d0f01`.

## Card result

Measured 2026-09-13 on H20 sm90 card 1 (`--features cuda`, release,
build id c0ee7deef…), 18 positive cases (CSA/HCA × chunks 128/2048/4096 ×
starts 63/128/255): all `pack=true indices=true topk=true`; HCA output
rel 0.00072–0.00083, lse abs 0.00001, maxlogit abs 0.00008–0.00011;
`ALL PASS`, rc=0. HCA output error is in the CSA range (~7e-4), as
predicted. The `--negative-control` invocation is recorded separately.
Tolerance constants at lines 94-96 are unchanged.

## Rule

A producer and a consumer of a row-major buffer must agree on the row
pitch; the capacity argument to the producer and the stride/alloc given to
the consumer are one invariant. When a gate fails on only one of two
shapes that share a consumer, diff the producer pitch before the consumer
math.
