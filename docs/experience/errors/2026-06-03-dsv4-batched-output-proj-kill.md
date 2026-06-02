# DSv4 Batched Output Projection Kill

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

After the batch FlashMLA arena substrate landed, a follow-up tried to batch the
compressed-attention output projection in `compute_top_level_logits_incremental_batch`.
The intended change was:

- row loop returns `local_attn`;
- batch `wo_a` over `[B, local_attn_width]`;
- batch `wo_b` over `[B, output_latent]`;
- reduce the projected hidden states.

That path was pushed as `ab1372382acaa85ef2ceea6ee56967b25c61a590`, then a
follow-up `8369947324e60ffe793e45ce018796f6f17ca778` changed the all-reduce
back to exact row buffers after fanout failed.

## Root Cause

The first failure was a real buffer contract bug:

`PostAttentionAllReduce buffer len 32768 does not match logical len 16384`

`LayerCommunicator::all_reduce_bf16` currently requires `CudaSlice::len()` to
match the logical collective length. The new batched `attn_out` scratch was
allocated for decode capacity, then used with a smaller current batch size, so
passing it directly to all-reduce was invalid.

The exact-row all-reduce follow-up fixed the HTTP 500 but did not make the path
correct. Remote debug-fallback validation at
`/tmp/dsv4_exact_row_attn_ar_20260603` showed:

- `decode64`: HTTP 200, 64 completion tokens, non-empty output.
- `math`: HTTP 200, output contained the correct `410`.
- `fanout_decode16`: 4/4 requests HTTP 200, 64 aggregate completion tokens.
- `dsv4_batched_decode_validate.py`: c4 batched outputs did not match c1 and
  degraded into repeated garbage tokens, while c8 had no HTTP errors.

That isolates the regression to the B>1 compressed-attention output projection
path, not to server startup or fanout transport. The code path uses DSv4
block-scaled FP8 batched linear dispatch (`dsv4_fp8_gemv_batch_cuda`) for
`wo_a` / `wo_b`; the old path used per-row single-token projection. Batched
Q/K/V projection is a separate already-validated path and is not killed by this
entry.

## Fix

Reverted both commits:

- `06ac4bfc` reverts the exact-row all-reduce follow-up.
- `5fd54d81` reverts the batched compressed-attention output projection.

The retained DSv4 tranche is the batch FlashMLA arena substrate. It does not
claim a TPOT win and does not change the old per-row output projection path.

## Rule

Do not batch DSv4 compressed-attention output projection until a focused parity
test proves B>1 `local_attn -> wo_a -> wo_b -> all-reduce` matches the per-row
path. Passing fanout without HTTP 500 is not correctness evidence; c1-vs-c4
decode output parity is the minimum gate before any performance discussion.
