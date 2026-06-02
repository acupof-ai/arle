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

Remote verification after the revert, commit
`0d8bc089c142cd3b3fc30e49bc0e2d62f46d43b2`:

- release-fast build passed in 18.47s using the DSv4 prebuilt CUDA artifact
  fast path: `/tmp/dsv4_revert_output_proj_20260603/build.log`.
- debug-fallback smoke returned HTTP 200 for decode64, prefill1k, prefill4k,
  math, write_zh, and fanout=4:
  `/tmp/dsv4_revert_output_proj_20260603/debug_fallback_trace_summary.json`.
- fanout=4 produced 64 aggregate completion tokens with no HTTP errors; outputs
  were normal text again, not the repeated garbage-token pattern from the killed
  output-projection path.
- `dsv4_batched_decode_validate.py` completed c8 with zero HTTP errors. Its
  c1-vs-c4 byte-identical check still reported `PARITY_OR_C8_FAIL`, but c4 rows
  contained the correct `406` answer text rather than garbage tokens. That
  script currently returns 0 even on the printed parity failure, so strict
  deterministic parity needs a separate gate fix before it can be used as the
  sole correctness verdict:
  `/tmp/dsv4_revert_output_proj_20260603/batched_decode_validate.log`.
- After the smoke, no `infer` process remained and `nvidia-smi` reported no
  compute apps.

## Rule

Do not batch DSv4 compressed-attention output projection until a focused parity
test proves B>1 `local_attn -> wo_a -> wo_b -> all-reduce` matches the per-row
path. Passing fanout without HTTP 500 is not correctness evidence; c1-vs-c4
decode output parity is the minimum gate before any performance discussion.
