# Mixed packed prefill review gaps

## Context

The first packed mixed decode+prefill refactor landed before CUDA runtime
validation. A follow-up review found correctness and validation gaps in the
mixed launch path.

## Root Cause

- The mixed model path copied decode logits into the shared decode context, but
  the scheduler's non-greedy fallback samples from per-slot logits. The regular
  decode path already scattered logits when `skip_logit_scatter=false`; mixed
  launch missed the equivalent fallback preparation.
- The shared-tail COW unit test asserted that a second append still fit after
  consuming the only free page. That contradicted the physical page accounting
  the test was meant to protect.
- Mixed scheduling allowed all selected prefill rows into one TC FlashInfer
  plan. Per-request chunk caps did not bound the combined QO row count.

## Fix

- Prepare per-slot sampling fallback logits for non-greedy mixed launches.
- Correct the COW budget expectation: after the COW page is consumed, the next
  page-growing append must not fit without another free page.
- Cap total mixed prefill tokens at the long-prefill cap and allocate the mixed
  Qwen3 FlashInfer metadata with the large split-KV workspace.

## Rule

Mixed decode+prefill must satisfy both decode contracts: greedy can consume
`decode_ctx.logits_batch` directly, but non-greedy fallback must see fresh
per-slot logits before readback sampling. Also, mixed packing needs a total QO
row cap; per-request prefill caps are not enough.
