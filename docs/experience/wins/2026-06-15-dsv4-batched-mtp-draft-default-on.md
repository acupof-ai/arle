# DSv4 batched MTP draft default ON

## Context

After `5d6eb0da` changed DSv4 batched `lm_head` projection from a per-row loop
to one cuBLAS GEMM over the batch, the old lever-2a verdict was stale. The prior
`ARLE_DSV4_BATCHED_MTP_DRAFT` experiment had been measured while
`lm_head_project_batch` still re-read the full bf16 vocab weight once per row,
so batching draft slots could not amortize the dominant read.

## What Worked

The lever was re-licensed after the batched `lm_head` fix:

| concurrency | delta |
|---:|---:|
| c=8 | +6.8% |
| c=16 | +11.1% |

The path now defaults ON. Operators can opt out with
`ARLE_DSV4_BATCHED_MTP_DRAFT=0`, which restores the proven per-slot draft chain.
The gate is only consulted when batched MTP is active.

## Rule

A sub-lever measured before its dependency becomes real is not a lasting verdict.
Once the batched `lm_head` path stopped looping per row, draft batching had to be
re-measured and defaulted based on the new same-binary A/B.
