# Short Requests Should Not Inherit Long-Prompt Prefetch And PD Split Assumptions

## Context

During the 2026-04-29 c=16 performance push, the active diagnosis was
centered on 4096-token prompt benchmarks and K2 prefill/decode interleave.
ckl corrected the scope: short requests do not necessarily benefit from the
same prefetch or prefill/decode split machinery.

## Root Cause

The optimization framing overgeneralized from long-prompt saturation benches.
For short prompts, the overhead and queueing complexity of prefix prefetch,
staged KV recall, or forced prefill/decode separation can dominate the small
amount of actual prefill work.

## Fix

When changing scheduler admission, prefetch, mixed batching, or PD split
behavior, explicitly segment by prompt length. Validate short-request behavior
separately from 4096-in canonical throughput runs.

## Rule

Do not apply long-prompt prefetch or PD-split assumptions to short requests
without a short-prompt benchmark. Short prompts need a bypass/fast path when
prefetch or split overhead exceeds the saved compute.
