# Qwen3.5 fp8 KV 0-token Completions

## Context

Qwen3.5 fp8 KV `guidellm` at c=16 / 4096 prompt / 256 output reported
TTFT/ITL p50 = 0 and server logs showed `Request N done: 0 tokens`. Raising
`--max-seq-len` from 4352 to 8192 did not fix it.

Trace run:

- Server: `/tmp/arle-target/release/infer --model-path infer/models/Qwen3.5-4B --port 8000 --num-slots 16 --max-seq-len 8192 --kv-cache-dtype fp8`
- Bench: `scripts/bench_guidellm.sh arle-qwen35-fp8kv-zero-token-trace --target http://localhost:8000 --model Qwen/Qwen3.5-4B --processor infer/models/Qwen3.5-4B --fast`
- Artefacts: `bench-output/2026-04-29-arle-qwen35-fp8kv-zero-token-trace-run2/`
- Server log: `bench-output/2026-04-29-arle-qwen35-fp8kv-zero-token-trace/server.log`

## Root Cause

The server was truly finishing some requests with zero generated tokens. This
was not an HTTP streaming or GuideLLM accounting bug.

Request-id tracing showed the first c=16 wave admitted many Qwen3.5 long
prefills into one scheduler step. Qwen3.5's hybrid paged-prefill path has large
per-row GDR scratch plus HD256 FlashInfer plan workspace. The fp8 KV pool was
sized with `est_workspace=0.0 GB`, so KV capacity consumed memory that the
lazy prefill buffers still needed. The failing requests then hit:

- `FlashInfer float_workspace alloc failed: CUDA_ERROR_OUT_OF_MEMORY`
- `Alloc a_tril failed: CUDA_ERROR_OUT_OF_MEMORY`

`finish_prefill_batch_error` called `finish_slot` for each row, and the
scheduler cleanup logged `Request N done: 0 tokens`. GuideLLM then received
terminal completions without visible output, causing TTFT/ITL p50 to collapse
to zero.

One secondary bug was exposed while tracing: `EmitCommand::Finish` only carried
the completion-token count. If the final sampled token had not yet been
appended to the emit worker, the final delta could miss the last token id.

## Fix

- Qwen3.5 now reports its scheduler runtime workspace: decode buffers, paged
  prefill buffers, GDR scratch, HD256 FlashInfer workspace, metadata, and a
  safety margin.
- KV-pool construction now reserves `max(static headroom, estimated
  workspace + 256 MiB)` before sizing the pool. With the fix, the Qwen3.5 fp8
  run logs `est_workspace=5.3 GB` and reduces the fp8 pool from 502,528 to
  398,624 tokens while keeping `mem_fraction_static=0.85`.
- Qwen3.5 caps prefill batching to one request per scheduler step via a
  model-side `max_concurrent_prefill_requests()` override. Decode still batches
  up to the configured c=16 slots.
- `EmitCommand::Finish` now carries the scheduler-side full generated token id
  list and uses it for the final stream delta if the emit worker has not yet
  appended the final token.

Verification run:

- `scripts/bench_guidellm.sh arle-qwen35-fp8kv-zero-token-prefillcap-trace --target http://localhost:8000 --model Qwen/Qwen3.5-4B --processor infer/models/Qwen3.5-4B --fast`
- Artefacts: `bench-output/2026-04-29-arle-qwen35-fp8kv-zero-token-prefillcap-trace-run2/`
- Result: valid GuideLLM output; TTFT p50 13,805.3 ms, ITL p50 58.61 ms,
  output tok/s 189.99 for the 30s diagnostic window.

## Rule

For hybrid models, KV capacity is not the only admission constraint. Model-owned
prefill scratch must be part of scheduler workspace sizing, and models with
large per-row prefill scratch need an explicit prefill-row cap instead of
relying only on token budget.
