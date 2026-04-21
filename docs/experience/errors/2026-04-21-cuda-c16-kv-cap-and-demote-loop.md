# CUDA c16 long-context underfill was effective-KV-cap plus broken reclaim

## Context

On 2026-04-21, the CUDA `Qwen/Qwen3-4B` serving path was traced under the
canonical long-context concurrency shape:

- `--num-slots 16`
- `--max-seq-len 4608`
- `--enable-mixed-chunk=true`
- `--chunked-prefill-size 4096`
- `--max-prefill-tokens 16384`
- GuideLLM data: `prompt_tokens=4096,output_tokens=256`

Relevant artefacts from this debugging pass:

- plan: [docs/plans/2026-04-21-cuda-c16-underfill-alignment.md](/content/workspace/agent-infer/docs/plans/2026-04-21-cuda-c16-underfill-alignment.md)
- earlier smoke after the first admission fix:
  `bench-output/2026-04-21-cuda-l4-c16-underfill-budget-smoke/`
- mixed-workspace crashfix smoke:
  `bench-output/2026-04-21-cuda-l4-c16-underfill-crashfix-smoke/`
- bounded-demote smoke:
  `bench-output/2026-04-21-cuda-l4-c16-underfill-bounded-demote-smoke/`

The latest clean smoke after bounded demote/drop produced:

- TTFT p50 `3490.0 ms`
- TTFT p99 `28444.6 ms`
- ITL p50 `45.99 ms`
- out tok/s `66.04`
- trace peak active `5`
- trace peak running_batch `5`
- trace peak kv_util `89.1%`

## Root Cause

The c16 underfill was not one scheduler bug. It was a stack of four issues,
with one hard physical limit underneath them:

1. Effective KV capacity was too small to hold 16 of these requests at once.
   After mixed runtime workspace reservation, the live pool on this L4 build
   was only `24352` tokens (`1522` pages). One request in this workload costs
   about `4097 + 256 = 4353` tokens, so the true steady-state ceiling is about
   `24352 / 4353 ≈ 5.6` requests, not 16.
2. Admission semantics were wrong. `max_prefill_tokens` had been collapsed to
   `min(max_prefill_tokens, chunked_prefill_size)`, which reduced a
   `16384`-token batch budget to `4096` and serialized long-prompt admission.
3. Mixed FlashInfer workspace sizing was too small. HD256 mixed prefill still
   used a `512 MiB` float workspace, and c16 / 4096-token mixed batches could
   overflow it in `batch_prefill_tmp_v`, crashing the server.
4. GPU-page reclaim under host-tier exhaustion was broken. `demote_block_to_host`
   removed `block_to_pages` before host reserve succeeded, so a failed demote
   left the prefix cache claiming a GPU block whose page span had been forgotten.
   The later fallback drop then "evicted" many blocks while reclaiming almost
   no GPU pages.
5. Even after that mapping bug, reclaim stayed too expensive because the
   scheduler kept trying hopeless host demotions after the host pool was full,
   spamming failure logs and stretching cleanup into hundreds of milliseconds.

## Fix

The debugging pass landed these corrections:

- restored SGLang-style prefill budgeting:
  - `chunked_prefill_size` = per-request chunk cap
  - `max_prefill_tokens` = batch-wide prefill budget
- aligned page budgeting:
  - truncated chunked prefill does not reserve decode headroom yet
  - prefill admission reserves one extra page per request
  - final prefill completion reserves explicit remaining decode budget
- increased HD256 mixed FlashInfer float workspace from `512 MiB` to `640 MiB`
- fixed `demote_block_to_host` so `block_to_pages` is only removed after a
  successful host write
- changed reclaim behavior under host-tier pressure to bounded demote plus
  direct GPU drop, instead of repeated impossible demote attempts

## Rule

When c16 long-context throughput looks far below SGLang, check the effective
KV pool first, not just scheduler slot count. If

- `max_live_tokens / per_request_total_tokens` is already near the observed
  `peak active`

then the problem is not "why not 16 active requests" but "why is runtime
workspace leaving so little T0 KV capacity". Also, never trust GPU cache
fallback logic unless failed demotion preserves the page-span mapping needed
for the subsequent hard-drop path.
