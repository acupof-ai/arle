# KV Quant Remote Validation Checklist

Date: 2026-04-14

## What changed

This patch optimizes the prefill -> paged-KV handoff without changing decode kernels:

1. Scheduler prefill now migrates only the newly materialized KV range into the paged pool.
2. Prefix-hit requests no longer re-copy the already committed prefix into the pool.
3. Added CUDA range-migration kernels for:
   - BF16 contiguous -> BF16 paged
   - INT8 contiguous -> INT8 paged
   - BF16 contiguous -> FP8 paged
4. TurboQuant prefill migration now copies only the new contiguous range into the NHD work buffer and quantizes it into packed paged storage.

## Expected impact

### Should improve

- Prefix-hit TTFT when the request has a reused prefix plus a short suffix.
- Exact prompt replay paths that only need to recompute a tiny suffix.
- CPU overhead and H2D metadata traffic on prefix-hit migrations, because we now upload only `new_token_indices` instead of the full slot token table.
- TurboQuant prompt prefill correctness and memory behavior.

### Should stay neutral

- Prefix-miss prompts with a single prefill chunk.
- Decode TPOT after prefill completes.
- BF16 decode throughput on workloads without prefix reuse.

### Not addressed by this patch

- The contiguous -> paged copy itself still exists; this patch makes it suffix-only, not zero-copy.
- `page_size = 1` is unchanged.
- No new CUDA graph capture was added around prefill migration.

## Recommended remote benchmark matrix

Run every case with:

- `bf16`
- `fp8`
- `int8`
- `tq3`

### 1. Prefix-hit TTFT

Goal: verify suffix-only migration removes prefix recopy cost.

- Warm one request with a long shared system prompt or long agent context.
- Send a second request sharing 90%+ of that prompt and adding a short suffix.
- Compare TTFT before/after.

Suggested shapes:

- shared prefix: 4k / 8k / 16k tokens
- suffix: 16 / 64 / 256 tokens

Primary metric:

- TTFT

Secondary metrics:

- scheduler prefill step time
- GPU KV utilization

Expected result:

- larger gain as `prefix_len / suffix_len` grows

### 2. Exact prompt replay

Goal: verify we no longer pay full-window migration when only replaying the final prompt token or a tiny suffix.

- Send the exact same prompt twice on the same slot reuse pattern.
- Also test a prompt that differs only in the last 1-8 tokens.

Primary metric:

- TTFT

Expected result:

- clear reduction vs pre-patch on exact-match or near-exact-match prompts

### 3. Prefix-miss neutrality

Goal: ensure no regression on the plain path.

- prompt lengths: 512 / 2048 / 8192
- output lengths: 64 / 256
- concurrency: 1 and 4

Primary metrics:

- TTFT
- TPOT
- tok/s

Expected result:

- within noise relative to pre-patch

### 4. TurboQuant prompt prefill

Goal: verify packed paged prompt KV now works end-to-end.

- `--kv-cache-dtype tq3`
- long prompt prefill followed by decode
- prefix-hit replay of the same prompt

Primary metrics:

- correctness of generated output
- absence of obvious prompt-forgetting behavior after prefill
- TTFT / TPOT

## Useful comparisons

- old commit: `335b08b^`
- new commit: `HEAD`

If you want the cleanest A/B:

1. restart the server between runs
2. use the same model weights and GPU
3. keep `--cuda-graph` unchanged across both runs
4. compare prefix-hit workloads separately from prefix-miss workloads

## Interpretation guide

If prefix-hit TTFT improves while prefix-miss stays flat, the patch did what it was supposed to do.

If prefix-miss regresses materially, inspect:

- extra scheduler work around slot allocation
- any unexpected repeated migration
- whether the benchmark unintentionally changed slot reuse behavior

If TurboQuant behaves differently from BF16/FP8/INT8, treat that as a correctness-first issue before reading too much into the performance numbers.
