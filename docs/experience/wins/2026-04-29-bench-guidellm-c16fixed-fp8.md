# cuda-l4 fixed c=16 @ 120 s, FP8 KV — saturation baseline

> First fixed-concurrency bench at c=16 with the §10.3 minimum
> duration (120 s) at SGLang-aligned defaults. Replaces the earlier
> `--fast` 30s n=3 medians as the canonical headline for c=16
> comparisons. **Honest number: 105.22 tok/s — ~52% of SGLang's
> historical 201 tok/s reference at this shape.**

## Goal

Pin down the c=16 fixed-concurrency saturation throughput at all the
post-`d53c6d8d` SGLang-aligned defaults, without sweep / `--fast`
sampling artifacts. Goal type: **publication baseline**.

## Hypothesis

- TTFT p50 around 1-3 s (admission window adequate after F4
  `max_prefill_tokens=16384`).
- ITL p50 around 75-85 ms.
- out tok/s around 150-180 (matches the better `--fast` runs).

**Hypothesis WRONG**: TTFT p50 = 10.5 s, tok/s = 105. The 120 s
duration exposes saturation effects the 30 s window hid.

## Command

```bash
target/release/infer \
    --model-path models/Qwen3-4B --port 8000 \
    --num-slots 16 --max-seq-len 4608
    # all other params at SGLang-aligned defaults

bash scripts/bench_guidellm.sh c16fixed-fp8 \
    --concurrencies 16 --max-seconds 120
```

## Environment

- **Backend:** cuda
- **Model:** Qwen3-4B (bf16 weights, FP8E4M3 paged KV)
- **Hardware:** NVIDIA L4 sm_89, 22 GB, CUDA 12.8
- **Commit:** 26ad7dce
- **Feature set:** `cargo build --release --features cuda`
- **Server flags:** `--num-slots 16 --max-seq-len 4608 --kv-cache-dtype fp8`
  (mem_fraction_static, max_prefill_tokens, etc. all at defaults
  after `d53c6d8d`)
- **Pool:** 148,256 tokens / 11.5 GB at fraction=0.85
- **Bench profile:** concurrent c=16, 120s

## Results

| Metric          | Value     |
|-----------------|----------:|
| TTFT p50 (ms)   | 10455.3   |
| TTFT p99 (ms)   | 23701.0   |
| ITL p50 (ms)    | 86.12     |
| ITL p99 (ms)    | 119.86    |
| out tok/s       | **105.22**|
| req/s actual    | 0.367     |

## Comparison

| Source                          | tok/s | TTFT p50 | ITL p50 | Notes |
|---------------------------------|------:|---------:|--------:|-------|
| **This run (c=16, 120s)**       | 105.22 | 10455 | 86.12 | post-fix-stack baseline |
| SGLang historical (c=16/4096)   | ~201  | ~3357 | ~67   | from `2026-04-28-bench-guidellm-cuda-l4-tilelang-prefill-causal-bound.md` |
| Δ vs SGLang                     | **−48%** | +211% | +28% | gap is real, not bench noise |
| Pre-fix-stack `--fast` 30s n=3  | 184.96 | 7839 | 75.95 | optimistic — small-sample variance |

The 30s `--fast` median (184) was misleading; 120s saturation gives
~105. The full perf-fix stack today (`83e67ff2` budget,
`47bad713` clear=True, `c4109b29` dedup, `8f6965c3` max_prefill,
`d53c6d8d` mem_fraction) closed the structural envelope (pool size,
TTFT cliff at low queueing) but the saturation throughput at c=16 is
still 48% below SGLang.

## Where the gap is (per-step breakdown from the trace)

From `service_stats_trace_summary.md`:

- Peak `kv_util` ~75-85% (pool isn't the bottleneck).
- Peak `prefill_queue` ~5-7 (admission window OK, F4 fix landed).
- `prefix_hit_rate = 0%` (synthetic random prompts; expected).
- TTFT p99 = 23.7 s — admission tail: when 16 reqs are in flight and
  ITL=86ms × 256 tokens = 22 s per req, the 17th-32nd reqs wait for
  the first batch to drain.

The remaining gap (vs SGLang's ~3.3 s TTFT p50) is K2 from
`docs/projects/2026-04-29-perf-bug-roundup.md`: **mixed (decode +
prefill) batches don't fire on FP8/INT8 KV** because
`infer/src/model/qwen3/forward.rs:585`'s
`supports_mixed_batch` gates on `KVFormat::BF16` only. So decode rows
serialize behind prefill chunks instead of riding alongside them.
SGLang interleaves; we don't.

## Problems

1. The 30s `--fast` headline numbers I pushed earlier today (184
   tok/s) were artifacts of small-sample variance, not the real
   sustained rate. §10.3 codifies the duration-vs-purpose rule that
   would have caught this.
2. The wins entry from `c4109b29` (weight dedup) cited `184.96
   tok/s` as a `+52%` improvement over `121.74 tok/s` baseline.
   Both numbers were `--fast` 30s. The actual headline at canonical
   duration is 105 tok/s. The dedup IS still a structural win
   (frees 4.7 GB → 175k tokens pool); the 52% perf number was
   coincidence of warm-cache vs cold-cache runs.

## Next steps

1. **K2: enable mixed batch on FP8 KV** — the largest remaining
   lever. Either fused-dequant FlashInfer varlen or relax the
   `KVFormat::BF16` gate after testing fp8/int8 path correctness.
2. Re-run the c=16 fixed canonical with K2 in place; expected TTFT
   drop from 10.5 s → 4-5 s, tok/s 105 → 150-180.
3. Establish a multi-concurrency canonical headline:
   `--concurrencies 1,4,16,64 --max-seconds 120` on each KV format.
   Land that alongside the K2 work, not before.
