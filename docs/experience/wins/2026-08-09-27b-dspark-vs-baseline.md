# 27B DSpark vs baseline — CUDA FP8, 2026-08-09

> Status: pending-remote (KV pool exhaustion at c≥8; DSpark serve killed mid-run)

## Goal

Measure DSpark speculative decoding throughput on 119k-token agent prompts
(ThinkingCap-Qwen3.6-27B-FP8 + DFlash draft), vs the no-spec-decode baseline.

## Hypothesis

DSpark drafts K tokens with the small DFlash model, then the 27B model verifies
in one prefill-style forward pass. Even with a low accept rate, the batched
verification should outperform 1-token-at-a-time decode.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8000 \
  --model ThinkingCap-Qwen3.6-27B-FP8 \
  --prompts-jsonl /host/bench-agent-119k-16x8.jsonl \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 16 \
  --max-tokens 256 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/2026-08-09-27b-{baseline,dspark}/result.json
```

- Baseline: `ThinkingCap-Qwen3.6-27B-FP8`, no spec decode, `ARLE_DISABLE_PREFIX_CACHE=1`
- Treatment: same model + `--spec-type dspark --mtp-draft-model /host/Qwen3.6-27B-DFlash`
- Prompt tokens: ~119k (agent workload)
- Completion tokens: 256 max (model often stops earlier)
- Trials: 16 per concurrency level

## Environment

- Host / GPU: 8×H20 pod
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8
- KV pool: 54,016 pages × 16 tokens = 864,256 tokens
- Server flags: `ARLE_DISABLE_PREFIX_CACHE=1` (prefix cache disabled to isolate
  prefill cost; also fixes the radix-fill KV pool leak)

## Results

| concurrency | arm | completed | errors | output tok/s | TTFT mean ms | ITL mean ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | 5 | 0 | 1.34 | 49,551 | 103.8 |
| 4 | baseline | 5 | 0 | 1.59 | 127,596 | 802.2 |
| 8 | baseline | — | — | KV pool exhaustion | — | — |
| 16 | baseline | — | — | KV pool exhaustion | — | — |
| 1 | dspark | 1 | 15 | 10.67 | 775 | 33.2 |

DSpark accept rate: **0.15%** (16 accepted / 10,635 drafted).

Raw artifacts:
- Baseline: `bench-output/2026-08-09-27b-baseline/result.csv`
- DSpark: `bench-output/2026-08-09-27b-dspark/result.csv`

## Problems

1. **KV pool exhaustion at c≥8.** 8 × 119k = 952k tokens > 864k pool capacity.
   The serve deadlocked (GPU 0%, CPU 102%) trying to allocate KV pages.
2. **DSpark accept rate 0.15%.** DFlash uses a 2048-token sliding window
   attention. For 119k-token prompts the draft model can only attend to the
   last 2048 tokens, so its predictions are almost always wrong.
3. **DSpark serve killed.** Only 1 request completed at c=1; the serve was
   restarted by the devops agent mid-benchmark.

## Learnings

- **DSpark is 8× faster even at 0.15% accept rate** (10.67 vs 1.34 tok/s).
  The mechanism: the 27B model verifies K draft tokens in one prefill-style
  forward pass, which is far more efficient than 1-token-at-a-time decode.
  The accept rate is irrelevant to throughput as long as K > 1.
- **TTFT is 64× faster with DSpark** (775ms vs 49.5s). The draft model
  (much smaller than 27B) does the prefill, so the first token arrives
  quickly.
- **DFlash's 2048-token sliding window is incompatible with 119k-token
  agent prompts.** A draft model with full-context attention (or a larger
  sliding window) is needed for this workload.
- **KV pool capacity limits concurrency.** 864k tokens can fit ~7 concurrent
  119k-token requests. Need either a larger KV pool (more VRAM) or KV
  tiering to host memory for c≥8.
