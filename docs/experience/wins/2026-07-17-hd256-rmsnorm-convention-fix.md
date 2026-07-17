# hd256 q/k RMSNorm convention fix — cuda, 2026-07-17

> Status: pending-remote

## Goal

Correct output quality for Qwen3.5/3.6 hd256 (27B/35B-A3B) models — the q/k
RMSNorm was applied with the wrong convention, producing garbage output.

## Hypothesis

q_norm/k_norm weights are STANDARD convention (mean 0.2–0.75, multiplicative
weights below 1), not OFFSET (centered at 0). The hd256 kernels applied
`(1 + weight)` instead of `weight`, scaling q/k vectors by 1.2–1.6x and
amplifying attention scores by 4–64x — sharp enough to destroy output quality.
The 4B model (hd128, STANDARD `weight`) verified correct, confirming the
diagnosis. The MTP `pre_fc_norm_embedding`/`pre_fc_norm_hidden` are OFFSET
convention (centered at ~0) but were loaded via `load_final_norm_offset`
(subtracts 1), producing negative multipliers.

## Parameters

```bash
# pending-remote: CUDA bench not available on Mac
python3 scripts/bench_throughput.py \
  --url <url> \
  --model Qwen3.6-27B-W4A16 \
  --prompts-jsonl <workload.jsonl> \
  --concurrency-grid 1,4,8,16 \
  --seconds-per-concurrency 120 \
  --max-tokens <n> \
  --seed 20260416 \
  --output bench-output/hd256-rmsnorm-fix/bench
```

- Baseline: `pre-fix commit, hd256 kernels with (1+weight) offset`
- Treatment: `b4b293f0c, hd256 kernels with weight (STANDARD)`
- Prompt tokens: `pending-remote`
- Completion tokens: `pending-remote`
- Trials: `pending-remote`

## Environment

- Host / GPU: `pending-remote (8×H20)`
- Driver / CUDA: `pending-remote`
- Model / dtype: `Qwen3.6-27B-W4A16, w4a16`
- TP / EP / slots / KV: `pending-remote`
- Server flags: `pending-remote`

## Results

| concurrency | arm | completed | errors | output tok/s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | | | | | | | — |
| 1 | treatment | | | | | | | |

Raw artifacts: `pending-remote`.

## Problems

`pending-remote` — CUDA benchmarks cannot run on Mac (no nvcc/GPU). Remote
H20 bench required to quantify the output quality recovery and any perf delta.

## Learnings

pending-remote. The fix is a correctness correction: hd256 q/k RMSNorm must
apply `weight` (STANDARD convention), matching hd128. Evidence: q_norm/k_norm
weights have mean 0.2–0.75 (all positive, multiplicative), not centered at 0
(OFFSET). 4B hd128 model (which already uses `weight`) produces correct output,
confirming the convention. Next wall: remote H20 bench to verify output quality
recovery (needle retrieval + self-consistency) and measure any perf delta from
the reduced q/k magnitude.
