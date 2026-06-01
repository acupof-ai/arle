# README Metal vs mlx-lm TTFT/RSS Figure Refresh

## Goal

Refresh the README Metal figure with one serial ARLE-vs-mlx-lm measurement path
after the bounded SSD KV fix.

## Hypothesis

ARLE should no longer show the old 12k TPOT collapse. RSS must use the same
process-level metric on both sides and must not mix old mlx-lm data with new
ARLE data.

## Command

```bash
python3 scripts/bench_readme_metal_chat_compare.py \
  --repeats 5 \
  --clear-arle-cache \
  --output /tmp/bench_readme_metal_chat_compare_n5_20260601.json \
  --plot /tmp/bench_readme_metal_chat_compare_n5_20260601.png \
  --log-dir /tmp/bench_readme_metal_chat_compare_n5_logs
```

## Environment

- Engine build: `ed0cb88b` (`fix(metal): bound default ssd kv cache`).
- Host: Apple Silicon, 48 GB unified memory.
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Shape: c=1, `/v1/chat/completions` streaming, `max_tokens=256`,
  temperature 0, `enable_thinking=false`.
- ARLE: `target/release/metal_serve --model-path
  mlx-community/Qwen3.6-35B-A3B-4bit --port 9071 --max-running-requests 1
  --max-batch-tokens 4096 --warmup 0`.
- mlx-lm: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m mlx_lm server
  --model mlx-community/Qwen3.6-35B-A3B-4bit --port 9072 --host 127.0.0.1`.
- RSS panel: cumulative process RSS high-water from request-window `psutil`
  samples. The raw JSON also keeps per-request RSS peaks.

Raw data:
`docs/experience/wins/assets/2026-06-01-readme-metal-vs-mlxlm-chat-essay-avg.json`

Figure:
`docs/assets/metal-vs-mlxlm-essay-avg.png`

## Results

ARLE and mlx-lm were not resident at the same time. ARLE logs show
`matched_tokens_max=0`, completion length was 256 tokens for every measured
request, and no `No space left on device` / cache publish failure appeared.
The bounded SSD KV cache ended at 5.6 GiB, below the 20 GiB default budget.

| target input | ARLE TTFT | mlx TTFT | ARLE TPOT | mlx TPOT | ARLE RSS high-water | mlx RSS high-water |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.27±0.00 s | 0.50±0.07 s | 11.8±0.0 ms | 13.0±0.7 ms | 14.54 GiB | 14.79 GiB |
| 256 | 0.40±0.00 s | 0.58±0.01 s | 11.8±0.0 ms | 12.4±0.4 ms | 15.33 GiB | 14.80 GiB |
| 512 | 0.69±0.01 s | 0.83±0.01 s | 11.9±0.0 ms | 12.0±0.1 ms | 16.12 GiB | 14.80 GiB |
| 1k | 1.24±0.01 s | 1.43±0.08 s | 12.0±0.1 ms | 13.0±1.1 ms | 17.03 GiB | 14.80 GiB |
| 2k | 2.36±0.00 s | 2.97±0.25 s | 12.3±0.3 ms | 14.1±1.2 ms | 17.41 GiB | 14.80 GiB |
| 4k | 5.03±0.04 s | 5.02±0.18 s | 12.8±0.5 ms | 13.0±0.5 ms | 17.41 GiB | 14.81 GiB |
| 8k | 10.52±0.34 s | 10.12±0.26 s | 14.6±2.3 ms | 13.4±0.2 ms | 17.41 GiB | 14.81 GiB |
| 12k | 16.78±0.28 s | 15.79±0.18 s | 14.3±0.6 ms | 14.3±0.2 ms | 17.41 GiB | 14.81 GiB |

## Problems

- The first n=3 sweep had an ARLE 2k TTFT outlier, so it was rejected and
  replaced by this n=5 sweep.
- ARLE 8k TPOT includes one first-long-context outlier at 18.6 ms/token; the
  next four samples were 13.0-14.2 ms/token, so the chart keeps the mean±std
  instead of hiding the variance.
- Per-request current RSS can fall after macOS reclaims non-wired MLX pages.
  The README chart therefore uses process RSS high-water for the memory panel.

## Learnings

- The old 12k TPOT collapse is gone: ARLE 12k is 14.3±0.6 ms/token, matching
  mlx-lm's 14.3±0.2 ms/token under this shape.
- ARLE TTFT is faster through 2k, parity around 4k, and slower at 8k/12k in
  this run.
- Same-script, same-endpoint RSS is the only acceptable README memory
  comparison. Mixing prior mlx-lm data with new ARLE samples created the earlier
  misleading chart.

## Delta vs baseline

Supersedes the previous README figure from `660c5149`. The new figure uses one
n=5 serial sweep for both backends and a cumulative process RSS high-water
memory axis.
