# Metal TTFT/TPOT Steady Sweep

## Goal

Refresh the README Metal chart with one ARLE-vs-mlx-lm figure that shows TTFT
and steady TPOT only.

## Hypothesis

After removing process-RSS from the chart and filtering transient jitter with a
fixed robust rule, TPOT should show the steady decode path instead of noisy
request-level outliers.

## Command

```bash
python3 scripts/bench_readme_metal_chat_compare.py \
  --repeats 5 \
  --clear-arle-cache \
  --output docs/experience/wins/assets/2026-06-02-metal-ttft-tpot-n5.json \
  --log-dir /tmp/2026-06-02-metal-ttft-tpot-n5-logs

python3 scripts/bench_readme_metal_chat_compare.py \
  --plot-input docs/experience/wins/assets/2026-06-02-metal-ttft-tpot-n5.json \
  --plot docs/assets/metal-vs-mlxlm-ttft-tpot.png \
  --plot-kind ttft-tpot
```

## Environment

- Engine build: `240fe9ee` plus this chart-script/doc update.
- Host: Apple Silicon M4 Pro, 48 GB unified memory.
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Shape: c=1, `/v1/chat/completions` streaming, `max_tokens=256`,
  temperature 0, `enable_thinking=false`, repeats=5.
- Run order: ARLE first, then mlx-lm. The two model servers were not resident
  at the same time.
- TPOT metric: steady inter-token latency; the token1->token2 interval is
  excluded because ARLE's scheduler can put prompt prefill work in that gap.
- Chart metric: stable mean. For each backend, length, and metric, the plot
  filters samples outside a median/MAD band, then averages the remaining stable
  cluster. If fewer than three samples remain, it falls back to dropping min and
  max.

Raw data:
`docs/experience/wins/assets/2026-06-02-metal-ttft-tpot-n5.json`

Figure:
`docs/assets/metal-vs-mlxlm-ttft-tpot.png`

## Results

| target input | ARLE TTFT | mlx-lm TTFT | ARLE TPOT | mlx-lm TPOT |
|---:|---:|---:|---:|---:|
| 128 | 0.27 s | 0.44 s | 11.92 ms | 12.19 ms |
| 256 | 0.41 s | 0.57 s | 11.97 ms | 12.01 ms |
| 512 | 0.68 s | 0.85 s | 11.91 ms | 12.18 ms |
| 1k | 1.26 s | 1.39 s | 12.19 ms | 12.56 ms |
| 2k | 2.58 s | 2.72 s | 12.38 ms | 13.60 ms |
| 4k | 5.14 s | 5.14 s | 12.40 ms | 13.56 ms |
| 8k | 10.40 s | 10.27 s | 13.38 ms | 14.03 ms |
| 12k | 15.93 s | 16.28 s | 14.89 ms | 14.30 ms |

## Problems

- Raw request samples still contain occasional long-context jitter. The README
  figure intentionally uses the stable-mean rule above instead of plotting
  noisy error bars.
- The previous RSS panel was removed. A follow-up vmmap check showed both ARLE
  and mlx-lm allocate about 18-20 GiB in `IOAccelerator (graphics)`; process
  RSS timing made the old README memory comparison misleading.

## Learnings

- For this README chart, TTFT and steady TPOT are the user-facing metrics.
  Metal memory needs `vmmap` / Metal-footprint accounting, not process RSS.
- ARLE steady TPOT remains in the same decode band as mlx-lm across 128-12k.
  The remaining long-context difference is mostly TTFT/prefill behavior and
  occasional scheduler/runtime jitter, not a decode collapse.
