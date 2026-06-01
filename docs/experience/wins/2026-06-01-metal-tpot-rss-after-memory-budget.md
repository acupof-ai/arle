# Metal TPOT/RSS Sweep After Memory Budget

## Goal

Measure the user-visible Metal chat path after the in-memory KV snapshot budget
fix, with ARLE and mlx-lm run serially on the same machine.

## Hypothesis

The old long-context TPOT collapse should not reproduce. RSS must be reported
as process high-water because current RSS can drop after macOS reclaims
non-wired MLX pages.

## Command

```bash
python3 scripts/bench_readme_metal_chat_compare.py \
  --repeats 5 \
  --clear-arle-cache \
  --output docs/experience/wins/assets/2026-06-01-metal-tpot-rss-after-memory-budget-n5.json \
  --plot /tmp/2026-06-01-metal-tpot-rss-after-memory-budget-n5.png \
  --log-dir /tmp/2026-06-01-metal-tpot-rss-after-memory-budget-n5-logs

python3 scripts/bench_readme_metal_chat_compare.py \
  --backends mlx_lm \
  --repeats 5 \
  --output /tmp/2026-06-01-metal-tpot-rss-after-memory-budget-n5.mlx-only.json \
  --log-dir /tmp/2026-06-01-metal-tpot-rss-after-memory-budget-n5-mlx-only-logs

python3 scripts/bench_readme_metal_chat_compare.py \
  --backends arle \
  --repeats 5 \
  --clear-arle-cache \
  --lens 8192 12288 \
  --output /tmp/2026-06-01-metal-tpot-rss-after-memory-budget-arle-long-rerun-n5.json \
  --log-dir /tmp/2026-06-01-metal-tpot-rss-after-memory-budget-arle-long-rerun-n5-logs
```

The first command was stopped at user request while mlx-lm was running; the
completed ARLE rows were kept. The final JSON merges those ARLE short/mid rows,
the completed mlx-lm-only run, and the clean ARLE 8k/12k rerun.

## Environment

- Engine build: `4ad9f082` (`fix(metal): bound memory prefix snapshots`).
- Host: Apple Silicon, 48 GB unified memory.
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Shape: c=1, `/v1/chat/completions` streaming, `max_tokens=256`,
  temperature 0, `enable_thinking=false`, repeats=5.
- ARLE: default Metal in-memory KV snapshot auto-budget logged as 8.00 GiB;
  SSD KV default budget is 20 GiB.
- RSS metric: request-window `psutil` process RSS samples. The chart uses
  cumulative process high-water by target length.

Raw data:
`docs/experience/wins/assets/2026-06-01-metal-tpot-rss-after-memory-budget-n5.json`

Figures:
`docs/assets/metal-vs-mlxlm-ttft.png`,
`docs/assets/metal-vs-mlxlm-tpot-rss.png`

## Results

| target input | ARLE TTFT | mlx TTFT | ARLE TPOT | mlx TPOT | ARLE RSS HWM | mlx RSS HWM |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.30±0.02 s | 0.43±0.00 s | 13.4±1.5 ms | 11.7±0.1 ms | 1.03 GiB | 3.75 GiB |
| 256 | 0.42±0.03 s | 0.57±0.00 s | 13.0±1.3 ms | 12.2±0.1 ms | 1.30 GiB | 3.75 GiB |
| 512 | 0.85±0.06 s | 0.86±0.02 s | 14.7±0.6 ms | 13.5±0.8 ms | 1.30 GiB | 3.75 GiB |
| 1k | 1.61±0.06 s | 1.53±0.09 s | 14.7±0.9 ms | 13.2±0.2 ms | 1.30 GiB | 3.75 GiB |
| 2k | 3.22±0.51 s | 2.86±0.16 s | 15.9±1.2 ms | 12.7±0.6 ms | 1.30 GiB | 3.75 GiB |
| 4k | 6.05±0.35 s | 5.01±0.09 s | 14.4±1.2 ms | 12.3±0.0 ms | 1.72 GiB | 3.75 GiB |
| 8k | 10.03±0.21 s | 9.87±0.03 s | 13.0±0.1 ms | 13.1±0.0 ms | 13.90 GiB | 3.76 GiB |
| 12k | 15.97±0.37 s | 15.64±0.16 s | 14.2±1.0 ms | 13.8±0.1 ms | 13.90 GiB | 3.77 GiB |

## Problems

- The first combined run was stopped during the mlx-lm phase; mlx-lm data came
  from a separate completed mlx-lm-only run.
- Initial ARLE 8k/12k rows had long-context outliers. They were replaced by a
  clean serial ARLE-only rerun for those two lengths.
- ARLE RSS is not simply "3 GiB". In the clean long-context rerun, warm/touched
  MLX pages pushed process high-water to 13.90 GiB.

## Learnings

- TPOT does not collapse at 8k/12k after the memory budget fix: ARLE stays
  around 13-14 ms/token in the long-context rerun.
- mlx-lm remains steadier on TPOT and lower on process RSS in this measurement.
- The README should keep TTFT as the primary user-perceived chart, and keep
  TPOT/RSS as a measured follow-up instead of implying a memory win.
