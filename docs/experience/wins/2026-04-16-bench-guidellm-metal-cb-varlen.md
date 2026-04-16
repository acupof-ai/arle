# guidellm sweep metal-cb-varlen — first Qwen3.5 Metal guidellm baseline, 2026-04-16

First canonical `scripts/bench_guidellm.sh` run on the local M4 Pro
with the new varlen CB + per-row RoPE fix. Establishes the Metal
Qwen3.5-4B-MLX-4bit guidellm baseline for future comparison.

## Context

- **Backend:** metal
- **Model:** mlx-community/Qwen3.5-4B-MLX-4bit
- **Hardware:** Apple M4 Pro (Metal, unified memory)
- **Commit:** a6d4525 (fix(metal): enable varlen packed decode via array-offset RoPE)
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** `--kv-pool` (server started with prefix cache)
- **Server launch:** `./target/release/metal_serve --model-path mlx-community/Qwen3.5-4B-MLX-4bit --port 8014 --kv-pool`

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://127.0.0.1:8014 \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --profile sweep \
  --data  prompt_tokens=1024,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir bench-output/2026-04-16-metal-cb-varlen/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh metal-cb-varlen --target http://127.0.0.1:8014 --model mlx-community/Qwen3.5-4B-MLX-4bit`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 1324.1 | 1515.1 | 27.89 | 29.06 | 30.98 | 0.117 |
| throughput | 32245.9 | 56275.6 | 0 | 19.78 | 57.24 | 0.2 |
| 0.127 r/s | 1388.4 | 1892.3 | 0 | 33.08 | 37.02 | 0.1 |
| 0.138 r/s | 1395.8 | 1585.1 | 0 | 29.15 | 40.05 | 0.133 |
| 0.148 r/s | 1379.4 | 1647.5 | 0 | 27.41 | 42.98 | 0.133 |
| 0.158 r/s | 1405.9 | 1821.7 | 0 | 26.26 | 44.89 | 0.133 |
| 0.169 r/s | 1499.8 | 1679.6 | 0 | 24.27 | 48.01 | 0.167 |
| 0.179 r/s | 1345.3 | 1550.4 | 0 | 22.71 | 50.84 | 0.167 |
| 0.190 r/s | 1495.1 | 1657.0 | 0 | 21.72 | 52.82 | 0.167 |
| 0.200 r/s | 1489.3 | 1777.0 | 0 | 21.14 | 55.67 | 0.200 |

## Artefacts

- Raw: `bench-output/2026-04-16-metal-cb-varlen/benchmarks.json`
- CSV:  `bench-output/2026-04-16-metal-cb-varlen/benchmarks.csv`
- HTML: `bench-output/2026-04-16-metal-cb-varlen/benchmarks.html`

## Delta vs previous snapshot

First canonical guidellm run on this hardware with this model. No prior
guidellm snapshot to diff against.

Closest reference: `M0.2` acceptance plan reports `512/256 C=4` via
`bench_throughput_sweep.py` (different tool, different prompt shape):
- Old serial reference: 65.8 tok/s, TTFT p50 7994 ms
- M0.2b live runtime: 58.7 tok/s, TTFT p50 1826 ms
- M0.2d same-length: 66.2 tok/s, TTFT p50 1757 ms

Not directly comparable (1024 vs 512 prompt, guidellm sweep vs quick
throughput sweep), but directionally:
- Sync single-stream 30.98 tok/s at 1024-prompt is consistent with
  the single-stream 52 tok/s at 128-prompt reported in the Metal
  optimization wins — longer prompts = slower prefill overhead.
- Throughput saturation 57.24 tok/s at 1024-prompt tracks the M0.2d
  66 tok/s at 512-prompt shape, scaled by the longer prompt / lower
  prefill throughput.

## Notes

- This is the first run after the per-row array-offset RoPE fix
  (`a6d4525`), which also fixed a silent production correctness bug
  where MLX 0.31.1's scalar `fast::rope` zeroed batch rows > 0. All
  prior `M0.2c/d` same-length batching numbers were measured under
  that bug — the RoPE fix may change the throughput profile because
  row > 0 outputs were positionally wrong before (and now compute more
  meaningful attention).
- The varlen admission path (`try_build_qwen35_packed_decode_batch`
  same-length gate lifted) is active but guidellm's sweep profile
  generates requests at constant rates, so concurrent request overlap
  at `max_active_requests=4` was low (median concurrency 1.0-1.5
  during constant-rate steps). A dedicated concurrent HTTP sweep
  with higher `C` values (e.g., `bench_throughput_sweep.py --quick
  --label metal-cb-varlen-c4`) would better exercise the varlen path.
- Follow-ups: run a `bench_throughput_sweep.py --quick` at C=2,4,8
  on 512/256 for a direct M0.2 baseline comparison, and a longer
  guidellm sweep at `--max-seconds 120` for steadier saturation
  numbers.
