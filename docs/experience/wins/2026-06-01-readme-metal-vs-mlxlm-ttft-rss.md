# README Metal vs mlx-lm TTFT/RSS Figure Refresh

## Context

The README benchmark image still showed the older multi-turn end-to-end view.
After the Metal memory fix, the user-facing chart needed the current
single-request sweep: ARLE vs mlx-lm only, with TTFT as the primary latency
axis, steady TPOT for decode parity, and process RSS as prompt length grows.

## Params / Env

- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Host: Apple Silicon, 48 GB unified memory.
- Shape: c=1, `/v1/completions` streaming, `max_tokens=256`, temperature 0.
- Prompt: stable completion prompt for both runtimes; the 8k/12k points use a
  paragraph request so mlx-lm does not early-stop.
- ARLE: `target/release/metal_serve --model-path
  mlx-community/Qwen3.6-35B-A3B-4bit --port 8915 --max-running-requests 1
  --max-batch-tokens 4096`.
- mlx-lm: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m mlx_lm server
  --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8916 --host 127.0.0.1`.
- RSS: process RSS sampled during streaming. On macOS unified memory,
  non-wired Metal pages may not all appear as process RSS; the raw JSON also
  records system-used memory.

Raw data:
`docs/experience/wins/assets/2026-06-01-readme-metal-vs-mlxlm-ttft-rss.json`

Figure:
`docs/assets/metal-vs-mlxlm-e2e.png`

Low-RSS analysis:
`docs/experience/wins/2026-06-01-metal-low-rss-analysis.md`

## Results

| input | ARLE TTFT | mlx TTFT | ARLE TPOT | mlx TPOT | ARLE RSS | mlx RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.22 s | 0.30 s | 14.3 ms | 16.3 ms | 4.54 GiB | 19.03 GiB |
| 256 | 0.34 s | 0.77 s | 15.4 ms | 16.2 ms | 4.54 GiB | 19.03 GiB |
| 512 | 0.63 s | 1.03 s | 14.4 ms | 16.2 ms | 4.55 GiB | 19.03 GiB |
| 1k | 1.16 s | 1.41 s | 14.5 ms | 15.8 ms | 4.55 GiB | 19.03 GiB |
| 2k | 2.27 s | 2.24 s | 14.3 ms | 15.5 ms | 4.55 GiB | 19.03 GiB |
| 4k | 4.90 s | 4.37 s | 15.0 ms | 15.1 ms | 4.56 GiB | 19.03 GiB |
| 8k | 9.75 s | 9.01 s | 15.6 ms | 15.1 ms | 5.36 GiB | 19.04 GiB |
| 12k | 15.04 s | 14.04 s | 15.8 ms | 15.2 ms | 6.04 GiB | 19.04 GiB |

## Problems

- The first prompt form made mlx-lm early-stop at 8k. The final figure uses one
  full retest with a stable prompt family, so every ARLE and mlx-lm point is
  present and connected.
- RSS is process-attributed. It is the right metric for the visible process-RSS
  regression, but not a full unified-memory pressure model; the raw JSON keeps
  `system_used_gb` for that follow-up.

## Learnings

- TTFT has the same shape in both runtimes: ARLE is faster on short prompts,
  mlx-lm is slightly faster at 4k+ in this retest.
- Decode is effectively parity: both stay around 14-16 ms/token.
- The visible difference is process RSS: ARLE is about 4.5-6.0 GiB across the
  sweep, while mlx-lm stays around 19.0 GiB.
- The README figure should stay focused on the user-facing comparison: ARLE vs
  mlx-lm. Internal residency tradeoffs belong in the dedicated memory-fix entry.

## Rule

README benchmark figures must include the memory axis when a performance claim
depends on Apple unified memory behavior. If a point is missing or only valid for
one metric, leave the gap visible and explain it in the raw evidence.
