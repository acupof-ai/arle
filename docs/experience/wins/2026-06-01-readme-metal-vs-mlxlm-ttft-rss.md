# README Metal vs mlx-lm TTFT/RSS Figure Refresh

## Context

The README benchmark image still showed the older multi-turn end-to-end view.
After the Metal memory fix, the user-facing chart needed the current
single-request sweep: ARLE vs mlx-lm only, with TTFT as the primary latency
axis, steady TPOT for decode parity, and process RSS as prompt length grows.

## Params / Env

- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Host: Apple Silicon, 48 GB unified memory.
- Shape: c=1, OpenAI streaming completions, `max_tokens=256`, temperature 0.
- ARLE: `target/release/metal_serve --max-running-requests 1
  --max-batch-tokens 4096`, default no-wired mode.
- mlx-lm: `mlx_lm server`, same model and host.
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
| 128 | 0.21 s | 0.16 s | 11.6 ms | 12.9 ms | 2.21 GiB | 10.77 GiB |
| 256 | 0.33 s | 0.48 s | 11.5 ms | 11.9 ms | 2.21 GiB | 10.77 GiB |
| 512 | 0.58 s | 0.72 s | 11.7 ms | 12.0 ms | 2.21 GiB | 10.78 GiB |
| 1k | 1.10 s | 1.23 s | 11.8 ms | 12.2 ms | 2.22 GiB | 10.78 GiB |
| 2k | 2.12 s | 2.32 s | 11.9 ms | 12.7 ms | 2.22 GiB | 10.78 GiB |
| 4k | 4.59 s | 4.55 s | 12.4 ms | 12.9 ms | 2.23 GiB | 10.78 GiB |
| 8k | 9.58 s | 10.86 s | 13.1 ms | - | 2.23 GiB | - |
| 12k | 16.64 s | 14.50 s | 14.3 ms | 15.3 ms | 2.23 GiB | 10.79 GiB |

## Problems

- The same-prompt mlx-lm 8k request emitted only one non-empty text chunk, so
  TPOT is not valid there. The README plot uses that retry only for TTFT and
  leaves mlx-lm 8k RSS/TPOT blank instead of fabricating a point.
- RSS is a process-attributed metric. The default no-wired ARLE curve is useful
  for the user-visible RSS regression, but it is not a complete unified-memory
  pressure model; use the raw `system_used` fields for that follow-up.

## Learnings

- The default memory issue was weight residency, not decode scaling. In the
  README-facing default configuration, nominal TTFT and TPOT stay at mlx-lm
  parity while process RSS no longer carries the previous wired footprint.
- The README figure should stay focused on the user-facing comparison: ARLE vs
  mlx-lm. Internal residency tradeoffs belong in the dedicated memory-fix entry.

## Rule

README benchmark figures must include the memory axis when a performance claim
depends on Apple unified memory behavior. If a point is missing or only valid for
one metric, leave the gap visible and explain it in the raw evidence.
