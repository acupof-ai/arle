# README Metal vs mlx-lm TTFT/RSS Figure Refresh

## Context

The README benchmark image still showed the older multi-turn end-to-end view.
After the Metal memory fix, the user-facing chart needed the current
single-request sweep: ARLE vs mlx-lm only, with TTFT as the primary latency
axis, steady TPOT for decode parity, and process RSS as prompt length grows.

## Params / Env

- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Host: Apple Silicon, 48 GB unified memory.
- Shape: c=1, `/v1/chat/completions` streaming, `max_tokens=256`,
  temperature 0, 3 repeats per point.
- Prompt: varied source document plus essay instruction. Prompts are
  cache-neutral per length/repeat and sized by chat-template token count.
  mlx-lm uses `chat_template_kwargs.enable_thinking=false`; ARLE accepts
  `chat_template_kwargs.thinking=false`.
- ARLE: `target/release/metal_serve --model-path
  mlx-community/Qwen3.6-35B-A3B-4bit --port 9035 --max-running-requests 1
  --max-batch-tokens 4096`.
- mlx-lm: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m mlx_lm server
  --model mlx-community/Qwen3.6-35B-A3B-4bit --port 9046 --host 127.0.0.1`.
- RSS: process RSS sampled during streaming. On macOS unified memory,
  non-wired Metal pages may not all appear as process RSS; the raw JSON also
  records system-used memory.

Raw data:
`docs/experience/wins/assets/2026-06-01-readme-metal-vs-mlxlm-chat-essay-avg.json`

Figure:
`docs/assets/metal-vs-mlxlm-essay-avg.png`

Low-RSS analysis:
`docs/experience/wins/2026-06-01-metal-low-rss-analysis.md`

## Results

| input | ARLE TTFT | mlx TTFT | ARLE TPOT | mlx TPOT | ARLE RSS | mlx RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.27±0.00 s | 0.52±0.06 s | 12.3±0.1 ms | 12.2±0.1 ms | 3.15±2.34 GiB | 7.23±0.01 GiB |
| 256 | 0.37±0.01 s | 0.55±0.00 s | 12.4±0.2 ms | 12.2±0.1 ms | 3.22±2.27 GiB | 7.23±0.01 GiB |
| 512 | 0.68±0.10 s | 0.83±0.04 s | 12.5±0.1 ms | 12.3±0.1 ms | 3.37±2.29 GiB | 7.23±0.01 GiB |
| 1k | 1.27±0.17 s | 1.32±0.01 s | 12.9±0.2 ms | 12.4±0.1 ms | 3.51±2.34 GiB | 7.23±0.01 GiB |
| 2k | 2.54±0.43 s | 2.43±0.06 s | 13.5±0.1 ms | 12.6±0.0 ms | 3.70±2.41 GiB | 7.23±0.01 GiB |
| 4k | 5.14±0.54 s | 4.77±0.07 s | 14.2±0.1 ms | 12.9±0.0 ms | 3.97±2.49 GiB | 7.23±0.01 GiB |
| 8k | 11.53±0.87 s | 9.79±0.24 s | 17.1±1.5 ms | 13.7±0.0 ms | 4.11±2.38 GiB | 7.23±0.01 GiB |
| 12k | 19.50±1.93 s | 15.34±0.36 s | 32.3±9.4 ms | 14.3±0.1 ms | 2.46±0.42 GiB | 7.24±0.00 GiB |

## Problems

- Repeated-word completions polluted TPOT/EOS behavior. The README figure now
  uses essay generation and mean±std over 3 runs.
- mlx-lm chat output needed thinking disabled; otherwise the stream reports
  reasoning tokens separately from content.
- RSS is process-attributed. It is the right metric for the visible process-RSS
  regression, but not a full unified-memory pressure model; the raw JSON keeps
  `system_used_gb` for that follow-up.

## Learnings

- TTFT has the same shape in both runtimes: ARLE is faster on short prompts,
  mlx-lm is faster at 2k+ in this retest.
- TPOT is stable for mlx-lm. ARLE is stable through 4k, then shows long-context
  tail variance at 8k/12k.
- RSS remains lower on ARLE by process accounting, but the ARLE error bars are
  large because macOS can reclaim non-wired pages between requests.
- The README figure should stay focused on the user-facing comparison: ARLE vs
  mlx-lm. Internal residency tradeoffs belong in the dedicated memory-fix entry.

## Rule

README benchmark figures must include the memory axis when a performance claim
depends on Apple unified memory behavior. If a point is missing or only valid for
one metric, leave the gap visible and explain it in the raw evidence.
