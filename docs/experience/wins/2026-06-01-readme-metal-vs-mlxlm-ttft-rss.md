# README Metal vs mlx-lm TTFT/RSS Figure Refresh

## Context

The README benchmark image still showed the older multi-turn end-to-end view.
After the Metal KV-boundary clear fix, the user-facing chart needed a fresh
single-request sweep: ARLE vs mlx-lm only, with TTFT as the primary latency
axis, steady TPOT for decode parity, and process RSS as prompt length grows.

## Params / Env

- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Host: Apple Silicon, 48 GB unified memory.
- Shape: c=1, `/v1/chat/completions` streaming, `max_tokens=256`,
  temperature 0. ARLE uses 5 repeats per point after
  `71494395`; mlx-lm uses the existing 3-repeat same-prompt sweep.
- Prompt: varied source document plus essay instruction. Prompts are
  cache-neutral per length/repeat and sized by chat-template token count.
  mlx-lm uses `chat_template_kwargs.enable_thinking=false`; ARLE accepts
  top-level `enable_thinking=false`.
- ARLE: `target/release/metal_serve --model-path
  mlx-community/Qwen3.6-35B-A3B-4bit --port 9035 --max-running-requests 1
  --max-batch-tokens 4096`.
- mlx-lm: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m mlx_lm server
  --model mlx-community/Qwen3.6-35B-A3B-4bit --port 9046 --host 127.0.0.1`.
- RSS: process RSS high-water. On macOS unified memory, non-wired Metal pages
  may not all appear as process RSS; the raw JSON also records system memory
  snapshots.

Raw data:
`docs/experience/wins/assets/2026-06-01-readme-metal-vs-mlxlm-chat-essay-avg.json`

Figure:
`docs/assets/metal-vs-mlxlm-essay-avg.png`

Low-RSS analysis:
`docs/experience/wins/2026-06-01-metal-low-rss-analysis.md`

## Results

| input | ARLE TTFT | mlx TTFT | ARLE TPOT | mlx TPOT | ARLE RSS | mlx RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.29±0.03 s | 0.52±0.06 s | 12.4±0.0 ms | 12.2±0.1 ms | 14.77 GiB | 7.23±0.01 GiB |
| 256 | 0.41±0.02 s | 0.55±0.00 s | 11.8±0.1 ms | 12.2±0.1 ms | 13.71 GiB | 7.23±0.01 GiB |
| 512 | 0.65±0.00 s | 0.83±0.04 s | 12.0±0.1 ms | 12.3±0.1 ms | 14.43 GiB | 7.23±0.01 GiB |
| 1k | 1.45±0.53 s | 1.32±0.01 s | 12.6±0.5 ms | 12.4±0.1 ms | 14.12 GiB | 7.23±0.01 GiB |
| 2k | 2.26±0.07 s | 2.43±0.06 s | 13.5±0.6 ms | 12.6±0.0 ms | 11.44 GiB | 7.23±0.01 GiB |
| 4k | 4.81±0.11 s | 4.77±0.07 s | 13.1±0.2 ms | 12.9±0.0 ms | 12.01 GiB | 7.23±0.01 GiB |
| 8k | 10.55±0.58 s | 9.79±0.24 s | 14.5±0.2 ms | 13.7±0.0 ms | 12.97 GiB | 7.23±0.01 GiB |
| 12k | 16.76±0.72 s | 15.34±0.36 s | 16.1±0.8 ms | 14.3±0.1 ms | 15.47 GiB | 7.24±0.00 GiB |

## Problems

- Repeated-word completions polluted TPOT/EOS behavior. The README figure uses
  essay generation and mean±std.
- Pre-fix ARLE had duplicate Metal cache-clears at 256-token KV boundaries.
  That caused the old 12k TPOT spike. The fix is documented in
  `2026-06-01-metal-kv-boundary-clear-tail-fix.md`.
- mlx-lm chat output needed thinking disabled; otherwise the stream reports
  reasoning tokens separately from content.
- RSS is process-attributed. It is the right metric for the visible process-RSS
  regression, but not a full unified-memory pressure model; the raw JSON keeps
  `system_used_gb` for that follow-up.

## Learnings

- TTFT has the same shape in both runtimes: ARLE is faster on short prompts,
  and mlx-lm is faster at 8k/12k in this retest.
- TPOT is now continuous after the ARLE fix: no 30ms 12k spike remains.
- Process RSS high-water is higher for ARLE in this run. The earlier low-RSS
  chart used current/request RSS, not process lifetime high-water.
- The README figure should stay focused on the user-facing comparison: ARLE vs
  mlx-lm. Internal residency tradeoffs belong in the dedicated memory-fix entry.

## Rule

README benchmark figures must include the memory axis when a performance claim
depends on Apple unified memory behavior. If a point is missing or only valid for
one metric, leave the gap visible and explain it in the raw evidence.
