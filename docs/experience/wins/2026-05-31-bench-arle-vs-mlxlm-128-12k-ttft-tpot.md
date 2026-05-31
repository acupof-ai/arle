# Bench: ARLE Metal vs mlx-lm — single-machine TTFT/TPOT sweep 128→12k (Qwen3.6)

## Goal
Single-machine (c=1) prompt-length sweep, 128 → 12 288 tokens, charting the two
canonical serving-latency metrics — **TTFT** (time to first token) and **TPOT**
(time per output token) — for ARLE Metal vs the mlx-lm reference on the canonical
Metal model. Hard constraint: never co-resident two 19 GB models on the 48 GB box
(a prior run hung that way).

## Hypothesis
TTFT is comparable once the prefix cache is defeated; TPOT is where the backends
diverge with context length.

## Params / Env
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` (same HF snapshot; 40 layers,
  2 KV heads, head_dim 256 → KV ~0.94 GB @12k).
- HW: Apple Silicon, 48 GB unified; macOS. c=1, greedy (temp 0), 64 output tokens.
- **Prefix cache defeated**: every request gets a unique nonce prefix
  (`make_prompt`), so prefill is always uncached and lengths are non-nested —
  the trap that voided the first eb14f29e A/B.
- ARLE: `metal_serve --max-running-requests 1 --max-batch-tokens 4096`, auto wired
  limit, streaming `/v1/completions`; TTFT + TPOT from client SSE timing.
- mlx-lm (engine): in-process `stream_generate`; TTFT = prompt_tokens/prompt_tps,
  TPOT = 1000/generation_tps. 
- mlx-lm (HTTP): `mlx_lm.server` `/v1/completions`, identical SSE client to ARLE.
- **Memory safety**: one 19 GB model resident at a time — ARLE `metal_serve` fully
  terminated and RAM reclaimed before any mlx-lm phase; in-script watchdog SIGKILLs
  the model if free RAM < 1.0 GB. Lowest free RAM with one model ≈ 5 GB; no hang.
- Driver `scripts/bench_mlx_vs_arle_sweep.py` (+ `bench_mlx_http_decode.py`),
  chart `scripts/plot_mlx_vs_arle_sweep.py`, doc `scripts/gen_mlx_vs_arle_wins.py`.

## Results

| prompt | ARLE TTFT (s) | mlx HTTP TTFT (s) | mlx eng prefill (s) | ARLE TPOT (ms) | mlx HTTP TPOT (ms) | mlx eng TPOT (ms) |
|--------|--------------:|-----------------:|-------------------:|---------------:|------------------:|-----------------:|
| 128 | 0.22 | 0.14 | 0.30 | 19.6 | 13.1 | 15.5 |
| 256 | 0.35 | 0.31 | 0.44 | 26.4 | 11.5 | 13.8 |
| 512 | 0.59 | 0.44 | 0.70 | 39.2 | 12.7 | 14.8 |
| 1k | 1.13 | 0.75 | 1.33 | 69.6 | 14.4 | 19.0 |
| 2k | 2.29 | 1.17 | 2.24 | 121.8 | 11.9 | 14.5 |
| 4k | 4.91 | 2.25 | 4.54 | 231.7 | 12.5 | 16.9 |
| 8k | 9.66 | 4.70 | 9.38 | 485.4 | 13.3 | 12.6 |
| 12k | 16.31 | — | 13.97 | 767.1 | — | 13.2 |

![ARLE vs mlx-lm TTFT/TPOT sweep](assets/2026-05-31-mlx-vs-arle-128-12k-sweep.png)

## Learnings
- **TTFT (prefill)**: ARLE is ~1.62× mlx-lm HTTP at ≤512 and
  ~2.06× at 8k — a widening gap at long context.
  (The earlier "ARLE 2× faster" was a prefix-cache artifact; with nonces the
  curves track closely.)
- **TPOT (decode) is the real divergence**: ARLE TPOT rises steeply with context
  (~1.50× mlx-lm HTTP at ≤512 → ~36.5× at 8k),
  while mlx-lm TPOT is near-flat. A context-dependent TPOT slope cannot come from
  fixed HTTP overhead — it is a real ARLE Metal decode characteristic: per-token
  decode cost grows with KV/context far faster than mlx-lm's. Decode is measured **the same way on both sides** (mlx-lm via its own `mlx_lm.server` `/v1/completions` with the identical SSE client), so the TPOT comparison is transport-matched and apples-to-apples.
- **Highest-value Metal item**: the long-context decode path (TPOT@8k–12k). Agent /
  multi-turn workloads sit exactly there. Profile the decode attention + per-step
  scheduler overhead vs context length (Xcode Metal capture / MLX trace).

## Rule
Report TTFT and TPOT separately, on a prompt-length sweep — a single c/shape hides
the divergence. Defeat the prefix cache with per-request nonces or prefill numbers
are fantasy. Measure decode the same way on both engines (both over HTTP, or both
in-engine); never compare HTTP-observed TPOT to engine-internal TPOT for absolute
claims. RAM-tight cross-backend A/B runs strictly sequentially with a watchdog.
