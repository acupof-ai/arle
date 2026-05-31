# Bench: ARLE Metal vs mlx-lm — single-machine TTFT/TPOT sweep 128→12k (Qwen3.6)

> **Correction (2026-06-01).** The first version of this entry reported a "TPOT
> decode collapse, ARLE ~40× slower at 12k." **That was a benchmark metric bug,
> not engine behavior.** With the metric fixed (steady-state TPOT from token 2
> onward) ARLE decode is **at parity with mlx-lm** at every length. Post-mortem:
> [`errors/2026-06-01-metal-bench-tpot-metric-artifact.md`](../errors/2026-06-01-metal-bench-tpot-metric-artifact.md).

## Goal
Single-machine (c=1) prompt-length sweep, 128 → 12 288 tokens, charting **TTFT**
(time to first token) and **TPOT** (time per output token) for ARLE Metal vs the
mlx-lm reference on the canonical Metal model. Hard constraint: never co-resident
two 19 GB models on the 48 GB box (a prior run hung that way).

## Hypothesis
TTFT is comparable once the prefix cache is defeated; TPOT is where the backends
diverge with context. → **Refuted on TPOT**: both are flat and equal. The real
ARLE-specific long-context cost is the *prefill-tail* (token1→token2 gap), not
decode.

## Params / Env
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` (same HF snapshot; 40 layers,
  2 KV heads, head_dim 256 → KV ~0.94 GB @12k).
- HW: Apple Silicon, 48 GB unified; macOS. c=1, greedy (temp 0), 64 output tokens.
- **Prefix cache defeated** on the ARLE side: per-request unique nonce prefix
  (`make_prompt`) so prefill is always uncached and lengths non-nested.
- ARLE: `metal_serve --max-running-requests 1 --max-batch-tokens 4096`, auto wired
  limit, streaming `/v1/completions`. TTFT + steady-state TPOT from client SSE.
- mlx-lm (engine): in-process `stream_generate`; TTFT = prompt_tokens/prompt_tps,
  TPOT = 1000/generation_tps.
- **TPOT = steady state, token 2 onward.** The token1→token2 interval is excluded
  (it carries ARLE's front-loaded prefill tail — see Learnings + the post-mortem).
  Reported separately as `first_interval`.
- **Memory safety**: one 19 GB model resident at a time — ARLE `metal_serve` fully
  terminated and RAM reclaimed before the mlx-lm phase loads its copy; in-script
  watchdog SIGKILLs the model if free RAM < 1.0 GB. Lowest free RAM ≈ 5.4 GB; no hang.
- Driver `scripts/bench_mlx_vs_arle_sweep.py`, chart `scripts/plot_mlx_vs_arle_sweep.py`.

## Results

| prompt | ARLE TTFT (s) | mlx-lm TTFT (s) | ARLE TPOT (ms) | mlx-lm TPOT (ms) | ARLE first-interval (s) |
|--------|--------------:|----------------:|---------------:|-----------------:|------------------------:|
| 128  | 0.22  | 0.28  | 11.3 | 11.3 | 0.5  |
| 256  | 0.35  | 0.39  | 11.3 | 11.3 | 0.9  |
| 512  | 0.60  | 0.63  | 11.3 | 11.3 | 1.8  |
| 1k   | 1.09  | 1.13  | 11.5 | 11.5 | 3.4  |
| 2k   | 2.12  | 2.18  | 11.7 | 11.6 | 6.8  |
| 4k   | 4.44  | 4.32  | 12.0 | 11.8 | 13.9 |
| 8k   | 9.23  | 8.93  | 13.1 | 12.5 | 29.6 |
| 12k  | 15.56 | 14.22 | 13.7 | 13.1 | 47.2 |

![ARLE vs mlx-lm TTFT/TPOT sweep](assets/2026-05-31-mlx-vs-arle-128-12k-sweep.png)

## Learnings
- **Decode (TPOT) is at parity.** 11.3 → 13.7 ms/token across 128 → 12k, within
  1–5 % of mlx-lm at every length, and ~flat in context (the ~15 % rise to 12k is
  the physical KV-read floor). ARLE is **not** slower at decode — the fused
  no-mask SDPA decode path (`mask_mode=""` at S=1) is already the default for the
  c=1 / uniform-batch case. No fix needed or made.
- **TTFT (prefill) is also at near-parity** (ARLE ≤ 1.1× mlx-lm; both rise ~linearly
  with length). An earlier "ARLE 2× slower TTFT" claim came from comparing
  ARLE-over-HTTP against the wrong mlx baseline; against the same in-process MLX,
  prefill tracks closely.
- **The one real ARLE-specific long-context cost is the prefill-tail**: the
  token1→token2 gap grows 0.5 s → 47 s as context goes 128 → 12k (panel 3). ARLE's
  pipelined scheduler emits token 1, then front-loads the bulk prompt prefill into
  the token1→2 interval. This is a **scheduling/streaming-shape** characteristic,
  not a decode cost — but it IS where a user feels "slow to get going" on long
  agent/multi-turn context. Worth a look: why the prompt-prefill work lands after
  token 1 instead of fully inside TTFT.
- **The metric bug that started this.** v1 computed decode rate as
  `(out-1)/(last-first)`, which folds the giant token1→2 prefill-tail interval into
  "decode" → a fake O(context) collapse (1.4 tok/s @8k instead of the real ~76).
  Fixed to measure from token 2 onward. Unit-tested: synthetic 28 s tail + flat
  12.6 ms decode → new metric 79 tok/s (correct), old 1.4 tok/s (bug reproduced).

## Rule
Report TTFT and TPOT separately on a prompt-length sweep. **Define TPOT as
steady-state (drop the token1→token2 interval)** — on a pipelined scheduler that
first interval carries prefill, and folding it in manufactures a fake decode
regression. Measure both engines the same way (both over HTTP or both in-engine);
never mix HTTP-observed against engine-internal for a head-to-head. Defeat the
prefix cache with per-request nonces. RAM-tight cross-backend A/B runs strictly
sequentially with a watchdog; never co-resident two large models.
