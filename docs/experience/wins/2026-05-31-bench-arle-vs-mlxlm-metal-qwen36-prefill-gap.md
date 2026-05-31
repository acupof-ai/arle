# A/B: ARLE Metal vs mlx-lm on Qwen3.6 — decode/short-prefill wash, long-prefill 2.3× gap

## Goal
Confirm-or-refute the user claim "ARLE Metal 性能没比 mlx python 好" / "prefill 很慢"
with a controlled A/B, and locate the gap if real.

## Hypothesis
Decode is roughly comparable; the felt slowness is prefill, and it widens with
prompt length (ARLE doesn't batch prefill as efficiently as mlx-lm).

## Params / Env
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` (same HF snapshot both sides).
- HW: Apple M4 Pro, 48 GB unified; macOS 26.3.1. ARLE after the macOS-26 MLX JIT
  preamble rebuild fix.
- c=1, greedy (temp 0), warm. Shapes: 512/128 (decode), 2048/16 + 2048/128 (prefill).
- ARLE: `metal_serve --max-batch-tokens 4096` (single-chunk prefill), `/v1/completions`
  streaming, `ignore_eos`. **Prefix cache defeated** with a unique nonce at the start
  of each prompt (an earlier best-of-2 run was invalid — the 2nd identical-prompt run
  hit the prefix cache and reported `prefill=54287 tok/s`, physically impossible).
- mlx-lm 0.31.2 `mlx_lm.generate --temp 0 --ignore-chat-template --verbose True`
  (fresh process per run = no cross-run cache), prompt/generation tok/s read from
  `--verbose`. Run sequentially (metal_serve killed before mlx-lm) to avoid 40 GB
  RAM contention on the 48 GB box.

## Results

| Metric | shape | ARLE | mlx-lm | Δ |
|---|---|---|---|---|
| decode tok/s | 512/128 | 89.7 | 86.2 | wash (+4%) |
| decode tok/s | 2048/128 | 87.7 | 89.3 | wash (−2%) |
| prefill tok/s | 512 | 222 | 214 | wash (+4%) |
| **prefill tok/s** | **2048** | **221** | **508** | **mlx-lm 2.3× faster** |
| TTFT | 2048 | 8.3 s | ~3.7 s | mlx-lm 2.2× faster |

Noise gate (`feedback_matched_ab_for_small_bench_effects`, ≤10% = wash): decode and
512-prefill are wash. The 2048-prefill gap is 130% — far beyond noise, single-session
conclusive.

## Problems
- First A/B pass was prefix-cache-polluted (best-of-2 → cache hit). Fixed by
  nonce-prefixed prompts + single cold run. Lesson re-confirmed: ARLE's prefix cache
  silently makes a repeated-prompt benchmark report fantasy prefill numbers.

## Learnings
- **Decode is at parity** (~88 tok/s both). ARLE is NOT slower at decode.
- **ARLE prefill tok/s is FLAT (~220) regardless of prompt length**, while mlx-lm
  scales (214 @512 → 508 @2048). So short prompts are a wash; long prompts (agent
  context, multi-turn history) are where ARLE falls 2.3× behind and "feels slow to get
  in." Root-cause hypothesis (unverified, for the next workstream): ARLE's Metal prefill
  is not running a single batched forward over all prompt tokens the way mlx-lm does —
  per-position / per-step encode overhead dominates and doesn't amortize with length.
  Raising `--max-batch-tokens` to 4096 did NOT help (still 221 tok/s for 2048), so it's
  not the chunking — it's the per-forward efficiency.
- **Actionable next step (separate from this session's scope):** profile the Metal
  prefill forward (Xcode Metal capture / MLX trace) on a 2048-token prompt to find why
  throughput doesn't scale; this is the single highest-value Metal perf item, and the
  one that would actually beat mlx-lm.

## Rule
"Not better than mlx-lm" decomposes: verify decode AND prefill AND prefill-vs-length
separately. A flat prefill-tok/s-vs-length curve (vs a rising one) is the signature of
an un-batched prefill and is the real gap — not decode.
