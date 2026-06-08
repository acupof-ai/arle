# DSv4 >128-token garbage fixed — RoPE base/YaRN is PER-LAYER (compressed layers use compress_rope_theta), not per-tensor-role

**Date:** 2026-06-08. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** root cause CONFIRMED + fixed; catastrophic collapse gone, needle
retrieves where it was garbage. A smaller long-context residual remains (see
Residual). Supersedes the "shared prefill corruption, not pinned" framing in
`errors/2026-06-08-dsv4-longctx-prefill-corruption-not-attention-kernel.md`.

## Context

The 900K-needle blocker: DSv4 produced coherent output only ≤~80 tokens, then
collapsed to garbage (`hit=False`, `" secret. \n- \n- \n-"`), position-
independent (a needle next to the query at depth 0.9 also failed). First prefill
token already wrong ⇒ prefill bug, not decode. Earlier same-binary env A/Bs left
it (FlashMLA / DSA-indexer / DeepGEMM all toggled, default = best), so it was in
a path SHARED by both attention kernels and both linear backends.

## Root cause

RoPE base + YaRN was selected **per-tensor-role** — Q / SW-K / output-inverse-rope
always `rope_theta=10000` + no YaRN, only the compressor's compressed-K using
`compress_rope_theta=160000` (`attention.rs:4044`, constant since the rewrite's
first DSv4 commit `9def46fb`). On every CSA (cr=4) / HCA (cr=128) layer this dots
a **10000-rotated Q against a 160000-rotated compressed-K** → the relative phase
is not a function of `pos_q − pos_k`, so attention corrupts in a position-
dependent way that worsens with length and breaks even near-query/high-abs-pos
needles. Short prompts survive because while everything fits the sliding window
(`sliding_window=128`) the SW path Q·SW-K is internally self-consistent (both at
the wrong-but-uniform 10000); once compressed blocks join the softmax (seq > ~80)
their mis-rotated logits dominate and drown the correct SW contribution.

**Ground truth = SGLang** (runs these exact weights coherently): RoPE base is
**per-LAYER**, one base for Q *and* K:
- `deepseek_v4.py:271` — `rope_base = compress_rope_theta if compress_ratio else rope_theta`.
- `fused_qk_norm_rope_swa_store` ropes Q **and** SW-K with one per-layer cos/sin
  cache (base 160000 on compressed layers) → Q, SW-K, compressed-K all 160000+YaRN there.
- `softmax_scale = head_dim**-0.5`, **no mscale** (line 251) — YaRN only touches the
  freq ramp, no attention-scale coupling.

The prior "always rope_theta, no YaRN" matched the old-tree `reference.rs` and
`errors/2026-05-29-dsv4-longctx-rope-conflation` (which asserted the model
"intentionally" dots Q@10000 against compressed-K@160000) — that was wrong;
reference.rs encoded the same bug. Docs are not truth.

## What Worked

`crates/infer-cuda/src/attention.rs`, `mla_attention`:
- `rope_base`/`original_seq_len` now per-layer: `compress_ratio > 0` →
  `(compress_rope_theta, original_max_position_embeddings)`; else
  `(rope_theta, 0)`. One source feeds Q-prep (`dsv4_prepare_qk_*`), SW-K prep, the
  SW + CSA/HCA hybrid output inverse-rope (all confirmed to read these locals).
- Core compressor `original_seq_len` `0 → original_seq_len` (YaRN on for
  compressed layers, matching Q + the indexer compressor + SGLang's compressor freqs_cis).

Theta-value swap, **zero added compute**. Built incrementally on pod (no git
reset — `git reset --hard` would wipe the tn-pushed fix), `ROPE_BUILD_EXIT=0`.

**Needle A/B** (serve `/v1/completions`, default config, greedy temp=0, needle
`738291` at depth 0; same binary, before = `errors/2026-06-08...`):

| prompt tok | before | after |
|---|---|---|
| 75  | `738291` ✓ | `738291` ✓ |
| 86  | partial `738` | `738291` ✓ |
| **115** | **garbage `\n- \n-`** | **`738291` ✓** (2/3 runs; 1/3 `738738` partial) |
| 241 | garbage | `738…` partial recall (coherent, no garbage) |
| France smoke | — | `Paris` ✓ |

n=115 went from *deterministic garbage* to *mostly-exact retrieval* — not
explainable by MoE run-to-run non-determinism (which can't turn deterministic
garbage into exact retrieval). The 1/3 partial at the 115 boundary is the known
MoE non-determinism (`reference_dsv4_moe_nondeterminism_confounds_4096_parity`).

**Perf (same serve, c=1):** decode ~39 tok/s / 25.5 ms-tok, TTFT ~32 ms —
neutral-to-slightly-better vs the ~27 ms-tok baseline
(`reference_dsv4_decode_6ms_path_state`). No regression (B=1 decode is GPU-bound;
a theta constant swap adds no compute).

## Residual (next)

seq ≥ 241 recalls the needle *region* (`738…`) but loses trailing digits
(`291`). No longer catastrophic — a separate, smaller mechanism (compression
fidelity at cr=4/128 for early-position fine detail, or a long-position rope/YaRN
detail). Investigate independently; do NOT conflate with the collapse above.

## Rule

- DSv4 RoPE base/YaRN is **per-LAYER**, not per-tensor-role: compressed layers
  (CSA cr=4 / HCA cr=128) rope Q + SW-K + output + compressor at
  `compress_rope_theta` + YaRN; pure-SW layers (cr=0) at `rope_theta`, no YaRN.
  Q MUST share the compressed-key theta or Q·compressed-K collapses past the SW
  window. The canonical reference is SGLang `deepseek_v4.py:271` +
  `fused_qk_norm_rope_swa_store`, NOT the old-tree `reference.rs`.
- "Garbage after a few tokens" on a compressed/sparse-attention model is a RoPE
  base/scaling-selection suspect first — and it's position-dependent (near-query
  high-abs-pos needle) when Q/K thetas mismatch, vs a global theta error which
  retrieves near-query needles fine.
- Validate the flip with a needle **determinism control** (same prompt ×3): a
  catastrophic→exact flip across the boundary length is the license; one partial
  run inside an otherwise-exact set is MoE non-determinism, not the old bug.
- Build the pod tree **incrementally** when the fix was tn-pushed (not committed
  to origin/main); `build3.sh`'s `git reset --hard origin/main` silently reverts it.
