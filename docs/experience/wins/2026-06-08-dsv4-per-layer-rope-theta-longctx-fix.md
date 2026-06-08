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

The prior "always rope_theta, no YaRN" matched the old-tree `reference.rs` (and a
2026-05-29 fix that "validated" the inverted direction — see the useful-error
trail in [`../errors/2026-06-08-dsv4-longctx-prefill-corruption-not-attention-kernel.md`]).
That direction was wrong; `reference.rs` encoded the same bug. SGLang > reference.rs.

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

## Long-context follow-on (`6e2e572e` — reconcile + caps)

Reaching past 32K needed two more unrelated fixes (committed `6e2e572e`), found
by sweeping the needle up the length axis:

- **`chunked_prefill_size = 4096`** (was `max(.., dsv4_max_seq_len())`). The
  prefix-cache fix (`702454fe`) pinned single-chunk prefill to dodge a
  *hypothesized* multi-rank NCCL chunk-boundary desync, but origin/main's
  `query_chunk` memory-bounding asserts each prefill call passes ≤
  `DSV4_PREFILL_QUERY_CHUNK`(4096) query tokens → >4096 prompts hit
  `M=6400 > query chunk 4096` and crash the engine. **The desync hypothesis was
  FALSE** (verified: a *single* 64K request = 13 contiguous chunks, multi-rank,
  completes in 29.5s, no hang — the "hang" seen first was two concurrent requests
  on a KV-budget-clamped 1-slot serve).
- **Lift `max_prompt_tokens`/`max_total_tokens` to `dsv4_max_seq_len()`** for
  DSv4 (`loaded.rs`). The 32768/65536 defaults silently returned an **empty**
  completion (0.1s, `out=''`) for any prompt >32K — the real 900K blocker, masked
  as "model can't go long."

Extended needle ladder (depth-0, default config): **32K (25,567 tok) → `738291`
exact ✓**; 64K (51,107) processes in 29.5s (recalls `738…`, digits noisy).

## Residual (next, separate)

1. **Borderline exact-digit retrieval >~2K.** Reliably finds the needle *region*
   (`738…`) but the trailing digits (`291`) are noisy/non-deterministic (8K = 1/2
   exact, 32K flips run-to-run). Compression fidelity (cr=4/128 blur of
   early-position detail) + MoE non-determinism. Not catastrophic; investigate
   apart from the collapse above.
2. **900K prefill is host-bound.** At 900K the `infer-engine` thread runs at 90%
   CPU while all 8 GPUs sit at 0% util / 120 W idle — per-chunk host-side prep
   (CSA selection / metadata over ~225K compressed blocks × ~220 chunks) starves
   the GPU, so prefill crawls (>10 min, no GPU progress; 32K/64K are fine). A
   perf wall at extreme length, **not** correctness. Profile + move per-chunk prep
   off the engine critical path.

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
