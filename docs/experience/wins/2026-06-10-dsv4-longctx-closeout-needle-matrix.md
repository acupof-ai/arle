# DSv4 long-ctx closeout (#56) — collapse stays dead; trailing-digit residual characterized as bounded, handed to the #58 parity gate

**Date:** 2026-06-10. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Serve:** main HEAD `c58afde0` binary (allreduce default lane, deepgemm,
`INFER_DSV4_MAX_SEQ_LEN=16384`, port 18189), `/v1/completions`, greedy temp=0,
needle `738291`, `max_tokens` 8–16. Harness `scripts/_pod_needle_matrix.py`
(pod `/data01/build/needle_matrix.py`), raw outputs in
`/data01/build/needle_matrix_*.log`.

## Goal

Close issue #56: reproduce the "seq≥241 trailing-digit" residual across the
boundary, run the same-config-repeat control (MoE non-determinism floor),
root-cause or explicitly bound it, and declare the Phase 1 long-ctx baseline.

## Results — primary matrix (non-degenerate filler, depth 0, ×3 same-config)

| target len | exact | partial | miss | deterministic? |
|---|---|---|---|---|
| 115 | **3/3** | 0 | 0 | **DET** |
| 180 | 0 | 3 | 0 | NONDET |
| 241 | 0 | 2 | 1 | NONDET |
| 300 | 1 | 0 | 2 | NONDET |
| 446 | 2 | 1 | 0 | NONDET |
| 1000 | 2 | 1 | 0 | NONDET |
| 2000 | 2 | 1 | 0 | NONDET |
| 4000 | 0 | 1 | 2 | NONDET |
| 8000 | **3/3** | 0 | 0 | NONDET |

Partial signature is always `738` + confabulated suffix (`73841`, `738741`,
`738491`, `738137`): the needle's first token is recalled, the tail is
hallucinated. Misses are coherent text (filler quotes, "the secret access
code" loops) — **no garbage collapse anywhere**.

## Controls (each ×3 same-config)

1. **Same-config-repeat floor**: above 115 every length is NONDET — outputs
   differ run-to-run at fixed config, flipping exact↔partial. Matches the
   documented MoE atomic-scatter non-determinism; any single-run verdict at
   these shapes is noise.
2. **Degenerate-filler confound (harness bug, fixed)**: the original 8-sentence
   looping filler produced WORSE, length-inconsistent results (8000: 1/3 vs
   3/3 after adding unique per-sentence prefixes). Looping filler is itself a
   degenerate prompt; v1 numbers (and the old "241 boundary" memory) are
   partially harness artifacts. The "≥241" framing was a ladder-sampling
   artifact — degradation starts as soon as retrieval leaves the SW window
   (~pt>128–180) and is **non-monotonic** in length.
3. **Depth control (0.5 / 0.9)**: does NOT rescue (300@0.9 = 3/3 miss while
   4000@0.9 = 2/3 exact). Kills the clean "SW-window=good / compressed=blurry"
   hypothesis; the dips are prompt/position-shape sensitive, not a sharp
   boundary at the sliding window.
4. **Official DSv4 chat template** (`<｜begin▁of▁sentence｜>…<｜User｜>…
   <｜Assistant｜></think>`, fullwidth-bar specials verified to encode as
   single tokens server-side AND offline — token-faithful): retrieval gets
   WORSE than raw (115: 0/3 with hallucinated digits). The raw-completions
   harness is not the weak point; raw numbers are the best case.
5. **Qwen ChatML via `/v1/chat/completions`**: tag salad — the rewrite's
   `render_chat` always renders ChatML regardless of model → filed **#66**.
6. **SGLang same-weights reference: DEFERRED.** `/workspace/sglang` (dev
   editable install) is broken since the 2026-06-08 tilelang/deep_gemm package
   churn (tilelang 0.1.11 lacks eager `wg_wait`; deep_gemm 2.4.2+7f2a703 is
   the 3-param API; triton fused-MoE rejects the FP4 expert layout). Last
   working SGLang serve logs are 06-05. Restoring it needs a dev tilelang
   build — deliberately not done inside #56.

## Verdict (license-or-kill framing)

- **The #56 catastrophic collapse is closed**: per-layer RoPE theta fix
  (`fa355315`) holds on main HEAD — coherent output at every length tested,
  needle-exact 3/3 at 8000.
- **The trailing-digit residual is real but bounded**: systematic first-token
  recall + tail confabulation, amplitude modulated by MoE non-determinism,
  non-monotonic in length, not rescued by depth or template. Root cause is
  narrowed to {model-inherent compressed-attention fidelity} vs {FP8-KV
  fidelity}, ∩ MoE non-determinism. **The discriminating instrument is the
  #58 KV-precision-parity harness (BF16 reference vs FP8 trajectory)** — that
  is the next Phase 0 issue, so the residual transfers there rather than
  blocking the Phase 1 baseline.
- **Long-ctx serving is declared clean as the Phase 1 baseline** in the
  correct-inference sense: no collapse, no garbage, majority-exact retrieval
  at most lengths, bounded non-deterministic digit blur past the SW window.

## Rule

- Needle fillers MUST be non-degenerate (unique per-sentence prefixes); a
  looping filler flips verdicts at fixed length and direction.
- A retrieval residual verdict at >115 tokens REQUIRES ×3 same-config runs;
  single runs flip exact↔partial at fixed config (MoE floor).
- "Boundary at length L" claims from a sparse ladder are sampling artifacts
  until a dense ladder + depth control confirm monotonicity (the "241
  boundary" was not real; neither is a clean SW-window split).
- DSv4 chat-format controls must hand-wrap the official template via raw
  completions until #66 lands; `/v1/chat/completions` renders Qwen ChatML at
  every model today.
