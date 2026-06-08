# DSv4 >128-token collapse — the misdiagnosis trail (RESOLVED: per-layer RoPE theta)

**Date:** 2026-06-08. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** RESOLVED — root cause was per-layer RoPE theta; fix + truth in
[`../wins/2026-06-08-dsv4-per-layer-rope-theta-longctx-fix.md`] (`fa355315`).
This entry keeps only the *useful errors* — the wrong turns that cost time, so
they aren't repeated.

## The bug (one line)

Past the sliding window (~80–128 tok) DSv4 output collapsed to garbage,
position-independent, first prefill token already wrong. Root cause: compressed
(CSA/HCA) layers roped Q+SW-K at `rope_theta`(10000) while the compressor roped
compressed-K at `compress_rope_theta`(160000) → Q·K phase mismatch. See the win.

## Useful errors (what wasted time)

1. **"Parity exhibits it too" is not corroboration when the harness is defective.**
   The `dsv4_parity` example reallocates the SW ring per `forward_tokens` call
   (`dsv4_parity.rs:14-21`), so its decode can't see prior-step KV — its garbage
   said nothing about the serve. Confirm on the production path with a clean
   controlled repro before attributing a bug to "pre-existing."

2. **A fixed-length divergence dismissed as "precision margin" was the real break.**
   `8bcd8ce3` saw a divergence at exactly ~122 tok and called it precision. It was
   catastrophic (failed retrieval), not a margin — a §0 narrow-window framing trap.
   Greedy-decode the actual tokens + a determinism control (same prompt ×2) + a
   position control (needle near vs far query) *before* calling it precision/decode.

3. **The inverted-direction trap — `reference.rs` is not ground truth, SGLang is.**
   The earlier "fix" (2026-05-29, old pre-rewrite tree, commits
   `d61d26f4`/`8105d5c6`/`003c8370`) went the **opposite** way: it forced Q/SW-K to
   `rope_theta` on all layers to match `reference.rs`, and a needle sweep *appeared*
   to validate it 11/12 to 2047. That direction is **wrong** — SGLang
   (`deepseek_v4.py:271` + `fused_qk_norm_rope_swa_store`) ropes Q **and** K at one
   per-layer base = `compress_rope_theta` on compressed layers, and reversing it
   (this fix) takes 115-tok garbage→`738291` exact. `reference.rs` encoded the same
   bug; the old "validation" was confounded (likely carried by that change's *other*
   half — the FlashMLA output inverse-rope — and/or a shape/needle-code that masked
   it). Gate a RoPE/theta direction on the **canonical upstream**, not your own CPU
   reference, and treat a passing needle on a wrong-direction fix as possible
   confounding, not proof.

4. **Stop toggling, start reading.** FlashMLA / DSA-indexer / DeepGEMM were each
   env-A/B'd off and all *left the bug* (default = best) — that localizes to shared
   forward code, but the answer came from **reading the SGLang rope path**, not from
   bisecting a commit window. When every component toggle leaves a bug and the
   all-default path is best, the defect is in shared config/selection logic; diff it
   against the canonical impl.

## Rule

- Validate long-context (seq > sliding_window) with an **unambiguous needle**
  retrieval + determinism + position controls — never "looks coherent."
- For compressed/sparse-attention RoPE: base/YaRN is **per-LAYER, not
  per-tensor-role**; diff Q and K theta against SGLang, not `reference.rs`.
- A "validated" fix can be the **wrong direction** if its validation shape/codes
  are confounded; reverse-A/B against the catastrophic case (here 115 tok) before
  trusting it.
