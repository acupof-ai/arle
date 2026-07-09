# DSpark partial-ctx drafting (P2.5) — LICENSED; prefix-hit lane 101–112 tok/s, RNG cleared

## Context

[P2.5](../../plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md): prefix-cache-hit
requests degraded to plain decode forever — the draft-ctx append gate required
`ctx_len == start_pos` (fresh slot 0 vs restored start_pos>0) and `pending`
required coverage from 0. At OPD's ~91% hit rate DSpark was near-inert.
Implementation `8edde59c7`; pod round 8×H20 GPU 1, Qwen3.6-27B-FP8 + z-lab
DFlash draft, greedy, prefix cache ON, plain anchor 42.6–43.6 tok/s.

## What Worked

`Qwen35DsparkSlotState` gains `ctx_base`/`ctx_end` (buffer row = abs − base;
RoPE/attention positions stay absolute via a `ctx_base`-row offset into the
cos/sin tables + buffer-relative kernel start_pos). Prefill/warm-decode rebase
the empty ctx at the gap position instead of bailing; sliding draft layers are
exact once the tail ≥ window (2048), only the 1 full-attention layer is
approximate. `ctx_base==0` reduces to the prior byte-identical arithmetic.

Pod verdict — **LICENSED on the production multi-turn shape**:

| arm | lane | accept | tok/s |
|---|---|---|---|
| csv / rust fresh anchors | base=0 | 8.70 / 3.11 | 148.7 / 83.0 |
| same-prompt resend (whole restore) | base>0 all | 2.88 (−67% vs 8.70) | **95.1**, output byte-identical |
| multi-turn t1 → t2 → t3 | base=0 → base>0 | 4.27 → 3.81 (−11%) → 3.34 (−22%) | 121 → 112 → 101 |
| needle fresh / hit ×2 | | 4.64 / 2.62 | 66–70, **738291 exact 3/3** |

- Re-seed proven: every post-restore step logs `base>0`; zero plain fallbacks.
- Multi-turn (the real OPD rollout shape) holds accept within 30% and tok/s
  2.3–2.6× the plain anchor. The degenerate whole-prompt-restore shape (drafter
  blind to 464/467 ctx tokens) drops accept −67% — outside the band but far
  from the ~1/16 collapse KILL signature, and tok/s 95.1 still ≥ anchor with
  correctness clean. Sidecar fallback NOT needed.
- **RNG cleared (task #13-①)**: same-seed-twice at temp 0.7 PASSES
  byte-identical with `ARLE_DISABLE_PREFIX_CACHE=1` (0 base>0 lines = cache
  genuinely off); the 07-10 "determinism bug" was the lane/ctx-state confound —
  run 2's prefix-hit changes drafter ctx → different proposal chains → a
  different (equally valid) sampled realization. Cache-ON same-seed still
  diverges by design.
- Env-sweep smoke (4d8c8c827..f7b2467cb): plain decode 42.7 tok/s decode-only
  vs 42.6–43.6 baseline band → **Δ≈0%**, flag-plumbing-only claim holds.

## Rule

- A spec-decode determinism gate must control engine cache state (disable or
  flush prefix cache): rejection sampling is distribution-equal, not
  draw-equal, across different drafter-ctx states.
- Accept-split telemetry (`base=` in the phase line) turns "silently inert on
  prefix hits" into a grep — keep per-lane counters in every degradation path.
