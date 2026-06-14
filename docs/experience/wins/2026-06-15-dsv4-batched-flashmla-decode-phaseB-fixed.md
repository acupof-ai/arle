# DSv4 batched FlashMLA decode Phase B — FIXED (indices pitch) + verified correct, +3% @c=8

## Context
Phase B (batch the per-row attention kernel) garbled rows≥1 (KILL,
[errors](../errors/2026-06-14-dsv4-batched-flashmla-decode-phaseB-correctness-kill.md)).
Root cause found by researching the official FlashMLA b=N contract + SGLang
(`先用最好的` / understand-until-simple — "not hard, just not-understood"): the
batched indices WRITER wrote row r at `r*shape.topk_unified` (this layer's actual
mode-dependent topk) but the READER used `self.max_topk_unified` (the scratch's
allocated max-over-all-layers pitch). For non-max layers (every SW layer:
topk=sliding_window ≪ max), row r≥1 read the wrong offset → garbage; row 0 (offset 0)
always correct. **One-line fix** (`b566b548`): reader `stride_indices =
shape.topk_unified` (== writer pitch), matching SGLang's builder-pitch==reader-pitch
invariant.

## Verify (pod, b566b548, 8×H20)
- **decode-read coherence — the REAL correctness gate — PASS.** c=4 distinct, true
  4-way batched decode (MAX_ACTIVE=4): France→Paris (Eiffel/Louvre), Italy→Rome
  (Colosseum/Vatican), Canada→Ottawa, Egypt→Cairo (Nile) — all coherent, NO
  `Parisian TheThe`/`{ { {`/repetition. needle 6000 3/3 exact, 512 5/5 exact.
- **B=1 identity:** `--spec-type mtp`: needle 6000 3/3, **42.2 ms/step** (≈42.7,
  −1.2%), 2.14 acc-tok/step — MTP/per-row not regressed.

## Bench A/B (matched same-binary, 3× repeats, σ≈0.01) — Phase-A vs Phase-B(fixed)
Aggregate decode tok/s (/v1/stats Δgen/Δwall, peak active):

| ctx | c | Phase-A (per-row attn) | Phase-B (batched attn) | Δ% |
|-----|---|---|---|---|
| 256 | 2 | 47.55 | 47.25 | −0.6% (wash) |
| 256 | 4 | 63.10 | 63.33 | +0.4% (wash) |
| 256 | 8 | 75.82 | **78.09** | **+3.0%** |
| 32K | 8 | 42.74 | 43.37 | +1.5% |

Phase-B c=8 78.09 > the 73.65 reference > Phase-A 75.82; same agg/step=8.00, tighter
window (40.2s vs 41.5s) = the launch-gap reclaim with a now-correct kernel.

## The honest perf framing (§0 wall-clock-is-ground-truth)
The batched-attention reclaim is **+3% @c=8**, NOT the ~20-40% the earlier
NVTX-window analysis implied (per-row attention "25% launch-gap" + "92% of step").
**Wall-clock matched A/B is ground truth; the NVTX-window launch-gap % overestimated
the reclaim.** The BIG concurrency win is the batched LANE itself vs MTP-per-row
(+58.8% @c=8, [c-sweep](2026-06-14-dsv4-batched-decode-csweep-threshold-n4.md));
Phase B (batched attn) is a correct +3% refinement ON TOP of that lane, wash at c≤4.

## Rule
- **decode-read coherent continuation is THE correctness gate for a batched decode
  kernel.** needle (sequential c=1, never populates rows≥1) AND the numdiff harness
  (see below) both FAILED to gate this — only reading the actual multi-row generation
  caught the garble and confirmed the fix. Extends
  [[feedback_correct_inference_not_baseline_identity]].
- **The numdiff infra (`1c4414dd`) is itself a broken gate — false-positives.** It
  reports exceedances even at row0 (where the pitch it tests is mathematically
  irrelevant, `0*pitch=0`), i.e. it compares the batched lane against a b=1 reference
  that doesn't actually match the batched setup → a test-framework artifact (cf.
  [[reference_dsv4_b1_tps_msstep_vs_tokstep_diagnostic]] family;
  errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact). It needs a follow-up fix
  before it can serve as a correctness gate; decode-read is the trustworthy gate.
- **A perf hypothesis from an NVTX-window fraction must be wall-clock-A/B-confirmed
  before you believe its magnitude** (§0 framing trap): the "25% launch-gap → big
  reclaim" became +3% in matched wall-clock.
- **"Can't go further" was not-understood, not hard** (ckl): the official FlashMLA
  b=N is verified-existing (SGLang runs it); comparing our writer/reader pitch to it
  found the one-line bug. Research the existing verified thing; don't defer it as
  "deep".
