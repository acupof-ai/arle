# The chunked-GDR GSM collapse was bf16 drift on a knife-edge harness, not a kernel bug — 2026-08-02

## Context

`--qwen35-gdr-chunked` (FlashQLA, −27% on 33K prefill) passed needle ×3 and
greedy-64 identity, defaulted on, then scored **GSM8K 11/100 vs recurrent
46/100** (raw 8-shot completion, greedy) — 35 one-directional failures, most
as an immediate-EOS first token after "Answer:". Default reverted
(`715c37a0c`), then a full adjudication.

## Root Cause

Not a kernel bug and not wiring. The evidence chain, in order:

1. Standalone same-input runs: bit-deterministic (no race), and chunked ≡
   recurrent to **rel-L2 4.5e-3** at every S ∈ {130..1358}, chained SEG=64,
   masked tails, in-place h0/ht, slow-decay g — all bf16-class.
2. `ARLE_FQ_PARITY` (in-serve probe, the failing request itself): all 96
   (layer, segment) pairs green — state ≤3.7e-3, output ≤5.7e-3.
3. Teacher-forcing: given " Janet has 16 - 3 - 4 =", the chunked arm
   continues flawlessly to "#### 18". Only the boundary token flips.
4. Sampling at temp 0.7 quantified the boundary: **the recurrent arm itself
   emits EOS 3/12 and think-mode 5/12** at that position — the raw few-shot
   format on a thinking model sits on an EOS knife edge. Chunked's drift
   moves EOS ≈25% → ≈50%; greedy collapses that into whole-item failures.
5. **Chat template (the real serving format): chunked 14/15 = recurrent
   14/15.** No quality difference where the model is actually served.

Fast-math was exonerated by A/B (5/30 with and without). The bf16
intermediates (V′, a_inv) vs the recurrent kernel's fp32 chain are the drift
source; the drift only matters where the format is fragile.

## Fix

Adjudication tooling kept: `ARLE_FQ_PARITY=1` replays the recurrent kernel
per (layer, segment) and logs rel-L2. Default stays OFF until a broader
chat-format battery (≥100 GSM-chat + a second task + needle) licenses the
flip; the raw-completion surface genuinely degrades and that must be a named,
accepted trade before any default change.

## Rule

**A same-binary accuracy A/B can indict the harness, not the kernel — decide
with a probability measurement, not a greedy one.** Greedy collapses a
25%→50% distribution shift into 0/1 item failures; sampling at temp>0
revealed the baseline was already on the cliff. And per-layer parity probes
beat end-to-end evals for locating this class: 96 green tensor comparisons
turned "the kernel is wrong" into "the harness is fragile" in one run.
