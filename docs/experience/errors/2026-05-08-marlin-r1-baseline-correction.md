# Marlin Round 1-3 baseline correction — production Marlin is 1.64× at FP8 KV, not 1.06×

> Self-correction issued during cron tick 12:43. Triggered by reading
> codex's KV W4A8 plan (`docs/plans/M_quant-kv-w4a8.md`, commit `1e713de`)
> §4 stack table citing `f6f3af3` Marlin license bench at 11.76 ms ITL.
>
> Round 1 entry [`8e73dad`] reported ARLE Marlin "1.06× ITL" with KILL-withheld
> "implementation gap" framing. Round 2 [`8ad6b90`] eliminated alloc_zeros
> hypothesis (NULL). Round 3 [`1888d8a`] eliminated checkpoint variant
> hypothesis (NULL). Round 4 prep [`b3f22ea`] surveyed W4A16BatchGemv as
> BF16-native alternative dispatch.
>
> **All four rounds compared Marlin-with-`--kv-cache-dtype bf16`-forced
> against an FP8-KV baseline**. The 1.06× ratio is the artifact of the KV
> format mismatch, not Marlin under-performance.

## Anti-pattern caught (skill rule #8)

`kernel-optimization` skill anti-pattern #8: "Production default ≠ A/B
baseline (matched-control violation)". My Round 1 entry already cited
this anti-pattern w.r.t. Phase 0 KILL ("Phase 0 forced BF16 KV (graph
compat) compared against production auto-FP8 baseline → contaminated
-0.8% TTFT comparison"). Then **Round 1 itself committed the same anti-pattern**
in the opposite direction:

| Arm | Weight | KV dtype | Source |
|---|---|---|---|
| Baseline cited | BF16 | **auto-FP8** (production default at `786a20a`) | from Phase 0 KILL `8b4a03b` reading |
| Round 1 Marlin | W4A16 Marlin | **BF16** (`--kv-cache-dtype bf16` forced) | my Round 1 bench `8e73dad` |

KV format differed between arms. Round 1's "1.06×" included the BF16-KV-vs-FP8-KV
overhead (BF16 KV reads cost 2× the HBM of FP8 KV per token), masking Marlin's
actual weight-bandwidth saving.

## Production-default Marlin (correctly matched)

Codex's W4A16 Marlin license bench at `f6f3af3` ran Marlin
**without** `--kv-cache-dtype bf16` — production default = auto-FP8 KV.
Both arms (BF16 baseline and Marlin) on FP8 KV. Matched.

| Arm | Weight | KV (auto) | TTFT p50 | ITL p50 | out tok/s |
|---|---|---|---:|---:|---:|
| ARLE BF16 baseline (`786a20a`) | BF16 | FP8 | 1976 ms | **19.27 ms** | 153.83 |
| **ARLE W4A16 Marlin (`f6f3af3`)** | W4A16 (Marlin) | FP8 | 2565 ms | **11.76 ms** | **191** |

Δ:
- ITL: −39% (1.64× faster) → license `≤ 12 ms` fired
- out tok/s: +24%
- TTFT: +30% (Marlin per-launch cost still hits prefill — confirms my Round 4
  prep launch-density survey)

## What stays valid from Rounds 1-4

The **NULL elimination of alloc_zeros overhead** (Round 2) is still valid —
that hypothesis was tested in BF16-vs-BF16 condition (`linear.rs` change
applied, both arms BF16-KV). Round 2's NULL conclusion holds.

The **NULL elimination of checkpoint variant** (Round 3) is still valid —
both arms used same KV format.

The **launch density survey** (Round 4 prep) remains the grounded mechanism:
6 launches/Marlin call vs 1 launch/W4A16BatchGemv call. The TTFT regression
in production-default Marlin (+30%) confirms launch overhead bites prefill —
matching Round 4 prep formula.

## What changes for Round 4 #6

Round 4 #6 (hybrid dispatch override) now has a different ROI framing:

- **Original framing** (per Round 4 plan tick log): "close 13% engagement gap from 1.06× toward 1.5× license"
- **Corrected framing**: production Marlin already at 1.64× (license fired).
  Round 4 #6's job is to **shave the +30% TTFT regression** by routing
  small-batch decode through W4A16BatchGemv (BF16-native, no FP16 round-trip),
  while keeping prefill on Marlin tensor-core path.

Refined Phase 4 formula:

```
Production Marlin: TTFT 2565 ms (+30% vs BF16 1976 ms), ITL 11.76 ms (1.64×)
Round 4 #6 hybrid: prefill on Marlin (TTFT unchanged ~2565), decode on
  W4A16BatchGemv (saves 2 conversion launches × 252 GEMMs = 3-5 ms ITL).
Predicted ITL: 11.76 - 3.5 = 8.26 ms  →  2.33× vs BF16 baseline
Predicted TTFT: ~2565 (unchanged because batch>8 prefill stays Marlin)
```

Soft license at 2.0× decode (ITL ≤ 9.6 ms); hard license at 1.5× plus
TTFT no regression.

The **multi-shape gate** in the plan now has cleaner intent: high-conc
1k/256/c=64 (batch=64 → Marlin path,unchanged) and multi-tenant must
not regress; long-ctx 4k/c=4 should improve.

## Methodology lesson

Two adjacent error entries that violate the SAME anti-pattern in opposite
directions:

1. Phase 0 KILL `8b4a03b` (codex): Phase 0 BF16-forced vs FP8-default
   baseline → -0.8% TTFT artifact (Marlin path didn't even apply, but
   rule was the same — BF16 vs FP8 confound).
2. Round 1 `8e73dad` (Claude): Marlin BF16-forced vs FP8-default baseline
   → 1.06× ITL artifact.

Both authors knew anti-pattern #8 (it's in the skill at `faffcb0` line item
8). Both still committed it. The lesson: the anti-pattern is hard to catch
in the FORWARD direction because the BF16-forcing is "motivated" by trying
to isolate weight quant from KV quant. The SKILL needs to add a stronger
warning: **"matching KV format" is one of the eight checked controls; don't
force-BF16 just to skip KV-quant-attribution and call it 'isolated'**.

Will land this as a Phase 5 matched-control checklist hardening in the next
skill version (v1.2.0).

## Cross-references

- Codex W4A16 license bench: [`docs/experience/wins/2026-05-08-m_quant-w4a16-marlin-bench.md`](../wins/2026-05-08-m_quant-w4a16-marlin-bench.md) (`f6f3af3`)
- Round 1 entry: [`2026-05-08-marlin-w4a16-bench-implementation-gap.md`](2026-05-08-marlin-w4a16-bench-implementation-gap.md) (`8e73dad`)
- Round 4 prep survey: same entry, R4 §
- Round 4 #6 plan: [`docs/plans/M_quant-marlin-round4-hybrid-dispatch.md`](../../plans/M_quant-marlin-round4-hybrid-dispatch.md) (`6781f46`) — reframes ROI per this correction
- KV W4A8 plan: [`docs/plans/M_quant-kv-w4a8.md`](../../plans/M_quant-kv-w4a8.md) (`1e713de`) — orthogonal axis
- Skill: [`.claude/skills/kernel-optimization/SKILL.md`](../../.claude/skills/kernel-optimization/SKILL.md) v1.1.0 — anti-pattern #8 needs hardening to v1.2.0

## Rule update

Round 4 #6 plan must rebench BF16 baseline + Marlin baseline + hybrid arm
all in **production-default auto-FP8 KV** (no `--kv-cache-dtype` override).
Three-arm A/B at FP8 KV; do not introduce BF16 KV variable.
