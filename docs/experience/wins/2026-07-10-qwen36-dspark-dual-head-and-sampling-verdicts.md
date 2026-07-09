# Qwen3.6 DSpark dual-head + P2 sampling round — z-lab stays, heads no-license, sampling KILL as-is

## Context

Follow-up to [2026-07-10 block-draft license](2026-07-10-qwen36-dspark-block-draft-licensed-2p4x.md)
(z-lab DFlash backbone, greedy, 36ms/step, 2.4×). This round gates the trained
DSpark heads (Markov + confidence) and the P2 rejection-sampling verify
(a5d738953) on the pod. 8×H20, GPU 1, Qwen3.6-27B-FP8, HEAD 90c51bda8,
500-tok csv/rust prompts, plain-decode anchor 42.6–43.6 tok/s. Checkpoints:
`Hikari07jp/DSpark-Qwen3.6-27B-{AEON-draft,FR}` (AEON config.json recovered
from the remote — the weight pull had skipped it), FR converted via
`scripts/convert_dspark_speculators.py` (same-position, block=16, markov=256,
conf head present).

## What Worked / Verdicts

Greedy (accept = csv/rust means; needle MAGENTA 3/3 + self-consistency clean on
every arm):

| arm | block | accept | step ms (draft/verify/total) | tok/s | verdict |
|---|---|---|---|---|---|
| z-lab control | 16 | 3.26/4.18 | 8.1 / 25.1 / 36 | 98.4/119.3 | baseline reproduced exactly |
| FR conf=0 | 16 | 4.11/3.84 | 16.6 / 24.8 / 45.0 | 96.3–98.2 | +2.25× vs plain, **≤ z-lab → Markov head NO LICENSE** |
| FR conf=0.5 | 16 | 2.92/3.71 | 16.6 / ~60 / ~78 | 45–51 | wash |
| FR conf=0.7 | 16 | ~1.9 | 16.4 / 41 / 60 | ~27 | −36% |
| AEON | 11 | 3.66/3.86 | 11.9 / **100.7** / 115.6 | 38–40 | **−9% KILL** |

- **Markov head raises accept (+0.3–0.9) but costs draft 8.1→16.6 ms** — a
  per-row host loop; the accept gain never pays back at B=1. z-lab backbone
  stays the default drafter.
- **Confidence truncation is strictly harmful here**: shorter chains fall back
  onto the row-serial verify lane; conf=0 dominates. `--dspark-conf-threshold`
  default should stay 0 until a drafter whose accept curve rewards truncation.
- **AEON kill is a routing artifact, not head quality**: block=11 → 12-row
  verify misses the B≥16 GEMM lanes (verify 100.7 vs 24.8 ms at 17 rows).
  Any future block<16 drafter needs the small-M routing floor lowered first.
- Mode-line bug (cosmetic): `dspark.rs:103 mode_label()` prints
  `dflash-backbone` for every same-position checkpoint even when markov/conf
  heads are live (the conf sweep proves they act).

## P2 sampling verify — KILL as-is (two independent failures)

FR conf=0, temp 0.7 / top_p 0.95 / seeded; plain-sampling control alongside:

| gate | spec | plain control |
|---|---|---|
| same-seed-twice byte-identical | **FAIL** (texts diverge) | PASS |
| diff-seed differs | PASS | PASS |
| needle ×3 (max_tokens 700) | 3/3 | 3/3 |
| tok/s | 34.8 (step 116.5: draft 71.7, commit 18.7, verify 25.1) | **37.6–37.8** |

1. **Determinism**: the spec lane breaks (seed,position) reproducibility while
   the plain engine lane holds it — our bug in the spec sampling path, root
   cause pending (do not file a hypothesis; decode the diverging step).
2. **Perf**: host-side per-row sampling inflates draft 16.6→71.7 ms and
   accept_commit 2.0→18.7 ms → **−7.5% vs plain sampling**. Sampled OPD
   rollouts gain nothing until sampling moves device-side/batched.

Greedy lane unaffected after the sampling runs (45.0 ms step, matches
pre-round).

## Rule

- A drafter head only licenses if its accept gain beats its draft-cost delta at
  the serving batch size — accept-rate alone is not a license.
- block_size interacts with GEMM lane routing: verify rows = block+1 must reach
  the batched lane (B≥16 today) or the whole arm drowns in row-serial GEMV.
- temp>0 spec decode needs its own determinism gate (same-seed-twice on the
  spec lane, plain lane as control) — greedy gates prove nothing about the
  sampling path.
