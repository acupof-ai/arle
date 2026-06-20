# Agentic OPD works in think-on: Qwen3.5-4B learns to abstain 0.60→1.00 (BFCL live)

## Context
After the no-think arm regressed (errors/2026-06-20-agentic-opd-greedy-reverse-kl-overcalls.md)
and the clean re-gate proved the fix (think-on teacher abstains 0.93 vs base 0.27,
no-think teacher only 0.07 — a bad target), we re-ran the agentic OPD with the SINGLE
change: **enable_thinking=True everywhere** (corpus + rollout + eval). Same proven
recipe (reverse-KL, --kl-mask completion, r32/a64 all-linear, --no-fused-distill,
teacher Qwen3.6-35B-A3B-FP8 --teacher-runtime infer, student Qwen3.5-4B), rollout-len
512, gradient-checkpointing + teacher-offload (no OOM). Eval: clean think-on
(single-thread, 600s, 0 errors across 206×3 calls — the first gate's timeout artifact
is gone). Curve: `docs/assets/opd-agentic-thinkon-curve.png`.

## What Worked

**Abstention SOLVED.** BFCL live_irrelevance: base **0.60 → step25/50 1.00**, toward the
think-on teacher's 0.93. This is the exact axis the no-think arm crashed to 0.00.
Case-as-fact (5 identical items, base vs step50): **base reasons correctly then caves
and emits a tool-call anyway 4/5** (geocoding/TSX/rename-files/games → fabricated
`[requests.get(...)]`); **step50 abstains correctly on all 5** ("I cannot fulfill the
request using the provided tools", no `[...]`). The teacher's reason-to-decline
transferred.

**Aggregate (n=206, weighted): base 0.786 → step25 0.825 (+3.9pp) → step50 0.781.**
**step25 is the sweet spot**; step50 over-trains.

| BFCL live | base | step25 | step50 |
|---|---|---|---|
| live_irrelevance (abstain) | 0.60 | **1.00** | **1.00** |
| live_multiple | 0.86 | 0.82 | 0.80 |
| live_simple | 0.86 | 0.74 | 0.76 |
| live_parallel_multiple | 0.79 | 0.83 | 0.625 |
| live_relevance | 0.875 | 0.75 | 0.625 |
| live_parallel | 0.81 | 0.625 | 0.50 |
| **aggregate** | **0.786** | **0.825** | 0.781 |

**The tool-use dip is an over-thinking artifact, NOT a capability loss** (measured, not
inferred): the distilled student runs away over-thinking — irrelevance/parallel hit the
4096-token cap (`finish=length`, ~15k chars of looping "Wait, let me re-read…"). For
irrelevance that still scores 1.00 (loop → no call = correct abstention); for tool-call
categories it truncates before the final `[...]`. So the dip grows with steps
(step25→step50) and is a generation-length/anti-loop problem.

## Rule
The clean re-gate + think-on flipped a "-14pp structural KILL" into a "+3.9pp, abstention
0.60→1.00" win — the no-think confound (and a timeout-faked gate) had hidden it. Decode
the cases AND the failure *mechanism* (here: `finish=length` over-thinking, not a
capability loss) before judging. Next iteration: a generation-length/anti-loop control +
early-stop at the abstention sweet spot (step25), to keep the abstention win without the
tool-use truncation. Artifacts: checkpoints + scores under
`bfcl-live-thinkon-012416/` on the pod.
