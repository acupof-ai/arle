# prep-norm shuffle reduce — correct + cleaner, but wall-clock wash (no license)

> Status: Verified wash — 2026-07-06

## Context

Kernel micro-opt pass. `decode_prep_paged{,_hd256}.cu` RMSNorm did the cross-warp
accumulation in a `tid==0` serial loop (`for i<NUM_WARPS`); `prefill_attention_paged_prep`
already uses a warp-0 shuffle reduce. Replaced the serial loop with the shuffle
idiom (commit `d566a8d9`) — mathematically identical (`1.0f/sqrtf(total/HEAD_DIM+eps)`).

## Result

Matched A/B on pod (H20, Qwen3-0.6B dense, HD128 paged decode-prep, guidellm
`--quick` 512-in/128-out, rate=1,2,4,8, same binary-vs-parent, same GPU 2):

| c | ITL baseline→changed | out tok/s baseline→changed |
|---|---|---|
| 1 | 4.928 → 4.863 ms (−1.3%) | 201.3 → 201.8 |
| 2 | 4.094 → 4.041 ms (−1.3%) | 243.1 → 243.2 |
| 4 | 3.516 → 3.457 ms (−1.7%) | 281.2 → 285.6 |
| 8 | 3.101 → 3.093 ms (−0.3%) | 307.9 → 308.7 |

ITL consistently −0.3–1.7%, but **inside the run-to-run noise band**: c=1 TTFT
alone swung 9.4 vs 16.1 ms across the two runs (>40% jitter), so a <2% ITL delta
carries no signal. Needle gate: coherent + DET, no correctness regression.

## Root cause of the wash

`NUM_WARPS` is 4 (HD128) / 8 (HD256). The removed serial section is a 4–8
iteration `float` add behind one existing `__syncthreads` — a few ns on a
per-layer-per-token prep that is itself a tiny fraction of the decode step. The
change removes a real single-thread section but there was never enough of it to
move wall-clock. Classic narrow-window-vs-end-to-end trap (cf.
`2026-05-21-arle-cuda-opd-swiglu-fused-kill.md`).

## Decision

**Keep the change** — it is a byte-equivalent readability/uniformity improvement
(the file now matches the prefill idiom, one canonical reduction shape) and does
not regress. But it is **NOT a perf win**; do not cite it as one. No default flip
was involved (it's on the always-on prep path, identity-preserving).

## Rule

Reduction-shape micro-opts on a loop bounded by `NUM_WARPS` (≤8) are wall-clock
wash by construction — the serial section is too small to profile. License a
prep-kernel reduction rewrite only if `NUM_WARPS` is large (≥32, i.e. ≥1024-wide
block) OR a paired nsys shows the prep kernel is a measurable decode-step fraction.
Otherwise treat it as a cleanup, not a perf item, and A/B only to confirm no
regression.
