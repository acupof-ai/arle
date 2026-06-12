# DSv4 B=1 −23% regression: trace-diff conviction + decode-band 64-aligned packing recovers +9%

## Context

DSv4-Flash no-spec B=1 decode (8×H20 TP=8, `arle_serve_allreduce.sh`, France
128-tok raw completions) regressed 43 → 33.5 tok/s across the 2026-06-11
commit window. Commit bisect produced mush — a docs-only window "dropped"
3.2 tok/s (boot/binary lottery ±3), so per-commit ladders could not converge.
Per ckl: fix the problem, never revert.

## What Worked

**Trace census diff over bisect.** Captured nsys at both endpoints
(d7be8c9b "ERA" 43 tok/s ×2 vs c7fe1aea "C2" 33.67) and diffed per-token
kernel totals. The entire GPU-work regression sat in ONE lane — the routed
MoE grouped-contiguous pipeline — while M=1 dense GEMMs were untouched:

| kernel (routed instance) | ERA | C2 | mechanism |
|---|---|---|---|
| pack_quantize grid | 192 | 28672 | rows 8 → 1152 (**149×**) |
| swiglu_quantize avg | 2.8µs | 20.7µs | 7.4× |
| scatter_all_route_slots avg | 1.7µs | 17µs | 10× |
| grouped GEMM pair | 22.4µs | 58.6µs/layer | block_m 64→128 templates |
| GPU busy /tok-unit | 2.865 | 3.368 | +4ms/tok real |

**Mechanism**: `ba1dd607` ("harden dsv4 review findings") switched the decode
MoE tail from compact packing (8 rows at B=1 — but in violation of the
MGroupedContiguous per-tile single-group contract; boundary rows silently
used the wrong expert whenever ≥2 local experts were active in a layer) to
128-aligned packing. Correct, but `deepgemm_contig_rows_cap(8, 32, 128)` =
1152 pad rows ground by every row-linear kernel, every MoE layer, every step.

**Fix forward (a1e15307)**: SM90 warpgroup MMA grants block_m ∈ {64,128}
only, so 64 is the smallest legal per-group alignment. The native bridge
takes `mk_align` and caps its block_m candidates at it (`GemmDesc.max_block_m`
— a tile can never span two 64-aligned groups); the DSv4 tail packs
64-aligned in the decode band (R ≤ 128 routes), 128 above. CPU contract test
`decode_band_64_alignment_needs_block_m_cap` pins both directions (64-align
holds at block_m=64, violates at 128).

## Results (same binary, same session)

| arm | B=1 tok/s | notes |
|---|---|---|
| C2 / 33-band (×7 probes) | 33.4–33.8 | regressed base |
| **64-align fix (a1e15307+68833c3f)** | **36.90 / 35.95** | two runs, n=4 each |
| ERA control d7be8c9b | 42.92 / 43.45 | re-measured this week |

nsys receipts at the fix: routed pack grid 28672→**14336**, swiglu/scatter
14336→**7168** (rows 1152→576 as designed); GPU busy 3.368→**2.823**/tok-unit
— *below* ERA's 2.865 (block_m=64 GEMM configs beat ERA's at this shape).

Needle gate ×3 same-config (512/2048/6000, depth 0.5): 512 exact DET,
6000 exact DET, 2048 partial×3 stable — the 2000-band exact↔partial flip
pre-exists in the locked baseline envelopes ("inside the MoE floor",
2026-06-10/12 entries). No miss, no garbage class. PASS.

## Remaining gap (open, next knife)

Wall 36.4 vs ERA 43 ≈ 5ms/tok now sits in launch GAPS, not GPU work and not
CUDA API volume (launches 2774 vs 2753, API time flat, HtoD −42/tok). The
second regression segment is host-side step latency — needs a per-step
launch-timeline probe, not more kernel work. The decode-band grouped-GEMV
lane (existing `dsv4_fp8_grouped_gemv_pair_batch` + `swiglu_clamped_routes`
+ UE8M0 scale tables) remains the compute end-game on top.

## Rule

- **Cumulative regressions don't bisect.** When a window's probes band into
  a continuum and "inert" commits move the number, stop laddering: trace
  BOTH endpoints and diff per-token kernel totals. Counts and grid dims are
  lottery-proof; per-launch avgs across boots are not.
- **A contract hardening that pads a hot lane needs a shape-banded cost
  check.** ba1dd607 was correct; its 128-alignment was sized for prefill and
  silently cost the decode band 23%. Alignment, like block size, is a
  per-band tunable bounded below by the hardware MMA granule.
- Build-gate on the script's own `INCR_BUILD_EXIT=0`, never a wrapper echo —
  a failed build silently re-serves the stale binary (one full bench+nsys
  cycle wasted here on exactly that; `ceil_div` fix 68833c3f unblocked HEAD).
