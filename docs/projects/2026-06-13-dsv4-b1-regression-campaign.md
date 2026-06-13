# DSv4-Flash B=1 decode regression campaign — full record

Pod: 8×H20 TP=8/EP=8, `arle_serve_allreduce.sh`, France-128tok raw-completion
B=1 bench (`dsv4_ab_bench.py`). Model: DeepSeek-V4-Flash (hidden 4096,
moe_intermediate 2048, n_routed 256, **top_k 6**, n_shared 1).

## The symptom

DSv4 no-spec B=1 decode had drifted from ~43 tok/s to ~33.5 over the
2026-06-11 commit window (`d7be8c9b`..HEAD). ckl's directive: **fix the
regression, never revert.**

## Why bisect failed → trace-diff pivot

Per-commit ladders banded into a noisy continuum (±3 tok/s boot lottery —
a docs-only window "dropped" 3.2). So instead of laddering, we traced
**both endpoints** with nsys and diffed per-token kernel totals:
`d7be8c9b` (ERA, 43) vs `c7fe1aea` (C2, 33.7). The entire GPU-work
regression sat in **one lane** — the routed MoE grouped-contiguous pipeline
— while M=1 dense GEMMs and attention were untouched.

## Root cause: a correctness fix that padded a hot lane

`ba1dd607` ("harden dsv4 review findings") switched the decode MoE tail
from **compact** packing to **128-aligned** packing. This was a *correct*
fix: the DeepGEMM contiguous scheduler resolves one expert per BLOCK_M=128
tile from `m_indices[tile_start]`, but `dsv4_fill_m_indices_from_counts`
writes `m_indices` **per row** (`m_indices[offset+row] = local_expert`). At
B=1 EP=8, when a rank holds ≥2 of the token's 6 routes (common by
balls-in-bins), those rows share tile 0 and **all get computed against the
first route's expert** — a wrong-expert perturbation. Compact packing was
*fast but subtly wrong*; alignment is the cost of correctness.

But 128-alignment pads each expert's ~1 real decode row to 128. The
row-linear ops (`pack_quantize`, `swiglu_quantize`, `scatter`) then grind
the pad rows every MoE layer every step: routed `pack_quantize` grid went
**192 (era, compact) → 28672 (C2, 128-aligned)**, ≈149×. That is the
−23% mechanism.

## Fix 1 (shipped): decode-band 64-aligned packing — `a1e15307`

SM90 warpgroup MMA grants block_m ∈ {64,128}, so 64 is the smallest legal
alignment. The native bridge now takes `mk_align` and caps its block_m
candidates at it (`GemmDesc.max_block_m`) so a tile never spans two
64-aligned groups; the DSv4 decode band (R ≤ 128) packs 64-aligned, prefill
keeps 128. Pad rows halve.

- **33.5 → 36.9 tok/s** (3 boots: 36.90 / 35.95 / 36.71 — solid, not lottery)
- nsys grids halved as designed; needle 512/6000 exact-DET, 2048
  partial-stable (in the locked envelope) — correctness held.
- Build fix `68833c3f`: `ceil_div` needed `flashinfer::` qualification under
  `dsv4_flash` kernel set (a pre-existing HEAD break from #88 U4 `40b9a0e7`).
- Process lesson (`feedback_build_exit_marker_not_wrapper_echo`): a tmux
  build wrapper's `echo BUILD_EXIT=$?` captures the echo's exit, not the
  build's — a failed build silently re-served the stale binary and cost one
  full bench+nsys cycle. Gate on the build script's own `INCR_BUILD_EXIT=0`.

## The probe ladder: where did the residual 36.9-vs-43 go?

Built `ARLE_DSV4_STEP_PROFILE` (4 nested buckets, plain `Instant`, no CUPTI):
outer gap/fwd → inner layers-launch/tail → sub attn-half/moe-half → tail
backlog/head_hc/norm/lm_head/sample. Ran the SAME probe on HEAD and on a
tn-pushed era binary (`d7be8c9b`) for a clean A/B:

- HEAD: layers-launch 19.9ms host + tail 6.9ms (backlog **6.5ms**)
- era:  layers-launch 22.5ms host + tail 1.7ms (backlog **1.2ms**)

HEAD's host *launches faster*; the GPU falls **5.3ms further behind** because
each padded MoE-lane kernel takes longer on the critical path. The residual
is the MoE padding tax, not a host knife. (nsys under-measures it ~9× —
CUPTI serializes the launch stream and hides the bubble.)

## Fix 2 attempt (killed): grouped-GEMV decode lane — `cef442ec`

Idea: compact pack (GEMV has no per-tile contract) → per-expert pointer-table
pair-GEMV → SwiGLU → w2 GEMV → same scatter tail. Zero pad rows, no
activation quantize.

- First boot falsified the UE8M0 assumption — MoE expert scales are
  **arbitrary f32**, not UE8M0 (that encoding is attention-side only). Fixed
  with `_f32s` kernel variants.
- Second boot: **34.98 < 36.5 baseline**, needle-512 flipped exact→partial.
  Root cause: the existing grouped GEMV uses **scalar 1-byte FP8 loads ≈ 25%
  HBM bandwidth** — loses to DeepGEMM's TMA pipeline grinding 9× the rows.
- KILLED to opt-in (`ARLE_DSV4_MOE_DECODE_GEMV=1`); code retained as the
  substrate for a *vectorized* decode kernel. (errors entry
  `2026-06-13-dsv4-decode-gemv-lane-bandwidth-kill`.)

## The reframe: "44" was incorrect-MoE-fast

`268005d7` documents 44.04 = `d7be8c9b` itself (matched A/B 40.29→44.04). So
**44 = era = the window base**, and era ran the subtly-wrong compact
contiguous MoE (per-tile violation above). The honest target is **era-class
speed WITH correctness**, which the contiguous kernel cannot give (compact ⊕
correct are mutually exclusive there). The 64-align fix already took the
correctly-recoverable half (33.5→36.9). Segment-1 (`874f8cfb`, 43→40) is
**not a real regression** — it is a Qwen-only change; the only DSv4-touching
piece (shared `dsv4_route` kernel) shows 0.041→0.041ms unchanged in the
trace. It was boot lottery.

## Remaining gap and the fix in flight

36.9 → ~43 is entirely the MoE decode padding. The contiguous kernel reads
each expert's weights once (good) but the row-linear ops process 64-padded
rows where only ~1 is real. The lever is a **compact + correct +
bandwidth-efficient** decode kernel.

The in-tree `moe_bf16_grouped_gemm_swiglu_decode` (`9e37bc77`) is exactly
that shape — compact (scales with real routes), **16-byte vector loads**,
exactly-once weight reads, per-route correct — but **BF16**, while DSv4
experts are FP8. **Fix in flight: port it to FP8** (dequant in the vector
load), wire it into the decode-band dispatch. This is the scalar-GEMV
failure fixed at the bandwidth root.

Beyond 44: TPOT is then NCCL-latency-bound (129 collectives/tok). The
already-written, not-yet-A/B'd **replicated-attn** (`f2601ba6`, −86
collectives/tok) + **NUMA pin** (`4bee3009`) are the levers past era.

## Ledger

`a1e15307` 64-align · `68833c3f` ceil_div · `9bcb30d6` wins · probes
`0e07d0ba`/`613b0c28`/`09fd950c` · GEMV `f3f24765`/`e3662c61` · kill
`cef442ec`. Next: FP8 decode-band kernel.
