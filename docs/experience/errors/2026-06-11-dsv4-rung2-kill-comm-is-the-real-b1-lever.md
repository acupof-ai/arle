# Rung 2 (HC-segment megakernel) KILLED before build — nsys says the real B=1 lever is the 34% in TP/EP comm, not HC fusion

## Context

Kernel-fusion ladder for DSv4-Flash B=1 decode (8×H20 TP=8/EP=8). Rung 1
landed (+9.3%, campaign 39.51→44.04: mhc_params warp tail + fused hc_pre/rms_norm
prologue + 1024 threads). Rung 3 (fused all-reduce + hc_post) was taken through
the correctness gate and killed (see the sibling entry — math exhaustively proven
correct by static audit, perf premise dead: overhead-removal on a GPU-bound path
≈ 1% wash). Commissioned task: "23串行做好" — bring Rung 2 (the multi-block
hc_enter / mix-GEMV segment megakernel, counter-sync, per
`docs/plans/2026-06-11-dsv4-rung23-fused-ar-and-segment-kernel.md`) to a verdict.

Per §0 (no kill on un-verified 占比; M_pf-graph Phase-0 lesson), licensed-or-killed
Rung 2 by **measuring the remaining fusable overhead first**, not by building the
~600-line cooperative kernel and discovering wash after.

## Evidence — `nsys stats --report cuda_gpu_kern_sum` on `nsys_b1_allreduce.nsys-rep`

Pre-Rung-1 B=1 decode trace (Jun 10 08:43; Rung 1 landed Jun 10 22:38–Jun 11 02:26),
one-shot-AR comm backend, the production B=1 lane. GPU-busy time by kernel:

| % GPU-busy | kernel | inst | med ns | what |
|-----------:|--------|-----:|-------:|------|
| **17.3** | `cross_device_reduce_1stage<bf16,8>` | 176,128 | 25,792 | one-shot **AR**, 2×/layer (attn+moe) |
| **16.6** | `cross_device_allgather_1stage<bf16,8>` | 88,032 | 51,136 | one-shot **AG**, 1×/layer |
| 12.9 | `dsv4_mhc_params_kernel` | 176,128 | 35,328 | Rung-1 target (→~8.5µs post-Rung-1) |
| 3.8 | `nvjet_…splitK_TNT` | 342,016 | 5,184 | GEMV — **shared by all model GEMMs**, not just mhc-mix |
| 2.1 | `cublasLt::splitKreduce` | 561,152 | 1,792 | GEMV reduce — shared |
| 0.8 | `dsv4_mhc_post_kernel` | 176,128 | 2,176 | hc_post |
| 0.6 | `dsv4_mhc_pre_kernel` | 176,128 | 1,536 | (now fused into prologue by Rung 1) |

**AR + AG = 33.9% of all GPU-busy time.** These are serial critical-path
dependencies in lockstep B=1 decode (each layer's reduce feeds the next op), so the
GPU-busy fraction maps to real wall-clock cost. Median AR 25.8µs / AG 51µs for a
**[4096] bf16 (~8 KB)** message is 25–50× the data-movement floor (~1µs on H20 HBM/NVLink)
→ the cost is barrier + launch + rank-skew **overhead**, not bytes → **reducible**.

## Root Cause (of the wrong-target premise)

Rung 2 fuses the HC segment (mix-GEMV + launch gaps + params/prologue). But:

1. **Rung 1 already ate the big HC pieces** — mhc_params 35µs→8.5µs, hc_pre+rms_norm
   fused. Post-Rung-1 the entire mhc-kernel family is single-digit %.
2. **The mix-GEMV Rung 2 targets is mostly irreducible model-GEMM compute.** The
   `nvjet…splitK` (3.8%) and `splitKreduce` (2.1%) are shared across q/k/v/gate/up/down
   projections — only a small slice is the 24×16K mhc-mix. A megakernel removes the
   launch boundary (~5µs), not the GEMV math.
3. **Net addressable by Rung 2 ≤ ~3%**, below the ±6% cross-session drift floor, and
   carries **regression risk** (hand-rolled GEMV vs tuned nvjet — the plan's own flagged risk).
4. Meanwhile **34% sits in comm**, untouched by any rung.

Building Rung 2 is the exact §0 framing trap: polishing a narrow window (≤3%) while the
real bottleneck (comm, 34%, 10× bigger) goes unexamined. `feedback_b1_decode_gpu_bound_overhead_removal_wash`
named this a year ago: only **less GPU compute / comm overlap / EAGLE** move B=1.
HC-fusion is overhead-removal → wash. Comm is the 34%.

## Fix / Redirect

- **Rung 2: KILLED before build.** No megakernel. The ladder is exhausted — Rung 1 was
  the one fusion lever with real headroom; Rungs 2/3 are overhead-removal on a GPU-bound path.
- **Next lever (licensed by this trace): TP/EP collective cost.** 2 AR + 1 AG per layer,
  median 25.8/51µs each, dominated by barrier/launch/skew not bytes. Candidate moves
  (each needs its own license): (a) **overlap** the collective with the next layer's
  independent compute; (b) **reduce collective count** (3/layer → fuse attn-AR+moe-AR, or
  cut the AG); (c) understand why the **AG is 51µs / what it gathers** (16.6% in one kernel
  is the single largest line — likely the EP combine or wide-HC-stream gather); (d) a
  lower-latency barrier than the per-block spin.
- **Confirmatory step before committing the comm campaign:** capture a fresh **post-Rung-1**
  B=1 nsys (comm absolute is unchanged but fraction rises as HC shrank) with clean
  single-rank wall-clock framing, and trace the AG call site in `tp.rs`/`dsv4.rs`.

## Rule

**License-or-kill a fusion lever by measuring the post-prior-rung kernel breakdown FIRST,
not by building it.** When a ladder's later rungs target single-digit-% overhead while a
profile shows a 10×-bigger category (here comm, 34%), the rung is a framing trap — kill it
and redirect to the dominant category, even if the dominant lever is harder. Fusion is
overhead-removal; on GPU-bound B=1 only less-compute / comm-overlap / EAGLE move the number
(`feedback_b1_decode_gpu_bound_overhead_removal_wash`).
