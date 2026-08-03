# CP×DP verified end-to-end; the training step is finally attributed — and the attribution re-ranks the campaign

**Date:** 2026-08-03 · **Commits:** 4aa6e5e02 (mesh) + 00e482f50 (CommAxis) + a644adab8 (GEMM f32 out) + e57c59793 (port fix) + 3cae75304 (count fix) · **Pod:** 8×H20, real 27B

## Context

Round-2 gate battery on the CP×DP mesh tranche plus the bf16-GEMM-writes-f32
change, with the nsys per-kernel budget extracted from the saved seq=8192 rep.

## What worked

**G1 (cph_parity, world=2): PASS** — `ce_cp_vs_cpu=8.534e-5` at the bf16 floor,
device/host CE agree to 1e-7.

**G2 (cp=2, seq=32768): PASS** — losses 4.805783/6.064485 identical to the
pre-change nccl reference at 6 decimals; fwd 102.0/102.4 s, bwd 384.2/384.3 s
(within 1%). The GEMM output-cast deletion is loss-neutral and time-neutral at
this scale (consistent with the nsys table: cuBLAS GEMMs are only ~10% of the
step).

**G3 (cp=2×dp=2): port fix verified** — distinct world-rank ports, dp replica 1
completed, walls match G2. Caught a real normalization bug: losses came back
exactly ÷4 (world size) because every cp rank contributed the same
replica-global count to the world-comm count reduce → fixed (3cae75304, only cp
rank 0 contributes). Re-gate pending.

**G4 (cp=4, seq=131072): PASS, clean exit** — full fwd+bwd+step in ~3100 s
(fwd ~683 s, bwd ~2415 s). **Host-RSS verdict: peak 170.4 GiB total
(~44.6 GB/rank) ≈ half the cp=2 343 GB reference → host RSS scales with the
per-rank shard.** The 131072 host-OOM is solved by cp=4; 256K extrapolates to
cp=8 at similar per-rank RSS. No host-allocator surgery needed to unblock.

## The attribution table (nsys, seq=8192 cp=2, ~50 s step, ~6% GPU idle)

| share | time | kernel |
|---|---|---|
| 31% | 30.2 s | ring_block_attention fwd_merge + bwd (CP ring SDPA, 16 full-attn layers) |
| 26% | 25.8 s | linear_attention chunk grad/transfer/carry f32 (GDN backward) |
| 13% | 12.3 s | kernel_kernel (unidentified — action item) |
| 10% | ~10 s | cuBLAS GEMMs (nvjet) |
| 7% | 6.6 s | gated_delta_rule_prefill_recurrent (23,808 launches) |
| — | **15.9 s HtoD, 75,076 ops** | host→device uploads (93% of mem time; max single copy 468 ms) |

Layout churn (transpose/slice ~23k launches) is visible but small (2.9 s).
Attention is O(s²): at 32768 the ring share only grows.

## Rule — the attribution re-ranks the campaign

The elementwise-fusion/bf16-bandwidth hypothesis (SOTA-survey rank #2) is NOT
the measured wall at this scale. The real top three TIME levers:
1. **Ring-attention kernel efficiency** (31% and growing with seq) — our
   hand-rolled per-block SDPA vs an FA-class kernel;
2. **HtoD staging** (75k uploads/step, 15.9 s) — find who uploads; this is also
   the prime host-RSS suspect;
3. **GDN chunk backward** (26%).

bf16-tape (D2) keeps its VRAM rationale (halves activations for 256K) but its
time claim is demoted pending the levers above. Formula-ranked suspects from
external surveys are hypotheses; only the local share table licenses the knife.
