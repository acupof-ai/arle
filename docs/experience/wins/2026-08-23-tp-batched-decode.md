# TP batched decode: rows>1 goes through the batched paged forward — CUDA, 2026-08-23

> Status: Verified (ce190fc20) — c≥2 ITL 1.55×–17.89×, aggregate 87 → 1,300 tok/s at c=32

## Context

After the TP decode graph landed (2026-08-22 entry), TP2 decode was still flat at
~88 tok/s aggregate from c=2 up: under TP, every rows>1 decode sub-batch fell
back to B per-row eager forwards (`executor/qwen35.rs` `is_single` gate). The
batched paged forward was already TP-complete — attention all-reduces inside the
attention kernels, FFN all-reduce per layer — and was only gated by a
never-validated single-GPU assumption.

## Change

`ce190fc20` — deleted the `is_single` branch and the per-row fallback loop in
`submit_decode_batch`; rows>1 always routes to `submit_decode_batch_paged`.
c=1 is untouched (single-row graph path).

## Verification

ThinkingCap-Qwen3.6-27B-NVFP4, 8×H20, TP2, FP8 KV, spec off, matched GPU pairs
(treatment 0,1 / baseline 6,7 — a cross-NUMA pair (2,4) cost +60% ITL on the
baseline and was discarded). Baseline = Phase 1 binary rebuilt with `cuda,nccl`
(per-row at c≥2, graph at c=1). Simultaneous A/B, 64 synthetic prompts, 128
tokens, c=1..32:

| c | batched ITL ms | per-row ITL ms | speedup | batched wall s | per-row wall s |
|---|---|---|---|---|---|
| 1 | 14.75 | 13.91 | wash | 122.9 | 115.7 |
| 2 | 14.59 | 22.56 | 1.55× | 62.8 | 96.0 |
| 4 | 13.42 | 44.94 | 3.35× | 29.7 | 94.4 |
| 8 | 16.29 | 90.32 | 5.55× | 18.1 | 94.1 |
| 16 | 14.68 | 181.09 | 12.34× | 8.9 | 93.9 |
| 32 | 20.25 | 362.40 | 17.89× | 6.3 | 93.9 |

Aggregate at c=32: 87 → 1,300 tok/s (14.9×). TTFT improves with it (936 → 580
ms at c=32, less CPU contention). Per-row ITL grows linearly with batch size
(B forwards per step); batched ITL stays flat — one forward per step.

Correctness: two needle ladders ×3 runs concurrently against the batched serve
(mixed batches, NEEDLE_MAX_TOKENS=512): 54/54 exact across all 9 lengths. The
Phase 1 len=300 counting-loop degeneracy did not reproduce (batch composition
changes MoE routing order). The bench repetition gate's per-arm mismatch
(batched 0 vs per-row 8 failures at c≥2) is the same routing-order effect, not
corruption — the needle content gate is clean.

## Rule

A per-row fallback loop under TP is a throughput ceiling, not a safety net:
once the batched forward has its collectives in place, the single-GPU gate is
unvalidated caution, and validating it is one A/B away. Match A/B GPU pairs for
NUMA as well as model — a cross-socket pair cost +60% ITL and masqueraded as a
treatment effect.

## Follow-up: where the c=32 step time goes (nsys, 2026-08-23)

nsys trace of the batched TP2 serve under the c=32 bench (490 decode steps,
5.73 s pure decode window): GPU busy 81 % on both ranks. The 19 % idle is the
upper bound for batched-under-graph, and part of it is non-capturable (per-row
sampling D2H, host PageMeta build, pointer staging) — realistic gain ~10 %.

The model is dense (no expert keys in config; FFN intermediate 17408). GPU-side
cost per step (kernel share of the window, per rank): the NVFP4 Marlin W4A16
GEMV fleet 41 % (8.6 ms of the 20.3 ms step — quantized qkv/o + gate/up/down
projections), linear-attention recurrent decode (gdr+conv1d) 11 %, DeepGEMM
fp8 10 %, NCCL all-reduce 6.5 %, paged attention 4.5 %.

Roofline: the quantized projections are ~20 B params × ~0.63 B/param (4-bit +
group scales) ≈ 12.7 GB, ~6.4 GB/rank at TP2, read once per step. 6.4 GB /
8.6 ms ≈ 740 GB/s — 18 % of the H20's 4 TB/s HBM peak. The bf16 lm_head GEMM
in the same trace runs at ~80 % of peak, so the gap is the Marlin path, not
the hardware. A 2× Marlin GEMV is worth ~+26 % decode throughput.

## Follow-up: kernel A/B harness (2026-08-23)

`scripts/slice_checkpoint.py` writes a self-contained N-layer checkpoint
(config truncated, layers 0..N-1 + all non-layer weights); the engine loads
it unmodified. The 4-layer slice of this model (must include ≥1 full-attention
layer — `full_attention_interval 4`, so `--layers 4` — or the KV-pool profiler
sees 0 full-attn layers and clamps slots to 1) is ~7 GB and loads in seconds.

Per-op timing: `ARLE_CUDA_PROFILE=1 arle serve --model-path <slice>` disarms
the graph (events cannot record on a capturing stream) and exposes exact
per-op µs on `/v1/stats` — the kernel A/B bed. Kernel-level: nsys on the slice
produces a 4 MB trace (vs 157 MB for the full model).
`scripts/nsys_op_attrib.py` joins kernels to NVTX op ranges, but NVTX ranges
are absent under graph replay — use the ARLE_CUDA_PROFILE path for op-level
attribution. `bench_throughput.py --ignore-warmup-gate` skips the warmup
correctness gate (a 4-layer model generates garbage by design).

Caveat: the NVFP4 baseline has a confirmed correctness bug on tool-rendered
prompts (`docs/experience/errors/2026-08-23-nvfp4-tool-calls-corrupt.md`) —
Marlin optimization on this quant is secondary to that.
