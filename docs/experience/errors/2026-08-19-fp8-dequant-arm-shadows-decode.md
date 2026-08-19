# One prefill arm ate concurrency, spec decode, and stability — CUDA, 2026-08-19

> Status: Root-caused, fix landed, re-measure pending

## Context

Qwen3.8-27B-NVFP4 on 1xH20. Three separate symptoms had three separate
explanations on file. All three were one defect.

| | Symptom | Explanation on file (wrong) |
|---|---|---|
| D1 | aggregate throughput falls 5x at c>=2 | not yet investigated |
| D2 | server exits at ~34K context | `CUDA_ERROR_ILLEGAL_ADDRESS`, cause unknown |
| D3 | MTP 14.0 / DSpark 17.0 against 66.6 tok/s no-spec | "verify runs depth+1 rows through the recurrent linear attention on 48 of 64 layers" |

## Root cause

`try_fp8_dequant_bf16_gemm_batch` (`ops/quant_linear.rs`) dequantises the WHOLE
weight matrix to a BF16 scratch and runs cuBLAS. It is a prefill trade: the
dequant is O(N*K) and independent of M, so it only pays when amortised over many
rows. Its gate was `M >= QWEN_FP8_DEQUANT_GEMM_MIN_M`, and that constant is 2.

Two things kept it harmless until they stopped:

1. It required the canonical 128x128 block shape, so on Hopper DeepGEMM claimed
   every weight it could have taken. Its own comment recorded this: *"No-op on
   Hopper (returns false)."*
2. The NVFP4 support plan removed the block-shape gate so per-channel FP8
   attention weights would get a fast prefill path. DeepGEMM takes only 128x128,
   so per-channel now falls through to this arm — at every M >= 2. The comment
   was not updated and still asserted the no-op.

It also sat FIRST in `gemm_batch`, ahead of both Marlin arms. Every other format
runs Marlin first and dequant second:

| format | chain |
|---|---|
| W8A16, FP4 | Marlin -> dequant+cuBLAS -> GEMV |
| FP8 | dequant+cuBLAS -> (no Marlin arm) -> GEMV |

`Qwen3.8-27B-NVFP4` is a mixed checkpoint: 145 of ~200 quantised GEMMs per
forward are FP8 (all 48 linear-attn in/out_proj, all 16 self-attn q/k/v/o, MLP
on 8 of 64 layers, lm_head), and only 56 layers' MLP is NVFP4. So the arm ran on
the majority of the model, and 11.56 G params were re-materialised to BF16 on
every step with more than one row.

Measured cost: 84.35 ms per forward of pure weight dequantisation, 84% of the
M=3 quantised-GEMM budget, performing no work on the activations.

## Why it explains all three

**D1.** Any decode batch >= 2 is M >= 2. Step time is 15.0 ms at batch 1 and a
flat 100.6-114 ms for every batch from 2 to 16 — flat because the dequant does
not scale with M. Distribution is tight (c=4: p10 100.98 / p50 101.13 /
p90 101.33), so it is a path switch, not contention.

**D3.** A spec verify submits depth+1 rows: M=3 for MTP d=2, M=7 for DSpark
block 6. Both cross the same threshold. Even at 100% acceptance MTP would be
1.65x worse than no spec, because the verify step costs 4.95x a decode step.
Measured acceptance was 35.1% (`/v1/stats` `spec_decode`), so acceptance was
never the problem.

The batched GEMV this arm shadowed was written for exactly this case —
`quantized_gemv.cu:2919` tiles the batch so the weight streams from HBM once per
tile, and its comment calls it *"the lever that makes MTP spec-decode a net
win"*. It was never reached at M=3.

**D2.** The dequant scratch is allocated at runtime, sized to the largest FP8
weight (lm_head 248320x5120 -> 2.54 GB BF16). The KV pool profiles free VRAM at
startup and takes `mem_fraction_static` 0.9 of it, so the scratch is claimed from
whatever is left. Second crash surfaced the arm by name:
`Qwen FP8 dense dequant BF16 GEMM failed: DriverError(CUDA_ERROR_UNKNOWN)`.

## Matched A/B — same binary, same box, same moment, same flags

`--kv-cache-dtype fp8`, synthetic prompt, 30 s per point, GPU0 NVFP4 / GPU1 FP8.

| c | NVFP4 ITL ms | NVFP4 agg tok/s | FP8 ITL ms | FP8 agg tok/s |
|---:|---:|---:|---:|---:|
| 1 | 15.01 | 66.6 | 17.60 | 56.8 |
| 2 | 100.59 | 19.9 | 20.21 | 99.0 |
| 4 | 102.02 | 39.2 | 20.58 | 194.4 |
| 8 | 105.85 | 75.6 | 22.43 | 356.7 |
| 16 | 113.95 | 140.4 | 25.46 | 628.4 |

Qwen3.6-27B-FP8 is immune because its weights are 128x128 block-scaled and
DeepGEMM claims them before this arm.

## Fix

Two structural changes, no new tuning:

- The arm moves below both Marlin arms, so the chain is uniform across formats.
- Its M floor becomes `QWEN_DEQUANT_GEMM_PREFILL_MIN_M` (512) — a routing
  invariant, above any decode batch (256 slots) and below
  `chunked_prefill_size` (2048). `QWEN_FP8_DEQUANT_GEMM_MIN_M = 2` stays for
  FP4/W8A16, where a Marlin arm claims M <= 1024 first.

Upgrade path: instantiate Marlin `kFE4M3fn` for per-channel FP8. The vendored
template already carries `dequant<nv_bfloat162, kFE4M3fn>` and the scalar type;
only the instantiation is missing, exactly as with `kFE2M1f`. That deletes the
scratch, the threshold, and D2's accounting exposure together.

## Rule

A gate that makes a path a no-op is load-bearing. Removing one ("accept any
block shape") without re-reading what the removed condition was protecting turns
a prefill path into a decode path silently — the arity gate `M >= 2` cannot tell
decode from prefill, and no test covers `2 <= M <= 13` because no non-spec path
submits it.

Corollary: when a format is added to a dispatch chain, place it in the same
order as every other format. FP8 having dequant-before-Marlin while FP4 and
W8A16 had Marlin-before-dequant is what let one arm shadow the kernel written
to handle its case.
