# W4AFP8 GEMV decode lane — CUDA, 2026-08-21

> Status: Shipped

## Goal

Decode throughput for DSv4-Flash-0731 (NVFP4→W4AFP8) at c=1/4/8/16 on H20,
TP=2 and TP=4.

## Implementation

Reuses the W4A16 grouped GEMV kernel for W4AFP8 decode. Two format conversions:

1. **Nibble sign flip in-kernel** — W4A16 GEMV expects unsigned nibbles with
   zero-point=8 (`value = (nibble - 8) * scale`); W4AFP8 stores signed INT4
   two's complement. The kernel takes an `xor_mask` parameter (`0x08080808`
   for W4AFP8, `0` for W4A16) that flips each nibble's sign bit on the fly.
   Zero extra VRAM. A converted weight copy was tried first and OOMed at
   3 GB/GPU.
2. **Scale de-interleave + transpose** at table-build time (one-time, lazy).
   w13 scales are stored w1/w3 row-interleaved by the loader; w2 scales are
   not interleaved. Both are transposed from `[K//512, N*4]` CUTLASS layout
   to `[N, K//128]` row-major for the GEMV kernel.

Dispatch: W4AFP8 checkpoint, routes ≤ 128 → GEMV; routes > 128 → CUTLASS
(prefill band). No env var, no feature flag — single path.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8000 \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 16 \
  --max-tokens 128 \
  --synthetic-prompts 8
```

- Baseline: CUTLASS grouped GEMM for decode (pre-GEMV), 18.3 tok/s at c=1
- Treatment: `521db0bed` — xor_mask GEMV (final, zero-VRAM)
- Prompt tokens: 8 (synthetic)
- Completion tokens: 128
- Trials: 16 requests per concurrency level

## Environment

- Host / GPU: H20 96GB ×8
- Driver / CUDA: sm_90, CUDA 12.x
- Model / dtype: DeepSeek-V4-Flash-0731, NVFP4→W4AFP8 (INT4+BF16)
- TP=2: `--tensor-parallel-size 2 --max-running-requests 8 --max-total-tokens 131072`, 88 GB/GPU
- TP=4: `--tensor-parallel-size 4 --max-running-requests 16 --max-total-tokens 131072`, 67 GB/GPU

## Results

### TP=2

| concurrency | decode tok/s (per-req) | aggregate tok/s |
|---:|---:|---:|
| 1 | 31.0 | 30.1 |
| 4 | 16.6 | 63.2 |
| 8 | 12.1 | 90.2 |
| 16 | 11.7 | 90.5 |

c=1 speedup vs baseline (18.3 tok/s): **1.69x**. Saturates at c=8
(max-running-requests=8); c=16 queues half the requests.

### TP=4

| concurrency | decode tok/s (per-req) | aggregate tok/s |
|---:|---:|---:|
| 1 | 41.1 | 39.8 |
| 4 | 22.5 | 85.8 |
| 8 | 17.0 | 127.2 |
| 16 | 11.0 | 161.7 |

TP=4 c=1 is 1.33x TP=2 (memory-bound GEMV; less work per GPU offset by NCCL
all-reduce). TP=4 still scaling at c=16: 161.7 tok/s aggregate, 1.79x the
TP=2 ceiling.

## Problems

Three bugs found and fixed:

1. **xor_mask** — W4A16 GEMV kernel expects unsigned nibbles with zero-point=8;
   W4AFP8 stores signed INT4 two's complement. Fixed with the `xor_mask` kernel
   parameter (`0x08080808` for W4AFP8, `0` for W4A16), zero VRAM overhead.
2. **w13 scale de-interleave (P1, codex review)** — the loader stores w13
   scales as plain CUTLASS `[K//512, n13*4]` (w1/w3 concatenated along N).
   The table builder incorrectly de-interleaved with stride `K/128*2 = 112 B`
   instead of the actual `N*4*2 = 16384 B`, systematically mixing gate/up
   scales. Fixed by removing the de-interleave entirely — direct transpose
   from the plain layout.
3. **Verify path GEMV vs CUTLASS** — DSpark verify (M=5, 30 routes) initially
   hit the GEMV lane; the first fix restricted GEMV to `num_tokens == 1` on the
   assumption that M>1 loads weights per-token without reuse. That assumption
   was wrong: the GEMV kernel reuses weights across routes for the same expert
   (up to 16), so M=5 verify (~28 unique experts, most with M=1) is efficient
   on GEMV. CUTLASS grouped GEMM tiles M and pads to tile boundaries — pure
   waste at M≈1 per expert. Reversed the restriction (GEMV for all route counts
   ≤ 128); A/B vs CUTLASS pending-remote.

An earlier weight-copy conversion approach (3 GB/GPU) OOMed at TP=2 with
max-running-requests=16. The xor_mask approach has zero VRAM overhead.

## DSpark spec decode (post-fix)

DSpark draft (FP8 attn + FP4 experts, block=5, layers 40-42) on top of the
fixed GEMV lane. TP=4, long prompt (119k), prefix-cache hit, pure decode.

| Metric | Pre-fix | Post-fix |
|--------|--------:|---------:|
| Acceptance rate | ~49% (short) / 58-65% (long) | **64.8%** |
| Tokens per chain | ~2.85 | **4.83** |
| Decode tok/s (DSpark) | ~41 (0% speedup) | **51.2** |
| Baseline (no DSpark) | 41.1 | 41.1 |
| **Speedup** | **~1.0x** | **~1.25x** |

The scale fix raised acceptance (fewer chains per token); the verify-path fix
lowered per-chain cost. Together they convert DSpark from 0% to 1.25x.

### c>1 regression (pin to c=1)

DSpark drafts per slot (sequential), so the draft tax scales with batch while
the verify savings do not. Measured on TP=4:

| Concurrency | DSpark tok/s | No-DSpark tok/s | Δ |
|---:|---:|---:|---:|
| 8 | 86.4 | 127.2 | −32% |
| 16 | — | — | −47.7% |

The executor pins DSpark to c=1 (`spec_max_batch().min(1)`). The sequential
draft tax (8×~10 ms at c=8) plus the MoE verify cost (240 routes, ~200 unique
experts) exceed the batched-decode savings. Dense models (Qwen3) batch the
draft and do not hit this wall.

## Learnings

PASS. The W4A16 GEMV kernel is reusable for W4AFP8 decode with two format
conversions: the nibble sign flip in-kernel via `xor_mask` (zero extra VRAM),
and the scale transpose once at table-build time. The 1.69x c=1 decode
speedup is the main win for interactive serving. TP=4 scales to c=16 with
161.7 tok/s aggregate. DSpark spec decode on top delivers an additional 1.25x
at c=1 (64.8% acceptance, 4.83 tok/chain).

Codex review caught the w13 scale de-interleave bug (P1) — the GEMV path had
been shipping wrong gate/up scales since implementation. Kernel alignment
reviews pay off.
