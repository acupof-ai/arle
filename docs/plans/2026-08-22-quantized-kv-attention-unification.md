# Quantized-KV attention: one tensor-core kernel for every Qwen — 2026-08-22

> Status: Open. Runtime `5759a2caa`, Qwen3.8-27B-NVFP4, 1×H20. Each phase
> closes on a measurement and a CHANGELOG line.

## Where we stand

`paged_attention_quantized_fa3_partial_kernel` (GQA-group dequant,
[entry](../experience/wins/2026-08-21-paged-attention-quantized-gqa-shared-dequant.md))
is the decode kernel for Qwen3.5/3.6/3.8 with FP8/INT8 KV. It is CUDA-core
scalar math: B=32, ctx 32 K, fp8 KV measures 5.77 ms against a 2.1 GB KV read
whose bandwidth floor on H20 is ≈0.5 ms. SGLang / FlashInfer run this on
tensor cores and sit near the floor. So the kernel is ≈10× off SOTA, and it
is still the largest decode item at c≥16.

Four code paths serve quantized-KV decode today:

| path | lines | quant scheme | consumer |
|---|---:|---|---|
| `paged_attention_quantized_fa3.cu` | 452 | per-token K/V | Qwen3.5/3.6/3.8 decode, head_dim 256 |
| `decode_attention_varlen_quantized.cu` | 443 | per-token | same family, FA3-disabled fallback |
| `decode_attention_quantized.cu` | 970 | KIVI per-channel K | Qwen3 dense (`attention.rs:816`) |
| FA3 quant shim (`arle_fa3_shim.cu`, dequant temp) | — | per-token | prefill rows, workspace overflow |

## Phase 1 — tensor-core partial kernel, the only decode path

- Rewrite the partial kernel: one CTA per (batch row, kv-head, split); the
  kv-head's q-heads (6 on Qwen3.8, 8 on Qwen3.6, 4 on Qwen3 dense) padded to
  the 16-row `mma.sync.m16n8k16` tile; K/V tiles dequantised once to bf16 in
  shared memory; S = Q·Kᵀ and O = P·V on MMA; online softmax per row in
  registers. head_dim 128/256, FP8/INT8. Merge kernel and Rust interface
  unchanged; `heads_per_cta` becomes the full GQA group.
- sm_80+ only (`__CUDA_ARCH__ >= 800`); the sm_70 lane supports BF16 KV only
  (`docs/environment.md`), so it never reaches this kernel.
- Delete the scalar partial kernel, `decode_attention_varlen_quantized.cu` and
  its fallback branch, the `head_dim == 256` gate. The shim keeps prefill rows.
- Gate: microbench vs the current kernel (bf16-ulp diffs), e2e c=1/16/32 on
  the 32 K chain, needle ×3, 200-item eval. Target: B≥16 within 2× of the
  bandwidth floor.

## Phase 2 — Qwen3 dense joins the same path

- Switch Qwen3 dense KV quantisation from KIVI per-channel K to per-token
  (the SGLang / vLLM scheme), route `attention.rs` decode to the Phase 1
  kernel, delete `decode_attention_quantized.cu` and the KIVI calibration
  scaffolding.
- Gate: needle ×3 + 200-item eval on a Qwen3 dense checkpoint with INT8 and
  FP8 KV against the KIVI baseline. If per-token loses quality on Qwen3 dense,
  KIVI stays and the phase closes as rejected with the numbers.

## Phase 3 — prefill on the same pool

The 32 K agent chain is 154:1 prefill to decode; after Phase 1 the end-to-end
number is set by prefill, which still dequantises the quantized prefix into a
bf16 temp for FA3 (5× the KV traffic). Land FA3's native FP8 KV prefill over
the paged pool; delete the shim. Gate: c=1/16 TTFT on the 32 K chain, needle.

## Phase 4 — fill the free GEMM rows

Marlin costs the same at M=1..8
([backlog #1](2026-08-21-nvfp4-decode-lever-backlog.md)). With decode
attention near its floor, MTP d=2/4 is the mechanism that fills those rows:
no-spec / d=2 / d=4 on one binary, ITL and end-to-end. Serving experiment,
no kernel work.

## Close-out

- `docs/support-matrix.md`: FP8 KV from opt-in to production once Phases 1–2
  carry a quality verdict on both families.
- CHANGELOG line per phase; phase exits cut a tag.
