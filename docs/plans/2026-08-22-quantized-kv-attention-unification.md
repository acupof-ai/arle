# Quantized-KV attention: one tensor-core kernel for every Qwen — 2026-08-22

> Status: Closed 2026-08-22. Runtime `5759a2caa` → `1df0acf68`, Qwen3.8-27B-NVFP4, 1×H20. Each phase
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

## Phase 1 — tensor-core partial kernel, the only decode path — CLOSED

Landed `97d28ba2c`; c=16 +33 %, c=32 +34 %, kernel 2.7–3.9×, B=32 now 4× off
the floor ([entry](../experience/wins/2026-08-22-paged-attention-quantized-tensor-core.md)).

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

## Phase 2 — Qwen3 dense joins the same path — CLOSED (family deleted)

No Qwen3 dense checkpoint is served on CUDA; the family and KIVI per-channel K
were deleted instead of ported (`1df0acf68`, −7,340 lines,
[entry](../experience/wins/2026-08-22-delete-qwen3-dense-cuda-and-kivi.md)).
Every CUDA quantized pool is per-(token, head) K+V on the Phase 1 kernel.

## Phase 3 — prefill on the same pool — CLOSED (rejected on measurement)

Per-op prefill profile, one 32 K prompt, Qwen3.8-27B-NVFP4, fp8 KV,
`ARLE_CUDA_PROFILE=1` (synchronising; shares, not absolutes): `dense_ffn`
4835 ms (48 %), `linear_attention` 2828 ms (28 %: in_proj 1213, gdr 800,
out_proj 443, conv1d 212), `full_attention` 2156 ms (22 %: FA3 + dequant shim
1566, qkv_gemm 352, o_proj 148). FA3's compute floor for 32 K × 16 layers
(4·L²·d·H/2 per layer ≈ 211 TFLOP) is ≈1.4 s on H20 bf16, so the 1566 ms is
the attention itself; the shim moves ≈2 GB across 16 layers, under 100 ms,
under 1 % of TTFT. A native fp8-pool prefill kernel has no TTFT to win.
TTFT levers, in order: the prefill GEMMs of `dense_ffn`, the linear-attention
projections, then FA3 fp8 compute (per-tensor descale only — incompatible
with per-token KV scales; low certainty).

## Phase 4 — fill the free GEMM rows — CLOSED (measured; no default to flip)

One binary (`1df0acf68`), Qwen3.8-27B-NVFP4, fp8 KV, 32 K chain, c=1/4/8/16,
per-request decode tok/s: no-spec 73.0 / 42.1 / 25.9 / 14.8; MTP d=2 **84.3**
/ 42.0 / 24.8 / 14.0; MTP d=4 83.0 / 42.1 / 26.3 / 15.1. TTFT identical.
MTP pays +15 % at c=1 only — the default `--spec-type auto` already selects
it — and is inert from c=4 because the verify step is not batched
([plan](2026-08-21-batched-mtp-verify.md)). The free Marlin rows (M ≤ 8) bound
the lever: c=1·d=2 verifies at M=3; c=4·d=2 would already be M=12. d=4 does
not beat d=2 (acceptance-limited). Caveat for future runs: with a drafter on,
`ITL p50` collapses (0.03 ms at d=4 — accepted tokens stream together); read
decode tok/s or ITL mean.

## Close-out

All four phases closed 2026-08-22. Remaining decode lever on this chain is
the batched MTP verify; remaining TTFT levers are GEMM-bound on H20 (no FP4
tensor cores) — see Phase 3.
