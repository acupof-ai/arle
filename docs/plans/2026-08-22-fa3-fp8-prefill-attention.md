# FA3 fp8 prefill attention over the quantized pool — 2026-08-22

> Status: Closed 2026-08-22 — landed `8f48ff6b4` + `7d58850dc`, routed at
> seqlen_k ≥ 64K and seqlen_q ≥ 256; 220K TTFT −17 %
> ([entry](../experience/wins/2026-08-22-fa3-fp8-prefill-attention-long-context.md)).

## Why

Prefill at long context is the attention's FLOPs. Per-op profile
(`ARLE_CUDA_PROFILE=1`, shares): at 32 K `full_paged/attention` is 16 % of
the forward; at 180 K it is 50 % (46.0 s of 92.3 s) and sits on FA3's bf16
compute floor (4·L²·d·H/2 × 16 layers ≈ 6,370 TFLOP ≈ 43 s at 148 TFLOPS).
The only lever below that floor is the fp8 tensor-core rate (2×). Expected
TTFT: 32 K −7 %, 128 K −21 %, 220 K −25 % (130 → ≈100 s).

## What exists

`vendor/flash-attention/hopper/flash_api.cpp:359` dispatches
`run_mha_fwd_<90, float_e4m3_t, 256, 256, Split, PagedKVNonTMA, …>`; descales
are per (batch, kv_head) f32 tensors (`q/k/v_descale`, `flash_api.cpp:694`).
The vendored tree carries bf16 instantiations only. Today's quantized-pool
prefill (`arle_fa3_shim.cu` Path A) dequantises the pages the table names into
a compact bf16 temp and runs the bf16 kernel.

## Steps

1. Generate the e4m3 hdim256 instantiations (paged, paged+split, packgqa) with
   the vendored `generate_kernels.py`; add them to `build.rs` next to the bf16
   set. Compile on the pod.
2. Shim: replace the bf16 temp with an e4m3 temp. Per (request, kv_head)
   scale `S = max_t s_t` over the request's per-token scales (tight: the
   per-token quantiser puts one element at full range), requant each byte as
   `e4m3(dequant(byte)·s_t / S)`; Q: bf16 → e4m3 with per (request, kv_head)
   absmax over the GQA group. Descale tensors `[batch, kv_heads]` f32 live in
   the same stream-ordered temp. Output stays bf16.
3. Rust: `qwen35_attention.rs` FA3 quant branch passes the new entry; the
   bf16 shim entry is deleted in the same tranche (no half-states).
4. Gate: microbench shim-bf16 vs shim-fp8 on one 32 K and one 180 K request
   (attention time, max diff); e2e TTFT at 32 K (c=1/16) and 220 K; needle ×3
   at 512/4096/16384/32768 plus 200000; 200-item eval. Accept on TTFT with
   needle 12/12 + eval within the n=200 spread; otherwise revert the tranche
   and record.

## Risks

- Per-(request, head) scale on K/V loses the per-token resolution the pool
  keeps; K is QK-normed on this family, V is not. The gate decides.
- FA3 fp8 paged path is `PagedKVNonTMA`; its throughput at page_size 16 is
  unmeasured — the microbench in step 4 is the first number.
