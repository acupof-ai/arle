# 2026-04-15 · Industry approaches to quantized KV cache decode attention

## Why this research

The 2026-04-15 long-seq bench
([`../experience/wins/2026-04-15-bench-longseq-int8-splits32.md`](../experience/wins/2026-04-15-bench-longseq-int8-splits32.md))
showed that our INT8 decode attention kernel
(`crates/infer-cuda-kernels/csrc/attention/decode_attention_quantized.cu`)
sits at **55.2 ms ITL at 25k** vs bf16's **33.1 ms** — a 22 ms
residual gap after all the tuning levers (SMEM-tile + cp.async +
splits=32) were pulled. User asked me to survey how the major serving
frameworks handle this class of kernel before we decide whether to
pursue Option B (`page_size = 16` pool lift) or an IMMA-based
tensor-core rewrite.

## What the state of the art actually is

### Per-format KV-cache-type support across frameworks

| Framework | INT8 KV | FP8 E4M3 KV | FP8 E5M2 KV | Paged block size | Scale granularity | Dequant strategy |
|---|---|---|---|---|---|---|
| **our in-tree** | yes (custom) | yes (custom) | no | **1** | per-token × per-head (f32) | fused in-register (split-KV partial kernel) |
| TensorRT-LLM | yes | yes | no | **8 / 16 / 32 / 64 / 128** | **per-tensor** (`kv_cache_scaling_factor` shape `[1]`) | fused in-kernel (XQA masked MHA) |
| vLLM | **no** | yes | yes | 16 or 32 typical | **per-tensor or per-head** | TRTLLM backend via FlashInfer (fused) |
| SGLang | **no** | yes | yes | 64 preferred (for TRTLLM XQA) | per-tensor / per-head | TRTLLM XQA backend (fused) |
| FlashInfer 0.6.6 (kernels we already link) | **not upstream** | yes (via `trtllm_batch_decode_with_kv_cache`) | yes (same API; dlpack caveat on e5m2) | runtime-variable, optimised via "prefetch page indices into SMEM" | per-tensor (with FP8 query quant auto-enabled when FP8 KV is) | fused (up to 2× faster than FP16 on the FP8 path) |
| AdaLLM (SM89-first, Feb 2026) | — | **yes, custom Triton** `fp8_kv_decode_group_kernel` | yes | **256** | per-tensor FP8 dtype (no separate scales) | fused in Triton kernel |

Observations:

1. **INT8 KV is becoming a minority format.** vLLM and SGLang have
   both deprecated / never added INT8 in favour of FP8. TRT-LLM is
   the main holdout that still ships INT8, and it uses **per-tensor
   scale**, not per-token per-head like our kernel.
2. **Our `page_size = 1` is an outlier.** Everyone else runs the
   quantised pool at 16+. TRT-LLM XQA picks from
   `{8, 16, 32, 64, 128}`. SGLang recommends 64 for XQA. AdaLLM
   uses 256. The pool layout choice drives a 16-128× difference in
   the number of per-token pointer-chases the decode kernel has to
   issue, which is exactly the residual-gap signature we saw on our
   kernel at 25k.
3. **Per-tensor scale is the canonical choice** because the scale
   load becomes a kernel-launch-time constant instead of a
   scattered per-token / per-head value. Our per-token × per-head
   symmetric INT8 preserves more quality at the cost of scale-load
   pressure that the other kernels don't pay.
4. **Everyone uses fused dequant-in-register.** Nobody materialises
   a full bf16 working buffer during decode. That is the path the
   contiguous-cache `prepare_layer` / `commit_layer` code in our
   tree takes, and it's only used by the legacy single-request
   engine, not the scheduler. Matches our codebase structure.
5. **Tensor cores + FP8 is where the real speedup lives.**
   FlashInfer's FP8 decode claim is "up to 2× over FP16" — that is
   achieved by reading FP8 from HBM and feeding it directly into
   `e4m3 → bf16` register casts with a shared-memory tile that
   feeds an IMMA-free FP16/BF16 dot product. FlashInfer's **
   `trtllm_batch_decode_with_kv_cache`** is the production path
   vLLM/SGLang both delegate to.

### Benchmark data points (framework authors' own numbers)

- **TRT-LLM INT8 KV cache** on decode-heavy workloads: **1.45×
  throughput improvement** over FP16 baseline (H100, per
  SqueezeBits blog). Prefill-heavy: 1.09× improvement. NOT a
  per-request speedup; the win comes from packing more sequences
  into the HBM the int8 cache frees up.
- **TRT-LLM FP8 KV cache**: similar shape, 1.0-1.4× throughput
  depending on workload.
- **AdaLLM FP8 KV on RTX 4090 (same SM89 as L4)** with Qwen3-8B:
  **~2.4× lower peak VRAM** vs FP16, **~20-25% throughput loss**.
  So even a 2026 Triton-based FP8 decode kernel on SM89 has a
  20-25% perf gap vs bf16 — close to our 20% gap at 4k / 8k.
- **FlashInfer FP8 decode**: "up to 2× faster than FP16 kernels"
  is the headline claim, but that is against FP16 attention-only
  kernels and measured on H100, not SM89. On SM89 with L4 memory
  bandwidth the gap narrows significantly because HBM bandwidth
  per SM is already closer to the compute ceiling.
- **Our kernel**: 20-80% slower than FlashInfer bf16 depending on
  context length, 10% slower than bf16 at short contexts, ~70%
  slower at 25k.

### How SGLang framed the consistency rule (matches our root cause)

> "when quantized KV cache must be dequantized before use in
> attention operations, performance can be **extremely slow** if
> dequantization is not fused with the attention kernel, so it's
> important to always verify that your chosen attention backend
> supports quantized KV cache, as backends without fused support
> may experience significant throughput degradation."

That is exactly what the single-request-engine `prepare_layer`
path does in our tree (`infer/src/model/kv_cache.rs:206`) — and why
Claude's initial hypothesis about the regression was wrong: the
scheduler path is already fused. Industry confirms the path we're
on; the residual gap is kernel quality, not kernel shape.

### How vLLM decided on FP8 over INT8

Per the vLLM docs, the decision was:

> "Quantizing the KV Cache to FP8 … offers significant advantages
> over traditional INT8 quantization by providing a larger dynamic
> range while maintaining similar compression benefits. By
> preserving dynamic range, FP8 quantization helps maintain model
> accuracy."

vLLM supports per-tensor and per-head FP8 scale granularity but
**not** per-token (the finer granularity we use). They explicitly
argue that per-token scales are overkill for KV quality once you're
on FP8's exponent-biased format.

SGLang made the same call, exposing only `fp8_e4m3` and `fp8_e5m2`
via `--kv-cache-dtype`.

## Applying this to our tree

Three things the industry survey suggests we do, ranked by cost:

### 1. Cheap: add **FP8 E4M3** as the canonical quant KV default, keep INT8 as an advanced flag

What: bump `KVFormat::FP8E4M3`'s `default_page_size()` from 1 to 16
alongside a fused per-tensor dequant decode kernel. Hide INT8
behind a `--kv-cache-dtype int8-per-token` flag with a warning.

Why: FP8 E4M3 is the industry default. Our per-token INT8 is
higher quality on paper but is slower than bf16 at every context
length we measure, so the quality advantage is moot. Shipping
FP8 as the default quant format aligns with vLLM / SGLang / TRT-LLM
and lets users reach for the format they expect.

Effort: ~300-500 LOC. Per-tensor scale is a kernel launch constant
instead of a per-token load; the kernel is strictly simpler than
ours. Can be implemented by stripping the per-token scale loads out
of `decode_attention_quantized.cu` and replacing with a constant.
Doesn't require page_size lift on its own, but benefits from it.

### 2. Medium: **`page_size = 16` pool lift** for all quant formats

This is the Option B already documented in
[`../plans/tiered-kv-cache-tasks.md:226`](../plans/tiered-kv-cache-tasks.md)
§1.3. Every framework we surveyed runs quant pools at 8+, usually
16-128. Our page_size=1 is the outlier and is the structural
source of the kernel latency difference. Lifting page_size is a
5-8 file change that rewrites:

- `kv_cache_to_paged_int8_kernel` at `csrc/kv/kv_cache_to_paged.cu`
- `quantize_paged_kv_fp8_kernel` at `csrc/quant/kv_quant.cu`
- `scatter_kv.cu`
- `decode_attention_int8_partial_kernel` (our tile loop)
- `decode_attention_fp8_partial_kernel` (same shape)
- `paged_kv.rs` pool format dispatch
- `batch_decode.rs:449,494` call sites
- `kv_types.rs` `default_page_size` dispatch table

Estimated ~500-800 LOC, matches M0.3's deferred P1.5 work.
Expected win: closes ~8-15 ms at 25k based on the TRT-LLM / SGLang
baseline numbers. Still won't match bf16 FlashInfer at 25k (that
requires TC / IMMA — Option C below).

### 3. Expensive: **FlashInfer `trtllm_batch_decode_with_kv_cache` integration**

vLLM and SGLang both delegate quant KV decode to this upstream
FlashInfer API. Our in-tree `flashinfer_decode.cu` wrapper is
BF16-only today, but the FlashInfer 0.6.6 API we already link has
a `trtllm_batch_decode_with_kv_cache` function that takes FP8 K/V
and handles the fused dequant + IMMA / TC path internally.

Effort: unknown without a prototype. The bindings would add:

- A new Rust FFI wrapper around `trtllm_batch_decode_with_kv_cache`
  in `crates/infer-cuda-kernels/src/ffi/kv.rs`
- A new Rust callable in `crates/infer-cuda-kernels/src/kv_quant.rs`
- Dispatch in `batch_decode.rs` for the quant decode path that
  prefers the FlashInfer path over our custom kernel when
  FP8 E4M3 + per-tensor scale
- Pool-format requirement: HND layout, per-tensor scale, page
  size 8+. Drives most of Option B's changes as prereqs.

This is the path that would get us into the 2×-FP16 territory
FlashInfer advertises. But it's the biggest project and should not
start until Option B lands.

### The counter-intuitive finding from AdaLLM

AdaLLM's ~2.4× VRAM savings for ~20-25 % throughput loss on SM89
is the **empirical ceiling** for an FP8 decode kernel of this
generation on this hardware. It matches our 4k / 8k numbers
(−1.6 % / −4.3 %) more closely than our 25k number (−10.8 %).
That says Option A + splits=32 already got us into the FP8-on-SM89
ballpark at short context; the 25k-specific degradation is a
separate issue that scales with `num_kv_heads × num_layers × seq_len`
and is more about the kernel's scaling profile than its absolute
efficiency.

The lesson: **don't expect to beat bf16 with quant KV on L4**.
Industry frontier on SM89 is parity-minus-15-to-25%. The INT8
per-token path we have is slightly-worse-than-parity at short
context, which is actually reasonable; the 25k regression is the
gap worth closing.

## Concrete recommendation

I would land these in this order:

1. **Accept the current state for short context.** 4k / 8k / 16k
   performance (1.6 - 7.3 % slower than bf16) is inside the industry
   norm for L4/SM89 FP8 decode kernels.
2. **Open the M0.4 Option B ticket now**, scoped to the 8-file diff
   above, as the next quant-KV perf investment. This is what closes
   the 25k-specific gap.
3. **Follow Option B with a FlashInfer `trtllm_batch_decode_*` FFI
   prototype** to see whether we can delegate the int8/fp8 decode
   hot path to the upstream library rather than maintaining our own
   fused kernel. If the FFI path works, we can deprecate our custom
   `decode_attention_quantized.cu` entirely and just forward to
   FlashInfer.
4. **Rename the quant story** in the CLI help text + project docs:
   FP8 is the default recommended format; INT8 per-token per-head
   is an advanced flag for users who need the extra quality
   headroom. Matches vLLM / SGLang / TRT-LLM framing.

Things **not** worth pursuing:

- A hand-rolled IMMA / tensor-core int8 decode kernel. FlashInfer
  and TRT-LLM already do this upstream; reimplementing it for
  our own tree is a 2-4 week project that would immediately go
  stale vs FlashInfer's own release cadence. Better to delegate.
- Further single-file tuning of
  `decode_attention_int8_partial_kernel`. We've extracted
  Option A + splits=32 wins already; the next marginal perf comes
  from kernel-architecture changes (TC / IMMA), not tuning.

## Sources

- [FlashInfer: Kernel Library for LLM Serving (GitHub)](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer 0.6.6 attention kernel reference](https://docs.flashinfer.ai/api/attention.html)
- [FlashInfer KV-Cache Layout tutorial](https://docs.flashinfer.ai/tutorials/kv_layout.html)
- [FlashInfer issue #721 — "How to use low bit KV Cache"](https://github.com/flashinfer-ai/flashinfer/issues/721)
- [vLLM Quantized KV Cache documentation](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [vLLM FP8 E4M3 KV Cache reference](https://docs.vllm.ai/en/v0.4.1/quantization/fp8_e4m3_kvcache.html)
- [SGLang Quantized KV Cache documentation](https://docs.sglang.io/advanced_features/quantized_kv_cache.html)
- [SGLang attention backend matrix (GitHub)](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md)
- [TensorRT-LLM — Multi-Head, Multi-Query, and Group-Query Attention (XQA + INT8/FP8 dequant)](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)
- [TensorRT-LLM — Numerical Precision reference](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)
- [SqueezeBits blog — vLLM vs TensorRT-LLM #8: KV Cache Quantization](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079)
- [AdaLLM — NVFP4 Inference on SM_89 with FP8 KV Cache and a Custom Decode Kernel (Feb 2026)](https://benchaliah.medium.com/adallm-nvfp4-inference-on-sm-89-e-g-rtx-4090-with-fp8-kv-cache-and-a-custom-decode-kernel-19d7b4a7ebf2)
- [FlashInfer launch blog — "Accelerating Self-Attentions for LLM Serving with FlashInfer"](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- [FlashInfer issue #1957 — "Is float8 supported for flashinfer trtllm attention?"](https://github.com/flashinfer-ai/flashinfer/issues/1957)
