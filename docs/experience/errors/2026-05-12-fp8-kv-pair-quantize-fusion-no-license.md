# FP8 KV pair quantize fusion no-license

## Context

Goal thread: optimize Qwen3.5 FP8/FP4 quantization and KV quantization
operators. After the single-token FP8 KV quantize thread-grouping kill, the
next plausible path was fusing the runtime K and V writes:

```rust
kv_quant::quantize_paged_kv_fp8(... K ...)
kv_quant::quantize_paged_kv_fp8(... V ...)
```

That fusion would touch CUDA C, FFI, Rust wrappers, model call sites, tests,
and benchmarks. Before doing that larger cross-layer change, a paired
component bench was added to measure the current two-call baseline under the
same synchronization framing as the existing Criterion CUDA benches:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_paged_kv_fp8_qwen35 --quiet
```

Bench shape:

| Param | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| batch_size | 8 |
| page_size | 16 |
| num_kv_heads | 4 |
| head_dim | 256 |
| kv_dim | 1024 |
| pair bench | K work buffer + V work buffer, two current kernel calls |
| KV format | BF16 HND work buffer -> FP8 E4M3 NHD + f32 scales |

## Root Cause

The static source survey was correct that Qwen3.5 FP8 decode/paged-prefill
paths enqueue separate K and V quantize calls. But static launch count is only
a hypothesis, not a license. The paired wall-clock component bench did not show
a meaningful two-launch penalty under the current harness.

Measured runs:

| Run | Candidate | Criterion time | Throughput |
|---|---|---:|---:|
| combined first run | single K/V-equivalent call | `8.6794-8.8283 us`, point `8.7234 us` | `939.08 Melem/s` |
| combined first run | current K+V pair, no intermediate sync | `8.5657-8.6780 us`, point `8.6131 us` | `1.9022 Gelem/s` |
| pair-only rerun | current K+V pair, no intermediate sync | `8.4689-8.4928 us`, point `8.4821 us` | `1.9316 Gelem/s` |
| single-only rerun | single K/V-equivalent call | `8.7174-8.9035 us`, point `8.7747 us` | `933.60 Melem/s` |

This framing measures a whole benchmark iteration with one pre-sync and one
post-sync. It does not prove that two GPU kernels are literally free; it does
prove that "two launches exist" is not enough evidence to justify a larger
runtime fusion change. The observed paired wall-clock is already within the
same small fixed-cost envelope as the single-call bench.

## Fix

Do not implement the K/V fused FP8 quantize runtime path from launch count
alone. Keep the current separate K and V calls for now.

The new `ops_cuda/quantize_paged_kv_fp8_qwen35_pair` bench remains committed
as the required baseline for any future fusion attempt. A future fusion must
beat the pair bench directly and should ideally include an `nsys` or CUDA-event
profile that isolates launch enqueue cost from GPU kernel time.

## Rule

Launch-count source survey is a hypothesis, not evidence. For tiny CUDA
operators, a fused-kernel rewrite is licensed only by a paired component A/B or
profile evidence under the same synchronization framing the runtime actually
uses. Do not spend a cross-FFI/model-call-site change on an unmeasured "two
launches must be slower" assumption.
