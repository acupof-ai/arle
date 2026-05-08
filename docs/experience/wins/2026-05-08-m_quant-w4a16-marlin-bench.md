# M_quant W4A16 Marlin Bench

## Goal

Validate whether the existing W4A16 + Marlin production path is the viable sm_89
weight-bandwidth axis after FP8 was killed on this machine.

## Hypothesis

W4A16 should reduce decode weight bandwidth enough to move longctx 4k/c=4 ITL
from the BF16 baseline `19.27 ms` toward the predicted `10.26 ms`. License
fires if measured ITL is `<= 12 ms`.

## Command

Checkpoint preparation:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  hf download JunHowie/Qwen3-4B-GPTQ-Int4 \
  --local-dir infer/models/Qwen3-4B-GPTQ-Int4 --max-workers 8

/home/ckl/projects/arle/.venv/bin/python scripts/convert_gptq.py \
  infer/models/Qwen3-4B-GPTQ-Int4 \
  --output infer/models/Qwen3-4B-GPTQ-Int4-converted

/home/ckl/projects/arle/.venv/bin/python scripts/marlin_repack.py \
  infer/models/Qwen3-4B-GPTQ-Int4-converted \
  --output infer/models/Qwen3-4B-GPTQ-Int4-marlin
```

The public GPTQ checkpoint loaded but failed the correctness smoke with repeated
non-English fragments, so it was not benchmarked. The measured path used ARLE's
internal symmetric W4A16 quantizer from the local BF16 checkpoint:

```bash
/home/ckl/projects/arle/.venv/bin/python scripts/quantize_weights.py \
  infer/models/Qwen3-4B \
  --bits 4 --group-size 128 \
  --output infer/models/Qwen3-4B-W4A16-sym-g128

/home/ckl/projects/arle/.venv/bin/python scripts/marlin_repack.py \
  infer/models/Qwen3-4B-W4A16-sym-g128 \
  --output infer/models/Qwen3-4B-W4A16-sym-g128-marlin
```

Server:

```bash
CUDA_HOME=/opt/cuda \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120
```

Bench:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh m_quant-w4a16-marlin-c4-r1 \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --processor infer/models/Qwen3-4B \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

Repeated as `r2` and `r3`.

## Environment

- Backend: CUDA
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Driver: 595.71.05
- CUDA: 13.2 (`cuda_13.2.r13.2/compiler.37668154_0`)
- Model: Qwen3-4B, ARLE self-quantized W4A16 symmetric group size 128
- Commit: `2fdfd26`
- Feature set: `cargo build --release -p infer --features cuda`
- Non-default flags: `--num-slots 8 --max-seq-len 5120`
- KV cache: auto FP8E4M3 paged pool

## Correctness Gate

Short deterministic smoke passed before benchmarking:

```text
Write a short sentence about stars.
=> Okay, the user wants a short sentence about stars. Let me think...
```

The public GPTQ-converted route failed this gate, so only the self-quantized
checkpoint is reported.

## Results

Per-run GuideLLM fixed-c results:

| run | TTFT mean (ms) | TTFT p50 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p50 (ms) | out tok/s | E2E mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 2458.9 | 2391.3 | 2573.5 | 11.75 | 11.73 | 191.64 | 5.45 |
| r2 | 2480.6 | 2566.5 | 2587.6 | 11.77 | 11.76 | 191.01 | 5.48 |
| r3 | 2481.3 | 2565.4 | 2581.0 | 11.79 | 11.78 | 190.83 | 5.49 |

n=3 stability:

| metric | mean | median | sigma | relative sigma |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 2507.73 | 2565.40 | 100.84 | 4.02% |
| ITL p50 (ms) | 11.76 | 11.76 | 0.03 | 0.21% |
| out tok/s | 191.16 | 191.01 | 0.43 | 0.22% |

Three-way comparison:

| Engine | TTFT p50 (ms) | ITL p50 (ms) | out tok/s | Notes |
|---|---:|---:|---:|---|
| ARLE BF16 baseline (`786a20a`) | 1976.4 | 19.27 | 153.83 | longctx 4k/c=4 |
| ARLE W4A16 Marlin, this run | 2565.4 | 11.76 | 191.16 | n=3 median for TTFT/ITL, mean tok/s |
| SGLang 0.5.11 BF16 | 972.9 | 19.44 | 164.3 | reference |
| vLLM s8 BF16 | 1177.0 | 19.4 | 159.1 | reference |

Delta table:

| metric | baseline | W4A16 | delta |
|---|---:|---:|---:|
| ITL p50 vs ARLE BF16 | 19.27 ms | 11.76 ms | 1.64x faster, -39.0% |
| out tok/s vs ARLE BF16 | 153.83 | 191.16 | +24.3% |
| out tok/s vs SGLang BF16 | 164.3 | 191.16 | +16.3% |
| out tok/s vs vLLM BF16 | 159.1 | 191.16 | +20.1% |
| TTFT p50 vs ARLE BF16 | 1976.4 ms | 2565.4 ms | +29.8% slower |

## Marlin Hit Evidence

- Loader detected W4A16 quantization: `bits=4`, `group_size=128`.
- The checkpoint contains `252` `.qweight` tensors and `252`
  `.marlin_qweight` plus `252` `.marlin_scales` side buffers.
- Server loader emitted `+ Marlin repacked` for every quantized linear weight
  during startup.
- Runtime logs do not expose a per-launch Marlin counter; dispatch evidence is
  from the loaded side buffers plus `linear.rs` selecting `MarlinW4Gemm` for
  aligned batched W4A16 linears.

## Decision

License fires for W4A16 as the sm_89 weight-bandwidth axis:

- ITL p50 `11.76 ms` is below the `<= 12 ms` gate.
- Decode speedup is real and stable across n=3.
- TTFT regresses, so W4A16 is not a prefill solution and should not be framed as
  closing the longctx prefill gap by itself.

Next step: pivot implementation effort to KV W4A8 (#33) as the parallel
quantization axis, with W4A16 retained as the measured weight-bandwidth path.

## Problems

- Official `Qwen/Qwen3-4B-GPTQ-Int4` was not available via Hugging Face.
- Public GPTQ conversion (`JunHowie/Qwen3-4B-GPTQ-Int4`) produced a loadable
  checkpoint but failed correctness, so external GPTQ compatibility remains
  unlicensed.
- Longctx TTFT is worse than BF16, consistent with prefill being dominated by
  non-decode work and quantized prefill conversion overhead.

## Learnings

- On RTX 4070 Ti SUPER / sm_89, FP8 GEMM is not viable, but W4A16 is viable for
  decode bandwidth.
- Wall-clock framing matters: W4A16 wins ITL and output throughput while losing
  TTFT. Treat it as a decode/token-throughput tool, not a prefill fix.
- GPTQ checkpoint compatibility needs a correctness gate before any perf number
  is accepted.

## Artefacts

- Raw r1: `bench-output/2026-05-08-m_quant-w4a16-marlin-c4-r1/benchmarks.json`
- CSV r1: `bench-output/2026-05-08-m_quant-w4a16-marlin-c4-r1/benchmarks.csv`
- HTML r1: `bench-output/2026-05-08-m_quant-w4a16-marlin-c4-r1/benchmarks.html`
- Trace r1: `bench-output/2026-05-08-m_quant-w4a16-marlin-c4-r1/service_stats_trace_summary.md`
- Raw r2: `bench-output/2026-05-08-m_quant-w4a16-marlin-c4-r2/benchmarks.json`
- Raw r3: `bench-output/2026-05-08-m_quant-w4a16-marlin-c4-r3/benchmarks.json`
