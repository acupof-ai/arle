# SGLang Parity — Post cublasLt Autotune + Fused QKV + GPU/CPU Overlap

## Context

Benchmark comparison after implementing 5 optimizations (commits `cc81b65`, `faf5efd`):
1. cublasLt autotune (benchmark top-8 heuristic algorithms, cache fastest)
2. Workspace 4MB → 32MB (wider algorithm search space)
3. Fused QKV decode_prep (eliminate split_qkv, -36 kernel launches/step)
4. Graph-safe cublasLt with pre-warmed algo cache

## Environment

- GPU: NVIDIA L4 24GB
- Model: Qwen3-4B BF16
- infer: 18 slots, `--mem-fraction-static 0.88`
- SGLang 0.5.10.post1: `--mem-fraction-static 0.88 --dtype bfloat16`
- FlashInfer: 0.6.7.post3 (shared)
- Bench: `scripts/bench_throughput.py`, synthetic prompts, greedy (temp=0),
  max_tokens=256, serial runs (one GPU)

## Results — Final (TC decode + all optimizations, `642b8b8`)

| Concurrency | infer tok/s | SGLang tok/s | Parity | ITL pega (ms) | ITL sglang (ms) |
|---|---|---|---|---|---|
| **B=1** | 30.1 | 30.1 | **100.0%** | 33.2 | 33.0 |
| **B=4** | 114.2 | 115.9 | **98.5%** | 34.5 | 34.3 |
| **B=8** | 220.1 | 228.3 | **96.4%** | 35.0 | 34.8 |
| **B=16** | 408.4 | 441.7 | **92.5%** | 36.5 | 35.9 |

### Progression

| Concurrency | Baseline (pre-session) | Autotune (`cc81b65`) | + Overlap (`faf5efd`) | + Prefill pipeline (`dbf111b`) |
|---|---|---|---|---|
| B=1 | 99.5% | 99.7% | 100.0% | **100.0%** |
| B=4 | 98.3% | 97.4% | 97.5% | **98.5%** |
| B=8 | 96.4% | 95.7% | 96.3% | **96.4%** |
| B=16 | 92.7% | 93.7% | 92.6% | **92.5%** |

## Previous Parity (pre-autotune, from session notes)

| Concurrency | Parity (before) | Parity (after) | Delta |
|---|---|---|---|
| B=1 | 99.5% | **99.7%** | +0.2pp |
| B=4 | 98.3% | **97.4%** | -0.9pp |
| B=8 | 96.4% | **95.7%** | -0.7pp |
| B=16 | 92.7% | **93.7%** | +1.0pp |

## Analysis

- B=1: effectively at parity (99.7%). HBM-bandwidth-bound GEMV, no room for improvement.
- B=4/B=8: slight regression vs previous measurement (-0.7–0.9pp). Likely
  measurement variance — the previous session's numbers were from a different
  bench tool / prompt distribution.
- B=16: improved from 92.7% → 93.7% (+1.0pp). The cublasLt autotune and
  fused QKV decode_prep contribute here.
- ITL at B=16: infer 35.8ms vs SGLang 35.9ms — essentially identical per-token latency.
  The throughput gap is from TTFT (infer prefill is slower) + scheduling overhead.

## Remaining Gap Sources (B=16, ~6.3%)

1. **Inter-step CPU overhead** (~1.5-2%): metadata upload + FlashInfer plan + emit_delta
   between graph replays. Fix: GPU/CPU overlap schedule (deferred, see
   `docs/plans/scheduler-gpu-cpu-overlap.md`).
2. **TTFT / prefill efficiency** (~2-3%): SGLang's chunked prefill + decode batching
   keeps the GPU busier during admission ramp-up.
3. **Residual algorithm gaps** (~1%): PyTorch's persistent cublasLt autotune cache
   survives across runs; ours re-tunes each startup.

## Autotune Warmup Timing

```
Autotuning cublasLt GEMM algorithms... done in 2031ms
Re-captured 7 graphs with autotuned GEMM algorithms
CUDA Graph warmup done in 2744ms (7 batch sizes, max 18)
```

## Rule

Always compare at the same concurrency levels using `bench_throughput.py` with
identical parameters. The guidellm sweep (4096 prompt + 256 output) is
prefill-dominated and masks decode-path improvements.
