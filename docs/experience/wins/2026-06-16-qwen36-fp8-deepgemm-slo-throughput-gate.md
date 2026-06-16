# Qwen3.6 FP8 DeepGEMM SLO throughput gate: memory pass, throughput pathologically slow

## SLO-shape probed? Y

4096 input tokens / 256 output tokens, c=1,2,4,8, H20 GPU0. The guidellm
generator was set to `prompt_tokens=4095` because its 4096 setting produces
4097 tokens under the server tokenizer and aborts before entering the engine.
The emitted server usage for valid requests was exactly 4096 prompt tokens and
256 completion tokens.

## Goal

Run the final FP8-vs-BF16 throughput gate on the prefill-heavy SLO shape after
the Qwen3.6 FP8 resident quant lane was routed through native DeepGEMM for
large-R prefill.

## Hypothesis

If the FP8 DeepGEMM port is correctly wired, it should be competitive with or
faster than BF16 on the 4096/256 prefill-heavy shape. A large regression here is
not evidence against FP8 or DeepGEMM as a method, and not evidence that H20 is
"compute-bound"; it is evidence that this ARLE Qwen FP8 prefill implementation
is still wrong or dominated by an unrooted overhead.

## Environment

- Remote tree: `/data01/arle-qwenfp8-smoke`.
- Binary: `/data01/arle-qwenfp8-smoke/target/release/arle`.
- Hardware: NVIDIA H20 GPU0.
- Runtime env: `ARLE_QWEN35_DEEPGEMM=1`,
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` build,
  `CUDA_HOME=/usr/local/cuda`,
  `ARLE_DEEPGEMM_JIT_CUDA_HOME=/usr/local/cuda-12.9`,
  `NVCC_CCBIN=/usr/bin/clang++-11`.
- Serve shape: `--num-slots 999 --total-pages 272 --page-size 16
  --max-total-tokens 4352 --max-prompt-tokens 4096`.
- Workload: guidellm concurrent profile, `prompt_tokens=4095`,
  `output_tokens=256`, `--max-seconds 60`, `--warmup 5`.

## Slot and memory result

| Backend | Effective slots | Peak VRAM | Slot delta |
|---|---:|---:|---:|
| FP8 DeepGEMM quant | 381 | ~91.9 GiB | +111.7% vs BF16 |
| BF16 baseline | 180 | ~93.9 GiB | baseline |

The FP8 memory/slot license remains valid on the SLO shape.

## SLO throughput table

Metric is completed output tokens divided by the raw wall interval from first
request start to last request end in the guidellm JSON. `inc` counts requests
that were still incomplete at the fixed 60s point.

| c | FP8 ok/inc | FP8 out tok/s | FP8 TTFT ms | FP8 ITL ms | BF16 ok/inc | BF16 out tok/s | BF16 TTFT ms | BF16 ITL ms | FP8 delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2/1 | 6.54 | 20740.7 | 72.3 | 8/1 | 31.50 | 1760.9 | 24.9 | -79.3% |
| 2 | 0/2 | 0.00 | 0.0 | 0.0 | 8/2 | 33.28 | 3467.6 | 46.4 | -100.0% |
| 4 | 0/4 | 0.00 | 0.0 | 0.0 | 12/4 | 39.09 | 6844.6 | 75.9 | -100.0% |
| 8 | 0/8 | 0.00 | 0.0 | 0.0 | 8/8 | 33.93 | 13457.6 | 183.8 | -100.0% |

## Verdict

FP8 still does not win on the SLO shape, but the magnitude is the important
signal: FP8 TTFT is 20.7s vs BF16 1.76s at c=1, output throughput is 6.54 tok/s
vs 31.50 tok/s, and FP8 completes 0 requests at c=2 while BF16 completes 8.
That is pathologically slow. The throughput gate is a KILL for this ARLE FP8
DeepGEMM prefill implementation as currently wired.

Do not interpret this as an H20 compute-bound verdict or as a throughput-method
verdict against FP8/DeepGEMM. Industry FP8 prefill should not be 12x slower than
BF16. The required next step is root-cause profiling of the ARLE Qwen FP8 prefill
path: JIT warmup, BF16-to-FP8 pack-quantize, generated grouped GEMM shape/config,
max-m padding, fallback gates, activation quantization, and scatter/combine.

Keep the memory/slot result: FP8 fits about 2.1x the long-shape slots in the
same VRAM budget. The raw-throughput result remains an unrooted implementation
bug until a stage breakdown proves otherwise.

Follow-up: `2026-06-16-qwen36-fp8-dense-deepgemm-cold-jit-fix.md` root-caused
and fixed the 12x cold-JIT part of this regression. The first 4K FP8 request
dropped to about 2.73s, but still does not beat the BF16 1.97s smoke, so the
throughput gate remains not passed.

## Artifacts

- FP8 JSON:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-fp8-deepgemm-slo-4096x256-c1-8/benchmark.json`
- FP8 CSV:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-fp8-deepgemm-slo-4096x256-c1-8/benchmark.csv`
- BF16 JSON:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-bf16-slo-4096x256-c1-8/benchmark.json`
- BF16 CSV:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-bf16-slo-4096x256-c1-8/benchmark.csv`

## Rule

When the synthetic bench generator and server tokenizer disagree by one token,
the apparent 4096-token run can be an ingress abort with zero engine steps. The
valid SLO gate is the server-usage shape, not the generator's nominal token
count. A double-digit FP8 slowdown on that valid SLO shape is our implementation
bug until profiled to a specific external limit.
